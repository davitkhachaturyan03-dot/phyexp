"""One return per metric per rollout (and per operation): the attribution-ready output of the evaluation layer.

Grounding
---------
- PoC p. 1: "per-metric and per-failure-type returns producing an episode × metric matrix".
- PoC §4.3 (p. 10): "Success within 30 s → binary return; RMST → negative capped time; AUC → win indicator
  against a sampled human time. HRT is a ratio, so it needs a first-order (linearized) approximation."
- PoC §9.1 Step 3 (p. 19): "Define one 'return' per metric for each rollout. For example, success within 30 s
  gives 1 or 0; for RMST, the return is negative time; for AUC, whether the rollout beat a sampled human
  time." "Repeat the analysis for subsets, such as a specific task or only rollouts with a specific failure
  type."
- PoC §3.4 (p. 7): "Operation-level (T, event) pairs give several outcome samples per rollout"; "Declared
  targets. Every attribution result states which functional it explains"; §4.4 (p. 10) "Label every claim
  with its metric and time cap τ". Every return column carries its metric and τ in its name.
- CUPID (docs/papers/CUPID_Influence_Functions_arXiv-2506.19121.pdf) §5.1 p. 5, Prop. 1 / Alg. 1:
  Ψ̂(ξ) = (1/m) Σ_i R(τ_i)/H Σ_{(s',a')∈τ_i} Σ_{(s,a)∈ξ} Ψ_a-inf(...): a scalar return R per rollout, and
  the rollout's (s', a') pairs. CUPID's released code uses a binary per-rollout ``success_mask`` (plan P25);
  this table gives one real-valued R per metric, keyed so a CUPID-style estimator can loop over columns.

Keys (lineage contract): every row carries ``eval_run_id`` and ``rollout_id`` (the rollout), and the policy's
``checkpoint_sha256`` and ``train_run_id``. The train run names its training subset, i.e. the exact
(``episode_id``, ``content_hash``) episode versions the episode × metric matrix will have as rows
(``attribution_frame``, checked with ``vlapoc.runs.check_scores``).

Usage::

    .venv/bin/python -m vlapoc.returns --eval-run <eval_run_id>          # writes returns.parquet etc. (data/eval/returns-<id>/)
    .venv/bin/python -m vlapoc.returns --eval-run <eval_run_id> --track  # the same, recorded as a 'returns' stage run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vlapoc import metrics_registry as _mr
from vlapoc.libero_convert import REPO_ROOT

TAU_S = 60.0        # RMST cap = τ_episode (plan: 60 s)
F_TAU_S = 30.0      # PoC §4.1 F(30 s)
HUMAN_REF = REPO_ROOT / "data" / "eval" / "human_reference" / "human_reference.parquet"
RETURN_COLUMNS = _mr.return_columns("W")  # family W per-rollout returns, from the single registry (plan r2 AC6.8)


def human_times(path: Path = HUMAN_REF) -> dict[tuple[str, int], np.ndarray]:
    """Human time-to-success per task (S3: replay of LIBERO's goal check; all 1000 v0 originals, the fixed HRT
    anchor): ``{(suite, libero_task_id): T_success_s array}``."""
    h = pd.read_parquet(path)
    h = h[h["success_any"].astype(bool)]
    return {(s, int(t)): g["T_success_s"].to_numpy(float) for (s, t), g in h.groupby(["suite", "libero_task_id"])}


def failure_type(event: int, n_ops: int, n_ops_done: int) -> str:
    """Failure taxonomy built from the operation rule (S4): ``success``, ``timeout_no_progress`` (no operation
    done), ``timeout_partial_<k>_of_<n>``. PoC §2.1 item 3 also names missed grasp, wrong object and ignored
    instruction, and §9.1 Step 2 names idle: LIBERO defines no such rule (plan P13), so they are not built;
    the raw per-step signals (conjunct trace, object positions, arm-action norm) are in the rollout record."""
    if event:
        return "success"
    return "timeout_no_progress" if n_ops_done == 0 else f"timeout_partial_{n_ops_done}_of_{n_ops}"


def _sample_rng(seed: int, rollout_id: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, int(hashlib.sha256(rollout_id.encode()).hexdigest()[:16], 16)]))


def auc_expected(T: float | None, censor_at: float, human: np.ndarray) -> float:
    """P(T_h > T) + ½ P(T_h = T) under the task's human reference, i.e. the expectation of the sampled win
    indicator (Mann-Whitney convention, ties ½). A censored rollout (T > ``censor_at``) wins against no human
    time it can be compared with: humans with ``T_h < censor_at`` are verified losses, the others are not
    comparable (PhAIL ``_auc_censored`` logic); 0.5 if nothing is comparable."""
    if T is None or np.isnan(T):
        comparable = human[human < censor_at]
        return 0.0 if len(comparable) else 0.5
    return float(np.mean(human > T) + 0.5 * np.mean(human == T))


def rollout_returns(rollouts: pd.DataFrame, human: dict | None = None, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """The per-rollout return table (one column per metric, named with its τ) and the HRT linearization weights.

    Columns (PoC §4.3 p. 10, §9.1 Step 3 p. 19, §3.4 p. 7):

    - ``ret_f30`` = 1[success ∧ T ≤ 30 s]; mean over a cell = F(30 s) (KM with censoring only at τ ≥ 30 s).
    - ``ret_rmst60`` = −min(T, 60 s), a censored rollout −60; mean = −RMST(60 s).
    - ``ret_auc_sampled`` = 1[T < T_h*] + ½·1[T = T_h*], T_h* drawn uniformly from the task's human times with
      ``SeedSequence([seed, sha256(rollout_id)])``: the PoC's literal "win indicator against a sampled human time"
      (ties ½: model and human times share the 20 Hz grid; plan r2).
    - ``ret_auc_expected`` = E[``ret_auc_sampled``] (``auc_expected``): the same target without sampling noise
      (a divergence: deterministic expectation of the PoC's sampled indicator).
    - ``ret_hrt60_lin`` = −w_o · min(T, 60 s), w_o = (1/J) · RMST_h,o(60) / RMST_m,o(60)² evaluated at this eval's
      policy (J cells). For HRT = (1/J) Σ_o RMST_h,o / RMST_m,o, a change in the rollouts' capped times gives
      ΔHRT ≈ Σ_o Δ mean_{i∈o}(ret_hrt60_lin) (first-order / delta method; §4.3 only says "first-order
      (linearized)"; the form is this module's derivation, checked numerically in S6 AC6.2). At the
      expansion point Σ_o mean_o(ret_hrt60_lin) = −HRT. The weights are returned for the record.
    - ``ret_censored60`` = 1[timed out] (§3.4 "the censored tail"); mean = ``evaluation.censored_rate``.
    - ``failure_type`` (``failure_type``).

    Family W (plan r2 "Declared functionals"; registry ``vlapoc.metrics_registry``): T is whole-task time-to-success in
    seconds (simulated, 20 Hz) and the human anchor is the whole-task reference of all 1000 v0 originals (S3), fixed.
    Censoring happens only at τ_episode = 60 s, so every cell mean is exactly the declared functional, and
    ``vlapoc.verdict`` (family W) verifies the same functional (AC6.10).
    """
    human = human if human is not None else human_times()
    df = rollouts.copy()
    df["cell"] = df["suite"] + "/" + df["libero_task_id"].astype(int).astype(str)
    T = df["T_s"].astype(float)
    capped = np.minimum(T.fillna(np.inf), TAU_S)
    out = df[["eval_run_id", "rollout_id", "train_run_id", "checkpoint_sha256", "external_checkpoint", "policy_type",
              "suite", "libero_task_id", "cell", "init_state_index", "seed", "event", "censored", "T_s", "n_ops",
              "n_ops_done"]].copy()
    out["failure_type"] = [failure_type(e, n, k) for e, n, k in zip(df["event"], df["n_ops"], df["n_ops_done"])]
    out["ret_f30"] = ((df["event"] == 1) & (T <= F_TAU_S)).astype(float)
    out["ret_rmst60"] = -capped
    h_of = [human[(s, int(t))] for s, t in zip(df["suite"], df["libero_task_id"])]
    hs = [h[_sample_rng(seed, rid).integers(0, len(h))] for h, rid in zip(h_of, df["rollout_id"])]
    out["ret_auc_sampled"] = [(float(t < x) + 0.5 * float(t == x)) if not np.isnan(t) else 0.0 for t, x in zip(T, hs)]
    out["ret_auc_expected"] = [auc_expected(None if np.isnan(t) else t, c / 20.0, h)
                               for t, h, c in zip(T, h_of, df["tau_episode_steps"])]
    # HRT linearization weights per cell at this policy
    J = df["cell"].nunique()
    weights = {}
    for cell, g in out.groupby("cell"):
        s, t = cell.split("/")
        rh = float(np.mean(np.minimum(human[(s, int(t))], TAU_S)))  # human RMST(60): all events, no censoring
        rm = float(-g["ret_rmst60"].mean())
        weights[cell] = {"rmst_human_60": rh, "rmst_model_60": rm, "w": (rh / rm ** 2) / J if rm > 0 else float("nan"),
                         "hrt_cell": rh / rm if rm > 0 else float("nan"), "n_rollouts": int(len(g))}
    out["ret_hrt60_lin"] = [weights[c]["w"] * r for c, r in zip(out["cell"], out["ret_rmst60"])]
    out["ret_censored60"] = df["censored"].astype(float)
    meta = {"tau_s": TAU_S, "f_tau_s": F_TAU_S, "auc_sample_seed": seed, "J_cells": J, "hrt_weights": weights,
            "hrt_macro_at_policy": float(np.nanmean([w["hrt_cell"] for w in weights.values()])),
            "human_reference": str(HUMAN_REF.relative_to(REPO_ROOT))}
    return out, meta


def operation_returns(rollouts: pd.DataFrame) -> pd.DataFrame:
    """Family O returns, one row per operation sample (PhAIL pairs: completed operations plus the one unfinished
    attempt per incomplete rollout; PoC §3.4 "Operation-level (T, event) pairs"): ``rollout_id``, ``op_k``,
    ``t_start_steps``/``t_end_steps`` (the segment whose (s', a') the operation's return can be credited to),
    ``T_op_s``, ``event`` and ``ret_rmst60_op_cap`` = −min(T_op, 60 s) for a completed operation and **−60 for the
    unfinished attempt** (plan r2, registry ``rmst60_op_cap``; analysis-r1 F-P3: an unfinished operation can never
    outrank a completed one). Cell mean over samples = −``evaluation.rmst_capped`` of the family-O arm."""
    rows = []
    for r in rollouts.itertuples():
        pairs = json.loads(r.pairs)
        t = 0
        for k, (T_op, e) in enumerate(pairs, start=1):
            steps = int(round(T_op * 20))
            rows.append({"eval_run_id": r.eval_run_id, "rollout_id": r.rollout_id, "train_run_id": r.train_run_id,
                         "checkpoint_sha256": r.checkpoint_sha256, "suite": r.suite, "libero_task_id": int(r.libero_task_id),
                         "op_k": k, "n_ops": int(r.n_ops), "t_start_steps": t, "t_end_steps": t + steps, "T_op_s": float(T_op),
                         "event": int(e), "ret_rmst60_op_cap": -min(float(T_op), TAU_S) if int(e) else -TAU_S})
            t += steps
    return pd.DataFrame(rows)


def metrics_registry() -> list[dict]:
    """The single registry (``vlapoc.metrics_registry``), serialized as ``metrics_registry.json`` with every returns run."""
    from vlapoc import metrics_registry as mr

    return mr.to_json()


def attribution_frame(train_run_id: str, lineage_root: Path | None = None) -> pd.DataFrame:
    """The empty episode × metric frame the attribution feature fills: rows = (``episode_id``, ``content_hash``)
    of the train run's training subset, one NaN column per metric with a return. ``vlapoc.runs.check_scores``
    rejects any frame whose keys are not exactly that subset (PoC §4.2 p. 9, §9.1 Step 3 p. 19)."""
    from vlapoc.runs import _training_subset, client

    subset = _training_subset(client(lineage_root).get_run(train_run_id), lineage_root)
    keys = subset.keys(lineage_root).reset_index(drop=True)
    for c in RETURN_COLUMNS:
        keys[c] = np.nan
    return keys


def load_eval_rollouts(eval_run_id: str, lineage_root: Path | None = None, make_out_dir: bool = True) -> tuple[pd.DataFrame, Path | None]:
    """The rollouts parquet an eval run logged (MLflow artifact, checked against the run's recorded sha256), and a
    dir to write returns next to it. The download goes to a ``TemporaryDirectory`` that is removed after reading
    (AC6.9: no ``data/scratch/returns_*`` leaks)."""
    import tempfile

    from vlapoc.catalog import sha256_file
    from vlapoc.runs import client

    c = client(lineage_root)
    run = c.get_run(eval_run_id)
    name = run.data.tags.get("output.rollouts.artifact", "rollouts.parquet")
    with tempfile.TemporaryDirectory(prefix="returns_", dir=REPO_ROOT / "data" / "scratch") as tmp:
        path = Path(c.download_artifacts(eval_run_id, name, tmp))
        if "output.rollouts.sha256" in run.data.tags and sha256_file(path) != run.data.tags["output.rollouts.sha256"]:
            raise ValueError(f"eval run {eval_run_id}: rollouts artifact differs from its recorded sha256")
        df = pd.read_parquet(path)
    if not make_out_dir:
        return df, None
    out_dir = REPO_ROOT / "data" / "eval" / f"returns-{eval_run_id[:12]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return df, out_dir


def returns_run(eval_run_id: str, name: str, seed: int = 0, lineage_root: Path | None = None, chain: str | None = None) -> dict:
    """Stage ``returns`` (PoC §9.1 Step 3 p. 19, "Define one 'return' per metric for each rollout"): the per-rollout
    and per-operation return tables of one eval run, recorded as an MLflow run whose input is that eval run's
    ``rollouts`` entity and whose outputs are ``returns.parquet`` / ``operation_returns.parquet`` (by sha256), so
    the attribution feature's runs can consume returns by digest and ``trace`` back to the checkpoint and data."""
    from vlapoc.runs import client, stage_run

    ev_run = client(lineage_root).get_run(eval_run_id)
    ro, out_dir = load_eval_rollouts(eval_run_id, lineage_root)
    ret, meta = rollout_returns(ro, seed=seed)
    ops = operation_returns(ro)
    paths = {"returns": out_dir / "returns.parquet", "operation_returns": out_dir / "operation_returns.parquet",
             "returns_meta": out_dir / "returns_meta.json", "metrics_registry": out_dir / "metrics_registry.json"}
    ret.to_parquet(paths["returns"], index=False)
    ops.to_parquet(paths["operation_returns"], index=False)
    paths["returns_meta"].write_text(json.dumps(meta, indent=1))
    paths["metrics_registry"].write_text(json.dumps(metrics_registry(), indent=1, ensure_ascii=False))
    with stage_run("returns", name, lineage_root, chain=chain) as r:
        r.input_entity("rollouts", ev_run.data.tags["output.rollouts.sha256"], "rollouts")
        r.params({"eval_run_id": eval_run_id, "auc_sample_seed": seed, "tau_s": TAU_S, "f_tau_s": F_TAU_S,
                  "return_columns": RETURN_COLUMNS, "human_reference": meta["human_reference"]})
        for kind, path in paths.items():
            r.output_file(kind, path, kind=kind)
        r.tags({"returns.rows": len(ret), "returns.operation_rows": len(ops),
                "returns.hrt_macro_at_policy": f"{meta['hrt_macro_at_policy']:.6g}"})
        return {"run_id": r.run_id, "returns": str(paths["returns"].relative_to(REPO_ROOT)), "rows": len(ret),
                "operation_rows": len(ops), "hrt_macro_at_policy": meta["hrt_macro_at_policy"]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m vlapoc.returns")
    p.add_argument("--eval-run", required=True)
    p.add_argument("--seed", type=int, default=0, help="seed for ret_auc_sampled")
    p.add_argument("--track", action="store_true", help="record as a 'returns' stage MLflow run (needs a clean git tree)")
    p.add_argument("--name", default="returns")
    a = p.parse_args(argv)
    if a.track:
        print(json.dumps(returns_run(a.eval_run, a.name, a.seed), indent=1))
        return 0
    ro, out_dir = load_eval_rollouts(a.eval_run)
    ret, meta = rollout_returns(ro, seed=a.seed)
    ops = operation_returns(ro)
    ret.to_parquet(out_dir / "returns.parquet", index=False)
    ops.to_parquet(out_dir / "operation_returns.parquet", index=False)
    (out_dir / "returns_meta.json").write_text(json.dumps(meta, indent=1))
    (out_dir / "metrics_registry.json").write_text(json.dumps(metrics_registry(), indent=1, ensure_ascii=False))
    print(json.dumps({"returns": str((out_dir / "returns.parquet").relative_to(REPO_ROOT)), "rows": len(ret),
                      "operation_rows": len(ops), "columns": RETURN_COLUMNS + ["failure_type"],
                      "hrt_macro_at_policy": meta["hrt_macro_at_policy"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
