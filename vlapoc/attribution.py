"""Metric-resolved CUPID attribution: the episode × metric influence table (replaces ``vlapoc.runs.attribute_standin``).

Grounding
---------
- PoC §4.2 (p. 9): "Instead of one influence score per episode, the PoC computes an influence vector: one score per
  metric ... Results are reported as an episode × metric matrix." §4.3 (p. 10): CUPID-style influence on returns is
  metric-resolved "with one return definition per metric"; the returns are ``vlapoc.returns``'s family-W columns.
- PoC §9.1 Step 3 (p. 19): "For each metric, CUPID-style influence estimates how much each training episode pushed
  that return up or down. Repeat the analysis for subsets, such as a specific task or only rollouts with a specific
  failure type." Columns ``__same_task`` restrict the rollouts to the episode's own task; ``infl_ft_<type>`` use the
  per-failure-type return 1[failure_type = type] (PoC p. 1 "per-metric and per-failure-type returns").
- PoC §4.4 (p. 11): "Agreement check. Report rank correlation (e.g. Kendall's τ) between the per-metric attribution
  vectors" (``agreement``, scipy ``kendalltau``).
- CUPID (docs/papers/CUPID_Influence_Functions_arXiv-2506.19121.pdf) §5.1 p. 5, Prop. 1 / Alg. 1, as quoted in
  ``vlapoc.returns``: Ψ̂(ξ) = (1/m) Σ_i R(τ_i)/H Σ_{(s',a')∈τ_i} Σ_{(s,a)∈ξ} Ψ_a-inf((s,a),(s',a')). The action
  influence is TRAK's: Ψ_a-inf = φ'ᵀ (ΦᵀΦ/N + λI)⁻¹ φ with projected gradients φ (``vlapoc.influence``).
- PoC §2.1 item 4 (p. 3): "Deliberately corrupt known subsets of training data ... Then check whether your
  attribution method flags exactly those." Owner decision 3 (ground-truth feature): a method under test reads data
  only through ``vlapoc.method_data.open_for_method``. ``attribute_run`` opens the training subset that way before
  it creates any run: integrity hash (``VersionIntegrityError``), fence/blindness, ground-truth-derived subsets and
  the one-lineage-per-process diff guard are all enforced (training-attribution S1).

The estimator, and how this implementation evaluates it
-------------------------------------------------------
Ψ_a-inf is bilinear in the two gradients, so the rollout × episode matrix is exact from per-group feature sums:
M[τ, ξ] = (Σ_{j∈τ} φ_j)ᵀ H⁻¹ (Σ_{i∈ξ} φ_i), one Cholesky solve for all pairs. Then for a return column R,
Ψ_R(ξ) = (1/m) Σ_τ (R(τ) − R̄) M[τ, ξ].

What is CUPID's (training-attribution S2; each line checked against the paper or the released code):
- **Sum over steps, no 1/H.** CUPID's Prop. 1 divides by the horizon H only "from the assumption that all
  trajectories have equal length" (§5.1 p. 5, footnote); the variable-length case, §D.2 p. 27 Prop. 2, is
  Ψ_π-inf(ξ) = E_τ[R(τ) Σ_{(s',a')∈τ} Σ_{(s,a)∈ξ} Ψ_a-inf] with no 1/H. A rollout's sum over its own steps
  (``rollout_mean=False``) is therefore CUPID's own variable-length form, not a departure from it.
- **The output function** is ℓ_square = E‖v_θ‖² on both sides by default (``vlapoc.influence`` docstring), with H the
  Gauss-Newton matrix of the training-side features (Eq. 9; ΦᵀΦ/N is its 1/(|D|H) normalisation).
- ``performance_influence(M, 2·r01 − 1, center=False) · m`` equals CUPID's released
  ``compute_demo_quality_scores(sum_of_sum, success_mask, "net")`` (AC2.2, against the vendored copy). **Since the owner's ruling
  of 2026-10-22 ("CUPID's +1/−1") this is the default for every yes/no column** (``return_encoding="pm1"``: the success, timed-out and
  failure-type columns); only the continuous metrics are centred.

What differs (``DIVERGENCES`` below, logged in every run's params), notably:
- **Centred returns for the continuous metrics (inferred baseline subtraction; not CUPID's).** For a yes/no column the
  centred form below is what ``--raw-returns`` gives; the default is CUPID's. CUPID uses R = ±1 with no centring (App. A.2 p. 19;
  code ``net`` = Σ_succ M − Σ_fail M). With p the success fraction over m rollouts and ms, mf the mean of M over
  successful and failed rollouts: centred Σ(R−R̄)M = p(1−p)·m·(ms − mf), whereas CUPID's net = m·[p·ms − (1−p)·mf]
  = m·[p(ms − mf) + (2p − 1)·mf]. The two rank episodes alike only when p = ½ or mf is constant across episodes;
  otherwise net also ranks by how much an episode moves the average *failed* rollout (AC2.3 demonstrates a
  counterexample). The earlier statement that centring "is CUPID's contrast scaled by p(1−p)" holds for a difference
  of *means*; CUPID's code takes a difference of *sums*. ``return_encoding="pm1"`` (the default) is CUPID's
  encoding for the binary returns; ``--raw-returns`` restores centring for them.
- Fewer draws, a Kronecker projection, damping and sample strides (see ``DIVERGENCES``).

Sign convention: Ψ_R(ξ) > 0 means episode ξ pushes return R up. Every family-W return is "higher is better" except
``ret_censored60`` (1 = timed out) and the per-failure-type indicators, where positive means harmful
(``GOOD_SIGN``). ``harm_score`` folds that in, so "most harmful" is always the most negative harm score.

Declared functionals (training-attribution S3; PoC §3.4 p. 7 "Every attribution result states which functional it explains",
§4.4 p. 10 "Label every claim with its metric and time cap τ"): ``column_declarations`` writes one row per score column
(return, family, τ, rollout subset, sign, centring, encoding, output functions) as ``functionals.json``, logged by
``attribute_run`` and ``ensemble_run`` and copied into the ranked column's declaration by ``select_run``.

Where it sits (docs/dataflow.md): Step 3 (Attribution), item 2. Inputs: a train run (its training subset, the rows,
and its checkpoint), the eval run of that checkpoint with ``record_trajectories`` (the rollouts' (s', a')), and the
returns run of that eval (Step 3, item 1). Output: ``attribution_scores.parquet`` keyed by (``episode_id``,
``content_hash``), the ``returns.attribution_frame`` contract, which ``vlapoc.runs.select_run`` consumes for Step 4.
``ensemble_run`` averages per-seed tables (z-scored per seed) into one ranking for the counterfactual retraining.

Usage::

    .venv/bin/python -m vlapoc.attribution --train-run <id> --eval-run <id> --returns-run <id> \\
        --train-features data/attribution/feats/<train>/train --rollout-features data/attribution/feats/<train>/rollouts \\
        --name attribute-<train name>
    .venv/bin/python -m vlapoc.attribution ensemble --runs <attr id> <attr id> <attr id> --name attribute-ensemble-<name>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from vlapoc.libero_convert import REPO_ROOT  # noqa: E402
from vlapoc.metrics_registry import return_columns  # noqa: E402

RETURN_COLUMNS = return_columns("W")  # ret_f30, ret_auc_expected, ret_auc_sampled, ret_rmst60, ret_hrt60_lin, ret_censored60
GOOD_SIGN = {**{c: 1.0 for c in RETURN_COLUMNS}, "ret_censored60": -1.0}  # +1: higher return is better
TASK_KEY = ("suite", "libero_task_id")
DEFAULT_OUT = REPO_ROOT / "data" / "attribution"
MIN_FAILURE_TYPE_ROLLOUTS = 5  # inferred (not from CUPID or the PoC): a failure type needs this many rollouts to get its own column
RETURN_ENCODINGS = ("raw", "pm1")  # pm1: binary returns {0,1} -> {-1,+1}, uncentred (CUPID App. A.2 p. 19)

# Every remaining departure from CUPID (Agia et al. 2025, docs/papers/CUPID_Influence_Functions_arXiv-2506.19121.pdf,
# and the released code agiachris/cupid @ 2941eba9). Logged as a run param by ``attribute_run`` so each score table
# carries its own list of caveats (PoC §3.4 p. 7: say which functional a result explains).
DIVERGENCES = {
    "centred_returns": "centring (R − mean R, an inferred baseline subtraction) now applies only to the continuous columns (RMST, AUC, "
                       "HRT); the yes/no columns use CUPID's R = ±1, uncentred (App. A.2 p. 19; error_util.compute_demo_quality_scores "
                       "'net' = Σ_succ M − Σ_fail M), the owner's ruling of 2026-10-22. For reference, centred = p(1−p)·m·(ms − mf) vs "
                       "net = m·[p·ms − (1−p)·mf]: same ranking only at p = ½ (module docstring). --raw-returns restores centring "
                       "for the yes/no columns.",
    "continuous_returns": "ours: one return per PoC metric, mostly continuous (RMST, AUC, HRT), per §4.3 p. 10; CUPID: binary "
                          "success/failure only. pm1 applies to binary columns only; the continuous ones are centred.",
    "draws": "ours: K = 4 shared (noise, time) draws averaged into each feature; CUPID: B = 64 (ε, i) samples (App. A.2 p. 19).",
    "projection": "ours: Kronecker per-layer projection, rank 8 per side, 7,488 dims over the trainable action-expert Linear "
                  "layers (LoGra/ATHENA §4.1 route); CUPID: dense TRAK projection d = 4000 over all parameters "
                  "(train_trak.sh proj_dim=4000).",
    "damping": "ours: λ = 1e-3 × mean diag(ΦᵀΦ/N); CUPID: lambda_reg = 0.0 (train_trak.sh).",
    "strides": "ours: train frames every 2nd, rollout frames every 4th (stage.py defaults, stored in the manifest); CUPID: "
               "every sample.",
    "numerics": "ours: features use vlapoc.runs.load_checkpoint's upcast-to-fp32 weights, the eval stage's numerics "
                "(G3b: median 13.5% feature difference from LeRobot's bf16 loader, attribution-curation log).",
    "model_family": "ours: SmolVLA flow matching (v_θ = action_out_proj); CUPID: diffusion-policy ε-prediction "
                    "(modelout_functions.py 'square' = pred² on ε_θ). ATHENA Eq. 14-15 p. 14 gives the flow-matching form.",
    "failure_type_returns": "ours: 1[failure_type = t] over all rollouts, only types with ≥ MIN_FAILURE_TYPE_ROLLOUTS (inferred "
                            "threshold); PoC §9.1 Step 3 says 'only rollouts with a specific failure type', which is the "
                            "__same_task-style subset of rollouts, not an indicator return.",
}


# ---------------------------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------------------------


def load_features(prefix: Path) -> tuple[list[np.ndarray], dict, list[dict]]:
    """A feature prefix written by ``vlapoc.influence`` (single file or shards, in shard order): the memmaps, the
    concatenated per-sample meta and the manifests."""
    prefix = Path(prefix)
    if Path(f"{prefix}.npy").exists():
        stems = [str(prefix)]
    else:
        stems = sorted(glob.glob(f"{prefix}.shard*.npy"), key=lambda f: int(f.rsplit(".shard", 1)[1].split(".")[0]))
        stems = [s[: -len(".npy")] for s in stems]
    if not stems:
        raise FileNotFoundError(f"no features at {prefix}")
    feats = [np.load(f"{s}.npy", mmap_mode="r") for s in stems]
    metas = [np.load(f"{s}.meta.npz") for s in stems]
    manifests = [json.loads(Path(f"{s}.manifest.json").read_text()) for s in stems]
    meta = {k: np.concatenate([m[k] for m in metas]) for k in ("episode_index", "frame_index", "loss")}
    n = [m["shard"][1] for m in manifests]
    if len(set(n)) != 1 or n[0] != len(stems):
        raise ValueError(f"{prefix}: found {len(stems)} shard files, manifests say {sorted(set(n))}")
    output_fn_of(manifests, str(prefix))
    return feats, meta, manifests


def output_fn_of(manifests: list[dict], where: str = "feature set") -> str:
    """The one output function of a set of shard manifests (``"loss"`` for a v1 manifest without the key);
    ``ValueError`` if the shards mix two (CUPID's Eq. 9 Hessian is the Gauss-Newton matrix of *one* output function)."""
    fns = {m.get("output_fn", "loss") for m in manifests}
    if len(fns) != 1:
        raise ValueError(f"{where}: shards mix output functions {sorted(fns)}")
    return fns.pop()


def check_compatible(*manifest_lists: list[dict]) -> dict:
    """Raise unless every manifest shares checkpoint, projection seed, rank, K and layer list, and each list (one
    side: train, or rollouts) uses a single output function; return the shared part plus ``output_fns`` (one per list).

    The train side and the rollout side may use different output functions (ATHENA's ``loss``/``square``), but within
    a side the Hessian must come from one (training-attribution S2).
    """
    keys = ("checkpoint_sha256", "seed", "rank", "k_draws", "dim", "layers")
    first = manifest_lists[0][0]
    for ms in manifest_lists:
        for m in ms:
            bad = [k for k in keys if m[k] != first[k]]
            if bad:
                raise ValueError(f"feature manifests disagree on {bad} ({m['dataset_root']} vs {first['dataset_root']})")
    fns = [output_fn_of(ms, f"manifest list {i}") for i, ms in enumerate(manifest_lists)]
    return {k: first[k] for k in keys if k != "layers"} | {"n_layers": len(first["layers"]), "output_fns": fns}


def _torch_device(device: str | None):
    import torch

    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def group_sums(feats: list[np.ndarray], groups: np.ndarray, n_groups: int, device, mean: bool = False):
    """Per-group sum (or mean) of feature rows, streamed over shards; ``groups`` are row ids in [0, n_groups)."""
    import torch

    out = torch.zeros(n_groups, feats[0].shape[1], device=device, dtype=torch.float64)
    counts = torch.zeros(n_groups, device=device, dtype=torch.float64)
    row = 0
    for f in feats:
        for s in range(0, len(f), 8192):
            x = torch.from_numpy(np.asarray(f[s: s + 8192], dtype=np.float32)).to(device, torch.float64)
            g = torch.from_numpy(np.asarray(groups[row + s: row + s + len(x)], dtype=np.int64)).to(device)
            out.index_add_(0, g, x)
            counts.index_add_(0, g, torch.ones(len(x), device=device, dtype=torch.float64))
        row += len(f)
    return out / counts.clamp_min(1)[:, None] if mean else out


def gram(feats: list[np.ndarray], device):
    """ΦᵀΦ / N over the training features (the Gauss-Newton Hessian TRAK and CUPID use)."""
    import torch

    dim = feats[0].shape[1]
    h = torch.zeros(dim, dim, device=device, dtype=torch.float64)
    n = 0
    for f in feats:
        for s in range(0, len(f), 8192):
            x = torch.from_numpy(np.asarray(f[s: s + 8192], dtype=np.float32)).to(device, torch.float64)
            h += x.T @ x
            n += len(x)
    return h / n


def influence_matrix(train_feats: list[np.ndarray], train_groups: np.ndarray, n_episodes: int,
                     rollout_feats: list[np.ndarray], rollout_groups: np.ndarray, n_rollouts: int,
                     damping: float = 1e-3, rollout_mean: bool = False, device: str | None = None) -> np.ndarray:
    """M[τ, ξ] = (Σ_{j∈τ} φ_j)ᵀ (ΦᵀΦ/N + λI)⁻¹ (Σ_{i∈ξ} φ_i), shape (n_rollouts, n_episodes).

    λ = ``damping`` × mean(diag ΦᵀΦ/N). Group ids index rows: ``train_groups[i]`` is sample i's episode row,
    ``rollout_groups[j]`` sample j's rollout row."""
    import torch

    dev = _torch_device(device)
    h = gram(train_feats, dev)
    h.diagonal().add_(damping * torch.diagonal(h).mean())
    chol = torch.linalg.cholesky(h)
    demo = group_sums(train_feats, train_groups, n_episodes, dev)
    roll = group_sums(rollout_feats, rollout_groups, n_rollouts, dev, mean=rollout_mean)
    return (roll @ torch.cholesky_solve(demo.T, chol)).cpu().numpy()


# ---------------------------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------------------------


def is_binary(r: np.ndarray) -> bool:
    """True if every value of ``r`` is 0 or 1 (a success/failure or failure-type indicator return)."""
    return bool(np.isin(np.unique(r), (0.0, 1.0)).all())


def column_plan(returns: dict[str, np.ndarray], center: bool = True, return_encoding: str = "pm1") -> dict[str, dict]:
    """Per return column: the ``encoding`` actually applied (``raw`` or ``pm1``) and whether it is ``centred``.

    ``pm1`` (CUPID's R ∈ {+1, −1}, uncentred; App. A.2 p. 19) is the default and applies to binary columns (success / timed-out /
    failure-type indicators); every other column stays raw and is centred iff ``center``. ``"raw"`` restores the earlier behaviour
    (binary columns centred like the rest). Owner ruling 2026-10-22 "CUPID's +1/−1" made ``pm1`` the default; centring stays
    only for continuous metrics. Used by ``score_table`` and by ``column_declarations`` so the declared and the applied agree.
    """
    if return_encoding not in RETURN_ENCODINGS:
        raise ValueError(f"return_encoding must be one of {RETURN_ENCODINGS}, got {return_encoding!r}")
    out = {}
    for name, r in returns.items():
        enc = "pm1" if return_encoding == "pm1" and is_binary(r) else "raw"
        out[name] = {"encoding": enc, "centred": bool(center) and enc == "raw"}
    return out


def performance_influence(m: np.ndarray, r: np.ndarray, mask: np.ndarray | None = None, center: bool = True,
                          return_encoding: str = "raw") -> np.ndarray:
    """Ψ_R(ξ) = (1/m) Σ_τ (R(τ) − R̄) M[τ, ξ] over the rollouts in ``mask`` (all by default); one value per column of M.

    ``return_encoding="pm1"`` is CUPID's R(τ) = +1 if the rollout succeeds and −1 otherwise (App. A.2 p. 19; Prop. 1
    with raw R, no centring): ``r`` must be binary {0, 1} and ``center`` False (``ValueError`` otherwise). Then
    ``Ψ · m`` is CUPID's ``compute_demo_quality_scores(sum_of_sum, success_mask, "net")`` (AC2.2).
    """
    if return_encoding not in RETURN_ENCODINGS:
        raise ValueError(f"return_encoding must be one of {RETURN_ENCODINGS}, got {return_encoding!r}")
    if return_encoding == "pm1":
        if center:
            raise ValueError("return_encoding='pm1' is CUPID's uncentred ±1 return; pass center=False")
        if not is_binary(r):
            raise ValueError("return_encoding='pm1' needs a binary {0, 1} return")
        r = 2.0 * np.asarray(r, float) - 1.0
    if mask is not None:
        m, r = m[mask], r[mask]
    if len(r) == 0:
        return np.full(m.shape[1], np.nan)
    adv = r - r.mean() if center else r
    return adv @ m / len(r)


def returns_of(ret: pd.DataFrame) -> dict[str, np.ndarray]:
    """The per-rollout return vectors that get a score column: every family-W return and every ``ft_<type>`` indicator."""
    returns = {c: ret[c].to_numpy(float) for c in RETURN_COLUMNS}
    returns.update(failure_type_returns(ret))
    return returns


def failure_type_returns(ret: pd.DataFrame, min_rollouts: int = MIN_FAILURE_TYPE_ROLLOUTS) -> dict[str, np.ndarray]:
    """``ft_<type>`` = 1[failure_type == type] for every non-success type with at least ``min_rollouts`` rollouts."""
    counts = ret["failure_type"].value_counts()
    return {f"ft_{t}": (ret["failure_type"] == t).to_numpy(float) for t in sorted(counts.index)
            if t != "success" and counts[t] >= min_rollouts}


def score_table(m: np.ndarray, episodes: pd.DataFrame, ret: pd.DataFrame, center: bool = True,
                return_encoding: str = "pm1") -> pd.DataFrame:
    """The episode × metric table. ``episodes`` (rows of M's columns, in order) carries ``episode_id``,
    ``content_hash`` and the task key; ``ret`` (rows of M's rows, in order) is the returns table.

    Columns: ``infl_<ret col>`` over all rollouts and ``infl_<ret col>__same_task`` over rollouts of the episode's own
    task, for every family-W return and every per-failure-type return; ``harm_<...>`` = −GOOD_SIGN × infl (most
    harmful = largest). ``return_encoding="pm1"`` applies CUPID's ±1 uncentred return to the binary columns
    (``column_plan``); the others stay raw.
    """
    returns = returns_of(ret)
    plan = column_plan(returns, center, return_encoding)
    out = episodes[["episode_id", "content_hash", "episode_index", *TASK_KEY, "task_name"]].reset_index(drop=True).copy()
    task_of_rollout = list(zip(ret["suite"], ret["libero_task_id"].astype(int)))
    task_of_episode = list(zip(out["suite"], out["libero_task_id"].astype(int)))
    for name, r in returns.items():
        col = f"infl_{name.removeprefix('ret_')}"
        enc, cen = plan[name]["encoding"], plan[name]["centred"]
        out[col] = performance_influence(m, r, center=cen, return_encoding=enc)
        same = np.full(len(out), np.nan)
        for task in sorted(set(task_of_episode)):
            cols = np.array([t == task for t in task_of_episode])
            rows = np.array([t == task for t in task_of_rollout])
            same[cols] = performance_influence(m[:, cols], r, rows, center=cen, return_encoding=enc)
        out[f"{col}__same_task"] = same
    for col in [c for c in out.columns if c.startswith("infl_")]:
        base = col.removeprefix("infl_").split("__")[0]
        sign = GOOD_SIGN.get(f"ret_{base}", -1.0)  # failure-type indicators: higher = more of that failure = harmful
        out[col.replace("infl_", "harm_", 1)] = -sign * out[col]
    return out


def agreement(table: pd.DataFrame, columns: list[str]) -> dict:
    """Kendall's τ (scipy ``kendalltau``, τ-b) between every pair of score columns (PoC §4.4 p. 11)."""
    from scipy.stats import kendalltau

    res = {}
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            ok = table[a].notna() & table[b].notna()
            t = kendalltau(table.loc[ok, a], table.loc[ok, b])
            res[f"{a}|{b}"] = {"tau": float(t.statistic), "p": float(t.pvalue), "n": int(ok.sum())}
    return res


def score_columns(table: pd.DataFrame) -> list[str]:
    """The ``infl_``/``harm_`` columns of a score table, in table order."""
    return [c for c in table.columns if c.startswith(("infl_", "harm_"))]


def zscore_ensemble(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Average of per-table z-scores for every ``infl_``/``harm_`` column (one table per training seed of the same
    subset). z-scoring puts seeds on one scale before averaging; the rows must be the same episode versions.

    The columns are the union over the tables (a failure type can have enough rollouts to get a column in one seed and
    not in another); a column is averaged over the tables that have it, and ``n_models_per_column`` says how many.
    """
    keys = tables[0][["episode_id", "content_hash"]]
    for t in tables[1:]:
        if not t[["episode_id", "content_hash"]].equals(keys):
            raise ValueError("ensemble tables do not key the same episode versions in the same order")
    cols = list(dict.fromkeys(c for t in tables for c in score_columns(t)))
    out = tables[0].drop(columns=score_columns(tables[0])).copy()
    for c in cols:
        z = [(t[c] - t[c].mean()) / t[c].std() for t in tables if c in t]
        out[c] = sum(z) / len(z)
    return out


def n_models_per_column(tables: list[pd.DataFrame]) -> dict[str, int]:
    """How many of the ensemble's tables contain each ``infl_``/``harm_`` column."""
    cols = list(dict.fromkeys(c for t in tables for c in score_columns(t)))
    return {c: sum(c in t for t in tables) for c in cols}


def _tau_s(tau: str, default: float | None) -> float | None:
    """The numeric time cap in a registry ``tau`` string (``"30 s"`` → 30.0, ``"60 s = τ_episode"`` → 60.0), else ``default``."""
    import re

    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*s", tau)
    return float(m.group(1)) if m else default


def weighting_label(p: dict) -> str:
    """Plain statement of how a column's return is weighted: CUPID's ±1, centred, or raw (PoC §3.4 p. 7: say what is explained)."""
    return "CUPID ±1, uncentred (success +1, otherwise −1)" if p["encoding"] == "pm1" else (
        "centred (R − mean R)" if p["centred"] else "raw R, uncentred")


def column_declarations(table: pd.DataFrame, ret: pd.DataFrame, center: bool = True, return_encoding: str = "pm1",
                        output_fns: tuple[str, str] = ("square", "square")) -> list[dict]:
    """One row per ``infl_``/``harm_`` column of ``table``: which functional the column explains (PoC §3.4 p. 7, §4.4 p. 10).

    Fields: ``column``; ``kind`` (``infl``: positive = the episode pushes the return up; ``harm``: positive = harmful,
    i.e. ``-good_sign`` × infl); ``return_column`` and ``return_definition``; ``metric_key``, ``family``,
    ``functional``, ``tau`` (the metric registry's string) and ``tau_s``; ``rollout_subset`` (``all`` rollouts, or
    ``same_task`` = the episode's own task, ATHENA §4.2 Eq. 7); ``good_sign``; ``centred`` and ``return_encoding``
    (what ``column_plan`` applied to this column); ``output_fns`` (train side, rollout side). A per-failure-type
    column's return is the indicator 1[failure_type = t] over the rollouts, family W, τ_episode.
    """
    from vlapoc.metrics_registry import FAMILY_TAU_S, entry, ENTRIES

    plan = column_plan(returns_of(ret), center, return_encoding)
    by_col = {c: e for e in ENTRIES if e["family"] == "W" and e["verdict"] is not None for c in e["return_columns"]}
    rows = []
    for col in score_columns(table):
        kind, _, rest = col.partition("_")
        base, _, suffix = rest.partition("__")
        name = base if base.startswith("ft_") else f"ret_{base}"
        if name not in plan:
            raise ValueError(f"column {col!r} has no return {name!r} in the returns table (cannot declare its functional)")
        if name.startswith("ft_"):
            e = entry("failure_type")
            row = {"return_definition": f"1[failure_type == {name.removeprefix('ft_')!r}] over the rollouts",
                   "metric_key": "failure_type", "functional": f"share of rollouts failing as {name.removeprefix('ft_')!r}",
                   "tau": e["tau"], "tau_s": FAMILY_TAU_S["W"]}
        else:
            e = by_col[name]
            row = {"return_definition": f"per-rollout return {name} (metrics_registry: {e['identity']})",
                   "metric_key": e["key"], "functional": e["functional"], "tau": e["tau"],
                   "tau_s": _tau_s(e["tau"], None)}
        rows.append({"column": col, "kind": kind, "return_column": name, **row, "family": e["family"],
                     "rollout_subset": "same_task" if suffix == "same_task" else "all",
                     "good_sign": GOOD_SIGN.get(name, -1.0), "centred": plan[name]["centred"],
                     "return_encoding": plan[name]["encoding"], "output_fns": list(output_fns),
                     "weighting": weighting_label(plan[name])})
    return rows


# ---------------------------------------------------------------------------------------------
# the attribute stage
# ---------------------------------------------------------------------------------------------


def _artifact_table(run, tag: str, lineage_root: Path | None) -> pd.DataFrame:
    """A run's parquet output ``tag`` (MLflow artifact), checked against its recorded sha256."""
    import tempfile

    from vlapoc.catalog import sha256_file
    from vlapoc.runs import client

    c = client(lineage_root)
    name = run.data.tags[f"output.{tag}.artifact"]
    (REPO_ROOT / "data" / "scratch").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="attr_", dir=REPO_ROOT / "data" / "scratch") as tmp:
        path = Path(c.download_artifacts(run.info.run_id, name, tmp))
        if sha256_file(path) != run.data.tags[f"output.{tag}.sha256"]:
            raise ValueError(f"run {run.info.run_id}: artifact {name} differs from its recorded sha256")
        return pd.read_parquet(path)


def feature_file_digests(prefix: Path, role: str) -> dict[str, str]:
    """``{input role: sha256}`` for every ``.npy`` and ``.meta.npz`` file of a feature prefix (single file or shards).

    The role is ``features.<role>.<suffix>`` (e.g. ``features.train.shard0.npy``), so the attribute run lists one
    ``features`` input per file and a regenerated or altered feature file changes a recorded digest (PoC §2.1 item 1
    p. 3; the feature files are intermediate, not produced by a recorded stage, so ``trace`` shows them as leaf inputs).
    """
    from vlapoc.catalog import sha256_file

    prefix = Path(prefix)
    out = {}
    for f in sorted(prefix.parent.glob(f"{prefix.name}*")):
        suffix = f.name[len(prefix.name):].lstrip(".")
        if f.name.endswith((".npy", ".meta.npz")):
            out[f"features.{role}.{suffix}"] = sha256_file(f)
    if not out:
        raise FileNotFoundError(f"no feature files at {prefix}")
    return out


def _artifact_json(run, tag: str, lineage_root: Path | None):
    """A run's JSON output ``tag`` (MLflow artifact), checked against its recorded sha256."""
    import tempfile

    from vlapoc.catalog import sha256_file
    from vlapoc.runs import client

    c = client(lineage_root)
    if f"output.{tag}.artifact" not in run.data.tags:
        raise ValueError(f"run {run.info.run_id[:12]} has no recorded output {tag!r} "
                         "(it predates training-attribution S3, or is not an attribute run); re-run the attribute stage")
    name = run.data.tags[f"output.{tag}.artifact"]
    (REPO_ROOT / "data" / "scratch").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="attr_", dir=REPO_ROOT / "data" / "scratch") as tmp:
        path = Path(c.download_artifacts(run.info.run_id, name, tmp))
        if sha256_file(path) != run.data.tags[f"output.{tag}.sha256"]:
            raise ValueError(f"run {run.info.run_id}: artifact {name} differs from its recorded sha256")
        return json.loads(path.read_text())


def declaration_of(score_run_id: str, column: str, lineage_root: Path | None = None) -> dict | None:
    """The ``functionals.json`` row of ``column`` in an attribute run, or ``None`` if that run recorded no declarations
    (a stand-in or pre-S3 run). ``ValueError`` if the run declares functionals but not this column."""
    from vlapoc.runs import client

    run = client(lineage_root).get_run(score_run_id)
    if "output.functionals.artifact" not in run.data.tags:
        return None
    rows = [r for r in _artifact_json(run, "functionals", lineage_root) if r["column"] == column]
    if len(rows) != 1:
        raise ValueError(f"run {score_run_id[:12]} declares {len(rows)} functionals for column {column!r}, expected exactly 1")
    return rows[0]


def attribute(train_feats_prefix: Path, rollout_feats_prefix: Path, episodes: pd.DataFrame, rollouts: pd.DataFrame,
              ret: pd.DataFrame, damping: float = 1e-3, center: bool = True, rollout_mean: bool = False,
              device: str | None = None, return_encoding: str = "pm1") -> tuple[pd.DataFrame, np.ndarray, dict]:
    """The untracked core of ``attribute_run``: features + keys → (score table, M, shared feature manifest).

    ``episodes`` are the training subset's catalog rows; ``rollouts`` the eval run's record (``traj_episode_index``
    maps a trajectory episode to its rollout); ``ret`` the returns table (one row per rollout, any order).
    """
    tf, tmeta, tman = load_features(train_feats_prefix)
    rf, rmeta, rman = load_features(rollout_feats_prefix)
    shared = check_compatible(tman, rman)
    episodes = episodes.sort_values("episode_index").reset_index(drop=True)
    row_of_ep = {int(e): i for i, e in enumerate(episodes["episode_index"])}
    unknown = sorted(set(np.unique(tmeta["episode_index"]).tolist()) - set(row_of_ep))
    if unknown:
        raise ValueError(f"train features contain episodes outside the training subset, e.g. {unknown[:3]}")
    missing = sorted(set(row_of_ep) - set(np.unique(tmeta["episode_index"]).tolist()))
    if missing:
        raise ValueError(f"{len(missing)} training episodes have no features, e.g. {missing[:3]}")
    ro = rollouts.dropna(subset=["traj_episode_index"]).copy()
    ro["traj_episode_index"] = ro["traj_episode_index"].astype(int)
    ro = ro.sort_values("traj_episode_index").reset_index(drop=True)
    row_of_traj = {int(e): i for i, e in enumerate(ro["traj_episode_index"])}
    if set(np.unique(rmeta["episode_index"]).tolist()) != set(row_of_traj):
        raise ValueError("rollout features and the eval record's traj_episode_index disagree")
    ret = ro[["rollout_id"]].merge(ret, on="rollout_id", how="left", validate="one_to_one")
    if ret[RETURN_COLUMNS].isna().any().any():
        raise ValueError("returns table lacks some featurized rollouts")
    m = influence_matrix(tf, np.array([row_of_ep[int(e)] for e in tmeta["episode_index"]]), len(episodes),
                         rf, np.array([row_of_traj[int(e)] for e in rmeta["episode_index"]]), len(ro),
                         damping, rollout_mean, device)
    table = score_table(m, episodes, ret, center, return_encoding)
    shared |= {"n_train_samples": int(len(tmeta["episode_index"])), "n_rollout_samples": int(len(rmeta["episode_index"])),
               "train_stride": tman[0]["stride"], "rollout_stride": rman[0]["stride"],
               "train_features": [m_["dataset_root"] for m_ in tman][:1], "rollout_features": [m_["dataset_root"] for m_ in rman][:1]}
    return table, m, shared


def attribute_run(train_run_id: str, eval_run_id: str, returns_run_id: str, train_features: Path, rollout_features: Path,
                  name: str, out_dir: Path | None = None, damping: float = 1e-3, center: bool = True,
                  rollout_mean: bool = False, device: str | None = None, lineage_root: Path | None = None,
                  chain: str | None = None, allow_dirty: bool = False, return_encoding: str = "pm1",
                  allow_partial: bool = False) -> dict:
    """Stage ``attribute`` (PoC §9.1 Step 3 p. 19): CUPID influence of every training episode version on every
    family-W return (and per-failure-type return) of one eval run, recorded as one MLflow run.

    Refuses (``LineageError``/``VersionIntegrityError``/``ValueError``, before any run exists) unless: the training
    subset opens through ``open_for_method`` (intact, not fenced, blind, not ground-truth-derived; PoC §2.1 item 4
    p. 3, owner decision 3); the eval run evaluated this train run's checkpoint and recorded trajectories; the returns run
    consumed that eval's rollouts; both feature sets were computed from this checkpoint with one projection; the train
    features cover exactly the training subset and were computed on that version's own copy (manifest ``dataset_root``), and the
    rollout features were computed on the trajectories copy this eval run recorded (digest equals ``output.trajectories``); and every
    rollout of the eval run has a recorded trajectory, unless ``allow_partial`` (then the count is recorded as param
    ``n_rollouts_without_trajectory``; PoC §2.1 item 1 p. 3 "Every eval rollout should be tied back to that checkpoint").

    Migration note: feature files written before training-attribution round 2 (an absolute or other-version ``dataset_root``
    in the manifest) are refused by the two data-binding checks above; regenerate them with ``scripts/stage.py features``.
    Rewriting a manifest's ``dataset_root`` by hand defeats the check, because the manifest is then no longer evidence of
    what the features were computed on.

    Records (training-attribution S3, PoC §2.1 item 1 p. 3 and §3.4 p. 7): the sha256 of every feature ``.npy`` and
    ``.meta.npz`` file it read as inputs of kind ``features`` (so a regenerated feature file is distinguishable), the
    estimator's settings and ``DIVERGENCES`` as params, and ``functionals.json`` (``column_declarations``) as an output.
    Returns ``{"run_id", "scores", "n_episodes", "n_rollouts", "agreement"}``.
    """
    from vlapoc.lineage import file_digests, rel, sha256_json
    from vlapoc.method_data import open_for_method
    from vlapoc.returns import load_eval_rollouts
    from vlapoc.runs import _training_subset, check_scores, client, stage_run

    c = client(lineage_root)
    tr = c.get_run(train_run_id)
    subset = _training_subset(tr, lineage_root)
    view = open_for_method(subset.subset_id, lineage_root)  # refuses fenced / leaky / ground-truth-derived / tampered input
    ev, rr = c.get_run(eval_run_id), c.get_run(returns_run_id)
    ckpt_sha = tr.data.tags["checkpoint.weights.sha256"]
    if ev.data.params.get("train_run_id") != train_run_id or ev.data.params.get("checkpoint_sha256") != ckpt_sha:
        raise ValueError(f"eval run {eval_run_id[:12]} did not evaluate train run {train_run_id[:12]}'s checkpoint")
    if "output.trajectories" not in ev.data.tags:
        raise ValueError(f"eval run {eval_run_id[:12]} recorded no trajectories (record_trajectories=True is required)")
    if rr.data.tags.get("input.rollouts") != f"rollouts:{ev.data.tags['output.rollouts.sha256']}":
        raise ValueError(f"returns run {returns_run_id[:12]} did not consume eval run {eval_run_id[:12]}'s rollouts")
    for prefix in (train_features, rollout_features):
        _, _, mans = load_features(prefix)
        if any(m["checkpoint_sha256"] != ckpt_sha for m in mans):
            raise ValueError(f"features {prefix} were computed from another checkpoint")
    # Bind each feature set to the data it claims (PoC §2.1 item 1 p. 3, "tied to exact dataset versions"; §9.2 p. 19): the train
    # features to the training version's own copy (the root ``open_for_method`` returned), the rollout features to the
    # trajectories this eval run recorded (the digest check ``stage.py features`` makes when it writes them).
    regen = ("; feature files written before training-attribution round 2 must be regenerated with `stage.py features` "
             "(editing a manifest's dataset_root by hand defeats this check)")
    vroot = rel(view.lerobot_root)
    _, _, tm0 = load_features(train_features)
    wrong = sorted({m["dataset_root"] for m in tm0} - {vroot})
    if wrong:
        raise ValueError(f"train features were computed on {wrong}, not on the training subset's version copy {vroot}{regen}")
    _, _, rm0 = load_features(rollout_features)
    roots = {m["dataset_root"] for m in rm0}
    recorded = ev.data.tags["output.trajectories"].split(":", 1)[-1]
    if len(roots) != 1 or sha256_json(file_digests(REPO_ROOT / next(iter(roots)))) != recorded:
        raise ValueError(f"rollout features were computed on {sorted(roots)}, which is not the trajectories copy eval run "
                         f"{eval_run_id[:12]} recorded{regen}")
    if sorted(tm0[0]["episodes"] or []) != subset.resolve(lineage_root):
        raise ValueError("train features were not computed on exactly the training subset's episodes")
    episodes = subset.episodes(lineage_root)
    rollouts, _ = load_eval_rollouts(eval_run_id, lineage_root, make_out_dir=False)
    n_no_traj = int(rollouts["traj_episode_index"].isna().sum())
    if n_no_traj and not allow_partial:
        raise ValueError(f"{n_no_traj} of {len(rollouts)} rollouts of eval run {eval_run_id[:12]} have no recorded trajectory, so "
                         "they would be dropped from the attribution silently; pass allow_partial=True to accept and record it")
    ret = _artifact_table(rr, "returns", lineage_root)
    feature_digests = {**feature_file_digests(train_features, "train"), **feature_file_digests(rollout_features, "rollouts")}
    table, m, shared = attribute(train_features, rollout_features, episodes, rollouts, ret, damping, center,
                                 rollout_mean, device, return_encoding)
    check_scores(table, train_run_id, lineage_root)
    cols = [f"infl_{c.removeprefix('ret_')}__same_task" for c in RETURN_COLUMNS]
    agree = agreement(table, cols)
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"attribution_scores": out_dir / "attribution_scores.parquet", "agreement": out_dir / "agreement.json",
             "influence_matrix": out_dir / "rollout_episode_influence.npy", "functionals": out_dir / "functionals.json"}
    table.to_parquet(paths["attribution_scores"], index=False)
    paths["agreement"].write_text(json.dumps(agree, indent=1))
    np.save(paths["influence_matrix"], m)
    decls = column_declarations(table, ret, center, return_encoding, tuple(shared["output_fns"]))
    paths["functionals"].write_text(json.dumps(decls, indent=1, ensure_ascii=False))
    with stage_run("attribute", name, lineage_root, chain=chain, allow_dirty=allow_dirty) as r:
        r.input_subset(subset, "training", None)
        r.input_entity("checkpoint", ckpt_sha, "checkpoint")
        r.input_entity("rollouts", ev.data.tags["output.rollouts.sha256"], "rollouts")
        r.input_entity("returns", rr.data.tags["output.returns.sha256"], "returns")
        for role, sha in feature_digests.items():
            r.input_entity("features", sha, role)
        r.params({"method": "CUPID influence on returns (TRAK action influence, Kronecker-projected gradients)",
                  "train_run_id": train_run_id, "eval_run_id": eval_run_id, "returns_run_id": returns_run_id,
                  "damping": damping, "center_returns": center, "return_encoding": return_encoding,
                  "return_weighting": {"binary_columns": "CUPID ±1, uncentred" if return_encoding == "pm1" else
                                       ("centred" if center else "raw"),
                                       "continuous_columns": "centred (R − mean R)" if center else "raw"},
                  "rollout_mean": rollout_mean,
                  "train_output_fn": shared["output_fns"][0], "rollout_output_fn": shared["output_fns"][1],
                  "divergences_from_cupid": DIVERGENCES, "min_failure_type_rollouts_inferred": MIN_FAILURE_TYPE_ROLLOUTS,
                  "return_columns": RETURN_COLUMNS, "good_sign": GOOD_SIGN,
                  "features": {k: v for k, v in shared.items() if k not in ("train_features", "rollout_features")},
                  "train_features": rel(Path(train_features)), "rollout_features": rel(Path(rollout_features)),
                  "n_rows": len(table), "n_rollouts": int(m.shape[0]), "allow_partial": allow_partial,
                  "n_rollouts_without_trajectory": n_no_traj, "n_feature_files": len(feature_digests)})
        for prefix, role in ((train_features, "train"), (rollout_features, "rollouts")):
            for mf in sorted(Path(prefix).parent.glob(f"{Path(prefix).name}*.manifest.json")):
                import mlflow

                mlflow.log_artifact(str(mf), artifact_path=f"features/{role}")
        r.output_file("attribution_scores", paths["attribution_scores"], kind="score_table")
        r.output_file("agreement", paths["agreement"], kind="agreement")
        r.output_file("functionals", paths["functionals"], kind="functionals")
        r.tags({"attribute.n_rollouts": int(m.shape[0]), "attribute.n_episodes": len(table)})
        return {"run_id": r.run_id, "scores": rel(paths["attribution_scores"]), "n_episodes": len(table),
                "n_rollouts": int(m.shape[0]), "agreement": agree}


ENSEMBLE_SAME_PARAMS = ("damping", "center_returns", "return_encoding", "train_output_fn", "rollout_output_fn")


def ensemble_run(attribute_run_ids: list[str], name: str, out_dir: Path | None = None, lineage_root: Path | None = None,
                 chain: str | None = None, allow_dirty: bool = False) -> dict:
    """Stage ``attribute`` (ensemble): z-scored average of per-seed attribution tables of one training subset.

    Grounding: PoC §2.3 p. 4 "2–3 seeds per data configuration", so the selection for Step 4 does not rest on one
    seed's noise (averaging influence over independently trained models is TRAK's ensembling, inferred to transfer).
    Refuses (``ValueError``, before any run) duplicate input runs, inputs that are themselves ensembles, inputs trained on
    different subsets, inputs whose ``damping`` / ``center_returns`` / ``return_encoding`` / output functions differ
    (their columns would not be the same functional), inputs without ``functionals.json``, and inputs that declare one
    column differently (PoC §3.4 p. 7). Records ``functionals.json`` for the ensemble with ``n_models`` per column
    (a failure-type column can exist in only some seeds). Returns ``{"run_id", "scores", "n_models"}``.
    """
    from vlapoc.lineage import Subset, rel
    from vlapoc.runs import client, stage_run

    if len(set(attribute_run_ids)) != len(attribute_run_ids):
        raise ValueError(f"ensemble inputs contain duplicate runs: {attribute_run_ids}")
    c = client(lineage_root)
    runs = [c.get_run(i) for i in attribute_run_ids]
    for r_ in runs:
        if r_.data.tags.get("stage") != "attribute" or "output.attribution_scores.sha256" not in r_.data.tags:
            raise ValueError(f"run {r_.info.run_id[:12]} is not an attribute run with a score table")
        if "attribute_run_ids" in r_.data.params:
            raise ValueError(f"run {r_.info.run_id[:12]} is itself an ensemble; ensembling z-scores twice would double-count seeds")
    subsets = {r.data.tags["input.training"] for r in runs}
    if len(subsets) != 1:
        raise ValueError(f"ensemble inputs were trained on different subsets: {sorted(subsets)}")
    for key in ENSEMBLE_SAME_PARAMS:
        vals = {r_.data.params.get(key) for r_ in runs}
        if len(vals) != 1:
            raise ValueError(f"ensemble inputs differ in {key!r}: {sorted(map(str, vals))}; their columns explain different functionals")
    decls = [_artifact_json(r_, "functionals", lineage_root) for r_ in runs]
    tables = [_artifact_table(r_, "attribution_scores", lineage_root) for r_ in runs]
    by_col: dict[str, dict] = {}
    for d in decls:
        for row in d:
            if by_col.setdefault(row["column"], row) != row:
                raise ValueError(f"ensemble inputs declare column {row['column']!r} differently")
    ens = zscore_ensemble(tables)
    n_models = n_models_per_column(tables)
    declared = [{**row, "n_models": n_models[col], "aggregation": "mean over seeds of per-seed z-scores (unit: seed s.d.)"}
                for col, row in by_col.items()]
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "attribution_scores.parquet"
    ens.to_parquet(path, index=False)
    fpath = out_dir / "functionals.json"
    fpath.write_text(json.dumps(declared, indent=1, ensure_ascii=False))
    subset = Subset.load(subsets.pop().split(":", 1)[1], lineage_root)
    with stage_run("attribute", name, lineage_root, chain=chain, allow_dirty=allow_dirty) as r:
        r.input_subset(subset, "training", None)
        for i, run in enumerate(runs):
            r.input_entity("score_table", run.data.tags["output.attribution_scores.sha256"], f"seed_table_{i}")
        r.params({"method": "ensemble: mean over seeds of per-seed z-scored influence columns",
                  "attribute_run_ids": attribute_run_ids, "n_models": len(runs), "n_rows": len(ens),
                  "n_models_per_column": n_models,
                  "aggregation": "mean over seeds of per-seed z-scored columns (unit: seed s.d.); a column present in only some seeds is "
                                 "averaged over those (n_models_per_column)", "divergences_from_cupid": DIVERGENCES,
                  **{k: runs[0].data.params.get(k) for k in ENSEMBLE_SAME_PARAMS}})
        r.output_file("attribution_scores", path, kind="score_table")
        r.output_file("functionals", fpath, kind="functionals")
        return {"run_id": r.run_id, "scores": rel(path), "n_models": len(runs)}


def main(argv: list[str] | None = None) -> int:
    """CLI (usage in the module docstring)."""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "ensemble":
        p = argparse.ArgumentParser(prog="python -m vlapoc.attribution ensemble")
        p.add_argument("--runs", nargs="+", required=True)
        p.add_argument("--name", required=True)
        a = p.parse_args(argv[1:])
        print(json.dumps(ensemble_run(a.runs, a.name), indent=1))
        return 0
    p = argparse.ArgumentParser(prog="python -m vlapoc.attribution")
    p.add_argument("--train-run", required=True)
    p.add_argument("--eval-run", required=True)
    p.add_argument("--returns-run", required=True)
    p.add_argument("--train-features", type=Path, required=True)
    p.add_argument("--rollout-features", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--damping", type=float, default=1e-3)
    p.add_argument("--no-center", action="store_true", help="CUPID Prop. 1 with raw returns")
    p.add_argument("--raw-returns", action="store_true",
                   help="centre the yes/no columns like the continuous ones (the pre-2026-10-22 behaviour); default is CUPID's ±1")
    p.add_argument("--allow-partial", action="store_true",
                   help="accept rollouts that have no recorded trajectory (recorded as a count)")
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    out = attribute_run(a.train_run, a.eval_run, a.returns_run, a.train_features, a.rollout_features, a.name,
                        damping=a.damping, center=not a.no_center, device=a.device,
                        return_encoding="raw" if a.raw_returns else "pm1", allow_partial=a.allow_partial)
    print(json.dumps({k: v for k, v in out.items() if k != "agreement"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
