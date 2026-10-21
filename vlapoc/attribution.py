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
  ``compute_demo_quality_scores(sum_of_sum, success_mask, "net")`` (AC2.2, against the vendored copy).

What differs (``DIVERGENCES`` below, logged in every run's params), notably:
- **Centred returns (inferred baseline subtraction; not CUPID's).** CUPID uses R = ±1 with no centring (App. A.2 p. 19;
  code ``net`` = Σ_succ M − Σ_fail M). With p the success fraction over m rollouts and ms, mf the mean of M over
  successful and failed rollouts: centred Σ(R−R̄)M = p(1−p)·m·(ms − mf), whereas CUPID's net = m·[p·ms − (1−p)·mf]
  = m·[p(ms − mf) + (2p − 1)·mf]. The two rank episodes alike only when p = ½ or mf is constant across episodes;
  otherwise net also ranks by how much an episode moves the average *failed* rollout (AC2.3 demonstrates a
  counterexample). The earlier statement that centring "is CUPID's contrast scaled by p(1−p)" holds for a difference
  of *means*; CUPID's code takes a difference of *sums*. ``return_encoding="pm1"`` (``--cupid-returns``) gives CUPID's
  encoding for the binary returns.
- Fewer draws, a Kronecker projection, damping and sample strides (see ``DIVERGENCES``).

Sign convention: Ψ_R(ξ) > 0 means episode ξ pushes return R up. Every family-W return is "higher is better" except
``ret_censored60`` (1 = timed out) and the per-failure-type indicators, where positive means harmful
(``GOOD_SIGN``). ``harm_score`` folds that in, so "most harmful" is always the most negative harm score.

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
    "centred_returns": "ours: Ψ uses R − mean(R) by default (inferred baseline subtraction); CUPID: R = ±1, no centring "
                       "(App. A.2 p. 19; error_util.compute_demo_quality_scores 'net' = Σ_succ M − Σ_fail M). centred = "
                       "p(1−p)·m·(ms − mf) vs net = m·[p·ms − (1−p)·mf]; same ranking only at p = ½ (module docstring). "
                       "--cupid-returns / return_encoding='pm1' gives CUPID's.",
    "continuous_returns": "ours: one return per PoC metric, mostly continuous (RMST, AUC, HRT), per §4.3 p. 10; CUPID: binary "
                          "success/failure only. pm1 applies to binary columns only; the rest stay raw.",
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


def column_plan(returns: dict[str, np.ndarray], center: bool = True, return_encoding: str = "raw") -> dict[str, dict]:
    """Per return column: the ``encoding`` actually applied (``raw`` or ``pm1``) and whether it is ``centred``.

    ``pm1`` (CUPID's R ∈ {+1, −1}, uncentred) applies to binary columns; every other column stays raw and is centred
    iff ``center``. Used by ``score_table`` and by ``column_declarations`` so the declared and the applied agree.
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


def failure_type_returns(ret: pd.DataFrame, min_rollouts: int = MIN_FAILURE_TYPE_ROLLOUTS) -> dict[str, np.ndarray]:
    """``ft_<type>`` = 1[failure_type == type] for every non-success type with at least ``min_rollouts`` rollouts."""
    counts = ret["failure_type"].value_counts()
    return {f"ft_{t}": (ret["failure_type"] == t).to_numpy(float) for t in sorted(counts.index)
            if t != "success" and counts[t] >= min_rollouts}


def score_table(m: np.ndarray, episodes: pd.DataFrame, ret: pd.DataFrame, center: bool = True,
                return_encoding: str = "raw") -> pd.DataFrame:
    """The episode × metric table. ``episodes`` (rows of M's columns, in order) carries ``episode_id``,
    ``content_hash`` and the task key; ``ret`` (rows of M's rows, in order) is the returns table.

    Columns: ``infl_<ret col>`` over all rollouts and ``infl_<ret col>__same_task`` over rollouts of the episode's own
    task, for every family-W return and every per-failure-type return; ``harm_<...>`` = −GOOD_SIGN × infl (most
    harmful = largest). ``return_encoding="pm1"`` applies CUPID's ±1 uncentred return to the binary columns
    (``column_plan``); the others stay raw.
    """
    returns = {c: ret[c].to_numpy(float) for c in RETURN_COLUMNS}
    returns.update(failure_type_returns(ret))
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


def zscore_ensemble(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Average of per-table z-scores for every ``infl_``/``harm_`` column (one table per training seed of the same
    subset). z-scoring puts seeds on one scale before averaging; the rows must be the same episode versions."""
    keys = tables[0][["episode_id", "content_hash"]]
    for t in tables[1:]:
        if not t[["episode_id", "content_hash"]].equals(keys):
            raise ValueError("ensemble tables do not key the same episode versions in the same order")
    cols = [c for c in tables[0].columns if c.startswith(("infl_", "harm_"))]
    out = tables[0].drop(columns=cols).copy()
    for c in cols:
        z = [(t[c] - t[c].mean()) / t[c].std() for t in tables if c in t]
        out[c] = sum(z) / len(z)
    return out


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


def attribute(train_feats_prefix: Path, rollout_feats_prefix: Path, episodes: pd.DataFrame, rollouts: pd.DataFrame,
              ret: pd.DataFrame, damping: float = 1e-3, center: bool = True, rollout_mean: bool = False,
              device: str | None = None, return_encoding: str = "raw") -> tuple[pd.DataFrame, np.ndarray, dict]:
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
                  chain: str | None = None, allow_dirty: bool = False, return_encoding: str = "raw") -> dict:
    """Stage ``attribute`` (PoC §9.1 Step 3 p. 19): CUPID influence of every training episode version on every
    family-W return (and per-failure-type return) of one eval run, recorded as one MLflow run.

    Refuses (``LineageError``/``VersionIntegrityError``/``ValueError``, before any run exists) unless: the training
    subset opens through ``open_for_method`` (intact, not fenced, blind, not ground-truth-derived; PoC §2.1 item 4
    p. 3, owner decision 3); the eval run evaluated this train run's checkpoint and recorded trajectories; the returns run
    consumed that eval's rollouts; both feature sets were computed from this checkpoint with one projection; the train
    features cover exactly the training subset and the rollout features exactly the recorded trajectories.
    Returns ``{"run_id", "scores", "n_episodes", "n_rollouts", "agreement"}``.
    """
    from vlapoc.lineage import rel
    from vlapoc.method_data import open_for_method
    from vlapoc.returns import load_eval_rollouts
    from vlapoc.runs import _training_subset, check_scores, client, stage_run

    c = client(lineage_root)
    tr = c.get_run(train_run_id)
    subset = _training_subset(tr, lineage_root)
    open_for_method(subset.subset_id, lineage_root)  # refuses fenced / leaky / ground-truth-derived / tampered input
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
        if mans[0]["checkpoint_sha256"] != ckpt_sha:
            raise ValueError(f"features {prefix} were computed from another checkpoint")
    _, _, tman = load_features(train_features)
    if sorted(tman[0]["episodes"] or []) != subset.resolve(lineage_root):
        raise ValueError("train features were not computed on exactly the training subset's episodes")
    episodes = subset.episodes(lineage_root)
    rollouts, _ = load_eval_rollouts(eval_run_id, lineage_root, make_out_dir=False)
    ret = _artifact_table(rr, "returns", lineage_root)
    table, m, shared = attribute(train_features, rollout_features, episodes, rollouts, ret, damping, center,
                                 rollout_mean, device, return_encoding)
    check_scores(table, train_run_id, lineage_root)
    cols = [f"infl_{c.removeprefix('ret_')}__same_task" for c in RETURN_COLUMNS]
    agree = agreement(table, cols)
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"attribution_scores": out_dir / "attribution_scores.parquet", "agreement": out_dir / "agreement.json",
             "influence_matrix": out_dir / "rollout_episode_influence.npy"}
    table.to_parquet(paths["attribution_scores"], index=False)
    paths["agreement"].write_text(json.dumps(agree, indent=1))
    np.save(paths["influence_matrix"], m)
    with stage_run("attribute", name, lineage_root, chain=chain, allow_dirty=allow_dirty) as r:
        r.input_subset(subset, "training", None)
        r.input_entity("checkpoint", ckpt_sha, "checkpoint")
        r.input_entity("rollouts", ev.data.tags["output.rollouts.sha256"], "rollouts")
        r.input_entity("returns", rr.data.tags["output.returns.sha256"], "returns")
        r.params({"method": "CUPID influence on returns (TRAK action influence, Kronecker-projected gradients)",
                  "train_run_id": train_run_id, "eval_run_id": eval_run_id, "returns_run_id": returns_run_id,
                  "damping": damping, "center_returns": center, "return_encoding": return_encoding,
                  "rollout_mean": rollout_mean,
                  "train_output_fn": shared["output_fns"][0], "rollout_output_fn": shared["output_fns"][1],
                  "divergences_from_cupid": DIVERGENCES, "min_failure_type_rollouts_inferred": MIN_FAILURE_TYPE_ROLLOUTS,
                  "return_columns": RETURN_COLUMNS, "good_sign": GOOD_SIGN,
                  "features": {k: v for k, v in shared.items() if k not in ("train_features", "rollout_features")},
                  "train_features": rel(Path(train_features)), "rollout_features": rel(Path(rollout_features)),
                  "n_rows": len(table), "n_rollouts": int(m.shape[0])})
        for prefix, role in ((train_features, "train"), (rollout_features, "rollouts")):
            for mf in sorted(Path(prefix).parent.glob(f"{Path(prefix).name}*.manifest.json")):
                import mlflow

                mlflow.log_artifact(str(mf), artifact_path=f"features/{role}")
        r.output_file("attribution_scores", paths["attribution_scores"], kind="score_table")
        r.output_file("agreement", paths["agreement"], kind="agreement")
        r.tags({"attribute.n_rollouts": int(m.shape[0]), "attribute.n_episodes": len(table)})
        return {"run_id": r.run_id, "scores": rel(paths["attribution_scores"]), "n_episodes": len(table),
                "n_rollouts": int(m.shape[0]), "agreement": agree}


def ensemble_run(attribute_run_ids: list[str], name: str, out_dir: Path | None = None, lineage_root: Path | None = None,
                 chain: str | None = None, allow_dirty: bool = False) -> dict:
    """Stage ``attribute`` (ensemble): z-scored average of per-seed attribution tables of one training subset.

    Grounding: PoC §2.3 p. 4 "2–3 seeds per data configuration", so the selection for Step 4 does not rest on one
    seed's noise (averaging influence over independently trained models is TRAK's ensembling, inferred to transfer).
    Every input must come from a train run on the same subset. Returns ``{"run_id", "scores", "n_models"}``.
    """
    from vlapoc.lineage import Subset, rel
    from vlapoc.runs import client, stage_run

    c = client(lineage_root)
    runs = [c.get_run(i) for i in attribute_run_ids]
    subsets = {r.data.tags["input.training"] for r in runs}
    if len(subsets) != 1:
        raise ValueError(f"ensemble inputs were trained on different subsets: {sorted(subsets)}")
    tables = [_artifact_table(r, "attribution_scores", lineage_root) for r in runs]
    ens = zscore_ensemble(tables)
    out_dir = Path(out_dir) if out_dir is not None else DEFAULT_OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "attribution_scores.parquet"
    ens.to_parquet(path, index=False)
    subset = Subset.load(subsets.pop().split(":", 1)[1], lineage_root)
    with stage_run("attribute", name, lineage_root, chain=chain, allow_dirty=allow_dirty) as r:
        r.input_subset(subset, "training", None)
        for i, run in enumerate(runs):
            r.input_entity("score_table", run.data.tags["output.attribution_scores.sha256"], f"seed_table_{i}")
        r.params({"method": "ensemble: mean over seeds of per-seed z-scored influence columns",
                  "attribute_run_ids": attribute_run_ids, "n_models": len(runs), "n_rows": len(ens)})
        r.output_file("attribution_scores", path, kind="score_table")
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
    p.add_argument("--cupid-returns", action="store_true",
                   help="CUPID's R = ±1, uncentred, for the binary return columns (return_encoding='pm1')")
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    out = attribute_run(a.train_run, a.eval_run, a.returns_run, a.train_features, a.rollout_features, a.name,
                        damping=a.damping, center=not a.no_center, device=a.device,
                        return_encoding="pm1" if a.cupid_returns else "raw")
    print(json.dumps({k: v for k, v in out.items() if k != "agreement"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
