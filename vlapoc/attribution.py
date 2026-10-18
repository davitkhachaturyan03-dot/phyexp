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
- PoC §2.1 item 4 (p. 3) and the evaluation layer's S0: a method under test only sees blind versions;
  ``attribute_run`` calls ``vlapoc.lineage.assert_blind`` on the training version first.

The estimator, and how this implementation evaluates it
-------------------------------------------------------
Ψ_a-inf is bilinear in the two gradients, so the rollout × episode matrix is exact from per-group feature sums:
M[τ, ξ] = (Σ_{j∈τ} φ_j)ᵀ H⁻¹ (Σ_{i∈ξ} φ_i), one Cholesky solve for all pairs. Then for a return column R,
Ψ_R(ξ) = (1/m) Σ_τ (R(τ) − R̄) M[τ, ξ].

Two choices, both recorded in every run's params:
- **Centred returns (divergence from Prop. 1's raw R; inferred).** Subtracting R̄ removes the part of every episode's
  score that is the same for all rollouts (the influence on the average rollout), so a metric's scores rank
  episodes by how they move that metric, not by how much they move the policy at all. For a binary return it is
  CUPID's success-minus-failure contrast scaled by p(1−p). ``center=False`` gives Prop. 1 literally.
- **1/H.** CUPID divides by the horizon H; a rollout's sum over its own steps is kept (``rollout_mean=False``), since
  family-W returns are already whole-rollout quantities and long timeouts are exactly the rollouts that matter.

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
MIN_FAILURE_TYPE_ROLLOUTS = 5  # a failure type needs this many rollouts to get its own return column


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
    return feats, meta, manifests


def check_compatible(*manifest_lists: list[dict]) -> dict:
    """Raise unless every manifest shares checkpoint, projection seed, rank, K and layer list; return the shared part."""
    keys = ("checkpoint_sha256", "seed", "rank", "k_draws", "dim", "layers")
    first = manifest_lists[0][0]
    for ms in manifest_lists:
        for m in ms:
            bad = [k for k in keys if m[k] != first[k]]
            if bad:
                raise ValueError(f"feature manifests disagree on {bad} ({m['dataset_root']} vs {first['dataset_root']})")
    return {k: first[k] for k in keys if k != "layers"} | {"n_layers": len(first["layers"])}


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


def performance_influence(m: np.ndarray, r: np.ndarray, mask: np.ndarray | None = None, center: bool = True) -> np.ndarray:
    """Ψ_R(ξ) = (1/m) Σ_τ (R(τ) − R̄) M[τ, ξ] over the rollouts in ``mask`` (all by default); one value per column of M."""
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


def score_table(m: np.ndarray, episodes: pd.DataFrame, ret: pd.DataFrame, center: bool = True) -> pd.DataFrame:
    """The episode × metric table. ``episodes`` (rows of M's columns, in order) carries ``episode_id``,
    ``content_hash`` and the task key; ``ret`` (rows of M's rows, in order) is the returns table.

    Columns: ``infl_<ret col>`` over all rollouts and ``infl_<ret col>__same_task`` over rollouts of the episode's own
    task, for every family-W return and every per-failure-type return; ``harm_<...>`` = −GOOD_SIGN × infl (most
    harmful = largest).
    """
    returns = {c: ret[c].to_numpy(float) for c in RETURN_COLUMNS}
    returns.update(failure_type_returns(ret))
    out = episodes[["episode_id", "content_hash", "episode_index", *TASK_KEY, "task_name"]].reset_index(drop=True).copy()
    task_of_rollout = list(zip(ret["suite"], ret["libero_task_id"].astype(int)))
    task_of_episode = list(zip(out["suite"], out["libero_task_id"].astype(int)))
    for name, r in returns.items():
        col = f"infl_{name.removeprefix('ret_')}"
        out[col] = performance_influence(m, r, center=center)
        same = np.full(len(out), np.nan)
        for task in sorted(set(task_of_episode)):
            cols = np.array([t == task for t in task_of_episode])
            rows = np.array([t == task for t in task_of_rollout])
            same[cols] = performance_influence(m[:, cols], r, rows, center=center)
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
              device: str | None = None) -> tuple[pd.DataFrame, np.ndarray, dict]:
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
    table = score_table(m, episodes, ret, center)
    shared |= {"n_train_samples": int(len(tmeta["episode_index"])), "n_rollout_samples": int(len(rmeta["episode_index"])),
               "train_stride": tman[0]["stride"], "rollout_stride": rman[0]["stride"],
               "train_features": [m_["dataset_root"] for m_ in tman][:1], "rollout_features": [m_["dataset_root"] for m_ in rman][:1]}
    return table, m, shared


def attribute_run(train_run_id: str, eval_run_id: str, returns_run_id: str, train_features: Path, rollout_features: Path,
                  name: str, out_dir: Path | None = None, damping: float = 1e-3, center: bool = True,
                  rollout_mean: bool = False, device: str | None = None, lineage_root: Path | None = None,
                  chain: str | None = None, allow_dirty: bool = False) -> dict:
    """Stage ``attribute`` (PoC §9.1 Step 3 p. 19): CUPID influence of every training episode version on every
    family-W return (and per-failure-type return) of one eval run, recorded as one MLflow run.

    Refuses (``LineageError``/``ValueError``, before any run exists) unless: the training version is blind
    (``assert_blind``); the eval run evaluated this train run's checkpoint and recorded trajectories; the returns run
    consumed that eval's rollouts; both feature sets were computed from this checkpoint with one projection; the train
    features cover exactly the training subset and the rollout features exactly the recorded trajectories.
    Returns ``{"run_id", "scores", "n_episodes", "n_rollouts", "agreement"}``.
    """
    from vlapoc.lineage import assert_blind, rel
    from vlapoc.returns import load_eval_rollouts
    from vlapoc.runs import _training_subset, check_scores, client, stage_run

    c = client(lineage_root)
    tr, ev, rr = c.get_run(train_run_id), c.get_run(eval_run_id), c.get_run(returns_run_id)
    subset = _training_subset(tr, lineage_root)
    assert_blind(subset.dataset_version_id, lineage_root)
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
                                 rollout_mean, device)
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
                  "damping": damping, "center_returns": center, "rollout_mean": rollout_mean,
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
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    out = attribute_run(a.train_run, a.eval_run, a.returns_run, a.train_features, a.rollout_features, a.name,
                        damping=a.damping, center=not a.no_center, device=a.device)
    print(json.dumps({k: v for k, v in out.items() if k != "agreement"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
