"""Per-subset normalization statistics: mean/std computed only from a training run's own episodes.

Grounding
---------
- Owner decision 3 (``tracker/features/2026-10-01-evaluation-layer/brief.md``): "Every training run normalizes
  with mean/std computed only from its own training episodes ... Wherever this feature trains or defines
  training, it applies." Statistics from data a run does not train on (corrupted episodes, clean originals,
  held-out references) would leak into it; per-subset statistics make normalization a function of the
  training data.
- PoC p. 1: "model, config, training schedule and evaluation protocol are held fixed so that data is the only
  variable" (a removal's effect on the statistics is part of its counterfactual effect). PoC §2.3 (p. 4):
  training runs of one experiment differ only in data and seed.

What is LeRobot's (reused, not re-implemented)
----------------------------------------------
- Every LeRobot v3 dataset stores per-episode statistics in ``meta/episodes/*.parquet`` as columns
  ``stats/<feature>/<stat>`` (min, max, mean, std, count, q01..q99), written at recording/conversion time.
- ``lerobot.datasets.compute_stats.aggregate_stats`` combines them (count-weighted mean, parallel-variance std,
  elementwise min/max, count-weighted quantiles). LeRobot's own ``meta/stats.json`` is exactly that aggregate
  over all episodes (AC8.1 measures ≤ 1e-9 here), and LeRobot's own episode-subset tools
  (``lerobot/datasets/dataset_tools.py``, the metadata copy used when deleting/splitting episodes) compute a
  subset's stats the same way, by aggregating the kept episodes' ``stats/`` columns.
- So a subset's stats are ``aggregate_stats`` over its episodes' stored stats: no frame reads, and no
  ``recompute_stats`` over a physical copy of the subset (plan P23).

The training hook is in ``vlapoc.lerobot_train_entry`` (it replaces ``dataset.meta.stats`` for every non-camera
feature before LeRobot builds the policy and its normalizer/unnormalizer); ``vlapoc.runs.train_run`` computes
the stats here, writes them next to the run, and records their digest (tag ``train.subset_stats.sha256``).
Camera features keep LeRobot's handling (ImageNet stats with ``use_imagenet_stats=true``; SmolVLA's VISUAL
normalization is IDENTITY).

    .venv/bin/python -m vlapoc.subset_stats --check v0          # AC8.1: aggregate over all episodes == stats.json
    .venv/bin/python -m vlapoc.subset_stats --subset v0/reference   # stats + digest of one subset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vlapoc.libero_convert import REPO_ROOT

NORMALIZED_FEATURES = ("observation.state", "action")  # the MEAN_STD-normalized non-camera features (ACT, SmolVLA)


def episode_stats(root: Path) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    """``{episode_index: {feature: {stat: array}}}`` for the non-camera features, from ``meta/episodes``."""
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    numeric = [k for k, f in info["features"].items() if f["dtype"] not in ("image", "video")]
    frames = [pd.read_parquet(p) for p in sorted((root / "meta" / "episodes").glob("*/*.parquet"))]
    df = pd.concat(frames, ignore_index=True)
    out = {}
    for r in df.to_dict("records"):
        st: dict = {}
        for k, v in r.items():
            if k.startswith("stats/"):
                feat, stat = k[len("stats/"):].rsplit("/", 1)
                if feat in numeric:
                    st.setdefault(feat, {})[stat] = np.asarray(v, dtype=np.float64)
        out[int(r["episode_index"])] = st
    return out


def subset_stats(root: Path, episodes: list[int]) -> dict[str, dict[str, np.ndarray]]:
    """LeRobot ``aggregate_stats`` over the stored per-episode stats of ``episodes`` (LeRobot episode indices)."""
    from lerobot.datasets.compute_stats import aggregate_stats

    per = episode_stats(root)
    missing = sorted(set(episodes) - set(per))
    if missing:
        raise KeyError(f"{len(missing)} episodes have no stored stats (e.g. {missing[:3]})")
    return aggregate_stats([per[e] for e in sorted(set(episodes))])


def to_json(stats: dict) -> dict:
    """Plain-JSON form (lists of floats), sorted keys: what the hook loads and what the digest hashes."""
    return {f: {s: np.asarray(v, dtype=np.float64).tolist() for s, v in sorted(d.items())} for f, d in sorted(stats.items())}


def stats_digest(stats: dict, features: tuple[str, ...] = NORMALIZED_FEATURES) -> str:
    """sha256 of the canonical JSON of the normalized features' stats (all stat fields, float64 repr)."""
    j = to_json({f: stats[f] for f in features})
    return hashlib.sha256(json.dumps(j, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check(version: str = "v0") -> dict:
    """AC8.1: ``aggregate_stats`` over all episodes vs the version's ``meta/stats.json`` (max |diff| per stat)."""
    from vlapoc.lineage import resolve_version

    v = resolve_version(version)
    root = REPO_ROOT / v["lerobot_root"]
    per = episode_stats(root)
    agg = subset_stats(root, sorted(per))
    ref = json.loads((root / "meta" / "stats.json").read_text())
    diffs = {f: {s: float(np.max(np.abs(np.asarray(agg[f][s], dtype=np.float64) - np.asarray(ref[f][s], dtype=np.float64))))
                 for s in ("mean", "std", "min", "max", "count")} for f in NORMALIZED_FEATURES}
    return {"version": version, "dataset_version_id": v["dataset_version_id"], "n_episodes": len(per), "max_abs_diff": diffs}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m vlapoc.subset_stats")
    p.add_argument("--check", metavar="VERSION")
    p.add_argument("--subset", metavar="SUBSET")
    a = p.parse_args(argv)
    if a.check:
        r = check(a.check)
        print(f"AC8.1 aggregate_stats over all {r['n_episodes']} episodes of {a.check} vs meta/stats.json, max |diff|: "
              + "; ".join(f"{f}: " + ", ".join(f"{s} {d:.2e}" for s, d in v.items()) for f, v in r["max_abs_diff"].items()))
        print(json.dumps(r))
    if a.subset:
        from vlapoc.lineage import Subset, resolve_version

        s = Subset.load(a.subset)
        root = REPO_ROOT / resolve_version(s.dataset_version_id)["lerobot_root"]
        st = subset_stats(root, s.resolve())
        print(json.dumps({"subset_id": s.subset_id, "n_episodes": len(s.resolve()), "digest": stats_digest(st),
                          **{f: {k: to_json(st)[f][k] for k in ("mean", "std")} for f in NORMALIZED_FEATURES}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
