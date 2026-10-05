"""Verify evaluation-layer S8: per-subset normalization statistics (owner decision 3) end to end.

- AC8.1  ``--check v0``: LeRobot ``aggregate_stats`` over all episodes' stored stats == ``meta/stats.json``.
- AC8.2  ``--train-smolvla v0/reference``: a 2-step SmolVLA fine-tune from ``lerobot/smolvla_base`` on MPS with
         per-subset stats, then ``--normalizer <train_run_id>``: the checkpoint's normalizer and unnormalizer
         state/action mean/std equal the subset stats and differ from the full ``stats.json``.
- AC8.3  ``--drop-subset v0/reference 5``, ``--train-act <subset>`` (twice), ``--diff <run_a> <run_b>``: two subsets
         differing by k episodes give different stats digests, and ``vlapoc.runs.diff`` reports it under ``data``.

Grounding: owner decision 3 (brief); PoC p. 1 (data is the only variable), §2.3 p. 4 (runs differ only in data and
seed). SmolVLA runs use a repo-local HF cache (``HF_HOME=data/hf_home``) holding only the VLM's config/tokenizer
(HuggingFaceTB/SmolVLM2-500M-Video-Instruct @ 7b375e1b, the repo ``SmolVLM2-500M-Instruct`` redirects to), with
``HF_HUB_OFFLINE=1``: the VLM weights come from ``smolvla_base``'s own ``model.safetensors`` (``load_vlm_weights=false``).

    .venv/bin/python scripts/verify_subset_stats_s8.py --check v0
    .venv/bin/python scripts/verify_subset_stats_s8.py --train-smolvla v0/reference
    .venv/bin/python scripts/verify_subset_stats_s8.py --normalizer <train_run_id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("HF_DATASETS_CACHE", str(REPO_ROOT / "data" / "hf_datasets_cache"))

ART = REPO_ROOT / "tracker" / "features" / "2026-10-01-evaluation-layer" / "artifacts"
SMOLVLA_BASE = REPO_ROOT / "data" / "hf_models" / "smolvla_base"  # lerobot/smolvla_base @ d9f33c94


def check(version: str) -> None:
    from vlapoc.subset_stats import check as _check

    r = _check(version)
    print(f"AC8.1 aggregate_stats over all {r['n_episodes']} episodes of {version} vs meta/stats.json, max |diff|: "
          + "; ".join(f"{f}: " + ", ".join(f"{s} {d:.2e}" for s, d in v.items()) for f, v in r["max_abs_diff"].items()))
    json.dump(r, open(ART / "s8_ac8_1_check.json", "w"), indent=1)


def train(subset: str, policy: str, name: str, steps: int, batch_size: int) -> None:
    from vlapoc.runs import train_run

    if policy == "smolvla":
        os.environ["HF_HOME"] = str(REPO_ROOT / "data" / "hf_home")
        os.environ["HF_HUB_OFFLINE"] = "1"
        out = train_run(subset, name, seed=0, steps=steps, batch_size=batch_size, policy_type="smolvla",
                        policy_overrides={"load_vlm_weights": False}, device="mps", pretrained_path=SMOLVLA_BASE)
    else:
        out = train_run(subset, name, seed=0, steps=steps, batch_size=batch_size, policy_type="act", device="mps")
    print(json.dumps(out, indent=1))


def normalizer(run_id: str) -> None:
    """AC8.2: compare the checkpoint's saved normalizer/unnormalizer stats with the subset stats and stats.json."""
    from safetensors.numpy import load_file

    from vlapoc.lineage import resolve_version
    from vlapoc.runs import _ckpt_files, _training_subset, client
    from vlapoc.subset_stats import NORMALIZED_FEATURES, stats_digest, subset_stats

    run = client().get_run(run_id)
    sub = _training_subset(run, None)
    root = REPO_ROOT / resolve_version(sub.dataset_version_id)["lerobot_root"]
    expect = subset_stats(root, sub.resolve())
    full = json.loads((root / "meta" / "stats.json").read_text())
    files = _ckpt_files(REPO_ROOT / run.data.tags["checkpoint.path"])
    out = {"run_id": run_id, "policy_type": run.data.params.get("policy_type"), "n_episodes": len(sub.resolve()),
           "digest_recomputed": stats_digest(expect), "digest_tag": run.data.tags.get("train.subset_stats.sha256"), "files": {}}
    for which in ("normalizer_stats", "unnormalizer_stats"):
        saved = load_file(str(files[which]))
        rows = {}
        for f in NORMALIZED_FEATURES:
            for s in ("mean", "std"):
                key = f"{f}.{s}"
                if key not in saved:
                    continue  # the unnormalizer holds only output features (action)
                v = saved[key].astype(np.float64).reshape(-1)
                rows[key] = {"max_abs_vs_subset": float(np.max(np.abs(v - np.asarray(expect[f][s]).reshape(-1)))),
                             "max_abs_vs_full_stats_json": float(np.max(np.abs(v - np.asarray(full[f][s]).reshape(-1)))),
                             "dtype": str(saved[key].dtype)}
        out["files"][files[which].name] = rows
    ok = all(r["max_abs_vs_subset"] <= 1e-6 and r["max_abs_vs_full_stats_json"] > 1e-6 for d in out["files"].values() for r in d.values())
    out["pass"] = ok and out["digest_recomputed"] == out["digest_tag"]
    for fname, rows in out["files"].items():
        for k, r in rows.items():
            print(f"AC8.2 {fname}: {k} ({r['dtype']}) max |saved - subset stats| = {r['max_abs_vs_subset']:.2e}; "
                  f"max |saved - full stats.json| = {r['max_abs_vs_full_stats_json']:.2e}")
    print(f"AC8.2 run {run_id[:12]} ({out['policy_type']}, {out['n_episodes']} episodes): train.subset_stats.sha256 tag == recomputed "
          f"digest: {out['digest_recomputed'] == out['digest_tag']} ({out['digest_tag'][:16]}…); pass: {out['pass']}")
    json.dump(out, open(ART / f"s8_ac8_2_normalizer_{out['policy_type']}_{run_id[:8]}.json", "w"), indent=1)


def drop_subset(parent: str, k: int) -> None:
    from vlapoc.lineage import Subset, drop_random_k

    child = drop_random_k(Subset.load(parent), k, seed=0)
    print(json.dumps({"parent": parent, "k": k, "subset_id": child.subset_id, "n_episodes": len(child.resolve())}))


def diff(a: str, b: str) -> None:
    from vlapoc.runs import diff as _diff

    d = _diff(a, b)
    print(f"AC8.3 diff {a[:8]} vs {b[:8]}: data.n_removed {d['data']['n_removed']}, n_added {d['data']['n_added']}; "
          f"data.normalization {json.dumps(d['data']['normalization'])}")
    print(f"AC8.3 params that differ: {sorted(d['params'])}; config keys that differ: {sorted(d['config'])}; git same: {d['git']['same']}")
    json.dump(d, open(ART / "s8_ac8_3_diff.json", "w"), indent=1, default=str)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check")
    p.add_argument("--train-smolvla")
    p.add_argument("--train-act")
    p.add_argument("--name")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--normalizer")
    p.add_argument("--drop-subset", nargs=2, metavar=("PARENT", "K"))
    p.add_argument("--diff", nargs=2)
    a = p.parse_args()
    if a.check:
        check(a.check)
    if a.train_smolvla:
        train(a.train_smolvla, "smolvla", a.name or "train-smolvla-2step-persubset", a.steps, a.batch_size)
    if a.train_act:
        train(a.train_act, "act", a.name or "train-act-2step-persubset", a.steps, a.batch_size)
    if a.normalizer:
        normalizer(a.normalizer)
    if a.drop_subset:
        drop_subset(a.drop_subset[0], int(a.drop_subset[1]))
    if a.diff:
        diff(*a.diff)
    return 0


if __name__ == "__main__":
    sys.exit(main())
