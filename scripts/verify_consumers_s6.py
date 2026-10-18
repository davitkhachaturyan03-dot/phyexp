"""Verify ground-truth S6: the injected version (v3 by default) is trainable and evaluable locally (AC6.1-AC6.3); the runbook is checked by reading it.

Grounding: evaluation-layer owner decision 3 (per-subset normalization) and S8; PoC §8.10 pp. 17-18 ("Verify before
claiming": policy-level effects need training and rollouts, which are pending server), §2.6 p.5 and §9.1 Step 4 p.19;
owner decision 3 of this feature (a label-derived subset must never land in ``data/lineage``).

    .venv/bin/python scripts/verify_consumers_s6.py            # ~3 min; SmolVLA 2-step smoke on MPS

All scratch (lineage copy, HF datasets cache, run directory) is under ``data/scratch/gt_s6/`` and deleted at the end.
The real ``data/lineage`` is only read (its subset count and ``mlflow.db`` sha256 are printed before and after).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from vlapoc.catalog import read_tasks  # noqa: E402
from vlapoc.groundtruth import load_answer_key  # noqa: E402
from vlapoc.lineage import Subset, resolve_version, version_episodes  # noqa: E402
from vlapoc.subset_stats import NORMALIZED_FEATURES, check, stats_digest, subset_stats  # noqa: E402

FEATURE_DIR = REPO_ROOT / "tracker" / "features" / "2026-10-15-ground-truth"
HUMAN_REF = REPO_ROOT / "data" / "eval" / "human_reference"
SMOLVLA_BASE = REPO_ROOT / "data" / "hf_models" / "smolvla_base"


def tree_digest(root: Path) -> tuple[str, int]:
    """sha256 over (relative path, file sha256) of every file under ``root`` and the file count."""
    h = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    for p in files:
        h.update(str(p.relative_to(root)).encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest(), len(files)


def sha(p: Path) -> str:
    """sha256 of a file."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v3")
    ap.add_argument("--prefix", default="s6_v3")
    a = ap.parse_args()
    V = a.version
    out: dict = {}
    ok = True
    t0 = time.time()
    db = REPO_ROOT / "data" / "lineage" / "mlflow.db"
    db_before, n_sub_before = sha(db), len(list((REPO_ROOT / "data" / "lineage" / "subsets").glob("*.json")))
    hr_before, hr_n = tree_digest(HUMAN_REF)
    newest = max(p.stat().st_mtime for p in HUMAN_REF.rglob("*") if p.is_file())
    print(f"before: mlflow.db sha256 {db_before[:16]}; data/lineage/subsets {n_sub_before} files; human_reference digest {hr_before[:16]} "
          f"({hr_n} files, newest mtime {time.strftime('%Y-%m-%d %H:%M', time.localtime(newest))})")

    # ---- AC6.1 ----------------------------------------------------------------------------------------------
    r = check(V)
    cli = subprocess.run([sys.executable, "-m", "vlapoc.subset_stats", "--check", V], cwd=REPO_ROOT, capture_output=True, text=True)
    print("AC6.1 CLI:", [l for l in cli.stdout.splitlines() if l.startswith("AC8.1")][0][:330])
    worst = max(d for f in r["max_abs_diff"].values() for d in f.values())
    cand0, cand2 = Subset.load("v0/candidate"), Subset.load(f"{V}/candidate")
    s0 = subset_stats(REPO_ROOT / resolve_version("v0")["lerobot_root"], cand0.resolve())
    s2 = subset_stats(REPO_ROOT / resolve_version(V)["lerobot_root"], cand2.resolve())
    d0, d2 = stats_digest(s0), stats_digest(s2)
    print(f"AC6.1 aggregate over all {V} episodes vs {V} stats.json: worst max|diff| {worst:.2e} (tolerance 1e-6); "
          f"v0/candidate digest {d0[:16]} vs {V}/candidate digest {d2[:16]}: differ {d0 != d2}")
    rows = []
    names = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
    for i, n in enumerate(names):
        rows.append({"action_dim": f"{i} {n}", "mean_v0cand": s0["action"]["mean"][i], "mean_target_cand": s2["action"]["mean"][i],
                     "d_mean": s2["action"]["mean"][i] - s0["action"]["mean"][i], "std_v0cand": s0["action"]["std"][i],
                     "std_target_cand": s2["action"]["std"][i], "std_ratio": s2["action"]["std"][i] / s0["action"]["std"][i]})
    tbl = pd.DataFrame(rows)
    print(tbl.round(5).to_string(index=False))
    tbl.to_csv(FEATURE_DIR / "artifacts" / f"{a.prefix}_action_stats_vs_v0.csv", index=False)
    ok1 = worst <= 1e-6 and d0 != d2
    out["AC6.1"] = {"worst_abs_diff": worst, "digest_v0_candidate": d0, "digest_target_candidate": d2, "pass": ok1}
    ok &= ok1

    # ---- AC6.3 (tasks) -------------------------------------------------------------------------------------------
    t0_, t2_ = read_tasks(REPO_ROOT / resolve_version("v0")["lerobot_root"]), read_tasks(REPO_ROOT / resolve_version(V)["lerobot_root"])
    tasks_equal = t0_ == t2_ and len(t2_) == 20
    print(f"AC6.3 {V}'s 20 task strings (task_index -> string) equal v0's: {tasks_equal}")

    # ---- AC6.2 ----------------------------------------------------------------------------------------------------
    base = REPO_ROOT / "data" / "scratch" / "gt_s6"
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)
    try:
        from vlapoc.runs import scratch_lineage_copy, train_run

        lr = scratch_lineage_copy(base / "lineage")
        ver = resolve_version(V, lr)
        lab = load_answer_key(V, lr)
        ve = version_episodes(V, lr)
        one_each = [g.sort_values("episode_id")["episode_id"].iloc[0] for _, g in lab.groupby("corruption_type")]
        clean = [e for e in cand2.episode_ids if e not in set(lab["episode_id"])]
        clean = list(ve[ve["episode_id"].isin(clean)].drop_duplicates("task_name").sort_values("episode_index")["episode_id"][:7])
        ids = one_each + clean
        sub = Subset(ver["dataset_version_id"], tuple(ids), {"op": "gt_smoke_subset", "groundtruth": True,
                                                            "note": "one labelled episode of each type + 7 clean candidates, built label-side"})
        sub.save(lr)
        types_in = sorted(lab.set_index("episode_id").loc[[i for i in ids if i in set(lab["episode_id"])], "corruption_type"])
        print(f"AC6.2 scratch subset {sub.subset_id[:12]}: {len(ids)} episodes, labelled types {types_in}; saved in {lr.relative_to(REPO_ROOT)}")
        os.environ["HF_DATASETS_CACHE"] = str(base / "hf_datasets_cache")
        os.environ["HF_HOME"] = str(REPO_ROOT / "data" / "hf_home")
        os.environ["HF_HUB_OFFLINE"] = "1"
        res = train_run(sub.subset_id, f"train-smolvla-2step-{V}gt-smoke", seed=0, steps=2, batch_size=2, policy_type="smolvla",
                        policy_overrides={"load_vlm_weights": False}, device="mps", pretrained_path=SMOLVLA_BASE,
                        pretrained_ref="lerobot/smolvla_base@d9f33c94a60fb382c90dea2164c96845bd955e28", smoke=True,
                        runs_root=base / "runs", lineage_root=lr)
        from vlapoc.runs import _ckpt_files, client

        # the same normalizer check as evaluation S8 AC8.2, against the scratch store
        from safetensors.numpy import load_file

        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{lr / 'mlflow.db'}")
        run = mlflow.tracking.MlflowClient().get_run(res["run_id"])
        root = REPO_ROOT / ver["lerobot_root"]
        expect = subset_stats(root, sub.resolve(lr))
        full = json.loads((root / "meta" / "stats.json").read_text())
        files = _ckpt_files(REPO_ROOT / run.data.tags["checkpoint.path"])
        rows_n = []
        for which in ("normalizer_stats", "unnormalizer_stats"):
            saved = load_file(str(files[which]))
            for f in NORMALIZED_FEATURES:
                for s in ("mean", "std"):
                    k = f"{f}.{s}"
                    if k in saved:
                        v = saved[k].astype(np.float64).reshape(-1)
                        rows_n.append({"file": files[which].name[-30:], "stat": k,
                                       "max_abs_vs_subset": float(np.max(np.abs(v - np.asarray(expect[f][s]).reshape(-1)))),
                                       "max_abs_vs_full_stats_json": float(np.max(np.abs(v - np.asarray(full[f][s]).reshape(-1))))})
        tag = run.data.tags.get("train.subset_stats.sha256")
        print(f"AC6.2 train run {res['run_id'][:12]} finished in {res['seconds']}s: checkpoint {res['checkpoint']}, episodes_match {res['episodes_match']}, "
              f"n_episodes {res['n_episodes']}; tag train.subset_stats.sha256 {tag[:16]} == recomputed {stats_digest(expect)[:16]}: {tag == stats_digest(expect)}")
        print(pd.DataFrame(rows_n).to_string(index=False))
        ok2 = res["episodes_match"] and tag == stats_digest(expect) and all(r_["max_abs_vs_subset"] <= 1e-6 and r_["max_abs_vs_full_stats_json"] > 1e-6 for r_ in rows_n)
        out["AC6.2"] = {"run_id": res["run_id"], "seconds": res["seconds"], "subset_id": sub.subset_id, "types": types_in,
                        "tag_subset_stats_sha256": tag, "normalizer_rows": rows_n, "pass": bool(ok2),
                        "weights_sha256": res["weights_sha256"]}
        ok &= bool(ok2)
    finally:
        shutil.rmtree(base, ignore_errors=True)
        print(f"scratch removed: {not base.exists()}")

    # ---- AC6.3 / 'nothing leaked to the real store' ------------------------------------------------------------------
    hr_after, _ = tree_digest(HUMAN_REF)
    db_after, n_sub_after = sha(db), len(list((REPO_ROOT / "data" / "lineage" / "subsets").glob("*.json")))
    print(f"AC6.3 human_reference digest before {hr_before[:16]} after {hr_after[:16]}: unchanged {hr_before == hr_after}")
    print(f"real store: mlflow.db sha256 {db_before[:16]} -> {db_after[:16]} (unchanged {db_before == db_after}); "
          f"data/lineage/subsets {n_sub_before} -> {n_sub_after} files")
    ok3 = tasks_equal and hr_before == hr_after and int(newest) < time.mktime((2026, 10, 15, 0, 0, 0, 0, 0, -1))
    ok &= bool(ok3) and db_before == db_after and n_sub_before == n_sub_after
    out["AC6.3"] = {"tasks_equal": tasks_equal, "human_reference_digest_before": hr_before, "after": hr_after,
                    "human_reference_newest_mtime_before_feature_start": int(newest) < time.mktime((2026, 10, 15, 0, 0, 0, 0, 0, -1)),
                    "pass": bool(ok3)}
    out["real_store"] = {"mlflow_db_unchanged": db_before == db_after, "subsets_before": n_sub_before, "subsets_after": n_sub_after}
    out["pass"] = bool(ok)
    (FEATURE_DIR / "artifacts" / f"{a.prefix}_verify.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nall S6 checks pass: {ok} ({time.time() - t0:.0f}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
