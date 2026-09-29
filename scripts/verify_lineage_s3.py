"""Verify S3 (dataset versions, splits, subsets) against AC3.1-AC3.6 and AC3.8.

Grounding: PoC §2.1 p.3 (exact dataset versions, episode IDs), §2.6 p.5 and §9.1 Step 4 p.19 (drop
top-k vs drop random-k), §2.1 item 5 p.3 and §10 p.20 (group leave-one-out), §8.3 p.14 / §8.7 p.15 and
RoboDrop App. A.1 p.13 (held-out clean references).

Everything this script creates lives in a fresh scratch lineage root under the gitignored
``data/scratch/`` (a copy of the version/subset/split manifests; no MLflow DB) and is deleted at the end,
so the real lineage store gains nothing. LeRobot loads run inside ``isolated_hf_datasets_cache()``, so the
shared HF cache gains nothing either. No absolute or session-bound path is hard-coded (AC3.9).

The drop-top-k demonstration uses a real catalog column (``smoothness``) as the score; it is a
mechanism check, not an attribution result, and no outcome numbers are invented.

    .venv/bin/python scripts/verify_lineage_s3.py [--summary <path>]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vlapoc.libero_convert import DEFAULT_REPO_ID  # noqa: E402
from vlapoc.lineage import (  # noqa: E402
    LINEAGE_ROOT,
    LineageError,
    Subset,
    VersionIntegrityError,
    drop_group,
    drop_random_k,
    drop_top_k,
    isolated_hf_datasets_cache,
    recompute_version_id,
    resolve_version,
    verify_version,
    version_episodes,
)

FEATURE_DIR = REPO_ROOT / "tracker" / "features" / "2026-09-28-data-lineage-layer"
SCRATCH_ROOT = REPO_ROOT / "data" / "scratch"  # gitignored (data/)


def scratch_lineage_root(base: Path) -> Path:
    """Copy of the real lineage manifests (versions, subsets, splits, names) without the MLflow DB."""
    lr = base / "lineage"
    lr.mkdir(parents=True)
    for d in ("versions", "subsets", "splits"):
        shutil.copytree(LINEAGE_ROOT / d, lr / d)
    shutil.copy(LINEAGE_ROOT / "names.json", lr / "names.json")
    return lr


def served_episode_ids(root: Path, subset: Subset, eps: pd.DataFrame, lr: Path, full_getitem: bool = False):
    """Episode IDs and frame count that LeRobot actually serves for ``episodes=subset.resolve()``."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(DEFAULT_REPO_ID, root=root, episodes=subset.resolve(lr))
    idx_to_id = dict(zip(eps["episode_index"], eps["episode_id"]))
    col = ds.hf_dataset.data.column("episode_index").to_numpy()
    served = {idx_to_id[int(i)] for i in set(col.tolist())}
    n = len(ds)
    if full_getitem:  # every frame through __getitem__, the path a training DataLoader uses
        served_gi = {idx_to_id[int(ds[i]["episode_index"])] for i in range(n)}
        assert served_gi == served, "getitem and column disagree"
    return served, n, len(col)


def absolute_path_markers() -> list[str]:
    """Strings that betray a machine-bound absolute path in a manifest (built at run time, AC3.9)."""
    return sorted({str(REPO_ROOT), str(Path.home()), str(Path(tempfile.gettempdir()).resolve()), tempfile.gettempdir()})


ABS_JSON_VALUE = re.compile(r'"/[^"\s]')  # a JSON string value that starts with "/"
ABS_CSV_FIELD = re.compile(r'(^|[,\n])"?/[^,\n]')  # a CSV field that starts with "/"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=FEATURE_DIR / "artifacts" / "s3_summary.json")
    a = ap.parse_args()
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    base = Path(tempfile.mkdtemp(prefix="verify_s3_", dir=SCRATCH_ROOT))
    lr = scratch_lineage_root(base)
    print(f"scratch lineage root: {lr.relative_to(REPO_ROOT)} (deleted at exit)")
    out: dict = {}
    try:
        _run(out, lr, base)
    finally:
        shutil.rmtree(base, ignore_errors=True)
        print(f"scratch removed: {not base.exists()}")
    a.summary.write_text(json.dumps(out, indent=2, default=str))
    print(f"summary -> {a.summary.relative_to(REPO_ROOT)}")
    return 0


def _run(out: dict, lr: Path, base: Path) -> None:
    v = resolve_version("v0", lr)
    vid = v["dataset_version_id"]
    root = REPO_ROOT / v["lerobot_root"]
    eps = version_episodes("v0", lr)

    # ---- AC3.1 recompute the version id from the stored bytes ----------------------------
    t0 = time.time()
    recomputed = recompute_version_id("v0", lineage_root=lr)
    print(f"AC3.1 registered dataset_version_id = {vid}")
    print(f"AC3.1 recomputed from dataset      = {recomputed}  match={recomputed == vid}  ({time.time() - t0:.0f}s)")
    vdir = lr / "versions" / vid
    files = json.loads((vdir / "files.json").read_text())
    print(f"AC3.1 version dir contents: {sorted(p.name for p in vdir.iterdir())}; files.json entries={len(files)} "
          f"(data parquet={sum(f['path'].startswith('data/') for f in files)}); lerobot_root={v['lerobot_root']}")
    out["AC3.1"] = {"registered": vid, "recomputed": recomputed, "match": recomputed == vid, "files": len(files)}

    # ---- AC3.2 splits -----------------------------------------------------------------------
    ref = Subset.load("v0/reference", lr)
    cand = Subset.load("v0/candidate", lr)
    inter = set(ref.episode_ids) & set(cand.episode_ids)
    union = set(ref.episode_ids) | set(cand.episode_ids)
    per_task = eps[eps["episode_id"].isin(ref.episode_ids)].groupby(["suite", "task_name"]).size()
    print(f"AC3.2 reference={len(ref.episode_ids)} candidate={len(cand.episode_ids)} overlap={len(inter)} "
          f"union==version: {union == set(eps['episode_id'])}; reference per task: {sorted(set(per_task.tolist()))} "
          f"over {len(per_task)} tasks; derivation seed={ref.derivation['seed']}")
    out["AC3.2"] = {"reference": len(ref.episode_ids), "candidate": len(cand.episode_ids), "overlap": len(inter),
                    "per_task": sorted(set(per_task.tolist())), "tasks": int(len(per_task))}

    # ---- AC3.3 / AC3.4 through LeRobot, inside an isolated HF cache ------------------------
    scores = cand.episodes(lr)[["episode_id", "content_hash", "smoothness"]].rename(columns={"smoothness": "score"})
    with isolated_hf_datasets_cache(base) as cache:
        res33 = {}
        for name, s, full in [("v0/reference", ref, True), ("v0/candidate", cand, False)]:
            served, n, ncol = served_episode_ids(root, s, eps, lr, full_getitem=full)
            want_frames = int(s.episodes(lr)["length_steps"].sum())
            ok = served == set(s.episode_ids) and n == want_frames == ncol
            print(f"AC3.3 {name}: subset_id={s.subset_id[:12]} resolve()={len(s.resolve(lr))} indices; served episodes "
                  f"== subset: {served == set(s.episode_ids)}; frames served={n} sum(length_steps)={want_frames}"
                  f"{' (every frame via __getitem__)' if full else ''} -> {'OK' if ok else 'FAIL'}")
            res33[name] = {"served_eq_subset": served == set(s.episode_ids), "frames": n, "expected": want_frames, "ok": ok}
        out["AC3.3"] = res33

        top = drop_top_k(cand, scores, metric="score", k=20, direction="highest",
                         score_source="catalog column 'smoothness' (mechanism demo, not an attribution score)", lineage_root=lr)
        expect = set(scores.sort_values(["score", "episode_id"], ascending=[False, True]).head(20)["episode_id"])
        low = drop_top_k(cand, scores, metric="score", k=20, direction="lowest", score_source="mechanism demo", lineage_root=lr)
        expect_low = set(scores.sort_values(["score", "episode_id"], ascending=[True, True]).head(20)["episode_id"])
        print(f"AC3.4 drop_top_k(highest): child {top.subset_id[:12]} n={len(top.episode_ids)}; dropped == 20 highest: "
              f"{set(top.derivation['dropped']) == expect}; drop_top_k(lowest): dropped == 20 lowest: "
              f"{set(low.derivation['dropped']) == expect_low}; derivation keys={sorted(top.derivation)}")
        r1 = drop_random_k(cand, k=20, seed=0, lineage_root=lr)
        r1b = drop_random_k(cand, k=20, seed=0, lineage_root=lr)
        r2 = drop_random_k(cand, k=20, seed=1, lineage_root=lr)
        print(f"AC3.4 drop_random_k(seed=0) twice -> same subset_id: {r1.subset_id == r1b.subset_id} ({r1.subset_id[:12]}); "
              f"seed=1 differs: {r1.subset_id != r2.subset_id}")
        stale = scores.copy()
        stale.loc[stale.index[0], "content_hash"] = "0" * 64
        try:
            drop_top_k(cand, stale, "score", 20, lineage_root=lr)
            stale_rejected = False
        except LineageError as e:
            stale_rejected = True
            print(f"AC3.4 stale content_hash rejected: {str(e)[:120]}...")
        served, n, _ = served_episode_ids(root, top, eps, lr)
        print(f"AC3.4 drop-top-k child round-trips through LeRobot: {served == set(top.episode_ids)}, frames={n}")
        cache_bytes = sum(p.stat().st_size for p in cache.rglob("*") if p.is_file())
        print(f"(isolated HF cache used by these loads: {cache_bytes / 1e9:.2f} GB, deleted with the block)")
    out["AC3.4"] = {"top_k_child": top.subset_id, "dropped_eq_expected": set(top.derivation["dropped"]) == expect,
                    "lowest_eq_expected": set(low.derivation["dropped"]) == expect_low,
                    "random_reproducible": r1.subset_id == r1b.subset_id, "seed_changes": r1.subset_id != r2.subset_id,
                    "stale_rejected": stale_rejected, "top_k_roundtrip": served == set(top.episode_ids),
                    "isolated_cache_gb": round(cache_bytes / 1e9, 2)}

    # ---- AC3.5 tamper detection on a scratch copy -----------------------------------------
    copy = base / "v0_tamper_copy"
    t0 = time.time()
    subprocess.run(["cp", "-c", "-R", str(root), str(copy)], check=True)  # APFS clone (copy-on-write)
    print(f"AC3.5 cloned {v['lerobot_root']} -> {copy.relative_to(REPO_ROOT)} in {time.time() - t0:.1f}s")
    ok_clean = verify_version("v0", root=copy, lineage_root=lr)
    target = copy / "data" / "chunk-000" / "file-017.parquet"
    size = target.stat().st_size
    with open(target, "r+b") as f:
        f.seek(size // 2)
        b = f.read(1)
        f.seek(size // 2)
        f.write(bytes([b[0] ^ 0x01]))
    try:
        verify_version("v0", root=copy, lineage_root=lr)
        caught = False
    except VersionIntegrityError as e:
        caught = True
        print(f"AC3.5 after flipping 1 bit at byte {size // 2} of data/chunk-000/file-017.parquet: "
              f"VersionIntegrityError: {str(e).replace(str(REPO_ROOT) + '/', '')}")
    ok_orig = verify_version("v0", lineage_root=lr)
    print(f"AC3.5 untouched clone verified first: {ok_clean['ok']} ({ok_clean['files_checked']} files); "
          f"tamper caught: {caught}; original still verifies: {ok_orig['ok']}")
    shutil.rmtree(copy)
    out["AC3.5"] = {"clean_copy_ok": ok_clean["ok"], "tamper_caught": caught, "original_ok": ok_orig["ok"]}

    # ---- AC3.6 no absolute paths in any record of the real lineage store --------------------
    markers = absolute_path_markers()
    offenders, checked = [], 0
    for p in sorted(LINEAGE_ROOT.rglob("*")):
        if p.suffix == ".json":
            text, pattern = p.read_text(), ABS_JSON_VALUE
        elif p.suffix == ".parquet":
            text, pattern = pd.read_parquet(p).astype(str).to_csv(index=False), ABS_CSV_FIELD
        else:
            continue
        checked += 1
        if any(m in text for m in markers) or pattern.search(text):
            offenders.append(p.relative_to(REPO_ROOT).as_posix())
    print(f"AC3.6 scanned {checked} lineage files (json + parquet, incl. mlartifacts copies) under data/lineage/: "
          f"absolute-path hits={offenders}")
    out["AC3.6"] = {"files_scanned": checked, "offenders": offenders}

    # ---- AC3.8 group leave-one-out on v1/candidate (no LeRobot load: R1) --------------------
    v1c = Subset.load("v1/candidate", lr)
    e1 = version_episodes("v1", lr)
    task = sorted(e1.loc[e1["suite"] == "libero_10", "task_name"].unique())[0]
    res38 = {}
    for column, value in (("task_name", task), ("suite", "libero_10")):
        child = drop_group(v1c, column, value, lineage_root=lr)
        remaining = e1[e1["episode_id"].isin(v1c.episode_ids) & (e1[column] != value)]
        ok = set(child.resolve(lr)) == set(remaining["episode_index"].astype(int)) and len(child.resolve(lr)) == len(remaining)
        print(f"AC3.8 drop_group(v1/candidate, {column}={value[:48]!r}): child {child.subset_id[:12]} "
              f"n={len(child.episode_ids)} dropped={len(child.derivation['dropped'])}; resolve() == episode_index set of "
              f"remaining episodes (from v1's episode table): {ok}; derivation keys={sorted(child.derivation)}")
        res38[column] = {"n": len(child.episode_ids), "dropped": len(child.derivation["dropped"]), "resolve_ok": ok,
                         "subset_id": child.subset_id}
    suite_child = Subset.load(res38["suite"]["subset_id"], lr)
    errors = {}
    for label, args in (("non-catalog column", ("corruption_type", "action_gaussian_noise")),
                        ("list column", ("objects", "x")),
                        ("identity column", ("content_hash", "0" * 64)),
                        ("no match", ("task_name", "no_such_task")),
                        ("matches all", ("suite", "libero_spatial"))):
        parent = suite_child if label == "matches all" else v1c
        try:
            drop_group(parent, *args, lineage_root=lr)
            errors[label] = "NOT RAISED"
        except LineageError as e:
            errors[label] = str(e)[:110]
        print(f"AC3.8 raises on {label}: {errors[label]}")
    out["AC3.8"] = {"task": task, "children": res38, "errors": errors}


if __name__ == "__main__":
    sys.exit(main())
