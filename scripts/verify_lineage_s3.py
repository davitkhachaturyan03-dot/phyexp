"""Verify S3 (dataset versions, splits, subsets) against AC3.1-AC3.6.

Grounding: PoC §2.1 p.3 (exact dataset versions, episode IDs), §2.6 p.5 and §9.1 Step 4 p.19 (drop
top-k vs drop random-k), §8.3 p.14 / §8.7 p.15 and RoboDrop App. A.1 p.13 (held-out clean references).

The drop-top-k demonstration uses a real catalog column (``smoothness``) as the score; it is a
mechanism check, not an attribution result, and no outcome numbers are invented.

    .venv/bin/python scripts/verify_lineage_s3.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
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
    drop_random_k,
    drop_top_k,
    load_names,
    recompute_version_id,
    resolve_version,
    verify_version,
    version_episodes,
)

FEATURE_DIR = REPO_ROOT / "tracker" / "features" / "2026-09-28-data-lineage-layer"
SCRATCH = Path("/private/tmp/claude-501/-Users-davit-khachaturyan-Projects-personal-vlaExp/"
               "ac005395-c032-4c99-b0c0-0f3ce43bc236/scratchpad")


def served_episode_ids(root: Path, subset: Subset, eps: pd.DataFrame, full_getitem: bool = False):
    """Episode IDs and frame count that LeRobot actually serves for ``episodes=subset.resolve()``."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(DEFAULT_REPO_ID, root=root, episodes=subset.resolve())
    idx_to_id = dict(zip(eps["episode_index"], eps["episode_id"]))
    col = ds.hf_dataset.data.column("episode_index").to_numpy()
    served = {idx_to_id[int(i)] for i in set(col.tolist())}
    n = len(ds)
    if full_getitem:  # every frame through __getitem__, the path a training DataLoader uses
        served_gi = {idx_to_id[int(ds[i]["episode_index"])] for i in range(n)}
        assert served_gi == served, "getitem and column disagree"
    return served, n, len(col)


def main() -> int:
    out: dict = {}
    v = resolve_version("v0")
    vid = v["dataset_version_id"]
    root = REPO_ROOT / v["lerobot_root"]
    eps = version_episodes("v0")

    # ---- AC3.1 recompute the version id from the stored bytes ----------------------------
    t0 = time.time()
    recomputed = recompute_version_id("v0")
    print(f"AC3.1 registered dataset_version_id = {vid}")
    print(f"AC3.1 recomputed from dataset      = {recomputed}  match={recomputed == vid}  ({time.time() - t0:.0f}s)")
    vdir = LINEAGE_ROOT / "versions" / vid
    files = json.loads((vdir / "files.json").read_text())
    print(f"AC3.1 version dir contents: {sorted(p.name for p in vdir.iterdir())}; files.json entries={len(files)} "
          f"(data parquet={sum(f['path'].startswith('data/') for f in files)}); lerobot_root={v['lerobot_root']}")
    out["AC3.1"] = {"registered": vid, "recomputed": recomputed, "match": recomputed == vid, "files": len(files)}

    # ---- AC3.2 splits -----------------------------------------------------------------------
    ref = Subset.load("v0/reference")
    cand = Subset.load("v0/candidate")
    inter = set(ref.episode_ids) & set(cand.episode_ids)
    union = set(ref.episode_ids) | set(cand.episode_ids)
    per_task = eps[eps["episode_id"].isin(ref.episode_ids)].groupby(["suite", "task_name"]).size()
    print(f"AC3.2 reference={len(ref.episode_ids)} candidate={len(cand.episode_ids)} overlap={len(inter)} "
          f"union==version: {union == set(eps['episode_id'])}; reference per task: {sorted(set(per_task.tolist()))} "
          f"over {len(per_task)} tasks; derivation seed={ref.derivation['seed']}")
    out["AC3.2"] = {"reference": len(ref.episode_ids), "candidate": len(cand.episode_ids), "overlap": len(inter),
                    "per_task": sorted(set(per_task.tolist())), "tasks": int(len(per_task))}

    # ---- AC3.3 subset -> LeRobot episodes= round trip ---------------------------------------
    res33 = {}
    for name, s, full in [("v0/reference", ref, True), ("v0/candidate", cand, False)]:
        served, n, ncol = served_episode_ids(root, s, eps, full_getitem=full)
        want_frames = int(s.episodes()["length_steps"].sum())
        ok = served == set(s.episode_ids) and n == want_frames == ncol
        print(f"AC3.3 {name}: subset_id={s.subset_id[:12]} resolve()={len(s.resolve())} indices; served episodes "
              f"== subset: {served == set(s.episode_ids)}; frames served={n} sum(length_steps)={want_frames}"
              f"{' (every frame via __getitem__)' if full else ''} -> {'OK' if ok else 'FAIL'}")
        res33[name] = {"served_eq_subset": served == set(s.episode_ids), "frames": n, "expected": want_frames, "ok": ok}
    out["AC3.3"] = res33

    # ---- AC3.4 drop_top_k / drop_random_k --------------------------------------------------
    scores = cand.episodes()[["episode_id", "content_hash", "smoothness"]].rename(columns={"smoothness": "score"})
    top = drop_top_k(cand, scores, metric="score", k=20, direction="highest",
                     score_source="catalog column 'smoothness' (mechanism demo, not an attribution score)")
    expect = set(scores.sort_values(["score", "episode_id"], ascending=[False, True]).head(20)["episode_id"])
    print(f"AC3.4 drop_top_k: child {top.subset_id[:12]} n={len(top.episode_ids)} (parent 960 - 20); dropped == 20 "
          f"highest scores: {set(top.derivation['dropped']) == expect}; derivation keys="
          f"{sorted(top.derivation)}; parent={top.derivation['parent_subset_id'][:12]} "
          f"score_table_digest={top.derivation['score_table_digest'][:12]}")
    r1 = drop_random_k(cand, k=20, seed=0)
    r1b = drop_random_k(cand, k=20, seed=0)
    r2 = drop_random_k(cand, k=20, seed=1)
    print(f"AC3.4 drop_random_k(seed=0) twice -> same subset_id: {r1.subset_id == r1b.subset_id} ({r1.subset_id[:12]}); "
          f"seed=1 differs: {r1.subset_id != r2.subset_id}; derivation keys={sorted(r1.derivation)}; "
          f"overlap(top-k dropped, random dropped)={len(set(top.derivation['dropped']) & set(r1.derivation['dropped']))}")
    stale = scores.copy()
    stale.loc[stale.index[0], "content_hash"] = "0" * 64
    try:
        drop_top_k(cand, stale, "score", 20)
        stale_rejected = False
    except LineageError as e:
        stale_rejected = True
        print(f"AC3.4 stale content_hash rejected: {str(e)[:140]}...")
    served, n, _ = served_episode_ids(root, top, eps)
    print(f"AC3.4 drop-top-k child round-trips through LeRobot: {served == set(top.episode_ids)}, frames={n}")
    out["AC3.4"] = {"top_k_child": top.subset_id, "dropped_eq_expected": set(top.derivation["dropped"]) == expect,
                    "random_reproducible": r1.subset_id == r1b.subset_id, "seed_changes": r1.subset_id != r2.subset_id,
                    "stale_rejected": stale_rejected, "top_k_roundtrip": served == set(top.episode_ids)}

    # ---- AC3.5 tamper detection on a scratch copy -----------------------------------------
    copy = SCRATCH / "v0_tamper_copy"
    if copy.exists():
        shutil.rmtree(copy)
    t0 = time.time()
    subprocess.run(["cp", "-c", "-R", str(root), str(copy)], check=True)  # APFS clone (copy-on-write)
    print(f"AC3.5 cloned {v['lerobot_root']} -> scratch copy in {time.time() - t0:.1f}s")
    ok_clean = verify_version("v0", root=copy)
    target = copy / "data" / "chunk-000" / "file-017.parquet"
    size = target.stat().st_size
    with open(target, "r+b") as f:
        f.seek(size // 2)
        b = f.read(1)
        f.seek(size // 2)
        f.write(bytes([b[0] ^ 0x01]))
    try:
        verify_version("v0", root=copy)
        caught = False
    except VersionIntegrityError as e:
        caught = True
        print(f"AC3.5 after flipping 1 bit at byte {size // 2} of data/chunk-000/file-017.parquet: "
              f"VersionIntegrityError: {str(e).replace(str(SCRATCH), '<scratch>')}")
    ok_orig = verify_version("v0")
    print(f"AC3.5 untouched clone verified first: {ok_clean['ok']} ({ok_clean['files_checked']} files); "
          f"tamper caught: {caught}; original still verifies: {ok_orig['ok']}")
    shutil.rmtree(copy)
    out["AC3.5"] = {"clean_copy_ok": ok_clean["ok"], "tamper_caught": caught, "original_ok": ok_orig["ok"]}

    # ---- AC3.6 no absolute paths in any lineage record --------------------------------------
    offenders = []
    checked = 0
    for p in sorted(LINEAGE_ROOT.rglob("*")):
        if p.suffix == ".json":
            text = p.read_text()
        elif p.suffix == ".parquet":
            text = pd.read_parquet(p).astype(str).to_csv(index=False)
        else:
            continue
        checked += 1
        if str(REPO_ROOT) in text or "/Users/" in text or "/private/" in text or '"/' in text or "\n/" in text:
            offenders.append(p.relative_to(REPO_ROOT).as_posix())
    print(f"AC3.6 scanned {checked} lineage files (json + parquet) under data/lineage/: absolute-path hits={offenders}")
    out["AC3.6"] = {"files_scanned": checked, "offenders": offenders}
    out["names"] = load_names()

    (FEATURE_DIR / "artifacts" / "s3_summary.json").write_text(json.dumps(out, indent=2, default=str))
    print("summary -> tracker/features/2026-09-28-data-lineage-layer/artifacts/s3_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
