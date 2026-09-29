"""Verify S4 (derived versions with ground-truth labels) against AC4.1-AC4.5.

Grounding: PoC §2.1 item 4 p.3, §2.5 p.5, §8.7 p.15 ("corrupt only the training copy"), §8.10 p.17
("pad some with idle or hesitant segments"), §10 p.20; RoboDrop App. A.1 p.13.

    .venv/bin/python scripts/verify_derive_s4.py --guard --padding   # AC4.3, AC4.4 (scratch lineage root)
    .venv/bin/python scripts/verify_derive_s4.py --v1                # AC4.1, AC4.2, AC4.5 on the real v1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vlapoc.catalog import read_episodes_meta  # noqa: E402
from vlapoc.derive import (  # noqa: E402
    derive_version,
    idle_padding_edit,
    load_labels,
    protected_episode_ids,
    read_stored_episode,
)
from vlapoc.libero_convert import DEFAULT_REPO_ID  # noqa: E402
from vlapoc.lineage import (  # noqa: E402
    LINEAGE_ROOT,
    LineageError,
    Subset,
    diff_versions,
    recompute_version_id,
    resolve_version,
    verify_version,
    version_episodes,
)

FEATURE_DIR = REPO_ROOT / "tracker" / "features" / "2026-09-28-data-lineage-layer"
SCRATCH_LINEAGE = REPO_ROOT / "data" / "lineage_scratch" / "s4_padding"
SCRATCH_OUT = REPO_ROOT / "data" / "lerobot" / "_scratch_s4_padding"


def guard(summary: dict) -> None:
    ref = Subset.load("v0/reference")
    victim = ref.episode_ids[0]
    target = REPO_ROOT / "data" / "lerobot" / "libero_spatial_10__should_not_exist"
    try:
        derive_version("v0", [idle_padding_edit(victim)], "should_not_exist", out_root=target)
        raised = None
    except LineageError as e:
        raised = str(e)
    print(f"AC4.3 editing reference episode {victim}: LineageError raised: {raised is not None}")
    print(f"AC4.3 message: {raised}")
    print(f"AC4.3 nothing written: {not target.exists()}; protected ids for v0: {len(protected_episode_ids('v0'))}")
    summary["AC4.3"] = {"raised": raised is not None, "message": raised, "nothing_written": not target.exists()}


def padding(summary: dict) -> None:
    for p in (SCRATCH_LINEAGE, SCRATCH_OUT):
        if p.exists():
            shutil.rmtree(p)
    v0 = resolve_version("v0")
    SCRATCH_LINEAGE.mkdir(parents=True)
    shutil.copytree(LINEAGE_ROOT / "versions" / v0["dataset_version_id"], SCRATCH_LINEAGE / "versions" / v0["dataset_version_id"])
    for d in ("subsets", "splits"):
        shutil.copytree(LINEAGE_ROOT / d, SCRATCH_LINEAGE / d)
    shutil.copy(LINEAGE_ROOT / "names.json", SCRATCH_LINEAGE / "names.json")
    cand = Subset.load("v0/candidate")
    target = cand.episode_ids[0]
    eps0 = version_episodes("v0")
    row0 = eps0[eps0["episode_id"] == target].iloc[0]
    t0 = time.time()
    out = derive_version("v0", [idle_padding_edit(target, 20)], "v0pad", out_root=SCRATCH_OUT,
                         note="AC4.4 test: 20 idle frames inserted into one episode", lineage_root=SCRATCH_LINEAGE)
    print(f"AC4.4 derive_version(v0, idle_padding x20 on {target}) -> {out['dataset_version_id'][:12]} "
          f"in {time.time() - t0:.0f}s, timing {out['timing_s']}")
    eps1 = version_episodes(out["dataset_version_id"], SCRATCH_LINEAGE)
    row1 = eps1[eps1["episode_id"] == target].iloc[0]
    meta = read_episodes_meta(SCRATCH_OUT)
    mrow = meta[meta["episode_index"] == row1["episode_index"]].iloc[0]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(DEFAULT_REPO_ID, root=SCRATCH_OUT, episodes=[int(row1["episode_index"])])
    full = LeRobotDataset(DEFAULT_REPO_ID, root=SCRATCH_OUT)
    contiguous = bool((meta["dataset_from_index"].to_numpy()[1:] == meta["dataset_to_index"].to_numpy()[:-1]).all())
    info = json.loads((SCRATCH_OUT / "meta" / "info.json").read_text())
    ep = read_stored_episode(SCRATCH_OUT, int(row1["episode_index"]))
    t = int(row0["length_steps"]) // 2
    idle_ok = bool((ep["action"][t:t + 20, :6] == 0).all() and (ep["image"][t:t + 20] == ep["image"][t]).all())
    d = diff_versions("v0", out["dataset_version_id"], SCRATCH_LINEAGE)
    rec = recompute_version_id(out["dataset_version_id"], lineage_root=SCRATCH_LINEAGE)
    print(f"AC4.4 length {int(row0['length_steps'])} -> {int(row1['length_steps'])} (catalog); meta/episodes length="
          f"{int(mrow['length'])}; LeRobotDataset(episodes=[{int(row1['episode_index'])}]) frames={len(ds)}; "
          f"info total_frames={info['total_frames']} (v0 200340 + 20); full load frames={full.num_frames}; "
          f"from/to contiguous={contiguous}; inserted frames idle={idle_ok}")
    print(f"AC4.4 diff v0 -> v0pad: changed {d['changed']}, identical {d['identical']}; recomputed id matches: "
          f"{rec == out['dataset_version_id']}; labels rows: {len(load_labels(out['dataset_version_id'], SCRATCH_LINEAGE))}")
    ok = (int(row1["length_steps"]) == int(row0["length_steps"]) + 20 == int(mrow["length"]) == len(ds)
          and info["total_frames"] == 200360 and contiguous and d["changed"] == 1 and rec == out["dataset_version_id"])
    summary["AC4.4"] = {"episode": target, "len_before": int(row0["length_steps"]), "len_after": int(row1["length_steps"]),
                        "meta_len": int(mrow["length"]), "served": len(ds), "total_frames": info["total_frames"],
                        "contiguous": contiguous, "idle_ok": idle_ok, "diff": {"changed": d["changed"], "identical": d["identical"]},
                        "recomputed_id_match": rec == out["dataset_version_id"], "timing_s": out["timing_s"], "pass": ok}
    shutil.rmtree(SCRATCH_OUT)
    shutil.rmtree(SCRATCH_LINEAGE)
    print(f"AC4.4 scratch dataset and scratch lineage root removed: {not SCRATCH_OUT.exists() and not SCRATCH_LINEAGE.exists()}")


def v1(summary: dict) -> None:
    v0, v1_ = resolve_version("v0"), resolve_version("v1")
    d = diff_versions("v0", "v1")
    labels = load_labels("v1")
    e1 = version_episodes("v1")
    e0 = version_episodes("v0").set_index("episode_id")
    print(f"AC4.1 version diff v0 v1: changed {d['changed']}, identical {d['identical']}, added {len(d['added'])}, "
          f"removed {len(d['removed'])}")
    print(f"AC4.1 v1 record parent_version_id == v0: {v1_['parent_version_id'] == v0['dataset_version_id']}; "
          f"changed ids == label ids: {set(d['changed_ids']) == set(labels['episode_id'])}")
    pch = e1.set_index("episode_id")["parent_content_hash"]
    print(f"AC4.1 parent_content_hash == v0 content_hash for all 1000 rows: {(pch == e0.loc[pch.index, 'content_hash']).all()}")
    t0 = time.time()
    rec = recompute_version_id("v1")
    ver = verify_version("v1")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(DEFAULT_REPO_ID, root=REPO_ROOT / v1_["lerobot_root"])
    info = json.loads((REPO_ROOT / v1_["lerobot_root"] / "meta" / "info.json").read_text())
    print(f"AC4.1 v1 loads: episodes={ds.num_episodes} frames={ds.num_frames}; info total_episodes={info['total_episodes']} "
          f"total_frames={info['total_frames']} total_tasks={info['total_tasks']} fps={info['fps']}; recomputed id matches "
          f"registered: {rec == v1_['dataset_version_id']}; verify_version files ok: {ver['ok']} ({ver['files_checked']}) "
          f"({time.time() - t0:.0f}s)")
    j = labels.merge(e1, on=["episode_id", "content_hash"], how="left", indicator=True)
    print(f"AC4.2 labels.parquet: {len(labels)} rows, columns {list(labels.columns)}; join with v1 episodes on "
          f"(episode_id, content_hash): {int((j['_merge'] == 'both').sum())}/{len(labels)} matched, 1:1 "
          f"(unique ids {labels['episode_id'].is_unique}); label parent hash == v0 hash: "
          f"{(labels.set_index('episode_id')['parent_content_hash'] == e0.loc[labels['episode_id'], 'content_hash']).all()}")
    ref = set(Subset.load("v0/reference").episode_ids)
    per_task = e1[e1["episode_id"].isin(labels["episode_id"])].groupby(["suite", "task_name"]).size()
    p0 = json.loads(labels["params"].iloc[0])
    print(f"AC4.5 corrupted per task: {sorted(set(per_task.tolist()))} over {len(per_task)} tasks; in reference split: "
          f"{len(set(labels['episode_id']) & ref)}; corruption_type {labels['corruption_type'].unique().tolist()}; "
          f"seed {labels['seed'].unique().tolist()}")
    print(f"AC4.5 params[0]: std_scale={p0['std_scale']} sigma_a={np.round(p0['sigma_a'], 4).tolist()} "
          f"dims='{p0['dims'][:40]}...' source='{p0['sigma_a_source']}'")
    # what actually changed in the corrupted episodes: only actions, by ~0.25 sigma
    sig = np.array(p0["sigma_a"])
    z, same_obs, same_len = [], 0, 0
    for eid in labels["episode_id"].tolist()[:: 10]:  # every 10th label -> 10 episodes, full frames
        a = read_stored_episode(REPO_ROOT / v0["lerobot_root"], int(e0.loc[eid, "episode_index"]))
        b = read_stored_episode(REPO_ROOT / v1_["lerobot_root"], int(e1.set_index("episode_id").loc[eid, "episode_index"]))
        same_len += int(len(a["action"]) == len(b["action"]))
        same_obs += int(np.array_equal(a["state"], b["state"]) and np.array_equal(a["image"], b["image"])
                        and np.array_equal(a["image2"], b["image2"]))
        z.append((b["action"].astype(np.float64) - a["action"].astype(np.float64)) / sig)
    z = np.concatenate(z)
    print(f"AC4.5 10 corrupted episodes checked frame by frame ({len(z)} frames): same length {same_len}/10, "
          f"observations bit-identical {same_obs}/10; (a_v1 - a_v0)/sigma_a per dim: mean "
          f"{np.round(z.mean(0), 3).tolist()} std {np.round(z.std(0), 3).tolist()} (expected 0 and 0.25)")
    summary["AC4.1"] = {"changed": d["changed"], "identical": d["identical"], "parent_ok": v1_["parent_version_id"] == v0["dataset_version_id"],
                        "recomputed_id_match": rec == v1_["dataset_version_id"], "frames": ds.num_frames}
    summary["AC4.2"] = {"rows": len(labels), "joined": int((j["_merge"] == "both").sum()), "columns": list(labels.columns)}
    summary["AC4.5"] = {"per_task": sorted(set(per_task.tolist())), "tasks": int(len(per_task)),
                        "in_reference": len(set(labels["episode_id"]) & ref), "z_mean": np.round(z.mean(0), 4).tolist(),
                        "z_std": np.round(z.std(0), 4).tolist(), "obs_identical": same_obs, "v1_version_id": v1_["dataset_version_id"]}
    labels.head(10).to_csv(FEATURE_DIR / "artifacts" / "s4_labels_head10.csv", index=False)


def columns(summary: dict) -> None:
    """AC4.7: a derived version's episode tables carry the parent's catalog columns + parent_content_hash only."""
    from vlapoc.catalog import CATALOG_COLUMNS

    v1_ = resolve_version("v1")
    tables = {
        "v0 version table": LINEAGE_ROOT / "versions" / resolve_version("v0")["dataset_version_id"] / "episodes.parquet",
        "v1 version table": LINEAGE_ROOT / "versions" / v1_["dataset_version_id"] / "episodes.parquet",
        "v1 catalog copy": REPO_ROOT / "data" / "lineage" / "catalog" / Path(v1_["lerobot_root"]).name / "episodes.parquet",
    }
    res = {}
    for label, path in tables.items():
        cols = list(pd.read_parquet(path).columns)
        label_cols = sorted(set(cols) & {"corruption_type", "params", "seed", "label", "corrupted"})
        res[label] = {"n_columns": len(cols), "label_columns": label_cols, "columns": cols}
        print(f"AC4.7 {label} ({path.relative_to(REPO_ROOT)}): {len(cols)} columns; label columns present: {label_cols}")
    v1cols = res["v1 version table"]["columns"]
    print(f"AC4.7 v1 columns == v0 catalog columns + ['parent_content_hash']: {v1cols == list(CATALOG_COLUMNS) + ['parent_content_hash']}; "
          f"v1 version table == catalog copy columns: {v1cols == res['v1 catalog copy']['columns']}; "
          f"v0 columns == CATALOG_COLUMNS: {res['v0 version table']['columns'] == list(CATALOG_COLUMNS)}")
    summary["AC4.7_columns"] = res


def small_parent(base: Path, lr: Path) -> str:
    """Convert and register a 20-episode parent (1 demo per task) in a scratch lineage root."""
    from vlapoc.catalog import hash_stored_dataset
    from vlapoc.libero_convert import convert, iter_raw_episodes
    from vlapoc.lineage import register_version

    root = base / "lerobot" / "small20"
    convert(out_root=root, limit_per_task=1, log_every=100)
    stored = hash_stored_dataset(root, workers=4)
    refs = list(iter_raw_episodes(limit_per_task=1))
    cat = pd.DataFrame([{"episode_id": f"{r.suite}/{r.task_name}/{r.source_demo}", "suite": r.suite, "task_name": r.task_name,
                         "libero_task_id": r.libero_task_id, "source_file": r.source_file, "source_demo": r.source_demo} for r in refs])
    for c in ("episode_index", "content_hash", "length_steps", "smoothness", "idle_fraction", "idle_steps", "gripper_toggles"):
        cat[c] = stored[c].to_numpy()
    cat["duration_s"] = cat["length_steps"] / 20
    cdir = lr / "catalog" / "small20"
    cdir.mkdir(parents=True)
    cat.to_parquet(cdir / "episodes.parquet", index=False)
    (cdir / "catalog.json").write_text(json.dumps({"lerobot_root": root.resolve().relative_to(REPO_ROOT).as_posix(),
                                                   "n_episodes": len(cat)}))
    return register_version(root, "small20", cdir, note="AC4.8 small parent", lineage_root=lr)["dataset_version_id"]


def cache_listing(d: Path) -> tuple[list[str], int]:
    files = sorted(p for p in d.rglob("*") if p.is_file()) if d.exists() else []
    return [str(p.relative_to(d)) for p in files], sum(p.stat().st_size for p in files)


def cache(summary: dict) -> None:
    """AC4.8: a derivation leaves the shared HF datasets cache unchanged (isolated temp cache, deleted)."""
    import tempfile

    import datasets

    scratch_root = REPO_ROOT / "data" / "scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    base = Path(tempfile.mkdtemp(prefix="verify_s4_cache_", dir=scratch_root))
    lr = base / "lineage"
    lr.mkdir()
    shared = Path(datasets.config.HF_DATASETS_CACHE)
    try:
        pvid = small_parent(base, lr)
        first = version_episodes(pvid, lr).sort_values("episode_index")["episode_id"].iloc[0]
        print(f"AC4.8 scratch parent small20 {pvid[:12]} (20 episodes) registered in {lr.relative_to(REPO_ROOT)}")
        before = cache_listing(shared)
        out = derive_version(pvid, [idle_padding_edit(first, 20)], "small20_pad_isolated", out_root=base / "lerobot" / "pad_isolated",
                             lineage_root=lr)
        after = cache_listing(shared)
        print(f"AC4.8 shared HF_DATASETS_CACHE = {shared.relative_to(REPO_ROOT) if shared.is_relative_to(REPO_ROOT) else shared}: "
              f"before {len(before[0])} files / {before[1]} bytes; after isolated derive_version {len(after[0])} files / {after[1]} bytes; "
              f"identical listing and bytes: {before == after}")
        print(f"AC4.8 derivation {out['dataset_version_id'][:12]}: 1 edit, timing {out['timing_s']}; derivation.json hf_datasets_cache: "
              f"{json.loads((lr / 'versions' / out['dataset_version_id'] / 'derivation.json').read_text())['hf_datasets_cache']!r}")
        leftovers = sorted(p.name for p in scratch_root.glob("hf_datasets_cache_*"))
        print(f"AC4.8 isolated temp caches left under data/scratch/: {leftovers}")
        # contrast: the round-1 behaviour, measured against a throwaway cache dir so the shared cache is untouched
        demo = base / "unisolated_cache_demo"
        demo.mkdir()
        old = datasets.config.HF_DATASETS_CACHE
        datasets.config.HF_DATASETS_CACHE = demo
        try:
            derive_version(pvid, [idle_padding_edit(first, 21)], "small20_pad_unisolated", out_root=base / "lerobot" / "pad_unisolated",
                           lineage_root=lr, isolate_hf_cache=False)
        finally:
            datasets.config.HF_DATASETS_CACHE = old
        files, nbytes = cache_listing(demo)
        arrow = [f for f in files if f.endswith(".arrow")]
        entries = sorted({f.split("/")[1] for f in files if f.startswith("parquet/") and f.count("/") >= 2})
        print(f"AC4.8 contrast, isolate_hf_cache=False: {len(files)} cache files ({len(arrow)} .arrow, {len(entries)} parquet builder "
              f"entries), {nbytes} bytes left behind by one 20-episode derivation")
        summary["AC4.8"] = {"shared_cache": str(shared.relative_to(REPO_ROOT)) if shared.is_relative_to(REPO_ROOT) else "outside repo",
                            "before_files": len(before[0]), "before_bytes": before[1], "after_files": len(after[0]),
                            "after_bytes": after[1], "unchanged": before == after, "leftover_temp_caches": leftovers,
                            "contrast_unisolated_files": len(files), "contrast_unisolated_arrow": len(arrow),
                            "contrast_unisolated_entries": len(entries), "contrast_unisolated_bytes": nbytes}
    finally:
        shutil.rmtree(base, ignore_errors=True)
        print(f"AC4.8 scratch removed: {not base.exists()}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--guard", action="store_true")
    p.add_argument("--padding", action="store_true")
    p.add_argument("--v1", action="store_true")
    p.add_argument("--columns", action="store_true", help="AC4.7 column listing")
    p.add_argument("--cache", action="store_true", help="AC4.8 small-scale derivation cache check")
    a = p.parse_args()
    out = FEATURE_DIR / "artifacts" / "s4_summary.json"
    summary = json.loads(out.read_text()) if out.is_file() else {}
    if a.guard:
        guard(summary)
    if a.padding:
        padding(summary)
    if a.v1:
        v1(summary)
    if a.columns:
        columns(summary)
    if a.cache:
        cache(summary)
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"summary -> {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
