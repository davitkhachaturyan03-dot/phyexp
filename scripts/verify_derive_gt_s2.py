"""Verify ground-truth S2 (instruction edits in ``derive_version``; the instruction counted as content in
``compare_blind``) against AC2.1-AC2.6, at small scale in a scratch lineage store.

Grounding: PoC §2.1 item 4 p.3 (the method must "flag exactly those" on its own), §2.5 p.5 ("Swapped or wrong
language instructions"), §8.7 p.15 (corrupt only the training copy); plan design D1 and premises P10-P12.

    .venv/bin/python scripts/verify_derive_gt_s2.py                      # everything, ~3-4 min
    .venv/bin/python scripts/verify_derive_gt_s2.py --child OUT.json     # one derivation (used for AC2.4)

All writes go under ``data/scratch/`` and are deleted at the end. The scratch parent has 20 episodes (1 demo per
task, converted from the raw HDF5) with the full v0 catalog columns (copied from v0 for the same ``episode_id``,
data-derived columns recomputed from the stored bytes), so the instruction columns are exercised. AC2.4 runs the
derivation in two subprocesses with ``PYTHONHASHSEED=0`` and ``=1`` and compares every output file.
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
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vlapoc import corruptions as C  # noqa: E402
from vlapoc.catalog import hash_stored_dataset, read_episodes_meta, read_tasks  # noqa: E402
from vlapoc.derive import EpisodeEdit, derive_version, load_labels, read_stored_episode  # noqa: E402
from vlapoc.lineage import (  # noqa: E402
    LineageError,
    assert_blind,
    check_blind,
    register_version,
    resolve_version,
    version_episodes,
)

FEATURE_DIR = REPO_ROOT / "tracker" / "features" / "2026-10-15-ground-truth"
LABEL_COLUMNS = ["episode_id", "parent_content_hash", "content_hash", "corruption_type", "params", "seed", "hypothesis"]


def full_small_parent(base: Path, lr: Path) -> str:
    """Convert and register a 20-episode parent (1 demo per task) whose catalog has v0's full columns."""
    from vlapoc.libero_convert import convert, iter_raw_episodes

    root = base / "lerobot" / "small20"
    convert(out_root=root, limit_per_task=1, log_every=1000)
    stored = hash_stored_dataset(root, workers=4)
    refs = list(iter_raw_episodes(limit_per_task=1))
    ids = [f"{r.suite}/{r.task_name}/{r.source_demo}" for r in refs]
    v0 = version_episodes("v0").set_index("episode_id")
    cat = v0.loc[ids].reset_index()[version_episodes("v0").columns.tolist()]
    for c in ("episode_index", "content_hash", "length_steps", "smoothness", "idle_fraction", "idle_steps", "gripper_toggles"):
        cat[c] = stored[c].to_numpy()
    cat["duration_s"] = cat["length_steps"] / 20
    same = int((cat["content_hash"].to_numpy() == v0.loc[ids, "content_hash"].to_numpy()).sum())
    print(f"scratch parent small20: {len(cat)} episodes; stored content hash == v0's for the same episode_id: {same}/{len(cat)}")
    cdir = lr / "catalog" / "small20"
    cdir.mkdir(parents=True)
    cat.to_parquet(cdir / "episodes.parquet", index=False)
    (cdir / "catalog.json").write_text(json.dumps({"lerobot_root": root.resolve().relative_to(REPO_ROOT).as_posix(),
                                                   "n_episodes": len(cat)}))
    return register_version(root, "small20", cdir, note="gt S2 scratch parent", lineage_root=lr)["dataset_version_id"]


def seven_edits(pvid: str, lr: Path, seed: int = 0) -> list[EpisodeEdit]:
    """One edit of each of the seven types on distinct eligible episodes of the scratch parent."""
    pe = version_episodes(pvid, lr).sort_values("episode_index").reset_index(drop=True)
    root = REPO_ROOT / resolve_version(pvid, lr)["lerobot_root"]
    stats = json.loads((root / "meta" / "stats.json").read_text())
    sigma = np.array(stats["action"]["std"], dtype=np.float64)
    suite_tasks = {s: sorted(g["language_instruction"].unique()) for s, g in pe.groupby("suite")}
    edits, used = [], set()
    for k, t in enumerate(C.TYPES):
        for i in range(k * 2 + 1, len(pe)):
            if i in used:
                continue
            ep = read_stored_episode(root, int(pe.loc[i, "episode_index"]))
            if C.eligible(t, ep):
                used.add(i)
                row = pe.loc[i]
                edits.append(C.make_edit(t, row["episode_id"], seed, sigma_a=sigma, original_task=ep["task"],
                                         same_suite_tasks=suite_tasks[row["suite"]]))
                break
    assert len(edits) == 7
    return edits


def file_digests(root: Path) -> dict[str, str]:
    """sha256 of every file under ``root``, keyed by relative path."""
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def derive_small(base: Path, name: str = "small20_gt"):
    """Build the scratch store (parent + one edit of each type) and return (lr, pvid, result, edits)."""
    lr = base / "lineage"
    lr.mkdir(parents=True)
    pvid = full_small_parent(base, lr)
    edits = seven_edits(pvid, lr)
    res = derive_version(pvid, edits, name, out_root=base / "lerobot" / name, lineage_root=lr)
    return lr, pvid, res, edits


def child_mode(out: Path) -> None:
    """One derivation in this process; write what AC2.4 compares to ``out``."""
    base = REPO_ROOT / "data" / "scratch" / "gt_s2_hashseed"
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)
    try:
        lr, pvid, res, _ = derive_small(base)
        croot = REPO_ROOT / res["lerobot_root"]
        cvid = res["dataset_version_id"]
        ce = version_episodes(cvid, lr).sort_values("episode_index")
        deriv = json.loads((lr / "versions" / cvid / "derivation.json").read_text())
        out.write_text(json.dumps({
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"), "dataset_version_id": cvid,
            "tasks_map": {k: int(v) for k, v in sorted({v: k for k, v in read_tasks(croot).items()}.items())},
            "tasks_parquet_sha256": hashlib.sha256((croot / "meta" / "tasks.parquet").read_bytes()).hexdigest(),
            "labels_rows_digest": deriv["labels_rows_digest"], "labels_sha256": deriv["labels_sha256"],
            "content_hashes": ce["content_hash"].tolist(), "files": file_digests(croot),
            "stats": json.loads((croot / "meta" / "stats.json").read_text()),
            "stats_text": (croot / "meta" / "stats.json").read_text(),
            "labels_params_sha256": hashlib.sha256(load_labels(cvid, lr)["params"].str.cat().encode()).hexdigest(),
        }, indent=1))
    finally:
        shutil.rmtree(base, ignore_errors=True)


def tamper_row(lr: Path, vid: str, episode_id: str, col: str, value) -> list[str]:
    """Set one cell of the derived version's table in the scratch store, run ``check_blind``, restore."""
    p = lr / "versions" / vid / "episodes.parquet"
    original = p.read_bytes()
    try:
        t = pd.read_parquet(p)
        t.loc[t["episode_id"] == episode_id, col] = value
        t.to_parquet(p, index=False)
        return check_blind(vid, lr)
    finally:
        p.write_bytes(original)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", type=Path, help="run one derivation and write its digests (AC2.4 helper)")
    ap.add_argument("--artifacts", type=Path, default=FEATURE_DIR / "artifacts")
    ap.add_argument("--skip-hashseed", action="store_true")
    a = ap.parse_args()
    if a.child:
        child_mode(a.child)
        return 0
    a.artifacts.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    t0 = time.time()
    base = REPO_ROOT / "data" / "scratch" / "gt_s2"
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)
    ok = True
    try:
        # ---- AC2.1 -------------------------------------------------------------------------------
        lr, pvid, res, edits = derive_small(base)
        cvid, croot = res["dataset_version_id"], REPO_ROOT / res["lerobot_root"]
        proot = REPO_ROOT / resolve_version(pvid, lr)["lerobot_root"]
        blind = check_blind(cvid, lr)
        labels = load_labels(cvid, lr)
        print(f"AC2.1 7 edits ({[e.corruption_type for e in edits]}) registered as {cvid[:12]} in {res['timing_s']['total']}s; "
              f"check_blind = {blind}; assert_blind passes: ", end="")
        assert_blind(cvid, lr)
        print("yes")
        print(f"AC2.1 labels.parquet: {len(labels)} rows, columns {list(labels.columns)}; columns == required: "
              f"{list(labels.columns) == LABEL_COLUMNS}; hypothesis per type: "
              f"{dict(zip(labels['corruption_type'], labels['hypothesis']))}")
        ok1 = blind == [] and list(labels.columns) == LABEL_COLUMNS and len(labels) == 7
        out["AC2.1"] = {"check_blind": blind, "labels_columns": list(labels.columns), "n_labels": len(labels),
                        "types": labels["corruption_type"].tolist(), "hypothesis": labels["hypothesis"].tolist(), "pass": ok1}
        ok &= ok1

        # ---- AC2.3 -------------------------------------------------------------------------------
        pt, ct = read_tasks(proot), read_tasks(croot)
        raw_p = pq.read_table(proot / "meta" / "tasks.parquet").to_pandas()
        raw_c = pq.read_table(croot / "meta" / "tasks.parquet").to_pandas()
        swap = next(e for e in edits if e.corruption_type == "instruction_swap")
        ce = version_episodes(cvid, lr).set_index("episode_id")
        pe = version_episodes(pvid, lr).set_index("episode_id")
        row = ce.loc[swap.episode_id]
        meta = read_episodes_meta(croot)
        files = sorted((croot / "data").rglob("*.parquet"))
        table = pd.concat(pq.read_table(f).to_pandas() for f in files)
        table = table[table["episode_index"] == int(row["episode_index"])]
        print(f"AC2.3 child data files: {[str(f.relative_to(croot)) for f in files]}; rows of the swapped episode: {len(table)}")
        target_idx = {v: k for k, v in pt.items()}[swap.new_task]
        orig_idx = {v: k for k, v in pt.items()}[pe.loc[swap.episode_id, "language_instruction"]]
        rows_idx = sorted(table["task_index"].unique().tolist())
        print(f"AC2.3 tasks.parquet equal to parent's (task -> task_index, from the raw files): {raw_p.equals(raw_c)}; "
              f"{len(ct)} tasks, same order: {pt == ct}")
        print(f"AC2.3 swapped episode {swap.episode_id.split('/')[-2][:40]}../{swap.episode_id.split('/')[-1]}: parent task_index {orig_idx} "
              f"-> data rows carry {rows_idx} (target task's parent index {target_idx}); meta tasks {list(meta.loc[meta.episode_index == row['episode_index'], 'tasks'].iloc[0])[:1]}")
        ok3 = raw_p.equals(raw_c) and pt == ct and rows_idx == [target_idx] and target_idx != orig_idx
        # every other episode keeps its parent's task_index
        others = []
        alltab = pd.concat(pq.read_table(f).to_pandas() for f in files)
        ptab = pd.concat(pq.read_table(f).to_pandas() for f in sorted((proot / "data").rglob("*.parquet")))
        pidx = ptab.groupby("episode_index")["task_index"].first()
        cidx = alltab.groupby("episode_index")["task_index"].first()
        diff_eps = sorted(int(i) for i in pidx.index if pidx[i] != cidx[i])
        print(f"AC2.3 episodes whose task_index differs from the parent's: {diff_eps} (expected only {int(row['episode_index'])})")
        ok3 = ok3 and diff_eps == [int(row["episode_index"])]
        out["AC2.3"] = {"tasks_equal": pt == ct, "raw_tasks_parquet_equal": bool(raw_p.equals(raw_c)), "swapped_rows_task_index": rows_idx,
                        "target_parent_index": target_idx, "original_index": orig_idx, "pass": bool(ok3)}
        ok &= bool(ok3)

        # ---- AC2.6 / blind table -----------------------------------------------------------------
        unedited = next(i for i in ce.index if i not in set(labels["episode_id"]))
        table_rows = [("derived child as written (swap + 6 other edits)", blind)]
        table_rows.append(("task_name changed on the swapped episode (provenance leak)", tamper_row(lr, cvid, swap.episode_id, "task_name", "x")))
        table_rows.append(("libero_task_id changed on the swapped episode", tamper_row(lr, cvid, swap.episode_id, "libero_task_id", 99)))
        table_rows.append(("source_demo changed on an edited (noise) episode",
                           tamper_row(lr, cvid, next(e.episode_id for e in edits if e.corruption_type == "action_noise"), "source_demo", "demo_999")))
        table_rows.append(("language_instruction of an UNEDITED episode changed, meta/episodes not",
                           tamper_row(lr, cvid, unedited, "language_instruction", swap.new_task)))
        table_rows.append(("swapped episode's language_instruction put back to the parent's, meta/episodes keeps the swap",
                           tamper_row(lr, cvid, swap.episode_id, "language_instruction", pe.loc[swap.episode_id, "language_instruction"])))
        print("\nAC2.6 blind-check table (scratch child; tampering is applied to the stored table and undone)")
        print(pd.DataFrame([{"case": c, "check_blind": str(r), "rejected": bool(r)} for c, r in table_rows]).to_string(index=False))
        t_leak = table_rows[1][1]
        ok6 = blind == [] and "task_name" in t_leak and all(bool(r) for _, r in table_rows[1:])
        print(f"AC2.6 provenance leak names task_name: {'task_name' in t_leak}")
        out["AC2.6"] = {"table": [{"case": c, "check_blind": r} for c, r in table_rows], "pass": ok6}
        ok &= ok6

        # ---- AC2.2 -------------------------------------------------------------------------------
        bad_target = pe.loc[swap.episode_id, "language_instruction"] + " (not a benchmark task)"
        eid2 = next(e.episode_id for e in edits if e.corruption_type == "action_noise")
        bad = EpisodeEdit(eid2, "instruction_swap", {}, 0, lambda ep: {**ep, "task": bad_target}, new_task=bad_target)
        target = base / "lerobot" / "should_not_exist"
        n_versions = len(list((lr / "versions").iterdir()))
        try:
            derive_version(pvid, [bad], "should_not_exist", out_root=target, lineage_root=lr)
            msg_bad = None
        except LineageError as e:
            msg_bad = str(e)
        tmp_left = [p.name for p in (base / "lerobot").iterdir() if "should_not_exist" in p.name]
        print(f"\nAC2.2 swap to a string not in the parent's tasks: LineageError: {msg_bad is not None}; message: {msg_bad}")
        print(f"AC2.2 nothing written: out_root exists {target.exists()}, leftover temp dirs {tmp_left}, versions in store "
              f"{n_versions} -> {len(list((lr / 'versions').iterdir()))}")
        # reference episode: the existing guard, on real v0 in a scratch lineage copy (the guard raises before any write)
        from vlapoc.lineage import Subset
        from vlapoc.runs import scratch_lineage_copy

        v0 = resolve_version("v0")
        lr0 = scratch_lineage_copy(base / "lineage_v0", dirs=(f"versions/{v0['dataset_version_id']}", "subsets", "splits"), mlflow_db=False)
        ref_id = Subset.load("v0/reference", lr0).episode_ids[0]
        v0e = version_episodes("v0", lr0).set_index("episode_id")
        suite = v0e.loc[ref_id, "suite"]
        alts = sorted(v0e[v0e["suite"] == suite]["language_instruction"].unique())
        tgt0 = base.parent.parent / "lerobot" / "libero_spatial_10__gt_s2_should_not_exist"
        try:
            derive_version("v0", [C.instruction_swap(ref_id, v0e.loc[ref_id, "language_instruction"], alts, 0)], "gt_s2_should_not_exist",
                           out_root=tgt0, lineage_root=lr0)
            msg_ref = None
        except LineageError as e:
            msg_ref = str(e)
        print(f"AC2.2 swap on reference episode {ref_id.split('/')[-1]}: LineageError: {msg_ref is not None}; message: {msg_ref}; "
              f"nothing written: {not tgt0.exists()}")
        ok2 = msg_bad is not None and not target.exists() and not tmp_left and msg_ref is not None and not tgt0.exists()
        out["AC2.2"] = {"bad_target_message": msg_bad, "reference_message": msg_ref, "nothing_written": ok2, "pass": bool(ok2)}
        ok &= bool(ok2)

        # ---- AC2.5 (own checks; the two existing verifiers are run separately) ----------------------
        c1, c0 = check_blind("v1"), check_blind("v0")
        try:
            assert_blind("v1")
            fence = None
        except LineageError as e:
            fence = str(e)
        print(f"\nAC2.5 check_blind(v1) = {c1}; check_blind(v0) = {c0}; assert_blind(v1) refused: {fence is not None} ({fence})")
        ok5 = c1 == ["episode_index"] and c0 == [] and fence is not None
        out["AC2.5_own"] = {"check_blind_v1": c1, "check_blind_v0": c0, "assert_blind_v1_refused": fence is not None, "pass": ok5}
        ok &= ok5
    finally:
        shutil.rmtree(base, ignore_errors=True)
        print(f"scratch removed: {not base.exists()}")

    # ---- AC2.4: PYTHONHASHSEED 0 vs 1, one subprocess each --------------------------------------------
    if not a.skip_hashseed:
        res4 = {}
        for seed in ("0", "1"):
            outp = REPO_ROOT / "data" / "scratch" / f"gt_s2_child_{seed}.json"
            env = {**os.environ, "PYTHONHASHSEED": seed}
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--child", str(outp)], env=env, cwd=REPO_ROOT,
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout[-2000:], r.stderr[-3000:])
                raise SystemExit(f"child PYTHONHASHSEED={seed} failed")
            res4[seed] = json.loads(outp.read_text())
            outp.unlink()
        a0, a1 = res4["0"], res4["1"]
        differing = sorted(f for f in set(a0["files"]) | set(a1["files"]) if a0["files"].get(f) != a1["files"].get(f))
        same = {k: a0[k] == a1[k] for k in ("tasks_map", "tasks_parquet_sha256", "labels_rows_digest", "content_hashes", "dataset_version_id")}
        print(f"\nAC2.4 PYTHONHASHSEED 0 vs 1: {same}; labels.parquet bytes equal: {a0['labels_sha256'] == a1['labels_sha256']}; "
              f"{len(a0['files'])} child files, differing files: {differing}")
        print(f"AC2.4 dataset_version_id (seed 0) {a0['dataset_version_id'][:16]}, (seed 1) {a1['dataset_version_id'][:16]}")
        if "meta/stats.json" in differing:
            def walk(x, y, path=""):
                """Largest absolute and relative difference between two nested stats structures."""
                if isinstance(x, dict):
                    return [r for k in x for r in walk(x[k], y[k], f"{path}/{k}")]
                xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
                d = np.abs(xa - ya)
                return [(path, float(d.max()), float((d / np.maximum(np.abs(xa), 1e-12)).max()))] if d.max() > 0 else []
            diffs = walk(a0["stats"], a1["stats"])
            print(f"AC2.4 meta/stats.json differs in {len(diffs)} entries; largest relative differences: "
                  f"{sorted(diffs, key=lambda r: -r[2])[:4]}")
            out["AC2.4_stats_json_diffs"] = diffs
            t0_, t1_ = a0["stats_text"], a1["stats_text"]
            first = next(i for i in range(min(len(t0_), len(t1_))) if t0_[i] != t1_[i]) if t0_[:min(len(t0_), len(t1_))] != t1_[:min(len(t0_), len(t1_))] else min(len(t0_), len(t1_))
            keys = lambda d: [list(d), {k: list(v) for k, v in d.items() if isinstance(v, dict)}]  # noqa: E731
            print(f"AC2.4 stats.json parsed equal: {a0['stats'] == a1['stats']}; lengths {len(t0_)} / {len(t1_)}; first differing char {first}: "
                  f"seed0 {t0_[max(0, first - 30):first + 40]!r} | seed1 {t1_[max(0, first - 30):first + 40]!r}; "
                  f"key order equal: {keys(a0['stats']) == keys(a1['stats'])}")
            out["AC2.4_stats_json_text_context"] = [t0_[max(0, first - 60):first + 80], t1_[max(0, first - 60):first + 80]]
        ok4 = all(same.values())
        out["AC2.4"] = {"same": same, "labels_parquet_bytes_equal": a0["labels_sha256"] == a1["labels_sha256"],
                        "n_files": len(a0["files"]), "differing_files": differing, "pass": ok4}
        ok &= ok4
    (a.artifacts / "s2_verify.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nall S2 checks pass: {ok} ({time.time() - t0:.0f}s); summary -> {a.artifacts / 's2_verify.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
