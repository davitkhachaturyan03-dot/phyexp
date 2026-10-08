"""Verify evaluation-layer S0: derived versions keep parent order and are blind to their edits.

Grounding: PoC §2.1 item 4 (p. 3), "Then check whether your attribution method flags exactly those":
a method under test must not be able to read the answer off the derived version's tables.

    .venv/bin/python scripts/verify_blind_s0.py     # ~1-2 min, all writes under data/scratch/ (deleted)

AC0.3: a 20-episode parent (1 demo per task, converted from the raw HDF5 into a scratch lineage root)
gets 3 edits, one of them length-changing (20 idle frames), none at the end. Every episode must keep its
parent's ``episode_index`` and ``check_blind`` must return ``[]``.
AC0.5: four tampered copies of the derived table (extra bookkeeping column, permuted ``episode_index``,
one changed bookkeeping value, shuffled row order) must each be rejected, and ``assert_blind`` must
refuse the fenced v1. Results go to ``artifacts/s0_verify_blind.json``.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from verify_derive_s4 import small_parent  # noqa: E402

from vlapoc.catalog import read_episodes_meta  # noqa: E402
from vlapoc.derive import derive_version, idle_padding_edit, load_labels, robodrop_action_noise  # noqa: E402
from vlapoc.lineage import (  # noqa: E402
    LineageError,
    assert_blind,
    check_blind,
    parent_hashes,
    resolve_version,
    version_episodes,
)

ART = REPO_ROOT / "tracker" / "features" / "2026-10-01-evaluation-layer" / "artifacts"


def _ranges(meta: pd.DataFrame) -> dict:
    """``{data file index: [first, last] episode_index}``: where the writer put the file boundaries."""
    g = meta.groupby("data/file_index")["episode_index"]
    return {int(k): [int(a), int(b)] for k, a, b in zip(g.min().index, g.min(), g.max())}


def tamper(lr: Path, vid: str, name: str, fn) -> list[str]:
    """Apply ``fn`` to the derived version's table in the scratch store, run check_blind, restore."""
    p = lr / "versions" / vid / "episodes.parquet"
    original = p.read_bytes()
    try:
        fn(pd.read_parquet(p)).to_parquet(p, index=False)
        got = check_blind(vid, lr)
    finally:
        p.write_bytes(original)
    print(f"AC0.5 {name}: check_blind -> {got}; rejected: {bool(got)}")
    return got


def main() -> int:
    """AC0.3 (order preserved, blind) on a scratch 20-episode derivation, then the AC0.5 negative tests."""
    out: dict = {}
    scratch = REPO_ROOT / "data" / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    base = Path(tempfile.mkdtemp(prefix="verify_s0_blind_", dir=scratch))
    lr = base / "lineage"
    lr.mkdir()
    try:
        pvid = small_parent(base, lr)
        pe = version_episodes(pvid, lr).sort_values("episode_index").reset_index(drop=True)
        print(f"AC0.3 scratch parent small20 {pvid[:12]}: {len(pe)} episodes, {int(pe['length_steps'].sum())} frames")
        stats = json.loads((REPO_ROOT / resolve_version(pvid, lr)["lerobot_root"] / "meta" / "stats.json").read_text())
        sigma = np.array(stats["action"]["std"])
        targets = {3: "idle_padding x20", 11: "robodrop action noise", 16: "robodrop action noise"}
        ids = {i: pe.loc[i, "episode_id"] for i in targets}
        edits = [idle_padding_edit(ids[3], 20), robodrop_action_noise(ids[11], sigma, 0), robodrop_action_noise(ids[16], sigma, 1)]
        t0 = time.time()
        res = derive_version(pvid, edits, "small20_s0", out_root=base / "lerobot" / "small20_s0", lineage_root=lr)
        dt = time.time() - t0
        cvid = res["dataset_version_id"]
        ce = version_episodes(cvid, lr)
        m = ce.merge(pe[["episode_id", "episode_index", "length_steps"]], on="episode_id", suffixes=("", "_parent"))
        same_idx = int((m["episode_index"] == m["episode_index_parent"]).sum())
        lab = load_labels(cvid, lr)
        edited_idx = sorted(int(ce.set_index("episode_id").loc[e, "episode_index"]) for e in lab["episode_id"])
        cmeta = read_episodes_meta(REPO_ROOT / res["lerobot_root"])
        pmeta = read_episodes_meta(REPO_ROOT / resolve_version(pvid, lr)["lerobot_root"])
        blind = check_blind(cvid, lr)
        ph = parent_hashes(cvid, lr).merge(pe[["episode_id", "content_hash"]], on="episode_id")
        print(f"AC0.3 derive_version: 3 edits at parent episode_index {sorted(targets)} ({list(targets.values())}); "
              f"{dt:.1f}s, timing {res['timing_s']}, mechanism '{res['mechanism']}'")
        print(f"AC0.3 episode_index equal to parent's: {same_idx}/{len(m)}; edited episodes now at {edited_idx}; "
              f"length change on the padded episode: {int(m.loc[m.episode_index == 3, 'length_steps_parent'].item())} -> "
              f"{int(m.loc[m.episode_index == 3, 'length_steps'].item())}")
        print(f"AC0.3 meta/episodes tasks per index equal to parent's: {bool((cmeta['tasks'].map(list) == pmeta['tasks'].map(list)).all())}; "
              f"data-file episode ranges child {_ranges(cmeta)} vs parent {_ranges(pmeta)}")
        print(f"AC0.3 parents.parquet == parent content_hash: {bool((ph['parent_content_hash'] == ph['content_hash']).all())} "
              f"({len(ph)} rows); child columns == parent columns: {list(ce.columns) == list(pe.columns)}")
        print(f"AC0.3 check_blind(child) = {blind}")
        frames = int(ce["length_steps"].sum())
        out["AC0.3"] = {"parent": pvid, "child": cvid, "n_episodes": len(m), "episode_index_equal": same_idx,
                        "edited_parent_index": sorted(targets), "edited_child_index": edited_idx,
                        "check_blind": blind, "timing_s": res["timing_s"], "frames_written": frames,
                        "file_ranges_child": _ranges(cmeta), "file_ranges_parent": _ranges(pmeta),
                        "write_frames_per_s": round(frames / res["timing_s"]["write"], 1),
                        "pass": same_idx == len(m) and blind == [] and edited_idx == sorted(targets)}

        # ---- AC0.5: negative tests --------------------------------------------------------------
        def extra_column(t):
            """Negative test: add the leaky ``parent_content_hash`` column back."""
            return t.assign(parent_content_hash=t["episode_id"].map(dict(zip(pe["episode_id"], pe["content_hash"]))))

        def permute_index(t):
            """Negative test: swap two rows' ``episode_index``."""
            t = t.copy()
            a, b = t.index[t.episode_index == 0][0], t.index[t.episode_index == 5][0]
            t.loc[[a, b], "episode_index"] = t.loc[[b, a], "episode_index"].to_numpy()
            return t

        def change_bookkeeping(t):
            """Negative test: change one bookkeeping value (``source_demo``)."""
            t = t.copy()
            t.loc[t.episode_index == 11, "source_demo"] = "demo_999"
            return t

        def shuffle_rows(t):
            """Negative test: same rows, shuffled order."""
            return t.sample(frac=1.0, random_state=0).reset_index(drop=True)

        neg = {name: tamper(lr, cvid, name, fn) for name, fn in [
            ("extra bookkeeping column parent_content_hash", extra_column),
            ("episode_index of parent rows 0 and 5 swapped", permute_index),
            ("source_demo changed on one edited row", change_bookkeeping),
            ("rows shuffled, values unchanged", shuffle_rows)]}
        print(f"AC0.5 check_blind after restore: {check_blind(cvid, lr)}")
        try:
            assert_blind("v1")
            v1_refused = None
        except LineageError as e:
            v1_refused = str(e)
        assert_blind(cvid, lr)  # must not raise
        print(f"AC0.5 assert_blind(v1) refused: {v1_refused}")
        print("AC0.5 assert_blind(scratch child) passed (no exception)")
        out["AC0.5"] = {"negatives": neg, "all_rejected": all(bool(v) for v in neg.values()),
                        "assert_blind_v1": v1_refused, "assert_blind_child": "passed"}
    finally:
        shutil.rmtree(base, ignore_errors=True)
        print(f"scratch removed: {not base.exists()}")
    (ART / "s0_verify_blind.json").write_text(json.dumps(out, indent=2, default=str))
    return 0 if out.get("AC0.3", {}).get("pass") and out.get("AC0.5", {}).get("all_rejected") else 1


if __name__ == "__main__":
    sys.exit(main())
