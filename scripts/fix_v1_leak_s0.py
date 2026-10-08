"""Evaluation-layer S0: close v1's column leak, fence its positional leak (owner ruling A, 2026-10-02).

Grounding: PoC §2.1 item 4 (p. 3), "Deliberately corrupt known subsets of training data ... Then check
whether your attribution method flags exactly those." v1's method-facing episode table (and its catalog
copy) carried ``parent_content_hash``, which differs from ``content_hash`` for exactly the 100 labelled
episodes. And v1's row order puts those 100 at ``episode_index`` 900-999 (``delete_episodes`` +
``merge_datasets`` appended them). Either one names the answer without the labels.

What this does (idempotent; owner ruling A: fix the code, fence v1, do not re-derive):

1. Moves ``parent_content_hash`` into the lineage-only ``versions/<v1>/parents.parquet``
   (``vlapoc.lineage.parent_hashes``), checked against v0's ``content_hash`` for all 1000 episodes.
2. Rewrites v1's episode table and catalog copy with exactly v0's 21 columns, rows in v0's row order.
   ``dataset_version_id`` is unchanged: it hashes (schema, sorted [episode_id, content_hash]) only.
3. Runs ``check_blind(v1)``. Expected: ``["episode_index"]``, the positional channel, which lives in
   the physical LeRobot copy and is not rewritten (that would change the S6 checkpoints' episode lists).
4. Fences v1 from method use: ``versions/<v1>/fence.json`` with the before/after table digests (the
   store's own trace of the rewrite) and the check result.

    .venv/bin/python scripts/fix_v1_leak_s0.py      # writes artifacts/s0_fix_v1.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vlapoc.catalog import CATALOG_COLUMNS  # noqa: E402
from vlapoc.derive import load_labels  # noqa: E402
from vlapoc.lineage import (  # noqa: E402
    LINEAGE_ROOT,
    blind_report,
    fence_record,
    parent_hashes,
    resolve_version,
    version_episodes,
    write_fence,
)

ART = REPO_ROOT / "tracker" / "features" / "2026-10-01-evaluation-layer" / "artifacts"


def sha(p: Path) -> str:
    """sha256 hex digest of a file's bytes."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    """Apply the four steps of the module docstring to v1 (idempotent) and write ``artifacts/s0_fix_v1.json``."""
    v0, v1 = resolve_version("v0"), resolve_version("v1")
    vdir = LINEAGE_ROOT / "versions" / v1["dataset_version_id"]
    tables = {"version_table": vdir / "episodes.parquet",
              "catalog_copy": LINEAGE_ROOT / "catalog" / Path(v1["lerobot_root"]).name / "episodes.parquet"}
    untouched = {"version.json": vdir / "version.json", "files.json": vdir / "files.json",
                 "labels.parquet": vdir / "labels.parquet"}
    before = {k: sha(p) for k, p in {**tables, **untouched}.items()}
    e0 = version_episodes("v0")
    out = {"v1": v1["dataset_version_id"], "sha256_before": before}

    t = pd.read_parquet(tables["version_table"])
    if "parent_content_hash" in t.columns:
        parents = (t.set_index("episode_id").loc[e0["episode_id"], ["parent_content_hash"]].reset_index())
        parents.to_parquet(vdir / "parents.parquet", index=False)
        for name, p in tables.items():
            tt = pd.read_parquet(p)
            if not tt.drop(columns=["parent_content_hash"]).set_index("episode_id").sort_index().equals(
                    t.drop(columns=["parent_content_hash"]).set_index("episode_id").sort_index()):
                raise SystemExit(f"{name} differs from the version table; refusing to rewrite")
            tt = tt.drop(columns=["parent_content_hash"]).set_index("episode_id").loc[e0["episode_id"]].reset_index()
            tt = tt[list(e0.columns)]
            tt.to_parquet(p, index=False)
        out["action"] = "moved parent_content_hash to parents.parquet; tables rewritten in v0 row order"
    else:
        out["action"] = "already migrated (no parent_content_hash column); nothing rewritten"

    after = {k: sha(p) for k, p in {**tables, **untouched}.items()}
    out["sha256_after"] = after
    out["unchanged_files"] = {k: before[k] == after[k] for k in untouched}
    e1 = version_episodes("v1")
    cat = pd.read_parquet(tables["catalog_copy"])
    ph = parent_hashes("v1").set_index("episode_id")["parent_content_hash"]
    labels = load_labels("v1")
    j = labels.merge(e1, on=["episode_id", "content_hash"], how="inner")
    out["AC0.1"] = {"v1_table_columns": list(e1.columns), "catalog_copy_columns": list(cat.columns),
                    "equal_v0_columns": list(e1.columns) == list(e0.columns) == list(cat.columns) == CATALOG_COLUMNS,
                    "n_columns": len(e1.columns), "dataset_version_id": resolve_version("v1")["dataset_version_id"],
                    "version_id_unchanged": resolve_version("v1")["dataset_version_id"] == v1["dataset_version_id"]}
    out["AC0.2"] = {"label_join_rows": int(len(j)), "labels": int(len(labels)),
                    "parent_hashes_rows": int(len(ph)),
                    "parent_hashes_equal_v0": bool((ph.loc[e0["episode_id"]].to_numpy() == e0["content_hash"].to_numpy()).all()),
                    "labels_parent_hash_equal_parents_parquet": bool(
                        (labels.set_index("episode_id")["parent_content_hash"] == ph.loc[labels["episode_id"]]).all())}
    rep = blind_report("v1")
    out["AC0.4"] = {"check_blind": rep["offending"], "tables": rep["tables"]}
    if fence_record("v1") is None or out["action"].startswith("moved"):
        write_fence("v1", reason=(
            "positional leak: the 100 edited (labelled) episodes are episode_index 900-999 of the physical copy "
            "(delete_episodes + merge_datasets appended them) and sit in their own data files 73-80, so row order "
            "names the answer without labels (PoC §2.1 item 4 p.3). Owner ruling A (2026-10-02): code fixed "
            "(derive_version preserves parent order), v1 not re-derived; the injection feature derives the real "
            "corrupted versions with the fixed code. v1 stays valid for lineage evidence (data+lineage S6)."),
            details={"check_blind": rep["offending"], "table_rewrite": {
                "what": "dropped parent_content_hash (moved to parents.parquet); rows reordered to v0's row order",
                "sha256_before": {k: before[k] for k in tables}, "sha256_after": {k: after[k] for k in tables},
                "by": "scripts/fix_v1_leak_s0.py (evaluation-layer S0)"}})
    out["fence"] = fence_record("v1")
    (ART / "s0_fix_v1.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: out[k] for k in ("action", "unchanged_files", "AC0.1", "AC0.2")}, indent=2, default=str))
    print(f"AC0.4 check_blind(v1) = {rep['offending']}; fenced: {out['fence'] is not None}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
