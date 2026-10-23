#!/usr/bin/env python
"""Repoint a copied lineage store at its new location (dashboard S8, "results back" step of the server runbook).

Why: MLflow keeps the artifact location of the experiment and of every run as an *absolute* URI in ``mlflow.db`` (``experiments.artifact_location``,
``runs.artifact_uri``; measured on ``data/lineage_demo``: ``file:///.../data/lineage_demo/mlartifacts/...``). A store copied from the server to another
machine therefore keeps pointing at the server's directories, and ``vlapoc.results`` (which downloads every output through MLflow) cannot read it until the
prefix is rewritten. ``runs.scratch_lineage_copy`` does this for the experiment only, for scratch stores that keep reading the real one.

Usage (stop every process that uses the store first; the script edits ``mlflow.db`` in place after writing ``mlflow.db.bak``)::

    .venv/bin/python scripts/repoint_lineage_store.py --root data/lineage_from_server --old /path/on/server/data/lineage --new $PWD/data/lineage_from_server [--dry-run]

``--old`` / ``--new`` are directory prefixes (the store root on the old and the new machine); every ``file://<old>...`` and plain ``<old>...`` location that starts with
``--old`` is rewritten. Prints how many experiment and run locations changed and how many of the run artifact directories exist afterwards (an artifact
directory that was not copied is reported, not an error). Grounding: PoC §2.1 item 1 p. 3 (lineage must survive being moved between machines).
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path


def repoint(db: Path, old: str, new: str, dry_run: bool = False) -> dict:
    """Rewrite ``old`` -> ``new`` in the experiment and run artifact locations of the SQLite store ``db``; returns the counts."""
    old, new = old.rstrip("/"), new.rstrip("/")
    con = sqlite3.connect(db)
    out = {}
    for table, col in (("experiments", "artifact_location"), ("runs", "artifact_uri")):
        rows = con.execute(f"select rowid, {col} from {table} where {col} is not null").fetchall()
        n = 0
        for rowid, loc in rows:
            for prefix in (f"file://{old}", old):
                if loc.startswith(prefix):
                    if not dry_run:
                        con.execute(f"update {table} set {col} = ? where rowid = ?", (loc.replace(prefix, f"file://{new}" if prefix.startswith("file://") else new, 1), rowid))
                    n += 1
                    break
        out[table] = {"rows": len(rows), "rewritten": n}
    if not dry_run:
        con.commit()
    uris = [r[0] for r in con.execute("select artifact_uri from runs").fetchall()]
    con.close()
    exist = sum(Path(u.removeprefix("file://")).is_dir() for u in uris)
    out["run_artifact_dirs_existing"] = {"existing": exist, "total": len(uris), "missing": len(uris) - exist}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True, help="the copied store (contains mlflow.db)")
    ap.add_argument("--old", required=True, help="the store root on the machine it was copied from")
    ap.add_argument("--new", required=True, help="the store root here (absolute)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    db = a.root / "mlflow.db"
    if not db.is_file():
        sys.exit(f"{db} not found")
    if not a.dry_run:
        shutil.copy(db, a.root / "mlflow.db.bak")
    res = repoint(db, a.old, a.new, a.dry_run)
    print(("DRY RUN " if a.dry_run else "") + str(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
