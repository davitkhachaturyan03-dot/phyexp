"""Verify ground-truth S3 (the method-facing loader and the ground-truth boundary) against AC3.1-AC3.6.

Grounding: PoC §2.1 item 4 p.3 ("flags exactly those" on its own), owner decision 3 (brief.md 2026-10-15,
"Enforce now"), evaluation-layer S0 (the v1 fence).

    .venv/bin/python scripts/verify_method_boundary_s3.py     # ~3 min; scratch under data/scratch/ (deleted)

Reads the real store for v0 and v1 (``verify_version`` hashes v0's 6.9 GB once, ~20 s); every tamper, oracle and
diff-guard case uses a scratch lineage store with a 20-episode parent and its derived child (the S2 helper).
"""

from __future__ import annotations

import ast
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from verify_derive_gt_s2 import derive_small  # noqa: E402

from vlapoc import method_data  # noqa: E402
from vlapoc.catalog import CATALOG_COLUMNS  # noqa: E402
from vlapoc.groundtruth import load_answer_key  # noqa: E402
from vlapoc.lineage import (  # noqa: E402
    LineageError,
    Subset,
    VersionIntegrityError,
    check_blind,
    version_episodes,
)
from vlapoc.method_data import open_for_method  # noqa: E402

FEATURE_DIR = REPO_ROOT / "tracker" / "features" / "2026-10-15-ground-truth"
FORBIDDEN_NAMES = {"load_labels", "parent_hashes", "diff_versions", "fence_record", "load_answer_key"}
FORBIDDEN_FILES = ("labels.parquet", "parents.parquet", "fence.json")


def scan_boundary(paths: list[Path]) -> list[str]:
    """AC3.4: AST scan of method-side source. Violations: importing ``vlapoc.groundtruth``; importing or
    referencing ``load_labels`` / ``parent_hashes`` / ``diff_versions`` / ``fence_record`` / ``load_answer_key``;
    a (non-docstring) string constant naming ``labels.parquet`` / ``parents.parquet`` / ``fence.json``."""
    out = []
    for p in paths:
        tree = ast.parse(p.read_text())
        docs = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and n.body \
                    and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant):
                docs.add(id(n.body[0].value))
        for n in ast.walk(tree):
            where = f"{p.name}:{getattr(n, 'lineno', '?')}"
            if isinstance(n, ast.Import):
                out += [f"{where} import {a.name}" for a in n.names if a.name.endswith("groundtruth")]
            elif isinstance(n, ast.ImportFrom):
                mod = n.module or ""
                if mod.endswith("groundtruth") or any(a.name == "groundtruth" for a in n.names):
                    out.append(f"{where} imports vlapoc.groundtruth")
                out += [f"{where} imports {a.name}" for a in n.names if a.name in FORBIDDEN_NAMES]
            elif isinstance(n, ast.Attribute) and n.attr in FORBIDDEN_NAMES:
                out.append(f"{where} uses .{n.attr}")
            elif isinstance(n, ast.Name) and n.id in FORBIDDEN_NAMES:
                out.append(f"{where} uses {n.id}")
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs \
                    and any(f in n.value for f in FORBIDDEN_FILES):
                out.append(f"{where} string {n.value!r}")
    return out


def method_sources() -> list[Path]:
    """``vlapoc/method_data.py`` and every module under ``vlapoc/methods/``."""
    return [REPO_ROOT / "vlapoc" / "method_data.py", *sorted((REPO_ROOT / "vlapoc" / "methods").rglob("*.py"))]


def attempt(label: str, fn, rows: list, expect: type | None):
    """Run ``fn``; record whether it opened or which exception it raised, and compare with ``expect``."""
    method_data._reset_session() if label.startswith("(fresh)") else None
    try:
        view = fn()
        res, msg = "opened", f"{len(view.episode_indices)} episodes"
        got = None
    except Exception as e:  # noqa: BLE001
        view, got, res, msg = None, type(e), "REFUSED " + type(e).__name__, str(e)
    ok = got is expect if expect is not None else got is None
    rows.append({"case": label.replace("(fresh) ", ""), "result": res, "reason": msg[:150], "as_expected": ok})
    return view, ok


def main() -> int:
    t0 = time.time()
    out: dict = {}
    rows: list[dict] = []
    ok_all = True
    base = REPO_ROOT / "data" / "scratch" / "gt_s3"
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)
    try:
        # ---- the real store: v1 fenced, v0 open ---------------------------------------------------------
        method_data._reset_session()
        _, a = attempt("v1/candidate (fenced, real store)", lambda: open_for_method("v1/candidate"), rows, LineageError)
        msg_v1 = rows[-1]["reason"]
        method_data._reset_session()
        _, a2 = attempt("v2/candidate (superseded by v3, real store)", lambda: open_for_method("v2/candidate"), rows, LineageError)
        msg_v2 = rows[-1]["reason"]
        print(f"AC3.1 (r3) v2/candidate refusal says superseded: {'fenced: superseded by v3' in msg_v2}: {msg_v2[:120]}")
        a = a and a2 and "fenced: superseded by v3" in msg_v2
        method_data._reset_session()
        v, b = attempt("v0/candidate (real store)", lambda: open_for_method("v0/candidate"), rows, None)
        print(f"AC3.1 v0/candidate view: {len(v.episode_indices)} episodes, {len(v.reference_ids)} reference ids, catalog columns "
              f"== CATALOG_COLUMNS: {list(v.catalog.columns) == CATALOG_COLUMNS}, root {v.lerobot_root.relative_to(REPO_ROOT)}")
        ep0 = next(iter(v.iter_episodes()))
        print(f"AC3.1 iter_episodes first: {ep0[0].split('/')[-1]} action {ep0[1]['action'].shape} image {ep0[1]['image'].shape}")
        print(f"AC3.1 v1/candidate refusal names the fence: {'fenced=True' in msg_v1}")
        ok_all &= a and b and list(v.catalog.columns) == CATALOG_COLUMNS and len(v.episode_indices) == 960 and len(v.reference_ids) == 40
        method_data._reset_session()

        # ---- scratch store: S2 derived child ------------------------------------------------------------
        lr, pvid, res, edits = derive_small(base)
        cvid = res["dataset_version_id"]
        ce = version_episodes(cvid, lr)
        pe = version_episodes(pvid, lr)
        child_all = Subset(cvid, tuple(ce["episode_id"]), {"op": "all"})
        child_all.save(lr)
        parent_all = Subset(pvid, tuple(pe["episode_id"]), {"op": "all"})
        parent_all.save(lr)
        from vlapoc.lineage import _bind_name
        _bind_name("subsets", "small20_gt/all", child_all.subset_id, lr)
        _bind_name("subsets", "small20/all", parent_all.subset_id, lr)

        _, c = attempt("(fresh) S2 scratch derived version (7 edits, blind)", lambda: open_for_method("small20_gt/all", lr), rows, None)
        ok_all &= c

        # tamper: one byte of a data file
        f = next((REPO_ROOT / res["lerobot_root"] / "data").rglob("*.parquet"))
        orig = f.read_bytes()
        try:
            b2 = bytearray(orig)
            b2[len(b2) // 2] ^= 0x01
            f.write_bytes(bytes(b2))
            method_data._reset_session()
            _, d = attempt("scratch child with 1 byte of a data file changed", lambda: open_for_method("small20_gt/all", lr), rows, VersionIntegrityError)
        finally:
            f.write_bytes(orig)
        ok_all &= d
        method_data._reset_session()

        # ---- AC3.2: ground-truth-derived subsets ----------------------------------------------------------
        from vlapoc.lineage import drop_random_k

        labelled = load_answer_key(cvid, lr)["episode_id"].tolist()
        oracle = Subset(cvid, tuple(i for i in ce["episode_id"] if i not in set(labelled)),
                        {"op": "oracle_drop", "parent_subset_id": child_all.subset_id, "groundtruth": True,
                         "dropped": "all labelled episodes"})
        oracle.save(lr)
        _bind_name("subsets", "small20_gt/oracle", oracle.subset_id, lr)
        grandchild = drop_random_k(oracle, 2, 0, lr)
        _bind_name("subsets", "small20_gt/oracle_then_random", grandchild.subset_id, lr)
        _, e1 = attempt("(fresh) oracle subset (op oracle_drop, labelled episodes removed)", lambda: open_for_method("small20_gt/oracle", lr), rows, LineageError)
        _, e2 = attempt("(fresh) random-k subset of the oracle subset (chain contains oracle_drop)", lambda: open_for_method("small20_gt/oracle_then_random", lr), rows, LineageError)
        ok_all &= e1 and e2

        # ---- AC3.3 diff guard --------------------------------------------------------------------------
        method_data._reset_session()
        _, g0 = attempt("(open) child, then", lambda: open_for_method("small20_gt/all", lr), rows, None)
        _, g1 = attempt("   ... its PARENT in the same process", lambda: open_for_method("small20/all", lr), rows, LineageError)
        method_data._reset_session()
        _, g2 = attempt("(open) parent, then", lambda: open_for_method("small20/all", lr), rows, None)
        _, g3 = attempt("   ... its CHILD in the same process", lambda: open_for_method("small20_gt/all", lr), rows, LineageError)
        method_data._reset_session()
        _, g4 = attempt("(open) child, then the same child again", lambda: open_for_method("small20_gt/all", lr), rows, None)
        _, g5 = attempt("   ... same version, another subset (reopen)", lambda: open_for_method("small20_gt/all", lr), rows, None)
        ok_all &= g0 and g1 and g2 and g3 and g4 and g5
        # real-store analogue: v0 opened, then v1 (descendant) is refused even apart from the fence
        method_data._reset_session()
        open_for_method("v0/candidate")
        try:
            open_for_method("v1/candidate")
            msg = None
        except LineageError as e:
            msg = str(e)
        print(f"AC3.3 real store: v0/candidate then v1/candidate refused: {msg is not None}")
        method_data._reset_session()

        # ---- AC3.5 -------------------------------------------------------------------------------------------
        method_data._reset_session()
        view = open_for_method("small20_gt/all", lr)
        doc = open(REPO_ROOT / "vlapoc" / "method_data.py").read()
        documented = all(f"``{c}``" in doc for c in CATALOG_COLUMNS)
        blind = check_blind(cvid, lr)
        cols_ok = list(view.catalog.columns) == CATALOG_COLUMNS and set(CATALOG_COLUMNS) <= set(ce.columns)
        print(f"\nAC3.5 catalog columns documented in the module docstring: {documented}; view columns == CATALOG_COLUMNS and "
              f"within the compared tables: {cols_ok}; check_blind(scratch child) = {blind}")
        out["AC3.5"] = {"documented": documented, "columns_ok": cols_ok, "check_blind": blind}
        ok_all &= documented and cols_ok and blind == []
        method_data._reset_session()
    finally:
        shutil.rmtree(base, ignore_errors=True)

    # ---- AC3.4 static boundary ---------------------------------------------------------------------------
    clean = scan_boundary(method_sources())
    planted = {}
    scratch = REPO_ROOT / "data" / "scratch" / "gt_s3_planted"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    try:
        cases = {"import groundtruth": "from vlapoc import groundtruth\n",
                 "import module": "import vlapoc.groundtruth as g\n",
                 "load_labels": "from vlapoc.derive import load_labels\n",
                 "parent_hashes attr": "import vlapoc.lineage as L\nx = L.parent_hashes('v1')\n",
                 "labels path": "p = 'versions/x/labels.parquet'\n",
                 "fence path": "p = 'fence.json'\n"}
        for name, src in cases.items():
            (scratch / "bad.py").write_text(src)
            planted[name] = scan_boundary([scratch / "bad.py"])
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"\nAC3.4 scanned {[p.relative_to(REPO_ROOT).as_posix() for p in method_sources()]}: violations {clean}")
    print(f"AC3.4 planted violations detected: { {k: bool(v) for k, v in planted.items()} }")
    ok4 = clean == [] and all(planted.values())
    out["AC3.4"] = {"clean_violations": clean, "planted_detected": {k: bool(v) for k, v in planted.items()}, "pass": ok4}
    ok_all &= ok4

    tbl = pd.DataFrame(rows)
    print("\nrefusal table (can a method open it?)")
    print(tbl.to_string(index=False))
    out["table"] = rows
    out["pass"] = bool(ok_all and tbl["as_expected"].all())
    (FEATURE_DIR / "artifacts" / "s3_verify.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nall S3 checks pass: {out['pass']} ({time.time() - t0:.0f}s)")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
