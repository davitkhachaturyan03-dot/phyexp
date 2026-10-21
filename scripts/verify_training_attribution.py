"""Verify the training-and-attribution feature (``tracker/features/2026-10-20-training-attribution/``), story by story.

Grounding: PoC §2.1 item 4 p.3, §3.4 p.7, §4.2-§4.5 pp.9-11, §9.1 Steps 3-4 p.19, §9.2 p.19; owner decision 3
(ground-truth feature); CUPID (docs/papers/CUPID_Influence_Functions_arXiv-2506.19121.pdf) App. A.1-A.2.

    .venv/bin/python scripts/verify_training_attribution.py --blind        # S1: refusal table (~2-3 min)
    .venv/bin/python scripts/verify_training_attribution.py --estimator    # S2: toy-torch, CUPID parity, centring algebra
    .venv/bin/python scripts/verify_training_attribution.py --functionals  # S3: functionals.json, ensemble, provenance
    .venv/bin/python scripts/verify_training_attribution.py --pipeline data/scratch/dry.txt   # S4: parse a dry run

Every MLflow-writing check runs on a scratch lineage store under ``data/scratch/`` (``runs.scratch_lineage_copy``);
the real ``data/lineage/mlflow.db`` sha256 is recorded before and after and must not change (plan rule R7). Scratch
runs use ``allow_dirty=True`` because this script itself is under ``scripts/``.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

FEATURE_DIR = REPO_ROOT / "tracker" / "features" / "2026-10-20-training-attribution"
ARTIFACTS = FEATURE_DIR / "artifacts"
SCRATCH = REPO_ROOT / "data" / "scratch" / "training_attribution"
REAL_DB = REPO_ROOT / "data" / "lineage" / "mlflow.db"


def db_sha() -> str:
    """sha256 of the real lineage store's MLflow database (must be unchanged by every check here, rule R7)."""
    return hashlib.sha256(REAL_DB.read_bytes()).hexdigest()


def n_runs(lr: Path) -> int:
    """Number of MLflow runs in the store at ``lr`` (all states)."""
    from vlapoc.runs import client, experiment_id

    return len(client(lr).search_runs([experiment_id(lr)], max_results=10000))


def fake_train_run(lr: Path, name: str, subset) -> str:
    """A FINISHED ``train`` run in the scratch store whose training input is ``subset`` (no model involved)."""
    import mlflow

    from vlapoc.runs import stage_run

    with stage_run("train", name, lr, allow_dirty=True) as r:
        r.input_subset(subset, "training", None)
        mlflow.set_tag("checkpoint.weights.sha256", "0" * 64)
        return r.run_id


# ---------------------------------------------------------------------------------------------
# S1: blindness
# ---------------------------------------------------------------------------------------------


def blind() -> int:
    """S1, AC1.1-AC1.6: refusal table over four entry points (attribute, features, train, select) x four inputs."""
    import stage as stage_mod  # scripts/stage.py

    from verify_derive_gt_s2 import derive_small
    from verify_method_boundary_s3 import method_sources, scan_boundary
    from vlapoc import method_data
    from vlapoc.attribution import attribute_run
    from vlapoc.lineage import LineageError, Subset, VersionIntegrityError, _bind_name, drop_random_k, version_episodes
    from vlapoc.runs import scratch_lineage_copy

    t0 = time.time()
    sha0 = db_sha()
    shutil.rmtree(SCRATCH / "blind", ignore_errors=True)
    (SCRATCH / "blind").mkdir(parents=True)
    rows: list[dict] = []
    ok_all = True
    dummy = "0" * 32

    def refuse(entry: str, case: str, fn, expect: type, lr: Path, contains: str):
        """Run ``fn``; it must raise ``expect`` mentioning ``contains``, and (for attribute) leave the run count alone."""
        method_data._reset_session()
        before = n_runs(lr)
        try:
            fn()
            got, msg = None, "opened"
        except Exception as e:  # noqa: BLE001
            got, msg = type(e), str(e)
        after = n_runs(lr)
        ok = got is expect and contains in msg and before == after
        rows.append({"entry_point": entry, "input": case, "result": "REFUSED " + got.__name__ if got else "OPENED",
                     "names_reason": contains in msg, "runs_before": before, "runs_after": after, "as_expected": ok,
                     "message": msg[:140]})
        return ok

    try:
        # ---- store A: a copy of the real lineage store (v0..v3, names, fences) -----------------------------
        lr = scratch_lineage_copy(SCRATCH / "blind" / "lineage_a")
        names = json.loads((lr / "names.json").read_text())["subsets"]
        v1c, v2c = Subset.load("v1/candidate", lr), Subset.load("v2/candidate", lr)
        v0c = Subset.load("v0/candidate", lr)
        # a 20-episode ground-truth-derived subset of the clean v0 (built the way verify_method_boundary_s3 builds one)
        oracle = Subset(v0c.dataset_version_id, tuple(v0c.episode_ids[:20]),
                        {"op": "oracle_drop", "parent_subset_id": v0c.subset_id, "groundtruth": True,
                         "dropped": "20 episodes (synthetic oracle subset for the S1 test)"})
        oracle.save(lr)
        _bind_name("subsets", "v0/oracle20", oracle.subset_id, lr)
        child = drop_random_k(oracle, 2, 0, lr)  # a descendant whose chain still contains oracle_drop
        _bind_name("subsets", "v0/oracle20_then_random", child.subset_id, lr)
        train = {"fenced v1": (fake_train_run(lr, "fake-train-v1", v1c), v1c, "fenced=True"),
                 "fenced v2 (superseded)": (fake_train_run(lr, "fake-train-v2", v2c), v2c, "fenced"),
                 "ground-truth-derived (oracle_drop)": (fake_train_run(lr, "fake-train-oracle", oracle), oracle,
                                                        "ground-truth-derived"),
                 "descendant of oracle_drop": (fake_train_run(lr, "fake-train-oracle-child", child), child,
                                               "ground-truth-derived")}
        for case, (rid, sub, contains) in train.items():
            ok_all &= refuse("attribute", case, lambda rid=rid: attribute_run(
                rid, dummy, dummy, Path("x"), Path("y"), "attr-test", lineage_root=lr, allow_dirty=True),
                LineageError, lr, contains)
            ok_all &= refuse("features --which train", case, lambda s=sub: stage_mod.method_view(s.subset_id, lr),
                             LineageError, lr, contains)
            ok_all &= refuse("train", case, lambda s=sub: stage_mod.method_view(s.subset_id, lr), LineageError, lr, contains)
            ok_all &= refuse("select (parent)", case, lambda s=sub: stage_mod.method_view(s.subset_id, lr),
                             LineageError, lr, contains)

        # positive control: the blind v3 opens (so the refusals above are not blanket), and its timing (premise P3)
        method_data._reset_session()
        t1 = time.time()
        view = method_data.open_for_method("v3/candidate", lr)
        t_open = time.time() - t1
        rows.append({"entry_point": "open_for_method (control)", "input": "v3/candidate (blind)", "result": "OPENED",
                     "names_reason": True, "runs_before": 0, "runs_after": 0, "as_expected": len(view.episode_indices) == 960,
                     "message": f"{len(view.episode_indices)} episodes, {t_open:.1f}s incl. integrity hash"})
        ok_all &= rows[-1]["as_expected"]
        method_data._reset_session()

        # ---- store B: a small derived version, one byte of a data file flipped (AC1.4) ----------------------
        lrb, pvid, res, _ = derive_small(SCRATCH / "blind" / "small")
        cvid = res["dataset_version_id"]
        ce = version_episodes(cvid, lrb)
        small = Subset(cvid, tuple(ce["episode_id"]), {"op": "all"})
        small.save(lrb)
        _bind_name("subsets", "small20_gt/all", small.subset_id, lrb)
        rid_b = fake_train_run(lrb, "fake-train-small", small)
        f = next((REPO_ROOT / res["lerobot_root"] / "data").rglob("*.parquet"))
        orig = f.read_bytes()
        try:
            b2 = bytearray(orig)
            b2[len(b2) // 2] ^= 0x01
            f.write_bytes(bytes(b2))
            ok_all &= refuse("attribute", "tampered version (1 byte flipped)", lambda: attribute_run(
                rid_b, dummy, dummy, Path("x"), Path("y"), "attr-test", lineage_root=lrb, allow_dirty=True),
                VersionIntegrityError, lrb, "")
            ok_all &= refuse("features --which train", "tampered version (1 byte flipped)",
                             lambda: stage_mod.method_view(small.subset_id, lrb), VersionIntegrityError, lrb, "")
            ok_all &= refuse("train", "tampered version (1 byte flipped)", lambda: stage_mod.method_view(small.subset_id, lrb),
                             VersionIntegrityError, lrb, "")
            ok_all &= refuse("select (parent)", "tampered version (1 byte flipped)",
                             lambda: stage_mod.method_view(small.subset_id, lrb), VersionIntegrityError, lrb, "")
        finally:
            f.write_bytes(orig)
        method_data._reset_session()
        view_b = method_data.open_for_method(small.subset_id, lrb)
        rows.append({"entry_point": "open_for_method (control)", "input": "same small version, byte restored",
                     "result": "OPENED", "names_reason": True, "runs_before": 0, "runs_after": 0,
                     "as_expected": len(view_b.episode_indices) == 20, "message": f"{len(view_b.episode_indices)} episodes"})
        ok_all &= rows[-1]["as_expected"]

        # ---- AC1.3 static: the stage.py commands actually call method_view ------------------------------------
        tree = ast.parse((REPO_ROOT / "scripts" / "stage.py").read_text())
        calls = {}
        for fn in tree.body:
            if isinstance(fn, ast.FunctionDef) and fn.name in ("cmd_train", "cmd_features", "cmd_select"):
                calls[fn.name] = any(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "method_view" for n in ast.walk(fn))
        attr_src = (REPO_ROOT / "vlapoc" / "attribution.py").read_text()
        calls["attribute_run uses open_for_method, not assert_blind"] = "open_for_method(" in attr_src and "assert_blind(" not in attr_src.split("def attribute_run")[1]
        print(f"AC1.3 stage commands call method_view / attribute_run uses open_for_method: {calls}")
        ok_all &= all(calls.values()) and len(calls) == 4

        # ---- AC1.5 boundary scan covers influence.py and attribution.py ---------------------------------------
        srcs = method_sources()
        names_scanned = [p.name for p in srcs]
        viol = scan_boundary(srcs)
        print(f"AC1.5 boundary scan over {names_scanned}: violations {viol}")
        ok_all &= {"influence.py", "attribution.py"} <= set(names_scanned) and viol == []
    finally:
        shutil.rmtree(SCRATCH / "blind", ignore_errors=True)

    tbl = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_colwidth", 70):
        print(tbl.drop(columns=["message"]).to_string(index=False))
    sha1 = db_sha()
    print(f"real mlflow.db sha256 before {sha0[:16]}... after {sha1[:16]}... unchanged: {sha0 == sha1}")
    ok_all &= bool(tbl["as_expected"].all()) and sha0 == sha1
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(ARTIFACTS / "s1_refusal_table.csv", index=False)
    (ARTIFACTS / "s1_blind.json").write_text(json.dumps(
        {"pass": bool(ok_all), "rows": rows, "static": calls, "boundary_scanned": names_scanned,
         "mlflow_db_sha256_before": sha0, "mlflow_db_sha256_after": sha1, "seconds": round(time.time() - t0)}, indent=1))
    print(f"S1 pass: {ok_all} ({time.time() - t0:.0f}s)")
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--blind", action="store_true", help="S1")
    a = ap.parse_args()
    if a.blind:
        return blind()
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
