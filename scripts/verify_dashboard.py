"""Verify the dashboard feature (``tracker/features/2026-10-22-dashboard/``), story by story. One mode per story group.

Grounding: PoC §2.7 p.5 (demonstration surface), §1 p.3 (aggregation by metadata), §3.4 p.7 (declared functional), §4.2 pp.9-10 (four
groups), §4.4 pp.10-11 (two-step verdict), §4.5 p.11, §9.1 Steps 1 and 4 pp.18-19, §9.2 p.19 (where "looks bad" and "hurts"
disagree), §2.1 item 4 p.3 (blindness); CUPID p.11 (group effects are only approximated by summed individual influence).

Run from the repo root; each mode exits 0 when every check passes::

    .venv/bin/python scripts/verify_dashboard.py --results       # S1: influence matrix recorded, results loader, provenance label
    .venv/bin/python scripts/verify_dashboard.py --clusters      # S2: the clusters stage (three lenses), sliceline
    .venv/bin/python scripts/verify_dashboard.py --planted       # S3: discovered clusters scored against the planted types
    .venv/bin/python scripts/verify_dashboard.py --discrepancy   # S9: planted/clean x harmful/not, estimated (PoC §9.2)

Every MLflow-writing check runs on a scratch lineage store under ``data/scratch/dashboard/`` (``runs.scratch_lineage_copy``); the real
``data/lineage/mlflow.db`` sha256 is recorded before and after and must not change. Scratch runs use ``allow_dirty=True`` because
this script is under ``scripts/``. Every score table built here is SYNTHETIC (noise plus planted structure) and is labelled as such
in the artifacts it writes; none of it is evidence about any policy.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

FEATURE_DIR = REPO_ROOT / "tracker" / "features" / "2026-10-22-dashboard"
ARTIFACTS = FEATURE_DIR / "artifacts"
SCRATCH = REPO_ROOT / "data" / "scratch" / "dashboard"
REAL_DB = REPO_ROOT / "data" / "lineage" / "mlflow.db"


def db_sha() -> str:
    """sha256 of the real lineage store's MLflow database (must be unchanged by every check here)."""
    return hashlib.sha256(REAL_DB.read_bytes()).hexdigest()


class Checker:
    """Collects ``[PASS]``/``[FAIL]`` lines, and writes them as a JSON artifact at the end."""

    def __init__(self, mode: str):
        self.mode, self.results, self.t0, self.sha0 = mode, [], time.time(), db_sha()

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}", flush=True)
        return bool(ok)

    def finish(self, extra: dict | None = None) -> int:
        self.check("R real data/lineage/mlflow.db sha256 unchanged by this mode", db_sha() == self.sha0, f"{self.sha0[:16]}...")
        ok = all(r[1] for r in self.results)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / f"verify_dashboard_{self.mode}.json").write_text(json.dumps(
            {"mode": self.mode, "pass": ok, "n_pass": sum(r[1] for r in self.results), "n": len(self.results), "seconds": round(time.time() - self.t0),
             "checks": [{"name": n, "pass": p, "detail": d} for n, p, d in self.results], **(extra or {})}, indent=1, default=str))
        print(f"{self.mode}: pass={ok} ({sum(r[1] for r in self.results)}/{len(self.results)}, {time.time() - self.t0:.0f}s)")
        return 0 if ok else 1


def imports_of(path: Path) -> set[str]:
    """Every module name imported anywhere in a source file (``import a.b`` and ``from a.b import c`` give ``a.b``; ``from a import b`` also ``a.b``)."""
    tree = ast.parse(path.read_text())
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module)
            out |= {f"{n.module}.{a.name}" for a in n.names}
    return out


GROUND_TRUTH_SIDE = ("vlapoc.groundtruth", "vlapoc.recovery", "vlapoc.discrepancy")


def forbidden_imports(path: Path, forbidden=GROUND_TRUTH_SIDE) -> list[str]:
    """Imported module names that equal or sit under a ground-truth-side module."""
    return sorted(m for m in imports_of(path) if any(m == f or m.startswith(f + ".") for f in forbidden))


def vt():
    """The training-attribution verifier's SYNTHETIC fixture builders (scratch store, fake train/eval runs, synthetic features)."""
    import verify_training_attribution as v

    return v


# ---------------------------------------------------------------------------------------------
# S1: results loader, recorded influence matrix, provenance
# ---------------------------------------------------------------------------------------------


def fake_chain(lr: Path, tag: str, train_tags: dict | None = None, eval_tags: dict | None = None, attr_tags: dict | None = None,
               attr_params: dict | None = None, stand_in: bool = False) -> str:
    """A recorded train -> eval -> attribute chain of fake runs in the scratch store (no model, no data); returns the attribute run id.
    ``train_tags``/``eval_tags`` default to the values the server writes (``vlapoc.smoke=false``, ``eval.attribution_grade=true``);
    pass ``{}`` to omit them, or other values to flip one condition."""
    import hashlib as h

    from vlapoc.runs import stage_run

    ck = h.sha256(f"ckpt-{tag}".encode()).hexdigest()
    ro = h.sha256(f"rollouts-{tag}".encode()).hexdigest()
    sc = h.sha256(f"scores-{tag}".encode()).hexdigest()
    tt = {"vlapoc.smoke": "false"} if train_tags is None else train_tags
    et = {"eval.attribution_grade": "true"} if eval_tags is None else eval_tags
    with stage_run("train", f"fake-train-{tag}", lr, allow_dirty=True, extra_tags=tt) as r:
        r.produced("checkpoint", ck, "checkpoint")
    with stage_run("eval", f"fake-eval-{tag}", lr, allow_dirty=True, extra_tags=et) as r:
        r.input_entity("checkpoint", ck, "checkpoint")
        r.produced("rollouts", ro, "rollouts")
    with stage_run("attribute", f"fake-attribute-{tag}", lr, stand_in=stand_in, allow_dirty=True, extra_tags=attr_tags) as r:
        r.input_entity("checkpoint", ck, "checkpoint")
        r.input_entity("rollouts", ro, "rollouts")
        if attr_params:
            r.params(attr_params)
        r.produced("score_table", sc, "attribution_scores")
        return r.run_id


def results() -> int:
    """S1, AC1.1-AC1.6."""
    import mlflow  # noqa: F401

    from vlapoc import results as res
    from vlapoc.runs import client, scratch_lineage_copy

    C = Checker("results")
    v = vt()
    shutil.rmtree(SCRATCH / "results", ignore_errors=True)
    base = SCRATCH / "results"
    base.mkdir(parents=True)
    try:
        # --- AC1.1: a new attribute_run records the matrix and its index; results.influence reads them back ---
        fx = v.make_fixture(base, parent="v3/candidate", cover_types=True, train_name="train-dash-s0")
        ev = v.add_eval(fx, "s0", eval_name="eval-dash-s0", returns_name="returns-dash-s0", seed=1)
        ra = v.run_attribute(fx, ev, "attribute-dash-s0")
        lr = fx["lr"]
        run = client(lr).get_run(ra["run_id"])
        recorded = sorted(k[len("output."):-len(".artifact")] for k in run.data.tags if k.startswith("output.") and k.endswith(".artifact"))
        m, index = res.influence(ra["run_id"], lr)
        tab = res.score_table(ra["run_id"], lr)
        n_roll, n_ep = int(run.data.tags["attribute.n_rollouts"]), int(run.data.tags["attribute.n_episodes"])
        C.check("1.1a attribute_run records outputs influence_matrix and influence_index",
                {"influence_matrix", "influence_index"} <= set(recorded), f"recorded outputs: {recorded}")
        C.check("1.1b results.influence matrix shape == (attribute.n_rollouts, attribute.n_episodes); index column order == score table episode_id order",
                m.shape == (n_roll, n_ep) and index["episode_id"] == tab["episode_id"].tolist() and index["content_hash"] == tab["content_hash"].tolist()
                and len(index["rollout_id"]) == n_roll and ra["n_rollouts"] == n_roll,
                f"matrix {m.shape}, tags ({n_roll}, {n_ep}), index {len(index['rollout_id'])} rollouts x {len(index['episode_id'])} episodes")
        ro_df = res.rollouts(ra["run_id"], lr)
        C.check("1.1c results.rollouts joins eval record and returns, with matrix_row == position in index['rollout_id']",
                len(ro_df) == n_roll and "failure_type" in ro_df and (ro_df["matrix_row"].astype(int).map(dict(enumerate(index["rollout_id"]))) == ro_df["rollout_id"]).all(),
                f"{len(ro_df)} rollouts, failure types {sorted(ro_df['failure_type'].unique())}")
        fun = res.functionals(ra["run_id"], lr)
        C.check("1.1d results.functionals declares every score column (PoC §3.4)", set(fun) == {c for c in tab.columns if c.startswith(("infl_", "harm_"))}, f"{len(fun)} declared columns")
        # --- AC1.2: sha mismatch refused; old run gives a clear 'not recorded' error ---
        art = Path(client(lr).download_artifacts(ra["run_id"], "rollout_episode_influence.npy", str(base / "dl")))
        stored = next((lr / "mlartifacts").rglob("rollout_episode_influence.npy"))
        orig = stored.read_bytes()
        bad = np.load(stored)
        bad[0, 0] += 1.0
        np.save(stored, bad)
        try:
            res.influence(ra["run_id"], lr)
            tamper = "did not raise"
        except res.NotRecordedError as e:
            tamper = f"wrong error: {e}"
        except ValueError as e:
            tamper = str(e)
        stored.write_bytes(orig)
        C.check("1.2a a tampered matrix artifact is refused (ValueError naming the sha256)", "sha256" in tamper and "refusing" in tamper, tamper[:140])
        del art
        stand = [r for r in res.find_runs("attribute", lr) if "stand-in" in r["name"]]
        try:
            res.influence(stand[0]["run_id"], lr)
            old = "did not raise"
        except res.NotRecordedError as e:
            old = str(e)
        C.check("1.2b the real store's stand-in attribute run: 'not recorded (run predates dashboard S1)'", bool(stand) and "predates dashboard S1" in old, f"{stand[0]['name'] if stand else None}: {old[:150]}")
        # --- AC1.3: provenance cases ---
        cases = {
            "all real (server-written tags)": ({}, None, None, None, "REAL"),
            "one vlapoc.synthetic=true": ({}, None, {"vlapoc.synthetic": "true"}, None, "SYNTHETIC"),
            "one vlapoc.smoke=true (train run)": ({"train_tags": {"vlapoc.smoke": "true"}}, None, None, None, "SMOKE"),
            "one eval.attribution_grade=false": ({"eval_tags": {"eval.attribution_grade": "false"}}, None, None, None, "SMOKE"),
            "one missing tag (train run has no vlapoc.smoke)": ({"train_tags": {}}, None, None, None, "SMOKE"),
            "one stand-in (attribute run)": ({"stand_in": True}, None, None, None, "SMOKE"),
            "extra: synthetic param True": ({}, None, None, {"synthetic": True}, "SYNTHETIC"),
        }
        rows, ok_all = [], True
        for i, (name, (kw, _, atags, aparams, want)) in enumerate(cases.items()):
            rid = fake_chain(lr, f"p{i}", attr_tags=atags, attr_params=aparams, **kw)
            p = res.provenance(rid, lr)
            ok = p["label"] == want and ((want == "REAL") == (p["reasons"] == [])) and p["n_runs"] == 3
            rows.append({"case": name, "expected": want, "label": p["label"], "reasons": p["reasons"], "ok": ok})
            ok_all &= ok
            print(f"      {name}: {p['label']}  {p['reasons']}")
        C.check(f"1.3 provenance unit check: {sum(r['ok'] for r in rows)}/{len(rows)} cases (6 required + 1 extra), each non-REAL case names its reason",
                ok_all and all(r["reasons"] for r in rows if r["expected"] != "REAL"), "")
        # provenance on the fixture attribute run (fake tags are absent -> not REAL), and on the real store's stand-in
        pf = res.provenance(ra["run_id"], lr)
        C.check("1.3b a fixture attribute run (train/eval without server tags) is not REAL", pf["label"] != "REAL", f"{pf['label']}: {pf['reasons'][:2]}")
        ens_none = res.seed_runs(ra["run_id"], lr)
        C.check("1.3c seed_runs of a seed run is itself", ens_none == [ra["run_id"]])
        (ARTIFACTS / "s1_provenance_cases.json").write_text(json.dumps({"note": "fake runs in a scratch store; no model or data", "cases": rows}, indent=1))
        # --- AC1.4 source scan, AC1.6 docstrings ---
        bad_imports = forbidden_imports(REPO_ROOT / "vlapoc" / "results.py")
        C.check("1.4 results.py imports nothing from vlapoc.groundtruth / recovery / discrepancy", not bad_imports, f"forbidden imports: {bad_imports}")
        doc = (REPO_ROOT / "vlapoc" / "results.py").read_text().split('"""')[1] + (REPO_ROOT / "vlapoc" / "attribution.py").read_text()
        C.check("1.6 grounding docstrings cite §2.7 p. 5, §3.4 p. 7, §2.1 items 1 and 4 p. 3", all(s in doc for s in ("§2.7 (p. 5)", "§3.4 (p. 7)", "§2.1 item 1 (p. 3)", "§2.1 item 4 (p. 3)")))
    finally:
        shutil.rmtree(base, ignore_errors=True)
    return C.finish()


MODES = {"results": results}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for m, h in (("results", "S1"), ("clusters", "S2"), ("planted", "S3"), ("discrepancy", "S9")):
        ap.add_argument(f"--{m}", action="store_true", help=h)
    a = ap.parse_args()
    ran = [m for m in MODES if getattr(a, m, False)]
    if not ran:
        ap.print_help()
        return 2
    return max(MODES[m]() for m in ran)


if __name__ == "__main__":
    sys.exit(main())
