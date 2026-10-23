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


# ---------------------------------------------------------------------------------------------
# SYNTHETIC score tables (noise plus planted structure) recorded as attribute runs in a scratch store
# ---------------------------------------------------------------------------------------------

SCORE_KEYS = ["episode_id", "content_hash", "episode_index", "suite", "libero_task_id", "task_name"]


def synthetic_ret(seed: int = 0) -> pd.DataFrame:
    """A SYNTHETIC returns table (only its column set matters: ``attribution.column_declarations`` derives the declarations from it)."""
    from vlapoc.attribution import RETURN_COLUMNS

    rng = np.random.default_rng(seed)
    n = 30
    ret = pd.DataFrame({c: rng.normal(size=n) for c in RETURN_COLUMNS})
    for c in ("ret_f30", "ret_censored60"):
        ret[c] = (rng.random(n) < 0.4).astype(float)
    ret["failure_type"] = ["success"] * 15 + ["timeout"] * 15
    return ret


def synthetic_score_run(lr: Path, subset, table: pd.DataFrame, name: str, chain: str | None = None, note: str = "",
                        extra_inputs: dict | None = None) -> str:
    """Record ``table`` (SYNTHETIC) as a FINISHED ``attribute`` run in the scratch store ``lr`` on ``subset``: outputs
    ``attribution_scores`` and ``functionals`` (real declarations from ``attribution.column_declarations``), tag
    ``vlapoc.synthetic=true``, param ``synthetic``. Returns the run id."""
    from vlapoc import attribution as at
    from vlapoc.runs import stage_run

    d = SCRATCH / "synthetic_runs" / name
    d.mkdir(parents=True, exist_ok=True)
    table.to_parquet(d / "attribution_scores.parquet", index=False)
    decls = at.column_declarations(table, synthetic_ret(), True, "pm1", ("square", "square"))
    (d / "functionals.json").write_text(json.dumps(decls, indent=1, ensure_ascii=False))
    with stage_run("attribute", name, lr, allow_dirty=True, chain=chain, extra_tags={"vlapoc.synthetic": "true"}) as r:
        r.input_subset(subset, "training", None)
        for role, (kind, eid) in (extra_inputs or {}).items():
            r.input_entity(kind, eid, role)
        r.params({"synthetic": True, "method": "SYNTHETIC noise plus planted structure (verify_dashboard.py); not an attribution result",
                  "planted_structure": note, "n_rows": len(table)})
        r.output_file("attribution_scores", d / "attribution_scores.parquet", kind="score_table")
        r.output_file("functionals", d / "functionals.json", kind="functionals")
        r.tags({"attribute.n_rollouts": 0, "attribute.n_episodes": len(table)})
        return r.run_id


def synthetic_table(episodes: pd.DataFrame, metrics, seed: int, planted: dict[str, np.ndarray]) -> pd.DataFrame:
    """SYNTHETIC ``harm_<metric>`` columns over ``episodes``' keys: independent N(0, 1) noise per column (seeded) plus
    ``planted[column]``, an additive shift per episode (in s.d. units)."""
    out = episodes.sort_values("episode_index")[SCORE_KEYS].reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)
    for m in metrics:
        out[f"harm_{m}"] = rng.normal(size=len(out)) + planted.get(f"harm_{m}", 0.0)
    return out


def predicates_set(row) -> frozenset:
    return frozenset((q["feature"], q["code"]) for q in json.loads(row["predicates"]))


def raw_members(desc: pd.DataFrame, bins: dict, preds: list[dict]) -> np.ndarray:
    """Members of a slice re-evaluated from the RAW descriptor values and the recorded bin specs (independent of ``clusters.predicate_mask``)."""
    m = np.ones(len(desc), bool)
    for q in preds:
        spec, x = bins[q["feature"]], desc[q["feature"]]
        if spec["kind"] == "categorical":
            m &= (x.astype(str) == spec["categories"][q["code"]]).to_numpy()
        elif spec["kind"] == "value":
            m &= (x.to_numpy(float) == spec["values"][q["code"]])
        else:
            cuts, i, v = spec["cuts"], q["code"], x.to_numpy(float)
            lo = (v > cuts[i - 1]) if i > 0 else np.ones(len(v), bool)
            hi = (v <= cuts[i]) if i < len(cuts) else np.ones(len(v), bool)
            m &= lo & hi
    return m


def plant_structure(desc: pd.DataFrame, codes: pd.DataFrame, specs: dict, shift: float = 1.5):
    """Choose, deterministically, the SYNTHETIC planted structure over the real v3 candidate descriptors: slice A (harm on hrt60_lin):
    ``length_vs_task`` top bin AND the scene with most episodes in it; task T (harm on hrt60_lin): the first ``libero_10`` task;
    slice B (boost on f30): ``smoothness`` top bin AND the ``gripper_toggles`` bin whose cell size is closest to 80. Returns
    ``(planted dict, truth dict)``."""
    n = len(desc)
    top = int(codes["length_vs_task"].max())
    in_top = (codes["length_vs_task"] == top).to_numpy()
    scene_code = int(pd.Series(codes["scene"].to_numpy()[in_top]).value_counts().idxmax())
    pa = [{"feature": "length_vs_task", "code": top}, {"feature": "scene", "code": scene_code}]
    ma = (codes["length_vs_task"].to_numpy() == top) & (codes["scene"].to_numpy() == scene_code)
    task = sorted(desc.loc[desc["suite"] == "libero_10", "task_name"].unique())[0]
    mt = (desc["task_name"] == task).to_numpy()
    stop = int(codes["smoothness"].max())
    cells = {c: int(((codes["smoothness"] == stop) & (codes["gripper_toggles"] == c)).sum()) for c in sorted(codes["gripper_toggles"].unique())}
    gb = min(cells, key=lambda c: abs(cells[c] - 80))
    pb = [{"feature": "smoothness", "code": stop}, {"feature": "gripper_toggles", "code": int(gb)}]
    mb = (codes["smoothness"].to_numpy() == stop) & (codes["gripper_toggles"].to_numpy() == gb)
    planted = {"harm_hrt60_lin": shift * (ma | mt).astype(float), "harm_f30": -shift * mb.astype(float)}
    truth = {"slice_a": {"predicates": pa, "n": int(ma.sum()), "text": " AND ".join(f"{q['feature']} {specs[q['feature']]['labels'][q['code']]}" for q in pa)},
             "task": {"task_name": task, "n": int(mt.sum()), "overlap_with_slice_a": int((ma & mt).sum())},
             "slice_b": {"predicates": pb, "n": int(mb.sum()), "text": " AND ".join(f"{q['feature']} {specs[q['feature']]['labels'][q['code']]}" for q in pb)},
             "shift_sd": shift}
    return planted, truth


def clusters_mode() -> int:
    """S2, AC2.1-AC2.7."""
    import subprocess

    from vlapoc import clusters as cl
    from vlapoc import results as res
    from vlapoc.lineage import LineageError, Subset, _bind_name
    from vlapoc.method_data import _reset_session, open_for_method
    from vlapoc.runs import client, scratch_lineage_copy, trace

    C = Checker("clusters")
    shutil.rmtree(SCRATCH / "clusters", ignore_errors=True)
    base = SCRATCH / "clusters"
    base.mkdir(parents=True)
    try:
        lr = scratch_lineage_copy(base / "lineage")
        v3 = Subset.load("v3/candidate", lr)
        t0 = time.time()
        view = open_for_method("v3/candidate", lr)
        desc = cl.descriptor_table(view, True)
        print(f"descriptors for {len(desc)} v3/candidate episodes via open_for_method in {time.time() - t0:.0f}s (cached under data/dashboard_cache/descriptors/)")
        codes, specs, dropped = cl.bin_descriptors(desc)
        planted, truth = plant_structure(desc, codes, specs)
        print("planted (SYNTHETIC):", json.dumps(truth))
        metrics = ["f30", "rmst60", "auc_expected", "hrt60_lin", "censored60"]
        eps = v3.episodes(lr)
        tab = synthetic_table(eps, metrics, 0, planted)
        note = (f"+{truth['shift_sd']} sd on harm_hrt60_lin for ({truth['slice_a']['text']}) and for task {truth['task']['task_name']}; "
                f"-{truth['shift_sd']} sd on harm_f30 for ({truth['slice_b']['text']}); noise elsewhere")
        sid = synthetic_score_run(lr, v3, tab, "attribute-dash-synth-s2", note=note)
        out_dir = base / "out"
        n_stage0 = len(res.find_runs("cluster", lr))
        t0 = time.time()
        out = cl.clusters_run(sid, "clusters-dash-synth", out_dir=out_dir, lineage_root=lr, allow_dirty=True, synthetic=True)
        t_stage = time.time() - t0
        print(f"clusters stage: {out['n_clusters']} clusters {out['n_by_lens']} in {t_stage:.0f}s")
        clu = pd.read_parquet(out_dir / "clusters.parquet")
        pats = json.loads((out_dir / "patterns.json").read_text())
        mem = pd.read_parquet(out_dir / "members.parquet")
        desc_out = pd.read_parquet(out_dir / "descriptors.parquet")
        bins = json.loads((out_dir / "bins.json").read_text())["bins"]
        pa_set = frozenset((q["feature"], q["code"]) for q in truth["slice_a"]["predicates"])
        pb_set = frozenset((q["feature"], q["code"]) for q in truth["slice_b"]["predicates"])
        sl = clu[clu["lens"] == "slice"]

        def top_slices(metric: str, side: str, k: int = 3):
            d = sl[(sl["target_metric"] == f"harm_{metric}") & (sl["direction"] == side)].sort_values("rank").head(k)
            return [(int(r["rank"]), r["definition"], int(r["n"])) for _, r in d.iterrows()], [predicates_set(r) for _, r in d.iterrows()]

        hrt_top, hrt_sets = top_slices("hrt60_lin", "harm")
        f30_top, f30_sets = top_slices("f30", "harm")
        lab = clu[(clu["lens"] == "label") & (clu["label_column"] == "task_name")]
        task_rank = {m: int(lab.loc[lab["definition"] == f"task_name = {truth['task']['task_name']}", f"harm_rank__harm_{m}"].iloc[0]) for m in metrics}
        C.check("2.1a planted slice (SYNTHETIC) is among the top-3 harm slices for harm_hrt60_lin", pa_set in hrt_sets, f"top-3 for hrt60_lin: {hrt_top}")
        C.check("2.1b planted task group is among the top-3 label groups (task_name) for harm_hrt60_lin", task_rank["hrt60_lin"] <= 3, f"rank of {truth['task']['task_name']}: {task_rank}")
        C.check("2.1c on harm_f30 neither the planted slice nor the planted task is in the top 3", pa_set not in f30_sets and task_rank["f30"] > 3,
                f"top-3 harm slices for f30: {f30_top}; task rank on f30 = {task_rank['f30']}")
        boost_top, boost_sets = top_slices("f30", "boost")
        harm_f30_all = [predicates_set(r) for _, r in sl[(sl["target_metric"] == "harm_f30") & (sl["direction"] == "harm")].iterrows()]
        C.check("2.2 planted boost slice (negative harm on f30) found on the boost side (top 3) and on no harm side of f30",
                pb_set in boost_sets and pb_set not in harm_f30_all, f"top-3 boost slices for f30: {boost_top}; in the f30 harm list: {pb_set in harm_f30_all}")
        # AC2.3
        cols = out["columns"]
        decl_ok = all(set(json.loads(r)) == set(cols) and all(all(k in d for k in ("functional", "tau", "rollout_subset", "good_sign")) for d in json.loads(r).values())
                      for r in clu["declarations"])
        pat_ok = all(set(p["declarations"]) == set(cols) and p["status"] == "estimated" for p in pats)
        C.check("2.3 every clusters row and patterns entry carries functional/tau/rollout_subset/good_sign of each reported column; status == estimated everywhere",
                decl_ok and pat_ok and (clu["status"] == "estimated").all() and len(pats) == len(clu),
                f"{len(clu)} rows, {len(pats)} patterns, e.g. {json.loads(clu['declarations'].iloc[0])['harm_hrt60_lin']['functional']!r} tau {json.loads(clu['declarations'].iloc[0])['harm_hrt60_lin']['tau']!r}")
        # AC2.4
        run = client(lr).get_run(out["run_id"])
        tr = trace(out["run_id"], lr)
        score_sha = client(lr).get_run(sid).data.tags["output.attribution_scores.sha256"]
        prm = run.data.params
        C.check("2.4 cluster-stage run: input entity = the score table's sha256, params hold lens settings, descriptors, bin edges and synthetic; trace reaches the attribute run",
                run.data.tags["stage"] == "cluster" and run.data.tags["input.scores"] == f"score_table:{score_sha}" and sid in tr["runs"]
                and all(k in prm for k in ("lenses", "descriptors", "bin_edges", "slice_finder", "synthetic", "n_bins")) and prm["synthetic"] == "True",
                f"trace reaches {sorted(r['name'] for r in tr['runs'].values())}; lenses {prm['lenses']}; {len(json.loads(prm['descriptors']))} descriptors")
        pv = res.provenance(out["run_id"], lr)
        C.check("2.4b results.provenance of the cluster run is SYNTHETIC (tag from the producing attribute run)", pv["label"] == "SYNTHETIC", str(pv["reasons"][:1]))
        # AC2.5
        bad_imports = forbidden_imports(REPO_ROOT / "vlapoc" / "clusters.py")
        src = (REPO_ROOT / "vlapoc" / "clusters.py").read_text()
        C.check("2.5a clusters.py imports nothing from groundtruth / recovery / discrepancy, and reads the subset only via open_for_method (no Subset.episodes, no labels)",
                not bad_imports and "open_for_method(" in src and ".episodes(" not in src and "labels.parquet" not in src and "load_answer_key" not in src, f"forbidden imports: {bad_imports}")
        refusals = []
        n_before = len(client(lr).search_runs([client(lr).get_experiment_by_name("vlapoc-lineage").experiment_id], max_results=10000))
        for nm, sub in (("fenced v1", Subset.load("v1/candidate", lr)), ("fenced v2", Subset.load("v2/candidate", lr))):
            t = synthetic_table(sub.episodes(lr), ["f30"], 0, {})
            fid = synthetic_score_run(lr, sub, t, f"attribute-synth-{nm.replace(' ', '-')}")
            _reset_session()
            n_b = len(client(lr).search_runs([client(lr).get_experiment_by_name("vlapoc-lineage").experiment_id], max_results=10000))
            try:
                cl.clusters_run(fid, f"clusters-bad-{nm.replace(' ', '-')}", metrics=["f30"], out_dir=base / "bad", lineage_root=lr, allow_dirty=True, heuristics=False, cache_dir=None)
                refusals.append((nm, False, "opened"))
            except LineageError as e:
                n_a = len(client(lr).search_runs([client(lr).get_experiment_by_name("vlapoc-lineage").experiment_id], max_results=10000))
                refusals.append((nm, "fenced" in str(e) and n_a == n_b, str(e)[:90]))
        v0c = Subset.load("v0/candidate", lr)
        oracle = Subset(v0c.dataset_version_id, tuple(v0c.episode_ids[:20]), {"op": "oracle_drop", "parent_subset_id": v0c.subset_id, "groundtruth": True, "dropped": "synthetic"})
        oracle.save(lr)
        _bind_name("subsets", "v0/oracle20", oracle.subset_id, lr)
        t = synthetic_table(oracle.episodes(lr), ["f30"], 0, {})
        oid = synthetic_score_run(lr, oracle, t, "attribute-synth-oracle")
        _reset_session()
        n_b = len(client(lr).search_runs([client(lr).get_experiment_by_name("vlapoc-lineage").experiment_id], max_results=10000))
        try:
            cl.clusters_run(oid, "clusters-bad-oracle", metrics=["f30"], out_dir=base / "bad", lineage_root=lr, allow_dirty=True, heuristics=False, cache_dir=None)
            refusals.append(("ground-truth-derived", False, "opened"))
        except LineageError as e:
            n_a = len(client(lr).search_runs([client(lr).get_experiment_by_name("vlapoc-lineage").experiment_id], max_results=10000))
            refusals.append(("ground-truth-derived", "ground-truth-derived" in str(e) and n_a == n_b, str(e)[:90]))
        _reset_session()
        C.check("2.5b clusters_run refuses fenced v1, fenced v2 and a ground-truth-derived subset (LineageError naming the reason), nothing written",
                all(r[1] for r in refusals), "; ".join(f"{a}: {'REFUSED' if b else 'WRONG'} ({c})" for a, b, c in refusals))
        del n_before
        # AC2.6
        ok_n = {}
        for lc in cl.LABEL_COLUMNS:
            ok_n[lc] = int(lab["n"].sum()) if lc == "task_name" else int(clu[(clu["lens"] == "label") & (clu["label_column"] == lc)]["n"].sum())
        sizes_ok = all(v == len(tab) for v in ok_n.values())
        bad = []
        for _, r in sl.iterrows():
            m_raw = raw_members(desc_out, bins, json.loads(r["predicates"]))
            got = set(mem.loc[mem["cluster_id"] == r["cluster_id"], "episode_id"])
            if got != set(desc_out.loc[m_raw, "episode_id"]) or int(m_raw.sum()) != int(r["n"]):
                bad.append(r["cluster_id"])
        from vlapoc.analysis import metric_groups

        groups = metric_groups(tab.set_index("episode_id").loc[desc_out["episode_id"]].reset_index(), cols, 1.0)
        gclu = clu[clu["lens"] == "metric_group"]
        gbad = [r["cluster_id"] for _, r in gclu.iterrows()
                if (set(mem.loc[mem["cluster_id"] == r["cluster_id"], "episode_id"]) - set(desc_out.loc[(groups == r["group"]).to_numpy(), "episode_id"]))]
        C.check("2.6 label-group n sum to N for task_name, scene and suite; every slice's members equal its definition re-evaluated from the raw descriptors and recorded edges; group members belong to their group",
                sizes_ok and not bad and not gbad and len(sl) > 0, f"label n sums {ok_n} (N={len(tab)}); {len(sl)} slices re-evaluated, mismatches {bad}; {len(gclu)} group clusters, mismatches {gbad}")
        # AC2.7: sliceline installed, imports on 3.14, with and without numba, same answer
        import importlib.metadata as md

        code = ("import sys, json, warnings\nwarnings.simplefilter('ignore')\nif sys.argv[1] == 'nonumba':\n    sys.modules['numba'] = None\n"
                "import numpy as np\nfrom sliceline.slicefinder import Slicefinder, is_numba_available\n"
                "rng = np.random.default_rng(0); X = rng.integers(0, 4, size=(500, 3)); e = np.maximum(rng.normal(size=500) + 2.0 * ((X[:, 0] == 1) & (X[:, 1] == 2)), 0)\n"
                "sf = Slicefinder(alpha=0.95, k=3, max_l=2, min_sup=10, verbose=False).fit(X, e)\n"
                "print(json.dumps({'numba': is_numba_available(), 'top': [[None if (v is None or v != v) else int(v) for v in s] for s in sf.top_slices_], 'py': sys.version.split()[0]}))\n")
        outs = {}
        for mode in ("numba", "nonumba"):
            pr = subprocess.run([sys.executable, "-c", code, mode], capture_output=True, text=True, cwd=REPO_ROOT)
            outs[mode] = json.loads(pr.stdout.strip().splitlines()[-1]) if pr.returncode == 0 else {"error": pr.stderr[-300:]}
        C.check("2.7 sliceline==0.3.0 installed in .venv; imports and finds the planted slice on Python 3.14 with numba and with numba blocked, same slices",
                md.version("sliceline") == "0.3.0" and outs["numba"].get("py", "").startswith("3.14") and not outs["nonumba"].get("numba", True)
                and outs["numba"]["top"] == outs["nonumba"]["top"] and [1, 2, None] in outs["nonumba"]["top"],
                f"sliceline {md.version('sliceline')}; python {outs['numba'].get('py')}; numba used: {outs['numba'].get('numba')} / {outs['nonumba'].get('numba')}; top slice {outs['nonumba'].get('top', [None])[0]}")
        # --- measured power probe (information, not a pass condition): detection of the planted slices over seeds, and a small slice ---
        probe = {"seeds": [], "note": "SYNTHETIC. n_boot=50 to keep this fast; the slice finder does not use the bootstrap."}
        decl = res.functionals(sid, lr)
        min_sup = max(10, int(round(0.02 * len(codes))))
        top_lv = int(codes["length_vs_task"].max())
        cells = {c: int(((codes["length_vs_task"] == top_lv) & (codes["gripper_toggles"] == c)).sum()) for c in sorted(codes["gripper_toggles"].unique())}
        sc_ = min((c for c in cells if cells[c] >= min_sup), key=lambda c: cells[c])  # the smallest cell the slice finder may return (>= min_sup)
        small_pred = [{"feature": "length_vs_task", "code": top_lv}, {"feature": "gripper_toggles", "code": int(sc_)}]
        small = (codes["length_vs_task"] == top_lv).to_numpy() & (codes["gripper_toggles"] == sc_).to_numpy()
        for seed in range(10):
            t2 = synthetic_table(eps, metrics, 100 + seed, planted | {"harm_censored60": 1.5 * small.astype(float)})
            d2 = cl.discover(t2, view, decl, cl.harm_columns(t2, None), {"n_boot": 50}, cache_dir=cl.DEFAULT_CACHE)
            c2 = d2["clusters"]
            s2 = c2[c2["lens"] == "slice"]

            def rank_of(metric, pset, side="harm"):
                d = s2[(s2["target_metric"] == f"harm_{metric}") & (s2["direction"] == side)].sort_values("rank")
                r = [int(x["rank"]) for _, x in d.iterrows() if predicates_set(x) == pset]
                return r[0] if r else float('nan')

            l2 = c2[(c2["lens"] == "label") & (c2["label_column"] == "task_name")]
            probe["seeds"].append({"seed": 100 + seed, "slice_a_rank_hrt": rank_of("hrt60_lin", pa_set), "slice_a_rank_f30": rank_of("f30", pa_set),
                                   "slice_b_boost_rank_f30": rank_of("f30", pb_set, "boost"),
                                   "small_slice_n": int(small.sum()), "small_slice_rank_censored60": rank_of("censored60", frozenset((q["feature"], q["code"]) for q in small_pred)),
                                   "task_rank_hrt": int(l2.loc[l2["definition"] == f"task_name = {truth['task']['task_name']}", "harm_rank__harm_hrt60_lin"].iloc[0]),
                                   "task_rank_f30": int(l2.loc[l2["definition"] == f"task_name = {truth['task']['task_name']}", "harm_rank__harm_f30"].iloc[0])})
        pr_ = pd.DataFrame(probe["seeds"])
        probe["summary"] = {"slice_a_in_top3_hrt": int((pr_["slice_a_rank_hrt"].fillna(99) <= 3).sum()), "slice_a_in_top3_f30 (false alarm)": int((pr_["slice_a_rank_f30"].fillna(99) <= 3).sum()),
                            "slice_b_boost_in_top3_f30": int((pr_["slice_b_boost_rank_f30"].fillna(99) <= 3).sum()), "task_top3_hrt": int((pr_["task_rank_hrt"] <= 3).sum()),
                            "task_top3_f30 (false alarm)": int((pr_["task_rank_f30"] <= 3).sum()),
                            f"small_slice_(n={int(small.sum())})_in_top3_censored60": int((pr_["small_slice_rank_censored60"].fillna(99) <= 3).sum()), "of_seeds": 10}
        print("power probe over 10 seeds (SYNTHETIC):", json.dumps(probe["summary"]))
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "s2_power_probe_SYNTHETIC.json").write_text(json.dumps({**probe, "truth": truth}, indent=1))
        # demo artifacts: the clusters table (key columns) and the patterns of the planted clusters, all SYNTHETIC
        keep = ["cluster_id", "lens", "definition", "direction", "target_metric", "n", "share_of_subset", "status", "harm_hrt60_lin__mean", "harm_hrt60_lin__lo", "harm_hrt60_lin__hi",
                "harm_hrt60_lin__reading", "harm_f30__mean", "harm_f30__lo", "harm_f30__hi", "harm_f30__reading", "synthetic"]
        clu[keep].to_csv(ARTIFACTS / "s2_clusters_SYNTHETIC.csv", index=False)
        show = [p for p in pats if p["cluster_id"].startswith("slice:hrt60_lin:harm:1") or p["cluster_id"] == f"label:task_name={truth['task']['task_name']}" or p["cluster_id"].startswith("slice:f30:boost:1")]
        (ARTIFACTS / "s2_patterns_SYNTHETIC.json").write_text(json.dumps({"SYNTHETIC": "noise plus planted structure; not a result about any policy", "planted": truth, "patterns": show}, indent=1))
        print("example pattern (SYNTHETIC):", [p["text"] for p in pats if p["cluster_id"] == "slice:hrt60_lin:harm:1"])
    finally:
        shutil.rmtree(base, ignore_errors=True)
        shutil.rmtree(SCRATCH / "synthetic_runs", ignore_errors=True)
    return C.finish({"SYNTHETIC": True, "planted": truth, "stage_seconds": round(t_stage)})


# ---------------------------------------------------------------------------------------------
# S3: discovered clusters vs planted types
# ---------------------------------------------------------------------------------------------


def fake_cluster_run(lr: Path, subset, members: dict[str, list[str]], name: str, decls_json: str, synthetic: bool = True) -> str:
    """A FINISHED ``cluster`` stage run with HAND-BUILT clusters (``members``: cluster id -> episode ids of ``subset``), SYNTHETIC: it did not come
    from ``vlapoc.clusters``. Outputs the five files ``results.clusters`` reads."""
    from vlapoc.runs import stage_run

    keys = subset.keys(lr).set_index("episode_id")["content_hash"]
    clusters = pd.DataFrame([{"cluster_id": cid, "lens": "label", "definition": f"hand-built {cid}", "direction": "none", "target_metric": "", "n": len(ids),
                              "status": "estimated", "declarations": decls_json, "synthetic": synthetic} for cid, ids in members.items()])
    mem = pd.DataFrame([{"cluster_id": cid, "episode_id": e, "content_hash": keys.loc[e], "rank": i + 1, "primary_metric": "harm_f30", "harm": 0.0}
                        for cid, ids in members.items() for i, e in enumerate(ids)])
    d = SCRATCH / "fake_clusters" / name
    d.mkdir(parents=True, exist_ok=True)
    clusters.to_parquet(d / "clusters.parquet", index=False)
    mem.to_parquet(d / "members.parquet", index=False)
    (d / "patterns.json").write_text("[]")
    pd.DataFrame({"episode_id": list(keys.index)}).to_parquet(d / "descriptors.parquet", index=False)
    (d / "bins.json").write_text(json.dumps({"bins": {}, "dropped": {}}))
    with stage_run("cluster", name, lr, allow_dirty=True, extra_tags={"vlapoc.synthetic": "true"} if synthetic else None) as r:
        r.input_subset(subset, "training", None)
        r.params({"synthetic": synthetic, "kind": "hand-built clusters for a verifier fixture (SYNTHETIC)"})
        for k in ("clusters", "members", "patterns", "descriptors", "bins"):
            r.output_file(k, d / (f"{k}.json" if k in ("patterns", "bins") else f"{k}.parquet"), kind="table")
        return r.run_id


def planted_mode() -> int:
    """S3, AC3.1-AC3.4."""
    import subprocess

    from scipy.stats import fisher_exact

    from vlapoc import clusters as cl
    from vlapoc import recovery as rc
    from vlapoc import results as res
    from vlapoc.groundtruth import load_answer_key
    from vlapoc.lineage import Subset
    from vlapoc.method_data import _reset_session, open_for_method
    from vlapoc.runs import client, scratch_lineage_copy, stage_run, trace

    C = Checker("planted")
    shutil.rmtree(SCRATCH / "planted", ignore_errors=True)
    base = SCRATCH / "planted"
    base.mkdir(parents=True)
    made = [REPO_ROOT / "data" / "attribution" / "score-clusters-dash"]
    try:
        lr = scratch_lineage_copy(base / "lineage")
        v3 = Subset.load("v3/candidate", lr)
        labels = load_answer_key("v3", lr)
        typ = dict(zip(labels["episode_id"], labels["corruption_type"]))
        eps = v3.episodes(lr)
        N = len(eps)
        n_total = len(client(lr).search_runs([client(lr).get_experiment_by_name("vlapoc-lineage").experiment_id], max_results=10000))
        decl_tab = synthetic_table(eps, ["f30", "hrt60_lin"], 0, {})
        from vlapoc import attribution as at

        decls_json = json.dumps({d["column"]: {k: d[k] for k in ("functional", "tau", "rollout_subset", "good_sign")}
                                 for d in at.column_declarations(decl_tab, synthetic_ret(), True, "pm1", ("square", "square"))})
        # --- AC3.2: a hand-built cluster = all 13 idle_padding + 13 clean, over the real v3 answer key ---
        idle = sorted(e for e, t in typ.items() if t == "idle_padding")
        clean = sorted(e for e in eps["episode_id"] if e not in typ)[:13]
        cid = fake_cluster_run(lr, v3, {"hand:idle13+clean13": idle + clean, "hand:clean13": clean}, "clusters-hand-built", decls_json)
        out = rc.cluster_recovery_run(cid, "v3", "score-clusters-hand", lineage_root=lr, allow_dirty=True, out_dir=base / "out_hand")
        t = pd.read_parquet(base / "out_hand" / "cluster_recovery.parquet")
        r0 = t[(t["cluster_id"] == "hand:idle13+clean13") & (t["planted_type"] == "idle_padding")].iloc[0]
        base_rate = 13 / N
        want = {"precision": 13 / 26, "recall": 1.0, "base_rate": base_rate, "lift": (13 / 26) / base_rate}
        p_want = fisher_exact([[13, 13], [0, N - 26]], alternative="greater")[1]
        indep = [abs(r0[k] - v) < 1e-12 for k, v in want.items()] + [abs(r0["p_fisher"] - p_want) < 1e-12]
        rp = t[(t["cluster_id"] == "hand:idle13+clean13") & (t["planted_type"] == "any_planted")].iloc[0]
        r_clean = t[(t["cluster_id"] == "hand:clean13") & (t["planted_type"] == "idle_padding")].iloc[0]
        best = t[(t["cluster_id"] == "hand:idle13+clean13") & t["is_best_type_for_cluster"]]
        bestc = t[(t["planted_type"] == "idle_padding") & t["is_best_cluster_for_type"]]
        C.check("3.2 hand-built SYNTHETIC cluster (13 idle_padding + 13 clean) over the real v3 key: precision 0.5, recall 1.0, lift = 0.5 / (13/960), Fisher p, recomputed independently",
                all(indep) and best["planted_type"].tolist() == ["idle_padding"] and bestc["cluster_id"].tolist() == ["hand:idle13+clean13"] and r_clean["n_overlap"] == 0 and r_clean["p_fisher"] == 1.0
                and abs(rp["precision"] - 0.5) < 1e-12 and abs(rp["recall"] - 13 / 91) < 1e-12,
                f"precision {r0['precision']}, recall {r0['recall']}, lift {r0['lift']:.3f} (= {want['lift']:.3f}), p_fisher {r0['p_fisher']:.3e} (= {p_want:.3e}); any_planted recall {rp['recall']:.3f} (=13/91); clean-only cluster: overlap {r_clean['n_overlap']}, p {r_clean['p_fisher']}")
        # --- AC3.4 ---
        C.check("3.4 output rows carry synthetic and the clusters' declarations", t["synthetic"].all() and t["declarations"].map(lambda x: bool(json.loads(x))).all(),
                f"{len(t)} rows, synthetic all {bool(t['synthetic'].all())}, declared columns {sorted(json.loads(t['declarations'].iloc[0]))}")
        # --- AC3.3 ---
        crun, srun = client(lr).get_run(cid), client(lr).get_run(out["run_id"])
        trs = trace(out["run_id"], lr)
        C.check("3.3a the scoring run was created after the cluster run and consumes its members sha256 and clusters sha256; trace(scoring run) reaches the cluster run",
                srun.info.start_time > crun.info.start_time and srun.data.tags["input.members"] == f"members:{crun.data.tags['output.members.sha256']}"
                and srun.data.tags["input.clusters"] == f"clusters:{crun.data.tags['output.clusters.sha256']}" and cid in trs["runs"],
                f"cluster run start {crun.info.start_time}, scoring run start {srun.info.start_time}; trace reaches {sorted(r['name'] for r in trs['runs'].values())[:4]}")
        tree_ = ast.parse((REPO_ROOT / "vlapoc" / "clusters.py").read_text())
        idents = {n.id for n in ast.walk(tree_) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree_) if isinstance(n, ast.Attribute)}
        scorer = {"cluster_recovery", "cluster_recovery_run", "recovery_matrix", "recovery_run", "load_answer_key", "score_recovery", "load_labels"}
        C.check("3.3b clusters.py has no code path to the scorer (no import of recovery/groundtruth/discrepancy; no code identifier of the scorer or the answer key)",
                not forbidden_imports(REPO_ROOT / "vlapoc" / "clusters.py") and not (idents & scorer),
                f"forbidden imports {forbidden_imports(REPO_ROOT / 'vlapoc' / 'clusters.py')}; scorer identifiers in code: {sorted(idents & scorer)}")
        # --- AC3.1: refusals, nothing written ---
        def n_score() -> int:
            return len(res.find_runs("score", lr))

        refusals = []

        def refuse(label, fn, contains):
            before = n_score()
            try:
                fn()
                refusals.append((label, False, "did not raise"))
            except Exception as e:  # noqa: BLE001
                refusals.append((label, contains in str(e) and n_score() == before, f"{type(e).__name__}: {str(e)[:110]}"))

        att = client(lr).search_runs([client(lr).get_experiment_by_name("vlapoc-lineage").experiment_id], "tags.stage = 'attribute' and attributes.status = 'FINISHED'", max_results=1)[0]
        refuse("a run that is not a cluster run (an attribute run)", lambda: rc.cluster_recovery_run(att.info.run_id, "v3", "score-bad", lineage_root=lr, allow_dirty=True, out_dir=base / "bad"), "not a FINISHED cluster run")
        try:
            with stage_run("cluster", "clusters-failed", lr, allow_dirty=True) as r:
                failed_id = r.run_id
                raise RuntimeError("fixture: this cluster run fails")
        except RuntimeError:
            pass
        refuse("a FAILED cluster run", lambda: rc.cluster_recovery_run(failed_id, "v3", "score-bad", lineage_root=lr, allow_dirty=True, out_dir=base / "bad"), "FINISHED")
        # a cluster run on v0 (no answer key)
        v0c = Subset.load("v0/candidate", lr)
        t0_ = synthetic_table(v0c.episodes(lr), ["f30"], 0, {})
        sid0 = synthetic_score_run(lr, v0c, t0_, "attribute-synth-v0")
        _reset_session()
        o0 = cl.clusters_run(sid0, "clusters-synth-v0", metrics=["f30"], out_dir=base / "o0", lineage_root=lr, allow_dirty=True, synthetic=True, heuristics=False, cache_dir=None)
        refuse("a cluster run whose version has no answer key (v0)", lambda: rc.cluster_recovery_run(o0["run_id"], "v0", "score-bad", lineage_root=lr, allow_dirty=True, out_dir=base / "bad"), "")
        # clusters found on v0 scored against v3: members are not v3 episode versions (hashes differ for the edited ones only; ids equal) -> check hashes
        # a ground-truth read upstream: the attribute run consumed a table produced by a score run
        with stage_run("score", "fake-score-gt", lr, allow_dirty=True) as r:
            r.produced("score_table", "ab" * 32, "gt_table")
        t1_ = synthetic_table(eps, ["f30"], 0, {})
        sid1 = synthetic_score_run(lr, v3, t1_, "attribute-synth-gtread", extra_inputs={"gt": ("score_table", "ab" * 32)})
        _reset_session()
        o1 = cl.clusters_run(sid1, "clusters-synth-gtread", metrics=["f30"], out_dir=base / "o1", lineage_root=lr, allow_dirty=True, synthetic=True, heuristics=False, cache_dir=None)
        refuse("a cluster run with a ground-truth read upstream (a score run in its trace)", lambda: rc.cluster_recovery_run(o1["run_id"], "v3", "score-bad", lineage_root=lr, allow_dirty=True, out_dir=base / "bad"), "ground-truth read upstream")
        C.check("3.1 cluster_recovery_run refuses (nothing written): a non-cluster run, a FAILED cluster run, a version without an answer key, a cluster run with a ground-truth read upstream",
                all(r_[1] for r_ in refusals), "; ".join(f"{a}: {'REFUSED' if b else 'WRONG'} ({c})" for a, b, c in refusals))
        # --- part B: the real chain, SYNTHETIC scores -> clusters_run -> score-clusters (python API and the stage CLI) ---
        _reset_session()
        view = open_for_method("v3/candidate", lr)
        desc = cl.descriptor_table(view, True)
        codes, specs, _ = cl.bin_descriptors(desc)
        planted, truth = plant_structure(desc, codes, specs)
        tab = synthetic_table(eps, ["f30", "rmst60", "auc_expected", "hrt60_lin", "censored60"], 0, planted)
        sidb = synthetic_score_run(lr, v3, tab, "attribute-dash-synth-s3", note=f"SYNTHETIC planted structure over descriptors: {truth['slice_a']['text']}; task {truth['task']['task_name']}; {truth['slice_b']['text']}")
        ob = cl.clusters_run(sidb, "clusters-dash-synth-s3", out_dir=base / "ob", lineage_root=lr, allow_dirty=True, synthetic=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "vlapoc", "scripts"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
        if not dirty:
            p_ = subprocess.run([sys.executable, "scripts/stage.py", "--lineage-root", str(lr), "score-clusters", "--clusters", "clusters-dash-synth-s3", "--version", "v3",
                                 "--subset", "candidate", "--name", "score-clusters-dash"], cwd=REPO_ROOT, capture_output=True, text=True)
            ok_cli = p_.returncode == 0
            sout = json.loads(p_.stdout[p_.stdout.index("{"):]) if ok_cli else {}
            detail = f"stage CLI rc={p_.returncode} {sout.get('n_clusters')} clusters" if ok_cli else p_.stderr[-300:]
            out_b_dir = REPO_ROOT / "data" / "attribution" / "score-clusters-dash"
        else:
            ok_cli, detail = True, "stage CLI skipped: vlapoc/scripts tree is dirty (the CLI refuses a dirty tree); re-run after committing"
            sout = rc.cluster_recovery_run(ob["run_id"], "v3", "score-clusters-dash", lineage_root=lr, allow_dirty=True, out_dir=base / "out_b")
            out_b_dir = base / "out_b"
        tb = pd.read_parquet(out_b_dir / "cluster_recovery.parquet")
        sb = json.loads((out_b_dir / "cluster_recovery_summary.json").read_text())
        C.check("3.5 the real chain (SYNTHETIC scores -> clusters_run -> score-clusters) produces one row per (cluster, planted type) incl. any_planted, flagged synthetic", ok_cli and
                len(tb) == pd.read_parquet(base / "ob" / "clusters.parquet").shape[0] * 8 and tb["synthetic"].all() and set(tb["planted_type"]) >= {"idle_padding", "gripper_release", "any_planted"}, detail + f"; {len(tb)} rows")
        top = tb[tb["planted_type"] != "any_planted"].sort_values("lift", ascending=False).head(8)
        print("highest-lift (cluster, planted type) pairs on SYNTHETIC scores (descriptor-based clusters vs the real planted labels; NOT evidence about attribution):")
        print(top[["cluster_id", "planted_type", "n_cluster", "n_overlap", "precision", "recall", "lift", "p_fisher"]].to_string(index=False))
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        tb.drop(columns=["declarations"]).sort_values(["planted_type", "p_fisher"]).to_csv(ARTIFACTS / "s3_cluster_recovery_SYNTHETIC.csv", index=False)
        (ARTIFACTS / "s3_cluster_recovery_summary_SYNTHETIC.json").write_text(json.dumps(sb, indent=1))
        n_after = len(client(lr).search_runs([client(lr).get_experiment_by_name("vlapoc-lineage").experiment_id], max_results=10000))
        del n_total, n_after
    finally:
        shutil.rmtree(base, ignore_errors=True)
        shutil.rmtree(SCRATCH / "synthetic_runs", ignore_errors=True)
        shutil.rmtree(SCRATCH / "fake_clusters", ignore_errors=True)
        for d in made:
            shutil.rmtree(d, ignore_errors=True)
    return C.finish({"SYNTHETIC": True})


# ---------------------------------------------------------------------------------------------
# S9: where "looks bad" and "hurts" disagree (estimated)
# ---------------------------------------------------------------------------------------------

READING_RE = r"no detectable harm on (\w+) \(estimated; CI \[-?\d+\.\d+, -?\d+\.\d+\]\)"
WORD = "harm" + "less"  # the word no output may contain (the verifier itself must not trip over it in its own source)


def output_text(paths: list[Path]) -> str:
    """Every string in the given output files (JSON text, parquet string cells, column names), joined, for the wording scan."""
    chunks = []
    for p in paths:
        if p.suffix == ".json":
            chunks.append(p.read_text())
        elif p.suffix == ".parquet":
            df = pd.read_parquet(p)
            chunks.append(" ".join(map(str, df.columns)))
            for c in df.columns:
                if df[c].dtype == object:
                    chunks.extend(map(str, df[c].dropna().unique()))
    return "\n".join(chunks)


def discrepancy_mode() -> int:
    """S9, AC9.1-AC9.6."""
    import re
    import subprocess

    from vlapoc import discrepancy as dc
    from vlapoc import results as res
    from vlapoc.groundtruth import load_answer_key
    from vlapoc.lineage import LineageError, Subset
    from vlapoc.method_data import _reset_session
    from vlapoc.runs import client, scratch_lineage_copy, stage_run

    C = Checker("discrepancy")
    shutil.rmtree(SCRATCH / "discrepancy", ignore_errors=True)
    base = SCRATCH / "discrepancy"
    base.mkdir(parents=True)
    made = [REPO_ROOT / "data" / "attribution" / "discrepancy-dash-cli"]
    try:
        lr = scratch_lineage_copy(base / "lineage")
        v3 = Subset.load("v3/candidate", lr)
        labels = load_answer_key("v3", lr)
        eps = v3.episodes(lr)
        typ = dict(zip(labels["episode_id"], labels["corruption_type"]))
        metrics = ["f30", "rmst60", "auc_expected", "hrt60_lin", "censored60"]
        order = eps.sort_values("episode_index")["episode_id"].tolist()
        idle = np.array([typ.get(e) == "idle_padding" for e in order])
        clean_ids = sorted(e for e in order if e not in typ)[:10]
        clean_pl = np.array([e in set(clean_ids) for e in order])
        planted = {"harm_hrt60_lin": 2.0 * idle.astype(float), "harm_f30": 4.0 * clean_pl.astype(float)}
        tab = synthetic_table(eps, metrics, 0, planted)
        note = "SYNTHETIC: idle_padding members +2 sd on harm_hrt60_lin only; action_noise members unshifted; 10 clean episodes +4 sd on harm_f30"
        sid = synthetic_score_run(lr, v3, tab, "attribute-dash-synth-s9", note=note)
        out_dir = base / "out"
        t0 = time.time()
        out = dc.discrepancy_run(sid, "v3", "discrepancy-dash-synth", out_dir=out_dir, lineage_root=lr, allow_dirty=True, synthetic=True, cache_dir=None if False else dc.DEFAULT_CACHE if hasattr(dc, "DEFAULT_CACHE") else None)
        print(f"discrepancy stage in {time.time() - t0:.0f}s")
        ty = pd.read_parquet(out_dir / "discrepancy_types.parquet")
        ep = pd.read_parquet(out_dir / "discrepancy_episodes.parquet")
        chd = pd.read_parquet(out_dir / "clean_harmful.parquet")
        fl = pd.read_parquet(out_dir / "discrepancy_filter.parquet")
        ex = json.loads((out_dir / "examples.json").read_text())
        sm = json.loads((out_dir / "discrepancy_summary.json").read_text())
        rd = lambda t, m: ty[(ty["planted_type"] == t) & (ty["metric"] == m)].iloc[0]  # noqa: E731
        c1 = rd("idle_padding", "hrt60_lin")["reading"] == "harmful" and rd("idle_padding", "hrt60_lin")["estimated_reading"].startswith("harmful on hrt60_lin")
        c2 = rd("idle_padding", "f30")["estimated_reading"].startswith("no detectable harm on f30 (estimated")
        c3 = all(r_["estimated_reading"].startswith(f"no detectable harm on {r_['metric']} (estimated") for _, r_ in ty[ty["planted_type"] == "action_noise"].iterrows())
        top20 = chd[chd["metric"] == "f30"]["episode_id"].tolist()
        c4 = set(clean_ids) <= set(top20)
        qs = {m: set(ep[f"quadrant__{m}"]) for m in metrics}
        c5 = any(len(v) == 4 for v in qs.values())
        C.check("9.1 SYNTHETIC over the real v3 key: idle_padding 'harmful on hrt60_lin' and 'no detectable harm on f30 (estimated)'; action_noise 'no detectable harm' on all five metrics; the 10 planted clean episodes are in clean_harmful top-20 for f30; all four quadrants non-empty for a metric",
                c1 and c2 and c3 and c4 and c5,
                f"idle_padding/hrt: {rd('idle_padding', 'hrt60_lin')['estimated_reading']}; idle_padding/f30: {rd('idle_padding', 'f30')['estimated_reading']}; "
                f"action_noise: {[r_['reading'] for _, r_ in ty[ty['planted_type'] == 'action_noise'].iterrows()]}; planted clean in f30 top-20: {len(set(clean_ids) & set(top20))}/10; "
                f"quadrant counts hrt60_lin {sm['quadrant_counts']['labels']['hrt60_lin']}, f30 {sm['quadrant_counts']['labels']['f30']}")
        # --- AC9.2: wording guard ---
        outs = sorted(out_dir.glob("*"))
        text = output_text(outs)
        sents = [x for x in ty["estimated_reading"] if x.startswith("no detectable")] + [x for x in fl["estimated_reading"] if x.startswith("no detectable")]
        C.check("9.2 the word 'harm'+'less' appears in no output file; every non-excluding CI reads 'no detectable harm on <metric> (estimated; CI [lo, hi])'",
                WORD not in text.lower() and all(re.fullmatch(READING_RE, x) for x in sents) and len(sents) > 0 and WORD not in (REPO_ROOT / "vlapoc" / "discrepancy.py").read_text().lower(),
                f"{len(outs)} output files scanned, {len(sents)} 'no detectable' sentences all well-formed, e.g. {sents[0]!r}")
        # --- AC9.3: the visibly-bad evidence, read from the recorded score runs, equals the ground-truth feature's published bands ---
        ev = dc.heuristic_evidence("v3", "candidate", lr)
        best = dc.best_heuristics(ev)
        bands = pd.read_csv(REPO_ROOT / "tracker" / "features" / "2026-10-15-ground-truth" / "artifacts" / "s5_v3_r3_bands.csv").set_index("type")
        bb = best.set_index("planted_type")
        same = all(abs(bb.loc[t, "auroc"] - bands.loc[t, "best_auroc_as_scored"]) < 1e-3 for t in bands.index if t in bb.index) and set(bands.index) <= set(bb.index)
        lab = labels.set_index("episode_id")
        pairs_ok = all(m_["most_harmful"]["original_content_hash"] == lab.loc[m_["most_harmful"]["episode_id"], "parent_content_hash"]
                       and m_["least_harmful"]["original_content_hash"] == lab.loc[m_["least_harmful"]["episode_id"], "parent_content_hash"]
                       and m_["most_harmful"]["content_hash"] != m_["most_harmful"]["original_content_hash"]
                       for e in ex["types"].values() for m_ in e["by_metric"].values())
        ev_ok = all(e["visibly_bad_evidence"] and abs(e["visibly_bad_evidence"]["auroc"] - bands.loc[t, "best_auroc_as_scored"]) < 1e-3 for t, e in ex["types"].items())
        C.check("9.3 per type: best-heuristic AUROC read from the recorded ground-truth `score` runs (not re-typed) equals the published bands (s5_v3_r3_bands.csv, tol 1e-3); examples.json pairs every example with its clean original (parent_content_hash, differs from the corrupted hash)",
                same and pairs_ok and ev_ok and len(ev["run_id"].unique()) == 9,
                f"{ev['run_id'].nunique()} recorded score runs; best per type: " + ", ".join(f"{t}={bb.loc[t, 'heuristic']}({bb.loc[t, 'auroc']:.3f})" for t in bb.index))
        # --- AC9.4: blindness / ordering ---
        run = client(lr).get_run(out["run_id"])
        srun = client(lr).get_run(sid)
        score_sha = srun.data.tags["output.attribution_scores.sha256"]
        refusals = []

        def n_score() -> int:
            return len(res.find_runs("score", lr))

        def refuse(label, fn, exc, contains):
            before = n_score()
            try:
                fn()
                refusals.append((label, False, "did not raise"))
            except exc as e:
                refusals.append((label, contains in str(e) and n_score() == before, f"{type(e).__name__}: {str(e)[:100]}"))
            except Exception as e:  # noqa: BLE001
                refusals.append((label, False, f"wrong error {type(e).__name__}: {str(e)[:100]}"))

        try:
            with stage_run("attribute", "attribute-failed", lr, allow_dirty=True) as r:
                failed_id = r.run_id
                raise RuntimeError("fixture: failed attribute run")
        except RuntimeError:
            pass
        att_f = client(lr).get_run(sid)
        refuse("a FAILED attribute run", lambda: dc.discrepancy_run(failed_id, "v3", "d-bad", out_dir=base / "bad", lineage_root=lr, allow_dirty=True, heuristics=False), ValueError, "FINISHED attribute run")
        with stage_run("cluster", "fake-cluster-for-refusal", lr, allow_dirty=True) as r:
            cl_id = r.run_id
        refuse("a non-attribute run (a cluster run)", lambda: dc.discrepancy_run(cl_id, "v3", "d-bad", out_dir=base / "bad", lineage_root=lr, allow_dirty=True, heuristics=False), ValueError, "FINISHED attribute run")
        v1c = Subset.load("v0/candidate", lr)
        t_v0 = synthetic_table(v1c.episodes(lr), ["f30"], 0, {})
        sid_v0 = synthetic_score_run(lr, v1c, t_v0, "attribute-synth-v0-for-s9")
        refuse("a version without an answer key (v0)", lambda: dc.discrepancy_run(sid_v0, "v0", "d-bad", out_dir=base / "bad", lineage_root=lr, allow_dirty=True, heuristics=False), LineageError, "no labels")
        small = Subset(v3.dataset_version_id, tuple(v3.episode_ids[:50]), {"op": "synthetic_small", "parent_subset_id": v3.subset_id})
        small.save(lr)
        t_small = synthetic_table(small.episodes(lr), metrics, 0, {})
        sid_small = synthetic_score_run(lr, small, t_small, "attribute-synth-small-for-s9")
        refuse("a score table that does not key exactly v3/candidate", lambda: dc.discrepancy_run(sid_small, "v3", "d-bad", out_dir=base / "bad", lineage_root=lr, allow_dirty=True, heuristics=False), LineageError, "keys")
        del att_f
        method_files = [REPO_ROOT / "vlapoc" / f for f in ("attribution.py", "clusters.py", "results.py", "analysis.py")]
        bad_imp = {f.name: forbidden_imports(f) for f in method_files if forbidden_imports(f)}
        C.check("9.4 refuses (nothing written): a FAILED attribute run, a non-attribute run, a version without an answer key, a table that does not key v3/candidate; the run consumes the score table's sha256 and was created after it; attribution/clusters/results/analysis import nothing from discrepancy, groundtruth or recovery",
                all(r_[1] for r_ in refusals) and run.data.tags["input.score_table" if "input.score_table" in run.data.tags else "input.scores"] == f"score_table:{score_sha}"
                and run.info.start_time >= srun.info.end_time and not bad_imp,
                "; ".join(f"{a}: {'REFUSED' if b else 'WRONG'} ({c})" for a, b, c in refusals) + f"; importers of the GT side among method files: {bad_imp}")
        # --- AC9.5: declarations and synthetic on every row ---
        def declared(df):
            return all(c in df.columns for c in ("functional", "tau", "rollout_subset", "good_sign", "harm_column", "synthetic")) and df["synthetic"].all() and df["functional"].notna().all()

        ep_ok = ep["synthetic"].all() and ep["declarations"].map(lambda x: set(json.loads(x)) == {f"harm_{m}" for m in metrics} and all("functional" in d for d in json.loads(x).values())).all()
        ex_ok = ex["synthetic"] and all("functional" in m_["declaration"] and "tau" in m_["declaration"] for e in ex["types"].values() for m_ in e["by_metric"].values())
        C.check("9.5 every row of discrepancy_types, discrepancy_filter and clean_harmful carries functional/tau/rollout_subset/good_sign of its harm column and synthetic; episodes rows carry the JSON declarations; examples.json entries carry the declaration",
                declared(ty) and declared(fl) and declared(chd) and ep_ok and ex_ok,
                f"types {len(ty)} rows, filter {len(fl)}, clean_harmful {len(chd)}, episodes {len(ep)}; e.g. {ty['functional'].iloc[0]!r} tau {ty['tau'].iloc[0]!r}")
        doc = (REPO_ROOT / "vlapoc" / "discrepancy.py").read_text().split('"""')[1]
        C.check("9.6 grounding docstring cites §9.2 p.19, §9.1 Step 1 p.18, §4.5 p.11 and CUPID Fig. 4 p.8", all(x in doc for x in ("§9.2 (p. 19)", "§9.1 Step 1 (p. 18)", "§4.5 (p. 11)", "Fig. 4 caption, p. 8")))
        # --- the secondary layer, in numbers ---
        anyf = fl[fl["filter"] == "any_H1_H8"].iloc[0]
        print(f"secondary layer (SYNTHETIC scores): any-of-H1-H8 filter flags {anyf['n_flagged']} of {sm['n']} episodes ({100 * anyf['share_flagged']:.0f}%), of which {anyf['n_flagged_planted']} planted "
              f"(precision {anyf['precision_vs_planted']:.2f}, recall of the 91 planted {anyf['recall_of_planted']:.2f}); per-heuristic: "
              + ", ".join(f"{r_['filter']} {r_['n_flagged']}/{r_['n_flagged_planted']}" for _, r_ in fl[fl['metric'] == 'f30'].iterrows() if r_['filter'] != "any_H1_H8"))
        C.check("9.7 the filter layer exists: 10 filters (any + 9 heuristics) x 5 metrics, quadrant counts per filter sum to N", len(fl) == 10 * 5 and all(
            (r_[[f'n_{q}' for q in dc.QUADRANTS]].sum() == sm["n"]) for _, r_ in fl.iterrows()), f"{len(fl)} rows; any-filter flagged {sm['n_flagged_filter']} (q = {sm['filter_q']:.4f})")
        # --- stage CLI (needs a clean tree) ---
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "vlapoc", "scripts"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
        if not dirty:
            _reset_session()
            p_ = subprocess.run([sys.executable, "scripts/stage.py", "--lineage-root", str(lr), "discrepancy", "--scores", "attribute-dash-synth-s9", "--version", "v3",
                                 "--subset", "candidate", "--name", "discrepancy-dash-cli"], cwd=REPO_ROOT, capture_output=True, text=True)
            ok_cli = p_.returncode == 0 and (REPO_ROOT / "data" / "attribution" / "discrepancy-dash-cli" / "discrepancy_types.parquet").is_file()
            C.check("9.8 the stage CLI (scripts/stage.py discrepancy) runs on the scratch store", ok_cli, f"rc={p_.returncode}" + ("" if ok_cli else p_.stderr[-300:]))
        else:
            print("stage CLI check skipped: vlapoc/scripts tree is dirty (the CLI refuses a dirty tree); re-run after committing")
        # --- measured: false-alarm rate of the readings on null cells (information) ---
        cells, hits, hits_w = 0, 0, 0
        for seed in range(20):
            t2 = synthetic_table(eps, metrics, 1000 + seed, {})
            e2 = dc.episode_table(t2, labels, [f"harm_{m}" for m in metrics], pd.DataFrame({"none": np.zeros(len(t2), bool)}), res.functionals(sid, lr), True)
            ty2 = dc.type_table(e2, labels, [f"harm_{m}" for m in metrics], res.functionals(sid, lr), pd.DataFrame(), True, n_boot=1000)
            cells += len(ty2)
            hits += int((ty2["reading"] != "no_detectable_harm").sum())
            hits_w += int((ty2["reading_welch"] != "no_detectable_harm").sum())
        print(f"null calibration (SYNTHETIC, 20 seeds, no planted structure, 13 vs 869): {hits}/{cells} cells read harmful or helpful by the percentile bootstrap, "
              f"{hits_w}/{cells} by the Welch cross-check (nominal two-sided 5%: {0.05 * cells:.0f})")
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "s9_null_calibration_SYNTHETIC.json").write_text(json.dumps({"SYNTHETIC": True, "seeds": 20, "cells": cells, "bootstrap_cells_reading_harmful_or_helpful": hits,
                                                                                  "bootstrap_rate": hits / cells, "welch_cells_reading_harmful_or_helpful": hits_w, "welch_rate": hits_w / cells,
                                                                                  "nominal_two_sided": 0.05, "note": "n_boot=1000; 13 episodes per type vs 869 clean; cells within a seed share the clean group, so they are not independent",
                                                                                  "single_cell_simulation": "13 vs 869 normal null, 1500 trials: percentile bootstrap 0.0853, Welch 0.0487 (data/scratch/dash_exec/draft/calib.py, artifacts/probes/calib.py)"}, indent=1))
        ty.drop(columns=["declarations"] if "declarations" in ty else []).to_csv(ARTIFACTS / "s9_discrepancy_types_SYNTHETIC.csv", index=False)
        fl.to_csv(ARTIFACTS / "s9_discrepancy_filter_SYNTHETIC.csv", index=False)
        (ARTIFACTS / "s9_discrepancy_summary_SYNTHETIC.json").write_text(json.dumps({"planted_structure": note, **sm}, indent=1))
        (ARTIFACTS / "s9_examples_SYNTHETIC.json").write_text(json.dumps(ex, indent=1))
        chd.to_csv(ARTIFACTS / "s9_clean_harmful_SYNTHETIC.csv", index=False)
    finally:
        shutil.rmtree(base, ignore_errors=True)
        shutil.rmtree(SCRATCH / "synthetic_runs", ignore_errors=True)
        for d in made:
            shutil.rmtree(d, ignore_errors=True)
    return C.finish({"SYNTHETIC": True})


MODES = {"results": results, "clusters": clusters_mode, "planted": planted_mode, "discrepancy": discrepancy_mode}


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
