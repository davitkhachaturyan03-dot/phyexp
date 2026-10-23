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


def synthetic_score_run(lr: Path, subset, table: pd.DataFrame, name: str, chain: str | None = None, note: str = "") -> str:
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


MODES = {"results": results, "clusters": clusters_mode}


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
