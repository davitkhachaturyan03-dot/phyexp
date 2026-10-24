"""Verify the dashboard feature (``tracker/features/2026-10-22-dashboard/``), story by story. One mode per story group.

Grounding: PoC §2.7 p.5 (demonstration surface), §1 p.3 (aggregation by metadata), §3.4 p.7 (declared functional), §4.2 pp.9-10 (four
groups), §4.4 pp.10-11 (two-step verdict), §4.5 p.11, §9.1 Steps 1 and 4 pp.18-19, §9.2 p.19 (where "looks bad" and "hurts"
disagree), §2.1 item 4 p.3 (blindness); CUPID p.11 (group effects are only approximated by summed individual influence).

Run from the repo root; each mode exits 0 when every check passes::

    .venv/bin/python scripts/verify_dashboard.py --results       # S1: influence matrix recorded, results loader, provenance label
    .venv/bin/python scripts/verify_dashboard.py --clusters      # S2: the clusters stage (three lenses), sliceline
    .venv/bin/python scripts/verify_dashboard.py --planted       # S3: discovered clusters scored against the planted types
    .venv/bin/python scripts/verify_dashboard.py --discrepancy   # S9: planted/clean x harmful/not, estimated (PoC §9.2)
    .venv/bin/python scripts/verify_dashboard.py --round-b       # S4: drop_cluster arms, the Round B chain, the confirm stage
    .venv/bin/python scripts/verify_dashboard.py --planted-arms  # S10: planted-type arms, ground-truth path, bounded verdict wording
    .venv/bin/python scripts/verify_dashboard.py --app           # S5: headless AppTest of every page on the demo store, video helper, server health
    .venv/bin/python scripts/verify_dashboard.py --render        # S5 (round 2): what a real browser draws: plot areas, row click, navigation, truncation
    .venv/bin/python scripts/verify_dashboard.py --demo          # S6: the demo store (data/lineage_demo, built by scripts/dashboard_demo.py)
    .venv/bin/python scripts/verify_dashboard.py --dropin        # S7: a real-code chain and the real smoke run render with no file changes

Every MLflow-writing check runs on a scratch lineage store under ``data/scratch/dashboard/`` (``runs.scratch_lineage_copy``); the real
``data/lineage/mlflow.db`` sha256 is recorded before and after and must not change. Scratch runs use ``allow_dirty=True`` because
this script is under ``scripts/``. Every score table built here is SYNTHETIC (noise plus planted structure) and is labelled as such
in the artifacts it writes; none of it is evidence about any policy.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import re
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
        codes, specs, dropped = cl.bin_descriptors(desc, 5, 19)  # the stage's own binning: min_bin = min_sup = 19
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
        prm_dedupe = client(lr).get_run(out["run_id"]).data.params["n_duplicate_member_sets_merged"]
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
        # AC2.9: no two clusters share a member set; AC2.8: Round A passes min-sup-fraction 0.01 and the run records min_sup
        sets = mem.groupby("cluster_id")["episode_id"].apply(frozenset)
        dup = int(sets.duplicated().sum())
        eq_rows = clu[clu["n_equivalent"] > 0]
        eq_example = (eq_rows.iloc[0]["definition"], json.loads(eq_rows.iloc[0]["equivalent_definitions"])[0]["definition"]) if len(eq_rows) else None
        C.check("2.9 clusters with identical member sets are reported once: no two rows of clusters.parquet share a member set; the kept row lists the equivalent definitions",
                dup == 0 and (len(eq_rows) == 0 or all(json.loads(x) for x in eq_rows["equivalent_definitions"])) and out["n_clusters"] == len(clu),
                f"{len(clu)} clusters, duplicate member sets {dup}; {len(eq_rows)} rows list equivalents, e.g. {eq_example}; {int(prm_dedupe)} merged")
        j_a, _ = dry_jobs("--subset", "v3/candidate", "--curation", "per_task", "--metrics", "f30", "hrt60_lin")
        cj = next(j for j in j_a if j["name"] == "clusters-t")["cmd"]
        out8 = cl.clusters_run(sid, "clusters-dash-synth-ms10", out_dir=base / "out_ms10", lineage_root=lr, allow_dirty=True, synthetic=True, params={"min_sup_fraction": 0.01})
        ms = json.loads(client(lr).get_run(out8["run_id"]).data.params["slice_finder"])["min_sup"]
        C.check("2.8 Round A on v3 passes --min-sup-fraction 0.01 to the clusters job (dry-run argv), and the cluster run's params record min_sup = 10, below the 13 members of a planted type",
                "--min-sup-fraction 0.01" in cj and ms == 10, f"clusters job argv: {cj[:140]}...; recorded min_sup {ms}")
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
            sub_ = clu[(clu["lens"] == "label") & (clu["label_column"] == lc)]
            # identical member sets are reported once (AC2.9): a label group merged into another cluster counts through that cluster's equivalent definitions
            ok_n[lc] = int(sub_["n"].sum()) + sum(int(r["n"]) for _, r in clu.iterrows() for e in json.loads(r["equivalent_definitions"]) if e["lens"] == "label" and e["definition"].startswith(f"{lc} = "))
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
        C.check("2.6 label-group n sum to N for task_name, scene and suite (counting groups merged into an equivalent cluster, AC2.9); every slice's members equal its definition re-evaluated from the raw descriptors and recorded edges; group members belong to their group",
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
        codes, specs, _ = cl.bin_descriptors(desc, 5, 19)
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

READING_RE = r"no detectable harm on (\w+) \(estimated; Welch 95% CI \[-?\d+\.\d+, -?\d+\.\d+\]\)\.( Uncorrected across \d+ types × \d+ metrics\.)? Holm across the \d+ metrics: adjusted p = \d\.\d+\."
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
        C.check("9.2 the word 'harm'+'less' appears in no output file; every non-excluding CI reads 'no detectable harm on <metric> (estimated; Welch 95% CI [lo, hi]). Uncorrected across 7 types × 5 metrics. Holm across the 5 metrics: ...'",
                WORD not in text.lower() and all(re.fullmatch(READING_RE, x) for x in sents) and all("Uncorrected across 7 types × 5 metrics." in x for x in ty["estimated_reading"]) and len(sents) > 0 and WORD not in (REPO_ROOT / "vlapoc" / "discrepancy.py").read_text().lower(),
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
        C.check("9.7b the filter layer exists: 10 filters (any + 9 heuristics) x 5 metrics, quadrant counts per filter sum to N", len(fl) == 10 * 5 and all(
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
        # --- AC9.7: the reading's interval is scipy's Welch interval, and the null calibration is re-run ---
        from scipy import stats as sps

        cols5 = [f"harm_{m}" for m in metrics]
        dev = 0.0
        for _, r_ in ty.iterrows():
            zt_ = ep.loc[ep["planted_type"] == r_["planted_type"], f"z__{r_['metric']}"].to_numpy()
            zc_ = ep.loc[ep["planted_type"] == "clean", f"z__{r_['metric']}"].to_numpy()
            ci_ = sps.ttest_ind(zt_, zc_, equal_var=False).confidence_interval(0.95)
            dev = max(dev, abs(ci_.low - r_["diff_lo"]), abs(ci_.high - r_["diff_hi"]))
        cells, hits, hits_b, holds = 0, 0, 0, 0
        for seed in range(20):
            t2 = synthetic_table(eps, metrics, 1000 + seed, {})
            e2 = dc.episode_table(t2, labels, cols5, pd.DataFrame({"none": np.zeros(len(t2), bool)}), res.functionals(sid, lr), True)
            ty2 = dc.type_table(e2, labels, cols5, res.functionals(sid, lr), pd.DataFrame(), True, n_boot=1000)
            cells += len(ty2)
            hits += int((ty2["reading"] != "no_detectable_harm").sum())
            hits_b += int((ty2["reading_boot"] != "no_detectable_harm").sum())
            holds += int(((ty2["reading"] != "no_detectable_harm") & (ty2["p_holm_metrics"] < 0.05)).sum())
        rate = hits / cells
        print(f"null calibration (SYNTHETIC, 20 seeds x 7 types x 5 metrics, no planted effect): Welch reading false-call rate {rate:.4f} ({hits}/{cells}); percentile bootstrap {hits_b / cells:.4f}; Holm-across-metrics {holds / cells:.4f}")
        C.check("9.7 the reading's difference CI equals scipy.stats.ttest_ind(..., equal_var=False).confidence_interval(0.95) to 1e-12 (no hand-rolled Welch); the stored null calibration (20 seeds x 7 types x 5 metrics, no planted effect) gives a reading false-call rate <= 0.06; written back with the Holm rate beside it",
                dev < 1e-12 and rate <= 0.06 and "p_holm_metrics" in ty and not hasattr(dc, "welch_diff_ci"), f"max |difference| to scipy {dev:.1e}; Welch false-call rate {rate:.4f}, bootstrap {hits_b / cells:.4f}, Holm {holds / cells:.4f}")
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "s9_null_calibration_SYNTHETIC.json").write_text(json.dumps({"SYNTHETIC": True, "seeds": 20, "cells": cells, "setup": "20 seeds x 7 types x 5 metrics, no planted effect, 13 vs 869", "bootstrap_cells_reading_harmful_or_helpful": hits_b,
                                                                                  "bootstrap_rate": hits_b / cells, "welch_cells_reading_harmful_or_helpful": hits, "welch_rate": rate, "holm_across_metrics_rate": holds / cells,
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


# ---------------------------------------------------------------------------------------------
# S4 / S10: Round B
# ---------------------------------------------------------------------------------------------


def dry_jobs(*args: str):
    """``server_pipeline.py --tag t --dry-run <args>`` parsed: ``(jobs, summary line)`` (the training-attribution verifier's parser)."""
    return vt().dry_run(*args)


def argv_minus(cmd: str, drop: tuple[str, ...] = ("--subset", "--name", "--train", "--chain")) -> list[str]:
    """A job's command split into tokens with the values of ``drop`` options removed (so two arms' argv can be compared apart from data and names)."""
    import shlex

    toks, out, skip = shlex.split(cmd), [], False
    for tk in toks:
        if skip:
            skip = False
            continue
        if tk in drop:
            skip = True
            continue
        out.append(tk)
    return out


def n_runs_in(lr: Path) -> int:
    from vlapoc.runs import client, experiment_id

    return len(client(lr).search_runs([experiment_id(lr)], max_results=20000))


def verdict_json(call: str | None, gate: bool = True, keys=("f30", "auc_h", "rmst60", "hrt60", "censored60"), only: dict | None = None,
                 mag: float = 0.1) -> dict:
    """A SYNTHETIC ``verdict.json`` with the fields ``step4`` / ``planted_arms`` read. ``call``: ``"T"`` (treated better, significant), ``"C"`` (control
    better), ``None`` (not significant), per metric overridable with ``only``. ``diff`` and ``ci95`` are A minus B (higher is better)."""
    def row(c):
        if c is None:
            return {"p_holm": 0.4, "diff_better": 0.01, "diff": 0.01, "ci95": [-0.1, 0.12], "higher_is_better": True}
        sgn = -1.0 if c == "T" else 1.0  # diff_better > 0: A (the control) better
        return {"p_holm": 0.004, "diff_better": sgn * mag, "diff": sgn * mag, "ci95": [sgn * mag - 0.05, sgn * mag + 0.05], "higher_is_better": True}
    return {"SYNTHETIC": True, "family": "W", "alpha": 0.05, "gate": {"fires": gate, "p": 0.01 if gate else 0.6},
            "metrics": {k: row((only or {}).get(k, call)) for k in keys}, "outcome": "i" if gate else "iii"}


def add_verify(lr: Path, base: Path, name: str, kind: str, treated: str, control: str, seed: int, verdict: dict, chain: str, family: str = "W") -> str:
    """A FINISHED ``verify`` run in the scratch store carrying a SYNTHETIC verdict and the ``step4.*`` tags."""
    from vlapoc.runs import stage_run

    path = base / "verdicts" / f"{name}.json"
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps({**verdict, "family": family}))
    with stage_run("verify", name, lr, chain=chain, allow_dirty=True, extra_tags={"step4.kind": kind, "step4.treated": treated, "step4.control": control, "step4.seed": seed}) as r:
        r.output_file("verdict", path, kind="verdict")
        return r.run_id


def round_b_mode() -> int:
    """S4, AC4.1-AC4.5."""
    from vlapoc import clusters as cl
    from vlapoc import confirm as cf
    from vlapoc import results as res
    from vlapoc.lineage import Subset, drop_random_k
    from vlapoc.method_data import _reset_session
    from vlapoc.runs import client, scratch_lineage_copy, select_run, stage_run, trace

    C = Checker("round_b")
    shutil.rmtree(SCRATCH / "round_b", ignore_errors=True)
    base = SCRATCH / "round_b"
    base.mkdir(parents=True)
    counts = {}
    try:
        # --- AC4.1: dry-run counts, parsed ---
        picks = ["slice:hrt60_lin:harm:1", "label:task_name=KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"]
        common = ["--subset", "v3/candidate", "--round-b", "--clusters", "clusters-t", "--pick", *picks]
        j2, s2 = dry_jobs(*common)
        j2m, s2m = dry_jobs(*common, "--matched-random")
        t2, t2m = [j for j in j2 if j["name"].startswith("train-")], [j for j in j2m if j["name"].startswith("train-")]
        by = {j["name"]: j for j in j2m}
        sm = by["step4-summary-t-roundb"]
        cj = by["confirm-t"]
        sel = by["select-t-roundb"]["cmd"]
        verifies = [j["name"] for j in j2m if j["name"].startswith("verify-")]
        counts = {"round_b_2_clusters": {"jobs": len(j2), "trainings": len(t2)}, "round_b_2_clusters_matched_random": {"jobs": len(j2m), "trainings": len(t2m)}}
        C.check("4.1a Round B for 2 picked clusters: 2 x (3 train + 3 eval + 3 verify) + 1 select + 1 step4-summary + 1 confirm = 21 jobs, 6 trainings (plan arithmetic)",
                len(j2) == 21 and len(t2) == 6, f"{len(j2)} jobs, {len(t2)} trainings")
        C.check("4.1b ... and with --matched-random the plan states 33 jobs / 12 trainings; the chain's own per-arm pattern gives the count below (each matched arm also gets a vs_base verify, and each cluster a curated_vs_random verify)",
                len(t2m) == 12, f"MEASURED {len(j2m)} jobs (plan arithmetic 33), {len(t2m)} trainings; difference: 2 matched arms x 3 vs_base verifies + 2 clusters x 3 curated_vs_random verifies = +12")
        C.check("4.1c job structure: one select job with 2 drop_cluster and 2 drop_random_matched selects; step4-summary depends on every verify; confirm depends on step4-summary",
                sel.count("--op drop_cluster") == 2 and sel.count("--op drop_random_matched") == 2 and sorted(sm["dep_names"]) == sorted(verifies) and cj["dep_names"] == ["step4-summary-t-roundb"]
                and all("--chain-id" in x["cmd"] for x in (sm, cj)), f"{len(verifies)} verify jobs; confirm deps {cj['dep_names']}")
        # --- AC4.5: argv of every arm equals the base recipe except data and names ---
        jb, _ = dry_jobs("--subset", "v3/candidate", "--curation", "per_task", "--metrics", "f30", "hrt60_lin")
        bad = []
        for s_ in (0, 1, 2):
            b_tr = argv_minus(next(j for j in jb if j["name"] == f"train-t-s{s_}")["cmd"])
            b_ev = argv_minus(next(j for j in jb if j["name"] == f"eval-t-s{s_}")["cmd"])
            for arm in ("cluster0", "crandom0", "cluster1", "crandom1"):
                a_tr = argv_minus(next(j for j in j2m if j["name"] == f"train-t-{arm}-s{s_}")["cmd"])
                a_ev = [x for x in argv_minus(next(j for j in j2m if j["name"] == f"eval-t-{arm}-s{s_}")["cmd"]) if x != "--no-trajectories"]
                if a_tr != b_tr or a_ev != b_ev:
                    bad.append((arm, s_))
        C.check("4.5 every arm's train argv equals the base recipe's (same seed) except --subset and --name; eval argv equal apart from --no-trajectories (AC7.5)", not bad and len(t2m) == 12,
                f"12 train + 12 eval jobs compared, mismatches {bad}; e.g. {argv_minus(next(j for j in j2m if j['name'] == 'train-t-cluster0-s0')['cmd'])}")
        # --- AC4.2 / AC4.4: the drop_cluster op on a SYNTHETIC cluster run over the real v3 ---
        lr = scratch_lineage_copy(base / "lineage")
        v3 = Subset.load("v3/candidate", lr)
        view = __import__("vlapoc.method_data", fromlist=["open_for_method"]).open_for_method("v3/candidate", lr)
        desc = cl.descriptor_table(view, True)
        codes, specs, _ = cl.bin_descriptors(desc, 5, 19)
        planted, truth = plant_structure(desc, codes, specs)
        eps = v3.episodes(lr)
        tab = synthetic_table(eps, ["f30", "rmst60", "auc_expected", "hrt60_lin", "censored60"], 0, planted)
        sid = synthetic_score_run(lr, v3, tab, "attribute-dash-synth-s4", note="SYNTHETIC planted structure (see S2)")
        out = cl.clusters_run(sid, "clusters-dash-synth-s4", out_dir=base / "oc", lineage_root=lr, allow_dirty=True, synthetic=True)
        crun = out["run_id"]
        clu = pd.read_parquet(base / "oc" / "clusters.parquet")
        mem = pd.read_parquet(base / "oc" / "members.parquet")
        cid = "slice:hrt60_lin:harm:1"
        ids = set(mem.loc[mem["cluster_id"] == cid, "episode_id"])
        sel_id, child = select_run("v3/candidate", "select-s4-cluster0", "drop_cluster", lineage_root=lr, allow_dirty=True, cluster_run_id=crun, cluster_id=cid)
        sel2_id, rnd = select_run("v3/candidate", "select-s4-crandom0", "drop_random_matched", seed=0, lineage_root=lr, allow_dirty=True, cluster_run_id=crun, cluster_id=cid)
        task = eps.set_index("episode_id")[["suite", "libero_task_id"]].astype(str).agg("/".join, axis=1)
        pt_c, pt_r = task.loc[list(ids)].value_counts().to_dict(), task.loc[list(set(v3.episode_ids) - set(rnd.episode_ids))].value_counts().to_dict()
        trs = trace(sel_id, lr)
        p_ = client(lr).get_run(sel_id).data.params
        C.check("4.2a drop_cluster child has exactly N - n episodes, its derivation lists the cluster's members, trace(select run) reaches the cluster run and the score run",
                len(child.episode_ids) == len(v3.episode_ids) - len(ids) and set(child.derivation["dropped"]) == ids and crun in trs["runs"] and sid in trs["runs"],
                f"cluster {cid}: n = {len(ids)}, child {len(child.episode_ids)} of {len(v3.episode_ids)}; trace reaches {sorted(r['name'] for r in trs['runs'].values())[:3]}")
        C.check("4.2b the matched random child has the same per-task counts as the cluster, drawn outside the cluster", pt_c == pt_r and not (set(v3.episode_ids) - set(rnd.episode_ids)) & ids,
                f"{len(pt_c)} tasks, counts {sorted(pt_c.values())}")
        exp = json.loads(p_["expected_by_metric"])
        C.check("4.2c the select run records expected_by_metric (stated before any retraining) from the estimated readings: harmful -> treated_better",
                exp.get("hrt60") == "treated_better" and all(v in ("treated_better", "treated_worse") for v in exp.values()) and "stated before" in p_["expected_by_metric_status"], f"expected_by_metric {exp}")
        # AC4.4 refusals
        refusals = []

        def refuse(label, fn, contains):
            before = n_runs_in(lr)
            try:
                fn()
                refusals.append((label, False, "did not raise"))
            except Exception as e:  # noqa: BLE001
                refusals.append((label, contains in str(e) and n_runs_in(lr) == before, f"{type(e).__name__}: {str(e)[:100]}"))

        other = drop_random_k(v3, 5, 0, lr)
        whole = f"label:task_name={truth['task']['task_name']}"
        refuse("unknown cluster_id", lambda: select_run("v3/candidate", "select-bad", "drop_cluster", lineage_root=lr, allow_dirty=True, cluster_run_id=crun, cluster_id="slice:nope:harm:9"), "unknown cluster_id")
        refuse("a cluster run found on another subset", lambda: select_run(other.subset_id, "select-bad", "drop_cluster", lineage_root=lr, allow_dirty=True, cluster_run_id=crun, cluster_id=cid), "another subset")
        refuse("a cluster that would empty a task (leave-one-task-out)", lambda: select_run("v3/candidate", "select-bad", "drop_cluster", lineage_root=lr, allow_dirty=True, cluster_run_id=crun, cluster_id=whole), "leave-one-task-out")
        refuse("a run that is not a FINISHED cluster run", lambda: select_run("v3/candidate", "select-bad", "drop_cluster", lineage_root=lr, allow_dirty=True, cluster_run_id=sid, cluster_id=cid), "not a FINISHED cluster run")
        sel3_id, whole_child = select_run("v3/candidate", "select-s4-whole", "drop_cluster", lineage_root=lr, allow_dirty=True, cluster_run_id=crun, cluster_id=whole, allow_whole_task=True)
        pw = client(lr).get_run(sel3_id).data.params
        C.check("4.4 refused with nothing written: unknown cluster_id, a cluster run of another subset, a cluster that would empty a task, a non-cluster run; with allow_whole_task the whole-task cluster is allowed and the flag is recorded as a param",
                all(r_[1] for r_ in refusals) and pw["allow_whole_task"] == "True" and len(whole_child.episode_ids) == len(v3.episode_ids) - 48,
                "; ".join(f"{a}: {'REFUSED' if b else 'WRONG'} ({c})" for a, b, c in refusals))
        # --- AC4.3: confirm on SYNTHETIC verdicts ---
        chain = "t-rb111111"
        decl = "{}"
        fcr = fake_cluster_run(lr, v3, {k: sorted(ids)[:5] for k in ("cA", "cB", "cC", "cD", "cE")}, "clusters-fake-s4", decl)
        cases = {  # arm -> (expected_by_metric, per-seed verdict calls)
            "cA": ({"hrt60": "treated_better"}, [verdict_json("T")] * 3),
            "cB": ({"hrt60": "treated_better"}, [verdict_json("T"), verdict_json(None), verdict_json(None)]),
            "cC": ({"hrt60": "treated_better"}, [verdict_json("T", gate=False), verdict_json("T", gate=False), verdict_json("T")]),
            "cD": ({"hrt60": "treated_better"}, [verdict_json("C"), verdict_json("C"), verdict_json(None)]),
            "cE": ({"hrt60": "treated_better", "f30": "treated_worse"}, [verdict_json(None, only={"hrt60": "T", "f30": "C"})] * 3),
        }
        for arm, (ebm, vds) in cases.items():
            with stage_run("select", f"select-t-{arm}", lr, chain=chain, allow_dirty=True) as r:
                r.params({"op": "drop_cluster", "cluster_run_id": fcr, "cluster_id": arm, "definition": f"hand-built {arm}", "expected_by_metric": ebm})
            for sd, vd in enumerate(vds):
                add_verify(lr, base, f"v-{arm}-s{sd}", "vs_base", arm, "base", sd, vd, chain)
                add_verify(lr, base, f"vO-{arm}-s{sd}", "vs_base", arm, "base", sd, vd, chain, family="O")  # family O alongside: must not be counted
        for sd in range(3):
            add_verify(lr, base, f"v-cA-vs-rand-s{sd}", "curated_vs_random", "cA", "crandom", sd, verdict_json("T" if sd < 2 else None), chain)
        outc = cf.confirm_run(fcr, "t", "confirm-t", chain=chain, out_dir=base / "oconf", lineage_root=lr, allow_dirty=True)
        st = pd.read_parquet(base / "oconf" / "cluster_status.parquet")
        get = lambda arm, m, comp="vs_base": st[(st["arm"] == arm) & (st["metric"] == m) & (st["comparison"] == comp)].iloc[0]  # noqa: E731
        got = {a: get(a, "hrt60")["status"] for a in cases}
        C.check("4.3a SYNTHETIC verdicts: met in 3/3 seeds -> confirmed; 1/3 -> not confirmed; gate fired in 1/3 -> not confirmed; opposite in 2/3 -> refuted",
                got["cA"] == "confirmed" and got["cB"] == "not confirmed" and got["cC"] == "not confirmed" and got["cD"] == "refuted", f"statuses {got}; calls cA {get('cA', 'hrt60')['calls_by_seed']}, cB {get('cB', 'hrt60')['calls_by_seed']}, cC {get('cC', 'hrt60')['calls_by_seed']}, cD {get('cD', 'hrt60')['calls_by_seed']}")
        C.check("4.3b a trade-off cluster (expected harm on HRT, boost on F30) is judged per metric: both confirmed; a metric with no expectation reads 'no expectation'; family O verdicts are not counted",
                get("cE", "hrt60")["status"] == "confirmed" and get("cE", "f30")["status"] == "confirmed" and get("cE", "rmst60")["status"] == "no expectation"
                and set(st["n_seeds"]) == {3} and cf.overall(st)["cE"] == "confirmed", f"cE hrt60 {get('cE', 'hrt60')['calls_by_seed']} f30 {get('cE', 'f30')['calls_by_seed']}; overall {cf.overall(st)}")
        rnd_row = get("cA", "hrt60", "vs_matched_random")
        C.check("4.3c the matched-random comparison is judged separately (vs_matched_random): T,T,0 -> confirmed (2 of 3, none contradicting)", rnd_row["status"] == "confirmed" and rnd_row["calls_by_seed"] == "T,T,0", f"{rnd_row['calls_by_seed']}")
        run = client(lr).get_run(outc["run_id"])
        C.check("4.3d the confirm run is a tracked summarize run with the table as an output, the rule as a param and the synthetic flag", run.data.tags["stage"] == "summarize" and "output.cluster_status.sha256" in run.data.tags
                and "ceil(2n/3)" in run.data.params["rule"] and run.data.params["synthetic"] == "True" and (st["synthetic"]).all(), f"{outc['rows']} rows; overall {outc['overall']}")
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        st.drop(columns=["verify_run_ids"]).to_csv(ARTIFACTS / "s4_cluster_status_SYNTHETIC.csv", index=False)
        (ARTIFACTS / "s4_dry_run_counts.json").write_text(json.dumps(counts, indent=1))
        (ARTIFACTS / "s4_round_b_dry_run_2clusters_matched.txt").write_text(f"# server_pipeline.py --tag t --dry-run {' '.join(common)} --matched-random\n" + "\n".join(f"{j['id']} {j['name']} deps={','.join(j['dep_names'])} :: {j['cmd']}" for j in j2m) + f"\n{s2m}\n")
        (ARTIFACTS / "s4_round_b_dry_run_2clusters.txt").write_text(f"# server_pipeline.py --tag t --dry-run {' '.join(common)}\n" + "\n".join(f"{j['id']} {j['name']} deps={','.join(j['dep_names'])} :: {j['cmd']}" for j in j2) + f"\n{s2}\n")
        _reset_session()
        del res
    finally:
        shutil.rmtree(base, ignore_errors=True)
        shutil.rmtree(SCRATCH / "synthetic_runs", ignore_errors=True)
        shutil.rmtree(SCRATCH / "fake_clusters", ignore_errors=True)
    return C.finish({"SYNTHETIC": True, "dry_run_counts": counts})


def planted_arms_mode() -> int:
    """S10, AC10.1-AC10.6."""
    import argparse as ap_
    import re

    import stage as stage_mod  # scripts/stage.py

    from vlapoc import attribution as at
    from vlapoc import clusters as cl
    from vlapoc import planted_arms as pa
    from vlapoc import results as res
    from vlapoc import step4
    from vlapoc.groundtruth import load_answer_key
    from vlapoc.lineage import LineageError, Subset
    from vlapoc.method_data import _reset_session, is_groundtruth_subset
    from vlapoc.runs import client, scratch_lineage_copy, stage_run

    C = Checker("planted_arms")
    shutil.rmtree(SCRATCH / "planted_arms", ignore_errors=True)
    base = SCRATCH / "planted_arms"
    base.mkdir(parents=True)
    counts = {}
    try:
        # --- AC10.1: dry-run counts ---
        args0 = ["--subset", "v3/candidate", "--round-b-planted"]
        jd, sd_ = dry_jobs(*args0)
        jn, _ = dry_jobs(*args0, "--no-oracle")
        j2, _ = dry_jobs(*args0, "--random-draws", "2")
        tr = lambda js: len([j for j in js if j["name"].startswith("train-")])  # noqa: E731
        counts = {"planted_default": {"jobs": len(jd), "trainings": tr(jd)}, "planted_no_oracle": {"jobs": len(jn), "trainings": tr(jn)},
                  "planted_random_draws_2": {"jobs": len(j2), "trainings": tr(j2), "delta_jobs": len(j2) - len(jd), "delta_trainings": tr(j2) - tr(jd)}}
        arms_d = re.search(r"arms: (\[.*\])", sd_).group(1)
        C.check("10.1 --round-b-planted default (owner option a): 11 arms, 33 trainings, 129 jobs (plan arithmetic); --no-oracle 27 trainings, 108 jobs; --random-draws 2: +3 trainings, +33 jobs",
                len(jd) == 129 and tr(jd) == 33 and len(jn) == 108 and tr(jn) == 27 and len(j2) - len(jd) == 33 and tr(j2) - tr(jd) == 3,
                f"MEASURED default {len(jd)} jobs / {tr(jd)} trainings; no-oracle {len(jn)} / {tr(jn)}; random-draws 2: {len(j2)} jobs (+{len(j2) - len(jd)}) / {tr(j2)} trainings (+{tr(j2) - tr(jd)}); arms {arms_d[:130]}...")
        by = {j["name"]: j for j in jd}
        sel_cmd = by["select-t-planted"]["cmd"]
        verifies = [j["name"] for j in jd if j["name"].startswith("verify-")]
        ok_struct = (sel_cmd.count("select-planted") == 11 and "--kind type" in sel_cmd and "--kind oracle" in sel_cmd and all(
            "--groundtruth-arm" in j["cmd"] for j in jd if j["name"].startswith("train-")) and sorted(by["discrepancy-verdict-t"]["dep_names"]) == sorted(verifies)
            and sum("curated_vs_random" in j["cmd"] for j in jd) == 8 * 3 + 3)
        C.check("10.1b structure: one select job with 11 select-planted commands (7 types, cleanharm, cleanrandom, oracle, oracle control), every train job has --groundtruth-arm, 27 curated_vs_random verifies (8 effect arms x 3 seeds + oracle x 3), discrepancy-verdict depends on every verify",
                ok_struct, f"{sel_cmd.count('select-planted')} select-planted, {sum('curated_vs_random' in j['cmd'] for j in jd)} curated_vs_random verifies, {len(verifies)} verify jobs")
        # --- AC10.5: argv equals the base recipe except --subset, --name, --groundtruth-arm ---
        jb, _ = dry_jobs("--subset", "v3/candidate", "--curation", "per_task", "--metrics", "f30", "hrt60_lin")
        bad = []
        for s_ in (0, 1, 2):
            b_tr = argv_minus(next(j for j in jb if j["name"] == f"train-t-s{s_}")["cmd"])
            for j in jd:
                if j["name"].startswith("train-") and j["name"].endswith(f"-s{s_}"):
                    if [x for x in argv_minus(j["cmd"]) if x != "--groundtruth-arm"] != b_tr:
                        bad.append(j["name"])
        C.check("10.5 every planted arm's train argv equals the base recipe's (same seed) except --subset, --name and --groundtruth-arm", not bad and tr(jd) == 33, f"33 train jobs compared, mismatches {bad[:3]}")
        # --- AC10.2: the arms over the real v3 answer key (scratch store) ---
        lr = scratch_lineage_copy(base / "lineage")
        stage_mod.LR = lr
        v3 = Subset.load("v3/candidate", lr)
        labels = load_answer_key("v3", lr)
        eps = v3.episodes(lr)
        task = eps.set_index("episode_id")[["suite", "libero_task_id"]].astype(str).agg("/".join, axis=1)
        lab_ids = set(labels["episode_id"])
        types = sorted(set(labels["corruption_type"]))
        made = {}
        for ty in types:
            made[f"type-{ty}"] = pa.planted_select_run("v3/candidate", f"select-t-type-{ty}", "type", corruption_type=ty, lineage_root=lr, allow_dirty=True)
        made["cleanrandom0"] = pa.planted_select_run("v3/candidate", "select-t-cleanrandom0", "cleanrandom", seed=0, lineage_root=lr, allow_dirty=True)
        made["oracle"] = pa.planted_select_run("v3/candidate", "select-t-oracle", "oracle", lineage_root=lr, allow_dirty=True)
        made["cleanrandom91-0"] = pa.planted_select_run("v3/candidate", "select-t-cleanrandom91-0", "cleanrandom_oracle", seed=0, lineage_root=lr, allow_dirty=True)
        tab = synthetic_table(eps, ["f30"], 0, {})
        for c in ("harm_f30__same_task",):
            tab[c] = np.random.default_rng(3).normal(size=len(tab))
        sid = synthetic_score_run(lr, v3, tab, "attribute-dash-synth-s10", note="SYNTHETIC noise for the clean-but-harmful arm's ranking")
        made["cleanharm-f30"] = pa.planted_select_run("v3/candidate", "select-t-cleanharm-f30", "cleanharm", scores_run_id=sid, metric="f30", lineage_root=lr, allow_dirty=True)
        sizes = {k: len(v[1].episode_ids) for k, v in made.items()}
        type_ok = all(sizes[f"type-{ty}"] == 947 and set(made[f"type-{ty}"][1].derivation["dropped"]) == set(labels.loc[labels["corruption_type"] == ty, "episode_id"]) for ty in types)
        cr = made["cleanrandom0"][1].derivation["dropped"]
        cr_ok = sizes["cleanrandom0"] == 947 and len(cr) == 13 and len({task[e] for e in cr}) == 13 and not (set(cr) & lab_ids)
        ch_ = made["cleanharm-f30"][1].derivation["dropped"]
        top_per_task = tab.assign(task=tab["episode_id"].map(task)).loc[~tab["episode_id"].isin(lab_ids)].sort_values(["harm_f30__same_task", "episode_id"], ascending=[False, True]).drop_duplicates("task").head(13)
        ch_ok = sizes["cleanharm-f30"] == 947 and set(ch_) == set(top_per_task["episode_id"]) and len({task[e] for e in ch_}) == 13 and not (set(ch_) & lab_ids)
        orc = made["oracle"][1].derivation["dropped"]
        pt_or = task.loc[sorted(lab_ids)].value_counts().to_dict()
        c91 = made["cleanrandom91-0"][1].derivation["dropped"]
        or_ok = sizes["oracle"] == 869 and set(orc) == lab_ids and sizes["cleanrandom91-0"] == 869 and task.loc[c91].value_counts().to_dict() == pt_or and not (set(c91) & lab_ids)
        C.check("10.2 each type arm's child has 947 episodes, exactly that type's 13 labelled episodes removed; cleanrandom: 947, 13 clean episodes from 13 distinct tasks; cleanharm: 947, the most harmful clean episode of each of the top 13 tasks; oracle: 869; oracle control: 869 with the oracle's per-task counts",
                type_ok and cr_ok and ch_ok and or_ok, f"sizes {sizes}; oracle per-task counts {sorted(set(pt_or.values()))}")
        # --- AC10.6: expectation stated before results, read back from the recorded select runs ---
        def recorded(arm):
            pr = client(lr).get_run(made[arm][0]).data.params
            return pr, (json.loads(pr["expected_by_metric"]) if "expected_by_metric" in pr else None)

        keys = ("f30", "auc_h", "rmst60", "hrt60", "censored60")
        exp_of = {}
        for arm in ("type-detour", "type-idle_padding", "type-gripper_release", "cleanharm-f30", "oracle"):
            pr, ebm = recorded(arm)
            exp_of[arm] = {m: step4.expected_direction("vs_base", pr["direction"], m, arm, "base", ebm) for m in keys}
        ok6 = (all(v == "treated_better" for v in exp_of["type-detour"].values()) and all(v == "treated_better" for v in exp_of["cleanharm-f30"].values())
               and all(v == "treated_better" for v in exp_of["oracle"].values())
               and {m for m, v in exp_of["type-idle_padding"].items() if v} == {"rmst60", "hrt60"} and {m for m, v in exp_of["type-gripper_release"].items() if v} == {"f30", "censored60"})
        C.check("10.6 expected_direction: type and cleanharm arms (and the oracle) treated_better on every metric, except where the label's hypothesis names metrics (idle_padding: rmst60, hrt60; gripper_release: f30, censored60: the others '-'); recorded in the select runs before any result",
                ok6, f"detour {set(exp_of['type-detour'].values())}; idle_padding {exp_of['type-idle_padding']}; gripper_release {exp_of['type-gripper_release']}")
        # --- AC10.3: blindness unchanged ---
        gt_id = made["type-idle_padding"][1].subset_id
        _reset_session()
        tagged_train = None
        with stage_run("train", "fake-train-gtarm", lr, allow_dirty=True, extra_tags={"vlapoc.groundtruth_arm": "true"}) as r:
            r.input_subset(made["type-idle_padding"][1], "training", None)
            tagged_train = r.run_id
        with stage_run("attribute", "attribute-fake-gtarm", lr, allow_dirty=True, extra_tags={"vlapoc.groundtruth_arm": "true"}) as r:
            tagged_score = r.run_id
        ns = lambda **kw: ap_.Namespace(**kw)  # noqa: E731
        refusals = []

        def refuse(label, fn, exc, contains):
            _reset_session()
            before = n_runs_in(lr)
            try:
                fn()
                refusals.append((label, False, "did not raise"))
            except exc as e:
                refusals.append((label, contains in str(e) and n_runs_in(lr) == before, f"{type(e).__name__}: {str(e)[:90]}"))
            except Exception as e:  # noqa: BLE001
                refusals.append((label, False, f"wrong error {type(e).__name__}: {str(e)[:90]}"))

        refuse("train without --groundtruth-arm on a planted-arm subset", lambda: stage_mod.cmd_train(ns(subset=gt_id, groundtruth_arm=False, name="x", seed=0, steps=1, batch_size=1, device="cpu", pretrained_ref="x", smoke=True, chain=None)),
               LineageError, "ground-truth-derived")
        refuse("features on a ground-truth arm's train run", lambda: stage_mod.cmd_features(ns(train=tagged_train, which="train", eval=None, root=None, shard="0/1", stride=None, train_stride=2, rollout_stride=4,
                                                                                             preset="cupid", output_fn=None, rank=8, k_draws=4, seed=0, batch_size=1, workers=1, device="cpu")), LineageError, "ground-truth arm")
        refuse("attribute on a ground-truth arm's train run", lambda: at.attribute_run(tagged_train, "0" * 32, "0" * 32, Path("x"), Path("y"), "attr-bad", lineage_root=lr, allow_dirty=True), LineageError, "ground-truth arm")
        refuse("select with a planted-arm subset as parent", lambda: stage_mod.cmd_select(ns(parent=gt_id, op="drop_random_k", scores=None, k=5, seed=0, name="s", metric=None, direction="highest", chain=None, headline=False,
                                                                                        pipeline_curation=None, clusters=None, cluster_id=None, allow_whole_task=False)), LineageError, "ground-truth-derived")
        refuse("select --scores from a ground-truth arm's run", lambda: stage_mod.cmd_select(ns(parent="v3/candidate", op="drop_top_k", scores=tagged_score, k=5, seed=None, name="s", metric="harm_f30", direction="highest", chain=None,
                                                                                           headline=False, pipeline_curation=None, clusters=None, cluster_id=None, allow_whole_task=False)), LineageError, "ground-truth arm")
        refuse("clusters on a ground-truth arm's run", lambda: cl.clusters_run(tagged_score, "clusters-bad", out_dir=base / "bad", lineage_root=lr, allow_dirty=True, heuristics=False, cache_dir=None), LineageError, "ground-truth arm")
        refuse("analyze on a ground-truth arm's run (extra)", lambda: __import__("vlapoc.analysis", fromlist=["analyze_run"]).analyze_run(tagged_score, "an-bad", lineage_root=lr, allow_dirty=True), LineageError, "ground-truth arm")
        _reset_session()
        guard = (is_groundtruth_subset(gt_id, lr), is_groundtruth_subset("v3/candidate", lr))
        C.check("10.3 stage.py train without --groundtruth-arm refuses a planted-arm subset; features, attribute, select --scores, clusters (and select parent, analyze) refuse planted arms and any vlapoc.groundtruth_arm=true run; "
                "nothing written; the flag accepts only ground-truth-derived subsets",
                all(r_[1] for r_ in refusals) and guard == (True, False), "; ".join(f"{a}: {'REFUSED' if b else 'WRONG'} ({c})" for a, b, c in refusals) + f"; is_groundtruth_subset(planted arm, v3/candidate) = {guard}")
        # --- AC10.4: bounded wording on SYNTHETIC verdicts ---
        def run_case(tag: str, calls_by_arm: dict, only_by_arm: dict | None = None):
            chain = f"{tag}-cc111111"
            for arm, kind, ty, nd in (("type-idle_padding", "type", "idle_padding", 13), ("type-detour", "type", "detour", 13), ("cleanharm-f30", "cleanharm", "", 13), ("oracle", "oracle", "", 91)):
                with stage_run("select", f"select-{tag}-{arm}", lr, chain=chain, allow_dirty=True) as r:
                    r.params({"op": "x", "arm_kind": kind, "groundtruth_arm": True, "corruption_type": ty, "n_dropped": nd, "child_n_episodes": 960 - nd, "direction": "highest"})
                ctrl = "cleanrandom91-0" if arm == "oracle" else "cleanrandom0"
                for sd in range(3):
                    add_verify(lr, base, f"{tag}-{arm}-s{sd}", "curated_vs_random", arm, ctrl, sd, verdict_json(calls_by_arm.get(arm), only=(only_by_arm or {}).get(arm)), chain)
            return pa.discrepancy_verdict_run(tag, f"dv-{tag}", out_dir=base / f"o_{tag}", lineage_root=lr, allow_dirty=True, synthetic=True), pd.read_parquet(base / f"o_{tag}" / "discrepancy_verdict.parquet")
        _, ta = run_case("ta", {"type-idle_padding": None}, {"type-idle_padding": {"hrt60": "T", "rmst60": "T"}})
        g = lambda t, arm, m: t[(t["arm"] == arm) & (t["metric"] == m)].iloc[0]  # noqa: E731
        h, f = g(ta, "type-idle_padding", "hrt60"), g(ta, "type-idle_padding", "f30")
        okA = (h["status_code"] == "harmful (confirmed)" and f["status_code"] == "no detectable harm, bounded" and f["sentence"].startswith("SYNTHETIC: ") and h["status"].startswith("SYNTHETIC: ")
               and "positive control in this round" in f["sentence"] and "shared by every type and seed" in f["sentence"] and re.search(r"spanned \[-?[\d.]+, -?[\d.]+\] \(union over runs\)", f["sentence"])
               and "not proof of no harm" in f["sentence"] and g(ta, "type-detour", "f30")["status_code"] == "no detectable harm, bounded")
        _, tb = run_case("tb", {})
        okB = set(tb["status_code"]) == {"inconclusive"} and tb["sentence"].str.startswith("SYNTHETIC: ").all() and tb["sentence"].str.contains("no arm showed an effect").all() and not tb["positive_control_in_round"].any()
        txt = " ".join(ta["sentence"]) + " " + " ".join(tb["sentence"]) + (base / "o_ta" / "discrepancy_verdict.json").read_text() + (base / "o_tb" / "discrepancy_verdict.json").read_text()
        C.check("10.4 SYNTHETIC verdicts, positive control present: idle_padding harmful (confirmed) on hrt60 and 'no detectable harm, bounded' on f30 with the union CI in the sentence; no arm confirmed anywhere -> every status inconclusive ('no arm showed an effect ...'); the word 'harm'+'less' never appears",
                bool(okA) and okB and ("harm" + "less") not in txt.lower(), f"A: hrt60 '{h['sentence'][:110]}...'; f30 '{f['sentence'][:200]}...'; B: {sorted(set(tb['status_code']))}")
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        ta.drop(columns=["verify_run_ids"]).to_csv(ARTIFACTS / "s10_discrepancy_verdict_positive_control_SYNTHETIC.csv", index=False)
        tb.drop(columns=["verify_run_ids"]).to_csv(ARTIFACTS / "s10_discrepancy_verdict_all_null_SYNTHETIC.csv", index=False)
        (ARTIFACTS / "s10_dry_run_counts.json").write_text(json.dumps(counts, indent=1))
        (ARTIFACTS / "s10_round_b_planted_dry_run_default.txt").write_text("# server_pipeline.py --tag t --dry-run --subset v3/candidate --round-b-planted\n" + "\n".join(f"{j['id']} {j['name']} deps={','.join(j['dep_names'])} :: {j['cmd']}" for j in jd) + f"\n{sd_}\n")
        del res
    finally:
        shutil.rmtree(base, ignore_errors=True)
        shutil.rmtree(SCRATCH / "synthetic_runs", ignore_errors=True)
        shutil.rmtree(SCRATCH / "fake_clusters", ignore_errors=True)
    return C.finish({"SYNTHETIC": True, "dry_run_counts": counts})


# ---------------------------------------------------------------------------------------------
# S5 / S6 / S7: the dashboard, the demo store, the drop-in proof
# ---------------------------------------------------------------------------------------------

APP = REPO_ROOT / "dashboard" / "app.py"
DEMO = REPO_ROOT / "data" / "lineage_demo"
PAGES = ("Overview", "Rollouts", "Episodes", "Clusters", "Bad vs harmful")


def app_test(lineage_root: str, scores: str, clusters: str = ""):
    """A headless Streamlit ``AppTest`` of ``dashboard/app.py`` on a store (the app reads its options from these environment variables)."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.cache_data.clear()
    os.environ.update({"DASH_LINEAGE_ROOT": lineage_root, "DASH_SCORES": scores, "DASH_CLUSTERS": clusters})
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    return at


def go(at, page: str):
    at.sidebar.radio(key="page").set_value(page).run()
    return at


def texts(at) -> list[str]:
    """Every user-visible string of the rendered page: markdown, captions, titles, banners, metric labels/values, widget labels and options."""
    out = []
    for kind in ("markdown", "caption", "title", "subheader", "error", "warning", "info", "success"):
        out += [str(e.value) for e in getattr(at, kind)]
    out += [f"{m.label} {m.value}" for m in at.metric]
    for kind in ("selectbox", "radio", "multiselect", "checkbox", "toggle", "slider"):
        for w in getattr(at, kind):
            out.append(str(w.label))
            out += [str(o) for o in getattr(w, "options", [])]
    return out


def chart_strings(at) -> list[str]:
    """Titles and axis titles of the page's vega-lite charts."""
    import re

    out = []
    for ch in at.get("vega_lite_chart"):
        out += [json.loads(f'"{x}"') for x in re.findall(r'"(?:title|text)": ?"((?:[^"\\]|\\.)*)"', ch.proto.spec)]
    return out


SNAKE = re.compile(r"[a-z]+_[a-z_]+")


def label_strings(at) -> list[str]:
    """Rendered *labels*: dataframe column names, widget labels and options, headings, markdown and captions (code blocks and backtick spans, which hold run names and ids, are left out),
    metric labels and chart titles. Cell values (task names, recorded sentences) are data, not labels."""
    out = [str(c) for df in at.dataframe for c in df.value.columns] + chart_strings(at) + [f"{m.label}" for m in at.metric]
    for kind in ("markdown", "caption", "title", "subheader", "error", "warning", "info", "success"):
        out += [str(e.value) for e in getattr(at, kind)]
    for kind in ("selectbox", "radio", "multiselect", "checkbox", "toggle", "slider"):
        for w in getattr(at, kind):
            out.append(str(w.label))
            out += [str(o) for o in getattr(w, "options", [])]
    return [re.sub(r"`[^`]*`", "", x) for x in out]


def snake_labels(at) -> list[str]:
    """AC5.9: labels that still read like code (``snake_case``) outside run names and ids."""
    return sorted({x[:80] for x in label_strings(at) if SNAKE.search(x)})


def unreached(at) -> list[str]:
    """AC5.4: short metric names shown on the page whose declared functional and tau are not reachable from it: some caption, markdown or tooltip must read ``<name>: <functional> [τ ...]``."""
    from dashboard import words

    labels = label_strings(at)
    helps = [str(w.help) for kind in ("selectbox", "radio", "multiselect", "checkbox", "toggle", "slider") for w in getattr(at, kind) if getattr(w, "help", None)]
    declared = [str(e.value) for kind in ("markdown", "caption") for e in getattr(at, kind)] + helps
    shown = [n for n in words.METRICS.values() if any(n in x for x in labels)]
    return [n for n in shown if not any(f"{n}: " in d and "τ" in d for d in declared)]


def app_mode() -> int:
    """S5, AC5.1-AC5.8."""
    import re
    import subprocess
    import urllib.request

    import streamlit
    from dashboard.views.episodes import groupable
    from vlapoc import results as res
    from vlapoc import video
    from vlapoc.analysis import aggregate_by
    from vlapoc.discrepancy import QUADRANTS
    from vlapoc.runs import client

    C = Checker("app")
    if not DEMO.exists():
        print("data/lineage_demo is missing: run scripts/dashboard_demo.py build --with-smoke first")
        return 2
    man = json.loads((DEMO / "demo_manifest.json").read_text())
    lr = DEMO
    t0 = time.time()
    at = app_test("data/lineage_demo", "attribute-demo-ens")
    bad_pages, no_banner, bare = [], [], {}
    pages = {}
    for pg in PAGES:
        go(at, pg)
        pages[pg] = at
        if at.exception:
            bad_pages.append((pg, [e.value[:100] for e in at.exception]))
        if not any("SYNTHETIC" in str(e.value) for e in at.error):
            no_banner.append(pg)
        b = snake_labels(at) + [f"functional not reachable: {n}" for n in unreached(at)]
        if b:
            bare[pg] = b
        print(f"  page {pg}: {len(at.dataframe)} dataframes, {len(at.get('video'))} videos, {len(at.get('vega_lite_chart'))} charts, {len(at.get('imgs'))} images")
    C.check("5.1 AppTest (headless) on the demo store: every page runs with no exception, and the SYNTHETIC banner is present on every page",
            not bad_pages and not no_banner, f"{len(PAGES)} pages in {time.time() - t0:.0f}s; exceptions {bad_pages}; pages without the banner {no_banner}")
    # --- AC5.2 rollouts ---
    go(at, "Rollouts")
    seed = at.selectbox(key="seed_run").value
    ro = res.rollouts(seed, lr)
    failed = ro[ro["failure_type"] != "success"]
    pick = failed[failed["task_name"].str.startswith("KITCHEN")].iloc[0] if (failed["task_name"].str.startswith("KITCHEN")).any() else failed.iloc[0]
    at.checkbox(key="only_failed").set_value(True)
    at.selectbox(key="rollout_pick").set_value(pick["rollout_id"]).run()
    ft = [str(e.value).split("  \n", 1)[1] for e in at.markdown if str(e.value).startswith("**Failure type**")]
    m_, idx = res.influence(seed, lr)
    row = m_[int(pick["matrix_row"])]
    want = [idx["episode_id"][j] for j in np.argsort(-row, kind="stable")[:10]]
    tbl = next(df.value for df in at.dataframe if "influence on this rollout" in df.value.columns).rename(columns={"episode": "episode_id"})
    ret = res._fetch(client(lr).get_run(json.loads(client(lr).get_run(seed).data.params["returns_run_ids"])[0 if pick["eval_run_id"] == json.loads(client(lr).get_run(seed).data.params["eval_run_ids"])[0] else 1]), "returns", lr, pd.read_parquet)
    ret_ft = ret.set_index("rollout_id").loc[pick["rollout_id"], "failure_type"]
    C.check("5.2 Rollouts: a selected failed rollout shows the failure type of returns.parquet, a video, and 10 episode rows in the order of argsort of its influence-matrix row (recomputed here)",
            ft and ft[0] == __import__("dashboard.words", fromlist=["plain"]).plain(ret_ft) and len(at.get("video")) >= 1 and list(tbl["episode_id"]) == want and len(tbl) == 10,
            f"rollout {pick['rollout_id']}: failure type {ft} (returns.parquet: {ret_ft}); {len(at.get('video'))} videos; top-10 order equal: {list(tbl['episode_id']) == want}")
    # --- AC5.3 episodes ---
    go(at, "Episodes")
    ens = man["scores_run_id"]
    t, decls = res.score_table(ens, lr), res.functionals(ens, lr)
    col = at.selectbox(key="ep_col").value
    from dashboard import words

    lab = words.metric_name(col)
    _w = words
    ranked = next(df.value for df in at.dataframe if lab in df.value.columns).rename(columns={"episode": "episode_id"})
    exp = t.sort_values([col, "episode_id"], ascending=[False, True])["episode_id"].head(len(ranked)).tolist()
    opts = at.selectbox(key="ep_group").options
    cat = res.training_episodes(ens, lr)[1]
    agg_shown = next(df.value for df in at.dataframe if any(str(c).startswith("mean ") for c in df.value.columns))
    agg_group = [c for c in agg_shown.columns if c not in ("n",) and not str(c).startswith(("mean ", "95%"))][0]
    assert agg_shown.columns[2] == f"mean {lab}"
    agg = aggregate_by(t, at.selectbox(key="ep_group").value, [col], cat)
    from vlapoc.catalog import CATALOG_COLUMNS
    from vlapoc.lineage import GROUP_COLUMNS_EXCLUDED

    full = [c for c in CATALOG_COLUMNS if c not in GROUP_COLUMNS_EXCLUDED]
    try:
        aggregate_by(t, "objects", [col], cat, n_boot=5)
        objects_ok = True
    except Exception as e:  # noqa: BLE001
        objects_ok, objects_err = False, type(e).__name__
    C.check("5.3 Episodes: the ranked list equals the score table sorted by the chosen harm column; 'aggregate by' offers the groupable catalog columns (CATALOG_COLUMNS minus GROUP_COLUMNS_EXCLUDED, minus the list-valued 'objects', which aggregate_by cannot group: measured); the table equals aggregate_by",
            list(ranked["episode_id"]) == exp and list(opts) == [words.plain(c) for c in groupable()] and groupable() == [c for c in full if c != "objects"] and not objects_ok
            and np.allclose(agg_shown.iloc[:, 2].to_numpy(float), agg[f"{col}__mean"].to_numpy(float)) and list(agg_shown["n"]) == list(agg["n"]),
            f"{len(ranked)} ranked rows equal; offered {list(opts)}; objects fails with {objects_err if not objects_ok else 'nothing'}; aggregate rows {len(agg)}")
    # --- AC5.4 labels ---
    C.check("5.4/5.9 plain language with the declared functional reachable: no rendered label (dataframe column, widget label or option, heading, caption, markdown, chart title) matches [a-z]+_[a-z_]+ outside run names and ids, and every metric name shown has "
            "its '<name>: <functional> [τ ...]' declaration in a caption or tooltip on the same page (all 5 pages scanned)", not bare, f"offending labels: {bare}")
    # --- AC5.5 clusters ---
    go(at, "Clusters")
    spec = " ".join(ch.proto.spec for ch in at.get("vega_lite_chart"))  # the quadrant labels are inline data values of a text layer
    cid0 = man["picked_clusters"]["cluster0"]
    at.selectbox(key="cl_metric").set_value(f"harm_{cid0.split(':')[1]}" if cid0.startswith("slice:") else "harm_hrt60_lin").run()
    tab = next(df.value for df in at.dataframe if "cluster" in df.value.columns and "status" in df.value.columns)
    from dashboard import words as _w

    cdef = res.output_table(man["clusters_run_id"], "clusters", lr).set_index("cluster_id")["definition"]
    st_of = dict(zip(tab["cluster"], tab["status"]))
    st_of = {cid_: st_of.get(_w.plain(cdef[cid_])) for cid_ in man["picked_clusters"].values() if cid_ in cdef.index} | {"_all": None}
    st_all = set(tab["status"])
    md = " ".join(str(e.value) for e in at.markdown)
    quad = all(q in spec for q in ("consistent harmers", "consistent helpers", "trade-off"))
    conf = man["confirm_overall"]
    C.check("5.5 Clusters: heatmap, a clickable scatter with the four quadrant labels, the cluster list with status from the confirm run when one exists (else estimated) and a lineage link per cluster",
            len(at.get("image")) >= 1 and len(at.get("vega_lite_chart")) >= 1 and quad and st_of.get(cid0) == "SYNTHETIC: " + conf[cid0]
            and all(x.removeprefix("SYNTHETIC: ") in {"estimated", "confirmed", "refuted", "not confirmed", "mixed"} for x in st_all) and "#/experiments/" in md and "vlapoc.lineage trace" in " ".join(c.value for c in at.code),
            f"quadrant labels in the chart spec: {quad}; statuses shown {sorted(st_all)}; {cid0} = {st_of.get(cid0)} (confirm run: {conf[cid0]})")
    # --- AC5.5b bad vs harmful ---
    go(at, "Bad vs harmful")
    drun = res.runs_with_param("score", "score_run_id", ens, lr)
    drun = [r for r in drun if "discrepancy_episodes" in r["outputs"]][0]
    ep = res.output_table(drun["run_id"], "discrepancy_episodes", lr)
    metric = at.selectbox(key="bvh_metric").value
    from dashboard import words as _w2

    shown = next(df.value for df in at.dataframe if "quadrant" in df.value.columns)
    inv = {v: k for k, v in _w2.QUADRANTS.items()}
    want_q = ep[f"quadrant__{metric}"].value_counts().reindex(QUADRANTS, fill_value=0)
    vruns = [r for r in res.find_runs("summarize", lr) if "discrepancy_verdict" in r["outputs"]]
    vt = res.output_table(vruns[0]["run_id"], "discrepancy_verdict", lr)
    vshown = next(df.value for df in at.dataframe if "sentence" in df.value.columns)
    alltxt = " ".join(texts(at)) + " ".join(str(c) for df in at.dataframe for c in df.value.to_numpy().ravel())
    nvid = len(at.get("video"))
    C.check("5.5b Bad-vs-harmful: quadrant counts equal discrepancy_episodes.parquet's; statuses and sentences equal the discrepancy-verdict run's verbatim; the word 'harm'+'less' appears nowhere on the page; the selected type shows two video elements (corrupted, original)",
            {inv[k]: v for k, v in zip(shown["quadrant"], shown["episodes"])} == {q: int(want_q[q]) for q in QUADRANTS} and set(vshown["sentence"]) == set(vt["sentence"]) and set(vshown["status"]) == set(vt["status"])
            and ("harm" + "less") not in alltxt.lower() and nvid == 2,
            f"quadrant counts {dict(zip(shown['quadrant'], shown['episodes']))}; {len(vshown)} verdict rows, statuses {sorted(set(vshown['status']))}; {nvid} videos")
    # --- AC5.6 video ---
    sid, cat2 = res.training_episodes(ens, lr)
    root = res.dataset_root(sid, lr)
    row = cat2.iloc[7]
    cache = SCRATCH / "video_cache"
    shutil.rmtree(cache, ignore_errors=True)
    b1 = video.episode_video(root, int(row["episode_index"]), row["content_hash"], cache_dir=cache)
    b2 = video.episode_video(root, int(row["episode_index"]), row["content_hash"], cache_dir=cache)
    n_ep = int(row["length_steps"])
    ro0 = res.rollouts(res.seed_runs(ens, lr)[0], lr).iloc[0]
    troot = res.trajectory_root(ro0["eval_run_id"], lr)
    r1 = video.episode_video(troot, int(ro0["traj_episode_index"]), "scratch-rollout", cache_dir=cache)
    r2 = video.episode_video(troot, int(ro0["traj_episode_index"]), "scratch-rollout", cache_dir=cache)
    n_roll = int(video.frames_of(troot, int(ro0["traj_episode_index"])).shape[0])
    C.check("5.6 vlapoc.video: a real training episode and a real rollout trajectory become H.264 mp4 whose decoded frame count equals the episode's frame count; the cache returns the same bytes on the second call",
            video.decode_frame_count(b1) == n_ep and video.decode_frame_count(r1) == n_roll and b1 == b2 and r1 == r2 and b1[4:8] == b"ftyp",
            f"training episode {row['episode_id'][-30:]}: {video.decode_frame_count(b1)} frames = {n_ep} ({len(b1)} bytes); rollout {ro0['rollout_id']}: {video.decode_frame_count(r1)} frames = {n_roll} ({len(r1)} bytes)")
    shutil.rmtree(cache, ignore_errors=True)
    # --- AC5.7 no ad-hoc paths; another store changes the content ---
    hard = [f.name for f in (REPO_ROOT / "dashboard").rglob("*.py") if re.search(r"data/(attribution|eval)|\"attribution\"|\"eval\" */", f.read_text())]
    at2 = app_test("", "")
    opts_real = [str(o) for o in at2.sidebar.selectbox[0].options] if at2.sidebar.selectbox else []
    opts_demo = [str(o) for o in app_test("data/lineage_demo", "attribute-demo-ens").sidebar.selectbox[0].options]
    msg_real = [str(e.value)[:120] for e in at2.error] + [str(e.value)[:120] for e in at2.warning]
    C.check("5.7 the app reads only through results.py/MLflow (no hard-coded data/attribution or data/eval path in dashboard/); pointing --lineage-root/--scores at another store (the real one) changes the content with no edit",
            not hard and opts_real != opts_demo and not at2.exception, f"hard-coded paths in {hard}; demo run options {opts_demo[:2]}...; real store options {opts_real[:2]}; messages {msg_real[:1]}")
    # --- AC5.8 streamlit installed and serving ---
    port = 8766
    srv = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(APP), "--server.headless", "true", "--server.port", str(port), "--browser.gatherUsageStats", "false", "--", "--lineage-root", "data/lineage_demo", "--scores", "attribute-demo-ens"],
                           cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    code, body, page = None, "", None
    try:
        for _ in range(60):
            try:
                r_ = urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=2)
                code, body = r_.status, r_.read().decode()
                page = urllib.request.urlopen(f"http://localhost:{port}/", timeout=5).status
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
    finally:
        srv.terminate()
        srv.wait(timeout=15)
    C.check("5.8 streamlit==1.66.1 is installed in .venv; `streamlit run dashboard/app.py` serves locally (HTTP 200 on /_stcore/health, and on /), then it is stopped",
            streamlit.__version__ == "1.66.1" and code == 200 and body == "ok" and page == 200, f"streamlit {streamlit.__version__}; health {code} {body!r}; page {page}; stopped")
    return C.finish({"SYNTHETIC": True})


def demo_mode() -> int:
    """S6, AC6.1-AC6.5."""
    from vlapoc import clusters as cl
    from vlapoc import results as res
    from vlapoc import synthetic as syn
    from vlapoc.lineage import LineageError
    from vlapoc.method_data import _reset_session
    from vlapoc.runs import client, experiment_id

    C = Checker("demo")
    if not DEMO.exists():
        print("data/lineage_demo is missing: run scripts/dashboard_demo.py build --with-smoke first")
        return 2
    man = json.loads((DEMO / "demo_manifest.json").read_text())
    C.check("6.1 the build with --with-smoke completed (manifest) and left data/lineage/mlflow.db unchanged (its sha256 before == after == now)",
            "smoke" in man and man["mlflow_db_sha256_before"] == man["mlflow_db_sha256_after"] == db_sha(), f"build took {man['seconds']}s; smoke attribute run {man['smoke']['attribute_name']} ({man['smoke']['n_rollouts']} rollouts); sha {db_sha()[:16]}...")
    lr = DEMO
    c = client(lr)
    real = {r.info.run_id for r in client(None).search_runs([experiment_id(None)], max_results=50000)}
    mine = [r for r in c.search_runs([experiment_id(lr)], max_results=50000) if r.info.run_id not in real]
    smoke_chain = [r for r in mine if r.data.tags.get("vlapoc.chain", "").startswith("demo-smoke")]
    other = [r for r in mine if r not in smoke_chain]
    untagged = [r.info.run_name for r in other if r.data.tags.get("vlapoc.synthetic") != "true"]
    smoke_tagged = [r.info.run_name for r in smoke_chain if r.data.tags.get("vlapoc.synthetic") == "true"]
    p_ens = res.provenance(man["scores_run_id"], lr)
    p_smoke = res.provenance(man["smoke"]["attribute_run"], lr)
    C.check("6.2 every run the builder created is tagged vlapoc.synthetic=true except the smoke chain's; provenance is SYNTHETIC for the demo ensemble and SMOKE for the smoke attribute run",
            not untagged and smoke_chain and not smoke_tagged and p_ens["label"] == "SYNTHETIC" and p_smoke["label"] == "SMOKE",
            f"{len(other)} synthetic-tagged runs, {len(smoke_chain)} smoke-chain runs (untagged by design); ensemble {p_ens['label']}; smoke {p_smoke['label']}: {p_smoke['reasons'][:1]}; untagged synthetic-side runs {untagged[:3]}")
    src = REPO_ROOT / "vlapoc" / "synthetic.py"
    _reset_session()
    try:
        syn.demo_inputs("v2/candidate", lr)
        refused = "opened"
    except LineageError as e:
        refused = str(e)[:80]
    _reset_session()
    prm = c.get_run(man["seed_runs"][0]).data.params
    C.check("6.3 the synthetic generator imports nothing from groundtruth / recovery / discrepancy and opens the subset only through open_for_method (a fenced v2 is refused); the planted structure is described in plain words in the run params",
            not forbidden_imports(src) and "open_for_method(" in src.read_text() and "fenced" in refused and "SYNTHETIC" in prm["planted_structure"] and "length_vs_task" in prm["planted_structure"],
            f"forbidden imports {forbidden_imports(src)}; v2 refused: {refused}; params: {prm['planted_structure'][:150]}...")
    # --- AC6.4 the demo's cluster run recovers the planted structure; S3 table exists ---
    view, desc, codes, specs = syn.demo_inputs("v3/candidate", lr)
    cells = syn.plant_cells(desc, codes, specs)
    top = int(codes["length_vs_task"].max())
    clu = res.output_table(man["clusters_run_id"], "clusters", lr)
    sl = clu[clu["lens"] == "slice"]

    # Clusters with identical member sets are reported once (AC2.9): the kept row lists the others as equivalent definitions, so the slices found for one metric and side are the kept rows' own
    # entries plus the equivalents that name that metric. A planted structure counts as recovered if the definition of a top-3 entry states it.
    entries = []
    for _, r in clu.iterrows():
        for e in [{"cluster_id": r["cluster_id"], "lens": r["lens"], "definition": r["definition"], "direction": r["direction"], "target_metric": r["target_metric"]}, *json.loads(r["equivalent_definitions"])]:
            if e["lens"] == "slice":
                entries.append({**e, "rank": int(e["cluster_id"].rsplit(":", 1)[1]), "n": int(r["n"])})
    ent = pd.DataFrame(entries)

    def top3(metric, side):
        d = ent[(ent["target_metric"] == f"harm_{metric}") & (ent["direction"] == side)].sort_values("rank").head(3)
        return [(int(r["rank"]), r["definition"], int(r["n"])) for _, r in d.iterrows()]

    def states(top, text):
        return any(all(part in d for part in text.split(" AND ")) for _, d, _ in top)

    t_h, t_f, t_b = top3("hrt60_lin", "harm"), top3("f30", "harm"), top3("f30", "boost")
    ok_h, ok_f, ok_b = states(t_h, cells["text"]["hrt_harm"]), states(t_f, cells["text"]["f30_harm"]), states(t_b, cells["text"]["f30_boost"])
    rec = [r for r in res.runs_with_param("score", "cluster_run_id", man["clusters_run_id"], lr) if "cluster_recovery" in r["outputs"]]
    tb = res.output_table(rec[0]["run_id"], "cluster_recovery", lr)
    best = tb[(tb["planted_type"] != "any_planted") & tb["is_best_cluster_for_type"]].sort_values("p_fisher").head(3)
    C.check("6.4 the demo's cluster run recovers the planted structure (planted HRT slice, F30 scene slice, F30 boost slice each in the top 3 of its metric and side), and its S3 table is produced; any overlap with the real planted types is SYNTHETIC and not evidence about attribution",
            ok_h and ok_f and ok_b and len(tb) == len(clu) * 8 and tb["synthetic"].all(),
            f"planted: {cells['text']}; found hrt {t_h[:2]}, f30 harm {t_f[:1]}, f30 boost {t_b[:1]} (identical member sets are listed once with their equivalents); S3 table {len(tb)} rows; best (cluster, type) by Fisher p: " + "; ".join(f"{r.planted_type}: {r.cluster_id} p={r.p_fisher:.1e}" for r in best.itertuples()))
    # --- AC6.5 disk ---
    def du(p: Path) -> int:
        return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())

    store, cache = du(DEMO), du(REPO_ROOT / "data" / "dashboard_cache")
    free = shutil.disk_usage(REPO_ROOT).free
    C.check("6.5 disk: the demo store plus the dashboard cache stay under 2 GiB and free disk stays above 20 GiB", store + cache < 2 * 2**30 and free > 20 * 2**30,
            f"demo store {store / 2**20:.0f} MiB (outputs of the smoke chain live under data/eval and data/attribution), cache {cache / 2**20:.0f} MiB, free {free / 2**30:.0f} GiB")
    del cl
    return C.finish({"SYNTHETIC": True, "manifest": {k: man[k] for k in ("planted", "warm", "confirm_overall", "picked_clusters", "seconds")}})


def dropin_mode() -> int:
    """S7, AC7.1-AC7.3."""
    import subprocess

    from vlapoc.runs import client

    C = Checker("dropin")
    v = vt()
    porcelain = lambda: subprocess.run(["git", "status", "--porcelain", "--", "dashboard", "vlapoc", "scripts"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    if porcelain():
        print(f"commit first (the stage CLI refuses a dirty tree and AC7.1 needs a clean one):\n{porcelain()}")
        return 2
    shutil.rmtree(SCRATCH / "dropin", ignore_errors=True)
    base = SCRATCH / "dropin"
    base.mkdir(parents=True)
    CH, T = "di-fixture0", "di"
    made = [REPO_ROOT / "data" / "attribution" / n for n in (f"attribute-{T}-s0", f"attribute-{T}-s1", f"attribute-{T}-ens", f"clusters-{T}")]
    try:
        fx = v.make_fixture(base, parent="v3/candidate", cover_types=True, chain=CH, train_name=f"train-{T}-s0")
        lr, feats = fx["lr"], base / "feats"
        tr0 = fx["train"]
        tr1 = v.make_train(fx, f"train-{T}-s1", fx["subset"], "cd" * 32, seed=1)
        v.add_eval(fx, "s0", train=tr0, eval_name=f"eval-{T}-s0", returns_name=f"returns-{T}-s0", seed=1)
        v.add_eval(fx, "s1", train=tr1, eval_name=f"eval-{T}-s1", returns_name=f"returns-{T}-s1", seed=2)
        Cli = lambda *a: v.stage_cli(list(a), lr, feats, CH)  # noqa: E731
        for s_ in (0, 1):
            Cli("attribute", "--train", f"train-{T}-s{s_}", "--eval", f"eval-{T}-s{s_}", "--returns", f"returns-{T}-s{s_}", "--name", f"attribute-{T}-s{s_}", "--device", "cpu")
        Cli("ensemble", "--runs", f"attribute-{T}-s0", f"attribute-{T}-s1", "--name", f"attribute-{T}-ens")
        Cli("clusters", "--scores", f"attribute-{T}-ens", "--name", f"clusters-{T}")
        before = porcelain()
        at = app_test(str(lr), f"attribute-{T}-ens")
        errs = []
        for pg in PAGES:
            go(at, pg)
            if at.exception:
                errs.append((pg, [e.value[:90] for e in at.exception]))
        after = porcelain()
        banner0 = [str(e.value)[:60] for e in at.warning] + [str(e.value)[:60] for e in at.error]
        C.check("7.1 the real attribute -> ensemble -> clusters code path (SYNTHETIC features, scratch store, run through the stage CLI) loads in the dashboard by run name, every page, with no file under dashboard/, vlapoc/ or scripts/ modified (git status --porcelain empty before and after)",
                not before and not after and not errs, f"porcelain before {before!r} after {after!r}; page errors {errs}; banner on the untagged fixture: {banner0[:1]}")
        # --- AC7.2 re-tag train/eval to the values the server writes ---
        c = client(lr)
        from vlapoc.runs import experiment_id

        for r in [*c.search_runs([experiment_id(lr)], "tags.stage = 'train'", max_results=1000), *c.search_runs([experiment_id(lr)], "tags.stage = 'eval'", max_results=1000)]:
            if r.data.tags.get("stage") == "train" and r.info.run_name.startswith(f"train-{T}"):
                c.set_tag(r.info.run_id, "vlapoc.smoke", "false")
            if r.data.tags.get("stage") == "eval" and r.info.run_name.startswith(f"eval-{T}"):
                c.set_tag(r.info.run_id, "eval.attribution_grade", "true")
        at = app_test(str(lr), f"attribute-{T}-ens")
        go(at, "Overview")
        flagged = [str(e.value)[:80] for e in at.error] + [str(e.value)[:80] for e in at.warning]
        caps = [str(e.value) for e in at.caption]
        real_ok = not any(("SYNTHETIC" in x or "SMOKE" in x) for x in flagged) and any(x.startswith("Provenance: REAL") for x in caps)
        seed_run = c.search_runs([experiment_id(lr)], f"tags.mlflow.runName = 'attribute-{T}-s0'")[0].info.run_id
        c.set_tag(seed_run, "vlapoc.synthetic", "true")
        at = app_test(str(lr), f"attribute-{T}-ens")
        go(at, "Overview")
        back = any("SYNTHETIC" in str(e.value) for e in at.error)
        C.check("7.2 the same chain with its train/eval runs re-tagged to what the server writes (vlapoc.smoke=false, eval.attribution_grade=true, nothing synthetic) renders with no SYNTHETIC or SMOKE banner (caption 'Provenance: REAL'); re-adding one synthetic tag brings the SYNTHETIC banner back",
                real_ok and back, f"re-tagged: banners {flagged}, caption {[x[:40] for x in caps if x.startswith('Provenance')]}; one synthetic tag re-added: banner back {back}")
        # --- AC7.3 the real smoke run in the demo store ---
        if DEMO.exists():
            man = json.loads((DEMO / "demo_manifest.json").read_text())
            at = app_test("data/lineage_demo", man["smoke"]["attribute_name"])
            go(at, "Rollouts")
            rows = next(df.value for df in at.dataframe if "rollout" in df.value.columns)
            at.checkbox(key="only_failed").set_value(False).run()
            rows = next(df.value for df in at.dataframe if "rollout" in df.value.columns)
            warn = [str(e.value)[:70] for e in at.warning]
            C.check("7.3 the real --with-smoke attribute run (real attribute_run output, no adapter) renders on the Rollouts page with its 2 real rollouts and the SMOKE banner",
                    len(rows) == 2 and any("SMOKE" in w for w in warn) and not at.exception and len(at.get("video")) >= 1,
                    f"{len(rows)} rollouts {list(rows['rollout'])}; banner {warn[:1]}; {len(at.get('video'))} videos (rollout + episodes)")
        else:
            C.check("7.3 demo store present", False, "run scripts/dashboard_demo.py build --with-smoke")
    finally:
        shutil.rmtree(base, ignore_errors=True)
        for d in made:
            shutil.rmtree(d, ignore_errors=True)
    return C.finish({"SYNTHETIC": "fixture features/rollouts/returns"})


def render_mode() -> int:
    """S5, AC5.10: what a real browser draws, measured in headless Chrome at a 1400 px window (AppTest cannot see any of it)."""
    import asyncio
    import subprocess
    import urllib.request

    import dashboard_screenshots as ds
    from vlapoc import results as res

    C = Checker("render")
    if not DEMO.exists():
        print("data/lineage_demo is missing: run scripts/dashboard_demo.py build --with-smoke first")
        return 2
    man = json.loads((DEMO / "demo_manifest.json").read_text())
    seed = res.seed_runs(man["scores_run_id"], DEMO)[0]
    ro = res.rollouts(seed, DEMO)
    ids = list(ro[ro["failure_type"] != "success"]["rollout_id"])  # the Rollouts table at its default filters, in display order
    port = 8767
    srv = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(APP), "--server.headless", "true", "--server.port", str(port), "--browser.gatherUsageStats", "false", "--",
                            "--lineage-root", "data/lineage_demo", "--scores", "attribute-demo-ens"], cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=2)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        m = asyncio.run(ds.measure(f"http://localhost:{port}", 9433, SCRATCH / "chrome-profile", 1400, 10.0, ids, row=4))
    finally:
        srv.terminate()
        srv.wait(timeout=15)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "s5_render_measurements.json").write_text(json.dumps(m, indent=1))
    plots = {pg: v["plots"] for pg, v in m["pages"].items()}
    small = {pg: [x for x in ps if x[1] < 200] for pg, ps in plots.items() if any(x[1] < 200 for x in ps)}
    axes = json.loads(next((REPO_ROOT / "data" / "dashboard_cache" / "figs").glob(f"{man['scores_run_id']}-heatmap-v2.axes.json")).read_text())
    imgs = m["pages"]["Clusters"]["images"]
    C.check("5.10a every Altair chart's plot area is at least 200 px tall at 1400 px (SVG path.background; Clusters and Bad vs harmful each have one) and the matplotlib heatmap's axes area and displayed image are at least 200 px tall",
            not small and len(plots["Clusters"]) >= 1 and len(plots["Bad vs harmful"]) >= 1 and axes["axes_height_px"] >= 200 and all(i[2] >= 200 for i in imgs),
            f"plot areas (w, h) {plots}; heatmap axes {axes['axes_width_px']:.0f}x{axes['axes_height_px']:.0f}px, displayed image heights {[i[2] for i in imgs]}")
    rc = m.get("row_click", {})
    C.check("5.10b clicking row 5 of the Rollouts table makes the page show that row's rollout_id as the selected rollout", rc.get("selected_after_click") is True and rc.get("before") != rc.get("after"),
            f"expected {rc.get('expected')!r}; selected before {rc.get('before')!r}, after {rc.get('after')!r}")
    trunc = [(pg, x["text"]) for pg, v in m["pages"].items() for x in v["metrics"] if x["truncated"]]
    exc = {pg: v["exceptions"] for pg, v in m["pages"].items() if v["exceptions"]}
    C.check("5.10c no st.metric value is truncated, the sidebar holds no auto-generated page links (only the page radio), and no page shows an exception",
            not trunc and m["sidebar_nav_links"] == 0 and not exc, f"truncated {trunc}; sidebar page links {m['sidebar_nav_links']}; exceptions {exc}")
    return C.finish({"measurements": m})


MODES = {"results": results, "clusters": clusters_mode, "planted": planted_mode, "discrepancy": discrepancy_mode, "round_b": round_b_mode, "planted_arms": planted_arms_mode, "app": app_mode, "render": render_mode, "demo": demo_mode, "dropin": dropin_mode}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for m, h in (("results", "S1"), ("clusters", "S2"), ("planted", "S3"), ("discrepancy", "S9"), ("round_b", "S4"), ("planted_arms", "S10"), ("app", "S5"), ("render", "S5 rendering"), ("demo", "S6"), ("dropin", "S7")):
        ap.add_argument(f"--{m.replace('_', '-')}", dest=m, action="store_true", help=h)
    a = ap.parse_args()
    ran = [m for m in MODES if getattr(a, m, False)]
    if not ran:
        ap.print_help()
        return 2
    return max(MODES[m]() for m in ran)


if __name__ == "__main__":
    sys.exit(main())
