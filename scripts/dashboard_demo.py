#!/usr/bin/env python
"""Build the dashboard's demo store ``data/lineage_demo/`` (dashboard S6): real videos, catalog and rollouts, SYNTHETIC scores labelled as such,
and, with ``--with-smoke``, the real 2-step SmolVLA chain shown as SMOKE.

Grounding: owner answer 2026-10-22 "Synthetic scores, labelled. Real LIBERO videos and catalog, the real (not attribution-grade) MPS smoke run, plus synthetic
influence scores clearly marked SYNTHETIC everywhere they appear" (brief.md); PoC §2.7 p.5 (the surface to demonstrate), §9.2 p.19, §4.4 pp.10-11.
The generator (``vlapoc.synthetic``) is method code and never reads the answer key; the ground-truth stages (``score-clusters``, ``discrepancy``, the planted arms)
are the real stages, run afterwards on the synthetic scores.

What it builds, in a ``runs.scratch_lineage_copy`` of the real store (so versions, subsets and the catalog resolve; the real ``data/lineage/mlflow.db`` is untouched)::

    returns         the real ``returns`` stage on the two real SmolVLA harness evals (20 rollouts, real videos, 11 real failures)
    attribute x3    SYNTHETIC seed runs on v3/candidate (960 episodes) x 20 rollouts: noise plus planted structure, scores from the real
                    ``attribution.score_table`` and ``column_declarations``; outputs include the influence matrix and index
    ensemble        the real ``ensemble_run`` over them;  analyze, clusters, score-clusters, discrepancy: the real stages
    Round B         the real ``drop_cluster`` / planted-arm select runs, SYNTHETIC verdicts, the real ``confirm`` and ``discrepancy-verdict`` stages
    --with-smoke    the real chain of ``verify_training_attribution.py --local-smoke`` (train run on v0/reference, 2 rollouts, features, attribute) kept in the
                    store; it shows as SMOKE

Every run the builder creates is tagged ``vlapoc.synthetic=true`` except the smoke chain's. Run from a clean ``vlapoc/``+``scripts/`` tree (the stage runs record the git SHA)::

    .venv/bin/python scripts/dashboard_demo.py build --with-smoke [--force]
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

os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

DEMO = REPO_ROOT / "data" / "lineage_demo"
REAL_DB = REPO_ROOT / "data" / "lineage" / "mlflow.db"
HARNESS = ("eval-smolvla-harness-spatial0", "eval-smolvla-harness-l10t8")  # the two real SmolVLA evals with recorded trajectories (20 rollouts)
SMOKE_TRAIN_REF = "0a33bc75bb71433ca2def5220014dde0"  # the tracked 2-step SmolVLA train run (training-attribution S8)
TAG = "demo"


def db_sha() -> str:
    return hashlib.sha256(REAL_DB.read_bytes()).hexdigest()


def run_ids(lr: Path) -> set[str]:
    from vlapoc.runs import client, experiment_id

    return {r.info.run_id for r in client(lr).search_runs([experiment_id(lr)], max_results=50000)}


def synthetic_attribute_run(lr: Path, subset, table, ret, matrix, index: dict, name: str, eval_ids: list[str], returns_ids: list[str], out: Path, truth: dict, note: str) -> str:
    """Record a SYNTHETIC ``attribute`` run: the table, its real declarations, the matrix and its index, with params the real ensemble stage needs."""
    import mlflow  # noqa: F401
    import numpy as np

    from vlapoc import attribution as at
    from vlapoc.runs import stage_run

    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "attribution_scores.parquet", index=False)
    np.save(out / "rollout_episode_influence.npy", matrix)
    (out / "influence_index.json").write_text(json.dumps(index))
    decls = at.column_declarations(table, ret, True, "pm1", ("square", "square"))
    (out / "functionals.json").write_text(json.dumps(decls, indent=1, ensure_ascii=False))
    with stage_run("attribute", name, lr, extra_tags={"vlapoc.synthetic": "true"}) as r:
        r.input_subset(subset, "training", None)
        r.params({"synthetic": True, "method": "SYNTHETIC influence matrix (noise plus planted structure), real score_table; not an attribution result",
                  "planted_structure": note, "generator_truth": truth, "damping": 1e-3, "center_returns": True, "return_encoding": "pm1",
                  "train_output_fn": "square", "rollout_output_fn": "square", "eval_run_ids": eval_ids, "returns_run_ids": returns_ids, "n_rows": len(table),
                  "n_rollouts": int(matrix.shape[0])})
        r.output_file("attribution_scores", out / "attribution_scores.parquet", kind="score_table")
        r.output_file("functionals", out / "functionals.json", kind="functionals")
        r.output_file("influence_matrix", out / "rollout_episode_influence.npy", kind="file")
        r.output_file("influence_index", out / "influence_index.json", kind="file")
        r.tags({"attribute.n_rollouts": int(matrix.shape[0]), "attribute.n_episodes": len(table)})
        return r.run_id


def warm_videos(lr: Path, scores_run: str, cluster_run: str, extra_runs: list[str]) -> dict:
    """Pre-encode the clips the demo shows (cached under ``data/dashboard_cache/video``): the 20 rollouts, each rollout's top-10 and bottom-10 episodes, the
    top-3 members of every cluster, and the planted-vs-original examples."""
    import numpy as np
    import pandas as pd

    from vlapoc import results as res
    from vlapoc import video

    sid, cat = res.training_episodes(scores_run, lr)
    root = res.dataset_root(sid, lr)
    n, t0 = 0, time.time()

    def ep_clip(row):
        video.episode_video(root, int(row["episode_index"]), row["content_hash"])

    seed0 = res.seed_runs(scores_run, lr)[0]
    m, idx = res.influence(seed0, lr)
    ro = res.rollouts(seed0, lr)
    ep_by_id = cat.set_index("episode_id")
    want = set()
    for i in range(m.shape[0]):
        o = np.argsort(-m[i])
        want |= {idx["episode_id"][j] for j in o[:10]} | {idx["episode_id"][j] for j in o[::-1][:10]}
    mem = res.output_table(cluster_run, "members", lr)
    want |= set(mem[mem["rank"] <= 3]["episode_id"])
    for e in sorted(want):
        ep_clip(ep_by_id.loc[e])
        n += 1
    nroll = 0
    for _, r in ro.iterrows():
        tr = res.trajectory_root(r["eval_run_id"], lr)
        if tr is not None and pd.notna(r["traj_episode_index"]):
            video.episode_video(tr, int(r["traj_episode_index"]), f"{r['eval_run_id'][:16]}-{int(r['traj_episode_index'])}")
            nroll += 1
    pv = res.parent_version(sid, lr)
    n_orig = 0
    for er in extra_runs:  # discrepancy runs: examples.json holds corrupted/original pairs
        ex = res.output_json(er, "examples", lr)
        for e in ex["types"].values():
            for md in e["by_metric"].values():
                for k in ("most_harmful", "least_harmful"):
                    d = md[k]
                    row = cat[cat["episode_id"] == d["episode_id"]].iloc[0]
                    video.episode_video(root, int(row["episode_index"]), row["content_hash"])
                    if pv is not None:
                        proot, ptab = pv
                        pr = ptab[ptab["episode_id"] == d["original_episode_id"]].iloc[0]
                        video.episode_video(proot, int(pr["episode_index"]), pr["content_hash"])
                        n_orig += 1
    return {"training_episode_clips": n, "rollout_clips": nroll, "original_clips": n_orig, "seconds": round(time.time() - t0)}


def build(with_smoke: bool, force: bool) -> int:
    import numpy as np
    import pandas as pd

    from vlapoc import attribution as at
    from vlapoc import clusters as cl
    from vlapoc import confirm as cf
    from vlapoc import planted_arms as pa
    from vlapoc import recovery as rc
    from vlapoc import results as res
    from vlapoc import synthetic as syn
    from vlapoc.analysis import analyze_run
    from vlapoc.discrepancy import discrepancy_run
    from vlapoc.lineage import Subset
    from vlapoc.returns import returns_run
    from vlapoc.runs import client, scratch_lineage_copy, select_run, stage_run

    t0, sha0 = time.time(), db_sha()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "vlapoc", "scripts"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    if dirty:
        print(f"commit first: the stage runs record the git SHA and refuse a dirty tree:\n{dirty}")
        return 2
    if DEMO.exists():
        if not force:
            print(f"{DEMO} exists; pass --force to rebuild it")
            return 2
        old_returns = None
        try:  # the smoke chain keeps files outside the store (stage.py writes eval and attribution outputs under data/); remove only those the builder named itself.
            import sqlite3  # plain sqlite, not MLflow: MLflow caches an engine per store path, which would break the store re-created below

            con = sqlite3.connect(DEMO / "mlflow.db")
            row = con.execute("select p.value from params p join runs r on r.run_uuid = p.run_uuid where r.name = 'returns-demo-smoke' and p.key = 'eval_run_id'").fetchone()
            con.close()
            old_returns = REPO_ROOT / "data" / "eval" / f"returns-{row[0][:12]}" if row else None
        except Exception:  # noqa: BLE001 - an incomplete earlier build has no such run
            pass
        for d in (REPO_ROOT / "data" / "eval" / f"eval-{TAG}-smoke", REPO_ROOT / "data" / "attribution" / f"attribute-{TAG}-smoke", old_returns):
            if d is not None and d.exists():
                shutil.rmtree(d)
        shutil.rmtree(DEMO)
    lr = scratch_lineage_copy(DEMO)
    out = DEMO / "outputs"
    before = run_ids(lr)
    manifest: dict = {"SYNTHETIC": "scores, clusters, discrepancy readings and Round B verdicts in this store are synthetic; videos, catalog and rollouts are real"}
    # --- real returns of the two real harness evals ---
    c = client(lr)
    ev_ids = [res.resolve_run(n, "eval", lr) for n in HARNESS]
    ret_dirs = [REPO_ROOT / "data" / "eval" / f"returns-{e[:12]}" for e in ev_ids]
    files_before = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for d in ret_dirs if d.exists() for f in d.iterdir()}
    rt_ids = [returns_run(e, f"returns-{TAG}-{n.removeprefix('eval-smolvla-harness-')}", lineage_root=lr)["run_id"] for e, n in zip(ev_ids, HARNESS)]
    files_after = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for d in ret_dirs if d.exists() for f in d.iterdir()}
    manifest["returns_files_unchanged"] = all(files_after.get(k) == v for k, v in files_before.items())
    ret = pd.concat([res._fetch(c.get_run(r), "returns", lr, pd.read_parquet) for r in rt_ids], ignore_index=True)
    print(f"real rollouts: {len(ret)}; failure types {ret['failure_type'].value_counts().to_dict()}; returns files unchanged: {manifest['returns_files_unchanged']}")
    # --- SYNTHETIC matrices through open_for_method ---
    v3 = Subset.load("v3/candidate", lr)
    view, desc, codes, specs = syn.demo_inputs("v3/candidate", lr)
    cells = syn.plant_cells(desc, codes, specs)
    episodes = v3.episodes(lr).sort_values("episode_index").reset_index(drop=True)
    assert episodes["episode_id"].tolist() == view.catalog["episode_id"].tolist()
    note = syn.PLANTED_TEXT + f" Planted: HRT-harm = {cells['text']['hrt_harm']} (n {cells['n']['hrt_harm']}); F30-harm = {cells['text']['f30_harm']} (n {cells['n']['f30_harm']}); F30-boost = {cells['text']['f30_boost']} (n {cells['n']['f30_boost']})."
    manifest["planted"] = {"text": cells["text"], "n": cells["n"], "description": note}
    seed_ids = []
    for s in range(3):
        M, truth = syn.synthetic_matrix(ret, cells["masks"], seed=s, n_episodes=len(episodes))
        table = at.score_table(M, episodes, ret, True, "pm1")
        index = {"rows": "rollout_id, row order of the SYNTHETIC matrix", "columns": "episode_id and content_hash, the score table's row order",
                 "rollout_id": ret["rollout_id"].tolist(), "episode_id": table["episode_id"].tolist(), "content_hash": table["content_hash"].tolist()}
        seed_ids.append(synthetic_attribute_run(lr, v3, table, ret, M, index, f"attribute-{TAG}-s{s}", ev_ids, rt_ids, out / f"attribute-{TAG}-s{s}", truth, note))
        print(f"synthetic seed {s}: matrix {M.shape}, table {table.shape}")
    ens = at.ensemble_run(seed_ids, f"attribute-{TAG}-ens", out_dir=out / f"attribute-{TAG}-ens", lineage_root=lr)["run_id"]
    an = analyze_run(ens, f"analyze-{TAG}", out_dir=out / f"analyze-{TAG}", lineage_root=lr, synthetic=True)
    cr = cl.clusters_run(ens, f"clusters-{TAG}", out_dir=out / f"clusters-{TAG}", lineage_root=lr, synthetic=True, params={"min_sup_fraction": 0.01})  # as the Round A chain runs it (min_sup 10)
    sc = rc.cluster_recovery_run(cr["run_id"], "v3", f"score-clusters-{TAG}", out_dir=out / f"score-clusters-{TAG}", lineage_root=lr)
    dr = discrepancy_run(ens, "v3", f"discrepancy-{TAG}", out_dir=out / f"discrepancy-{TAG}", lineage_root=lr)
    print(f"stages: analyze {an['run_id'][:8]}, clusters {cr['n_clusters']} ({cr['n_by_lens']}), score-clusters {sc['n_clusters']}, discrepancy {dr['run_id'][:8]}")
    # --- Round B for 3 picked clusters: real select runs, SYNTHETIC verdicts, the real confirm stage ---
    clu = res.output_table(cr["run_id"], "clusters", lr)
    picks = [cid for cid in ("slice:hrt60_lin:harm:1", "slice:f30:boost:1") if cid in set(clu["cluster_id"])]
    extra = clu[(clu["lens"] == "slice") & (clu["direction"] == "harm") & ~clu["cluster_id"].isin(picks)].sort_values("n", ascending=False)
    picks += [extra.iloc[0]["cluster_id"]] if len(extra) else []
    chain = f"{TAG}-rb000000"
    arms = {}
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from verify_dashboard import add_verify, verdict_json  # SYNTHETIC verdict builders (verifier helpers; a demo, not a result)

    outcomes = ["confirmed", "not confirmed", "refuted"]
    for i, cid in enumerate(picks):
        sel_id, child = select_run("v3/candidate", f"select-{TAG}-cluster{i}", "drop_cluster", lineage_root=lr, cluster_run_id=cr["run_id"], cluster_id=cid, chain=chain)
        exp = json.loads(c.get_run(sel_id).data.params["expected_by_metric"])
        arms[f"cluster{i}"] = (cid, exp)
        for sd in range(3):
            only = {}
            for m_, d in exp.items():
                want = "T" if d == "treated_better" else "C"
                other = "C" if want == "T" else "T"
                only[m_] = [want, want, want][sd] if outcomes[i % 3] == "confirmed" else ([want, None, None][sd] if outcomes[i % 3] == "not confirmed" else [other, other, None][sd])
            add_verify(lr, DEMO / "verdicts", f"v-{TAG}-cluster{i}-s{sd}", "vs_base", f"cluster{i}", "base", sd, verdict_json(None, only=only), chain)
    cf_out = cf.confirm_run(cr["run_id"], TAG, f"confirm-{TAG}", chain=chain, out_dir=out / f"confirm-{TAG}", lineage_root=lr)
    print(f"confirm: {cf_out['overall']}")
    # --- Round B planted: the real ground-truth-side select runs, SYNTHETIC verdicts, the real discrepancy-verdict stage ---
    pchain, ptag = "demop-rb000000", "demop"
    parms = {"type-idle_padding": ("type", "idle_padding"), "type-time_shift": ("type", "time_shift"), "type-detour": ("type", "detour"), "cleanharm-f30": ("cleanharm", ""), "oracle": ("oracle", "")}
    only_of = {"type-idle_padding": {"hrt60": "T", "rmst60": "T"}, "type-time_shift": {"f30": "C"}, "type-detour": {}, "cleanharm-f30": {"f30": "T"},
               "oracle": {"hrt60": "T", "rmst60": "T", "f30": "T", "censored60": "T"}}
    for arm, (kind, ty) in parms.items():
        pa.planted_select_run("v3/candidate", f"select-{ptag}-{arm}", kind, corruption_type=ty or None, scores_run_id=ens if kind == "cleanharm" else None, metric="f30" if kind == "cleanharm" else None, scope="all",
                              lineage_root=lr, chain=pchain)
        ctrl = "cleanrandom91-0" if arm == "oracle" else "cleanrandom0"
        for sd in range(3):
            add_verify(lr, DEMO / "verdicts", f"v-{ptag}-{arm}-s{sd}", "curated_vs_random", arm, ctrl, sd, verdict_json(None, only=only_of[arm]), pchain)
    dv = pa.discrepancy_verdict_run(ptag, f"discrepancy-verdict-{ptag}", out_dir=out / f"discrepancy-verdict-{ptag}", lineage_root=lr, synthetic=True)
    print(f"discrepancy-verdict: {dv['rows']} rows, positive control {dv['positive_control']}")
    # --- tag everything built so far SYNTHETIC (the real stages have no tag option) ---
    mid = run_ids(lr) - before
    for rid in mid:
        if c.get_run(rid).data.tags.get("vlapoc.synthetic") != "true":
            c.set_tag(rid, "vlapoc.synthetic", "true")
    manifest["synthetic_runs"] = len(mid)
    # --- the real smoke chain, kept ---
    smoke = {}
    if with_smoke:
        from verify_training_attribution import stage_cli

        feats = DEMO / "feats"
        chn, T = f"{TAG}-smoke0", f"{TAG}-smoke"
        steps = [("eval", ["eval", "--train", SMOKE_TRAIN_REF, "--name", f"eval-{T}", "--smoke", "--tasks", "libero_spatial:0", "--n", "2", "--batch-size", "2", "--device", "mps"]),
                 ("returns", ["returns", "--eval", f"eval-{T}", "--name", f"returns-{T}"]),
                 ("features-train", ["features", "--train", SMOKE_TRAIN_REF, "--which", "train", "--train-stride", "100000", "--batch-size", "8", "--workers", "0", "--device", "mps"]),
                 ("features-rollouts", ["features", "--train", SMOKE_TRAIN_REF, "--which", "rollouts", "--eval", f"eval-{T}", "--rollout-stride", "150", "--batch-size", "8", "--workers", "0", "--device", "mps"]),
                 ("attribute", ["attribute", "--train", SMOKE_TRAIN_REF, "--eval", f"eval-{T}", "--returns", f"returns-{T}", "--name", f"attribute-{T}", "--device", "cpu"])]
        for name, args in steps:
            t1 = time.time()
            smoke[name] = stage_cli(args, lr, feats, chn)
            print(f"[ok] smoke {name} {time.time() - t1:.0f}s", flush=True)
        manifest["smoke"] = {"chain": chn, "attribute_run": smoke["attribute"]["run_id"], "attribute_name": f"attribute-{T}", "n_rollouts": smoke["attribute"]["n_rollouts"], "train_run": SMOKE_TRAIN_REF}
    # --- pre-warm the clips, record the manifest ---
    manifest["warm"] = warm_videos(lr, ens, cr["run_id"], [dr["run_id"]])
    manifest |= {"scores_run": f"attribute-{TAG}-ens", "scores_run_id": ens, "clusters_run": f"clusters-{TAG}", "clusters_run_id": cr["run_id"], "seed_runs": seed_ids,
                 "picked_clusters": {a: v[0] for a, v in arms.items()}, "confirm_overall": cf_out["overall"], "mlflow_db_sha256_before": sha0, "mlflow_db_sha256_after": db_sha(),
                 "seconds": round(time.time() - t0)}
    (DEMO / "demo_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("planted",)}, indent=1, default=str)[:2500])
    return 0 if db_sha() == sha0 else 1


def main() -> int:
    """Command line: ``build [--with-smoke] [--force]`` (module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--with-smoke", action="store_true", help="also run the real 2-step SmolVLA chain on this Mac (MPS, about 8 minutes) and keep it as a SMOKE run")
    b.add_argument("--force", action="store_true", help="replace an existing data/lineage_demo")
    a = ap.parse_args()
    return build(a.with_smoke, a.force)


if __name__ == "__main__":
    sys.exit(main())
