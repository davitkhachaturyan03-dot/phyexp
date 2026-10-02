"""Verify S5 (every stage as an MLflow run; trace / consumers / diff; check_scores) against AC5.1-AC5.6.

Runs in an isolated scratch lineage root (``data/lineage_scratch/s5``: copies of the version, subset and
split manifests, its own ``mlflow.db``), so these tiny test runs never mix with the real chain (S6).

Grounding: PoC §2.1 p.3 (tie-in list; "what changed between the good run and the bad run?"), §2.3 p.4
(seeds), §9.1-9.2 pp.18-19 (eval tied to checkpoint; screening vs attribution keys).

    .venv/bin/python scripts/verify_runs_s5.py          # AC5.1-AC5.6 (round 1; 4 tiny training runs)
    .venv/bin/python scripts/verify_runs_s5.py --r2     # AC5.8-AC5.10 (round 2; selections only, no training)

Both record MLflow runs (in scratch DBs), so both refuse to start while ``vlapoc/`` or ``scripts/`` has
uncommitted changes. The round-1 mode leaves ``data/lineage_scratch/s5`` (~2.3 GB) for inspection
and replaces it on the next run; ``--r2`` uses a ``data/scratch/`` dir and deletes it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from vlapoc.catalog import sha256_file  # noqa: E402
from vlapoc.derive import load_labels  # noqa: E402
from vlapoc.lineage import LINEAGE_ROOT, LineageError, Subset, resolve_version, version_episodes  # noqa: E402
from vlapoc import runs  # noqa: E402

FEATURE_DIR = REPO_ROOT / "tracker" / "features" / "2026-09-28-data-lineage-layer"
LR = REPO_ROOT / "data" / "lineage_scratch" / "s5"
RUNS = LR / "runs"


def setup() -> None:
    """(Re)create the round-1 scratch lineage root from copies of the real manifests (no MLflow DB)."""
    if LR.exists():
        shutil.rmtree(LR)
    LR.mkdir(parents=True)
    for d in ("versions", "subsets", "splits"):
        shutil.copytree(LINEAGE_ROOT / d, LR / d)
    shutil.copy(LINEAGE_ROOT / "names.json", LR / "names.json")


def main() -> int:
    """Round-1 checks AC5.1-AC5.6: build, split, 4 tiny train runs, a select, an eval (real since evaluation-layer S5), queries."""
    setup()
    s: dict = {}
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    print(f"scratch lineage root: {LR.relative_to(REPO_ROOT)}; tracking uri scheme: {runs.tracking_uri(LR).split(':')[0]}; HEAD {head[:12]}")

    # upstream runs so trace() can reach raw files
    b = runs.build_run("v0", lineage_root=LR, verify_raw=False)
    sp, _ = runs.split_run("v0", 2, 0, lineage_root=LR)
    ref = Subset.load("v0/reference", LR)
    # run A: train on v0/reference; select: drop 5 random; run B: train on the child
    a = runs.train_run("v0/reference", "s5-train-A", seed=0, steps=2, batch_size=2, runs_root=RUNS, lineage_root=LR)
    sel, child = runs.select_run("v0/reference", "s5-select-random5", "drop_random_k", k=5, seed=0, lineage_root=LR)
    bb = runs.train_run(child.subset_id, "s5-train-B", seed=0, steps=2, batch_size=2, runs_root=RUNS, lineage_root=LR)
    a1 = runs.train_run("v0/reference", "s5-train-A-seed1", seed=1, steps=2, batch_size=2, runs_root=RUNS, lineage_root=LR)
    # eval_standin was replaced by the real eval stage (evaluation-layer S5): one 20-step rollout, smoke scale
    ev_out = runs.eval_run(f"train:{a['run_id']}", [("libero_spatial", 0)], 1, "s5-eval-A", out_dir=RUNS / "s5-eval-A",
                           tau_episode_steps=20, lineage_root=LR)
    ev, plan = ev_out["run_id"], REPO_ROOT / ev_out["rollouts"]
    c = runs.client(LR)
    exp = runs.experiment_id(LR)

    # ---- AC5.1 ------------------------------------------------------------------------------------
    allr = c.search_runs([exp])
    stages = sorted({r.data.tags["stage"] for r in allr})
    found = c.search_runs([exp], filter_string=f"datasets.digest = '{ref.subset_id[:32]}'")
    names_found = sorted(r.info.run_name for r in found)
    ra = c.get_run(a["run_id"])
    di = [d for d in ra.inputs.dataset_inputs][0]
    print(f"AC5.1 runs: {len(allr)}, stages {stages} (all in {runs.STAGES}: {set(stages) <= set(runs.STAGES)})")
    print(f"AC5.1 train-A input: name={di.dataset.name} digest={di.dataset.digest} source={di.dataset.source} "
          f"tags={ {t.key: t.value[:16] for t in di.tags} }")
    print(f"AC5.1 search_runs(datasets.digest = '{ref.subset_id[:32]}') -> {names_found}")
    print(f"AC5.1 eval output tags: output.rollouts.sha256={c.get_run(ev).data.tags['output.rollouts.sha256'][:16]}... "
          f"== sha256(file): {c.get_run(ev).data.tags['output.rollouts.sha256'] == sha256_file(plan)}")
    s["AC5.1"] = {"runs": len(allr), "stages": stages, "digest_search": names_found}

    # ---- AC5.2 ------------------------------------------------------------------------------------
    p, t = ra.data.params, ra.data.tags
    arts = [x.path for x in c.list_artifacts(a["run_id"])] + [x.path for x in c.list_artifacts(a["run_id"], "checkpoint")] \
        + [x.path for x in c.list_artifacts(a["run_id"], "subset")]
    ck = REPO_ROOT / t["checkpoint.path"]
    norm = sorted(ck.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))[0]
    fields = {
        "dataset_version_id": p["dataset_version_id"][:12], "subset_id": p["subset_id"][:12],
        "episode_id_list_artifact": [x for x in arts if x.startswith("subset/")],
        "mixture_weights": p["mixture_weights"], "augmentation": p["augmentation"],
        "train_config_artifacts": [x for x in arts if x.endswith(".json") and "subset/" not in x],
        "git_sha": t["mlflow.source.git.commit"][:12], "git_sha_is_HEAD": t["mlflow.source.git.commit"] == head,
        "git_dirty": t["vlapoc.git_dirty"], "seed": p["seed"], "checkpoint.path": t["checkpoint.path"],
        "weights_sha256_matches_disk": t["checkpoint.weights.sha256"] == sha256_file(ck / "model.safetensors"),
        "normalizer_sha256_matches_disk": t["checkpoint.normalizer_stats.sha256"] == sha256_file(norm),
        "stats_json_sha256": t["dataset.stats_json.sha256"][:12], "episodes_match": t["checkpoint.train_config.episodes_match_subset"],
        "normalization_stats": p["normalization_stats"][:60] + "...",
    }
    for k, v in fields.items():
        print(f"AC5.2 {k}: {v}")
    s["AC5.2"] = fields

    # ---- AC5.3 ------------------------------------------------------------------------------------
    re_ = c.get_run(ev)
    ep = re_.data.params
    print(f"AC5.3 eval run: checkpoint_sha256 == train-A weights: {ep['checkpoint_sha256'] == a['weights_sha256']}; "
          f"train_run_id == A: {ep['train_run_id'] == a['run_id']}; suites={ep['suites']}; libero_task_ids={ep['libero_task_ids']}; "
          f"init_state_indices={ep['init_state_indices']}; seeds={ep['seeds']}; loaded={re_.data.tags['eval.checkpoint.loaded']} "
          f"({re_.data.tags['eval.checkpoint.n_params']} params); stand_in={re_.data.tags['vlapoc.stand_in']}")
    pl = pd.read_parquet(plan)
    print(f"AC5.3 rollout plan rows={len(pl)} outcomes={pl['outcome'].unique().tolist()} columns={list(pl.columns)}")
    s["AC5.3"] = {"ckpt_match": ep["checkpoint_sha256"] == a["weights_sha256"], "train_run": ep["train_run_id"] == a["run_id"],
                  "plan_rows": len(pl), "outcomes": pl["outcome"].unique().tolist()}

    # ---- AC5.4 ------------------------------------------------------------------------------------
    tr = runs.trace(bb["run_id"], LR)
    chain = [f"{v['stage']}:{v['name']}" for v in tr["runs"].values()]
    print(f"AC5.4 trace(s5-train-B) runs: {chain}; raw files reached: {len(tr['raw_files'] or [])}")
    dropped = child.derivation["dropped"][0]
    h = version_episodes("v0", LR).set_index("episode_id").loc[dropped, "content_hash"]
    cons = runs.consumers(dropped, h, LR)
    cnames = [x["name"] for x in cons]
    print(f"AC5.4 consumers({dropped.split('/')[-2][:30]}.../{dropped.split('/')[-1]}@{h[:12]}): {cnames}; "
          f"lists train-A: {'s5-train-A' in cnames}; lists train-B (dropped it): {'s5-train-B' in cnames}")
    d_ab = runs.diff(a["run_id"], bb["run_id"], LR)
    d_seed = runs.diff(a["run_id"], a1["run_id"], LR)
    print(f"AC5.4 diff(A, B): removed {d_ab['data']['n_removed']}, added {d_ab['data']['n_added']}, changed_version "
          f"{d_ab['data']['n_changed_version']}; params {list(d_ab['params'])}; config {list(d_ab['config'])}; git same {d_ab['git']['same']}")
    print(f"AC5.4 diff(A, A-seed1): data removed/added/changed {d_seed['data']['n_removed']}/{d_seed['data']['n_added']}/"
          f"{d_seed['data']['n_changed_version']}; params {d_seed['params']}; config {d_seed['config']}")
    s["AC5.4"] = {"trace": chain, "raw_files": len(tr["raw_files"] or []), "consumers": cnames,
                  "diff_ab": {k: d_ab["data"][k] for k in ("n_removed", "n_added", "n_changed_version")},
                  "diff_seed_params": d_seed["params"], "diff_seed_config": d_seed["config"]}

    # ---- AC5.5 check_scores ---------------------------------------------------------------------
    labels = load_labels("v1", LR)
    corrupted = labels["episode_id"].tolist()[:6]
    ref_ids = list(Subset.load("v1/reference", LR).episode_ids[:4])
    mix = Subset(resolve_version("v1", LR)["dataset_version_id"], tuple(corrupted + ref_ids), {"op": "manual", "note": "AC5.5 test"})
    mix.save(LR)
    rv1 = runs.train_run(mix.subset_id, "s5-train-v1-mix", seed=0, steps=2, batch_size=2, runs_root=RUNS, lineage_root=LR)
    good = mix.keys(LR)
    ok = runs.check_scores(good.assign(score=0.0), rv1["run_id"], LR)
    e0 = version_episodes("v0", LR).set_index("episode_id")
    stale = good.copy()
    stale["content_hash"] = e0.loc[stale["episode_id"], "content_hash"].to_numpy()  # clean v0 hashes
    try:
        runs.check_scores(stale, rv1["run_id"], LR)
        rejected = None
    except LineageError as e:
        rejected = str(e)
    print(f"AC5.5 run trained on v1 subset (6 corrupted + 4 reference episodes): v1-keyed table accepted {ok}")
    print(f"AC5.5 same episodes keyed by clean v0 hashes -> rejected: {rejected}")
    s["AC5.5"] = {"accepted": ok, "rejected": rejected}

    # ---- AC5.6 ------------------------------------------------------------------------------------
    s["AC5.6"] = {"tracking_uri_scheme": runs.tracking_uri(LR).split(":")[0],
                  "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"), "wandb_installed": shutil.which("wandb") is not None}
    print(f"AC5.6 tracking uri scheme={s['AC5.6']['tracking_uri_scheme']}; HF_HUB_OFFLINE={s['AC5.6']['hf_hub_offline']}; "
          f"wandb CLI installed={s['AC5.6']['wandb_installed']}")
    s["run_ids"] = {"build": b, "split": sp, "A": a["run_id"], "select": sel, "B": bb["run_id"], "A_seed1": a1["run_id"],
                    "eval": ev, "v1mix": rv1["run_id"]}
    (FEATURE_DIR / "artifacts" / "s5_summary.json").write_text(json.dumps(s, indent=2, default=str))
    (FEATURE_DIR / "artifacts" / "s5_diff_A_B.json").write_text(json.dumps(d_ab, indent=2, default=str))
    print("summary -> tracker/features/2026-09-28-data-lineage-layer/artifacts/s5_summary.json")
    return 0


def r2() -> int:
    """Round-2 checks AC5.8-AC5.10: selections only, no training, in a scratch lineage root.

    The scratch root holds copies of the real manifests and of the real ``mlflow.db`` (so ``trace`` can
    reach v0 through the recorded build/split/derive runs); on the copy only, the experiment's
    ``artifact_location`` is repointed to the scratch dir so nothing can be written into the real
    ``data/lineage/mlartifacts``. The real DB's sha256 and the real artifact tree are compared before/after.
    """
    import hashlib
    import sqlite3
    import tempfile

    import numpy as np

    from vlapoc.lineage import drop_random_k

    real_db = LINEAGE_ROOT / "mlflow.db"
    db_sha_before = hashlib.sha256(real_db.read_bytes()).hexdigest()
    art_before = sorted(str(p.relative_to(LINEAGE_ROOT)) for p in (LINEAGE_ROOT / "mlartifacts").rglob("*"))
    scratch_root = REPO_ROOT / "data" / "scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    base = Path(tempfile.mkdtemp(prefix="verify_s5_r2_", dir=scratch_root))
    lr = base / "lineage"
    lr.mkdir()
    for d in ("versions", "subsets", "splits"):
        shutil.copytree(LINEAGE_ROOT / d, lr / d)
    shutil.copy(LINEAGE_ROOT / "names.json", lr / "names.json")
    shutil.copy(real_db, lr / "mlflow.db")
    (lr / "mlartifacts").mkdir()
    con = sqlite3.connect(lr / "mlflow.db")
    con.execute("UPDATE experiments SET artifact_location = ? WHERE name = ?", ((lr / "mlartifacts").resolve().as_uri(), runs.EXPERIMENT))
    con.commit()
    con.close()
    print(f"scratch lineage root {lr.relative_to(REPO_ROOT)} (copy of real manifests + mlflow.db; artifact location repointed)")
    s: dict = {}
    try:
        c = runs.client(lr)
        exp = runs.experiment_id(lr)
        n0 = len(c.search_runs([exp]))
        attr = [r for r in c.search_runs([exp], filter_string="tags.stage = 'attribute'") if r.info.status == "FINISHED"][0]
        score_path = REPO_ROOT / "data" / "runs" / attr.data.tags["vlapoc.chain"] / "attribute" / attr.data.tags["output.attribution_scores.artifact"]
        table = pd.read_parquet(score_path)
        v1c = Subset.load("v1/candidate", lr)

        # ---- AC5.8 direction ---------------------------------------------------------------
        rid_low, low = runs.select_run("v1/candidate", "r2-select-drop-lowest-20", "drop_top_k", 20, score_path=score_path,
                                       metric="placeholder_score", score_run_id=attr.info.run_id, direction="lowest", lineage_root=lr)
        rid_high, high = runs.select_run("v1/candidate", "r2-select-drop-highest-20", "drop_top_k", 20, score_path=score_path,
                                         metric="placeholder_score", score_run_id=attr.info.run_id, lineage_root=lr)
        exp_low = sorted(table.sort_values(["placeholder_score", "episode_id"], ascending=[True, True]).head(20)["episode_id"])
        exp_high = sorted(table.sort_values(["placeholder_score", "episode_id"], ascending=[False, True]).head(20)["episode_id"])
        pl, ph = c.get_run(rid_low).data.params, c.get_run(rid_high).data.params
        s6_child = [r for r in c.search_runs([exp], filter_string="attributes.run_name = 'select-drop-top-20'")][0].data.tags["output.subset"].split(":")[1]
        print(f"AC5.8 lowest: dropped == 20 lowest placeholder scores (tie-break episode_id asc): {low.derivation['dropped'] == exp_low}; "
              f"derivation direction={low.derivation['direction']!r}; run params direction={pl['direction']!r} metric={pl['metric']!r} k={pl['k']}")
        print(f"AC5.8 default (highest): dropped == 20 highest: {high.derivation['dropped'] == exp_high}; params direction={ph['direction']!r}; "
              f"child id == S6 select-drop-top-20 child ({s6_child[:12]}): {high.subset_id == s6_child}; low/high overlap: "
              f"{len(set(low.derivation['dropped']) & set(high.derivation['dropped']))}")
        try:
            runs.select_run("v1/candidate", "bad", "drop_top_k", 20, score_path=score_path, metric="placeholder_score", direction="up", lineage_root=lr)
            bad = "NOT RAISED"
        except ValueError as e:
            bad = str(e)
        print(f"AC5.8 invalid direction rejected before any run starts: {bad!r}; runs added so far: {len(c.search_runs([exp])) - n0}")
        s["AC5.8"] = {"lowest_ok": low.derivation["dropped"] == exp_low, "derivation_direction": low.derivation["direction"],
                      "param_direction": pl["direction"], "highest_ok": high.derivation["dropped"] == exp_high,
                      "highest_equals_s6_child": high.subset_id == s6_child, "run_low": rid_low, "run_high": rid_high}

        # ---- AC5.9 drop_group as a recorded selection ---------------------------------------------
        e1 = version_episodes("v1", lr)
        task = sorted(e1.loc[e1["suite"] == "libero_10", "task_name"].unique())[0]
        rid_g, g = runs.select_run("v1/candidate", "r2-select-leave-one-task-out", "drop_group", column="task_name", value=task, lineage_root=lr)
        rg = c.get_run(rid_g)
        tr = runs.trace(rid_g, lr)
        chain = [f"{v['stage']}:{v['name']}" for v in tr["runs"].values()]
        print(f"AC5.9 drop_group(task_name={task[:40]}...): child n={len(g.episode_ids)} (960 - 48); params op={rg.data.params['op']!r} "
              f"column={rg.data.params['column']!r} value={rg.data.params['value'][:40]!r}... n_dropped={rg.data.params['n_dropped']}; "
              f"select.dropped tag present: {'select.dropped' in rg.data.tags}; produced subset tag: "
              f"{rg.data.tags['output.subset'][:20]}...; == child: {rg.data.tags['output.subset'] == 'subset:' + g.subset_id}")
        print(f"AC5.9 trace(leave-one-task-out select) runs: {chain}; raw files reached: {len(tr['raw_files'] or [])}; "
              f"every edge has one producer: {all(len(e['produced_by']) == 1 for e in tr['edges'])}")
        s["AC5.9"] = {"n": len(g.episode_ids), "params": {k: rg.data.params[k] for k in ("op", "column", "value", "n_dropped")},
                      "trace": chain, "raw_files": len(tr["raw_files"] or []), "run": rid_g}

        # ---- AC5.10 random subsets for datamodels ------------------------------------------------
        N, n = len(v1c.episode_ids), 480
        rand = []
        for seed in (0, 1, 2):
            rid, child = runs.select_run("v1/candidate", f"r2-select-random-subset-n{n}-s{seed}", "drop_random_k", N - n, seed=seed, lineage_root=lr)
            again = drop_random_k(v1c, N - n, seed, lineage_root=lr)  # reproduce from the seed alone
            rp = c.get_run(rid).data.params
            rand.append((rid, child, again.subset_id == child.subset_id, rp))
            print(f"AC5.10 seed {seed}: select run {rid[:12]} k={rp['k']} subset_size={rp['subset_size']} -> subset {child.subset_id[:12]} "
                  f"n={len(child.episode_ids)}; reproduced from seed: {again.subset_id == child.subset_id}")
        ids = [ch.subset_id for _, ch, _, _ in rand]
        cols = list(v1c.episode_ids)
        records = [json.loads((lr / "subsets" / f"{i}.json").read_text()) for i in ids]  # subset records alone
        M = np.array([[1 if e in set(r["episode_ids"]) else 0 for e in cols] for r in records], dtype=np.int8)
        print(f"AC5.10 distinct subset ids: {len(set(ids))}/3; membership matrix from subset records: shape {M.shape}, "
              f"row sums {M.sum(1).tolist()}, episodes in 0/1/2/3 subsets: {np.bincount(M.sum(0), minlength=4).tolist()}")
        pd.DataFrame(M.T, index=cols, columns=[f"seed{sd}:{i[:12]}" for sd, i in zip((0, 1, 2), ids)]).rename_axis("episode_id") \
            .to_csv(FEATURE_DIR / "artifacts" / "s5_r2_membership_matrix.csv")
        s["AC5.10"] = {"subset_ids": ids, "distinct": len(set(ids)), "sizes": [len(ch.episode_ids) for _, ch, _, _ in rand],
                       "reproducible": [r for _, _, r, _ in rand], "matrix_shape": list(M.shape), "row_sums": M.sum(1).tolist()}
        s["runs_added_to_scratch_db"] = len(c.search_runs([exp])) - n0
    finally:
        shutil.rmtree(base, ignore_errors=True)
    db_sha_after = hashlib.sha256(real_db.read_bytes()).hexdigest()
    art_after = sorted(str(p.relative_to(LINEAGE_ROOT)) for p in (LINEAGE_ROOT / "mlartifacts").rglob("*"))
    print(f"runs added to the scratch DB: {s['runs_added_to_scratch_db']}; real mlflow.db sha256 unchanged: {db_sha_before == db_sha_after} "
          f"({db_sha_after[:12]}); real mlartifacts tree unchanged: {art_before == art_after} ({len(art_after)} paths); scratch removed: {not base.exists()}")
    s["real_db_unchanged"] = db_sha_before == db_sha_after
    s["real_artifacts_unchanged"] = art_before == art_after
    (FEATURE_DIR / "artifacts" / "s5_r2_summary.json").write_text(json.dumps(s, indent=2, default=str))
    print("summary -> tracker/features/2026-09-28-data-lineage-layer/artifacts/s5_r2_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(r2() if "--r2" in sys.argv[1:] else main())
