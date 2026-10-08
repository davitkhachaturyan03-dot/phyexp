"""Lineage dry run (data+lineage S6): every downstream PoC stage type, end to end, on the real v0/v1 data.

Records this chain as MLflow runs (PoC §9.1 pp.18-19 Steps 1-4, §2.6 p.5, §10 p.20):

    build(v0) -> split(v0) -> derive(v1) -> screen [stand-in] -> train-full (v1/candidate, seed 0 and 1)
    -> eval (seed 0) -> attribute [stand-in] -> select drop-top-k / drop-random-k (same k)
    -> train on each -> eval on each

build/split/derive are reused when a run that produced v0 / v0's split / v1 already exists (they are
content-addressed and were recorded by the S4 commands). Train steps are real LeRobot training runs
(ACT, a few steps, MPS, wandb off, per-subset normalization since evaluation-layer S8). Eval steps are real
LIBERO rollouts at smoke scale since evaluation-layer S5 (libero_spatial task 0, ``--eval-n`` init states,
``--eval-tau`` steps), named "[smoke]" (their ``vlapoc.smoke`` tag stays ``unspecified``). Stand-ins carry ``vlapoc.stand_in=true`` and produce only
lineage keys plus placeholder payloads: screening scores are NaN, attribution scores are a hash of the key.
No outcome numbers are invented.

    .venv/bin/python scripts/lineage_dry_run.py --k 20 --seed 0      # full chain: records runs in data/lineage
    .venv/bin/python scripts/lineage_dry_run.py --tamper-only        # AC6.5 only; records no run
    .venv/bin/python scripts/lineage_dry_run.py --eval-smoke         # evaluation-layer S9 chain (see ``eval_smoke``)

The full chain writes real runs (and ~2.3 GB of checkpoints under ``data/runs/``), refuses to start
with uncommitted changes under ``vlapoc/`` or ``scripts/``, and overwrites the S6 files in the
feature's ``artifacts/``. Its order matters: attribute runs before select, because select consumes
attribute's score table, which is what makes ``trace(train-drop-top-k)`` reach the full training run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from vlapoc import runs  # noqa: E402
from vlapoc.derive import load_labels  # noqa: E402
from vlapoc.lineage import (  # noqa: E402
    LINEAGE_ROOT,
    LineageError,
    Subset,
    VersionIntegrityError,
    resolve_version,
    version_episodes,
)

FEATURE_DIR = REPO_ROOT / "tracker" / "features" / "2026-09-28-data-lineage-layer"
ART = FEATURE_DIR / "artifacts"  # the full chain's evidence (data+lineage feature)
EVAL_FEATURE_ART = REPO_ROOT / "tracker" / "features" / "2026-10-01-evaluation-layer" / "artifacts"  # --eval-smoke's evidence
SCRATCH_ROOT = REPO_ROOT / "data" / "scratch"  # gitignored (data/); never a session-bound path (AC3.9)
EVAL_TASKS = [("libero_spatial", 0)]  # smoke-scale evals in the dry run (evaluation-layer S5); full protocol: server runbook


def log(msg: str) -> None:
    """Print and flush (the chain's progress lines are read live from long runs)."""
    print(msg, flush=True)


def tamper_check(v1: dict, steps: int = 5, chain: str | None = None) -> dict:
    """AC6.5: flip one byte in a clone of v1 and show that ``train_run`` refuses to launch.

    The clone lives in a fresh ``tempfile.mkdtemp`` dir under ``data/scratch/`` and is deleted afterwards.
    A refused launch creates no MLflow run, so the main DB only gets read.
    """
    c = runs.client()
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="tamper_", dir=SCRATCH_ROOT))
    copy = scratch / "v1_tamper_copy"
    try:
        subprocess.run(["cp", "-c", "-R", str(REPO_ROOT / v1["lerobot_root"]), str(copy)], check=True)
        target = sorted((copy / "data").rglob("*.parquet"))[-1]
        size = target.stat().st_size
        with open(target, "r+b") as f:
            f.seek(size // 3)
            byte = f.read(1)
            f.seek(size // 3)
            f.write(bytes([byte[0] ^ 0xFF]))
        n_before = len(c.search_runs([runs.experiment_id()]))
        runs_dir = REPO_ROOT / "data" / "runs"
        dirs_before = set(p.name for p in runs_dir.iterdir()) if runs_dir.is_dir() else set()
        try:
            runs.train_run("v1/candidate", "train-on-tampered-copy", seed=0, steps=steps, dataset_root=copy, chain=chain)
            refused = None
        except VersionIntegrityError as e:
            refused = str(e).replace(str(REPO_ROOT) + "/", "")
        n_after = len(c.search_runs([runs.experiment_id()]))
        dirs_after = set(p.name for p in runs_dir.iterdir()) if runs_dir.is_dir() else set()
        log(f"AC6.5 flipped byte {size // 3} of {target.relative_to(copy)} in a clone of v1 at "
            f"{copy.relative_to(REPO_ROOT)}; train_run refused: {refused is not None}")
        log(f"AC6.5 {refused}")
        log(f"AC6.5 MLflow runs before/after the refused launch: {n_before}/{n_after}; new data/runs dirs: "
            f"{sorted(dirs_after - dirs_before)}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    log(f"AC6.5 scratch clone removed: {not scratch.exists()}")
    return {"refused": refused is not None, "message": refused, "runs_before": n_before, "runs_after": n_after,
            "new_run_dirs": sorted(dirs_after - dirs_before)}


def reuse_or(entity_id: str, fn, label: str) -> str:
    """Run id of the run that already produced ``entity_id``, or of a new run made by calling ``fn()``.

    Entities are content-addressed, so if v0, its split or v1 already has a producer, re-running that
    stage would only duplicate the record.
    """
    prods = runs.producers(entity_id)
    if prods:
        log(f"[reuse] {label}: run {prods[0].info.run_id[:12]} ({prods[0].info.run_name}) already produced {entity_id[:12]}")
        return prods[0].info.run_id
    rid = fn()
    log(f"[run] {label}: {rid[:12]}")
    return rid


def graph(run_ids: list[str]) -> dict:
    """Union of upstream traces of the given runs as a node/edge graph (runs + entities)."""
    nodes, edges = {}, set()
    raw = None
    for rid in run_ids:
        tr = runs.trace(rid)
        raw = raw or tr["raw_files"]
        for r_id, n in tr["runs"].items():
            nodes[r_id] = {"type": "run", **n}
            run = runs.client().get_run(r_id)
            for k, v in run.data.tags.items():
                if k.startswith("produced."):
                    kind, _, eid = v.partition(":")
                    nodes.setdefault(eid, {"type": kind, "id": eid})
                    edges.add((r_id, eid, "produces"))
        for e in tr["edges"]:
            kind, _, eid = e["consumes"].partition(":")
            nodes.setdefault(eid, {"type": kind, "id": eid})
            edges.add((eid, e["run"], e["role"]))
    if raw:
        nodes["raw:libero"] = {"type": "raw_files", "count": len(raw), "files": raw}
        for n_id, n in nodes.items():
            if n.get("stage") == "build":
                edges.add(("raw:libero", n_id, "source"))
    names = json.loads((LINEAGE_ROOT / "names.json").read_text())
    alias = {v: k for k, v in {**names["versions"], **names["subsets"]}.items()}
    for n_id, n in nodes.items():
        if n_id in alias:
            n["name"] = alias[n_id]
    return {"nodes": nodes, "edges": [{"from": a, "to": b, "label": l} for a, b, l in sorted(edges)]}


def mermaid(g: dict) -> str:
    """Mermaid ``flowchart LR`` text for a ``graph()`` result: runs as boxes, entities as stadiums."""
    def nid(x):
        """A stable Mermaid node id for an entity or run id."""
        return "n" + hashlib.sha256(x.encode()).hexdigest()[:10]

    lines = ["flowchart LR"]
    for k, n in g["nodes"].items():
        if n["type"] == "run":
            label = f"{n['stage']}: {n['name']}" + (" [stand-in]" if n.get("stand_in") and "stand-in" not in n["name"] else "")
            lines.append(f'  {nid(k)}["{label}"]')
        elif n["type"] == "raw_files":
            lines.append(f'  {nid(k)}[("raw LIBERO HDF5 x{n["count"]}")]')
        else:
            label = f"{n['type']} {n.get('name', '')} {k[:8]}".strip()
            lines.append(f'  {nid(k)}(["{label}"])')
    for e in g["edges"]:
        lines.append(f'  {nid(e["from"])} -->|{e["label"]}| {nid(e["to"])}')
    return "\n".join(lines) + "\n"


def eval_smoke(subset: str = "v0/reference", k: int = 5, seed: int = 0, steps: int = 2, batch_size: int = 2,
               eval_n: int = 2, eval_tau: int = 1200, n_boot: int = 1000) -> dict:
    """Evaluation-layer S9: the evaluation chain end to end at smoke scale, as MLflow runs of one chain.

    Grounding: PoC §9.1 Steps 2-4 pp. 18-19 (rollouts tied to the checkpoint; one return per metric per rollout;
    the two-step verdict on a counterfactual pair), §2.1 item 1 p. 3 (every rollout tied back to its checkpoint
    and data), §10 p. 20 (Evaluation layer). The pair has the shape of the PoC's random-k control (§9.1 Step 4):

        build(v0) [reuse] -> split(v0) [reuse] -> select drop-random-k of ``subset``
        -> train on ``subset`` and on its child (ACT, ``steps`` steps, per-subset normalization, S8)
        -> eval each (real LIBERO rollouts, libero_spatial task 0, ``eval_n`` init states, τ ``eval_tau`` steps, S5)
        -> returns of each (S6) -> verify (macro-KS gate + per-metric CIs, S7) -> trace(verify)

    Smoke scale: 2-step policies on 2 rollouts each exercise every stage's code and the lineage, not a PoC
    claim. Writes ``artifacts/s9_*`` of the evaluation-layer feature (not the data+lineage feature's files).
    """
    from vlapoc import verdict
    from vlapoc.returns import returns_run

    chain = f"eval-smoke-k{k}-s{seed}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out = REPO_ROOT / "data" / "runs" / chain
    t0 = time.time()
    log(f"chain {chain}; git HEAD {runs.git_state()['sha'][:12]} dirty={runs.git_state()['dirty']}")
    R: dict[str, str] = {}
    v0 = resolve_version("v0")
    R["build"] = reuse_or(v0["dataset_version_id"], lambda: runs.build_run("v0", chain=chain), "build(v0)")
    R["split"] = reuse_or(Subset.load("v0/candidate").subset_id, lambda: runs.split_run("v0", 2, 0, chain=chain)[0], "split(v0)")
    parent = Subset.load(subset)
    R["select"], child = runs.select_run(parent.subset_id, f"select-drop-random-{k}-s{seed} [eval-smoke]", "drop_random_k", k,
                                         seed=seed, chain=chain)
    log(f"[run] select drop-random-{k}: child {child.subset_id[:12]} ({len(child.episode_ids)} of {len(parent.episode_ids)} episodes)")
    ta = runs.train_run(subset, "train-ref [eval-smoke]", seed=0, steps=steps, batch_size=batch_size, chain=chain, smoke=True)
    tb = runs.train_run(child.subset_id, f"train-ref-drop-random-{k} [eval-smoke]", seed=0, steps=steps, batch_size=batch_size,
                        chain=chain, smoke=True)
    R["train_a"], R["train_b"] = ta["run_id"], tb["run_id"]
    log(f"[run] train: {ta['run_id'][:12]} ({ta['n_episodes']} eps), {tb['run_id'][:12]} ({tb['n_episodes']} eps)")
    # smoke runs: tagged vlapoc.smoke=true and eval.attribution_grade=false (plan r2 AC5.9/AC9.1); the same batch plan for both
    ea = runs.eval_run(f"train:{ta['run_id']}", EVAL_TASKS, eval_n, "eval-train-ref [eval-smoke]", out / "eval-a",
                       tau_episode_steps=eval_tau, batch_size=eval_n, device="cpu", chain=chain, smoke=True)
    eb = runs.eval_run(f"train:{tb['run_id']}", EVAL_TASKS, eval_n, f"eval-train-ref-drop-random-{k} [eval-smoke]", out / "eval-b",
                       tau_episode_steps=eval_tau, batch_size=eval_n, device="cpu", chain=chain, smoke=True)
    R["eval_a"], R["eval_b"] = ea["run_id"], eb["run_id"]
    log(f"[run] eval: {ea['run_id'][:12]} ({ea['n_success']}/{ea['n_rollouts']} success), {eb['run_id'][:12]} ({eb['n_success']}/{eb['n_rollouts']})")
    R["returns_a"] = returns_run(ea["run_id"], "returns-train-ref [eval-smoke]", chain=chain)["run_id"]
    R["returns_b"] = returns_run(eb["run_id"], f"returns-train-ref-drop-random-{k} [eval-smoke]", chain=chain)["run_id"]
    v = verdict.verify_run(ea["run_id"], eb["run_id"], f"verify-ref-vs-drop-random-{k} [eval-smoke]", family="W", n_boot=n_boot,
                           out_dir=out / "verify", chain=chain)
    R["verify"] = v["run_id"]
    log("[run] verify: " + verdict.describe(v).splitlines()[0])
    c = runs.client()
    rows = []
    for key, rid in R.items():
        run = c.get_run(rid)
        rows.append({"step": key, "run_id": rid, "name": run.info.run_name, "stage": run.data.tags["stage"],
                     "stand_in": run.data.tags.get("vlapoc.stand_in"), "status": run.info.status,
                     "smoke": run.data.tags.get("vlapoc.smoke", ""),
                     "attribution_grade": run.data.tags.get("eval.attribution_grade", run.data.tags.get("verify.attribution_grade", "")),
                     "chain": run.data.tags.get("vlapoc.chain", "(earlier command)")})
    table = pd.DataFrame(rows)
    log("AC9.1 chain runs:\n" + table.drop(columns=["run_id"]).to_string(index=False))
    tr = runs.trace(R["verify"])
    stages = sorted({n["stage"] for n in tr["runs"].values()})
    stand_ins = [n["name"] for n in tr["runs"].values() if n.get("stand_in")]
    kinds = sorted({e["consumes"].split(":")[0] for e in tr["edges"]})
    dead = [e["consumes"] for e in tr["edges"] if not e["produced_by"]]
    raw = tr["raw_files"] or []
    log(f"AC9.1 trace(verify {R['verify'][:12]}): {len(tr['runs'])} runs, stages {stages}; consumed kinds {kinds}; "
        f"dead ends {dead}; stand-in runs in the trace: {stand_ins}; raw files reached: {len(raw)}"
        + (f" (first {raw[0]['source_file'].split('/')[-1][:48]}… sha256 {raw[0]['sha256'][:16]})" if raw else ""))
    summary = {"chain": chain, "run_ids": R, "runs": rows,
               "verdict": {k_: v[k_] for k_ in ("outcome", "direction", "gate", "label", "family", "attribution_grade", "batch_plan_match")},
               "trace": {"n_runs": len(tr["runs"]), "stages": stages, "consumed_kinds": kinds, "dead_ends": dead,
                         "stand_in_runs": stand_ins, "raw_files": len(raw)},
               "seconds": round(time.time() - t0, 1)}
    table.to_csv(EVAL_FEATURE_ART / "s9_chain_runs.csv", index=False)
    (EVAL_FEATURE_ART / "s9_trace_verify.json").write_text(json.dumps(tr, indent=1, default=str))
    (EVAL_FEATURE_ART / "s9_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    g = graph([R["verify"], R["returns_a"], R["returns_b"]])
    (EVAL_FEATURE_ART / "s9_lineage_graph.mmd").write_text(mermaid(g))
    log(f"done in {summary['seconds']}s -> tracker/features/2026-10-01-evaluation-layer/artifacts/s9_summary.json")
    return summary


def main() -> int:
    """Record the chain (usage in the module docstring), then check AC6.1-AC6.7 and write the artifacts."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--eval-n", type=int, default=1, help="init states per eval task (real LIBERO rollouts, smoke scale)")
    ap.add_argument("--eval-tau", type=int, default=60, help="eval episode budget in steps (smoke scale)")
    ap.add_argument("--eval-smoke", action="store_true", help="evaluation-layer S9 chain (train -> eval -> returns -> verify)")
    ap.add_argument("--tamper-only", action="store_true",
                    help="run only the AC6.5 tamper refusal against the existing v1 (records no run)")
    a = ap.parse_args()
    if a.eval_smoke:
        # the S9 chain's own defaults (k 5, 2 training steps) apply unless --k / --steps were set to something else
        eval_smoke(k=5 if a.k == 20 else a.k, seed=a.seed, steps=2 if a.steps == 5 else a.steps)
        return 0
    if a.tamper_only:
        res = tamper_check(resolve_version("v1"), steps=a.steps)
        (ART / "s6_tamper_only.json").write_text(json.dumps(res, indent=2))
        log("summary -> tracker/features/2026-09-28-data-lineage-layer/artifacts/s6_tamper_only.json")
        return 0 if res["refused"] and res["runs_before"] == res["runs_after"] else 1
    chain = f"dry-run-k{a.k}-s{a.seed}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out = REPO_ROOT / "data" / "runs" / chain
    t_start = time.time()
    log(f"chain {chain}; git HEAD {runs.git_state()['sha'][:12]} dirty={runs.git_state()['dirty']}")
    R: dict[str, str] = {}

    v0 = resolve_version("v0")
    R["build"] = reuse_or(v0["dataset_version_id"], lambda: runs.build_run("v0", chain=chain), "build(v0)")
    R["split"] = reuse_or(Subset.load("v0/candidate").subset_id, lambda: runs.split_run("v0", 2, 0, chain=chain)[0], "split(v0)")
    try:
        v1 = resolve_version("v1")
    except LineageError:
        v1 = None
    if v1 is None:
        R["derive"], _ = runs.derive_run("v0", "v1", 5, 0, chain=chain)
        v1 = resolve_version("v1")
        log(f"[run] derive(v1): {R['derive'][:12]}")
    else:
        prods = runs.producers(v1["dataset_version_id"])
        if not prods:
            raise SystemExit("v1 exists but no MLflow derive run produced it; derive it with `python -m vlapoc.lineage derive`")
        R["derive"] = prods[0].info.run_id
        log(f"[reuse] derive(v1): run {R['derive'][:12]} ({prods[0].info.run_name}) produced {v1['dataset_version_id'][:12]}")

    R["screen"], screen_path = runs.screen_standin("v1/candidate", "v1/reference", "screen-v1 [stand-in]", out / "screen", chain=chain)
    log(f"[run] screen stand-in: {R['screen'][:12]}")
    tf0 = runs.train_run("v1/candidate", "train-full-s0", seed=0, steps=a.steps, batch_size=a.batch_size, chain=chain)
    log(f"[run] train-full-s0: {tf0['run_id'][:12]} ({tf0['seconds']}s, {tf0['n_episodes']} episodes)")
    tf1 = runs.train_run("v1/candidate", "train-full-s1", seed=1, steps=a.steps, batch_size=a.batch_size, chain=chain)
    log(f"[run] train-full-s1: {tf1['run_id'][:12]} ({tf1['seconds']}s)")
    R["train_full_s0"], R["train_full_s1"] = tf0["run_id"], tf1["run_id"]
    R["eval_full_s0"] = runs.eval_run(f"train:{tf0['run_id']}", EVAL_TASKS, a.eval_n, "eval-train-full-s0 [smoke]",
                                      out / "eval-full-s0", tau_episode_steps=a.eval_tau, chain=chain)["run_id"]
    R["attribute"], attr_path, attr_digest = runs.attribute_standin(tf0["run_id"], R["eval_full_s0"], "attribute-train-full-s0 [stand-in]",
                                                                    out / "attribute", chain=chain)
    log(f"[run] eval (real, smoke scale) + attribute stand-in: {R['eval_full_s0'][:12]}, {R['attribute'][:12]}")
    parent = runs.client().get_run(tf0["run_id"]).data.tags["subset_id"]  # train-full's training subset
    R["select_top"], top = runs.select_run(parent, f"select-drop-top-{a.k}", "drop_top_k", a.k, score_path=attr_path,
                                           metric="placeholder_score", score_run_id=R["attribute"], chain=chain)
    R["select_rand"], rnd = runs.select_run(parent, f"select-drop-random-{a.k}-s{a.seed}", "drop_random_k", a.k,
                                            seed=a.seed, chain=chain)
    log(f"[run] selects: top-k child {top.subset_id[:12]}, random-k child {rnd.subset_id[:12]}")
    tt = runs.train_run(top.subset_id, f"train-drop-top-{a.k}", seed=0, steps=a.steps, batch_size=a.batch_size, chain=chain)
    tr_ = runs.train_run(rnd.subset_id, f"train-drop-random-{a.k}", seed=0, steps=a.steps, batch_size=a.batch_size, chain=chain)
    R["train_top"], R["train_rand"] = tt["run_id"], tr_["run_id"]
    R["eval_top"] = runs.eval_run(f"train:{tt['run_id']}", EVAL_TASKS, a.eval_n, f"eval-train-drop-top-{a.k} [smoke]",
                                  out / "eval-top", tau_episode_steps=a.eval_tau, chain=chain)["run_id"]
    R["eval_rand"] = runs.eval_run(f"train:{tr_['run_id']}", EVAL_TASKS, a.eval_n, f"eval-train-drop-random-{a.k} [smoke]",
                                   out / "eval-rand", tau_episode_steps=a.eval_tau, chain=chain)["run_id"]
    log(f"[run] train on drop-top-k {tt['run_id'][:12]}, drop-random-k {tr_['run_id'][:12]}; evals {R['eval_top'][:12]}, {R['eval_rand'][:12]}")
    summary: dict = {"chain": chain, "run_ids": R, "k": a.k, "seed": a.seed, "steps": a.steps}

    # ---- AC6.1 -----------------------------------------------------------------------------------
    c = runs.client()
    rows = []
    for key, rid in R.items():
        run = c.get_run(rid)
        rows.append({"step": key, "run_id": rid, "name": run.info.run_name, "stage": run.data.tags["stage"],
                     "stand_in": run.data.tags.get("vlapoc.stand_in"), "status": run.info.status,
                     "git": run.data.tags.get("mlflow.source.git.commit", "")[:12], "chain": run.data.tags.get("vlapoc.chain", "(S4 command)")})
    table = pd.DataFrame(rows)
    log("AC6.1 chain runs:\n" + table.drop(columns=["run_id"]).to_string(index=False))
    table.to_csv(ART / "s6_chain_runs.csv", index=False)
    summary["AC6.1"] = rows

    # ---- AC6.2 -----------------------------------------------------------------------------------
    ac62 = {}
    for key in ("train_full_s0", "train_full_s1", "train_top", "train_rand"):
        run = c.get_run(R[key])
        ck = REPO_ROOT / run.data.tags["checkpoint.path"]
        cfg = json.loads((ck / "train_config.json").read_text())
        want = Subset.load(run.data.tags["subset_id"]).resolve()
        ac62[key] = {"episodes_in_train_config": len(cfg["dataset"]["episodes"]), "equal_to_subset": sorted(cfg["dataset"]["episodes"]) == want,
                     "policy": cfg["policy"]["type"], "steps": cfg["steps"], "device": cfg["policy"]["device"],
                     "seed": cfg["seed"], "wandb": cfg["wandb"]["enable"], "seconds": run.data.tags.get("train.seconds")}
        log(f"AC6.2 {run.info.run_name}: train_config.json episodes={ac62[key]['episodes_in_train_config']} == subset.resolve(): "
            f"{ac62[key]['equal_to_subset']}; policy={ac62[key]['policy']} steps={ac62[key]['steps']} device={ac62[key]['device']} "
            f"seed={ac62[key]['seed']} wandb={ac62[key]['wandb']} ({ac62[key]['seconds']}s)")
    summary["AC6.2"] = ac62

    # ---- AC6.3 -----------------------------------------------------------------------------------
    d_top = runs.diff(R["train_full_s0"], R["train_top"])
    d_seed = runs.diff(R["train_full_s0"], R["train_full_s1"])
    d_rand = runs.diff(R["train_full_s0"], R["train_rand"])
    log(f"AC6.3 diff(train-full-s0, train-drop-top-{a.k}): removed {d_top['data']['n_removed']}, added {d_top['data']['n_added']}, "
        f"changed_version {d_top['data']['n_changed_version']}; params {d_top['params']}; config {d_top['config']}; git same {d_top['git']['same']}")
    log(f"AC6.3 diff(train-full-s0, train-full-s1): removed/added/changed {d_seed['data']['n_removed']}/{d_seed['data']['n_added']}/"
        f"{d_seed['data']['n_changed_version']}; params {d_seed['params']}; config {d_seed['config']}; git same {d_seed['git']['same']}")
    log(f"AC6.3 diff(train-full-s0, train-drop-random-{a.k}): removed {d_rand['data']['n_removed']}; params {d_rand['params']}")
    for n, d in (("top", d_top), ("seed", d_seed), ("rand", d_rand)):
        (ART / f"s6_diff_full_s0_vs_{n}.json").write_text(json.dumps(d, indent=2, default=str))
    summary["AC6.3"] = {"top": {k: d_top["data"][k] for k in ("n_removed", "n_added", "n_changed_version")} | {"params": d_top["params"], "config": d_top["config"]},
                        "seed": {k: d_seed["data"][k] for k in ("n_removed", "n_added", "n_changed_version")} | {"params": d_seed["params"], "config": d_seed["config"]}}

    # ---- AC6.4 -----------------------------------------------------------------------------------
    tr = runs.trace(R["train_top"])
    path = [f"{n['stage']}:{n['name']}" for n in tr["runs"].values()]
    log(f"AC6.4 trace(train-drop-top-{a.k}) reaches {len(tr['runs'])} runs: {path}")
    log(f"AC6.4 raw files reached: {len(tr['raw_files'] or [])}; first: {tr['raw_files'][0]['source_file'].split('/')[-1][:50]}... "
        f"sha256 {tr['raw_files'][0]['sha256'][:16]}")
    (ART / "s6_trace_train_drop_top.json").write_text(json.dumps(tr, indent=2, default=str))
    labels = load_labels("v1")
    dropped = set(top.derivation["dropped"])
    corrupted_dropped = sorted(set(labels["episode_id"]) & dropped)
    probe = corrupted_dropped[0] if corrupted_dropped else sorted(set(labels["episode_id"]) & set(Subset.load(parent).episode_ids))[0]
    h1 = version_episodes("v1").set_index("episode_id").loc[probe, "content_hash"]
    h0 = version_episodes("v0").set_index("episode_id").loc[probe, "content_hash"]
    cons = runs.consumers(probe, h1)
    cn = [x["name"] for x in cons]
    log(f"AC6.4 corrupted episodes among the {a.k} dropped by top-k: {len(corrupted_dropped)}; probe {probe} (dropped: {probe in dropped})")
    log(f"AC6.4 consumers({probe.split('/')[-1]}@v1 {h1[:12]}): {cn}")
    log(f"AC6.4 lists train-full-s0: {'train-full-s0' in cn}; lists train-drop-top-{a.k}: {f'train-drop-top-{a.k}' in cn}")
    cons0 = runs.consumers(probe, h0)
    log(f"AC6.4 consumers(same episode @ clean v0 {h0[:12]}): {[x['name'] for x in cons0]}")
    (ART / "s6_consumers_probe.json").write_text(json.dumps({"probe": probe, "v1_hash": h1, "v0_hash": h0, "consumers_v1": cons,
                                                             "consumers_v0": cons0, "corrupted_dropped": corrupted_dropped}, indent=2))
    summary["AC6.4"] = {"trace_runs": path, "raw_files": len(tr["raw_files"] or []), "probe": probe, "probe_dropped": probe in dropped,
                        "corrupted_dropped": len(corrupted_dropped), "consumers": cn}

    # ---- AC6.5 tamper -----------------------------------------------------------------------------
    summary["AC6.5"] = tamper_check(v1, steps=a.steps, chain=chain)

    # ---- AC6.6 ------------------------------------------------------------------------------------
    scr = pd.read_parquet(screen_path)
    v1cand = Subset.load("v1/candidate").keys()
    in_cand = scr.merge(v1cand, on=["episode_id", "content_hash"], how="inner")
    joined = scr.merge(labels, on=["episode_id", "content_hash"], how="left", indicator=True)
    lab_in_scr = labels.merge(scr, on=["episode_id", "content_hash"], how="left", indicator=True)
    chk = runs.check_scores(pd.read_parquet(attr_path), R["train_full_s0"])
    log(f"AC6.6 screening stand-in rows {len(scr)}; keyed to v1/candidate episode versions: {len(in_cand)}/{len(v1cand)}; "
        f"left join with labels.parquet on (episode_id, content_hash): {int((joined['_merge'] == 'both').sum())} labelled corrupt, "
        f"{int((joined['_merge'] == 'left_only').sum())} unlabelled; labels found in screening table: "
        f"{int((lab_in_scr['_merge'] == 'both').sum())}/{len(labels)}")
    log(f"AC6.6 attribution stand-in vs train-full-s0: check_scores {chk}")
    summary["AC6.6"] = {"screen_rows": len(scr), "screen_in_candidate": len(in_cand), "labelled": int((joined["_merge"] == "both").sum()),
                        "labels_found": int((lab_in_scr["_merge"] == "both").sum()), "check_scores": chk}

    # ---- AC6.7 ------------------------------------------------------------------------------------
    g = graph([R["train_top"], R["train_rand"], R["eval_top"], R["eval_rand"], R["train_full_s1"], R["screen"]])
    (ART / "s6_lineage_graph.json").write_text(json.dumps(g, indent=2, default=str))
    (ART / "s6_lineage_graph.mmd").write_text(mermaid(g))
    n_runs = sum(1 for n in g["nodes"].values() if n["type"] == "run")
    log(f"AC6.7 lineage graph: {len(g['nodes'])} nodes ({n_runs} runs), {len(g['edges'])} edges -> artifacts/s6_lineage_graph.json, .mmd")
    summary["AC6.7"] = {"nodes": len(g["nodes"]), "runs": n_runs, "edges": len(g["edges"])}
    summary["seconds"] = round(time.time() - t_start, 1)
    (ART / "s6_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"done in {summary['seconds']}s -> tracker/features/2026-09-28-data-lineage-layer/artifacts/s6_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
