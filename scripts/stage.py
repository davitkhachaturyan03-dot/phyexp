#!/usr/bin/env python
"""One tracked stage per call, for the server pipeline (``scripts/server_pipeline.py``) and by hand.

Every subcommand records exactly the run the runbook (``artifacts/server-runbook.md``) prescribes and prints its
result as JSON. Run arguments (``--train``, ``--eval``, ``--returns``, ``--runs``) take an MLflow run id or a run
*name*; a name resolves to the latest FINISHED run of that name and stage, so a chain of SLURM jobs can refer to
runs that do not exist yet when the chain is submitted.

Blindness (PoC §2.1 item 4 p.3; owner decision 3): ``train``, ``features --which train`` and ``select`` open their
subset through ``method_view`` (= ``vlapoc.method_data.open_for_method``); ``attribute`` does the same inside
``attribute_run``. A fenced, leaky, ground-truth-derived or tampered version is refused (training-attribution S1).

Stages (docs/dataflow.md; PoC §9.1 pp. 18-19):

- ``train``    AC8.5: ``train_run`` with SmolVLA from ``data/hf_models/smolvla_base`` (``load_vlm_weights=False``),
               20,000 steps, batch 64, per-subset normalization, ``vlapoc.smoke=false`` unless ``--smoke``.
- ``eval``     AC5.8: all 20 tasks x init states 0-49, seed_base 0, τ 1200 steps, batch 50, asserted
               attribution-grade; trajectories recorded unless ``--no-trajectories`` (retrained arms, AC7.5);
               output in ``data/eval/<name>`` (where ``features`` finds it).
- ``returns``  AC6.7: ``vlapoc.returns.returns_run``.
- ``features`` Step 3 input: ``vlapoc.influence.featurize`` of the train run's checkpoint on its training subset
               (``--which train``) or on an eval run's recorded trajectories (``--which rollouts``), one shard.
               ``--preset cupid`` (default: ℓ_square on both sides, CUPID App. A.1), ``athena`` (loss / square) or
               ``branch`` (loss / loss, the pre-S2 behaviour); the output function is stored in the manifest.
- ``attribute``/``ensemble``  Step 3: ``vlapoc.attribution.attribute_run`` / ``ensemble_run``.
- ``select``   Step 4: ``vlapoc.runs.select_run`` (per-task ops by default). A later ``train --subset
               select:<select run name>`` trains on the child subset that run produced.
- ``verify``   Step 4 gate: ``vlapoc.verdict.verify_run`` between two eval runs; ``--comparison`` tags it as a Step 4
               comparison (``vs_base`` | ``curated_vs_random`` | ``head_to_head``).
- ``analyze``  the PoC §4.2/§4.5 outputs of an attribute or ensemble run: four groups (descriptive, unverified), aggregation by
               catalog column, episode x metric heatmap, throughput-vs-reliability scatter (``vlapoc.analysis``).
- ``score``    ground-truth recovery (``vlapoc.recovery``): AUROC / recall@k per corruption type and harm column next to
               each type's best simple heuristic; only for a version with an answer key (v3), after attribution exists.
- ``step4-summary``  one table over a chain's verify runs: per arm and comparison, how many seeds went each way per
               metric (``vlapoc.step4``; PoC §9.1 Step 4, §2.6, §4.5).

Usage::

    .venv/bin/python scripts/stage.py train --subset v0/candidate --name train-candidate-s0 --seed 0
    .venv/bin/python scripts/stage.py eval --train train-candidate-s0 --name eval-candidate-s0
    .venv/bin/python scripts/stage.py returns --eval eval-candidate-s0 --name returns-candidate-s0
    .venv/bin/python scripts/stage.py features --train train-candidate-s0 --which train --shard 0/4
    .venv/bin/python scripts/stage.py features --train train-candidate-s0 --which rollouts --eval eval-candidate-s0
    .venv/bin/python scripts/stage.py attribute --train train-candidate-s0 --eval eval-candidate-s0 \\
        --returns returns-candidate-s0 --name attribute-candidate-s0
    .venv/bin/python scripts/stage.py ensemble --runs attribute-candidate-s0 attribute-candidate-s1 \\
        attribute-candidate-s2 --name attribute-candidate-ens
    .venv/bin/python scripts/stage.py select --parent v0/candidate --scores attribute-candidate-ens \\
        --op drop_top_k_per_task --metric harm_f30__same_task --k 5 --name select-harm-f30-k5
    .venv/bin/python scripts/stage.py train --subset select:select-harm-f30-k5 --name train-harm-f30-k5-s0 --seed 0
    .venv/bin/python scripts/stage.py verify --a eval-candidate-s0 --b eval-harm-f30-k5-s0 --name verify-harm-f30-k5-s0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vlapoc.libero_convert import REPO_ROOT  # noqa: E402

PRETRAINED_DIR = REPO_ROOT / "data" / "hf_models" / "smolvla_base"
# (train-side, rollout-side) output functions per preset: CUPID App. A.1 (square/square, the default), ATHENA App. A.1
# Eq. 14-15 p. 14 for flow-matching VLAs (loss on the training side, square on rollouts), and the branch's earlier
# behaviour (loss/loss). Training-attribution S2.
FEATURE_PRESETS = {"cupid": ("square", "square"), "athena": ("loss", "square"), "branch": ("loss", "loss")}
PRETRAINED_REF = "lerobot/smolvla_base@d9f33c94a60fb382c90dea2164c96845bd955e28"  # runbook AC8.5
FEATS_ROOT = REPO_ROOT / "data" / "attribution" / "feats"  # ``--feats-root`` overrides (tests use a scratch directory)


# Per-invocation context set by ``main`` from the global options: the lineage store every call reads and writes
# (``--lineage-root``, e.g. a ``runs.scratch_lineage_copy``), the chain prefix names are resolved within, and ``--latest``.
LR: Path | None = None
CHAIN_PREFIX: str | None = None
LATEST = False


def chain_prefix_of(chain: str | None) -> str | None:
    """``"<tag>-"`` for a chain id ``"<tag>-<sha8>"`` (``server_pipeline.py`` builds ``f"{tag}-{sha[:8]}"``), else ``None``."""
    return chain.rsplit("-", 1)[0] + "-" if chain and "-" in chain else None


def run_id(ref: str, stage: str, chain_prefix: str | None = None, latest: bool = False, lineage_root: Path | None = None) -> str:
    """``ref`` itself if it is a run id, else the FINISHED ``stage`` run named ``ref``.

    With ``chain_prefix`` (``"<tag>-"``), only runs whose ``vlapoc.chain`` tag starts with it are candidates, so a
    re-submitted or partly re-run chain cannot silently pick up another chain's run of the same name (PoC §2.1 item 1
    p. 3: every run tied to what it consumed). Two or more candidates raise ``ValueError`` unless ``latest``, which takes
    the most recent. Without ``chain_prefix`` (hand use) the most recent run of that name wins, as before.
    """
    from vlapoc.runs import client, experiment_id

    if re.fullmatch(r"[0-9a-f]{32}", ref):
        return ref
    runs = client(lineage_root).search_runs([experiment_id(lineage_root)], f"tags.mlflow.runName = '{ref}' and tags.stage = '{stage}' "
                                            "and attributes.status = 'FINISHED'", order_by=["attributes.start_time DESC"], max_results=1000)
    if chain_prefix:
        runs = [r for r in runs if r.data.tags.get("vlapoc.chain", "").startswith(chain_prefix)]
        if len(runs) > 1 and not latest:
            raise ValueError(f"{len(runs)} FINISHED {stage} runs named {ref!r} in chains {sorted({r.data.tags['vlapoc.chain'] for r in runs})}; "
                             "pass --latest to take the most recent, or use a run id")
    if not runs:
        raise SystemExit(f"no FINISHED {stage} run named {ref!r}" + (f" in a chain starting {chain_prefix!r}" if chain_prefix else ""))
    return runs[0].info.run_id


def rid(ref: str, stage: str) -> str:
    """``run_id`` with this invocation's lineage root, chain prefix and ``--latest``."""
    return run_id(ref, stage, CHAIN_PREFIX, LATEST, LR)


def run_name(run: str) -> str:
    from vlapoc.runs import client

    return client(LR).get_run(run).info.run_name


def feats_prefix(train_rid: str, which: str, eval_rid: str | None = None) -> Path:
    """``data/attribution/feats/<train run name>/train`` or ``.../rollouts-<eval run name>``."""
    base = Path(FEATS_ROOT) / run_name(train_rid)
    return base / "train" if which == "train" else base / f"rollouts-{run_name(eval_rid)}"


def subset_ref(ref: str) -> str:
    """A subset reference, where ``select:<select run>`` means the child subset that select run produced."""
    from vlapoc.runs import client

    if not ref.startswith("select:"):
        return ref
    return client(LR).get_run(rid(ref.split(":", 1)[1], "select")).data.tags["output.subset"].split(":", 1)[1]


def method_view(ref: str, lineage_root: Path | None = None):
    """Open a subset for the method chain through ``vlapoc.method_data.open_for_method`` and return its ``MethodView``.

    Grounding: PoC §2.1 item 4 p.3 ("flags exactly those" only means something if the method cannot read the
    answer) and owner decision 3 (ground-truth feature): the attribution chain's trainer, feature extractor and
    selector read data only through the method-facing loader, which hashes the version (``VersionIntegrityError``),
    refuses fenced / leaky versions and ground-truth-derived subsets (``LineageError``). ``train_run`` itself is
    left unchanged because the ground-truth feature's oracle arms legitimately train on ``oracle_drop`` subsets; the
    check sits here, at the chain's entry (training-attribution S1). ``ref`` may be ``select:<select run>``.
    """
    from vlapoc.method_data import open_for_method

    return open_for_method(subset_ref(ref), lineage_root if lineage_root is not None else LR)


def cmd_train(a) -> dict:
    from vlapoc.runs import train_run

    method_view(a.subset)
    return train_run(subset_ref(a.subset), a.name, seed=a.seed, steps=a.steps, batch_size=a.batch_size, policy_type="smolvla",
                     policy_overrides={"load_vlm_weights": False}, device=a.device, pretrained_path=PRETRAINED_DIR,
                     pretrained_ref=a.pretrained_ref, smoke=a.smoke, chain=a.chain, lineage_root=LR)


def cmd_eval(a) -> dict:
    from vlapoc.runs import CANONICAL_TASKS, eval_run

    tasks = CANONICAL_TASKS if not a.tasks else [(s, int(t)) for s, t in (x.split(":") for x in a.tasks.split(","))]
    out_dir = REPO_ROOT / "data" / "eval" / a.name
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} is not empty (a previous attempt?); move it aside before re-running")
    return eval_run(f"train:{rid(a.train, 'train')}", tasks, a.n, a.name, out_dir=out_dir, batch_size=a.batch_size,
                    device=a.device, record_trajectories=not a.no_trajectories, chain=a.chain,
                    attribution_grade=not a.smoke, smoke=a.smoke, lineage_root=LR)


def cmd_returns(a) -> dict:
    from vlapoc.returns import returns_run

    return returns_run(rid(a.eval, "eval"), a.name, chain=a.chain, lineage_root=LR)


def cmd_features(a) -> dict:
    from vlapoc.influence import featurize
    from vlapoc.lineage import file_digests, sha256_json
    from vlapoc.runs import _training_subset, client

    c = client(LR)
    tid = rid(a.train, "train")
    tr = c.get_run(tid)
    ckpt = REPO_ROOT / tr.data.tags["checkpoint.path"]
    if a.which == "train":
        view = method_view(_training_subset(tr, LR).subset_id)  # S1: root and episodes come from the MethodView
        root, episodes, eid = view.lerobot_root, list(view.episode_indices), None
    else:
        eid = rid(a.eval, "eval")
        ev = c.get_run(eid)
        root = Path(a.root) if a.root else REPO_ROOT / "data" / "eval" / ev.info.run_name / "trajectories"
        recorded = ev.data.tags.get("output.trajectories", "").split(":", 1)[-1]
        if sha256_json(file_digests(root)) != recorded:
            raise SystemExit(f"{root} is not the trajectories copy eval run {eid[:12]} recorded")
        episodes = None
    si, sn = (int(x) for x in a.shard.split("/"))
    stride = a.stride if a.stride is not None else (a.train_stride if a.which == "train" else a.rollout_stride)
    output_fn = a.output_fn or FEATURE_PRESETS[a.preset][0 if a.which == "train" else 1]
    m = featurize(ckpt, root, feats_prefix(tid, a.which, eid), episodes, a.rank, a.k_draws, stride, a.batch_size,
                  a.workers, (si, sn), a.seed, a.device, output_fn=output_fn)
    return {k: m[k] for k in ("checkpoint_sha256", "dataset_root", "stride", "shard", "n_samples", "dim", "output_fn")}


def cmd_attribute(a) -> dict:
    from vlapoc.attribution import attribute_run

    tid, eid = rid(a.train, "train"), rid(a.eval, "eval")
    out = attribute_run(tid, eid, rid(a.returns, "returns"), feats_prefix(tid, "train"),
                        feats_prefix(tid, "rollouts", eid), a.name, damping=a.damping, center=not a.no_center,
                        device=a.device, chain=a.chain, return_encoding="pm1" if a.cupid_returns else "raw",
                        allow_partial=a.allow_partial, lineage_root=LR)
    return {k: v for k, v in out.items() if k != "agreement"}


def cmd_ensemble(a) -> dict:
    from vlapoc.attribution import ensemble_run

    return ensemble_run([rid(r, "attribute") for r in a.runs], a.name, chain=a.chain, lineage_root=LR)


def select_params(op: str, headline: bool = False, pipeline_curation: str | None = None) -> dict:
    """Params every select run of the chain records besides the op's own: the curation unit of the arm (``per_task`` for
    the ``*_per_task`` ops, ``global`` for ``drop_top_k``/``drop_random_k``), whether it is a headline arm, and the pipeline's
    ``--curation`` setting (owner answer Q1, 2026-10-20: both, per-task headline; PoC §9.1 Step 4.1 p.19)."""
    mode = "per_task" if op.endswith("_per_task") else ("global" if op in ("drop_top_k", "drop_random_k") else "group")
    return {"curation": mode, "headline": bool(headline), "pipeline_curation": pipeline_curation or "unspecified"}


def cmd_select(a) -> dict:
    import tempfile

    from vlapoc.attribution import _artifact_table
    from vlapoc.runs import client, select_run

    method_view(a.parent)  # S1: the parent opens through the method loader (fence / integrity / ground-truth chain)
    score_path, sid = None, None
    if a.op.startswith("drop_top_k"):
        sid = rid(a.scores, "attribute")
        (REPO_ROOT / "data" / "scratch").mkdir(parents=True, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix="select_", dir=REPO_ROOT / "data" / "scratch")
        score_path = Path(tmp) / "attribution_scores.parquet"
        _artifact_table(client(LR).get_run(sid), "attribution_scores", LR).to_parquet(score_path, index=False)
    extra = select_params(a.op, a.headline, a.pipeline_curation)
    sel_id, child = select_run(subset_ref(a.parent), a.name, a.op, k=a.k, seed=a.seed, score_path=score_path,
                            metric=a.metric, score_run_id=sid, direction=a.direction, chain=a.chain,
                            require_declaration=a.op.startswith("drop_top_k"), extra_params=extra, lineage_root=LR)
    return {"run_id": sel_id, "subset_id": child.subset_id, "n_episodes": len(child.episode_ids)}


def cmd_verify(a) -> dict:
    from vlapoc.verdict import verify_run

    tags = {}
    if a.comparison and not (a.treated and a.control and a.seed_id is not None):
        raise SystemExit("--comparison needs --treated, --control and --seed-id")
    if a.comparison:  # Step 4 pipeline: which comparison this verdict is (read back by ``step4-summary``)
        tags = {"step4.kind": a.comparison, "step4.treated": a.treated, "step4.control": a.control, "step4.seed": a.seed_id}
    v = verify_run(rid(a.a, "eval"), rid(a.b, "eval"), a.name, family=a.family, chain=a.chain, extra_tags=tags, lineage_root=LR)
    return {k: v[k] for k in ("run_id", "label", "attribution_grade") if k in v}


def cmd_score(a) -> dict:
    """Ground-truth side: recovery of the planted corruptions by an attribute run's scores (``vlapoc.recovery``)."""
    from vlapoc.recovery import recovery_run

    sid = rid(a.scores, "attribute")
    return recovery_run(sid, a.version, a.name, subset=a.subset, chain=a.chain, lineage_root=LR, synthetic=a.synthetic)


def cmd_analyze(a) -> dict:
    """The §4.2 / §4.5 outputs of an attribute or ensemble run (``vlapoc.analysis``): groups, aggregation, heatmap, scatter."""
    from vlapoc.analysis import analyze_run

    return analyze_run(rid(a.scores, "attribute"), a.name, z=a.z, by=tuple(a.by), chain=a.chain, lineage_root=LR, synthetic=a.synthetic)


def cmd_step4_summary(a) -> dict:
    """Step 4's table over the chain's verify runs (``vlapoc.step4``): seed consistency per arm and comparison."""
    from vlapoc.step4 import summary_run

    out = summary_run(a.tag, a.name, chain=a.chain_id, run_chain=a.chain, lineage_root=LR)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scripts/stage.py", description=__doc__.split("\n\n")[0])
    p.add_argument("--chain", default=None, help="tag vlapoc.chain on the run (groups one experiment); run names are "
                   "resolved only among runs of chains starting with its <tag>-")
    p.add_argument("--lineage-root", type=Path, default=None,
                   help="a lineage store other than data/lineage (e.g. runs.scratch_lineage_copy, for tests); every stage reads and writes it")
    p.add_argument("--feats-root", type=Path, default=None, help="where feature files live (default data/attribution/feats)")
    p.add_argument("--latest", action="store_true", help="when two FINISHED runs in the chain scope share a name, take the most recent")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("train")
    s.add_argument("--subset", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--steps", type=int, default=20000)
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--device", default="cuda")
    s.add_argument("--pretrained-ref", default=PRETRAINED_REF)
    s.add_argument("--smoke", action="store_true")

    s = sub.add_parser("eval")
    s.add_argument("--train", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--tasks", default=None, help="suite:task_id,... (smoke only; default all 20)")
    s.add_argument("--batch-size", type=int, default=50)
    s.add_argument("--device", default="cuda")
    s.add_argument("--smoke", action="store_true", help="not attribution-grade; the name must contain 'smoke'")
    s.add_argument("--no-trajectories", action="store_true", help="retrained arms (AC7.5): rollouts only")

    s = sub.add_parser("returns")
    s.add_argument("--eval", required=True)
    s.add_argument("--name", required=True)

    s = sub.add_parser("features")
    s.add_argument("--train", required=True)
    s.add_argument("--which", choices=["train", "rollouts"], required=True)
    s.add_argument("--eval", default=None, help="the eval run whose trajectories to featurize (--which rollouts)")
    s.add_argument("--root", default=None, help="trajectories dir (default data/eval/<eval run name>/trajectories)")
    s.add_argument("--shard", default="0/1")
    s.add_argument("--stride", type=int, default=None)
    s.add_argument("--train-stride", type=int, default=2)
    s.add_argument("--rollout-stride", type=int, default=4)
    s.add_argument("--preset", choices=sorted(FEATURE_PRESETS), default="cupid",
                   help="output functions (train side, rollout side): cupid = square/square (default), "
                        "athena = loss/square, branch = loss/loss")
    s.add_argument("--output-fn", choices=["loss", "square"], default=None, help="override the preset for this call")
    s.add_argument("--rank", type=int, default=8)
    s.add_argument("--k-draws", type=int, default=4)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--device", default="cuda")

    s = sub.add_parser("attribute")
    for k in ("--train", "--eval", "--returns", "--name"):
        s.add_argument(k, required=True)
    s.add_argument("--damping", type=float, default=1e-3)
    s.add_argument("--no-center", action="store_true")
    s.add_argument("--cupid-returns", action="store_true", help="CUPID's R = ±1 (uncentred) for the binary return columns")
    s.add_argument("--allow-partial", action="store_true",
                   help="accept eval rollouts that have no recorded trajectory (the count is recorded)")
    s.add_argument("--device", default=None)

    s = sub.add_parser("ensemble")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--name", required=True)

    s = sub.add_parser("select")
    s.add_argument("--parent", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--op", default="drop_top_k_per_task")
    s.add_argument("--k", type=int, required=True, help="episodes to drop (per task for the *_per_task ops)")
    s.add_argument("--scores", default=None, help="attribute run (drop_top_k*)")
    s.add_argument("--metric", default=None)
    s.add_argument("--direction", default="highest")
    s.add_argument("--seed", type=int, default=None)
    s.add_argument("--headline", action="store_true", help="record that this arm is a headline arm (per-task arms; owner Q1)")
    s.add_argument("--pipeline-curation", default=None, choices=["per_task", "global", "both"],
                   help="the pipeline's --curation setting, recorded with the arm (PoC §9.1 Step 4.1; owner Q1)")

    s = sub.add_parser("verify")
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--family", default="W", choices=["W", "O"])
    s.add_argument("--comparison", default=None, choices=["vs_base", "curated_vs_random", "head_to_head"],
                   help="Step 4 comparison kind; with --treated/--control/--seed-id tags the verify run for step4-summary")
    s.add_argument("--treated", default=None, help="arm B (bare arm name, e.g. harm-f30-k5)")
    s.add_argument("--control", default=None, help="arm A (bare arm name, or 'base')")
    s.add_argument("--seed-id", default=None, help="the training seed both arms share")

    s = sub.add_parser("score")
    s.add_argument("--scores", required=True, help="the attribute run (name or id) whose score table to score")
    s.add_argument("--version", required=True, help="a version with an answer key (e.g. v3)")
    s.add_argument("--subset", default="candidate", help="subset of that version the table must key exactly")
    s.add_argument("--synthetic", action="store_true", help="the scores are test data: recorded in the run's params")
    s.add_argument("--name", required=True)

    s = sub.add_parser("analyze")
    s.add_argument("--scores", required=True, help="the attribute or ensemble run (name or id)")
    s.add_argument("--name", required=True)
    s.add_argument("--z", type=float, default=1.0, help="group threshold on the z-scored harm columns (declared, inferred default 1)")
    s.add_argument("--by", nargs="+", default=["task_name"], help="catalog columns to aggregate by (PoC §1 p.3)")
    s.add_argument("--synthetic", action="store_true", help="the scores are test data: put SYNTHETIC in every figure title")

    s = sub.add_parser("step4-summary")
    s.add_argument("--tag", required=True, help="the chain's tag (runs are named <stage>-<tag>-...)")
    s.add_argument("--name", required=True)
    s.add_argument("--chain-id", default=None, help="full chain id <tag>-<sha8> if the tag matches several chains")

    a = p.parse_args(argv)
    global LR, CHAIN_PREFIX, LATEST, FEATS_ROOT
    LR, CHAIN_PREFIX, LATEST = a.lineage_root, chain_prefix_of(a.chain), a.latest
    if a.feats_root is not None:
        FEATS_ROOT = a.feats_root
    out = globals()[f"cmd_{a.cmd.replace('-', '_')}"](a)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
