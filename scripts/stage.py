#!/usr/bin/env python
"""One tracked stage per call, for the server pipeline (``scripts/server_pipeline.py``) and by hand.

Every subcommand records exactly the run the runbook (``artifacts/server-runbook.md``) prescribes and prints its
result as JSON. Run arguments (``--train``, ``--eval``, ``--returns``, ``--runs``) take an MLflow run id or a run
*name*; a name resolves to the latest FINISHED run of that name and stage, so a chain of SLURM jobs can refer to
runs that do not exist yet when the chain is submitted.

Stages (docs/dataflow.md; PoC §9.1 pp. 18-19):

- ``train``    AC8.5: ``train_run`` with SmolVLA from ``data/hf_models/smolvla_base`` (``load_vlm_weights=False``),
               20,000 steps, batch 64, per-subset normalization, ``vlapoc.smoke=false`` unless ``--smoke``.
- ``eval``     AC5.8: all 20 tasks x init states 0-49, seed_base 0, τ 1200 steps, batch 50, asserted
               attribution-grade; trajectories recorded unless ``--no-trajectories`` (retrained arms, AC7.5);
               output in ``data/eval/<name>`` (where ``features`` finds it).
- ``returns``  AC6.7: ``vlapoc.returns.returns_run``.
- ``features`` Step 3 input: ``vlapoc.influence.featurize`` of the train run's checkpoint on its training subset
               (``--which train``) or on an eval run's recorded trajectories (``--which rollouts``), one shard.
- ``attribute``/``ensemble``  Step 3: ``vlapoc.attribution.attribute_run`` / ``ensemble_run``.
- ``select``   Step 4: ``vlapoc.runs.select_run`` (per-task ops by default). A later ``train --subset
               select:<select run name>`` trains on the child subset that run produced.
- ``verify``   Step 4 gate: ``vlapoc.verdict.verify_run`` between two eval runs.

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
PRETRAINED_REF = "lerobot/smolvla_base@d9f33c94a60fb382c90dea2164c96845bd955e28"  # runbook AC8.5
FEATS_ROOT = REPO_ROOT / "data" / "attribution" / "feats"


def run_id(ref: str, stage: str) -> str:
    """``ref`` itself if it is a run id, else the latest FINISHED ``stage`` run named ``ref``."""
    from vlapoc.runs import client, experiment_id

    if re.fullmatch(r"[0-9a-f]{32}", ref):
        return ref
    runs = client().search_runs([experiment_id()], f"tags.mlflow.runName = '{ref}' and tags.stage = '{stage}' "
                                "and attributes.status = 'FINISHED'", order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise SystemExit(f"no FINISHED {stage} run named {ref!r}")
    return runs[0].info.run_id


def run_name(rid: str) -> str:
    from vlapoc.runs import client

    return client().get_run(rid).info.run_name


def feats_prefix(train_rid: str, which: str, eval_rid: str | None = None) -> Path:
    """``data/attribution/feats/<train run name>/train`` or ``.../rollouts-<eval run name>``."""
    base = FEATS_ROOT / run_name(train_rid)
    return base / "train" if which == "train" else base / f"rollouts-{run_name(eval_rid)}"


def subset_ref(ref: str) -> str:
    """A subset reference, where ``select:<select run>`` means the child subset that select run produced."""
    from vlapoc.runs import client

    if not ref.startswith("select:"):
        return ref
    return client().get_run(run_id(ref.split(":", 1)[1], "select")).data.tags["output.subset"].split(":", 1)[1]


def cmd_train(a) -> dict:
    from vlapoc.runs import train_run

    return train_run(subset_ref(a.subset), a.name, seed=a.seed, steps=a.steps, batch_size=a.batch_size, policy_type="smolvla",
                     policy_overrides={"load_vlm_weights": False}, device=a.device, pretrained_path=PRETRAINED_DIR,
                     pretrained_ref=a.pretrained_ref, smoke=a.smoke, chain=a.chain)


def cmd_eval(a) -> dict:
    from vlapoc.runs import CANONICAL_TASKS, eval_run

    tasks = CANONICAL_TASKS if not a.tasks else [(s, int(t)) for s, t in (x.split(":") for x in a.tasks.split(","))]
    out_dir = REPO_ROOT / "data" / "eval" / a.name
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} is not empty (a previous attempt?); move it aside before re-running")
    return eval_run(f"train:{run_id(a.train, 'train')}", tasks, a.n, a.name, out_dir=out_dir, batch_size=a.batch_size,
                    device=a.device, record_trajectories=not a.no_trajectories, chain=a.chain,
                    attribution_grade=not a.smoke, smoke=a.smoke)


def cmd_returns(a) -> dict:
    from vlapoc.returns import returns_run

    return returns_run(run_id(a.eval, "eval"), a.name, chain=a.chain)


def cmd_features(a) -> dict:
    from vlapoc.influence import featurize
    from vlapoc.lineage import file_digests, resolve_version, sha256_json
    from vlapoc.runs import _training_subset, client

    c = client()
    tid = run_id(a.train, "train")
    tr = c.get_run(tid)
    ckpt = REPO_ROOT / tr.data.tags["checkpoint.path"]
    if a.which == "train":
        subset = _training_subset(tr, None)
        root = REPO_ROOT / resolve_version(subset.dataset_version_id)["lerobot_root"]
        episodes, eid = subset.resolve(), None
    else:
        eid = run_id(a.eval, "eval")
        ev = c.get_run(eid)
        root = Path(a.root) if a.root else REPO_ROOT / "data" / "eval" / ev.info.run_name / "trajectories"
        recorded = ev.data.tags.get("output.trajectories", "").split(":", 1)[-1]
        if sha256_json(file_digests(root)) != recorded:
            raise SystemExit(f"{root} is not the trajectories copy eval run {eid[:12]} recorded")
        episodes = None
    si, sn = (int(x) for x in a.shard.split("/"))
    stride = a.stride if a.stride is not None else (a.train_stride if a.which == "train" else a.rollout_stride)
    m = featurize(ckpt, root, feats_prefix(tid, a.which, eid), episodes, a.rank, a.k_draws, stride, a.batch_size,
                  a.workers, (si, sn), a.seed, a.device)
    return {k: m[k] for k in ("checkpoint_sha256", "dataset_root", "stride", "shard", "n_samples", "dim")}


def cmd_attribute(a) -> dict:
    from vlapoc.attribution import attribute_run

    tid, eid = run_id(a.train, "train"), run_id(a.eval, "eval")
    out = attribute_run(tid, eid, run_id(a.returns, "returns"), feats_prefix(tid, "train"),
                        feats_prefix(tid, "rollouts", eid), a.name, damping=a.damping, center=not a.no_center,
                        device=a.device, chain=a.chain)
    return {k: v for k, v in out.items() if k != "agreement"}


def cmd_ensemble(a) -> dict:
    from vlapoc.attribution import ensemble_run

    return ensemble_run([run_id(r, "attribute") for r in a.runs], a.name, chain=a.chain)


def cmd_select(a) -> dict:
    import tempfile

    from vlapoc.attribution import _artifact_table
    from vlapoc.runs import client, select_run

    score_path, sid = None, None
    if a.op.startswith("drop_top_k"):
        sid = run_id(a.scores, "attribute")
        (REPO_ROOT / "data" / "scratch").mkdir(parents=True, exist_ok=True)
        tmp = tempfile.mkdtemp(prefix="select_", dir=REPO_ROOT / "data" / "scratch")
        score_path = Path(tmp) / "attribution_scores.parquet"
        _artifact_table(client().get_run(sid), "attribution_scores", None).to_parquet(score_path, index=False)
    rid, child = select_run(subset_ref(a.parent), a.name, a.op, k=a.k, seed=a.seed, score_path=score_path,
                            metric=a.metric, score_run_id=sid, direction=a.direction, chain=a.chain)
    return {"run_id": rid, "subset_id": child.subset_id, "n_episodes": len(child.episode_ids)}


def cmd_verify(a) -> dict:
    from vlapoc.verdict import verify_run

    v = verify_run(run_id(a.a, "eval"), run_id(a.b, "eval"), a.name, family=a.family, chain=a.chain)
    return {k: v[k] for k in ("run_id", "label", "attribution_grade") if k in v}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scripts/stage.py", description=__doc__.split("\n\n")[0])
    p.add_argument("--chain", default=None, help="tag vlapoc.chain on the run (groups one experiment)")
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

    s = sub.add_parser("verify")
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--family", default="W", choices=["W", "O"])

    a = p.parse_args(argv)
    out = globals()[f"cmd_{a.cmd}"](a)
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
