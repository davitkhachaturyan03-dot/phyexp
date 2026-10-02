"""Evaluation-layer S5: run the rollout harness (``vlapoc.rollouts``) from the command line.

Grounding: PoC §9.1 Step 2 p. 18 (rollouts with fixed initial states, success and time from the benchmark's
goal check), §2.1 items 1 and 3 p. 3 (tied to the checkpoint; fixed seeds and initial states).

    # tracked eval stage (MLflow run; needs a clean git tree):
    .venv/bin/python scripts/eval_smoke.py --policy noop --tasks libero_spatial:0,libero_10:8 --n 3 --track --name eval-noop
    .venv/bin/python scripts/eval_smoke.py --policy train:<train_run_id> --tasks libero_spatial:0 --n 2 --track --name eval-act
    .venv/bin/python scripts/eval_smoke.py --policy hf:HuggingFaceVLA/smolvla_libero@6721902b:data/hf_models/smolvla_libero \\
        --render 256 --device mps --tasks libero_spatial:0 --n 10 --track --record-trajectories --name eval-smolvla
    # untracked (records to --out only; determinism / batching checks):
    .venv/bin/python scripts/eval_smoke.py --policy noop --tasks libero_spatial:0 --n 4 --batch-size 4 --out data/eval/x

Prints one line per rollout and a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MUJOCO_GL", "cgl")
os.environ.setdefault("HF_DATASETS_CACHE", str(REPO_ROOT / "data" / "hf_datasets_cache"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="noop | train:<run_id> | hf:<repo>@<rev>:<local dir>")
    p.add_argument("--tasks", required=True, help="suite:task_id[,suite:task_id...]")
    p.add_argument("--n", type=int, default=3, help="init states per task")
    p.add_argument("--init-offset", type=int, default=0)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--tau", type=int, default=1200, help="tau_episode in steps (20 Hz)")
    p.add_argument("--render", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--record-trajectories", action="store_true")
    p.add_argument("--track", action="store_true", help="record as an MLflow eval stage run")
    p.add_argument("--name", default="eval-smoke")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    tasks = [(t.split(":")[0], int(t.split(":")[1])) for t in a.tasks.split(",")]
    if a.track:
        from vlapoc.runs import eval_run

        out = eval_run(a.policy, tasks, a.n, a.name, out_dir=a.out, seed_base=a.seed_base, init_offset=a.init_offset,
                       tau_episode_steps=a.tau, render_hw=a.render, batch_size=a.batch_size, device=a.device,
                       record_trajectories=a.record_trajectories)
    else:
        from vlapoc import rollouts as ro

        if a.policy == "noop":
            spec = ro.noop_policy()
        elif a.policy.startswith("hf:"):
            ext, local = a.policy[3:].rsplit(":", 1)
            spec = ro.checkpoint_policy(REPO_ROOT / local, device=a.device, external=ext)
        else:
            raise SystemExit("untracked mode supports noop and hf: policies; use --track for train:<run_id>")
        out_dir = a.out or REPO_ROOT / "data" / "eval" / a.name
        out_dir.mkdir(parents=True, exist_ok=True)
        df, traj = ro.run_rollouts(spec, ro.plan_rollouts(tasks, a.n, a.seed_base, a.init_offset), a.tau, a.render,
                                   a.batch_size, record_trajectories=a.record_trajectories, traj_root=out_dir / "trajectories")
        errs = ro.validate_record(df)
        df.to_parquet(out_dir / "rollouts.parquet", index=False)
        out = {"rollouts": str((out_dir / "rollouts.parquet").relative_to(REPO_ROOT)), "n_rollouts": len(df),
               "n_success": int(df["event"].sum()), "schema_errors": errs, "trajectories": traj}
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
