#!/usr/bin/env python
"""Submit the whole attribution-and-curation experiment to SLURM as one dependency chain of tracked stages.

Each job runs one ``scripts/stage.py`` subcommand (one MLflow stage run), in the runbook's order
(``tracker/features/2026-10-01-evaluation-layer/artifacts/server-runbook.md``) extended by Steps 3-4:

    base    per seed S: train (AC8.5) → eval with trajectories (AC5.8) → returns (AC6.7)
            → features (train subset, sharded; rollouts, sharded) → attribute (one per seed)
    rank    ensemble of the per-seed attribution tables (z-scored mean)
    select  one select run per arm, all in one CPU job (selects are seconds; one writer for names and subsets):
              harm-<metric>-k<k>  drop_top_k_per_task on harm_<metric>__same_task (the k most harmful per task)
              random<r>-k<k>      drop_random_k_per_task, seed r: the matched control (same per-task counts)
              drop-random-20      AC7.5's gate: drop_random_k 20, seed 0
    arms    per arm and seed: train on the child subset → eval (no trajectories; AC7.5) → verify vs the base eval
            of the same seed, family W (declared) and O (alongside)

Grounding: PoC §9.1 Steps 3-4 (p. 19): "Rank training episodes by harmful influence ... Retrain without the
selected episodes and compare against the original policy"; PoC §2.6 (p. 5) and §10 (p. 20): every curated arm is
compared with a random removal of the same size, here matched per task. PoC §2.3 (p. 4): 2-3 seeds per data
configuration; every arm uses the base recipe unchanged (PoC p. 1: data is the only variable).

Run names are ``<stage>-<tag>[-<arm>]-s<S>``; ``stage.py`` resolves them to run ids when each job starts, so the
chain can be submitted before any run exists. The chain refuses to submit from a dirty tree and every job refuses
to start on a commit other than the one it was submitted from (``slurm/guard.sh``).

Usage::

    .venv/bin/python scripts/server_pipeline.py --tag v0cand --dry-run       # print the sbatch commands
    .venv/bin/python scripts/server_pipeline.py --tag v0cand                  # submit everything
    .venv/bin/python scripts/server_pipeline.py --tag v0cand --from select    # base already done: arms only
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = "python scripts/stage.py"
PHASES = ("base", "rank", "select", "arms")


class Chain:
    def __init__(self, dry: bool, after: list[str], sha: str, chain: str, partition: str):
        self.dry, self.after, self.sha, self.chain, self.partition = dry, after, sha, chain, partition
        self.n = 0

    def submit(self, script: str, cmd: str, name: str, deps: list[str] = (), array: int | None = None,
               time: str | None = None) -> str:
        """sbatch ``slurm/<script>`` with CMD; returns the job id (or a placeholder in a dry run)."""
        args = ["sbatch", "--parsable", f"--job-name={name}", f"--partition={self.partition}"]
        if array:
            args.append(f"--array=0-{array - 1}")
        if time:
            args.append(f"--time={time}")
        deps = [d for d in (*self.after, *deps) if d]
        if deps:
            args.append("--dependency=afterok:" + ":".join(deps))
        args.append(str(REPO / "slurm" / script))
        cmd = cmd.replace(PY, f"{PY} --chain {shlex.quote(self.chain)}")
        env = {"CMD": cmd, "EXPECT_SHA": self.sha, "PHYEXP_DIR": str(REPO)}
        self.n += 1
        if self.dry:
            print(f"[dry {self.n}] {name}  deps={','.join(deps) or '-'}\n    {cmd}")
            return f"J{self.n}"
        job = subprocess.run(args, env={**os.environ, **env}, check=True, capture_output=True, text=True).stdout.strip()
        job = job.split(";")[0]
        print(f"[{job}] {name}  deps={','.join(deps) or '-'}  :: {cmd}")
        return job


def arms(a) -> list[dict]:
    """The selection arms: name, select-stage arguments."""
    out = []
    for k in a.k_per_task:
        for m in a.metrics:
            out.append({"arm": f"harm-{m}-k{k}", "args": f"--op drop_top_k_per_task --k {k} --metric harm_{m}__same_task "
                        f"--direction highest --scores attribute-{a.tag}-ens"})
        for r in range(a.random_draws):
            out.append({"arm": f"random{r}-k{k}", "args": f"--op drop_random_k_per_task --k {k} --seed {r}"})
    if a.ac75:
        out.append({"arm": "drop-random-20", "args": "--op drop_random_k --k 20 --seed 0"})
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scripts/server_pipeline.py", description=__doc__.split("\n\n")[0])
    p.add_argument("--tag", required=True, help="names every run of this experiment (e.g. v0cand)")
    p.add_argument("--subset", default="v0/candidate", help="the base training subset (blind; AC8.5)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--metrics", nargs="+", default=["f30", "rmst60", "auc_expected"],
                   help="family-W returns (without ret_) whose harm ranking defines a curated arm")
    p.add_argument("--k-per-task", type=int, nargs="+", default=[5], help="episodes dropped per task (48 per task in v0/candidate)")
    p.add_argument("--random-draws", type=int, default=2, help="matched random arms per k")
    p.add_argument("--no-ac75", dest="ac75", action="store_false", help="skip AC7.5's drop-random-20 arm")
    p.add_argument("--train-shards", type=int, default=4)
    p.add_argument("--rollout-shards", type=int, default=4)
    p.add_argument("--eval-batch", type=int, default=50, help="AC5.8: 50, or one smaller divisor of 50 for every arm")
    p.add_argument("--partition", default="a100")
    p.add_argument("--from", dest="start", choices=PHASES, default="base", help="first phase to submit")
    p.add_argument("--to", dest="stop", choices=PHASES, default="arms", help="last phase to submit")
    p.add_argument("--after", default="", help="comma-separated job ids every submitted job waits for")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)

    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "vlapoc", "scripts", "slurm"], cwd=REPO,
                           check=True, capture_output=True, text=True).stdout.strip()
    if dirty and not a.dry_run:
        sys.exit(f"uncommitted changes under vlapoc/, scripts/ or slurm/; commit first:\n{dirty}")
    ch = Chain(a.dry_run, [x for x in a.after.split(",") if x], sha, f"{a.tag}-{sha[:8]}", a.partition)
    on = set(PHASES[PHASES.index(a.start): PHASES.index(a.stop) + 1])
    t, eb = a.tag, a.eval_batch

    attr_jobs: list[str] = []
    eval_base: dict[int, str] = {}
    if "base" in on:
        for s in a.seeds:
            tr = ch.submit("gpu.sbatch", f"{PY} train --subset {a.subset} --name train-{t}-s{s} --seed {s}", f"train-{t}-s{s}")
            ev = ch.submit("gpu.sbatch", f"{PY} eval --train train-{t}-s{s} --name eval-{t}-s{s} --batch-size {eb}",
                           f"eval-{t}-s{s}", [tr])
            eval_base[s] = ev
            rt = ch.submit("cpu.sbatch", f"{PY} returns --eval eval-{t}-s{s} --name returns-{t}-s{s}", f"returns-{t}-s{s}", [ev])
            ft = ch.submit("gpu.sbatch", f"{PY} features --train train-{t}-s{s} --which train --shard {{i}}/{a.train_shards}",
                           f"feat-{t}-s{s}-train", [tr], array=a.train_shards, time="1-00:00:00")
            fr = ch.submit("gpu.sbatch", f"{PY} features --train train-{t}-s{s} --which rollouts --eval eval-{t}-s{s} "
                           f"--shard {{i}}/{a.rollout_shards}", f"feat-{t}-s{s}-roll", [ev], array=a.rollout_shards,
                           time="1-00:00:00")
            attr_jobs.append(ch.submit("gpu.sbatch", f"{PY} attribute --train train-{t}-s{s} --eval eval-{t}-s{s} "
                                       f"--returns returns-{t}-s{s} --name attribute-{t}-s{s}", f"attribute-{t}-s{s}",
                                       [rt, ft, fr], time="06:00:00"))
    ens = ""
    if "rank" in on:
        runs = " ".join(f"attribute-{t}-s{s}" for s in a.seeds)
        ens = ch.submit("cpu.sbatch", f"{PY} ensemble --runs {runs} --name attribute-{t}-ens", f"ensemble-{t}", attr_jobs)
    sel = ""
    plan = arms(a)
    if "select" in on:
        cmd = " && ".join(f"{PY} select --parent {a.subset} --name select-{t}-{x['arm']} {x['args']}" for x in plan)
        sel = ch.submit("cpu.sbatch", cmd, f"select-{t}", [ens])
    if "arms" in on:
        for x in plan:
            arm = f"{t}-{x['arm']}"
            for s in a.seeds:
                tr = ch.submit("gpu.sbatch", f"{PY} train --subset select:select-{t}-{x['arm']} --name train-{arm}-s{s} --seed {s}",
                               f"train-{arm}-s{s}", [sel])
                ev = ch.submit("gpu.sbatch", f"{PY} eval --train train-{arm}-s{s} --name eval-{arm}-s{s} --batch-size {eb} "
                               "--no-trajectories", f"eval-{arm}-s{s}", [tr])
                cmd = " && ".join(f"{PY} verify --a eval-{t}-s{s} --b eval-{arm}-s{s} --family {f} --name verify-{arm}-s{s}-{f}"
                                  for f in ("W", "O"))
                ch.submit("cpu.sbatch", cmd, f"verify-{arm}-s{s}", [ev, eval_base.get(s, "")])
    print(f"{'would submit' if a.dry_run else 'submitted'} {ch.n} jobs; chain {ch.chain}; arms: {[x['arm'] for x in plan]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
