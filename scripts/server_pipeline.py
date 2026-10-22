#!/usr/bin/env python
"""Submit the whole attribution-and-curation experiment to SLURM as one dependency chain of tracked stages.

Each job runs one ``scripts/stage.py`` subcommand (one MLflow stage run), in the runbook's order
(``tracker/features/2026-10-01-evaluation-layer/artifacts/server-runbook.md``) extended by Steps 3-4:

    base    per seed S: train (AC8.5) → eval with trajectories (AC5.8) → returns (AC6.7)
            → features (train subset, sharded; rollouts, sharded; output function from ``--preset``) → attribute
    rank    ensemble of the per-seed attribution tables (z-scored mean)
    select  one select run per arm, all in one CPU job (selects are seconds; one writer for names and subsets).
            ``--curation`` (owner answer Q1, 2026-10-20: "Both, per-task headline") picks the arm families:
              per_task  harm-<m>-k<k>   drop_top_k_per_task on harm_<m>__same_task (the k most harmful per task)
                        help-<m>-k<k>   the same op, ``--direction lowest`` (the k most helpful per task)
                        random<r>-k<k>  drop_random_k_per_task, seed r: the matched control (same per-task counts)
              global    gharm-<m>-n<N>  drop_top_k on the all-rollout harm_<m> (the N most harmful overall; N = k × tasks,
                        the same total as the per-task arms): the PoC §9.1 Step 4.1 literal version
                        ghelp-<m>-n<N>  the same, ``--direction lowest``
                        grandom<r>-n<N> drop_random_k N, seed r
              and AC7.5's gate drop-random-20 (drop_random_k 20, seed 0). ``both`` (the default) emits both families,
              per-task is the reported headline.
    arms    per arm and seed: train on the child subset → eval (no trajectories; AC7.5), then the verdicts:
              vs_base            every arm vs the base eval of the same seed, family W (declared) and O (alongside)
              curated_vs_random  each harm/help arm vs each random control of its family (PoC §2.6: improve more /
                                 degrade more than after random removal), family W
              head_to_head       harm-hrt60_lin vs harm-f30 (PoC §4.5 "same dataset, opposite curations"), family W
    analyze one CPU job after the ensemble: ``stage.py analyze`` = the PoC §4.2/§4.5 groups, aggregation by task, episode x
            metric heatmap and throughput-vs-reliability scatter (``vlapoc.analysis``)
    score   (only when the base subset's version has an answer key, e.g. ``--subset v3/candidate``) one CPU job after the
            ensemble: ``stage.py score`` = recovery of the planted corruptions per type and metric, next to each type's best
            heuristic (``vlapoc.recovery``; PoC §2.1 item 4, §4.5). The injection run is ``--subset v3/candidate``.
    summary one CPU job after every verify: ``stage.py step4-summary`` (seed-consistency table, ``vlapoc.step4``)

Grounding: PoC §9.1 Steps 3-4 (p. 19): "Rank training episodes by harmful influence ... Retrain without the
selected episodes and compare against the original policy"; "5 Repeat for the top-k most helpful episodes (removing
them should make things worse)"; PoC §2.6 (p. 5): "the policy should improve more than after removing k random
episodes"; PoC §4.5 (p. 11): curate once for HRT and once for reliability and "show that each curated policy wins on
its own metric"; PoC §2.3 (p. 4): 2-3 seeds per data configuration; every arm uses the base recipe unchanged (PoC p. 1:
data is the only variable). Per-task top-k is grounded in ATHENA §3.2 p. 3 and §4.2 Eq. 7 p. 5 (not in the PoC), global
top-k is the PoC's literal Step 4.1; the owner approved building both (training-attribution brief, Q1).

Cost (inferred, see the batch-2 log): the default is large because ``--curation both`` doubles the arms. Shrink it with
``--curation per_task``, fewer ``--metrics`` / ``--seeds`` / ``--random-draws``, or ``--no-ac75``; every arm costs
(seeds) x (one train + one eval + its verifies).

Run names are ``<stage>-<tag>[-<arm>]-s<S>``; ``stage.py`` resolves them to run ids when each job starts, so the
chain can be submitted before any run exists. The chain refuses to submit from a dirty tree and every job refuses
to start on a commit other than the one it was submitted from (``slurm/guard.sh``).

Usage::

    cp slurm/site.env.example slurm/site.env                                   # once per machine; edit it
    .venv/bin/python scripts/server_pipeline.py --tag v0cand --dry-run       # print every job with its full sbatch argv
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


def load_site(path: Path) -> dict[str, str]:
    """Parse a site file (``slurm/site.env``; format of ``slurm/site.env.example``): ``KEY=VALUE`` lines, ``${KEY}`` expands
    to an earlier key, ``#`` comments. Raises ``SystemExit`` naming the missing keys (SCRATCH_ROOT, PARTITION, LOG_DIR)."""
    site: dict[str, str] = {}
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, _, v = ln.removeprefix("export ").partition("=")
        for key, val in site.items():
            v = v.replace("${" + key + "}", val)
        site[k.strip()] = v.strip().strip('"')
    site.setdefault("LOG_DIR", f"{site['SCRATCH_ROOT']}/logs" if "SCRATCH_ROOT" in site else "")
    missing = [k for k in ("SCRATCH_ROOT", "PARTITION", "LOG_DIR") if not site.get(k)]
    if missing:
        sys.exit(f"{path}: missing {missing}; see slurm/site.env.example")
    return site


class Chain:
    def __init__(self, dry: bool, after: list[str], sha: str, chain: str, partition: str, log_dir: str = "", site_file: str = ""):
        self.dry, self.after, self.sha, self.chain, self.partition = dry, after, sha, chain, partition
        self.log_dir, self.site_file = log_dir, site_file
        self.n = 0

    def submit(self, script: str, cmd: str, name: str, deps: list[str] = (), array: int | None = None,
               time: str | None = None) -> str:
        """sbatch ``slurm/<script>`` with CMD; returns the job id (or a placeholder in a dry run)."""
        args = ["sbatch", "--parsable", f"--job-name={name}", f"--partition={self.partition}",
                f"--output={self.log_dir}/{'%x-%A_%a.out' if array else '%x-%j.out'}"]
        if array:
            args.append(f"--array=0-{array - 1}")
        if time:
            args.append(f"--time={time}")
        deps = [d for d in (*self.after, *deps) if d]
        if deps:
            args.append("--dependency=afterok:" + ":".join(deps))
        args.append(str(REPO / "slurm" / script))
        cmd = cmd.replace(PY, f"{PY} --chain {shlex.quote(self.chain)}")
        env = {"CMD": cmd, "EXPECT_SHA": self.sha, "PHYEXP_DIR": str(REPO), **({"SITE_ENV": self.site_file} if self.site_file else {})}
        self.n += 1
        if self.dry:
            print(f"[dry {self.n}] {name}  deps={','.join(deps) or '-'}\n    {cmd}\n    {shlex.join(args)}")
            return f"J{self.n}"
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)  # sbatch does not create the --output directory (round 2)
        job = subprocess.run(args, env={**os.environ, **env}, check=True, capture_output=True, text=True).stdout.strip()
        job = job.split(";")[0]
        print(f"[{job}] {name}  deps={','.join(deps) or '-'}  :: {cmd}")
        return job


def has_answer_key(subset: str) -> bool:
    """Does the base subset's version carry an answer key (``labels.parquet``, i.e. an injected version such as v3)?
    The pipeline only schedules the ground-truth ``score`` stage then (PoC §2.1 item 4 p.3). Existence check only."""
    sys.path.insert(0, str(REPO))
    from vlapoc.lineage import _root, resolve_version

    try:
        vid = resolve_version(subset.split("/")[0])["dataset_version_id"]
    except Exception:  # noqa: BLE001 - no lineage store here (e.g. a fresh clone): no key
        return False
    return (_root(None) / "versions" / vid / "labels.parquet").exists()


def arms(a) -> list[dict]:
    """The selection arms. Each: ``arm`` (name), ``args`` (select-stage arguments), ``mode`` (per_task / global / ac75),
    ``kind`` (harm / help / random / ac75), ``metric`` (family-W return name or None), ``k`` (the arm's size parameter) and
    ``control_of``: the random arms an arm is compared with (same mode and k)."""
    out = []
    modes = ("per_task", "global") if a.curation == "both" else (a.curation,)
    for k in a.k_per_task:
        for mode in modes:
            pt = mode == "per_task"
            n = k if pt else k * a.n_tasks  # global arms drop the same total as the per-task arms
            pre, size = ("", f"k{k}") if pt else ("g", f"n{n}")
            op_top = "drop_top_k_per_task" if pt else "drop_top_k"
            op_rand = "drop_random_k_per_task" if pt else "drop_random_k"
            col = (lambda m: f"harm_{m}__same_task") if pt else (lambda m: f"harm_{m}")
            common = f"--pipeline-curation {a.curation}" + (" --headline" if pt and a.curation == "both" else "")
            rnd = [f"{pre}random{r}-{size}" for r in range(a.random_draws)]
            for m in a.metrics:
                for kind, direction in (("harm", "highest"), ("help", "lowest")):
                    out.append({"arm": f"{pre}{kind}-{m}-{size}", "mode": mode, "kind": kind, "metric": m, "k": n,
                                "args": f"--op {op_top} --k {n} --metric {col(m)} --direction {direction} "
                                        f"--scores attribute-{a.tag}-ens {common}", "controls": rnd})
            for r, name in enumerate(rnd):
                out.append({"arm": name, "mode": mode, "kind": "random", "metric": None, "k": n, "controls": [],
                            "args": f"--op {op_rand} --k {n} --seed {r} {common}"})
    if a.ac75:
        out.append({"arm": "drop-random-20", "mode": "ac75", "kind": "ac75", "metric": None, "k": 20, "controls": [],
                    "args": "--op drop_random_k --k 20 --seed 0"})
    return out


def head_to_head(plan: list[dict], metrics: list[str]) -> list[tuple[dict, dict]]:
    """PoC §4.5 pairs: (treated = harm-hrt60_lin arm, control = harm-f30 arm) per mode and k."""
    if "hrt60_lin" not in metrics or "f30" not in metrics:
        return []
    by = {(x["mode"], x["kind"], x["metric"], x["k"]): x for x in plan}
    return [(by[(mode, "harm", "hrt60_lin", k)], by[(mode, "harm", "f30", k)])
            for (mode, kind, m, k) in sorted(by) if kind == "harm" and m == "hrt60_lin" and (mode, "harm", "f30", k) in by]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="scripts/server_pipeline.py", description=__doc__.split("\n\n")[0])
    p.add_argument("--tag", required=True, help="names every run of this experiment (e.g. v0cand)")
    p.add_argument("--subset", default="v0/candidate", help="the base training subset (blind; AC8.5)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--metrics", nargs="+", default=["f30", "rmst60", "auc_expected", "hrt60_lin"],
                   help="family-W returns (without ret_) whose harm ranking defines a curated arm; hrt60_lin enables the "
                        "PoC §4.5 head-to-head against f30")
    p.add_argument("--k-per-task", type=int, nargs="+", default=[5], help="episodes dropped per task (48 per task in v0/candidate)")
    p.add_argument("--curation", choices=["per_task", "global", "both"], default="both",
                   help="owner answer Q1 (2026-10-20): both, per-task is the headline; global = PoC §9.1 Step 4.1 literal")
    p.add_argument("--n-tasks", type=int, default=20, help="tasks in the base subset (v0/candidate: 20); global arms drop k x this")
    p.add_argument("--random-draws", type=int, default=2, help="matched random arms per k")
    p.add_argument("--no-ac75", dest="ac75", action="store_false", help="skip AC7.5's drop-random-20 arm")
    p.add_argument("--train-shards", type=int, default=4)
    p.add_argument("--rollout-shards", type=int, default=4)
    p.add_argument("--eval-batch", type=int, default=50, help="AC5.8: 50, or one smaller divisor of 50 for every arm")
    p.add_argument("--site", type=Path, default=None,
                   help="site file with SCRATCH_ROOT, PARTITION, LOG_DIR (default slurm/site.env; a dry run falls back to "
                        "slurm/site.env.example, the cluster's values are in docs/site-profile-cluster.env)")
    p.add_argument("--partition", default=None, help="override the site file's PARTITION")
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
    site_path = a.site or (REPO / "slurm" / "site.env")
    if not site_path.exists():
        if not a.dry_run:
            sys.exit(f"{site_path} not found: copy slurm/site.env.example to slurm/site.env and edit it (docs/site-profile-cluster.env)")
        site_path = REPO / "slurm" / "site.env.example"
    site = load_site(site_path)
    ch = Chain(a.dry_run, [x for x in a.after.split(",") if x], sha, f"{a.tag}-{sha[:8]}", a.partition or site["PARTITION"],
               site["LOG_DIR"], str(a.site) if a.site else "")
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
        ch.submit("cpu.sbatch", f"{PY} analyze --scores attribute-{t}-ens --name analyze-{t}", f"analyze-{t}", [ens])  # PoC §4.2, §4.5 outputs
        if has_answer_key(a.subset):  # injection run (e.g. --subset v3/candidate): score the ranking against the planted truth
            vname, _, sname = a.subset.partition("/")
            ch.submit("cpu.sbatch", f"{PY} score --scores attribute-{t}-ens --version {vname} --subset {sname} --name score-{t}",
                      f"score-{t}", [ens])
    sel = ""
    plan = arms(a)
    if "select" in on:
        cmd = " && ".join(f"{PY} select --parent {a.subset} --name select-{t}-{x['arm']} {x['args']}" for x in plan)
        sel = ch.submit("cpu.sbatch", cmd, f"select-{t}", [ens])
    if "arms" in on:
        ev_of: dict[tuple[str, int], str] = {}
        verifies: list[str] = []
        for x in plan:
            arm = f"{t}-{x['arm']}"
            for s in a.seeds:
                tr = ch.submit("gpu.sbatch", f"{PY} train --subset select:select-{t}-{x['arm']} --name train-{arm}-s{s} --seed {s}",
                               f"train-{arm}-s{s}", [sel])
                ev = ch.submit("gpu.sbatch", f"{PY} eval --train train-{arm}-s{s} --name eval-{arm}-s{s} --batch-size {eb} "
                               "--no-trajectories", f"eval-{arm}-s{s}", [tr])
                ev_of[(x["arm"], s)] = ev
                tag = f"--comparison vs_base --treated {x['arm']} --control base --seed-id {s}"
                cmd = " && ".join(f"{PY} verify --a eval-{t}-s{s} --b eval-{arm}-s{s} --family {f} --name verify-{arm}-s{s}-{f} {tag}"
                                  for f in ("W", "O"))
                verifies.append(ch.submit("cpu.sbatch", cmd, f"verify-{arm}-s{s}", [ev, eval_base.get(s, "")]))
        for x in plan:  # PoC §2.6 / §9.1 Step 4.1-4.5: each harm and help arm against every random control of its family
            for ctrl in x["controls"]:
                for s in a.seeds:
                    name = f"{t}-{x['arm']}-vs-{ctrl}"
                    cmd = (f"{PY} verify --a eval-{t}-{ctrl}-s{s} --b eval-{t}-{x['arm']}-s{s} --family W --name verify-{name}-s{s}-W "
                           f"--comparison curated_vs_random --treated {x['arm']} --control {ctrl} --seed-id {s}")
                    verifies.append(ch.submit("cpu.sbatch", cmd, f"verify-{name}-s{s}", [ev_of[(x["arm"], s)], ev_of[(ctrl, s)]]))
        for hrt, f30 in head_to_head(plan, a.metrics):  # PoC §4.5 "same dataset, opposite curations"
            for s in a.seeds:
                name = f"{t}-{hrt['arm']}-vs-{f30['arm']}"
                cmd = (f"{PY} verify --a eval-{t}-{f30['arm']}-s{s} --b eval-{t}-{hrt['arm']}-s{s} --family W --name verify-{name}-s{s}-W "
                       f"--comparison head_to_head --treated {hrt['arm']} --control {f30['arm']} --seed-id {s}")
                verifies.append(ch.submit("cpu.sbatch", cmd, f"verify-{name}-s{s}", [ev_of[(hrt["arm"], s)], ev_of[(f30["arm"], s)]]))
        ch.submit("cpu.sbatch", f"{PY} step4-summary --tag {t} --name step4-summary-{t} --chain-id {ch.chain}", f"step4-summary-{t}", verifies)
    print(f"{'would submit' if a.dry_run else 'submitted'} {ch.n} jobs; chain {ch.chain}; arms: {[x['arm'] for x in plan]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
