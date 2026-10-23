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
    clusters (always) one CPU job after the ensemble: ``stage.py clusters`` (dashboard S2: label groups, rule-like slices, the four metric groups, each with a
            pattern and an *estimated* effect); with an answer key also ``score-clusters`` (S3, after ``clusters``) and ``discrepancy`` (S9, PoC §9.2)
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

Migration note: feature files written before training-attribution round 2 (an absolute or other-version ``dataset_root`` in
their manifest) are refused by ``attribute_run``; regenerate them with ``stage.py features`` (this chain does it in its ``feat``
jobs). Editing a manifest's ``dataset_root`` by hand defeats the check.

Run names are ``<stage>-<tag>[-<arm>]-s<S>``; ``stage.py`` resolves them to run ids when each job starts, so the
chain can be submitted before any run exists. The chain refuses to submit from a dirty tree and every job refuses
to start on a commit other than the one it was submitted from (``slurm/guard.sh``).

FIRST SERVER RUN (owner answers 2026-10-22: Round A is the planted-bad-data ``v3/candidate``; the earlier ruling "112 jobs, ~96 GPU-h" on ``v0/candidate``
is replaced, dashboard AC8.2b): per-task curation with metrics F30 and HRT only::

    .venv/bin/python scripts/server_pipeline.py --tag v3cand --subset v3/candidate --curation per_task --metrics f30 hrt60_lin

That is 116 jobs (measured by ``--dry-run``), 24 of them training runs (3 base seeds + 7 arms x 3 seeds), about 96 training GPU-hours (inferred: 4 h per
train, runbook; eval time not included). On v3 the chain also runs the CPU jobs ``clusters`` (dashboard S2), ``score`` (recovery), ``score-clusters`` (S3)
and ``discrepancy`` (S9) after the ensemble: the measured 113 jobs of the plan's first count plus ``clusters`` + ``score-clusters`` + ``discrepancy``.
The clean variant ``--subset v0/candidate`` (``--tag v0cand``) is 113 jobs (112 plus the ``clusters`` job), 24 trainings. The script's own defaults are unchanged
(``--curation both``, four metrics, ``v0/candidate``: 314 jobs); the owner ruled the size of the first run, not the default. Global curation
(``--curation global``, the PoC §9.1 Step 4.1 literal version) runs later, reusing the finished base with ``--from select``.
``scripts/verify_training_attribution.py --pipeline`` checks these counts and writes the dry run to ``artifacts/s4_dry_run_first_run.txt``.

ROUND B (dashboard S4 and S10; PoC §9.1 Step 4 p.19, §2.1 item 5 p.3, §9.2 p.19), each its own chain after Round A, same ``--tag`` (names resolve within ``<tag>-``)::

    # discovered clusters: the owner picks cluster ids (the dashboard lists them); optional matched random arm per cluster
    .venv/bin/python scripts/server_pipeline.py --tag v3cand --subset v3/candidate --round-b --clusters clusters-v3cand \\
        --pick 'slice:hrt60_lin:harm:1' 'label:task_name=<task>' [--matched-random]
    # the planted data (v3 only; owner answer Q2 option a): 7 type arms, cleanharm-f30, cleanrandom0, the oracle and its control
    .venv/bin/python scripts/server_pipeline.py --tag v3cand --subset v3/candidate --round-b-planted [--no-oracle] [--random-draws N]

``--round-b``: per picked cluster one ``select --op drop_cluster`` (and with ``--matched-random`` a ``drop_random_matched`` arm with the same per-task counts),
then per arm and seed train -> eval (no trajectories) -> ``verify`` vs the base policy (family W declared, O alongside), per cluster a curated_vs_random
verify against its matched arm, ``step4-summary`` and ``confirm`` (status confirmed / refuted / not confirmed per cluster and metric, ``vlapoc.confirm``).
Two clusters: 21 jobs, 6 trainings; with ``--matched-random``: 45 jobs, 12 trainings (measured; each matched arm also gets a vs_base verify).
``--round-b-planted``: ground-truth-side arms (``select-planted``, trained with ``train --groundtruth-arm``; every method entry point refuses them), each type
and the clean-but-harmful arm against each clean random control, the oracle against its control, ``step4-summary`` and ``discrepancy-verdict`` (the bounded
sentences of ``vlapoc.planted_arms``). Default: 11 arms, 33 trainings, 129 jobs (~132 GPU-h inferred at 4 h per training); ``--no-oracle``: 27 trainings,
108 jobs; ``--random-draws 2``: 36 trainings, 162 jobs. All counts measured by ``--dry-run``.

Usage::

    cp slurm/site.env.example slurm/site.env                                   # once per machine; edit it
    .venv/bin/python scripts/server_pipeline.py --tag v3cand --subset v3/candidate --curation per_task --metrics f30 hrt60_lin --dry-run  # print every job
    .venv/bin/python scripts/server_pipeline.py --tag v3cand --subset v3/candidate --curation per_task --metrics f30 hrt60_lin          # submit everything
    .venv/bin/python scripts/server_pipeline.py --tag v3cand --subset v3/candidate --from select    # base already done: arms only
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
    """Submits jobs with ``sbatch``, each depending on the job ids it is given; ``dry`` prints the full argv instead."""

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


PLANTED_TYPES = ("action_noise", "detour", "dropped_frames", "gripper_release", "idle_padding", "instruction_swap", "time_shift")  # the v3 injection's types


def _retrain_arm(ch: Chain, a, arm: str, sel: str, gt: bool, ev_of: dict, verifies: list, eval_base: dict) -> None:
    """Jobs of one retrained arm per seed: train on ``select:select-<tag>-<arm>`` (``--groundtruth-arm`` for planted arms), eval without trajectories,
    and the vs_base verdicts (family W declared, O alongside), as in the Round A chain's ``arms`` phase."""
    t, eb = a.tag, a.eval_batch
    for s in a.seeds:
        tr = ch.submit("gpu.sbatch", f"{PY} train --subset select:select-{t}-{arm} --name train-{t}-{arm}-s{s} --seed {s}" + (" --groundtruth-arm" if gt else ""),
                       f"train-{t}-{arm}-s{s}", [sel])
        ev = ch.submit("gpu.sbatch", f"{PY} eval --train train-{t}-{arm}-s{s} --name eval-{t}-{arm}-s{s} --batch-size {eb} --no-trajectories",
                       f"eval-{t}-{arm}-s{s}", [tr])
        ev_of[(arm, s)] = ev
        tag = f"--comparison vs_base --treated {arm} --control base --seed-id {s}"
        cmd = " && ".join(f"{PY} verify --a eval-{t}-s{s} --b eval-{t}-{arm}-s{s} --family {f} --name verify-{t}-{arm}-s{s}-{f} {tag}" for f in ("W", "O"))
        verifies.append(ch.submit("cpu.sbatch", cmd, f"verify-{t}-{arm}-s{s}", [ev, eval_base.get(s, "")]))


def _vs_random(ch: Chain, a, arm: str, ctrl: str, ev_of: dict, verifies: list) -> None:
    """Per seed: ``arm`` (treated, B) against ``ctrl`` (control, A), kind curated_vs_random, family W (PoC §2.6)."""
    t = a.tag
    for s in a.seeds:
        cmd = (f"{PY} verify --a eval-{t}-{ctrl}-s{s} --b eval-{t}-{arm}-s{s} --family W --name verify-{t}-{arm}-vs-{ctrl}-s{s}-W "
               f"--comparison curated_vs_random --treated {arm} --control {ctrl} --seed-id {s}")
        verifies.append(ch.submit("cpu.sbatch", cmd, f"verify-{t}-{arm}-vs-{ctrl}-s{s}", [ev_of[(arm, s)], ev_of[(ctrl, s)]]))


def round_b(ch: Chain, a) -> list[str]:
    """Round B for discovered clusters (dashboard S4; PoC §2.1 item 5 p.3 leave-group-out, §9.1 Step 4 p.19): per picked cluster one ``drop_cluster``
    select, 3 seeds of train -> eval (no trajectories) -> vs_base verify, optionally a matched random arm (same per-task counts, ``--matched-random``)
    with its own arms and a curated_vs_random verify, then ``step4-summary`` and ``confirm``. Returns the arm names."""
    t = a.tag
    if not a.clusters or not a.pick:
        sys.exit("--round-b needs --clusters <cluster run name> and --pick <cluster_id> ...")
    arms_, cmds = [], []
    wt = " --allow-whole-task" if a.allow_whole_task else ""
    for i, cid in enumerate(a.pick):
        base = f"--parent {a.subset} --clusters {a.clusters} --cluster-id {shlex.quote(cid)}{wt}"
        cmds.append(f"{PY} select --name select-{t}-cluster{i} --op drop_cluster {base}")
        arms_.append(f"cluster{i}")
        if a.matched_random:
            cmds.append(f"{PY} select --name select-{t}-crandom{i} --op drop_random_matched --seed 0 {base}")
            arms_.append(f"crandom{i}")
    sel = ch.submit("cpu.sbatch", " && ".join(cmds), f"select-{t}-roundb")
    ev_of: dict = {}
    verifies: list[str] = []
    eval_base = {s: "" for s in a.seeds}
    for arm in arms_:
        _retrain_arm(ch, a, arm, sel, False, ev_of, verifies, eval_base)
    if a.matched_random:
        for i in range(len(a.pick)):
            _vs_random(ch, a, f"cluster{i}", f"crandom{i}", ev_of, verifies)
    sm = ch.submit("cpu.sbatch", f"{PY} step4-summary --tag {t} --name step4-summary-{t}-roundb --chain-id {ch.chain}", f"step4-summary-{t}-roundb", verifies)
    ch.submit("cpu.sbatch", f"{PY} confirm --clusters {a.clusters} --tag {t} --name confirm-{t} --chain-id {ch.chain}", f"confirm-{t}", [sm])
    return arms_


def round_b_planted(ch: Chain, a) -> list[str]:
    """Round B for the planted data (dashboard S10; PoC §9.2 p.19): ground-truth-side removal arms of ``v3/candidate``: ``type-<T>`` per planted type,
    ``cleanharm-<m>`` (the most harmful clean episodes), ``cleanrandom<r>`` (13 clean episodes: the control), and unless ``--no-oracle`` the oracle (all
    planted episodes) with ``cleanrandom91-0`` (its control). Each arm: ``select-planted``, 3 seeds of ``train --groundtruth-arm`` -> eval -> vs_base verify;
    each type / cleanharm arm against each cleanrandom (curated_vs_random), the oracle against its control; then ``step4-summary`` and ``discrepancy-verdict``."""
    t = a.tag
    cmds, effect, ctrls = [], [], []
    for ty in a.types:
        cmds.append(f"{PY} select-planted --parent {a.subset} --kind type --type {ty} --name select-{t}-type-{ty}")
        effect.append(f"type-{ty}")
    for m in a.clean_harm_metrics:
        cmds.append(f"{PY} select-planted --parent {a.subset} --kind cleanharm --metric {m} --scores attribute-{t}-ens --name select-{t}-cleanharm-{m}")
        effect.append(f"cleanharm-{m}")
    for r in range(a.random_draws):
        cmds.append(f"{PY} select-planted --parent {a.subset} --kind cleanrandom --seed {r} --name select-{t}-cleanrandom{r}")
        ctrls.append(f"cleanrandom{r}")
    oracle = []
    if a.oracle:
        cmds.append(f"{PY} select-planted --parent {a.subset} --kind oracle --name select-{t}-oracle")
        cmds.append(f"{PY} select-planted --parent {a.subset} --kind cleanrandom_oracle --seed 0 --name select-{t}-cleanrandom91-0")
        oracle = ["oracle", "cleanrandom91-0"]
    sel = ch.submit("cpu.sbatch", " && ".join(cmds), f"select-{t}-planted")
    ev_of: dict = {}
    verifies: list[str] = []
    eval_base = {s: "" for s in a.seeds}
    all_arms = [*effect, *ctrls, *oracle]
    for arm in all_arms:
        _retrain_arm(ch, a, arm, sel, True, ev_of, verifies, eval_base)
    for arm in effect:
        for c in ctrls:
            _vs_random(ch, a, arm, c, ev_of, verifies)
    if a.oracle:
        _vs_random(ch, a, "oracle", "cleanrandom91-0", ev_of, verifies)
    ch.submit("cpu.sbatch", f"{PY} step4-summary --tag {t} --name step4-summary-{t}-planted --chain-id {ch.chain}", f"step4-summary-{t}-planted", verifies)
    ch.submit("cpu.sbatch", f"{PY} discrepancy-verdict --tag {t} --name discrepancy-verdict-{t} --chain-id {ch.chain}", f"discrepancy-verdict-{t}", verifies)
    return all_arms


def main(argv: list[str] | None = None) -> int:
    """Parse the flags, build the arm lists, and submit (or dry-run print) every job in dependency order."""
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
    p.add_argument("--random-draws", type=int, default=None, help="matched random arms per k (default 2); with --round-b-planted the number of cleanrandom "
                   "controls (default 1)")
    p.add_argument("--round-b", action="store_true", help="dashboard S4: confirmation arms for discovered clusters (needs --clusters and --pick)")
    p.add_argument("--clusters", default=None, help="--round-b: the cluster run (name) whose clusters to remove, e.g. clusters-v3cand")
    p.add_argument("--pick", nargs="+", default=[], help="--round-b: cluster ids to remove, e.g. 'slice:hrt60_lin:harm:1' (the owner picks; see the dashboard)")
    p.add_argument("--matched-random", action="store_true", help="--round-b: also train a matched random arm per picked cluster (needed to claim 'more than random removal')")
    p.add_argument("--allow-whole-task", action="store_true", help="--round-b: allow a cluster that holds every episode of a task (leave-one-task-out)")
    p.add_argument("--round-b-planted", action="store_true", help="dashboard S10 (v3 only): planted-type arms, a clean-but-harmful arm, clean random controls, the oracle")
    p.add_argument("--types", nargs="+", default=list(PLANTED_TYPES), help="--round-b-planted: the planted corruption types")
    p.add_argument("--clean-harm-metrics", nargs="+", default=["f30"], help="--round-b-planted: metrics for the clean-but-harmful arms (harm_<m>__same_task)")
    p.add_argument("--no-oracle", dest="oracle", action="store_false", help="--round-b-planted: skip the oracle arm and its matched control (the positive control)")
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
    if a.round_b and a.round_b_planted:
        sys.exit("--round-b and --round-b-planted are separate chains; submit one at a time")
    if a.random_draws is None:
        a.random_draws = 1 if a.round_b_planted else 2
    if a.round_b_planted and not has_answer_key(a.subset):
        sys.exit(f"--round-b-planted needs a subset of a version with an answer key (e.g. v3/candidate); {a.subset!r} has none")
    if a.round_b or a.round_b_planted:
        names = round_b(ch, a) if a.round_b else round_b_planted(ch, a)
        n_train = sum(1 for _ in names) * len(a.seeds)
        print(f"{'would submit' if a.dry_run else 'submitted'} {ch.n} jobs; chain {ch.chain}; {n_train} training runs; arms: {names}")
        return 0
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
        cl = ch.submit("cpu.sbatch", f"{PY} clusters --scores attribute-{t}-ens --name clusters-{t}", f"clusters-{t}", [ens])  # dashboard S2: clusters + patterns
        if has_answer_key(a.subset):  # injection run (e.g. --subset v3/candidate): score the ranking against the planted truth
            vname, _, sname = a.subset.partition("/")
            ch.submit("cpu.sbatch", f"{PY} score --scores attribute-{t}-ens --version {vname} --subset {sname} --name score-{t}",
                      f"score-{t}", [ens])
            ch.submit("cpu.sbatch", f"{PY} score-clusters --clusters clusters-{t} --version {vname} --subset {sname} --name score-clusters-{t}",
                      f"score-clusters-{t}", [cl])  # dashboard S3: do the discovered clusters land on the planted types?
            ch.submit("cpu.sbatch", f"{PY} discrepancy --scores attribute-{t}-ens --version {vname} --subset {sname} --name discrepancy-{t}",
                      f"discrepancy-{t}", [ens])  # dashboard S9: PoC §9.2, where "looks bad" and "hurts" disagree (estimated)
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
