---
name: server-run
description: Step-by-step runbook for a Claude Code session on the GPU/SLURM server (cluster) that runs the VLA training-data attribution pipeline end to end on real GPUs. Use whenever the task is to run, submit, monitor, recover or bring back results of the pipeline on the server, the cluster, SLURM or the GPUs - preflight, smoke, Round A (attribution on v3/candidate), clusters, the real dashboard, Round B (confirmation retraining), results back. Stops and asks the owner at every marked point.
---

# Server run: preflight -> smoke -> Round A -> clusters -> dashboard -> Round B -> results back

You are a fresh Claude Code session on the GPU server. This skill is your whole task. Read it fully before running anything. Everything in it was produced
from the repo's code and local dry runs on the owner's Mac; **nothing here has run on a server yet**. Each step is tagged **[verified locally]** (its commands
ran on the Mac with the output quoted) or **[unverified until a server session]** (it is the first time anyone runs it: report exactly what you see). Expected
outputs are labelled (measured), (quoted) or (inferred).

Project background (read, do not re-derive): `CLAUDE.md`; the spec is `docs/VLA_Data_Attribution_PoC.pdf`; this feature's record is
`tracker/features/2026-10-22-dashboard/`; the earlier server notes are `tracker/features/2026-10-01-evaluation-layer/artifacts/server-runbook.md`.

## Rules (they override everything below)

1. **No code edits.** Do not edit anything under `vlapoc/`, `scripts/`, `slurm/` or `dashboard/`. If something needs a code change, that is a stop-and-ask.
2. **No `git push`, no merge, no `git commit`** except where a step names one. No deletes: never `rm -rf` data, runs, checkpoints, lineage or logs. Never `scancel` a job
   you did not submit in this session without asking.
3. **Stop and ask the owner** at every **STOP AND ASK** marker below, and at anything odd: a failed check, a failed job, a number that looks wrong, low disk, a banner you did not
   expect. Say what you ran, what you saw, and what you think it means, then wait.
4. **Label every claim you write** with how you know it: quoted (a source and its text), measured (a command you ran, with its output), or inferred (your reasoning, unverified).
   A claim without a label is treated as a guess. Never write that attribution "worked" or a finding is "real" from a single number.
5. Scratch files go under `data/scratch/`, never `/tmp`. Keep at least 20 GiB free on the scratch disk plus the estimate of the step you are about to run.
6. A tracked stage refuses to start from a dirty tree: `git status --porcelain -- vlapoc scripts slurm` must be empty. Do not "fix" it; ask.

## Step 1: Preflight [unverified until a server session]

Run these and compare with the expected values. Every command is read-only.

```bash
git rev-parse HEAD
git status --porcelain -- vlapoc scripts slurm dashboard
python --version
.venv/bin/python -m pip check
nvidia-smi
df -h .
ls slurm/site.env
```

Expected (inferred): `HEAD` equals the commit the owner names in the request (if the owner named none, ask); the status output is empty; Python is 3.12 or 3.13
(quoted, evaluation-layer server runbook: "lerobot 0.6.1 declares 3.12/3.13 only"); `pip check` prints `No broken requirements found.`; `nvidia-smi` lists the GPUs the
job partition uses; at least 100 GiB free on the scratch filesystem (Round A needs about 60-95 GiB, inferred from the evaluation-layer runbook's totals);
`slurm/site.env` exists (create it from `slurm/site.env.example`; the owner's cluster values are in `docs/site-profile-cluster.env`).

**STOP AND ASK (preflight failure):** any of: wrong commit, a dirty tree, Python outside 3.12-3.13, `pip check` errors, no GPU, free disk below 100 GiB, no `slurm/site.env`.

Install what the dashboard and the cluster stage add to the evaluation-layer environment (pure wheels, measured to install on the Mac's Python 3.14):

```bash
.venv/bin/python -m pip install streamlit==1.66.1 sliceline==0.3.0
```

## Step 2: Inputs and environment [unverified until a server session]

Copy from the owner's machine (sizes measured on the Mac): the v0 dataset `data/lerobot/libero_spatial_10/` (6.9 GiB) **and the v3 dataset
`data/lerobot/libero_spatial_10__v3/`** (the injected version Round A trains on; about the same size), `data/lineage/` (the store; copy it, append on the server, copy it back),
`data/eval/human_reference/`, `data/libero_assets/`, `data/libero_config/`, `data/hf_models/smolvla_base/`, `data/hf_home/`. Then the environment of the evaluation-layer runbook:

```bash
source slurm/env.sh
export LIBERO_CONFIG_PATH=$PWD/data/libero_config
export HF_HOME=$PWD/data/hf_home
export MLFLOW_DISABLE_AGENT_HINT=1
.venv/bin/python -m vlapoc.lineage version verify v3
.venv/bin/python -m vlapoc.lineage version verify v0
```

Expected (inferred): both verifications report every file matching its recorded sha256 (about 3 s each in the data+lineage record, measured on the Mac); the lineage store was
copied with its absolute MLflow artifact paths from the Mac, so new runs on the server write server paths (this matters in Step 14).

Check the `smolvla_base` revision. The code pins `lerobot/smolvla_base@d9f33c94a60fb382c90dea2164c96845bd955e28` (`scripts/stage.py` `PRETRAINED_REF`); the cluster's Hugging Face cache may
hold a different one (`5e8d12a6`, quoted from the owner's brief):

```bash
cat data/hf_models/smolvla_base/config.json | head -5
ls data/hf_models/smolvla_base
```

**STOP AND ASK (revision mismatch):** if the weights you will train from are not the pinned `d9f33c94...` revision (or you cannot show they are), stop: runs trained from another
revision are not comparable with anything recorded. Do not download or swap weights yourself.

## Step 3: Smoke on the server [unverified until a server session]

The same chain passes on the Mac (CPU for the first command, MPS for the second; measured: the eval-layer record gives 122.6 s for the first, and the dashboard demo build ran the second chain's
stages in 489 s on MPS). On the server run them on CUDA:

```bash
.venv/bin/python scripts/lineage_dry_run.py --eval-smoke
.venv/bin/python scripts/verify_training_attribution.py --local-smoke --device cuda
```

Expected (inferred for the server; the Mac outputs are measured): the first ends with a `trace` that reaches 20 raw files and no stand-in; the second finishes six stages and prints a
score table with 40 rows (`n_episodes 40`, the v0/reference subset) and `all_finite_harm_columns True`. Then the throughput probe and the two-process determinism check
(evaluation-layer runbook rules 4 and 5; `<run>` is a train run id the smoke created):

```bash
.venv/bin/python scripts/eval_smoke.py --policy train:<run> --tasks libero_spatial:0 --n 10 --batch-size 10 --tau 30 --device cuda --out data/scratch/tp
```

Run that probe twice (two processes) and compare the two `rollouts.parquet` files as the evaluation-layer runbook rule 5 says; expect no differing columns (inferred: unmeasured on CUDA).
Write the seconds per env step down: they size every later step.

**STOP AND ASK (failed check):** a smoke stage fails, the trace shows a stand-in or fewer than 20 raw files, or the two probes differ.

## Step 4: Round A dry run and the count [verified locally]

Round A is the planted-bad-data version `v3/candidate` (owner answer 2026-10-22), per-task curation with metrics F30 and HRT. Print the job list first; do not submit yet:

```bash
.venv/bin/python scripts/server_pipeline.py --tag v3cand --subset v3/candidate --curation per_task --metrics f30 hrt60_lin --dry-run
```

Expected (measured, `scripts/server_pipeline.py --dry-run` on the Mac): the last line reads `would submit 116 jobs; chain v3cand-<sha8>; arms: [harm-f30-k5, help-f30-k5, harm-hrt60_lin-k5, help-hrt60_lin-k5, random0-k5, random1-k5, drop-random-20]`.
<!-- count: round_a_v3 116 24 -->
**Round A (v3): 116 jobs, 24 of them training runs** (3 base seeds + 7 arms x 3 seeds), about 96 training GPU-hours (inferred: 4 h per training, quoted LeRobot SmolVLA docs). After the ensemble
the chain also runs the CPU jobs `analyze`, `clusters`, `score`, `score-clusters` and `discrepancy`. The clean variant (`--subset v0/candidate --tag v0cand`) is a labelled alternative, not the plan:
<!-- count: round_a_v0_variant 113 24 -->
113 jobs, 24 trainings (measured).

Report the count to the owner. **STOP AND ASK (odd numbers):** if your dry run prints any other job or training count, stop before submitting.

## Step 5: Round A submit [unverified until a server session]

```bash
.venv/bin/python scripts/server_pipeline.py --tag v3cand --subset v3/candidate --curation per_task --metrics f30 hrt60_lin
```

Expected (inferred): 116 `[<job id>] <name> deps=... :: ...` lines and `submitted 116 jobs; chain v3cand-<sha8>`. Write down the chain id and the first and last job ids. The pipeline refuses a dirty tree and every job
refuses to start on another commit (`slurm/guard.sh`); do not work around either.

## Step 6: Monitor [unverified until a server session]

```bash
squeue -u $USER
sacct -j <first job id>,<last job id> --format=JobID,JobName,State,Elapsed
ls <LOG_DIR from slurm/site.env> | tail
```

and a short MLflow query for the stage runs finished so far (read-only):

```bash
.venv/bin/python -c "
from vlapoc import results
for stage in ('train','eval','returns','attribute','cluster','score','summarize'):
    print(stage, [r['name'] for r in results.find_runs(stage, chain='v3cand-')][:6], len(results.find_runs(stage, chain='v3cand-')))"
```

Healthy looks like (inferred): trainings run about 4 h each (quoted: "roughly ~4 hrs on a single A100"), evals of the base policies with trajectories take the longest, feature jobs are array jobs, the CPU jobs after the
ensemble take minutes (the cluster stage 5 s plus 85 s once for the frame scan, measured on the Mac). Check disk after each phase: `df -h .`.

**STOP AND ASK (failed job):** any job in `FAILED`, `TIMEOUT`, `OUT_OF_MEMORY` or `CANCELLED`: collect `tail -50` of its log, the `sacct` line and the MLflow run state, report them, and wait. Do not resubmit.
**STOP AND ASK (disk below threshold):** free disk under 20 GiB plus the next phase's estimate (trajectories 11-22 GB per base seed, inferred in the evaluation-layer runbook).

## Step 7: Recovery [unverified until a server session]

Only after the owner agrees. The chain's phases are `base`, `rank`, `select`, `arms`; a finished base can be reused:

```bash
.venv/bin/python scripts/server_pipeline.py --tag v3cand --subset v3/candidate --curation per_task --metrics f30 hrt60_lin --from select --dry-run
```

An interrupted job leaves its MLflow run RUNNING; close it with `MlflowClient().set_terminated(run_id, "KILLED")` plus a tag saying why (evaluation-layer runbook rule 3), only on the owner's yes. Never merge partial evals (the batch plan is part of the protocol).

**STOP AND ASK (any needed code change):** if a failure looks like a bug (a traceback inside `vlapoc/` or `scripts/`), report it; do not patch it.

## Step 8: Clusters, planted-type scoring and the discrepancy table, by hand [verified locally]

The chain already runs these as CPU jobs after the ensemble. Run them by hand only if those jobs failed or you are asked to re-run them (they only read the score table and write new runs):

```bash
.venv/bin/python scripts/stage.py --chain v3cand-<sha8> clusters --scores attribute-v3cand-ens --name clusters-v3cand
.venv/bin/python scripts/stage.py --chain v3cand-<sha8> score-clusters --clusters clusters-v3cand --version v3 --subset candidate --name score-clusters-v3cand
.venv/bin/python scripts/stage.py --chain v3cand-<sha8> discrepancy --scores attribute-v3cand-ens --version v3 --subset candidate --name discrepancy-v3cand
```

Expected (measured on SYNTHETIC scores on the Mac): `clusters` prints a JSON with `n_clusters` around 80-100 and `n_by_lens` for `label`, `slice`, `metric_group`; `score-clusters` prints one row per (cluster, planted type) incl. `any_planted`;
`discrepancy` prints the per-metric quadrant counts. Cluster discovery reads the data only through the blind loader; the two ground-truth stages read the answer key afterwards. Every cluster is **estimated** at this point.

## Step 9: The real dashboard [verified locally]

The app was run headless against the demo store on the Mac (every page rendered with no exception, `http://localhost:<port>/_stcore/health` returned 200, measured). On the server:

```bash
.venv/bin/streamlit run dashboard/app.py --server.headless true --server.port 8501 -- --lineage-root data/lineage --scores attribute-v3cand-ens --clusters clusters-v3cand
```

View it through an SSH tunnel from the owner's machine: `ssh -L 8501:localhost:8501 <user>@<server>` then open `http://localhost:8501` (the tunnel is [unverified until a server session]).
Expected (measured on the Mac with the demo store, `scripts/verify_dashboard.py --app`): the five pages (Overview, Rollouts, Episodes, Clusters, Bad vs harmful) load with no exception, the health URL returns 200 `ok`, and a SYNTHETIC banner shows on the demo. Expected (inferred) on the server: **no SYNTHETIC or SMOKE banner** on the real ensemble (the Overview page shows `Provenance: REAL`); the five pages load; the Rollouts page needs a per-seed run (the ensemble has no matrix of its own) and plays rollout
video only where the eval's trajectories exist on the machine that runs the app.

**STOP AND ASK (SYNTHETIC or SMOKE banner on real results):** if a banner appears on the real ensemble, stop and report the listed reasons verbatim (the banner is computed from the recorded runs; do not try to remove it).

## Step 10: Round B for discovered clusters, dry run [verified locally]

Round B retrains without a cluster and asks the two-step verdict whether removing it helps. The owner picks the clusters (from the Clusters page of the dashboard). Before they pick, you may show the cost with placeholder ids:

```bash
.venv/bin/python scripts/server_pipeline.py --tag v3cand --subset v3/candidate --round-b --clusters clusters-v3cand --pick 'slice:hrt60_lin:harm:1' 'label:task_name=<task name>' --dry-run
.venv/bin/python scripts/server_pipeline.py --tag v3cand --subset v3/candidate --round-b --clusters clusters-v3cand --pick 'slice:hrt60_lin:harm:1' 'label:task_name=<task name>' --matched-random --dry-run
```

Expected (measured, `--dry-run` on the Mac):
<!-- count: round_b_clusters_2 21 6 -->
two picked clusters: **21 jobs, 6 trainings** (about 24 GPU-h, inferred at 4 h per training);
<!-- count: round_b_clusters_2_matched 45 12 -->
with `--matched-random`: **45 jobs, 12 trainings** (about 48 GPU-h, inferred). (The plan's arithmetic said 33 jobs for the second; the chain's own per-arm pattern gives 45 because each matched arm also gets a vs_base verify and each cluster a verify against its matched arm.)

## Step 11: Round B for clusters, submit and confirm [unverified until a server session]

**STOP AND ASK (the Round B cluster pick):** do not choose clusters yourself. Show the owner the Clusters page, quote each candidate's definition, n and estimated reading (labelled estimated), the
cost above, and wait for the exact `--pick` ids. A cluster that holds every episode of a task is refused unless the owner says `--allow-whole-task` (that is leave-one-task-out, not a pattern).

```bash
.venv/bin/python scripts/server_pipeline.py --tag v3cand --subset v3/candidate --round-b --clusters clusters-v3cand --pick <ids the owner gave> [--matched-random]
```

The chain ends with `step4-summary` and `confirm`. Expected (inferred): after all jobs finish, `confirm-v3cand` exists and the dashboard's Clusters page shows `confirmed`, `refuted` or `not confirmed` per picked cluster (rule, stated before any result:
confirmed if the expectation is met in at least 2 of 3 seeds with the macro-KS gate fired and contradicted in none). Report the status table; do not call a cluster "confirmed" unless the `confirm` run says so.

## Step 12: Round B for the planted data, dry run [verified locally]

This tests the section 9.2 claim causally: remove each planted type, a clean-but-harmful group, and clean random controls, plus the all-planted oracle as the positive control. v3 only.

```bash
.venv/bin/python scripts/server_pipeline.py --tag v3cand --subset v3/candidate --round-b-planted --dry-run
.venv/bin/python scripts/server_pipeline.py --tag v3cand --subset v3/candidate --round-b-planted --no-oracle --dry-run
```

Expected (measured, `--dry-run` on the Mac):
<!-- count: round_b_planted 129 33 -->
default (owner option a): **129 jobs, 33 trainings** (about 132 GPU-h, inferred at 4 h per training);
<!-- count: round_b_planted_no_oracle 108 27 -->
`--no-oracle`: **108 jobs, 27 trainings** (about 108 GPU-h, inferred). `--random-draws 2` adds 33 jobs and 3 trainings (measured: 162 jobs, 36 trainings).

## Step 13: Round B planted, submit and verdict [unverified until a server session]

**STOP AND ASK (cost approval):** show the owner the cost table (33 trainings, about 132 GPU-h inferred) and submit only after an explicit yes.

```bash
.venv/bin/python scripts/server_pipeline.py --tag v3cand --subset v3/candidate --round-b-planted
```

Expected (inferred): the chain ends with `step4-summary` and `discrepancy-verdict`; the dashboard's "Bad vs harmful" page then lists a status and a sentence per arm and metric. **Report the sentences exactly as the
`discrepancy-verdict` run wrote them.** The wording rule was fixed before any result: "harmful (confirmed)", "helpful (confirmed)", "no detectable harm, bounded" (only when the round has a positive control, with the union
of the confidence intervals and "This is not proof of no harm."), or "inconclusive" (and every arm is inconclusive if no arm in the round shows an effect, because 13 episodes are 1.4% of the data and the test may lack power). Never write that
a type is "harmless", and never turn a non-significant gate into "no effect".

## Step 14: Results back [unverified until a server session]

Copy to the owner's Mac (never delete on the server): the lineage store including `mlflow.db` and `mlartifacts/`, `data/attribution/<tag>*`, `data/eval/returns-*` and the `rollouts.parquet` of each eval, but **not** the eval `trajectories/` (11-22 GB per base seed, inferred in the evaluation-layer runbook):

```bash
rsync -a --info=progress2 <user>@<server>:<repo>/data/lineage/ data/lineage_from_server/
rsync -a <user>@<server>:<repo>/data/attribution/ data/attribution_from_server/
shasum -a 256 data/lineage_from_server/mlflow.db
```

then compare that sha256 with the same command run on the server (they must be equal). MLflow stores **absolute** artifact paths in `mlflow.db`, so the copied store still points at the server's directories: rewrite them on the Mac
(measured to make a copied demo store read from its new location; it keeps a `mlflow.db.bak`):

```bash
.venv/bin/python scripts/repoint_lineage_store.py --root data/lineage_from_server --old <server repo>/data/lineage --new $PWD/data/lineage_from_server --dry-run
.venv/bin/python scripts/repoint_lineage_store.py --root data/lineage_from_server --old <server repo>/data/lineage --new $PWD/data/lineage_from_server
.venv/bin/streamlit run dashboard/app.py -- --lineage-root data/lineage_from_server --scores attribute-v3cand-ens --clusters clusters-v3cand
```

Expected (inferred): the dashboard shows the real results with no banner; rollout video reads "not on this machine" for rollouts whose trajectories were not copied.

## Step 15: Known server-only risks [unverified until a server session]

- **SQLite locking** (inferred): MLflow's store is a SQLite file and many SLURM jobs write to it concurrently. Symptom: `database is locked` (or `OperationalError`) in a job log. **STOP AND ASK (database is locked):** stop, report the job names and the log lines, do not retry blindly and do not copy or "repair" `mlflow.db`.
- **`smolvla_base` revision** `5e8d12a6` in the cluster cache vs the pinned `d9f33c94` (Step 2).
- **CUDA nondeterminism** (inferred): the two-process probe in Step 3 is the check; report differences, do not hide them.
- **HF datasets cache** about 7 GB per distinct training subset (measured on the Mac); Round A, Round B arms and planted arms each add subsets. Put `HF_DATASETS_CACHE` on the big disk (`slurm/env.sh` does it from `SCRATCH_ROOT`).
- **Queue limits** (inferred): 24 + 6 to 12 + 27 to 33 trainings at about 4 h each may not fit one allocation; ask before splitting across submissions.
- **Trajectory size**: 11-22 GB per base seed (inferred); the arms' evals run with `--no-trajectories`.

Finish by writing up (with labels) what ran, what you saw at each step, every stop you made, and the exact commit, chain ids and run names, so the owner can reproduce it.
