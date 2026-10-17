# Ground truth (injection experiment): server runbook (pending-server items), round 2: version v3

Written by the executor (round 1, batch 2, 2026-10-15) for v2; **rewritten in round 2 (2026-10-17) for v3**: owner ruling A changed `time_shift` to a trim, v3 replaced v2, and v2 is fenced ("superseded by v3") and must not be used. This runbook is **separate** from the evaluation feature's
`tracker/features/2026-10-01-evaluation-layer/artifacts/server-runbook.md` and does not edit it; it reuses that file's
environment section (section 0), operational rules and its AC8.5 / AC5.8 / AC6.7 commands, and only swaps in the
injected data. Every command exists in the repo at the commit that adds this file. Each number is labelled
**measured** (on this Mac, artifact named), **quoted** (a source) or **inferred** (not run anywhere).
**Nothing in this file has been run on a server.** No item here is an acceptance condition of this feature: they are
all "pending server", and their outcomes are findings.

## What is already done locally (measured)

| Fact | Evidence |
|---|---|
| `v3` = parent v0 with 91 injected candidates (13 each of 7 types, clipped to [-1, 1]), `check_blind(v3) = []` | `artifacts/s4_v3_run.txt` |
| v3 version id `002a68e4caf285c0b4b4d06118c50d106b161a92a092fc6b529859b9925414c1`, 1000 episodes, 200,860 frames, 6.82 GiB = 7.33 GB of files (see the size note below) | `artifacts/s4_v3_run.txt`, `s4_v3_derive_cli_output.txt` |
| `v3/candidate` = `8d80a5d4731ee7321c8f3525fa9bf96f7687b22a22f876e6c39676c0e30d5ad1` (960); `v3/reference` = `5b1bb0c8ca10...` (40, identical episodes and hashes to v0's); `v0/candidate` = `8ee096b72b5a4e2b8697cc29f35f1b81aeb270b1f78d593304db9b1845f9410e` | `data/lineage/names.json` |
| Per-subset stats work on v3; `v3/candidate` digest differs from `v0/candidate`'s; action std shifts by at most 0.14% | `artifacts/s6_v3_run.txt`, `s6_v3_action_stats_vs_v0.csv` |
| A SmolVLA 2-step fine-tune on a v3 subset with all seven types runs on MPS (15.1 s) | `artifacts/s6_v3_run.txt` |
| v3's 20 task strings equal v0's, so the evaluation task list is unchanged; the human reference (all 1000 v0 originals) is untouched | `artifacts/s6_v3_run.txt` |

**The training data of the injected arm is the corrupted set.** `train_run` takes any subset; it does not go through
`vlapoc.method_data.open_for_method`, and must not (the loader is for methods that must not see the answer; the
injected training arm is the experiment). Methods that screen or attribute over `v3/candidate` do use the loader.

## Known issue for the attribution feature (AC6.6)

`vlapoc.runs.train_run` (and the eval path, `eval_run`) do **not** call `open_for_method` or `assert_blind`. That is correct for
this feature's training arms (GT-R1, GT-R5 train on the injected or oracle subsets on purpose). But a later method that trains
policies itself (datamodels, CUPID) could pass a ground-truth-derived subset to `train_run` and would not be refused, and
`train_run("v1/candidate", ...)` would run on the fenced v1 (measured in the S3 note: the loader protects methods, not trainers).
**The attribution feature must route its training inputs through `open_for_method`** (or add the same checks to `train_run`
for method-initiated calls). Not fixed here: out of scope for this feature (plan r2 R11).

## Order

| Order | Id | What | Depends on |
|---|---|---|---|
| 1 | GT-R1 | SmolVLA trained on `v3/candidate`, AC8.5 config and seeds | v3 + lineage store copied |
| 2 | GT-R2 | each GT-R1 checkpoint evaluated with the AC5.8 protocol | GT-R1 |
| 3 | GT-R3 | returns for every GT-R2 rollout | GT-R2 |
| 4 | GT-R4 | verdict `v3/candidate` vs `v0/candidate` (macro-KS gate, then per-metric CIs) | GT-R3, the evaluation feature's AC8.5/AC5.8/AC6.7 runs on `v0/candidate` |
| 5 | GT-R5 (optional) | oracle arm: `v3/candidate` minus the 91 labelled episodes | GT-R1 config |
| 6 | GT-R6 | hand-off: a screening or attribution score table goes through `score_recovery` | later features |

The arm `v0/candidate` (the evaluation feature's AC8.5 + AC5.8 + AC6.7) is the **clean control**. GT-R4 needs it from
the same seeds and the same batch plan; run it first if it has not been run.

## Inputs to copy (in addition to the evaluation runbook's list)

| What | Path | Size | Why |
|---|---|---|---|
| v3 dataset | `data/lerobot/libero_spatial_10__v3/` | 6.82 GiB = 7.33 GB of file bytes (measured, sum of file sizes). `du -sh` can print up to 7.4G on a freshly written copy because of allocated-but-unused blocks (measured, execution-log round 2 step 2); plan disk with `df` free space, not `du` | `train_run` verifies it byte for byte against the registered version |
| Lineage store incl. v3's version record, subsets, labels | `data/lineage/` | about 8 MB (measured) | Joins the server runs to the same graph; copy it back afterwards. Note `versions/<v3 id>/labels.parquet` is the answer key: keep it out of any directory a method under test can read |

Environment: the evaluation runbook's section 0 (Python 3.12/3.13, `LIBERO_CONFIG_PATH`, `MUJOCO_GL=egl`,
`HF_DATASETS_CACHE=/big/disk/hf_datasets_cache`, `HF_HUB_OFFLINE=1`, `HF_HOME=$PWD/data/hf_home`,
`MLFLOW_DISABLE_AGENT_HINT=1`). The git tree must be clean under `vlapoc/` and `scripts/` when a tracked stage starts.

## GT-R1 (pending server): SmolVLA on `v3/candidate`

- **Inputs:** version v3 `002a68e4caf2...`; subset `v3/candidate` `8d80a5d4731e...` (960 episodes, 192,945 frames, measured
  with `Subset.episodes()`; `v0/candidate` has 192,425).
- **Command** (one per seed S, exactly the evaluation feature's AC8.5 configuration, only the subset and the name differ):

```bash
HF_HOME=$PWD/data/hf_home .venv/bin/python -c "
from vlapoc.runs import train_run
print(train_run('v3/candidate', 'train-smolvla-v3cand-s0', seed=0, steps=20000, batch_size=64, policy_type='smolvla',
                policy_overrides={'load_vlm_weights': False}, device='cuda', pretrained_path='data/hf_models/smolvla_base',
                pretrained_ref='lerobot/smolvla_base@d9f33c94a60fb382c90dea2164c96845bd955e28', smoke=False))"
```

- **Outputs:** train runs with checkpoints under `data/runs/train-smolvla-v3cand-s<S>-<stamp>/`; tags
  `train.subset_stats.sha256` (per-subset normalization is the default; this digest differs from the `v0/candidate`
  run's, measured locally for the subset stats) and `vlapoc.smoke=false`.
- **Check:** `scripts/verify_subset_stats_s8.py --normalizer <run_id>` prints `pass: True`.
- **Disk:** HF datasets cache about 7 GB for the 960-episode subset (inferred from the evaluation feature's measured
  6.4-6.6 GB per 940-960-episode subset), about 1.2 GB per checkpoint (measured, 2-step fine-tune).
- **Time:** about 4 h per seed on one A100 (quoted, LeRobot SmolVLA docs, 20k steps); unmeasured on our server.

## GT-R2 (pending server): evaluate each GT-R1 checkpoint

The evaluation runbook's AC5.8 command with `--policy train:<GT-R1 run id>` and `--name eval-smolvla-v3cand-s<S>`,
the same 20 tasks, 50 init states, `--seed-base 0`, `--tau 1200`, `--render 128`, `--batch-size 50`,
`--attribution-grade`. **The batch plan must equal the `v0/candidate` arm's**, or the verdict's guard refuses.
`--record-trajectories` is only needed for the arm that becomes an attribution base policy (the injected arm is the
likely one: record it). Outputs, checks and interruption handling: as in the evaluation runbook.
**Disk (inferred):** 11-22 GB of trajectories per seed with `--record-trajectories`, about 15 MB of rollout rows.

## GT-R3 (pending server): returns

`.venv/bin/python -m vlapoc.returns --eval-run <GT-R2 run id> --seed 0 --track --name returns-eval-smolvla-v3cand-s<S>`
(evaluation AC6.7). Disk negligible. Check: `scripts/verify_returns_s6.py --real data/eval/<eval dir>`.

## GT-R4 (pending server): verdict `v3/candidate` vs `v0/candidate`

```bash
.venv/bin/python -m vlapoc.verdict --eval-runs <eval v0cand s0> <eval v3cand s0> --family W --name verify-v0cand-vs-v3cand-s0-W
.venv/bin/python -m vlapoc.verdict --eval-runs <eval v0cand s0> <eval v3cand s0> --family O --name verify-v0cand-vs-v3cand-s0-O
```

- The gate runs the macro-KS check first, then the per-metric CIs with the family declared (W is the declared verdict,
  O is reported alongside, labelled). One verdict per seed.
- **The outcome is a finding, not an acceptance condition** (PoC §8.10 p.17-18 "Verify before claiming").
- What it can show: whether the pooled injection changed the policy at all (gate) and which metrics moved. It cannot
  separate the seven types (one combined version, design D2); per-type harm needs seven training arms (out of scope).
- Check: `python -m vlapoc.lineage diff <train v0cand sS> <train v3cand sS>` shows the data-side differences only (the
  subset and the normalization stats), `params {}` and `config {}`.

## GT-R5 (pending server, optional): the oracle arm

RoboDrop Table 7's "Oracle" removes exactly the corrupted episodes (quoted, p. 14). Build it **ground-truth side**, on the
server, with an op the method loader refuses:

```bash
.venv/bin/python -c "
from vlapoc.lineage import Subset, _bind_name
from vlapoc.groundtruth import load_answer_key
cand = Subset.load('v3/candidate')
bad = set(load_answer_key('v3')['episode_id'])
oracle = Subset(cand.dataset_version_id, tuple(e for e in cand.episode_ids if e not in bad),
                {'op': 'oracle_drop', 'parent_subset_id': cand.subset_id, 'groundtruth': True, 'dropped': '91 labelled episodes'})
oracle.save(); _bind_name('subsets', 'v3/oracle_candidate', oracle.subset_id, None)
print(oracle.subset_id, len(oracle.episode_ids))"
```

Expected: 869 episodes. Then GT-R1/R2/R3 with `v3/oracle_candidate` and the same seeds. Its content is the clean
episodes of `v0/candidate` (inferred: unedited episodes keep their v0 hashes, measured in S4 AC4.4), so it should behave
like the clean control; running it checks the removal arm of PoC §9.1 Step 4. `open_for_method` refuses this subset
(measured, S3), which is intended. Disk: about 7 GB more HF cache, 1.2 GB per checkpoint.

## GT-R6 (pending later features): the hand-off

A screening or attribution method run over `v3/candidate` through `open_for_method` produces a score table keyed by
(`episode_id`, `content_hash`) of the 960 candidates. Score it:

```bash
.venv/bin/python -m vlapoc.groundtruth score --version v3 --scores <table.parquet> --column <score column> --direction highest
```

It reports recall@k (k = 91, 2 x 91), precision@k, AUROC, best balanced accuracy and average precision, pooled and per type
(13 injected vs 869 clean), each with the random-ranker expectation, and records an MLflow `score` run. **Compare against the
heuristic baselines** in `artifacts/s5_v3_recovery_heuristics.csv` and `s5_v3_bands.csv`: four of the seven types are saturated by one-line statistics (`action_noise`, `dropped_frames`, `idle_padding`, `instruction_swap`)
(measured), `detour` is partly found (AUROC 0.73), and `gripper_release` and `time_shift` are the types no heuristic finds at its a-priori direction. `time_shift` is now a trim: the 40-step shortening leaves a length residual (AUROC 0.68 read in the opposite direction, 0.32 as scored; reported openly, execution-log round 2 step 3a). The PoC §4.5
metric check (idle padding should move time-based metrics, gripper release reliability) needs per-metric attribution, which is a
later feature; the labels already carry the hypothesis in `labels.parquet` (`hypothesis` column).

## What cannot be shown here (and why)

- **LIBERO has no ghost-event (drop) rule** (evaluation brief, decision 6: ghost events move out of scope where LIBERO
  gives no rule). PoC §4.5 aims `gripper_release` at "drops". In LIBERO a premature release can only show up as a lower
  success rate, a lower F(tau), or more timeouts, never as a ghost-event rate. Do not look for a drop metric in
  the GT-R4 output.
- **Policy-level harm per corruption type** needs one training arm per type. This feature trains one pooled injected
  arm (design D2).
- **Altered viewpoint** is not injected (owner decision 2: it needs simulator re-rendering).
- **Dropped frames, paused frames and detours reuse stored frames**, so they are exact duplicates or replays; the
  S5 heuristics find held and paused frames (duplicate detector) and swapped instructions (provenance cross-check) with
  AUROC 1.0 (measured). A policy trained on them sees real frames.

## Expected totals (inferred)

| Item | Disk |
|---|---|
| Inputs copied (v3 + lineage) | about 7.3 GB (6.82 GiB) |
| HF datasets caches (`v3/candidate`, optionally the oracle subset) | about 7-14 GB |
| Checkpoints (3 seeds, 1-2 arms) | about 3.6-7 GB |
| Injected-arm trajectories (3 seeds) | about 33-66 GB |
| **Total** | about 50-95 GB, plus 20 GiB headroom |
