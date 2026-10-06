# Evaluation layer: server runbook (pending-server ACs)

Written by the executor (round 1, batch 3, 2026-10-04; revised in round 2, 2026-10-06, for plan r2). Every command below exists in the repo at the commit
that adds this file. Every number is labelled **measured** (on this Mac, with the artifact named) or
**inferred** (an estimate, not yet run anywhere). Nothing in this file has been run on a server.

The four pending-server ACs, in the order they must run:

| Order | AC | What | Depends on |
|---|---|---|---|
| 1 | AC8.5 | SmolVLA trained on `v0/candidate` (960 episodes), per-subset stats, 2–3 seeds | — |
| 2 | AC5.8 | each AC8.5 checkpoint × 50 init states × 20 tasks, τ 60 s, 128², trajectories | AC8.5 |
| 3 | AC6.7 | returns for every AC5.8 rollout | AC5.8 |
| 4 | AC7.5 | full-data vs drop-20-random retrain, both evaluated as AC5.8, then the verdict | AC8.5, AC5.8 |

## 0. Environment and inputs

**Python and packages.**
- Use Python 3.12 or 3.13. lerobot 0.6.1 declares 3.12/3.13 only. On 3.14 the `lerobot-train` CLI is broken
  (draccus #69). `vlapoc.lerobot_train_entry` works around it on any version.
- Pin what ran locally (measured, `data/scratch/logs/venv_freeze_after_smolvla.txt`): lerobot 0.6.1,
  hf-libero 0.1.4 (`--no-deps`), robosuite 1.4.0, bddl 1.0.1, mujoco 3.x, transformers 5.5.4, num2words 0.5.14,
  torch with CUDA.

**Environment variables**, every session:

```bash
export LIBERO_CONFIG_PATH=$PWD/data/libero_config      # never ~/.libero (plan P12)
export MUJOCO_GL=egl                                    # headless Linux (LeRobot LIBERO docs, quoted)
export HF_DATASETS_CACHE=/big/disk/hf_datasets_cache    # about 7 GB per distinct 960-episode training subset (measured locally, data+lineage R1)
export HF_HUB_OFFLINE=1                                 # after the downloads below
export MLFLOW_DISABLE_AGENT_HINT=1
```

**Copy from this machine (sizes measured with `du`):**

| What | Path | Size | Why |
|---|---|---|---|
| v0 dataset (lossless LeRobot v3) | `data/lerobot/libero_spatial_10/` | 6.9 GiB | `train_run` checks it byte for byte against the registered version before any run (`verify_version`) |
| Lineage store + MLflow DB | `data/lineage/` | — | Version, subset and run ids, so server runs join the same graph. Copy it, append on the server, copy it back |
| Human reference | `data/eval/human_reference/` | — | HRT/AUC anchor for returns and the verdict (all 1000 v0 originals) |
| LIBERO assets | `data/libero_assets/` | — | Pinned at rev `0b3ea86b…`, or re-fetch with `scripts/human_reference.py --fetch-assets` |
| LIBERO config | `data/libero_config/` | — | Repo-local |

**Model initialization for AC8.5: fine-tune `lerobot/smolvla_base@d9f33c94a60fb382c90dea2164c96845bd955e28`**
(plan r2, F-P4).
- **Source (quoted, LeRobot SmolVLA docs):** "Use `smolvla_base`, our pretrained 450M model, and fine-tune it on your
  data". The weights are 906,712,520 bytes, owner-approved and downloaded to `data/hf_models/smolvla_base`.
- It needs the VLM's config/tokenizer only: copy `data/hf_home/` (4.7 MB, `refs/main = 7b375e1b`) and set
  `HF_HOME=$PWD/data/hf_home`.
- **Divergence, recorded.** Round 1 specified `--policy.type=smolvla --policy.load_vlm_weights=true`, the LIBERO
  page example, which initializes from the 2.03 GB SmolVLM2 backbone. The owner did not approve that download, and no
  PoC page requires it (§2.3 p. 4: "SmolVLA via LeRobot").
- **Steps and batch size:** `steps=20000`, `batch_size=64`, from the LeRobot SmolVLA docs' fine-tuning command
  (quoted: `--policy.path=lerobot/smolvla_base … --batch_size=64 --steps=20000`; "Training the model for 20k steps
  will roughly take ~4 hrs on a single A100 GPU").
- These are fixed for every seed and arm; data is the only variable (PoC p. 1).

**First command on the server: the local smoke chain** (S9, measured locally: 122.6 s, chain
`eval-smoke-k5-s0-20261005T033607Z`):

```bash
.venv/bin/python scripts/lineage_dry_run.py --eval-smoke
```

It runs select → train ×2 → eval ×2 → returns ×2 → verify on `v0/reference`. If `trace(verify)` reaches 20 raw
files and no stand-in, the environment, the copied lineage store and the sim stack all work. Its evals use
`--device cpu` (ACT smoke); the AC commands below set the device explicitly.

## Operational rules (measured locally in batch 2)

1. **Fix the batch plan across all seeds, arms and re-runs** (AC5.8: batch 50, one task per batch, fixed task and init
   order, eval seed 0).
   - A stochastic policy's outcome depends on the batch plan. SmolVLA, same init state and seed, batch 1 vs 5:
     init 4 succeeded at step 71 vs never (`artifacts/s5_smolvla_batch_effect.txt`).
   - The init state and scene do not depend on batching (`s5_ac5_6_pairing.json`, `s5_ac5_2b_resets_r2.json`).
   - `batch_size` is recorded in every rollout row and MLflow eval run.
2. **One env object per rollout.** `vlapoc.rollouts.run_rollouts` does this already; do not change it.
   - In 4/20 tasks, LIBERO moves fixtures on every re-reset of the same env object, even with the same seed (task
     8's stove: 7.2 mm per reset).
   - The first seeded reset of a fresh env is reproducible (`s5_ac5_2b_resets_r2.json`).
3. **Run long jobs under the server's job runner** (slurm, or tmux/screen). Locally a session restart killed one
   eval mid-run (`8fae29c4`, marked KILLED); `screen -dmS … caffeinate -i …` survived.
   - One task or seed per eval run, so an interruption loses at most one.
   - An interrupted run is left RUNNING in MLflow. Close it by hand:
     `MlflowClient().set_terminated(run_id, "KILLED")` plus a tag saying why.
4. **Measure throughput before sizing.**
   - Locally, SmolVLA (`smolvla_libero`, `n_action_steps 1`) cost ≈ 0.29–0.4 s per env-step on MPS. That is ≈ 6 min
     of policy compute per failed 1200-step rollout (`artifacts/s5_smolvla_throughput.txt`).
   - Our trained policy's `n_action_steps` comes from its config. smolvla_base's is 50, which re-plans every 50
     steps, so it should be much cheaper per step (inferred).
   - First run a 30-step probe:
     `scripts/eval_smoke.py --policy train:<run> --tasks libero_spatial:0 --n 10 --batch-size 10 --tau 30 --device cuda --out data/scratch/tp`.
5. **Check determinism on the GPU before the big run.**
   - Run the same 30-step probe twice in two processes. Compare them with
     `verify_rollouts_s5._compare(pd.read_parquet(a), pd.read_parquet(b))`; expect `differing: {}`.
   - On MPS it was identical (`artifacts/s5_smolvla_mps_determinism.txt`). CUDA kernels may not be (unmeasured). If
     they differ, record it and keep the verdict's unpaired bootstrap, which does not need bit-identical reruns.
6. **The git tree must be clean** under `vlapoc/` and `scripts/` when any tracked stage starts (`stage_run` refuses
   otherwise). Commit server-side edits first.

## AC8.5: SmolVLA fine-tuned from smolvla_base with per-subset stats (2–3 seeds)

- **Inputs:** version v0 = `3c4dca68d6f196c10d249b43bd94ae32f51b905735f25c7865e3c943caf553b7`; subset
  `v0/candidate` = `8ee096b72b5a4e2b8697cc29f35f1b81aeb270b1f78d593304db9b1845f9410e` (960 episodes, 192,425 frames,
  measured).
- **Command** (one per seed S ∈ {0, 1, 2}; identical except `seed` and `name`):

```bash
HF_HOME=$PWD/data/hf_home .venv/bin/python -c "
from vlapoc.runs import train_run
print(train_run('v0/candidate', 'train-smolvla-v0cand-s0', seed=0, steps=20000, batch_size=64, policy_type='smolvla',
                policy_overrides={'load_vlm_weights': False}, device='cuda', pretrained_path='data/hf_models/smolvla_base',
                pretrained_ref='lerobot/smolvla_base@d9f33c94a60fb382c90dea2164c96845bd955e28', smoke=False))"
```

- `smoke=False` is asserted explicitly, so the train run is tagged `vlapoc.smoke=false`. Only such a run can feed an
  attribution-grade eval (AC5.9). `normalization='per_subset'` is the default.
- **The run records:**
  - params `policy.path=lerobot/smolvla_base@d9f33c94…` (hub id and revision), `pretrained_path=data/hf_models/smolvla_base`,
    and `policy_overrides` (`load_vlm_weights: false`, i.e. not used: the VLM weights come from smolvla_base's own file);
  - tag `train.pretrained.weights.sha256` (`7cd549ac…`);
  - tag `train.subset_stats.sha256` and artifact `subset_stats.json`;
  - the checkpoint's weights and normalizer sha256.
- **Check after each seed:** `scripts/verify_subset_stats_s8.py --normalizer <run_id>` must print `pass: True`.
- **Render and train at 128²**, the dataset's resolution.
- **Outputs:** 2–3 train runs with checkpoints under `data/runs/train-smolvla-v0cand-s<S>-<stamp>/`.
- **Disk:**
  - HF datasets cache for the 960-episode subset ≈ 7 GB, once (measured for v1/candidate in the data+lineage feature).
  - ≈ 1.2 GB per checkpoint. The 2-step local fine-tune's `model.safetensors` is 1,197,789,224 bytes (measured).
  - 1 checkpoint per seed: ≈ 3.6 GB for 3 seeds.
- **Time:** ≈ 4 h per seed on one A100 (quoted, LeRobot SmolVLA docs, 20k steps); unmeasured on our server. A 2-step
  local fine-tune took 18 s on MPS.

## AC5.8: full-scale evaluation of each AC8.5 checkpoint (attribution-grade, fixed batch plan)

- **Inputs:** each AC8.5 `train_run_id`.
- **Protocol** (plan r2 AC5.8; `vlapoc.runs.attribution_protocol_errors` refuses anything else):
  - all 20 tasks in one eval run, in `libero_task_id` order, libero_spatial before libero_10;
  - init states 0–49 in order, eval seed_base 0;
  - τ_episode 1200 steps (60 s), 128², `control_mode=relative`, `record_trajectories=True`;
  - **batch size 50: one task per batch.** If GPU memory forbids 50, use one smaller divisor of 50 (e.g. 10) for
    **every** seed and arm, and record it.
- **Command** (one per seed; the same batch plan for every seed and every arm it will be compared with):

```bash
.venv/bin/python scripts/eval_smoke.py --policy train:<train_run_id> \
  --tasks libero_spatial:0,libero_spatial:1,libero_spatial:2,libero_spatial:3,libero_spatial:4,libero_spatial:5,libero_spatial:6,libero_spatial:7,libero_spatial:8,libero_spatial:9,libero_10:0,libero_10:1,libero_10:2,libero_10:3,libero_10:4,libero_10:5,libero_10:6,libero_10:7,libero_10:8,libero_10:9 \
  --n 50 --seed-base 0 --tau 1200 --render 128 --batch-size 50 --device cuda --record-trajectories --track \
  --attribution-grade --name eval-smolvla-v0cand-s<S>
```

- `--attribution-grade` is asserted **only** in this command and in AC7.5's. Every other eval (smoke, harness
  validation, the local chain) records `eval.attribution_grade=false`.
- The run records param `eval.batch_plan` and its digest `eval.batch_plan.sha256`, plus `eval.attribution_grade=true`
  and `eval.attribution_grade.asserted=true`.
- **Outputs per eval run:**
  - `data/eval/<name>-<stamp>/rollouts.parquet` (33 columns, schema-validated, incl. `t0_model_sha256`);
  - `trajectories/` (a lossless LeRobot v3 dataset);
  - an MLflow `eval` run whose input is the checkpoint sha and whose outputs are the rollouts sha and the
    trajectories digest.
- **Checks:**
  - `scripts/verify_rollouts_s5.py --trajectories <dir>` must print frames == Σ sim_steps and `frame_index == policy
    step`. `--summary <dir>` prints per-task outcomes and KM.
  - `t0_model_sha256` must be identical per (task, init state) across seeds (fixture placement is part of the fixed
    initial state, AC5.10).
- **Disk (inferred from measured per-frame costs):**
  - At 128² our dataset is 6.9 GiB / 200,340 frames ≈ 37 KB per frame (measured). The 256² trajectories in AC5.4 cost
    ≈ 108 KB per frame (measured).
  - Frames per seed: 1000 rollouts × (successes ≈ 70–500 steps, failures 1200) ≈ 300k–600k (inferred, plan R6), so
    ≈ **11–22 GB of trajectories per seed** and ≈ 33–66 GB for 3 seeds.
  - Rollout rows: 10.6–18.1 KB each (measured), so ≈ 11–18 MB per 1000 rollouts.
  - Keep > 20 GiB free.
- **Interruption:** one run holds all 20 tasks. If it dies, close it with `MlflowClient().set_terminated(run_id,
  "KILLED")` plus a reason tag, and re-run the whole command. Partial runs are never merged, because the batch plan
  is part of the protocol.

## AC6.7: returns for every AC5.8 rollout

- **Inputs:** each AC5.8 eval run id.
- **Command** (one per eval run):

```bash
.venv/bin/python -m vlapoc.returns --eval-run <eval_run_id> --seed 0 --track --name returns-<eval name>   # family W returns + family O operation returns
```

- **Outputs:**
  - `data/eval/returns-<id12>/returns.parquet`: one row per rollout, with `ret_f30`, `ret_rmst60`,
    `ret_auc_sampled`, `ret_auc_expected`, `ret_hrt60_lin`, `ret_censored60` and `failure_type`;
  - `operation_returns.parquet` (libero_10 per sub-task);
  - `returns_meta.json` (HRT weights);
  - `metrics_registry.json`;
  - a `returns` stage MLflow run that consumes the rollouts sha. `trace` reaches the train run and the raw files.
- **Note:** `ret_hrt60_lin`'s weights use the cells in one eval run. With the AC5.8 plan, one run holds all 20 tasks,
  so J = 20.
- All per-rollout returns are **family W** (whole-task T). `operation_returns.parquet` carries family O's
  `ret_rmst60_op_cap` (an unfinished attempt scores −60).
- **Check:** `scripts/verify_returns_s6.py --real data/eval/<eval dir>` must print the identities at ≤ 1e-9.
  Then `--join data/eval/returns-<id12>/returns.parquet` must print a 960 × 6 frame with `check_scores ok`.
- **Disk:** negligible (MB).

## AC7.5: the first LIBERO counterfactual gate (full data vs drop-20-random)

- **Inputs:** parent subset `v0/candidate` (`8ee096b7…`); k = 20; selection seed 0.
- **Commands:**

```bash
# 1. the drop-20-random child, as a select stage run (prints the child subset id)
.venv/bin/python -c "
from vlapoc.runs import select_run
rid, child = select_run('8ee096b72b5a4e2b8697cc29f35f1b81aeb270b1f78d593304db9b1845f9410e', 'select-v0cand-drop-random-20-s0', 'drop_random_k', 20, seed=0)
print(rid, child.subset_id, len(child.episode_ids))"
# 2. train on the child with exactly the AC8.5 config and the same seeds S (per-subset stats are automatic)
HF_HOME=$PWD/data/hf_home .venv/bin/python -c "
from vlapoc.runs import train_run
print(train_run('<child_subset_id>', 'train-smolvla-v0cand-drop20-s0', seed=0, steps=20000, batch_size=64, policy_type='smolvla',
                policy_overrides={'load_vlm_weights': False}, device='cuda', pretrained_path='data/hf_models/smolvla_base',
                pretrained_ref='lerobot/smolvla_base@d9f33c94a60fb382c90dea2164c96845bd955e28', smoke=False))"
# 3. evaluate it with the IDENTICAL AC5.8 command (same tasks, init states, seed_base, tau, render, --batch-size 50, --attribution-grade)
# 4. the verdict per seed: family W is the declared verdict; family O is reported alongside, labelled
.venv/bin/python -m vlapoc.verdict --eval-runs <eval full s0> <eval drop20 s0> --family W --name verify-full-vs-drop20-s0-W
.venv/bin/python -m vlapoc.verdict --eval-runs <eval full s0> <eval drop20 s0> --family O --name verify-full-vs-drop20-s0-O
```

- **Checks:**
  - The verdict's batch-plan guard must pass **without** `--allow-plan-mismatch`: both eval runs carry the same
    `eval.batch_plan` (AC7.8).
  - `verify.attribution_grade=true` (both evals asserted) and a label without "smoke".
  - `python -m vlapoc.lineage diff <train full sS> <train drop20 sS>` must show `data.n_removed 20`,
    `data.normalization.differ true`, `params {}` and `config {}` (the S8 AC8.3 shape, measured locally).
  - `python -m vlapoc.lineage trace <verify run>` must reach both train runs, the select run, v0 and the 20 raw files.
- **Outputs:** per seed, two `verify` MLflow runs (W declared, O alongside) with `verdict.json`: outcome
  (i)/(ii)/(iii)/gate_only, per-metric Holm-adjusted CIs, per-cell points, and the family in every label.
- **Disk:**
  - The drop-20 HF cache ≈ 7 GB (one more distinct subset), and 1 checkpoint per seed ≈ 1.2 GB.
  - The drop-20 arm's evals use the identical plan but may omit `--record-trajectories` (trajectories are only needed
    for the attribution base policy). That saves ≈ 11–22 GB per seed (inferred). The batch plan is unchanged, because
    trajectory capture does not alter rollouts.

## Expected totals (inferred)

| Item | Disk |
|---|---|
| Inputs copied | ≈ 7 GB |
| HF datasets caches (2 subsets) | ≈ 14 GB |
| Checkpoints (3 seeds × 2 arms) | ≈ 7 GB |
| Base-policy trajectories (3 seeds) | ≈ 33–66 GB |
| **Total** | **≈ 60–95 GB**, plus 20 GiB headroom |
