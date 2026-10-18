# Execution log: attribution, curation and server orchestration (2026-10-18, cluster `dgx`)

Every number here is **measured**, with its artifact in `artifacts/`, unless labelled otherwise.

## Environment (server)

- venv `/raid/chem/gor/phyexp/venv` (Python 3.12), linked as `.venv`: an exact `--no-deps` replica of the cluster's
  working LeRobot env (`/raid/chem/gor/phyexp/requirements-lerobot-env.txt`) plus mlflow 3.16.1 and pypdf. lerobot
  0.6.1, as the runbook pins. `data/` → `/raid/chem/gor/phyexp/data`. Both links are in `.git/info/exclude`.
- `data/hf_models/SmolVLM2-500M-Instruct`: config and tokenizer (no weights) from the HF snapshot `7b375e1b`, the
  revision the repo pins.
- **Not on the server:** v0 (`data/lerobot/libero_spatial_10`), the lineage store `data/lineage/`, the human
  reference, LIBERO assets. They live on the owner's Mac (runbook §0), so the tracked chain cannot run here yet.
- **Discrepancy, recorded:** the cluster's cached `lerobot/smolvla_base` is revision `5e8d12a6…`, not the runbook's
  `d9f33c94…`. `scripts/stage.py train` records `d9f33c94…` as `policy.path` by default, so download that revision
  into `data/hf_models/smolvla_base` (or pass `--pretrained-ref`) before AC8.5.

## Checks

| Check | Result | Artifact |
|---|---|---|
| A1 `influence_matrix` = brute-force Σ_{j∈τ} Σ_{i∈ξ} φ_jᵀH⁻¹φ_i (2 shards each) | max rel err 6.2e-16 | `verify_synthetic.json` |
| A2 `center=False` = Prop. 1; centred = raw − R̄·mean row; constant R → 0 | pass | `verify_synthetic.json` |
| A3 `__same_task` = per-task by hand; harm signs; ft columns only with ≥ 5 rollouts; Kendall τ | pass | `verify_synthetic.json` |
| A4 z-score ensemble; misaligned keys refused | pass | `verify_synthetic.json` |
| A5 per-task ops: 2 per task from 6 tasks, ties by episode_id, seeded, matched counts, stale table / emptying k refused | pass (16/16 overall) | `verify_synthetic.json` |
| A6 ported `influence_matrix` on the cluster's clean_full_s0 features vs its `rollout_demo_influence.npy` (500 × 379) | max rel err 3.2e-13 | `verify_regress_clean_full_s0.json` |
| A7 G1 `featurize` loads a cluster checkpoint through `checkpoint_policy`; 117 trainable Linear layers, dim 7,488 | pass | `verify_gpu_smoke_clean_full_s0.json` |
| A7 G3 `featurize` vs the cluster code's loader/processors/projector, same 422 samples, draws and parameter dtypes | max rel diff 0 (min cos 0.9999999) | same |
| A7 G4 seeded draws: two `featurize` calls in one process | max rel diff 0 | same |
| A8 `scripts/server_pipeline.py --tag v0cand --dry-run` | 74 jobs, 6 arms, dependencies as designed | below |

## Findings

1. **The cluster code's flow-matching times were not seeded (G5).** `vlacur.influence.fixed_draws` seeded the noise
   but drew the K times from torch's global RNG, whose initial seed is random per process (two fresh processes:
   `[0.913, 0.650, 0.679, 0.836]` vs `[0.984, 0.162, 0.921, 0.958]`). Every featurize job (4 train shards and 5 rollout
   parts per seed) therefore used its own times, so the October influence matrices pair gradients of different
   losses. A re-run of the cluster code does not reproduce its own feature files (median rel diff 0.47 on 64 rows of
   shard 0). **Fixed in the port:** `vlapoc.influence.fixed_draws` draws noise and times from one seeded generator
   (exact Beta(1.5, 1) by inverse CDF, u^(1/1.5)). The October attribution scores (clean_full_s0/1/2) should be
   recomputed before anyone relies on them (inferred).
2. **`vlapoc.runs.load_checkpoint` upcasts bf16 weights (G3b).** The checkpoint stores 474 bf16 and 26 fp32 tensors;
   LeRobot's `from_pretrained` keeps those dtypes, while `load_checkpoint` (used by the eval stage and therefore by
   `featurize`) leaves 225 parameters in fp32 (values identical). On the same samples and draws the features differ by
   median 13.5% (min cos 0.857). `featurize` keeps the eval stage's loader, so rollouts and features come from the same
   numerics; that is a choice, recorded (inferred). The eval stage itself running the vision tower in fp32 rather than
   training's bf16 is an existing behaviour, not changed here.
3. The cluster featurizer hard-coded fps 10 and strided a global index; `featurize` uses the dataset's fps (20 for
   v0 and trajectories) and strides per episode (`frame_index % stride == 0`, G2).
4. A torchcodec decode error appears when DataLoader workers fork after the main process has decoded video
   (lerobot/libero_10 only, in the smoke's reference path). v0 and the recorded trajectories are PNG datasets, so the
   tracked chain does not decode video (inferred from `write_trajectories(use_videos=False)` and the runbook's
   "lossless" v0).

## Pipeline dry run (A8)

`--tag v0cand`, seeds 0-2, metrics f30 / rmst60 / auc_expected, k_per_task 5, 2 random draws, AC7.5 on:
per seed train → eval (trajectories) → returns, train features (4-shard array) and rollout features (4-shard array)
→ attribute; one ensemble; one select job (6 select runs); per arm × seed train → eval (`--no-trajectories`) → verify
(W then O). 74 jobs. Arms: `harm-f30-k5`, `harm-rmst60-k5`, `harm-auc_expected-k5`, `random0-k5`, `random1-k5`,
`drop-random-20`.

## Process

vla-build single pass (read the section, check prior art, apply, ground, record). No analyzer / `report.html`
pipeline, no multi-agent `feature-pipeline`. This page is the execution record.

## Not done

- The tracked chain end to end (needs v0 + the lineage store on the server).
- Concurrency risk (inferred): many SLURM jobs append to one SQLite `mlflow.db` on local `/raid`; MLflow retries
  on lock, but a "database is locked" failure would fail that stage's job, and `afterok` stops its dependents.
