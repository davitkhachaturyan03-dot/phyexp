# Changelog: attribution, curation and server orchestration (2026-10-18)

## Added

- `vlapoc/influence.py`: per-sample projected gradients of a SmolVLA checkpoint (Kronecker per-layer projection,
  K shared seeded (noise, time) draws, per-episode stride, sharding, manifest). Loads checkpoints through
  `vlapoc.rollouts.checkpoint_policy`. CLI `python -m vlapoc.influence`.
- `vlapoc/attribution.py`: CUPID influence of every training episode version on every family-W return and every
  per-failure-type return, all-rollout and same-task columns, `harm_` sign-folded columns, Kendall's τ agreement,
  z-scored seed ensemble. `attribute_run` / `ensemble_run` record stage `attribute` runs (replacing
  `attribute_standin` in real use). CLI `python -m vlapoc.attribution`.
- `vlapoc/lineage.py`: `drop_top_k_per_task`, `drop_random_k_per_task` (`TASK_GROUP = (suite, libero_task_id)`).
- `scripts/stage.py`: one tracked stage per call (train / eval / returns / features / attribute / ensemble / select /
  verify) with the runbook's recipes; run arguments resolve by run name; `train --subset select:<run>`.
- `scripts/server_pipeline.py`: the whole experiment as one SLURM dependency chain (`--dry-run`, `--from`, `--to`).
- `slurm/env.sh`, `slurm/gpu.sbatch`, `slurm/cpu.sbatch`, `slurm/guard.sh` (refuses a job on another commit).
- `scripts/verify_attribution.py`: synthetic checks, regression vs the cluster's influence matrix, GPU smoke.

## Changed

- `vlapoc/runs.py`: `SELECT_OPS` gains the two per-task ops; `select_run` takes `allow_dirty`; docstrings point at
  the real attribute stage.
- `vlapoc/__init__.py`: module list.

## Ported from VLA-data-curation (`src/vlacur/`)

`influence.py` (projector and loss loop unchanged; draws now seeded), `attribution.py` (estimator unchanged,
verified to 3.2e-13), `curate.py` (per-task drop and matched random → lineage ops), `scripts/pipeline.py` and
`slurm/*.sbatch` (rewritten around tracked stages). Not ported: its rollouts, evaluate, metrics, catalog, corrupt
(superseded here; see `brief.md`).
