#!/bin/bash
#SBATCH --job-name=phyexp-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#
# One tracked stage on one GPU. CMD may contain {i} (replaced by SLURM_ARRAY_TASK_ID) for sharded stages:
#   CMD="python scripts/stage.py train --subset v0/candidate --name train-x-s0 --seed 0" sbatch slurm/gpu.sbatch
#   CMD="python scripts/stage.py features --train train-x-s0 --which train --shard {i}/4" sbatch --array=0-3 slurm/gpu.sbatch
# Partition and log directory come from slurm/site.env, passed by scripts/server_pipeline.py as --partition / --output=<LOG_DIR>/%x-%A_%a.out
# (#SBATCH lines cannot expand variables); for a hand submission add them yourself: sbatch --partition=<P> --output=<LOG_DIR>/%x-%A_%a.out ...
# EXPECT_SHA (set by scripts/server_pipeline.py) makes the job refuse to run on any other commit.

set -euo pipefail
source "${PHYEXP_DIR:?set PHYEXP_DIR (scripts/server_pipeline.py exports it)}/slurm/env.sh"
source slurm/guard.sh
: "${CMD:?set CMD}"
CMD=${CMD//\{i\}/${SLURM_ARRAY_TASK_ID:-0}}
echo "[gpu.sbatch] $(hostname) $(git rev-parse --short HEAD) :: $CMD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
eval "$CMD"
