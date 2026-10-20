#!/bin/bash
#SBATCH --job-name=phyexp-gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/raid/chem/gor/phyexp/logs/%x-%A_%a.out
#
# One tracked stage on one GPU. CMD may contain {i} (replaced by SLURM_ARRAY_TASK_ID) for sharded stages:
#   CMD="python scripts/stage.py train --subset v0/candidate --name train-x-s0 --seed 0" sbatch slurm/gpu.sbatch
#   CMD="python scripts/stage.py features --train train-x-s0 --which train --shard {i}/4" sbatch --array=0-3 slurm/gpu.sbatch
# EXPECT_SHA (set by scripts/server_pipeline.py) makes the job refuse to run on any other commit.

set -euo pipefail
source "${PHYEXP_DIR:-/auto/home/gor/projects/phyexp}/slurm/env.sh"
source slurm/guard.sh
: "${CMD:?set CMD}"
CMD=${CMD//\{i\}/${SLURM_ARRAY_TASK_ID:-0}}
echo "[gpu.sbatch] $(hostname) $(git rev-parse --short HEAD) :: $CMD"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
eval "$CMD"
