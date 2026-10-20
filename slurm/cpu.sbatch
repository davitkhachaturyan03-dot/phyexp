#!/bin/bash
#SBATCH --job-name=phyexp-cpu
#SBATCH --partition=a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/raid/chem/gor/phyexp/logs/%x-%j.out
#
# Tracked stages that need no GPU (returns, ensemble, select, verify); CMD may chain several with `&&`:
#   CMD="python scripts/stage.py returns --eval eval-x-s0 --name returns-x-s0" sbatch slurm/cpu.sbatch

set -euo pipefail
source "${PHYEXP_DIR:-/auto/home/gor/projects/phyexp}/slurm/env.sh"
source slurm/guard.sh
export CUDA_VISIBLE_DEVICES=
: "${CMD:?set CMD}"
echo "[cpu.sbatch] $(hostname) $(git rev-parse --short HEAD) :: $CMD"
eval "$CMD"
