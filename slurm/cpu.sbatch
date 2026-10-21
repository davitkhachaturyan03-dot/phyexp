#!/bin/bash
#SBATCH --job-name=phyexp-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#
# Partition and log directory come from slurm/site.env, passed by scripts/server_pipeline.py as --partition / --output=<LOG_DIR>/%x-%j.out
# (#SBATCH lines cannot expand variables); for a hand submission add them yourself.
# Tracked stages that need no GPU (returns, ensemble, select, verify); CMD may chain several with `&&`:
#   CMD="python scripts/stage.py returns --eval eval-x-s0 --name returns-x-s0" sbatch slurm/cpu.sbatch

set -euo pipefail
source "${PHYEXP_DIR:?set PHYEXP_DIR (scripts/server_pipeline.py exports it)}/slurm/env.sh"
source slurm/guard.sh
export CUDA_VISIBLE_DEVICES=
: "${CMD:?set CMD}"
echo "[cpu.sbatch] $(hostname) $(git rev-parse --short HEAD) :: $CMD"
eval "$CMD"
