# Server environment for every tracked stage (runbook §0). Source from the repo root: `source slurm/env.sh`.
# data/ and .venv are machine-local (symlinks to /raid on this server; see .git/info/exclude).
export PHYEXP_DIR=${PHYEXP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export SCRATCH_ROOT=${SCRATCH_ROOT:-/raid/chem/gor/phyexp}

export LIBERO_CONFIG_PATH=$PHYEXP_DIR/data/libero_config   # never ~/.libero (plan P12)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HOME=$PHYEXP_DIR/data/hf_home
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE=$SCRATCH_ROOT/hf_datasets_cache  # ~7 GB per distinct training subset
export MLFLOW_DISABLE_AGENT_HINT=1
export TOKENIZERS_PARALLELISM=false
export TMPDIR=$SCRATCH_ROOT/tmp
# torchcodec needs FFmpeg shared libs; appended so they never shadow the system libs EGL/MuJoCo use.
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/raid/chem/gor/vla-curation/envs/ffmpeg/lib

mkdir -p "$HF_DATASETS_CACHE" "$TMPDIR" "$SCRATCH_ROOT/logs"
source "$PHYEXP_DIR/.venv/bin/activate"
cd "$PHYEXP_DIR"
