# Server environment for every tracked stage (runbook §0). Source from the repo root: `source slurm/env.sh`.
# Machine-specific values (scratch disk, partition, log dir, FFmpeg libs) live in slurm/site.env, a per-machine copy of
# slurm/site.env.example (gitignored); see also docs/site-profile-cluster.env. data/ and .venv are machine-local too
# (on the owner's server data/ is a symlink to a scratch disk; see .git/info/exclude).
export PHYEXP_DIR=${PHYEXP_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SITE_ENV=${SITE_ENV:-$PHYEXP_DIR/slurm/site.env}
if [[ ! -f "$SITE_ENV" ]]; then
  echo "[env.sh] missing $SITE_ENV: copy slurm/site.env.example to slurm/site.env and edit it (cluster values: docs/site-profile-cluster.env)" >&2
  return 2 2>/dev/null || exit 2
fi
set -a
# shellcheck disable=SC1090
source "$SITE_ENV"
set +a
: "${SCRATCH_ROOT:?set SCRATCH_ROOT in $SITE_ENV}"
LOG_DIR=${LOG_DIR:-$SCRATCH_ROOT/logs}

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
if [[ -n "${FFMPEG_LIB:-}" ]]; then export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$FFMPEG_LIB; fi

mkdir -p "$HF_DATASETS_CACHE" "$TMPDIR" "$LOG_DIR"
source "$PHYEXP_DIR/.venv/bin/activate"
cd "$PHYEXP_DIR"
