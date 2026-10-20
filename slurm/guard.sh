# Sourced by the sbatch scripts: a queued job must run the commit its chain was submitted from.
# (stage_run separately refuses a dirty vlapoc/ or scripts/.)
if [[ -n "${EXPECT_SHA:-}" && "$(git rev-parse HEAD)" != "$EXPECT_SHA" ]]; then
  echo "[guard] HEAD $(git rev-parse --short HEAD) != submitted ${EXPECT_SHA:0:12}; refusing" >&2
  exit 3
fi
