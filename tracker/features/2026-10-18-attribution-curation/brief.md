# Brief: attribution, curation and server training orchestration

## Request (verbatim)

> please clone the https://github.com/davitkhachaturyan03-dot/phyexp.git repo under /projects folder, and integrate
> the codes under VLA-data-curation folder into that repo. The codes inside the repo span data coruption and model
> evaluation concepts and missing data curation and model training parts, that is actually implemented inside this
> folder, so please integrate them properly.

## Interpretation

`VLA-data-curation` (the cluster project, `src/vlacur/`) ran the PoC end to end on lerobot/libero_10, with its own
rollouts, metrics, corruption and catalog. This repo already has better-grounded versions of those (the data,
lineage and evaluation layers), so they are **not** ported. The missing parts are:

- **Step 3 (attribution):** `attribute_standin` produced placeholder scores. Port the projected-gradient featurizer
  (`vlacur.influence`) and the CUPID estimator (`vlacur.attribution`), keyed to this repo's lineage
  (`episode_id`, `content_hash`), its eval stage's recorded trajectories and its family-W returns, recorded as an
  `attribute` stage run.
- **Step 4 (curation):** `vlacur.curate`'s per-task drops and matched random controls, as lineage selection ops
  wired into `select_run`.
- **Training at scale:** `vlacur`'s SLURM scripts and `scripts/pipeline.py`, rewritten so every job is one tracked
  stage (`train_run`, `eval_run`, ...) in the runbook's order.

Process: the `vla-build` discipline (CLAUDE.md: read the section, check prior art, apply, ground, record), not the
multi-agent `feature-pipeline`. No `report.html` / analyzer round was run for this feature.

## Not ported (and why)

- `vlacur.rollouts`, `vlacur.evaluate`, `vlacur.metrics`: superseded by `vlapoc.rollouts` / `vlapoc.returns` /
  `vlapoc.verdict` (fixed batch plan, one env per rollout, attribution-grade assertion).
- `vlacur.catalog`: superseded by `vlapoc.catalog` + `vlapoc.lineage` (content-addressed versions).
- `vlacur.corrupt`: `vlapoc.derive` is the injection hook with ground truth kept apart and the blind fence. The
  cluster's corruption types (action noise, gripper flips, frame drops, ...) could be added as derive ops later.
- The cluster's training recipe (lerobot-train CLI, 30k steps, camera rename): AC8.5's recipe (20k steps, batch 64,
  `load_vlm_weights=False`, per-subset stats) is the repo's declared one and is used unchanged.
