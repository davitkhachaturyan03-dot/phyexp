# Brief: data layer + lineage layer

**Date:** 2026-09-28
**Feature folder:** `tracker/features/2026-09-28-data-lineage-layer/`

## Request (verbatim)

> build the data layer and the lineage layer FEATURE. The output must be a dataset and a
> lineage layer that will enable all downstream tasks inside the poc

## What "downstream tasks inside the PoC" means (orchestrator's pointer, not spec)

The downstream consumers are the PoC's own stages; read them in the PDF, don't take this list
as the spec: corruption injection with ground-truth labels (§2.5), screening — DemInf and
RoboDrop-style gradient scoring against a clean reference (§8, §9), training SmolVLA on a
chosen episode subset (§2), rollout evaluation with PhAIL primitives (§3), per-metric
attribution over episodes (§4), and counterfactual retraining on "drop top-k" subsets with
verification (§9, §10). The data + lineage layer must let each of those name exactly which
episodes (and which version of each episode) it consumed and produced. The lineage
requirements are in §2.1 (the PoC names W&B / MLflow / DVC there). `docs/dataflow.md` is a
derived reading of §9 — useful map, not authority.

## Current state of the repo (verified by the orchestrator, 2026-09-28)

- Raw data on disk: `data/raw/libero/libero_spatial/` (10 HDF5 files) and
  `data/raw/libero/libero_10/` (10 HDF5 files), ~19 GB total. Original LIBERO robomimic-format
  HDF5. No other data, no processed dataset, no code (`main.py` is the PyCharm stub).
- An earlier session built a "LIBERO dataset and episode catalog" (adopted `lerobot/libero`,
  added a metadata layer + local browser). **That work no longer exists on disk** — its report
  file was removed from `tracker/reports/` on 2026-09-28 and no code or processed data remains.
  Its description (from the old index, now removed): "Adopted the official `lerobot/libero`
  dataset, then built the metadata layer it lacks — suite, scene, objects, goal structure,
  action statistics — plus a local browser with video. 14 dataset columns to 21 groupable
  ones." Treat that as a lead about a past decision, not as something that exists.
- The tracker was restructured on 2026-09-28: the feature folder is the unit of record,
  `tracker/index.html` lists features and holds the spec-coverage table (empty so far), and
  the old `tracker/reports/` + `GROUNDING.md` were retired.
- Python env: `.venv/bin/python` (3.14). Check `.venv/bin/pip list` for what is installed.

## Constraints carried from this conversation / CLAUDE.md

- Check off-the-shelf first (LeRobot dataset format, robomimic, RLDS, HF Hub revisions,
  W&B Artifacts / MLflow / DVC) and verify against the docs and the bytes, not memory.
- Known verified fact from a prior session (CLAUDE.md / vla-build): every one of the 1000
  LIBERO demos ends with reward == 1 — there are no failed demonstrations; do not adopt
  OpenVLA's 189-demo filter on that premise. Re-verify on this data if you rely on it.
- Label every claim quoted / measured / inferred.
- PoC scope: smallest thing that actually enables the downstream stages, with real numbers.

## Owner answers to plan questions (2026-09-28)

- **BLOCKING Q1, tracking backend: "MLflow, local (Recommended)".** Local SQLite at
  `data/lineage/mlflow.db`, no account, offline. `mlflow` may be added as a dependency.
- **BLOCKING Q2, git: "git init, local only (Recommended)".** Initialise a local git repo at the
  repo root, no remote, with `data/`, `.venv/`, `outputs/` (and other large or generated paths)
  in `.gitignore`. The owner chose this so that every run records the exact code version, which
  needs commits: the executor may create **local** commits of source code for that purpose
  (never push, never add a remote, never commit anything under `data/` or `.venv/`). Commit
  messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Perturbation scope (asked 2026-09-29): "Separate feature (Recommended)".** This feature ships
  the injection *mechanism* (S4: derived versions, ground-truth labels, reference-set guard,
  length-changing edits) plus the one demo corruption (RoboDrop action noise) as planned. The full
  §2.5 / §8.10 corruption suite is deliberately a separate, later injection feature; its absence
  here is owner-approved scope, not a gap.

## Owner rulings

(none yet)
