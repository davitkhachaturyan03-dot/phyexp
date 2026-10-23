# Brief: dashboard (demonstration surface), cluster and pattern discovery, server runbook

**Date:** 2026-10-22
**Feature folder:** `tracker/features/2026-10-22-dashboard/`
**Git branch:** `dashboard`, created from `training-attribution` at `1d18247` (that branch is PASSED but not merged into
`main`; the owner said "Not yet"). **Never push**; never merge unless the owner asks.

## Request (verbatim, in order)

> i want you to start off the dashboard feature first. I think it's the most useful to present the project to any future
> stakeholders. Also, I think as a Poc we need to pinpoint to exact clusters of data and identify patterns within those
> clusters that harm or boost a specific policy. tell me exactly how we're able to do that on a high level before kicking
> off this new and final feature

> if we build the dashboard feature, will it be possible to create an exact step-by-step runbook for a new claude session
> inside the actual gpu server? In case I want to run the pipeline end-to-end on real gpus the claude-code session must
> know exactly what to do with this code.

> will runbook guide all the way to attribution and real dashboard creation?

> ok, kick off the feature for dashboard and include the runbook creation as a separate story there. start now

The owner calls this the "new and final feature" of the PoC milestone on this machine.

## Owner answers (binding, 2026-10-22)

- **Cluster lenses to build:** "Label groups (task, scene), Rule-like slices, Four metric groups".
  **Not** visual-similarity (image-embedding) clusters.
- **Dashboard form: "Live local app"** (a small web app the owner runs locally, with filtering and click-through).
  The planner still checks off-the-shelf options (LeRobot's dataset visualizer, Renumics Spotlight, FiftyOne, Streamlit /
  Gradio / Panel / Dash, etc.) and the analyzer audits the choice — "live local app" is the form, not a mandate to
  hand-roll a framework.
- **Data until server results exist: "Synthetic scores, labelled".** Real LIBERO videos and catalog, the real (not
  attribution-grade) MPS smoke run, plus synthetic influence scores clearly marked SYNTHETIC everywhere they appear. Real
  server results must drop in later with no code change and the SYNTHETIC label disappearing only when the data is real.
- **Runbook: a separate story in this feature.** A step-by-step runbook a fresh Claude Code session on the GPU server can
  follow end to end: preflight → smoke → submit the attribution chain → monitor → clusters/patterns stage → real
  dashboard. Discussed form (orchestrator's proposal, not yet a ruling): a repo skill (e.g.
  `.claude/skills/server-run/SKILL.md`) loaded automatically, pointed to from project docs, with expected outputs per step,
  explicit stop-and-ask points (failed check, failed job, odd numbers, low disk, any code change needed), recovery
  (`--from`), the known server-only risks (SQLite locking under concurrent jobs; `smolvla_base` revision `5e8d12a6` vs
  pinned `d9f33c94`), bringing results back (lineage DB + outputs), and rules (no code edits, no push, no deletes).
  It cannot be executed here: commands and expected outputs come from the code and local dry runs; server-specific steps
  are marked unverified until a first real session.
- **Two-round confirmation.** Clusters found from real scores are *estimated*; a cluster is *confirmed* only after
  removing it, retraining and passing the two-step verdict (macro-KS gate, then per-metric CIs, Holm). So the runbook
  covers Round A (the attribution chain) and Round B (one removal arm per chosen cluster, ~3 trainings each), and the
  dashboard marks findings estimated vs confirmed.
- **Open, not yet answered:** which dataset the runbook's Round A uses — the planted-bad-data `v3/candidate` (113 jobs,
  adds recovery scoring; orchestrator's recommendation) or the clean `v0/candidate` (112 jobs; the documented first-run
  command from the training-attribution owner ruling). The planner should raise this as a question (BLOCKING only if it
  changes the plan).

## What the orchestrator told the owner (high level, to keep consistent)

- Inputs that exist (measured): per-episode, per-metric CUPID influence scores (family-W returns and per-failure-type
  columns, `harm_` sign folding), the ensemble table, and `rollout_episode_influence.npy` (rollout × training-episode
  matrix written by `attribution.attribute_run`), `functionals.json`, the catalog, verdicts, `vlapoc.analysis`
  (`metric_groups`, `aggregate_by`, heatmap, trade-off scatter), `vlapoc.recovery`, `vlapoc.step4`.
- **Lens 1, label groups:** `aggregate_by` over catalog columns. Measured limitation: the LIBERO catalog has
  `operator = None`, `collection_date = None`, a single `source`; only `task_name` (20), `scene` (10), `suite` (2),
  `source_file` (20, = task) vary. "Operator 3"-style findings need data that records them.
- **Lens 2, rule-like slices:** a slice finder over per-episode behaviour descriptors (catalog `length_steps`,
  `duration_s`, `smoothness`, `idle_fraction`, `idle_steps`, `gripper_toggles`; the ground-truth feature's label-free
  heuristics H1–H8 incl. task-relative length and post-release share) against each metric's harm score. Candidate
  off-the-shelf tool: `sliceline` (`Slicefinder.fit(X, errors)`, `top_slices_`, `top_slices_statistics_`; params alpha,
  k, max_l, min_sup — quoted from its docs). **Unverified:** whether it accepts continuous scores / needs discretised X,
  and whether it installs on this venv's Python 3.14. PoC example to match: "episodes over 400 steps are net harmful"
  (§1 p.3).
- **Lens 3, four metric groups:** `metric_groups` (owner-approved ≥2-of-5 rule), heatmap, scatter quadrants (§4.2, §4.5).
- **Patterns:** per cluster, how its descriptors/labels differ from the rest, which metrics it moves and which way, the
  most influential member videos. On v3 the planted corruption labels (answer key) are used only *after* the method, to
  score whether discovered clusters match planted types (blindness unchanged).
- **Causal caveat (quoted, CUPID p.11):** summed individual influence approximates group effects; "one could approximate
  group effects ... via random sampling". Hence estimated vs confirmed.
- **Patterns are per policy and metric** (SmolVLA on that subset); cross-policy comparison is the PoC scale-up stage.
- **Dashboard views (PoC §2.7 p.5, quoted):** "select a failed rollout, see its failure type, the most influential training
  episodes (with video) and their source datasets; or browse a ranked list of harmful episodes, aggregated by any catalog
  column." PhAIL's run explorer is the PoC's design reference for the rollout side. Plus a cluster view (heatmap, scatter,
  slice list with example videos, verdict per confirmed cluster, linked via lineage).

## PoC sections to hold the feature to

§1 p.3 (granularity; "aggregated by metadata"), §2.6 p.5, **§2.7 p.5 (demonstration surface)**, §3.4 (declared
functional on every result), §4.2 p.9–10 (four groups), §4.4 (two-step verdict), §4.5 p.11 (heatmap, scatter, metric-
targeted hypotheses), §9.2 p.19 ("which kind of harm each data group causes"), §10 p.20 "Verification & demo"
(counterfactual removal tests; dashboard linking failed rollouts to influential episodes), §2.1 item 5 (group leave-one-out
— the confirmation arms). Papers: CUPID (`docs/papers/CUPID_Influence_Functions_arXiv-2506.19121.pdf`), DataMIL (cluster-
level influence, p.6), PhAIL (rollout-side design reference), ATHENA.

## Owner decisions carried (binding)

SmolVLA; per-subset normalization; all PoC metrics, family W declared; ghost events out of scope; owner ruling B (n_boot
1000); ground-truth rulings (clip, all types but viewpoint, ruling A, limitation D: beat each type's best heuristic);
training-attribution rulings (scope; Q1 both curation modes, per-task headline; CUPID ±1 for yes/no columns; first
server run 112 jobs / ~96 GPU-h per_task + f30 hrt60_lin; 2-of-5 group rule owner-approved). Reports are written for a
non-specialist (summarizer rule). Model routing: planner and analyzer Opus 5.5; executor, refactorer, summarizer Sonnet 5.5.

## Constraints

- No CUDA or cluster access from this session. Never fabricate results; synthetic data is always labelled SYNTHETIC.
- Blindness: dashboard and cluster code that reads training data goes through `open_for_method`; answer-key labels are
  read only by scoring code, never by cluster discovery.
- The dashboard must read results from the lineage store (MLflow runs, recorded artifacts), not from ad-hoc paths, so
  real server results replace synthetic ones by pointing at different runs.
- New dependencies need a check that they install on this venv (Python 3.14) and a reason; prefer what is installed.
- Scratch under `data/scratch/` only, never `/tmp`. Disk: keep above 20 GiB free.
- Git: branch `dashboard`; commit locally in small steps with explicit paths (never `git commit -a`); messages end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; never push; don't commit the owner's uncommitted files
  (`.claude/skills/call-notes/SKILL.md`, `CLAUDE.md`, `calls/README.md`, `calls/instructions/call-instructions.md`).
  Note: a pointer to the runbook skill in `CLAUDE.md` would touch an owner-modified file — propose the exact line in the
  plan; the orchestrator asks the owner.
- Agents here stall and hit usage limits: write and commit incrementally; write the first draft of any file early.
- Refused writes: return the full content in your reply; the orchestrator saves it verbatim.

## Owner rulings

(none yet)
