# Brief: curation headline results in the dashboard

**Date:** 2026-11-05
**Feature folder:** `tracker/features/2026-11-05-curation-results/`
**Git branch:** `curation-results`, created from `dashboard` at `bad0241` (the owner's own commit on top of the dashboard
feature). `training-attribution` and `dashboard` are PASSED but not merged into `main`. **Never push; never merge** unless
the owner asks.

## Request (verbatim)

> can you add the curation headline results to the dashboard? treat this effort as a feature

## Context from the conversation (what the orchestrator told the owner, 2026-11-05)

The owner asked "are all of these results accurately represented inside the dashboard you created?". The orchestrator
searched `dashboard/` for every result type each stage writes (measured) and reported:

- **Not shown anywhere in the dashboard:** Round A's curation verdicts — each curated arm (`harm-f30-k5`, `help-f30-k5`,
  `harm-hrt60_lin-k5`, `help-hrt60_lin-k5`, `random0-k5`, `random1-k5`, `drop-random-20`) judged by the two-step verdict
  (`vs_base`, `curated_vs_random`), the PoC §4.5 p.11 **"same dataset, opposite curations"** head-to-head
  (`head_to_head`: harm-hrt60_lin vs harm-f30), which the PoC calls "the headline result of the PoC", and the Step 4
  seed-consistency summary (`step4-summary`, `vlapoc.step4`; gate-aware since training-attribution round 2). These are
  recorded as MLflow runs/artifacts by the chain (`scripts/server_pipeline.py`, `scripts/stage.py`, `vlapoc.verdict`,
  `vlapoc.step4`) but the dashboard never reads them.
- **Other gaps the orchestrator listed (NOT requested in this message):**
  1. how well the attribution ranking recovered the planted episodes (`score` stage, `vlapoc.recovery`: per-type
     AUROC / recall next to each type's best heuristic) — only cluster-vs-planted overlap is shown;
  2. a "planted type" column on the Episodes ranked list and on the Rollouts page's influential-episode list;
  3. Kendall's τ metric agreement and the evaluation layer's per-policy time-to-success (KM) curves;
  4. cluster effect intervals still use the percentile bootstrap (`vlapoc/clusters.py` docstring: "optimistic ... These are
     estimates to confirm, not findings"); the analyzer switched only the §9.2 readings to Welch.
  The planner should list these under "Out of scope / owner question" rather than silently adding them; it may raise
  a question if one of them is needed for the headline to make sense to a stakeholder (e.g. 1 or 2).

## What "done" means (orchestrator's reading, for the planner to ground)

A stakeholder opening the dashboard can see, in plain language, with the declared functional reachable (§3.4):
- per curated arm and metric: the two-step verdict (macro-KS gate, then per-metric bootstrap CI, Holm across metrics; owner
  ruling B n_boot 1000), vs the base and vs random removal, per seed and across seeds (the step-4 rule);
- the §4.5 head-to-head: does curating for speed (HRT) win on speed and curating for reliability (F30) win on reliability —
  stated with the same bounded, "never 'harmless'" wording discipline as the §9.2 page, and "inconclusive" where the gate
  did not fire;
- which training episodes each curated arm removed (link to the select run; with lineage);
- SYNTHETIC / SMOKE / REAL provenance enforced exactly as on the other pages; until the server runs, a labelled synthetic
  demo of the page built into `data/lineage_demo` by the existing demo builder (`scripts/dashboard_demo.py`), with the
  drop-in proof that real Round A runs render with no code change.

## PoC sections

§4.5 p.11 ("same dataset, opposite curations" — headline), §4.4 p.10 (two-step verdict, Holm), §2.6 p.5 (curated vs random
removal), §9.1 Step 4 p.19 (removal tests; "Output: Confirmation, or refutation, that the attribution is causally
correct"), §2.3 (2–3 seeds), §2.7 p.5 (demonstration surface), §3.4 (declared functional), §10 p.20 "Verification & demo".

## Owner decisions carried (binding)

All earlier rulings (see `tracker/features/2026-10-22-dashboard/brief.md` and `2026-10-20-training-attribution/brief.md`):
SmolVLA; family W; owner ruling B (n_boot 1000); Q1 curation both modes, per-task headline; Round A on v3/candidate
(116 jobs, `--curation per_task --metrics f30 hrt60_lin`); CUPID ±1 for yes/no columns; 2-of-5 group rule; live local
Streamlit app; synthetic data labelled; plain-language presentation layer (dashboard round 2, AC5.9/5.10 incl. real-browser
render checks). Reports are written for a non-specialist.

## Constraints

- **No change to the server chain's jobs or counts** unless the plan proves one is needed: the runbook (`server-run` skill)
  and its verified counts (Round A 116 / 24, planted Round B 129 / 33, cluster Round B 21 / 6) must stay valid;
  `scripts/verify_runbook.py` must still pass. If a needed output is not recorded by the chain today, that is a plan item
  with an explicit runbook/count consequence.
- Read results from the lineage store only (MLflow runs/artifacts), like the other pages.
- The dashboard's real-browser checks (`verify_dashboard.py --render`) must cover the new page.
- Scratch under `data/scratch/` only, never `/tmp`. Real store `data/lineage/mlflow.db` sha256 must stay
  `306af4123fda7e635892d459a2e34008ba1cb62f6c4fce152c546e0da7bdde79`.
- **`data.zip` in the repo root (about 90 GB, untracked, the owner's) must never be added, moved, read or deleted.** Never
  use `git add -A` / `git add .` / `git commit -a`; add explicit paths only.
- Git: commit locally in small steps; messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; never
  push; never touch `CLAUDE.md`, `calls/`, or `.claude/skills/call-notes/`.
- Agents stall and hit usage limits: write and commit incrementally; write first drafts early.
- Refused writes: return the full content in your reply; the orchestrator saves it verbatim.

## Owner rulings

(none yet)
