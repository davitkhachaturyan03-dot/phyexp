# Brief: training and attribution (compliance and gap-closing for the new branch)

**Date:** 2026-10-20
**Feature folder:** `tracker/features/2026-10-20-training-attribution/`
**Git branch:** `training-attribution` (local, created from `origin/attribution-curation-merged` at `efee1db`).
Merges into `main` only after this pipeline passes. **Never push** unless the owner asks.

## Request (verbatim)

> Okay, so you have to treat what I say as a feature. So you have to spool up a feature and use all your skills
> to do that in order to make sure that the codes which came as a part of the new branch, the new codes, they
> basically comply to a new training and attribution feature, and also they adhere to everything we have in our
> POC document. And in case you notice any gaps, they have to be fixed by the corresponding agents which will
> work when you develop the new feature.

## What this feature is (orchestrator's reading)

The branch `attribution-curation-merged` (3 commits on top of `main` at `625b953`, 22 files, about 2,000 lines)
ported code from the owner's cluster project `VLA-data-curation` (`src/vlacur/`) into this repo. Its own
`brief.md` (read it in `tracker/features/2026-10-18-attribution-curation/`) says it used only `vla-build`:
**no plan review, no analyzer round, no refactor, and a 56-line report.** Nobody independently verified it.

This feature **plans, verifies and fixes that code against the PoC and our earlier features**, covering:
- **Step 3, attribution:** `vlapoc/influence.py` (projected per-sample gradients of a SmolVLA checkpoint) and
  `vlapoc/attribution.py` (the CUPID estimator: influence of every training episode on every family-W return and
  every per-failure-type return, with same-task and all-rollout columns, `harm_` sign folding, Kendall's τ,
  z-scored seed ensembles).
- **Step 4, curation:** `lineage.drop_top_k_per_task`, `drop_random_k_per_task`, `runs.select_run` ops.
- **Training at scale:** `scripts/stage.py` (one tracked stage per call), `scripts/server_pipeline.py` (the whole
  experiment as one SLURM dependency chain), `slurm/env.sh`, `gpu.sbatch`, `cpu.sbatch`, `guard.sh`.
- **Verification:** `scripts/verify_attribution.py` and the artifacts under the branch's feature folder.

"Comply with the PoC" means every requirement of the sections below that this code touches is met, or the
divergence is recorded and owner-approved. Where a gap is found, **the planner writes a story, the executor fixes
it, and the analyzer re-checks**, as in every earlier feature.

## PoC sections to hold the code to (read them yourself, from the PDF)

§2.1 (lineage first; evaluation protocol; injection; "Group-level leave-one-out"; random subsets) · §2.3–2.4 ·
§3.4 ("Every attribution result states which functional it explains") · §4 (metric-resolved attribution, §4.3 one
return per metric per rollout, §4.4 two-step verdict, §4.5 hypotheses) · §6 (reference papers: CUPID is "the best
starting point to fork") · §8.10 · §9 (architecture) and §9.1 Steps 3 and 4 and §9.2 · §10 rows "Fine-grained
attribution", "Metric-resolved attribution" and "Verification & demo". Defining paper:
`docs/papers/CUPID_Influence_Functions_arXiv-2506.19121.pdf` (read its formulation and its released code at
`agiachris/cupid`; also ATHENA and QoQ for the gradient-projection and aggregation choices).

## Things the orchestrator already knows the planner must check (leads, not conclusions)

1. **Blindness.** Ground-truth feature known issue: `train_run` and `eval_run` bypass `open_for_method`, so an
   attribution method that trains or reads data could read a fenced or leaky version. Does the new code route its
   inputs through `open_for_method`? v1 and v2 are fenced; the corrupted version is v3.
2. **CUPID fidelity.** Is the estimator the CUPID one (influence on returns, Hessian handling, projection,
   per-demo aggregation), and what exactly differs from the paper and the released code? The branch's changelog
   claims the estimator matches the cluster version to 3.2e-13, which only proves it matches the port source, not
   the paper.
3. **The key contract (§9.2).** Scores keyed by (`episode_id`, `content_hash`); `check_scores` must reject scores
   for episode versions a run did not train on. Does the attribute stage enforce it?
4. **Per-task curation vs §9.1 Step 4.** The PoC says remove the top-k most harmful, with random-k as the control.
   The branch drops per task and uses "matched" random controls. Does it match the spec, and is the choice
   declared and owner-approved?
5. **Which functional is explained (§3.4).** Family W returns are the declared attribution family; the verdict must
   verify the same functional. Is the attribution's target declared per column?
6. **The verification gate.** After retraining on curated subsets, the evaluation layer's verdict (macro-KS gate,
   then per-metric CIs, Holm) is the check; owner ruling B says n_boot stays 1000 and near-boundary verdicts may
   depend on the seed. Is it wired in?
7. **Normalization.** Per-subset statistics are the default (owner decision). The cluster recipe differed
   (`lerobot-train` CLI, 30k steps, camera rename). The branch says it uses the repo recipe unchanged: check.
8. **Attribution-grade assertion and the fixed batch plan** from the evaluation layer. Do the stages respect them?
9. **Lineage and records.** Every stage a tracked MLflow run; the guard refuses dirty code or a different commit;
   paths relative (the branch touched path recording for symlinked `data/`). Do older verifiers still pass?
10. **Hardcoded environment.** `slurm/env.sh` and the sbatch scripts hardcode `/raid/chem/gor/...` and
    `/auto/home/gor/...`. Is that acceptable as server config, documented, and separable?
11. **Out-of-scope PoC pieces** this feature does not own (Step 1 screening with DemInf/RoboDrop-style scores, the
    §9.2 screening-vs-attribution comparison, real-robot stage, scale-up): list them as gaps with a reason, and
    do not silently expand. The owner can decide whether to add stories.

## Owner decisions carried (binding)

- SmolVLA; per-subset normalization; all PoC metrics (family W is the declared attribution family); ghost events out
  of scope; owner ruling B (n_boot 1000); ground-truth rulings (clip, all types except viewpoint, ruling A,
  detectability limitation D: methods must beat each type's best simple heuristic).
- Reports are written for a non-specialist reader (summarizer instructions: plain words, glossary).
- Model routing: planner and analyzer on Opus 5.5; executor, refactorer, summarizer on Sonnet 5.5.

## Constraints

- **This Mac has no CUDA and no cluster access from this session.** What the branch ran on the cluster is evidence
  only in its artifacts (for example `verify_gpu_smoke_clean_full_s0.json`); the planner must separate what can be
  re-verified here (synthetic checks, CPU/MPS smoke, dry-run of the SLURM chain, static review) from what needs the
  server, and write the latter as "pending server" with runbook commands. Never fabricate results.
- The cluster's `data/` is a symlink to `/raid`; on this Mac `data/` is local. Disk: about 73 GiB free; keep above
  20 GiB.
- Scratch under `data/scratch/` only, never `/tmp`. `HF_DATASETS_CACHE=$PWD/data/hf_datasets_cache`.
- **Git:** work on branch `training-attribution`; commit locally in small steps; messages end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; never push; don't commit the owner's uncommitted files
  (`.claude/skills/call-notes/SKILL.md`, `CLAUDE.md`, `calls/README.md`, `calls/instructions/call-instructions.md`).
- Agents here stall on long silent stretches and hit usage limits: write and commit incrementally; write the first
  draft of any file early.
- Refused writes: return the full content in your reply; the orchestrator saves it verbatim.

## Owner answers to plan questions

- **2026-10-20, scope: "Just dismiss the parts that need the server and fix everything else".**
  - Anything that can only be checked or fixed on the server (GPU training, rollouts, feature extraction on
    real checkpoints, running the SLURM chain for real, verdicts on retrained arms) is **out of scope** for this
    feature. List it once, briefly, under "Out of scope (needs the server)" so the report stays honest. Write no
    stories, acceptance criteria or new runbook work for it.
  - **Every other gap is in scope and must be fixed:** code, logic, PoC compliance, blindness, key contracts,
    declared functionals, curation rules, lineage, tests on synthetic data, CPU/MPS smoke checks, the SLURM chain's
    `--dry-run`, config and documentation.

## Owner rulings

(none yet)
