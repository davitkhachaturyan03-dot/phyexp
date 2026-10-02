# Brief: evaluation layer (PhAIL primitives)

**Date:** 2026-10-01
**Feature folder:** `tracker/features/2026-10-01-evaluation-layer/`

## Request (verbatim)

> also fix the leak now as a separate story inside the feature.
>
> start building the evaluation FEATURE, make sure to include the two stories mentioned as of
> now. make sure to cover a maximal scope of what's given in the document as the evaluation
> feature. per metric attribution and phail usage is crucial for this POC.

"The two stories mentioned as of now" are (orchestrator's reading, from the conversation):
1. **Answer-key leak fix** (details below), as its own story inside this feature.
2. **Per-sub-task time-to-success T for libero_10**, as its own story.

## Owner decisions carried into this feature (2026-10-01, from the conversation)

These are binding. Don't re-ask; plan around them.

1. **Compute:** the owner will arrange a remote Linux server *later*. LIBERO rollouts and SmolVLA
   training at scale run there. The server does not exist yet. The owner said "start building"
   now, so build and verify everything that can run on this Mac, and mark server-dependent work
   explicitly (see Constraints).
2. **Policy to evaluate:** SmolVLA (PoC §10 Foundation: "LIBERO (one or two suites) + SmolVLA").
3. **Normalization statistics: per-subset.** Every training run normalizes with mean/std computed
   only from its own training episodes. The owner's reasoning: statistics computed over data a run
   doesn't train on (corrupted episodes, clean originals or held-out references) leak into it.
   Per-subset makes normalization a function of the training data, so data stays the only variable
   (PoC p. 1), and a removal's effect on the statistics is part of its counterfactual effect.
   Measured fact: SmolVLA normalizes STATE and ACTION with MEAN_STD (installed
   `lerobot/policies/smolvla/configuration_smolvla.py`); LeRobot currently uses the full
   version's `meta/stats.json` even with `episodes=`. Implementation lead (inferred, unverified):
   compute subset stats from the subset's frames and pass them to LeRobot `train()` via
   `vlapoc/lerobot_train_entry.py`, instead of `recompute_stats` on a ~7 GB copy. Wherever this
   feature trains or defines training, it applies.
4. **Metrics: cover all metrics the PoC specifies**, not one headline. §3.2 primitives
   (time-to-success T, ghost events, right-censoring, Kaplan–Meier CDF, Success@τ, median, RMST,
   UPH, completion, MTBF/A, HRT, episode-clustered bootstrap CIs, macro-KS, P-P plots) and the
   §4/§10 attribution targets (F(30 s), HRT, RMST, AUC, ghost-event rate). Where a metric doesn't
   transfer to simulation, say so rather than silently dropping it.
5. **Units of T:** probably seconds, but the planner decides from the PoC (§3.5A allows "time
   (or steps)"). The dataset is 20 Hz (`info.json` fps 20).
6. **Ghost events:** use LIBERO's own state/goal rules if LIBERO defines any that fit "object off
   the table, out of reach" (§3.5A). If LIBERO gives no ghost-event rules, move ghost events out
   of scope and state that explicitly in this feature's final report.
7. **Per-sub-task T for libero_10:** build it in this feature as a separate story, baked into the
   existing code if worth it (§3.5A: "for long-horizon LIBERO-10 tasks, per sub-task").
8. **Crucial emphasis (owner):** per-metric attribution readiness (PoC §4, §10 "Metric-resolved
   attribution") and PhAIL usage (§3, especially §3.4 and §3.5). The evaluation outputs must be
   shaped so the later attribution feature can compute one return per metric per rollout
   (§3.4 "Finer rollout signal for CUPID-style influence on returns"; §4).

## Story: the answer-key leak (owner: fix now, separate story)

Verified by the orchestrator on 2026-10-01 against the bytes. v1's episode table
(`data/lineage/versions/16076b9c…/episodes.parquet`, and its catalog copy) has columns
`content_hash` and `parent_content_hash`. They differ for exactly 100 of 1000 rows, and that set
equals the `episode_id` set in `labels.parquet` (`changed set == labelled set: True`). So any
method reading the version's episode table (`Subset.episodes()` / `version_episodes()`) can
identify the corrupted episodes without the labels. The plan of the data+lineage feature required
the column there (its AC4.1); its analyzer missed this in both rounds.

The leak is to the *methods under test* (screening, attribution, and the agents that will write
them), which must be blind to ground truth (PoC §2.1 item 4, "check whether your attribution
method flags exactly those"). The fix direction the owner accepted: move the pre-corruption
fingerprint out of the method-facing episode table into lineage-only storage, keep v1's version id
and the label join working, and add a check that no bookkeeping column of a derived version's
method-facing table distinguishes edited episodes. Data-derived columns that legitimately differ
(e.g. `gripper_toggles` on noised episodes) are not a leak. The data+lineage feature's report
needs a corresponding note; record it through that feature's `changelog.md` per vla-build step 5.

## Current state of the repo (verified by the orchestrator, 2026-10-01)

- **Data + lineage layer is built and passed** (`tracker/features/2026-09-28-data-lineage-layer/`:
  read its `report.html` "Open decisions and known issues", `plan.md` Risks R1–R7, and
  `analysis-r2.md` Non-blocking).
- **Datasets:** v0 `data/lerobot/libero_spatial_10/` (1000 demos, 200,340 frames, fps 20, 128²,
  lossless) and v1 `data/lerobot/libero_spatial_10__v1/`.
- **Lineage store:** `data/lineage/`, with MLflow at `data/lineage/mlflow.db`.
- **Code:** `vlapoc/` (`libero_convert`, `catalog`, `lineage`, `derive`, `runs`,
  `lerobot_train_entry`) and `scripts/`.
- **Catalog:** has per-episode `duration_s` (human time-to-success, the §3.5 HRT reference; the last
  step is the only reward, 1000/1000).
- **Evaluation stand-in:** the S6 eval stand-in (`runs.py`) records rollout plans with outcomes
  `not_run`. This feature replaces or extends it.
- **Checkpoints:** only 5-step ACT smoke checkpoints exist; no SmolVLA checkpoint.
- **Git:** local repo at commit `599b923`, no remote.
- **Disk:** 118 GiB free. Each distinct LeRobot training subset writes ~6.5 GB of HF cache (R1).
- **Python:** `.venv/bin/python` 3.14. lerobot 0.6.1 declares Python 3.12/3.13 only; the
  `lerobot-train` CLI is broken on 3.14 (draccus #69), and the entry point works around it.
- **Machine:** Apple M4 Pro, 24 GB, MPS, no CUDA.

## Facts the orchestrator fetched (verify further yourself; they are leads)

- **LeRobot LIBERO docs** (https://huggingface.co/docs/lerobot/libero, fetched 2026-10-01):
  - "LIBERO requires Linux (`sys_platform == 'linux'`)".
  - Eval flags include `--env.task`, `--env.task_ids`, `--eval.n_episodes`, `--env.init_states`,
    `--env.control_mode`.
  - Recommended "10 episodes per task … averaging over 3 seeds". The PoC §2.1 asks for "roughly 50
    or more, fixed seeds and initial states".
  - Actions are `Box(-1, 1, shape=(7,))`.
  - The page doesn't say whether `lerobot-eval` records the step at which each episode succeeds,
    or exposes simulator state for ghost detection. Check the source.
- **The LeRobot LIBERO env default render size** is quoted in the data+lineage plan as 360; our
  dataset is 128². Unverified by the orchestrator.
- **PhAIL's analysis pipeline** appears to be public at
  https://github.com/Positronic-Robotics/phail-paper, and the positronic framework at
  https://github.com/Positronic-Robotics/positronic (found via web search, not opened).
  - PoC p. 8: "Confirm which parts of the dataset, including the fine-tuning set, are downloadable
    and under what license."
  - PhAIL publishes ~990 evaluation episodes with "synchronized video, telemetry and event
    annotations" (§3.1, quoted).
  - Check this before writing any KM/HRT/bootstrap/macro-KS code. Also check whether PhAIL's
    published rollouts could validate our metric implementations on real data (orchestrator's
    suggestion, inferred). Standard libraries (lifelines, scipy) are also prior art to check.

## Constraints

- **Check off-the-shelf first and verify against the artifact,** per CLAUDE.md and vla-build.
  Label every claim quoted / measured / inferred.
- **No server:** LIBERO rollouts can't run here via LeRobot (Linux-only, quoted), and real SmolVLA
  training needs a GPU. Build and verify locally whatever can be verified locally: the metric
  layer on real data such as the human demo durations, LIBERO goal and sub-task rules from BDDL,
  the rollout harness code and record schema, and per-metric return definitions. Mark every
  server-dependent AC explicitly as "pending server", with exactly what will be run there.
  - **Never fabricate rollout outcomes.** Synthetic test data must be labelled synthetic.
  - Whether LIBERO itself (outside LeRobot's extra) can run on macOS is an open question to check
    (inferred possibility, unverified). Don't spend the session on it if it fails.
- **Disk:** keep free space above 20 GiB. Use `HF_DATASETS_CACHE=$PWD/data/hf_datasets_cache`;
  scratch goes under `data/scratch/`.
- **Lineage:** every new stage must keep the lineage contract (episode versions, run records in
  MLflow).
- **Git:** local commits only, never push, nothing under `data/` or `.venv/`. Messages end with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Owner answers to plan questions

Answered 2026-10-02:

- **BLOCKING Q1, v1's positional leak: "A. Fix code, fence v1 (Recommended)".**
  - Fix `derive_version` to preserve the parent's episode order.
  - Add the `check_blind` guard, and fence v1 off from method use.
  - The injection feature derives the real corrupted versions with the fixed code.
  - Don't re-derive v1 now; the data+lineage S6 evidence stays intact.
- **Downloads: "Approve all (Recommended)".** Approved:
  - the PhAIL metadata sidecars (~180 MB) and the phail-paper repo files (~52 MB);
  - LIBERO simulation assets from the HF Hub;
  - the `HuggingFaceVLA/smolvla_libero` checkpoint (harness validation only);
  - the SmolVLA base model for the S8 2-step check.

  Record source, revision and size for every download. PhAIL's 13.95 GB fine-tuning set is **not**
  approved for download in this feature.
- **CLAUDE.md / vla-build reward wording: "Yes, correct it".** The orchestrator makes this edit
  (not an agent), after verifying the planner's P6 finding against LIBERO's `create_dataset.py`.
  S3 still records the correction in the data+lineage changelog.
- **Non-blocking defaults stand** (no override from the owner): T in seconds of simulated time;
  τ_episode = 60 s; ghost events out of scope for LIBERO (decision 6), stated in the report; S8
  stays in this feature; the HRT anchor is the v0 originals.

## Owner rulings

(none yet)
