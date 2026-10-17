# Brief: ground truth (injection experiment)

**Date:** 2026-10-15
**Feature folder:** `tracker/features/2026-10-15-ground-truth/`

## Request (verbatim)

> start building the ground truth feature

Context from the conversation: the orchestrator had proposed this as the next feature — PoC §10 p. 20
"Ground truth | Injection experiment with 5–10% corrupted episodes, using LIBERO's original demos as clean
references", with the corruption types of §2.5 p. 5 and the metric-targeted corruptions of §8.10 p. 17. The
data+lineage feature deferred the full corruption suite to this feature (owner-approved, 2026-09-29).

## Owner decisions for this feature (2026-10-15, binding)

1. **Clipping: "Clip to [−1, 1]".** Every injected corruption keeps actions inside LIBERO's valid command
   range (`Box(-1, 1, shape=(7,))`, quoted LeRobot LIBERO docs), so corrupted episodes look like plausible bad
   data rather than being detectable by impossible values. Record the clipping (and how much it bit: fraction
   of clipped values) in each label's `params`. The existing v1 (unclipped, fenced) is not rebuilt.
2. **Scope: "All except viewpoint".** In scope: §2.5 noisy actions, time-shifted actions, swapped
   instructions, dropped frames, low-skill / hesitant demos; §8.10 idle/hesitant padding and gripper-release
   noise. **Out of scope: altered viewpoint** (needs simulator re-rendering) — record it as owner-approved out
   of scope in the report.
3. **Blindness enforcement: "Enforce now".** Add a single method-facing loader that always calls
   `lineage.assert_blind` (and respects the v1 fence), so later screening / attribution features cannot read a
   leaky or fenced version by accident. The answer key stays readable only through an explicitly
   ground-truth-side path (for scoring recovery@k).

## Decisions carried from earlier features (binding)

- Policy is SmolVLA; **per-subset normalization** is the training default (evaluation feature, owner decision 3).
- Owner ruling B (evaluation feature): verdict n_boot stays 1000; near-boundary verdicts may depend on seed.
- MLflow local; git local only; local commits allowed, never push, nothing under `data/` or `.venv*`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## What already exists (verify yourself; these are pointers)

- **Mechanism (data+lineage S4, evaluation S0):** `vlapoc/derive.py` `derive_version` (order-preserving since
  the S0 fix; ~8.9 min and a full ~7 GB copy per full-scale derivation, inferred by the S0 executor), ground
  truth only in `labels.parquet`, parent hashes in lineage-only `parents.parquet`, `check_blind` /
  `assert_blind` and the v1 `fence.json` in `vlapoc/lineage.py`, the reference-split guard (40 clean references,
  2 per task; 960 candidates), length-changing edits supported.
- **One demo corruption exists:** v1 = RoboDrop App. A.1 action noise (0.25·σ_a, all 7 dims, **unclipped**) on
  100 candidates (5 per task). v1 is fenced (positional leak) and stays as is.
- **Evaluation layer:** rollouts, per-metric returns (family W and O), verdict gate — the downstream consumers.
  `tracker/features/2026-10-01-evaluation-layer/report.html` "Open decisions and known issues".
- **Defining sources:** RoboDrop App. A.1 p. 13 (action corruption "zero-mean Gaussian noise with standard
  deviation 0.25σa", temporal corruption "shifts the action sequence by two seconds", 10% per task, 2 clean
  references per task); PoC §2.5 p. 5, §8.7 p. 15, §8.10 p. 17 ("pad some with idle or hesitant segments
  (should hurt speed, not reliability) and add gripper-release noise to others"). Dataset fps is 20, so two
  seconds = 40 steps.

## Constraints

- Check off-the-shelf first and verify against the artifact; label claims quoted / measured / inferred.
- **No server yet.** Locally you can verify that corruptions are applied correctly, labelled, blind, clipped and
  traceable. You cannot show that a corruption *hurts* a trained policy (that needs SmolVLA training and
  rollouts) — mark those ACs "pending server" and add them to the evaluation feature's server runbook pattern.
  Never fabricate outcomes.
- **Disk: 91 GiB free.** A full-scale derived version costs ~7 GB plus transient. Plan the number of
  full-scale derivations explicitly; prefer one combined derived version or small-scale tests where they prove
  the same thing. Keep free space above 20 GiB. `HF_DATASETS_CACHE=$PWD/data/hf_datasets_cache`; scratch under
  `data/scratch/`; `LIBERO_CONFIG_PATH=$PWD/data/libero_config`.
- Known quirk: LeRobot's `task_index` order in derived datasets depends on `PYTHONHASHSEED` (content hashes
  unaffected). Swapped-instruction corruptions touch tasks, so check this interacts safely.
- Agents in this repo stall on long silent stretches and hit usage limits: write and commit incrementally.

## Owner answers to plan questions

(none yet)

## Owner rulings

### 2026-10-17: `dispute-r1.md` (corruption detectability)

- **Question put to the owner:** "Ground truth dispute: 5 of 7 corruption types are found by one-line
  statistics. Which detectability leaks should be removed in the single re-derivation?"
- **Owner's decision (verbatim):** "A. Fix time_shift only (rec.)".
- **Meaning, per the case file's Option A:**
  - Replace the `time_shift` tail fill (40 steps of arm = 0) with a trim: obs[0:T−40] paired with a[40:T],
    with the length change recorded in the label.
  - Keep everything else.
  - Report recovery per type against the heuristic baselines, saying plainly which types test anything beyond
    one-line statistics.
  - B (hiding provenance, correlated noise) and C (re-rendered idle) are not built; record them as open
    questions or known limitations.
- **Final and owner-approved.** The analyzer does not re-open it.
