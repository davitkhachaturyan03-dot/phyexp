# Plan: Evaluation layer (PhAIL primitives on LIBERO, validated on PhAIL's real-robot data)

**Round:** 3 (owner ruling on `dispute-r2.md`, 2026-10-07). Round 2 (plan rework after `analysis-r1.md`) was
2026-10-05; round 1 was written 2026-10-02.
**Feature folder:** `tracker/features/2026-10-01-evaluation-layer/`. **Brief:** [`brief.md`](brief.md) (binding,
including the owner answers of 2026-10-02).

Evidence labels used throughout: **quoted** (from a source, with the quote or file/line), **measured** (run in
a planning session, or measured by the executor/analyzer and cited from `execution-log.md` /
`analysis-r1.md`), **inferred** (planner's reasoning, unverified).

## Changes in this revision (r3)

This applies the owner ruling of 2026-10-06 on `dispute-r2.md`: **"B. Accept 4/5 (executor)"**, final (brief.md,
"Owner rulings"). Nothing else changed.

1. **AC7.1 restated.** "GR00T vs ACT → (ii) in ≥ 4/5 seeds; the gate fires in 5/5 and RMST favours GR00T in 5/5;
   the AUC direction (ACT) is at the Holm boundary (Holm p 0.024–0.054)". The other three pairs are unchanged
   (5/5).
2. **n_boot stays 1000**, including for AC7.5's server verdict commands. AC7.5 now notes that, under this
   ruling, a server verdict near the Holm boundary may depend on the bootstrap seed.
3. **Owner-approved divergence, 2026-10-06, `dispute-r2.md`.** The PhAIL GR00T–ACT trade-off (ii) is not
   robust to the bootstrap seed at n_boot 1000. The summarizer carries this into the report and the
   spec-coverage table (§3.3 / §4.4 rows) as owner-approved.

## Changes in this revision (r2)

Routed to the planner by `analysis-r1.md` (F-P1 to F-P5), plus the executor items turned into checkable ACs.
Round-2 work for the executor is every AC marked **(r2)**.

1. **F-P2: one declared functional per metric on libero_10** (new section "Declared functionals (r2)"):
   - **Choice: option (a).** Per-rollout returns (S6) and the verdict's gate and per-metric CIs (S7) both use
     **whole-task T** for the five rollout metrics: F(30 s), RMST(60), AUC vs human, HRT(60), censored(60).
   - Per-sub-task T (§3.5 A) is kept as a **separately declared operation family `*_op`**. It has its own
     verdict mode and its own operation-sample returns.
   - Justification and a counterexample showing why option (b) fails: see that section.
   - New identity ACs AC6.10 and AC7.7: mean(return) equals the verdict's cell metric to ≤ 1e-9, for both
     families, with the documented weighting for HRT.
   - The AUC functional on LIBERO is declared with ties counted ½. PhAIL validation keeps PhAIL's
     ties-dropped rule.
2. **F-P3: the operation return for an unfinished attempt is −60** (the cap), so it cannot outrank a completed
   operation. The operation family's attribution functional is renamed `rmst60_op_cap` and is declared with
   the same convention. AC6.11 checks the ranking on real data. The PhAIL-semantics operation KM (censored)
   stays as the descriptive per-sub-task CDF (S4).
3. **F-P1: AC7.1 is restated from PhAIL's released artifact** (`dashboard_per_obj_macro.json`), not its p. 9
   prose:
   - OpenPI–GR00T → (iii); OpenPI–ACT → (iii), with the discrepancy recorded; GR00T–ACT → (ii); any vs
     SmolVLA → (i).
   - Each holds in 5/5 seeds on the deterministic export.
4. **F-P4: AC8.5 fine-tunes from `lerobot/smolvla_base@d9f33c94`.** The round-1 spec
   (`--policy.type=smolvla --policy.load_vlm_weights=true`, 2.03 GB SmolVLM2 backbone, not approved by the
   owner) is recorded as a divergence under S8.
5. **F-P5: a fixed batch plan is written into AC5.8 and AC7.5.** `compare_policies` refuses arms whose batch
   plans differ (new AC7.8).
6. **E1 → AC5.9 and AC7.9.** Attribution grade is asserted by the caller and checked against the protocol.
   Smoke runs are labelled "smoke"; `lineage_dry_run --eval-smoke` gives `verify.attribution_grade=false`.
   The mis-tagged round-1 runs are documented.
7. **E2 → AC6.8.** One metric registry is consumed by S1, S6 and S7. UPH = 3600 / RMST(τ_episode), matching
   `evaluation.uph`. The registry carries each metric's family and T definition.
8. **E3 → AC2.9.** The PhAIL cohort export is sorted by `s3_url` and byte-identical across `PYTHONHASHSEED`
   values. The S2 and S7 artifacts are regenerated, and p-values are disclosed as Monte-Carlo draws
   conditional on episode order.
9. **E5 → AC6.9.** No temporary directories leak from `returns` or `verdict`; the 11 leaked ones are removed.
10. **Non-blocking `t0_model_sha256`: IN (AC5.10).** `t0_state_sha256` cannot see fixture placement (measured,
    AC5.2b). A model-placement hash makes the "fixed initial states" of §2.1 item 3 p. 3 checkable per
    rollout.
11. **Non-blocking items folded in:**
    - AC5.3 reworded to "JSON-equal".
    - S7 records the accepted `gate_only` state and the fixed human anchor in the difference bootstrap.
    - The report must state the not-built failure types next to the ghost-event statement (AC6.4 note).
    - The fence must be enforced by later features (Out of scope).
    - pos3 is named as the reproduction route (AC2.9 note).
    - BLOCKING Q1 is marked resolved (owner: A).
12. **No rebuttals.** Every analyzer item is accepted.

---

## Declared functionals (r2)

This answers F-P2. The registry (AC6.8) is the single place that records it.

**Families.**

| Family | T | Censoring | Used by |
|---|---|---|---|
| **W (whole-task), the declared attribution family, both suites** | Time to full task success (`T_s`, the env's own `_check_success`). On libero_spatial this is the single operation | Only at τ_episode = 60 s = τ (a harness invariant: a rollout ends early only by success). So min(T, 60) is exactly observed for every rollout | S6 per-rollout returns; S7 gate (macro-KS on whole-task CDFs per task) and per-metric CIs; S1 headline tables; human anchor = whole-task `human_reference.parquet` |
| **O (operation), PoC §3.5 A "per sub-task"** | Inter-completion time T_k of operation k (S4 rule; PhAIL §3.1 semantics) | (1) Descriptive: PhAIL KM with the unfinished attempt as a censored pair (S4 CDFs, `rmst60_op`, `hrt60_op`), labelled descriptive. (2) Attribution: `rmst60_op_cap`, which counts the unfinished attempt as reaching the cap (60 s) | S4 reporting; S6 `operation_returns.parquet`; S7 `family="op"` (its own gate and CIs, labelled); human anchor = `human_operations.parquet` |

**Metrics and their per-rollout returns (family W).** "Cell" is (policy, task); macro is the equal-weight mean
over tasks.

| Metric (registry key) | Declared functional | Return column | Identity (per cell, ≤ 1e-9) |
|---|---|---|---|
| `f30` | F(30 s) of whole-task T | `ret_f30` = 1[event ∧ T ≤ 30] | mean = F(30) |
| `rmst60` | RMST(60) = ∫₀⁶⁰ (1 − F) | `ret_rmst60` = −min(T, 60) (censored → −60) | mean = −RMST(60) |
| `auc_h` | P(T < T_h) + ½ P(T = T_h), T_h from the task's whole-task human reference (Mann–Whitney; ties ½ because model and human share the 20 Hz grid) | `ret_auc_expected` = P(T_h > T) + ½ P(T_h = T); `ret_auc_sampled` = 1[T < T_h*] + ½·1[T = T_h*], recorded seed | mean(expected) = AUC; mean(sampled) = AUC up to sampling noise |
| `hrt60` | macro over tasks of RMST_h,o(60) / RMST_m,o(60) | `ret_hrt60_lin` = w_o·(−min(T, 60)), with w_o = (1/J)·RMST_h,o / RMST_m,o², evaluated at this policy | Σ_cells mean(`ret_hrt60_lin`) = −HRT(60) at the expansion point (the documented weighting) |
| `censored60` | 1 − F(60) | `ret_censored60` = 1[no success by 60 s] | mean = 1 − F(60) |

The ghost-event rate has no LIBERO return (owner decision 6). Median, UPH, completion and MTBF/A are
scoring-only (registry).

**Family O returns (operation samples).** In `operation_returns.parquet`, one row per operation sample: the
PhAIL pairs, i.e. completed operations plus the one unfinished attempt per incomplete rollout.
- `ret_rmst60_op_cap` = −min(T_k, 60) for a completed operation, and **−60 for the unfinished attempt**. It
  cannot exceed the return of any completed operation, since a completed operation has −min(T, 60) ≥ −60.
- Each row carries `t_start`/`t_end` steps, the segment whose (s′, a′) the operation's return can be credited
  to (§3.4 "several outcome samples per rollout").
- Identity, per cell over operation samples: mean(`ret_rmst60_op_cap`) = −`rmst60_op_cap`, as the S7 op arm
  computes it.

**Why (a), from the PoC:**
1. **The returns are per rollout.** §9.1 Step 3 p. 19: "Define one 'return' per metric for each rollout".
   §4.3 p. 10 lists rollout-level returns. CUPID's estimator weights a whole rollout by R(τ) (Prop. 1 p. 5)
   (quoted).
2. **Exactness.** In family W every rollout's capped time is observed, because censoring occurs only at
   τ_episode = τ. So each cell mean of a return *is* the KM functional, and the gate verifies exactly the
   attributed functional (§3.4 p. 7 "Declared targets"; §4.4 p. 10 "for that metric") (inferred from the
   harness invariant; checked by AC6.10).
3. **Option (b) cannot be both KM-consistent and progress-monotone** (inferred, worked example).
   - Under PhAIL's pair rule, a rollout that never finishes operation 1 contributes one censored pair. A
     rollout that finishes operation 1 at 10 s and then stalls contributes an event plus a censored pair.
   - Any per-rollout sum of KM-consistent per-pair terms (each censored term is the KM conditional
     expectation E[min(T, 60) | T > c] ≥ c) gives the second rollout ≈ −10 − 55 = −65. The first gets ≈ −60.
   - So rollouts that made progress would rank **below** those that made none.
4. **§3.5 A p. 7 ("for long-horizon LIBERO-10 tasks, per sub-task") is still honoured.** The per-sub-task T is
   built (S4), reported as the PhAIL-style per-operation CDF, verifiable by S7 in `family="op"`, and it
   carries operation-sample returns for the finer §3.4 signal. Its declared attribution functional is
   `rmst60_op_cap` (F-P3).
5. **Divergence, recorded.** On libero_10 the headline per-metric results are whole-task, not per-sub-task.
   The registry, report and every claim label state the family (§4.4 p. 10 "Label every claim with its metric
   and time cap τ").

**Why −60 for the unfinished attempt (F-P3), and how it squares with §3.2's censoring rule.**
- §3.2 p. 6's rule ("recorded as censored, not failed") governs how F is *estimated*. S1 and S4 keep it for
  every CDF.
- The return is an attribution target. §3.4 p. 7 already lets returns carry penalties ("a return based on
  negative time-to-success with a penalty for ghost events").
- `rmst60_op_cap` is therefore a declared, pessimistic functional: it equals the KM operation RMST only when no
  operation is censored before 60 s. Its `_cap` name and registry entry say so.
- A KM-consistent imputation (E[min(T, 60) | T > c]) would keep the identity with the descriptive KM. But it can
  still rank a censored attempt above a slow completed one, which F-P3 rules out (inferred).

---

## Request

The owner asked to (1) fix the answer-key leak as its own story inside this feature, and (2) build the
evaluation feature. That feature must include per-sub-task time-to-success for libero_10 as its own story, cover
"a maximal scope of what's given in the document as the evaluation feature", and make "per metric attribution and
phail usage" the crucial emphasis ([brief.md](brief.md), verbatim request plus eight binding decisions).

This plan turns PoC §3 (PhAIL primitives) into an evaluation layer for LIBERO (§3.5 A). It validates that layer
on PhAIL's own released real-robot data (§3.5 B, and the p. 8 caveat). It also shapes every output so the
later attribution feature gets **one return per metric per rollout**, keyed by the lineage contract (§3.4, §4.3,
§9.1 Step 3).

There is no Linux/GPU server yet. Planning measured that the LIBERO simulator **does run on this Mac**
(predicate replay and offscreen rendering, details below). So far more is locally verifiable than the brief
assumed: the human reference, the per-sub-task T, the rollout harness and small real rollouts. Only
attribution-grade SmolVLA training and the full 1000-rollout evaluations stay **pending server**.

---

## Spec grounding

### PoC sections and the rules they impose (all quoted from `docs/VLA_Data_Attribution_PoC.pdf`, read in full this session)

- **§2.1 item 3 (p. 3)**, "Make evaluation statistically honest":
  - "Use enough rollouts per task (roughly 50 or more), fixed seeds and initial states, and report confidence
    intervals. Otherwise you'll 'attribute' noise."
  - "Also log per-rollout outcomes with a simple failure taxonomy (missed grasp, wrong object, ignored
    instruction, timeout)."
- **§2.1 item 4 (p. 3):** plant ground truth, then "check whether your attribution method flags exactly those".
  This is the reason the methods under test must be blind (S0).
- **§2.1, "Connection to the evaluation layer" (p. 4):** "Section 3 upgrades 'success rate with confidence
  intervals' to PhAIL's distributional methodology."
- **§2.3 (p. 4):** Controls are "Fixed config and steps; 2–3 seeds per data configuration". SmolVLA "is also
  one of PhAIL's baselined models".
- **§2.4 (p. 4):** "The primary signal comes from closed-loop evaluation rollouts, scored with the PhAIL
  primitives in Section 3 and broken down by task or object and by failure mode."
- **§2.6 (p. 5):** counterfactual comparisons must "use the PhAIL significance gate (Section 3.4) to decide whether
  the change is real before claiming it".
- **§3.1 (p. 6):** PhAIL publishes "~990 evaluation episodes including 396 same-fixture human-teleoperation
  reference rollouts; every rollout published with synchronized video, telemetry and event annotations".
  "analysis pipeline published on GitHub".
- **§3.2 (p. 6), the primitives table.** Every row is in scope:
  - **Time-to-success T:** "Wall-clock time to complete one operation; each rollout yields several".
  - **Ghost events:** "(T = ∞) Unrecoverable outcomes ... count as completed at infinite time"; "The gap
    between the CDF's asymptote and 1 is exactly the hard-failure rate".
  - **Right-censoring:** "Timeouts are slow-but-recoverable, recorded as censored, not failed".
  - **Kaplan–Meier CDF:** "F(t) = P(T ≤ t) estimated with censoring"; "Success@τ, median, RMST, UPH,
    completion, MTBF/A are all projections of F".
  - **HRT:** "RMST_human / RMST_model per object, macro-averaged".
  - **Clustered bootstrap:** "95% CIs resampling whole episodes, not operations".
  - **Macro-KS:** "Two-sample Kolmogorov–Smirnov per object, averaged; p-values from pooled-resample bootstrap".
  - **P-P plots:** "F_human(t) vs F_model(t); area = P(T_model < T_human)".
  - **Rule:** "scoring (a scalar with a confidence interval, used for ranking and direction) and significance (the
    macro-KS test, used only to decide whether two policies differ at all). KS cannot rank on its own."
- **§3.3 (p. 7):** "Four principled scalars on the same data produced three different top-1 models";
  "SmolVLA was clearly worst"; "Equal hard-failure rates can hide different causes".
- **§3.4 (p. 7), the integration table:**
  - Verification gate: "Macro-KS is the significance gate; the HRT or RMST confidence interval gives the
    direction".
  - Richer targets: "ghost-event rate ..., RMST or HRT ..., and the censored tail".
  - Declared targets: "Every attribution result states which functional it explains".
  - Strata: "Per-object cells and macro-aggregation map directly to per-task attribution".
  - Finer rollout signal: "Operation-level (T, event) pairs give several outcome samples per rollout, e.g. a
    return based on negative time-to-success with a penalty for ghost events".
  - Failure taxonomy: "Ghost-event types ... and censored timeouts form an operational failure taxonomy".
- **§3.5 A (pp. 7–8), our adaptation to LIBERO:**
  - "T = time (or steps) to task success per episode; for long-horizon LIBERO-10 tasks, per sub-task."
  - "Ghost events = unrecoverable states a simulator can detect directly (object off the table, out of reach);
    timeouts are censored."
  - "Human reference = durations of LIBERO's original human-teleoperated demonstrations on the same tasks. This
    is approximate, since demos and rollouts don't share initial states."
  - "the distributional targets and failure semantics carry over unchanged."
- **§3.5 B and Practical caveats (p. 8):**
  - "The 449-episode fine-tuning set is small enough for retraining-based attribution."
  - "Use the human reference rollouts both as the HRT anchor and as the clean reference set."
  - "Confirm which parts of the dataset, including the fine-tuning set, are downloadable and under what
    license."
- **§4.1–4.4 (pp. 9–11):**
  - The four yardsticks are F(30 s), HRT (per-object macro mean), global RMST and AUC vs human.
  - The paper adds "as the per-rollout time budget grows, throughput-oriented UPH falls while
    reliability-oriented MTBF/A rises".
  - §4.3 defines the returns: "Success within 30 s → binary return; RMST → negative capped time; AUC → win
    indicator against a sampled human time. HRT is a ratio, so it needs a first-order (linearized)
    approximation."
  - §4.4: "Two-step verdict per intervention. First the macro-KS gate ... Then each metric's bootstrap CI gives
    the direction"; "Crossing curves mean trade-offs"; "correct for it (e.g. Bonferroni or Holm across
    metrics)"; "Label every claim with its metric and time cap τ".
- **p. 1:** "per-metric and per-failure-type returns producing an episode × metric matrix".
- **§9.1 Step 2 (p. 18):**
  - "Run many rollouts per task (e.g. 50) with fixed initial states."
  - "For each rollout, record whether and when it succeeded, using the benchmark's goal check."
  - "Classify failures with rules over simulator state: ghost events ..., censored timeouts, and failure types
    (wrong object, missed grasp, idle)."
  - "Compute the metrics: time-to-success curves, success within 30 s, RMST, HRT against human demo durations,
    and AUC against humans."
  - Output: "per task, per metric and per failure type".
- **§9.1 Step 3 (p. 19):** "Define one 'return' per metric for each rollout. For example, success within 30 s
  gives 1 or 0; for RMST, the return is negative time; for AUC, whether the rollout beat a sampled human time."
  "Repeat the analysis for subsets, such as a specific task or only rollouts with a specific failure type."
- **§9.1 Step 4 (p. 19):**
  - "Evaluate both policies as in Step 2."
  - "Apply the macro-KS test: did the time-to-success distribution change at all?"
  - "If it did, check each metric's confidence interval for the direction of the change."
- **§10 (p. 20):**
  - Evaluation layer: "Kaplan–Meier time-to-success CDFs with ghost-event and censoring rules; HRT against
    LIBERO demo durations; episode-clustered bootstrap CIs; macro-KS test; P-P plots".
  - Metric-resolved attribution: "Influence vector per episode across F(30 s), HRT, RMST and AUC".
  - Foundation: "LIBERO (one or two suites) + SmolVLA".
  - Open decision: "Which metric is the declared headline objective for curation".

### Papers (defining formulations)

- **PhAIL** (`docs/papers/PhAIL_Real-Robot_VLA_Benchmark_arXiv-2605.29710.pdf`, read in full):
  - **§3.1 p. 4: the T-pair extraction.**
    - "A successful placement contributes Ti equal to the elapsed time since the previous successful placement
      (or since episode start, for the first), with Ei = 1."
    - Unfinished episodes add "one right-censored pair (Ttail, 0) where Ttail = duration − last placement time".
    - Ghosts are "(T=∞, E=1)". "Timeouts are explicitly not ghost events".
    - "Episodes with zero successes contribute a single right-censored observation of length τepisode ... we
      never drop them".
    - "for the integrated-metric calculation ... we set Tghost → τ".
  - **Table 1 p. 5: the scalars as functionals of F.**
    - F(τ); median inf{t: F(t) ≥ 0.5}; RMST(τ) = ∫₀^τ (1 − F).
    - UPH = 3600/E[T | T < τepisode] ∝ 1/RMST(τepisode).
    - Completion = F(τepisode); MTBF/A = RMST(τepisode)/(1 − F(τepisode)).
  - **§3.2 p. 5:**
    - HRT(m,o) = RMST_Human,o(τ)/RMST_m,o(τ), τ = 240 s, "macro-aggregated across objects with equal weights".
    - P-P "area under the curve equals P(Tmodel < THuman)".
    - CIs: "We resample whole episodes".
  - **§3.3 p. 6:**
    - D̄ = (1/J) Σ_o D_o, with p-values from a "pooled-resample episode-clustered bootstrap".
    - "Direction ... comes from a scalar"; logrank with Bonferroni "as a sanity-check".
  - **§5 pp. 7–10:** the numbers this plan reproduces in S2:
    - Table 2: RMST / HRT with CIs and per-model episode counts.
    - F(30 s) leaders 0.51 / 0.47 / 0.44.
    - Macro-AUC: ACT 0.134 [0.108, 0.162], OpenPI 0.100, GR00T 0.095, SmolVLA 0.027.
    - The closest pair: "per-object macro KS (p = 0.12)".
  - **App. F p. 15:** drops are a per-operation ghost; safety stops are a per-episode ghost; "Timeouts ...
    censored, not ghosts".
- **CUPID** (`docs/papers/CUPID_Influence_Functions_arXiv-2506.19121.pdf`):
  - **§4 p. 4:** binary returns: "R(τ)=1 and R(τ)=−1".
  - **§5.1 p. 5, Prop. 1 / Alg. 1:** Ψ̂(ξ) = (1/m) Σᵢ R(τᵢ)/H Σ_{(s′,a′)∈τᵢ} Σ_{(s,a)∈ξ} Ψ_a-inf((s′,a′),(s,a))
    (quoted).
  - **What this plan takes:** attribution needs, per rollout, **a scalar return and every state-action pair
    (s′, a′) the rollout visited**. So rollout trajectories must be stored, not only outcomes (S5, S6).
- **LIBERO** (`docs/papers/LIBERO_Benchmark_arXiv-2306.03310.pdf`) plus its released code (MIT, hf-libero 0.1.4
  and GitHub `8f1084e`):
  - Success is the conjunction of BDDL goal predicates (code quoted under Prior art). The libero_10 goals are
    conjunctions of 1–3 predicates (measured, S4).
  - The paper is cited for task suites and goal checks. The predicate semantics come from the source, which is
    the authority over the prose.

### Units of T (owner decision 5: the planner decides)

**T is measured in seconds of simulated time, T_s = T_steps / 20, and both are stored.**

- **The spec states its targets in seconds.** The PoC says "F(30 s)", "within thirty seconds" (§4.1); PhAIL
  says "wall-clock time" (§3.2) (quoted).
- **The data is 20 Hz.** `info.json` fps 20, LeRobot `fps: int = 20  # Must match robosuite's default
  control_freq (20 Hz)` (quoted, data+lineage plan). The human reference is already in seconds.
- **Divergence, recorded.** Sim time excludes policy inference latency, which PhAIL's real wall-clock includes
  (inferred from the sim stepping only when an action is applied). LIBERO T therefore measures behaviour, not
  compute speed. The report must say so.

### Episode time budget τ_episode (declared, inferred, non-blocking: owner may override)

**τ_episode = 60 s (1200 steps) for both suites. The RMST cap is τ = τ_episode = 60 s.**

- **Why not LeRobot's caps (measured, quoted source).** LeRobot's defaults are 280 steps (14 s, libero_spatial)
  and 520 steps (26 s, libero_10) (`lerobot/envs/libero.py TASK_SUITE_MAX_STEPS`). With those caps "F(30 s)"
  cannot be measured at all, and on libero_10 F(τ_episode) would collapse onto completion.
- **Why 60 s (inferred):**
  - PhAIL's cap is "30 s/item ... ∼10× the ∼2.7 s/item human pace" (quoted).
  - 60 s is about 10× the libero_spatial human median of 6.2 s, and about 4.6× the libero_10 median of 13.15 s
    (measured catalog).
  - That keeps F(30 s) an interior point of the CDF, and leaves room for slow-but-recoverable runs in the
    censored tail (§3.4).
- **Comparability with LeRobot's binary success** is kept for free. F(14 s) and F(26 s) are projections of the
  same CDF (Table 1 logic).
- **Cost (inferred):** about 2–4× more sim steps on failed rollouts only.

---

## Premises verified against the artifacts (planning session, 2026-10-01/02)

| # | Claim | Evidence | Label |
|---|---|---|---|
| P1 | v1's episode table has `parent_content_hash`; it differs from `content_hash` for exactly 100 rows, and that set equals the labelled set | pandas over `data/lineage/versions/16076b9c…/episodes.parquet` + `labels.parquet`: `changed 100 == labelled True` | measured |
| P2 | **A second leak:** the 100 corrupted episodes are exactly `episode_index` 900–999 in v1 (contiguous tail), so row position identifies them without any column | same script: `labelled episode_index min/max 900 999 contiguous tail? True`; `derive.py` docstring: "edited episodes move to the end of the derived dataset" | measured + quoted |
| P3 | `dataset_version_id` is order-independent: H(schema, sorted [episode_id, content_hash]); `content_hash` covers task, action, state and images only, not `episode_index` | `vlapoc/lineage.py` docstring + `version_id` (l. 213); `vlapoc/catalog.py:127-155` | quoted (source) |
| P4 | Data-derived columns that legitimately differ v0→v1: `smoothness`, `gripper_toggles` (100 rows each, = labelled set), `idle_fraction`/`idle_steps` (9 rows) | v0/v1 row comparison | measured |
| P5 | `duration_s == length_steps / 20` for 1000/1000; per-task human medians 4.75–7.43 s (spatial), 9.33–20.43 s (libero_10); suite medians 6.2 s / 13.15 s; max 9.85 s / 25.85 s | catalog parquet | measured |
| P6 | **The HDF5 reward is not a goal check.** LIBERO's `scripts/create_dataset.py` (which wrote the released files) sets `rewards = np.zeros(len(actions)); rewards[-1] = 1` and the same for `dones`. Stored `states` are the original collection states with the first 5 steps dropped (`cap_index = 5`) | LIBERO GitHub `8f1084e` `scripts/create_dataset.py` | quoted (source) |
| P7 | Human collection stops only after the goal has held for 10 consecutive steps: `task_completion_hold_count = 10 # reset count on first success timestep` | LIBERO `scripts/collect_demonstration.py` | quoted (source) |
| P8 | **Replay check:** with each demo's own `model_file` XML, the goal predicate holds at some stored state in **1000/1000** demos (now verified by the goal check, not the hard-coded reward). The last stored state comes **after** the first success by min 8–10, median 10–18 (per task), max 81 steps. So `duration_s` overstates human time-to-success by about 0.5–0.9 s at the median | probe `probe_replay_xml.py` (scratch venv, see P10) over all 20 tasks × 50 demos | measured |
| P9 | Building the env from the BDDL (not the demo XML) fails to reproduce success for at least one demo (libero_10 task 8 `demo_1`: `on moka_pot_2 ...` never true); with the demo XML it succeeds at stored state 428 | probe `probe_predicates.py` vs `probe_replay_xml.py` | measured |
| P10 | **LIBERO runs on this Mac** (macOS 26.5.1, M4 Pro) in a uv Python 3.12 venv: `hf-libero 0.1.4` installed `--no-deps` + robosuite 1.4.0, bddl 1.0.1, mujoco 3.8.1, numba 0.68. Predicate replay works without a renderer; offscreen rendering 2×128² works with `MUJOCO_GL=cgl` (26.1 steps/s) and `glfw` (55.0 steps/s), single process; **50 fixed init states for every one of the 20 tasks**; with `LIBERO_CONFIG_PATH` set, nothing is written to `~/.libero` | scratch probes `probe_predicates.py`, `probe_render.py`, init-state count over both suites | measured |
| P11 | The plain `hf-libero` install fails on macOS: `robomimic==0.2.0` → `egl-probe` → "CMake must be installed" (EGL build). lerobot gates the extra: `hf-libero<0.2.0,>=0.1.4; sys_platform == "linux" and extra == "libero"`. LeRobot docs: "LIBERO requires Linux (`sys_platform == 'linux'`)" | uv install log; lerobot METADATA; docs page | measured + quoted |
| P12 | Importing LIBERO writes `~/.libero/config.yaml` unless `LIBERO_CONFIG_PATH` is set (`libero/__init__.py:5-6`). The planning probe created it; it was removed after the probe. Assets are not in the wheel: `get_assets_path` "Downloading from HuggingFace Hub" | probe + source | measured + quoted |
| P13 | LIBERO success = conjunction over `parsed_problem["goal_state"]` (`_check_success`); registered predicates are `true, false, in, on, up, printjointstate, open, close, turnon, turnoff`; **no failure, off-table or out-of-reach rule** anywhere in `envs/` | LIBERO source `predicates/__init__.py`, `problems/*_manipulation.py:136-161`, grep | quoted + measured (grep) |
| P14 | libero_10 goals: tasks 0, 1, 2, 3, 4, 6, 7, 9 have 2 conjuncts, task 5 has 1, task 8 has 3. Task 8's `(Turnon flat_stove_1)` is also in `:init` (true at t0 in all 50 demos); no other conjunct is true at t0 in any demo. libero_spatial: all 10 tasks `(On akita_black_bowl_1 plate_1)` | BDDL files + replay | measured |
| P15 | `lerobot-eval`'s `eval_policy` keeps only `sum_reward, max_reward, success, seed` per episode (no success step). `rollout()` returns a per-step `success` tensor and calls `render_callback(env)` after every step. `--eval.recording` writes `use_videos=True` (lossy) | `lerobot/scripts/lerobot_eval.py:167-412, 653-668` | quoted (source) |
| P16 | LeRobot `LiberoEnv`: config default render 360² (class default 256²); `num_steps_wait=10` no-op settle steps on reset; terminates on `check_success()`; `init_state_id = episode_index`, `+= n_envs` per reset | `lerobot/envs/libero.py`, `envs/configs.py:333-334` | quoted (source) |
| P17 | LeRobot docs, paired comparison: "use the same `--seed`, keep `--env.init_states=true`, and run each task in a single batch"; recommended "10 episodes per task ... averaging over 3 seeds" | LeRobot LIBERO docs page | quoted |
| P18 | PhAIL `build/stats.py` (Apache-2.0, repo `18ce72d`, 286 lines, numpy only): `episode_to_T_pairs`, `kaplan_meier` (handles `inf`), `km_step`, `km_median`, `rmst` (ghosts integrate to τ), `ks_distance`, `episode_cluster_bootstrap_ks`, `per_obj_macro_bootstrap_ks`, `pct_of_human` | clone + read | quoted (source) |
| P19 | PhAIL's script producing `out/paper/tab_headline_ci.json` (Table 2 HRT CIs) is **not in the repo**; the JSON output is | grep for `headline_ci`/`hrt_ci` finds no producer | measured |
| P20 | lifelines raises `TypeError("Infs were detected in the dataset...")` on infinite durations; scipy 1.18.1 `ecdf(CensoredData(...))` raises "`uncensored` must contain only finite values"; scipy has `logrank`; `ks_2samp` has no censoring | lifelines source `utils/__init__.py:1171-1192`; scipy probe | quoted + measured |
| P21 | **PhAIL data access (p. 8 caveat answered).** Public, anonymous HTTPS (Nebius S3); license CC BY 4.0. Fine-tuning set `datasets/phail/v1.0/training/` = **449 episodes, 13.95 GB** (= `phail/v1.0/dataset/teleoperation/`). `inference/` 524, `rollouts/` 594, `human/` 40 episodes. `croissant.json` says "training subset (352 episodes)" (discrepancy with 449). `s3://inference/phail_act_groot/` (142 cohort episodes per `out/data_audit/episodes.csv`) returns HTTP 403 | S3 ListObjectsV2 count script; `dataset/croissant.json` `"license": "https://creativecommons.org/licenses/by/4.0/"` | measured + quoted |
| P22 | PhAIL's Table 2 human anchor "Teleop (396)" is the teleoperated fine-tuning cohort (`REFERENCE_MODEL = 'Robot teleoperated by Human'`, `episodes.csv` source `training`), the direct analogue of the PoC's "LIBERO's original demos" as reference | `build/tab_leaderboard.py`, `out/data_audit/episodes.csv` | quoted + measured |
| P23 | LeRobot's own `aggregate_stats` over the per-episode stats stored in `meta/episodes` reproduces `meta/stats.json` for `observation.state` and `action` (max abs diff ≤ 2.4e-10). `lerobot_train.py` feeds `dataset.meta.stats` to the processors (l. 314, 351, 376) | probe in `.venv` | measured + quoted |
| P24 | `HuggingFaceVLA/smolvla_libero` exists: Apache-2.0, rev `6721902b`, ~605M params, inputs 2×256² + state 8, `chunk_size` 50, `n_action_steps` 1, STATE/ACTION `MEAN_STD`. Card says `datasets: unknown` | HF API + `config.json` | measured + quoted |
| P25 | CUPID's released code (MIT) aggregates influences with a **binary** per-rollout `success_mask` (`eval_demonstration_scores.py:489-586`, metrics `succ/fail/net`) | clone + read | quoted (source) |
| P26 | Disk 117 GiB free; uv 0.11.15 present; only Python 3.14 system-wide (uv can provision 3.11/3.12) | `df`, `which` | measured |

**Discrepancies recorded** (the artifact wins):

- **(a) CLAUDE.md / vla-build reward premise.** They say every demo ends with "reward == 1: the goal predicate
  verified true". The reward is hard-coded by `create_dataset.py` (P6), so it is not a goal check. The
  conclusion "no failed demos" still holds, but its evidence is P8, not the reward. For the owner; the planner
  does not edit CLAUDE.md.
- **(b) The catalog's `duration_s` is not human time-to-success.** It includes the 10-step success hold (P7, P8).
  The data+lineage report and the index coverage row say otherwise.
- **(c) LeRobot docs say LIBERO requires Linux.** Measured: the simulator, predicates and rendering run on macOS
  (P10, P11).
- **(d) Fine-tuning set size.** PhAIL `croissant.json` says 352; the bucket and the paper say 449 (P21).

---

## Prior art

| Candidate | What it covers | What it misses | Evidence |
|---|---|---|---|
| **PhAIL `build/stats.py`** (Apache-2.0, `18ce72d`) | The defining implementation of §3.1–3.3: T-pair extraction with censored tail and T=∞ ghosts, KM (Greenwood var), median, RMST with ghosts→τ, KS distance, episode-clustered pooled-resample bootstrap, per-object macro-KS | No HRT-with-CI function (the producer of Table 2 CIs is absent, P19), no censored AUC, UPH, MTBF/A or P-P (these live in scripts), no LIBERO adapter, no per-rollout returns | P18 (quoted) |
| **PhAIL scripts** (`explore_pp_plot._auc_censored`, `_pp_curve`, `_macro_auc`; `fig_qq_tradeoff` UPH = 3600/RMST, MTBF/A = RMST/(1−F); `tab_leaderboard` F(20/30/60 s); `explore_separation._rmst_bootstrap`, `_logrank`) | Reference formulas for every remaining scalar, plus sanity tests | Script-level code with `configuronic`/`pos3`/loader coupling. **UPH has three formulas**: Table 1 `3600/E[T∣T<τ]`, `fig_qq_tradeoff` `3600/RMST`, `common.py` per-episode `3600·S/T` (measured read) | quoted (source) |
| **PhAIL released dataset** (CC BY 4.0, public bucket) | Real-robot (T, E) data with published numbers to reproduce. **The validation set for our metric layer**, and the §3.5 B real-robot stage data | Part of the published cohort may sit in a private bucket (P21: 403) | measured |
| **positronic** (Apache-2.0, PyPI 0.2.1, `Requires-Python <3.12,>=3.11`) | Real-robot runtime for the PhAIL station / replica (§3.5 B) | Not needed for simulation evaluation; Python 3.11 only | measured (PyPI, GitHub API) |
| **lifelines** | KM, RMST (`restricted_mean_survival_time`), CIs | **Rejects infinite durations** (P20), so ghost events cannot enter; no macro-KS bootstrap | quoted (source) |
| **scipy 1.18.1** (installed) | `stats.ecdf` with `CensoredData` (KM + Greenwood CI), `stats.logrank`, `ks_2samp` | Rejects `inf`; `ks_2samp` has no censoring; no clustered bootstrap. **Used as an independent cross-check** of KM on finite data and for the logrank sanity check | measured |
| **LeRobot 0.6.1** `lerobot-eval` / `rollout()` / `LiberoEnv` / `LiberoProcessorStep` | The exact policy I/O and env stepping SmolVLA is evaluated with; per-step `success`; per-step `render_callback` hook; fixed init states; seeds | Drops the success step (P15); no simulator-state probing; recording only as lossy video; init-state assignment depends on batching (P16) | quoted (source) |
| **LIBERO / hf-libero** (MIT) | BDDL goal predicates, `_eval_predicate` per conjunct, `_check_success`, fixed init states, demo replay recipe (`create_dataset.py`: `reset_from_xml_string(postprocess_model_xml(...))`) | **No failure or ghost rules** (P13); its rewards are synthetic (P6); a plain install is broken on macOS (P11) | quoted + measured |
| **CUPID code** (MIT) | The influence estimator the attribution feature will fork; stores rollouts plus per-rollout success | Binary `success_mask` only (P25). Needs a real-valued return per metric, which is what S6 emits | quoted (source) |
| **MLflow 3.16.1** (existing lineage backend) | Run records, inputs and outputs; already the lineage contract | Nothing new needed | existing feature |
| **statsmodels** `multipletests(method="holm")` | Holm correction across metrics (§4.4) | Not installed (adds a dependency) | to verify during execution (not checked) |

**Verdict: adapter over the PhAIL reference implementation (vendored verbatim) plus LeRobot's eval loop and
LIBERO's own predicates. Custom code only where nothing exists:**

1. **The LIBERO → (T, E) adapter**, including the per-sub-task operation rule (S4) and the replay-based human
   reference (S3).
2. **HRT macro with an episode-clustered bootstrap CI.** PhAIL's producer is not released (P19).
3. **The per-rollout / per-operation return table and metric registry (S6).** No tool has one: CUPID is
   binary-only (P25).
4. **The two-step verdict wrapper (S7)** around PhAIL's macro-KS plus per-metric CIs.
5. **The per-subset normalization hook (S8)**, using LeRobot's own `aggregate_stats`.
6. **Lineage glue** (eval stage runs, keys).

**Decisions:**

- **Vendor `build/stats.py` unmodified** into `vlapoc/third_party/phail_stats.py`, with its LICENSE and an
  attribution header.
  - Installing `phail-paper` as a package is not viable: its package name is `build`, and it pulls `positronic`
    from git and `pos3` at import (quoted pyproject / `common.py`).
  - GitHub's API reports `NOASSERTION` for the repo license. The LICENSE file is the Apache-2.0 text with
    "Copyright 2025-2026 Positronic Robotics" (measured).
- **Do not build on lifelines.** It cannot represent T=∞ (P20). scipy is the cross-check, not the
  implementation.
- **Reuse LeRobot's `rollout()` preprocessing path; don't write a second policy-I/O pipeline.** Add only the
  per-step probe and explicit init-state assignment.

---

## Stories

**Layout (proposed; the executor may refine names):**
- **Code:** `vlapoc/third_party/phail_stats.py` (vendored), `vlapoc/evaluation.py` (S1 adapter and metrics),
  `vlapoc/libero_goals.py` (S3–S4: BDDL ops, replay, operation rule), `vlapoc/rollouts.py` (S5 harness),
  `vlapoc/returns.py` (S6), `vlapoc/verdict.py` (S7), and a `vlapoc/subset_stats.py` hook used by
  `vlapoc/lerobot_train_entry.py` (S8).
- **Scripts:** `scripts/phail_validate.py` (S2), `scripts/human_reference.py` (S3–S4),
  `scripts/eval_smoke.py` (S5, S9).
- **Data (gitignored):** `data/external/phail-paper/` (clone at `18ce72d`), `data/phail/` (metadata only, about
  180 MB), `data/libero_assets/`, `data/libero_config/`, `data/eval/<eval_run>/`.
- **Evidence:** `tracker/features/2026-10-01-evaluation-layer/artifacts/`.

**Environment rule for every story that touches the simulator.**
- Install the sim stack (`hf-libero==0.1.4 --no-deps`, `robosuite==1.4.0`, `bddl==1.0.1`, `mujoco` 3.x,
  `termcolor`, `easydict`, `hydra-core`, `numba`) into the main `.venv` if it resolves there. The pieces have
  cp314 macOS wheels: mujoco 3.8.1, numba 0.68 (measured, PyPI). Otherwise create a uv-managed Python 3.12
  `.venv-sim` with `lerobot` + the sim stack; this path is proven by P10.
- Always set `LIBERO_CONFIG_PATH=$PWD/data/libero_config` and `MUJOCO_GL=cgl` or `glfw`, and pin the assets.
  Never write `~/.libero` (P12).
- Record the chosen environment, package versions, LIBERO commit/version and an assets digest in every run.

---

### S0: Fix the answer-key leak (method-facing tables must not reveal edited episodes)

- **Why:**
  - Owner request "fix the leak now as a separate story".
  - PoC §2.1 item 4 p. 3: methods must "flag exactly those" without seeing the answer.
  - P1 (column leak) and **P2 (positional leak, new)**.
- **Depends on:** —
- **Deliverables:**
  - **Lineage-only storage of pre-edit fingerprints.** For example `versions/<id>/parents.parquet`, or rely on
    `labels.parquet`, which already holds `parent_content_hash` for the edited rows. The executor chooses.
    Accessor `parent_hashes(version)` is documented as lineage-only.
  - **`derive_version` preserves parent order.** Every episode keeps its parent's `episode_index`; edited
    episodes are rewritten in place, not appended (mechanism: executor's choice, e.g. one writer pass in parent
    order).
  - **`check_blind(version_ref)` guard.**
    - Compares the derived version's method-facing tables with its parent: `version_episodes()`, the catalog
      copy `data/lineage/catalog/<dataset>/episodes.parquet`, and the LeRobot `meta/episodes` order.
    - Rule 1: column sets are identical.
    - Rule 2: every **bookkeeping** column (explicit list: `episode_index`, `source_file`, `source_demo`,
      `init_state_sha256`, `suite`, `task_name`, `libero_task_id`, `language_instruction`, `scene`, `objects`,
      `source`, `operator`, `collection_date`) equals the parent's value per `episode_id`.
    - **Data-derived** columns are exempt by an explicit list: `content_hash`, `length_steps`, `duration_s`,
      `smoothness`, `idle_fraction`, `idle_steps`, `gripper_toggles`.
    - It returns the offending columns.
  - Update `scripts/verify_derive_s4.py` AC4.7 (it currently asserts the leaky column set), and the docstrings
    of `version_episodes()` ("it never holds ground truth") and `derive.py` ("edited episodes move to the end").
  - **v1 (default, see BLOCKING Q1):**
    - Apply the column fix to v1's tables. The version id and `labels.parquet` are unchanged.
    - Keep v1's physical copy. `check_blind(v1)` then fails on `episode_index` only, and v1 is fenced from
      method use, recorded in lineage-only storage and in the report.
    - The injection feature derives its real corrupted versions with the fixed `derive_version`.
- **Acceptance criteria:**
  - [ ] AC0.1 After the fix, v1's `version_episodes()` and catalog copy have exactly v0's 21 columns (no `parent_content_hash`). `version show v1` prints the same `dataset_version_id` (`16076b9c…`) [brief: keep v1's version id]
  - [ ] AC0.2 The label join still works: `labels.parquet` joins v1's table on (`episode_id`, `content_hash`) for 100/100 rows, and `parent_hashes(v1)` reproduces v0's `content_hash` for all 1000 episode_ids [brief; PoC §2.1 item 4 p. 3]
  - [ ] AC0.3 Order preservation, measured on a scratch derivation (temporary lineage root, ≤ 20-episode parent, ≥ 2 edits incl. one length-changing edit): every episode's `episode_index` equals its parent's, and `check_blind` returns no offending column [P2; PoC §2.1 item 4 p. 3]
  - [ ] AC0.4 `check_blind(v1)` reports exactly `["episode_index"]` (positional) under default A, or nothing if the owner picks B. The result is written to `artifacts/` [P2]
  - [ ] AC0.5 Negative test: a scratch derived table with an extra bookkeeping column, or a permuted `episode_index`, is rejected by `check_blind` [vla-build: prefer a runnable test]
  - [ ] AC0.6 A `changelog.md` entry in `tracker/features/2026-09-28-data-lineage-layer/` records the two leak channels, the fix, the v1 disposition, and that its AC4.1/AC4.7 required the leaky column. The report's "Open decisions and known issues" gets a dated note [vla-build step 5; brief]
  - [ ] AC0.7 Grounding docstrings cite PoC §2.1 item 4 p. 3 [vla-build]
- **Verify by:**
  - `.venv/bin/python -c "from vlapoc.lineage import ...; print(check_blind('v1'))"`.
  - The scratch-derivation test script.
  - `scripts/verify_derive_s4.py` (updated).
- **Demo value:** a two-row table, "v1 before: 2 leak channels (column, row order) → after: 0 columns, row order
  fenced/fixed", plus the guard that blocks future regressions.

### S1: Metric layer: PhAIL primitives over (T, E) pairs, with LIBERO-ready cells

- **Why:**
  - PoC §3.2 p. 6 (every primitive) and §10 p. 20, Evaluation layer row.
  - Owner decision 4: all metrics.
- **Depends on:** —
- **Deliverables:**
  - **`vlapoc/third_party/phail_stats.py`:** verbatim `build/stats.py` + LICENSE + header naming the repo, the
    commit `18ce72d` and "unmodified".
  - **`vlapoc/evaluation.py`:**
    - (i) A cell abstraction: (policy/arm, cell = task or object) → list of episodes, each with its
      operation-level (T, E) pairs. LIBERO episodes carry `place_ts` = operation completion times,
      `duration_s` = the rollout's observed duration, `total_items` = n_ops and `n_inf_events` = 0.
      **(r2)** A LIBERO cell is built in either declared family: W (one whole-task pair per rollout, the
      headline) or O (operation pairs). The family comes from the single registry (AC6.8).
    - (ii) Scalars per cell and macro: F(τ) for any τ, median, RMST(τ), UPH (= 3600/RMST(τ_episode), following
      PhAIL's code, with Table 1's form noted), completion F(τ_episode), MTBF/A, ghost-event rate,
      censored-tail rate.
    - (iii) HRT per cell and macro (equal weights), human anchor passed in.
    - (iv) Censored AUC vs reference (PhAIL `_auc_censored` semantics) and macro-AUC.
    - (v) P-P curves.
    - (vi) Episode-clustered bootstrap CIs for every scalar, resampling episodes within each cell. For HRT,
      both the policy and the human anchor are resampled, as a documented option. P19 means the PhAIL variant
      must be determined against the published CIs in S2.
    - (vii) Macro-KS with pooled-resample p (vendored).
    - (viii) Stratified pooled logrank via scipy as the §3.3 sanity check.
    - (ix) Figures: KM CDF with CI bands, P-P plot, UPH vs MTBF/A sweep over τ ≤ τ_episode (App. L).
  - **Ghost events are fully supported** (T = ∞ through the vendored KM). LIBERO simply supplies none (see S5,
    owner decision 6).
- **Acceptance criteria:**
  - [ ] AC1.1 The vendored file is byte-identical to `build/stats.py` at `18ce72d` (sha256 recorded) and its license header is present [Apache-2.0 §4; vla-build prior art]
  - [ ] AC1.2 Independent cross-check on finite, censored data (synthetic, labelled synthetic): KM survival agrees with `scipy.stats.ecdf(CensoredData)` at every event time to ≤ 1e-12 [PoC §3.2 p. 6 "estimated with censoring"; vla-build "do not hand-roll"]
  - [ ] AC1.3 Ghost semantics (synthetic): with g of n operations at T=∞ and no censoring, `1 − F(t→∞)` equals g/n exactly, and RMST(τ) equals the mean of `min(T, τ)` with ghosts set to τ [PoC §3.2 p. 6; PhAIL §3.1 p. 4]
  - [ ] AC1.4 Censoring semantics (synthetic): an episode with zero successes contributes one censored pair of length τ_episode, and is never dropped. An episode that completes all its ops contributes no censored tail [PhAIL §3.1 p. 4; `episode_to_T_pairs`]
  - [ ] AC1.5 Clustered vs per-operation bootstrap (synthetic, ≥ 3 correlated ops per episode): the clustered 95% CI is wider than the per-operation CI. The ratio is reported [PoC §3.2 p. 6 "per-operation resampling gives CIs that are too narrow"]
  - [ ] AC1.6 Every §3.2 primitive and Table 1 scalar has a named function with a docstring giving its functional of F and its τ. The registry lists F(τ), median, RMST(τ), UPH, completion, MTBF/A, HRT, AUC, KS / macro-KS, P-P, ghost-event rate, censored rate [PoC §3.2 p. 6; owner decision 4]
  - [ ] AC1.7 Grounding docstrings cite PoC §3.2 p. 6 and PhAIL §3.1–3.3 pp. 4–6 [vla-build]
- **Verify by:**
  - `.venv/bin/python -m pytest` (or an equivalent script) over the synthetic suite.
  - `sha256sum` of the vendored file vs the clone.
- **Demo value:** the metric registry table (name → functional of F → τ → status), the foundation every later
  number cites.

### S2: Validate the metric layer on PhAIL's released real-robot data, and answer the §3.5 B data question

- **Why:**
  - Owner emphasis "phail usage is crucial".
  - PoC §3.1/§3.3 pp. 6–7 (published evidence), §3.5 B and Practical caveats p. 8, §4.1 p. 9 ("four yardsticks").
  - vla-build: verify against the artifact.
- **Depends on:** S1
- **Deliverables:**
  - **`scripts/phail_validate.py`**, three steps:
    - (a) Fetch the metadata sidecars only (`static.json`, `meta.json`; exclude `*.parquet`, `*.mp4`; about
      180 MB per the PhAIL README, quoted) into `data/phail/`, plus the repo's 461 annotation JSONs and
      `out/release_classifier/candidates.jsonl` (51.5 MB, in repo).
    - (b) Build the "mixed" cohort with PhAIL's own loader logic, then export per-episode `place_ts`,
      `duration_s`, `total_items`, `n_inf_events`, `model`, `object` and `cohort_source` to a parquet. PhAIL's
      loader can be run in a throwaway uv env from the clone.
    - (c) Compute every S1 scalar on that parquet and compare.
  - **`artifacts/phail_validation.json`** and figures.
  - **A short "PhAIL data access" table:** what is downloadable, size, license, counts, discrepancies.
- **Acceptance criteria:**
  - [ ] AC2.1 Implementation equivalence on identical input: for every (model, object) cell and the macro, our F(20/30/60 s), median, RMST(240 s), KS distances and censored AUC equal PhAIL's own functions/scripts on the same exported cohort to ≤ 1e-9. Macro-KS p with the same RNG seed is identical [PhAIL §3 pp. 4–6; vla-build verify against source]
  - [ ] AC2.2 Cohort reconciliation: per-model episode counts of the exported cohort vs Table 2 (165/165/151/118, Teleop 396) are reported. If any source is unreachable (P21: `s3://inference/` → 403), the missing count is stated per model [PhAIL Table 2 p. 7; P21]
  - [ ] AC2.3 Published-number reproduction, reported side by side with the published values: Table 2 RMST and HRT (points + 95% CIs, n_boot = 1000, seed 0), F(30 s) for GR00T/OpenPI/ACT (0.51/0.47/0.44), macro-AUC (ACT 0.134, OpenPI 0.100, GR00T 0.095, SmolVLA 0.027), and OpenPI-vs-GR00T macro-KS p (0.12). **If AC2.2 shows a complete cohort:** points within ±0.1 s (RMST), ±0.1 pp (HRT), ±0.005 (F, AUC); CIs within ±10% of the published CI width; macro-KS p within ±0.03. **Inferred tolerances:** bootstrap RNG order differs and the HRT CI producer is unreleased (P19); 0.03 ≈ 3 Monte-Carlo SE at p = 0.12, n_boot = 1000. **If incomplete:** the deltas are reported and explained by the missing episodes, and AC2.1 carries the pass [PhAIL §5 pp. 7–10]
  - [ ] AC2.4 The "four yardsticks" table (F(30 s), HRT macro, global RMST, AUC vs human: value, CI, winner) is regenerated from the data. It reports whether "four principled scalars, three different winners" (GR00T / OpenPI / GR00T / ACT) is reproduced [PoC §3.3 p. 7, §4.1 p. 9]
  - [ ] AC2.5 HRT CI method: both bootstrap variants (policy-only resampling vs policy + human-anchor resampling) are compared with `tab_headline_ci.json`. The variant that matches is adopted as S1's default, with the evidence [P19; PhAIL §3.2 p. 5]
  - [ ] AC2.6 The p. 8 caveat is answered with measured facts:
    - Fine-tuning set 449 episodes / 13.95 GB at `datasets/phail/v1.0/training/`, CC BY 4.0.
    - The human reference used in Table 2 is the teleop fine-tuning cohort (P22).
    - The 352-vs-449 `croissant.json` discrepancy.
    - Which cohort sources are private.
    - The listing command's output is saved [PoC §3.5 B p. 8, §10 Open decisions p. 20]
  - [ ] AC2.7 Figures on real data:
    - One (model, object) KM CDF with clustered-bootstrap bands against the human reference.
    - Its P-P plot (cf. PhAIL Fig. 2).
    - The macro UPH-vs-MTBF/A sweep (cf. Fig. 10) [PoC §3.2 p. 6, §4.1 p. 9; PhAIL App. L]
  - [ ] AC2.8 The UPH formula discrepancy (P-table "Prior art": three forms) is recorded, with the one adopted [vla-build: record discrepancies]
  - [ ] AC2.9 **(r2, E3)** Deterministic cohort export.
    - **Why:** PhAIL's loader iterates a Python set (`for url in candidate_urls:`, quoted by the analyzer), so
      row order depends on hash randomization and every bootstrap p moves.
    - `scripts/phail_export_cohort.py` sorts rows by `s3_url` before writing.
    - Two exports in fresh processes with `PYTHONHASHSEED` 0 and 1 (and one unset) give byte-identical parquet
      files; their sha256 is recorded.
    - `s2_phail_reference.json`, `phail_validation.json` and the S7 artifacts are regenerated from the sorted
      export, and the p-values quoted in the log are updated.
    - The report states that bootstrap p-values are Monte-Carlo draws conditional on episode order and seed;
      D and the point estimates do not depend on order.
    - The report names pos3 (`pos3.download(..., profile=PUBLIC)`, the phail-paper README recipe) as the
      reproduction route for the fetch.
    - Sources: [analysis-r1 E3; PhAIL §3.3 p. 6 "pooled-resample ... bootstrap"; vla-build reproducibility]
- **Verify by:**
  - `.venv/bin/python scripts/phail_validate.py --out artifacts/phail_validation.json`.
  - The executor saves the cohort-count and comparison tables.
- **Demo value:** the headline PhAIL-usage result. On PhAIL's own real-robot data, our metric layer reproduces
  the paper's numbers and the "same data, opposite rankings" table. That table is the finding §4 builds the
  PoC's main contribution on.

### S3: LIBERO simulator stack locally, and the human reference by replaying the demos' goal check

- **Why:**
  - PoC §3.5 A p. 8 ("Human reference = durations of LIBERO's original human-teleoperated demonstrations"),
    §10 p. 20 ("HRT against LIBERO demo durations"), §9.1 Step 2.2 ("using the benchmark's goal check").
  - P6–P8 show `duration_s` is not time-to-success, so the anchor must come from the goal predicate.
- **Depends on:** — (S1 for the summary metrics)
- **Deliverables:**
  - The sim environment (Environment rule above).
  - `vlapoc/libero_goals.py`:
    - Parse each task's BDDL goal conjuncts through LIBERO's own parser (`parsed_problem["goal_state"]`).
    - Replay a demo: `reset_from_xml_string(postprocess_model_xml(model_file))`, plus remapping of the
      authors' absolute asset paths (`/home/…/assets/`, `/Users/…/assets/`) to the local assets. Then
      `set_state_from_flattened(states[t])` + `forward()` per stored state.
    - Per state, evaluate every conjunct with `_eval_predicate` and the full goal with `_check_success`.
  - `scripts/human_reference.py` → `data/eval/human_reference/human_reference.parquet`, one row per v0 episode:
    `episode_id`, `content_hash`, `T_success_steps` (first stored state where the goal holds), `T_success_s`,
    `duration_s`, the gap, and per-conjunct first-true steps.
  - **The HRT anchor is fixed:** all 1000 v0 originals, constant across every compared policy (inferred: a
    ratio's denominator must not move between arms).
- **Acceptance criteria:**
  - [ ] AC3.1 The sim stack installs in a project env without touching `~/.libero` (`ls ~/.libero` → absent). Versions and assets digest are recorded in `artifacts/sim_env.json` [P11, P12]
  - [ ] AC3.2 The goal is reached at some stored state for 1000/1000 demos under demo-XML replay. Any demo that fails is listed, not dropped [PoC §9.1 Step 2.2 p. 18; P8]
  - [ ] AC3.3 For every demo, `T_success_s < duration_s`. The per-task distribution of the gap (min/median/max steps) is reported and compared with the planning measurement (min 8–10, median 10–18, max 81) [P7, P8]
  - [ ] AC3.4 A human reference table per task and per suite: n, median, RMST(60 s), F(30 s) from the KM of `T_success_s` (all events, no censoring). HRT computed with `duration_s` instead is shown beside it, to quantify the bias the fix removes [PoC §3.5 A p. 8, §10 p. 20]
  - [ ] AC3.5 Alignment conventions are documented in the docstring and the report: human T counts from stored state 0 (original step 5, `cap_index = 5`); rollout T counts from the post-settle state (10 no-op steps, P16). The residual uncertainty (≤ 5 steps = 0.25 s, inferred) and "approximate ... don't share initial states" (PoC p. 8) are stated [PoC §3.5 A p. 8]
  - [ ] AC3.6 The BDDL-built env vs demo-XML discrepancy (P9) is reproduced and recorded. Rollouts run in the BDDL env (LeRobot's), the human reference uses the demo XML, and the report says so [vla-build: record discrepancies]
  - [ ] AC3.7 All 20 tasks expose exactly 50 fixed init states (LeRobot `get_task_init_states`) [PoC §2.1 item 3 p. 3, §9.1 Step 2.1 p. 18; P10 measured 50 × 20 in the scratch env]
  - [ ] AC3.8 A `changelog.md` entry in the data+lineage folder corrects the catalog/report wording ("duration_s = human time-to-success", "last step is the only reward" as evidence), with P6–P8 as evidence. The index coverage row for §3.5 A is updated [vla-build step 5]
- **Verify by:** `<sim-python> scripts/human_reference.py --suites libero_spatial,libero_10` (about 2–3 min
  measured for 1000 demos), then the summary script in `.venv`.
- **Demo value:**
  - The real human-reference CDFs for 20 LIBERO tasks.
  - A measured correction: the old anchor overstated human time by about 0.5–0.9 s at the median, about 8–15%
    on libero_spatial (inferred from P5/P8). That would have inflated every HRT.

### S4: Per-sub-task time-to-success T for libero_10 (required story)

- **Why:**
  - Owner decision 7 and the request.
  - PoC §3.5 A p. 7: "for long-horizon LIBERO-10 tasks, per sub-task".
  - §3.4 p. 7: "Operation-level (T, event) pairs give several outcome samples per rollout".
  - §3.2 p. 6: "each rollout yields several".
- **Depends on:** S3, S1
- **Deliverables:** one function, used by both the human replay (S3) and the rollout harness (S5),
  `operations_from_trace(conjunct_truth[t, c], success[t], τ_end) → place_ts, n_ops, censored tail`. The rule
  is declared in its docstring:
  - **Operations** = goal conjuncts that are false at t0 (the post-settle initial state). Conjuncts true at t0
    are **maintained conditions**, not operations; e.g. task 8 `(Turnon flat_stove_1)`, P14.
  - **Operation k < n completes** at the first step where ≥ k operation-conjuncts hold simultaneously
    (counting, not identity: conjunct order is free in the BDDL).
  - **Operation n completes** at the env's own success step (`_check_success`), so the last operation time
    equals whole-task T by construction.
  - **T_k = t_k − t_{k−1}** (t₀ = 0). An incomplete rollout adds the censored tail (τ_end − t_last, 0). This
    is exactly PhAIL §3.1 / `episode_to_T_pairs`, with `multi_grip_threshold_s = 0`: a divergence, because
    sim timestamps are exact and PhAIL's 0.3 s clustering exists for annotation jitter (quoted docstring).
  - The identity of the conjunct that completed each operation is kept for diagnostics.
  - libero_spatial tasks have one operation each, so their T is whole-task T.
- **Acceptance criteria:**
  - [ ] AC4.1 The operations table for all 20 tasks (task → goal conjuncts → maintained conjuncts → n_ops) is generated from the installed BDDL files and matches P14: libero_10 n_ops = [2, 2, 2, 2, 2, 1, 2, 2, 2, 2], task 8 having one maintained conjunct; libero_spatial n_ops = 1 [PoC §3.5 A p. 7; LIBERO BDDL]
  - [ ] AC4.2 Unit tests on labelled-synthetic traces cover: two conjuncts in either order; a conjunct that toggles off and on; a maintained conjunct broken mid-rollout (success delayed until restored); simultaneous completion (T = 0 for the second op); timeout after k < n ops (censored tail present); zero ops done (one censored pair of length τ_end) [PhAIL §3.1 p. 4]
  - [ ] AC4.3 On the 500 libero_10 human demos: Σ_k T_k equals whole-task `T_success_s` for 500/500, and every demo has n_ops events and no censored pair [consistency with S3]
  - [ ] AC4.4 Human per-operation CDFs per libero_10 task (KM, all events) and a per-op-index breakdown (first vs second operation) are produced. The clustered bootstrap is used, since operations within one demo are correlated. **(r2)** These PhAIL-semantics operation CDFs (`rmst60_op`, `hrt60_op`; unfinished attempts censored) are labelled *descriptive* in the registry. The family-O attribution functional is `rmst60_op_cap` (S6, "Declared functionals (r2)") [PoC §3.2 p. 6 clustered bootstrap, §3.5 A p. 7]
  - [ ] AC4.5 The harness (S5) imports this same function. There is one implementation, enforced by a grep in the verify script [vla-build; single record system]
  - [ ] AC4.6 Grounding docstring cites PoC §3.5 A p. 7, §3.4 p. 7, PhAIL §3.1 p. 4 and the LIBERO source lines for `_check_success` [vla-build]
- **Verify by:** `pytest` on the synthetic traces; `<sim-python> scripts/human_reference.py --suites libero_10
  --operations`.
- **Demo value:** per-operation human CDFs for the 10 long-horizon tasks, and the operation counts that turn
  500 libero_10 rollouts into about 950 operation samples (inferred: 19 ops × 50 per task).

### S5: Rollout harness, rollout record and the eval stage (replaces the stand-in)

- **Why:**
  - PoC §9.1 Step 2 p. 18: "Run many rollouts per task (e.g. 50) with fixed initial states"; "record whether
    and when it succeeded, using the benchmark's goal check".
  - §2.1 item 1 p. 3: "Every eval rollout should be tied back to that checkpoint".
  - §2.1 item 3 p. 3: fixed seeds and initial states.
  - CUPID Prop. 1 p. 5: attribution needs each rollout's (s′, a′).
- **Depends on:** S3, S4
- **Deliverables:**
  - **`vlapoc/rollouts.py`**, a thin adapter over LeRobot's `LiberoEnv` and `lerobot_eval.rollout()`
    preprocessing path (`LiberoProcessorStep` + policy pre/post processors, unchanged):
    - (i) Explicit init-state assignment: rollout j of a task uses init state j and seed `seed_base + j`,
      independent of batching (P16, P17).
    - (ii) A per-step probe, via `render_callback` or a `LiberoEnv` subclass method reached with
      `env.call(...)` (the executor chooses). It records goal-conjunct truth (S4 input) and the positions of the
      BDDL-named task objects. These are raw signals; no rule is applied (see Failure types).
    - (iii) τ_episode (default 1200 steps = 60 s), render size (128² for policies trained on our data, the
      checkpoint's own size otherwise) and `control_mode=relative`, all recorded.
  - **Rollout record** `data/eval/<run>/rollouts.parquet`, one row per rollout:
    - Keys and identity: `eval_run_id`, `rollout_id`, `suite`, `libero_task_id`, `task_name`,
      `init_state_index`, `seed`.
    - The policy: `checkpoint_sha256`, `train_run_id` (or `external_checkpoint` = hub id@rev),
      `policy_type`.
    - Protocol: `tau_episode_steps`, `render_hw`.
    - Outcome: `T_steps`/`T_s` (null if no success), `event`, `censored`, `n_ops`, `n_ops_done`, `ops` (k,
      conjunct, t_k).
    - Diagnostics: `conjunct_trace` (compact), per-step arm-action norm (a raw signal, no idle threshold applied),
      `sim_steps`, `wall_seconds`. The catalog's `idle_fraction` rule does **not** transfer: it counts steps
      whose six arm action dims are exactly 0.0 (`vlapoc/catalog.py:57, 275`, quoted). That is a teleop-device
      artifact; continuous policy outputs are almost never exactly zero (inferred).
  - **Trajectories**, when `record_trajectories=True` (the base policy that attribution will analyse):
    - A **lossless** LeRobot v3 dataset in the training schema (PNG, 128²), plus `rollout_id` and
      `policy_step`/`chunk_index`, so (s′, a′) and the observation each executed action was predicted from are
      recoverable.
    - LeRobot's built-in recording is lossy video (P15), so it is not used.
  - **The eval stage as an MLflow run** (`vlapoc/runs.py`), replacing `eval_standin` (single record system):
    - Inputs: the checkpoint (sha), linked to its train run when one exists.
    - Params: protocol, sim-stack versions, LIBERO version, assets digest.
    - Outputs: `rollouts` (sha), plus the trajectories dataset digest.
    - `stand_in=false`.
    - `scripts/lineage_dry_run.py` is switched to the real stage.
  - **Ghost events:** none for LIBERO. Owner decision 6 applies because LIBERO defines no off-table or
    out-of-reach rule (P13). The record has no ghost column, and the report states this explicitly.
- **Acceptance criteria (local):**
  - [ ] AC5.1 No-op policy smoke (dummy action): on 1 libero_spatial and 1 libero_10 task × 3 init states, every rollout is censored at τ_episode, n_ops_done = 0, and the record validates against the schema [PoC §3.2 p. 6 right-censoring]
  - [ ] AC5.2 Determinism: re-running the no-op smoke (same init state, seed) gives bit-identical records except `wall_seconds`. The same holds for the ACT checkpoint on CPU (MPS determinism: to verify during execution) [PoC §2.1 item 3 p. 3]
  - [ ] AC5.3 Lineage: the ACT smoke checkpoint `train-full-s0` (128², on our data) is evaluated on ≥ 2 tasks × 2 init states. The eval run's input is that checkpoint's sha, `trace(eval_run)` reaches v0's raw files (the CLI output is JSON-equal to the stored trace artifact; formatting may differ), and `stand_in` is false. The run is a smoke run and is tagged `eval.attribution_grade=false` (AC5.9) [PoC §2.1 item 1 p. 3]
  - [ ] AC5.4 Real-policy harness validation on `HuggingFaceVLA/smolvla_libero@6721902b` (256², per its config, P24), on ≥ 1 libero_spatial task and the libero_10 task with the most operations (task 8), × ≥ 10 init states each. Real outcomes; T, ops and censoring recorded; per-task KM computed with S1. Labelled "external checkpoint (not trained on our data), harness validation, not attribution-grade". Throughput on MPS is measured first; if it is too slow, the count is reduced and the reason recorded (to verify during execution) [PoC §9.1 Step 2 p. 18; owner decision 2 SmolVLA]
  - [ ] AC5.5 Trajectory store: for the AC5.4 rollouts, the dataset loads with `LeRobotDataset`. Frame count equals Σ sim_steps, and every frame maps to (`rollout_id`, `policy_step`) [CUPID Prop. 1 p. 5]
  - [ ] AC5.6 Init-state pairing: rollout j uses init state j regardless of `batch_size` (checked by running the same 4 rollouts with batch sizes 1 and 4: identical `init_state_index` → sim state at t0 sha) [P16, P17]
  - [ ] AC5.7 Grounding docstrings cite PoC §9.1 Step 2 p. 18, §2.1 items 1 and 3 p. 3, §3.5 A pp. 7–8, and LeRobot's `rollout()` source [vla-build]
- **Acceptance criteria (pending server: exactly what will be run there):**
  - [ ] AC5.8 (pending server) The SmolVLA base policy from S8 is evaluated at 50 rollouts per task × 20 tasks (init states 0–49, seed_base 0), τ_episode 60 s, 128², `record_trajectories=True`, for each of the 2–3 training seeds. Output: rollouts parquet + trajectories + MLflow eval runs.
    - **(r2, F-P5) The batch plan is fixed.**
      - One task per batch, `batch_size = 50` (all 50 init states of the task in one batch).
      - Tasks in `libero_task_id` order, libero_spatial before libero_10; init states 0–49 in order.
      - Eval seed 0. This is identical for every training seed and for every arm it will ever be compared with.
      - Source (quoted, LeRobot LIBERO docs): "use the same `--seed`, keep `--env.init_states=true`, and run
        each task in a single batch (`--eval.batch_size` equal to episodes per task)".
      - If server GPU memory forbids 50, use one smaller divisor of 50 (e.g. 10) for **all** arms and seeds, and
        record it.
    - The plan (batch size, task order, init order, eval seed) is stored as the `eval.batch_plan` param with
      its digest.
    - **Why:** measured — SmolVLA outcomes depend on the batch plan (same init and seed, batch 1 vs 5: init 4
      succeeds at step 71 vs never; analysis-r1, reproduced).
    - The runs are asserted attribution-grade by the runbook command (AC5.9).
    - Sources: [PoC §2.1 item 3 p. 3, §2.3 p. 4, §9.1 Step 2 p. 18; analysis-r1 F-P5]
  - [ ] AC5.9 **(r2, E1)** Attribution grade is asserted by the caller.
    - `eval_run(..., attribution_grade=False)` is the default.
    - `true` is accepted only if the caller asserts it **and** the protocol check passes:
      - 50 init states per task for every task of both suites;
      - τ_episode = 1200 steps;
      - a fixed `eval.batch_plan`;
      - the train run is not tagged `smoke`.
    - Otherwise it raises.
    - Smoke runs carry `vlapoc.smoke=true` and a run name containing "smoke".
    - Tests: (a) an eval with `attribution_grade=True` on 2 init states is refused; (b) the default gives
      `eval.attribution_grade=false`.
    - The round-1 runs mis-tagged `true` (`eval-act-train-full-s0-a`, `eval-act-train-full-s0-spatial0-n10`,
      `verify-ref-vs-drop-random-5 [eval-smoke]`) are listed in the log with how to read them ("tag meant
      'has a train run'"). They are not rewritten; MLflow history stays append-only.
    - Sources: [analysis-r1 E1; PoC §2.1 item 1 p. 3; §2.4 p. 4 "final claims should be about closed-loop behavior"]
  - [ ] AC5.10 **(r2, t0_model_sha256)** Fixture placement is part of the "fixed initial state".
    - Every rollout record carries `t0_model_sha256`: a sha256 of the post-reset model placement (`body_pos`
      and `site_pos` arrays; the executor chooses the exact bytes and documents them) next to
      `t0_state_sha256`.
    - Equality: for the same (task, init state), it is identical across rollouts, eval runs and batch sizes.
    - Negative test: a reused env reset 3× on libero_10 task 8 yields a different `t0_model_sha256` (the
      measured 7.2 mm stove drift).
    - `validate_record` requires the column.
    - **Why:** measured AC5.2b — `t0_state_sha256` misses fixture placement.
    - Sources: [PoC §2.1 item 3 p. 3 "fixed seeds and initial states"; analysis-r1 Non-blocking]
- **Verify by:** `<sim-python> scripts/eval_smoke.py --policy noop|ckpt:<train_run>|hf:HuggingFaceVLA/smolvla_libero@6721902b
  --tasks ... --n ...`; `python -m vlapoc.lineage trace <eval_run>`.
- **Demo value:** the first real closed-loop rollouts in the repo: SmolVLA on LIBERO, run on this Mac, with T
  and sub-task T per rollout, linked to the checkpoint in MLflow.

### S6: One return per metric per rollout (and per operation): the attribution-ready output

- **Why:**
  - Owner emphasis "per metric attribution ... is crucial".
  - PoC §4.3 p. 10 and §9.1 Step 3 p. 19 (the return definitions).
  - §3.4 p. 7 (operation-level returns, "Declared targets").
  - p. 1 ("per-metric and per-failure-type returns producing an episode × metric matrix").
  - P25: CUPID is binary-only.
- **Depends on:** S1, S3, S4, S5
- **Deliverables:**
  - **`vlapoc/returns.py`** → `returns.parquet`, keyed (`eval_run_id`, `rollout_id`) and carrying
    `train_run_id`, `checkpoint_sha256` and the cell.
    - **(r2) All per-rollout returns are family W: whole-task T, whole-task human anchor**, exactly as tabled
      in "Declared functionals (r2)".
    - One column per metric, each named with its τ:
      - `ret_f30` = 1[event ∧ T ≤ 30 s] (§4.3 "binary return").
      - `ret_rmst60` = −min(T, 60 s), with censored → −60 (§4.3 "negative capped time").
      - `ret_auc_sampled` = 1[T < T_h*] + ½·1[T = T_h*], T_h* drawn from the same task's whole-task human
        reference with a recorded seed (§9.1 Step 3, literal; ties ½ in r2).
      - `ret_auc_expected` = P(T_h > T) + ½ P(T_h = T) under the human reference: the deterministic
        expectation of the sampled indicator (a divergence, recorded; inferred lower variance).
      - `ret_hrt60_lin` = w_o · (−min(T, 60 s)), with w_o = (1/J) · RMST_h,o / RMST_m,o² evaluated at the
        current policy. This is the first-order (delta-method) linearization of the macro ratio: the form is
        inferred and must be confirmed by the attribution feature; §4.3 only says "first-order (linearized)".
      - `ret_censored60` = 1[no success by 60 s] (§3.4 "the censored tail").
      - `failure_type`.
    - Round-1 executor names (`ret_hrt60_lin`, `ret_censored60`) are kept.
  - **`operation_returns.parquet`** (libero_10, family O): one row per operation sample (PhAIL pairs).
    - Columns: `rollout_id`, `op_k`, `t_start`/`t_end` steps (the segment whose (s′, a′) the operation's
      return can be credited to), `T_op`, `event`.
    - **`ret_rmst60_op_cap` = −min(T_op, 60) for a completed operation, −60 for the unfinished attempt (r2,
      F-P3).** This replaces round 1's `ret_rmst60_op`, which used −T_tail for the unfinished attempt.
    - This answers §3.4 "Operation-level (T, event) pairs".
  - **(r2, E2) One metric registry**: a single module (e.g. `vlapoc/metrics_registry.py`) that S1
    (`evaluation`), S6 (`returns`) and S7 (`verdict`) all import. `metrics_registry.json` is its serialization.
    It replaces `evaluation.METRICS`, `returns.REGISTRY` and `verdict.METRICS`.
  - Per entry: key, family (W / O / scoring-only), T definition, τ, censoring convention, return column(s),
    verdict metric name, and the identity relation. For every PoC metric:
    - The functional of F, τ, and the return definition or "scoring only, no per-rollout return in the PoC"
      (median, UPH, completion, MTBF/A).
    - **Transfer status to LIBERO, with the reason:**
      - Ghost-event rate: not available, no LIBERO rule (P13, owner decision 6).
      - Intervention / safety-stop rate: no operator in simulation.
      - Blind randomized scheduling: replaced by fixed init states + seeds, which allow scene-level pairing
        that PhAIL cannot do. PhAIL: "initial physical configurations are not replayed", quoted.
      - Wall-clock T: sim time, excludes inference latency.
      - UPH: a projection only; production UPH needs reset time (PhAIL p. 5, quoted). **(r2) Declared
        functional `3600 / RMST(τ_episode)`, matching `evaluation.uph` and S2 AC2.8.** Table 1's
        `3600 / E[T | T < τ_episode]` and `common.py`'s per-episode `3600·S/T` are named as notes; they
        differ by up to 3× (measured, S2).
      - MTBF/A: "failures/assists" become non-completions.
      - F(30 s) transfers literally with τ_episode = 60 s.
  - **Failure types**, PoC-named, with what is and isn't built:
    - Built: `success`, `timeout_no_progress`, `timeout_partial_k_of_n` (from S4).
    - **Not built:** wrong object, missed grasp, ignored instruction, and **idle**. LIBERO defines no such rule
      (P13). The catalog's exact-zero idle rule does not transfer to policy actions, and the PoC gives no idle
      threshold. By the same reasoning as owner decision 6 they are deferred, but the raw per-step signals
      (goal-conjunct truth, task-object positions, arm-action norm) are in the record, so they can be added
      without re-running rollouts.
- **Acceptance criteria:**
  - [ ] AC6.1 **(r2 restated)** Family-W return identities on synthetic data and on **both** AC5.4 eval runs (libero_spatial:0 and libero_10:8), against the whole-task KM. Per cell, to ≤ 1e-9:
    - mean(`ret_rmst60`) = −RMST(60);
    - mean(`ret_f30`) = F(30);
    - mean(`ret_censored60`) = 1 − F(60);
    - mean(`ret_auc_expected`) = AUC_h (ties ½).
    - Real libero_10:8 must now give the whole-task values (the analyzer measured RMST(60) 48.400 s, F(30)
      0.300 as the whole-task numbers).
    - Sources: [PoC §4.3 p. 10, §3.2 p. 6; "Declared functionals (r2)"]
  - [ ] AC6.2 HRT linearization check on synthetic perturbations (labelled synthetic): for policy-time perturbations with |ΔHRT|/HRT ≤ 5%, Σ_rollouts Δ(mean `ret_hrt60_lin`) predicts ΔHRT_exact with relative error ≤ 10% (inferred: the first-order error is O(Δ²)). The measured error curve is saved [PoC §4.3 p. 10 "first-order (linearized) approximation"]
  - [ ] AC6.3 Attribution join contract (no influence computed): `returns.parquet` + the train run's training subset produce the empty episode × metric frame (rows = (`episode_id`, `content_hash`) of the subset that passes `check_scores`; columns = registry metrics with a return). Its shape is printed [PoC §4.2 p. 9, §9.1 Step 3 p. 19; lineage contract]
  - [ ] AC6.4 The registry covers 100% of §3.2 primitives, Table 1 scalars and the §4/§10 targets (F(30 s), HRT, RMST, AUC, ghost-event rate). Each has a transfer status and a reason; none is silently missing. The report states the not-built failure types (missed grasp, wrong object, ignored instruction, idle) next to the ghost-event statement [owner decision 4; brief "say so rather than silently dropping it"; PoC §2.1 item 3 p. 3]
  - [ ] AC6.5 Per-failure-type subsets: returns can be filtered by `failure_type` and by task, and S1 metrics recomputed on the subset. Demonstrated on AC5.4 [PoC §9.1 Step 3.3 p. 19]
  - [ ] AC6.6 Every return column name carries its metric and τ (e.g. `ret_rmst60`); the registry states the declared functional [PoC §3.4 p. 7 "Declared targets", §4.4 p. 10 "Label every claim with its metric and time cap τ"]
  - [ ] AC6.7 (pending server) Returns for every AC5.8 rollout (1000 per seed) [PoC §9.1 Step 3 p. 19]
  - [ ] AC6.8 **(r2, E2) One registry.**
    - A grep finds no metric-definition table outside the registry module (no `METRICS =`/`REGISTRY =` in
      `evaluation.py`, `returns.py` or `verdict.py`), and all three import it.
    - `metrics_registry.json` is regenerated, and its UPH entry reads `3600 / RMST(τ_episode)`.
    - Every entry with a return or a verdict metric has a `family` and a `T` field matching "Declared
      functionals (r2)".
    - `uph()`'s output on the S2 cohort equals 3600 / `rmst(τ_episode)` to ≤ 1e-9.
    - Sources: [analysis-r1 E2; single record system; PoC §3.4 p. 7 "Declared targets"]
  - [ ] AC6.9 **(r2, E5) No temp-dir leaks.**
    - The count of `data/scratch/verdict_*` and `data/scratch/returns_*` is the same before and after one
      `returns` run and one `verdict --eval-runs` run (`TemporaryDirectory`, or deletion after reading).
    - The 11 leaked directories from round 1 are removed, and their count before and after is logged.
    - Sources: [analysis-r1 E5; brief: disk]
  - [ ] AC6.10 **(r2, F-P2) Returns and verdict agree on the declared functional.** On both AC5.4 eval runs,
    for every family-W metric and every cell:
    - mean(return) equals the metric that `verdict.compare_policies` reports for that arm and cell (its point
      estimate) to ≤ 1e-9, with sign/weighting per the identity column: `ret_rmst60` ↔ −RMST(60);
      Σ_cells mean(`ret_hrt60_lin`) ↔ −HRT(60).
    - For family O: mean over operation samples of `ret_rmst60_op_cap` equals −`rmst60_op_cap` from the
      verdict's `family="op"` arm to ≤ 1e-9.
    - The comparison table is saved, and it replaces the analyzer's mismatch table (48.4 vs 37.0 s).
    - Sources: [analysis-r1 F-P2; PoC §3.4 p. 7, §4.4 p. 10]
  - [ ] AC6.11 **(r2, F-P3) The operation return ranks correctly.**
    - On `operation_returns.parquet` of the libero_10:8 eval run, every unfinished attempt has
      `ret_rmst60_op_cap` = −60, and no unfinished attempt's return exceeds any completed operation's return in
      the same cell.
    - Explicitly: `init005` op 2 (event 0) now scores −60, below `init003` op 2 (−15.50) and `init006` op 2
      (−12.10).
    - Synthetic test: an unfinished attempt with a short tail never outranks a completed operation.
    - Sources: [analysis-r1 F-P3; PoC §3.4 p. 7]
- **Verify by:** `.venv/bin/python -m vlapoc.returns --eval-run <id>` + the identity test script;
  `scripts/verify_returns_s6.py --real --against-verdict` (r2) on both AC5.4 eval runs, against a scratch lineage
  copy.
- **Demo value:** the episode × metric matrix's input made concrete. One table where each rollout carries five
  returns, each a declared functional with its τ. This is what CUPID's binary `success_mask` becomes in this PoC.

### S7: The verification gate: the two-step verdict (macro-KS, then per-metric CIs)

- **Why:**
  - PoC §2.6 p. 5 and §3.4 p. 7 (verification gate).
  - §4.4 p. 10: "Two-step verdict per intervention", "Crossing curves mean trade-offs", "correct for it (e.g.
    Bonferroni or Holm across metrics)".
  - §9.1 Step 4 p. 19.
  - PhAIL §3.3 p. 6 outcomes (i)/(ii)/(iii).
- **Depends on:** S1, S2 (for the real-data demo), S6
- **Deliverables:**
  - **`vlapoc/verdict.py`**, `compare_policies(arm_a, arm_b, metrics, n_boot, seed) →`:
    - The macro-KS D̄ and pooled-resample p (vendored).
    - Per-metric differences with episode-clustered bootstrap 95% CIs and two-sided bootstrap p.
    - Holm-adjusted p across metrics (statsmodels, or a tested small function: to verify during execution).
    - The outcome:
      - (i) gate fires and all significant metrics agree in sign;
      - (ii) gate fires and metrics disagree (CDFs cross, the trade-off signature);
      - (iii) gate does not fire, so the arms are indistinguishable at this N.
      - **(r2, accepted divergence) `gate_only`:** the gate fires but no metric's Holm-adjusted CI excludes 0,
        so no direction is established. PoC §4.4 p. 10: "each metric's bootstrap CI gives the direction"
        (quoted); `verdict.py:101-102`.
    - Stratified logrank as the sanity line.
  - **(r2, F-P2) Families.** LIBERO arms default to **family W**: whole-task (T, E) per rollout, a whole-task
    macro-KS gate per task, the family-W metrics, and the whole-task human anchor.
    - `family="op"` gives the operation-family verdict: operation pairs, operation-level gate,
      `rmst60_op_cap`, and the `human_operations.parquet` anchor.
    - The family is a recorded param and appears in every verdict label.
    - PhAIL arms keep PhAIL's own semantics (operation pairs, ties-dropped AUC) for S2/S7 reproduction.
  - **(r2, recorded) The human anchor is held fixed in the difference bootstrap.** Inferred: this removes
    common-mode noise. S2's single-policy HRT CI resamples it, to match PhAIL Table 2.
  - **(r2, F-P5) Batch-plan guard.** `compare_policies` reads each eval run's `eval.batch_plan` and refuses arms
    whose plans differ, unless `allow_plan_mismatch=True`, which labels the verdict "unpaired plan".
  - **CLI** over two eval runs, recorded as a `verify` MLflow run, or under `select`/`eval` per the existing
    stage list (the executor decides; extend `STAGES` if needed).
- **Acceptance criteria:**
  - [ ] AC7.1 **(r2 restated from PhAIL's released artifact, F-P1; r3 per owner ruling 2026-10-06)** On the S2 cohort (sorted export, AC2.9), with n_boot = 1000 and bootstrap seeds 0–4:
    - **OpenPI vs GR00T → (iii)** in 5/5 seeds.
    - **OpenPI vs ACT → (iii) at this N** in 5/5 seeds. PhAIL's released
      `out/explore/ks_power/dashboard_per_obj_macro.json` gives OpenPI–ACT `p 0.10294852573713144, status WEAK`
      (measured, analysis-r1). Our gate measured 0.105–0.123.
    - **(r3) GR00T vs ACT → (ii) in ≥ 4/5 seeds; the gate fires in 5/5 and RMST favours GR00T in 5/5; the AUC
      direction (ACT) is at the Holm boundary (Holm p 0.024–0.054).** This is the §5.3 crossing; the released
      JSON gives p 0.016 `HOLDS`. This is an **owner-approved divergence**, 2026-10-06, `dispute-r2.md`, ruling
      "B. Accept 4/5 (executor)" (brief, "Owner rulings").
    - **Any model vs SmolVLA → (i)** in 5/5 seeds.
    - **The prose-vs-artifact discrepancy is recorded in the report.** PhAIL p. 9 says "ACT and OpenPI/GR00T
      are different (KS rejects)" (quoted), but its own released macro-KS does not reject OpenPI–ACT at
      α = 0.05. PoC §3.3 p. 7's "resolved … OpenPI vs. ACT at 30 rollouts per … cell" is a detection rate from
      PhAIL §5.2's subsampling experiment at τ = 120 s (p. 8), not a full-cohort rejection.
    - The artifact wins.
    - Sources: [PhAIL §3.3 p. 6, §5.1–5.3 pp. 7–10, released `dashboard_per_obj_macro.json`; PoC §3.3 p. 7, §4.4 p. 10]
  - [ ] AC7.2 Null calibration smoke (cf. PhAIL App. J): over ≥ 200 random half-splits of one PhAIL model's episodes, the empirical rejection rate at α = 0.05 is reported. PhAIL reports 0.042–0.078 (quoted); ours is expected inside [0.02, 0.10] (inferred band at 200 trials) [PhAIL App. J p. 18–19]
  - [ ] AC7.3 KS is never used for direction: the verdict object exposes direction only from scalar CIs, and a unit test using PhAIL App. I's three intransitive CDFs shows sup-signs cycling while the verdict's direction comes from RMST [PoC §3.2 p. 6 "KS cannot rank on its own"; PhAIL App. I]
  - [ ] AC7.4 On LIBERO smoke data (AC5.3 ACT vs AC5.4 SmolVLA on a shared task and init states): the verdict runs end to end and is recorded as a run. Labelled smoke: the policies differ in training data and architecture, so this exercises the code, not a PoC claim [PoC §9.1 Step 4 p. 19]
  - [ ] AC7.5 (pending server) Full-data SmolVLA vs a "drop k random" retrain (k = 20, existing `drop_random_k`), each evaluated per AC5.8 → verdict recorded. This is the first real counterfactual gate on LIBERO.
    - **(r2, F-P5)** Both arms are trained with the same seeds and evaluated with the **identical** fixed batch
      plan of AC5.8: batch 50 = one task per batch, task and init order fixed, eval seed 0. If a smaller divisor
      is forced, the same one is used for both arms and all seeds.
    - The verdict's batch-plan guard (AC7.8) passes without override.
    - Family W is the declared verdict. Family O is reported alongside, labelled.
    - **(r3, owner ruling 2026-10-06, `dispute-r2.md`)** The server verdict commands use `n_boot = 1000`. Under
      this ruling, a server verdict whose Holm-adjusted p lands near α (as GR00T–ACT's AUC did on PhAIL data:
      Holm p 0.024–0.054 across seeds) may depend on the bootstrap seed. The verdict records its seed, and the
      report states this.
    - Sources: [PoC §2.6 p. 5, §9.1 Step 4 p. 19; LeRobot LIBERO docs, paired comparisons (quoted in AC5.8); owner ruling 2026-10-06]
  - [ ] AC7.6 **(r2)** The verdict artifacts of AC7.1–AC7.4 are regenerated from the sorted PhAIL export (AC2.9) and the family-W LIBERO arms. The logged p-values are updated, and the S7 log notes that the p-values are Monte-Carlo draws conditional on episode order and seed [analysis-r1 E3, E4]
  - [ ] AC7.7 **(r2, F-P2)** Family consistency on LIBERO: for both AC5.4 eval runs, the verdict's per-arm family-W metrics equal the S6 cell means of the matching returns (AC6.10 run from the verdict side), and the op-family `rmst60_op_cap` equals the operation-sample mean of `ret_rmst60_op_cap`. A label check: every verdict label contains its family [PoC §3.4 p. 7, §4.4 p. 10]
  - [ ] AC7.8 **(r2, F-P5)** The batch-plan guard: `compare_policies` on two eval runs with different `eval.batch_plan` raises without the override. With the override, the label contains "unpaired plan". Two runs with the same plan pass [LeRobot docs paired comparison; measured batch effect]
  - [ ] AC7.9 **(r2, E1)** Smoke verdicts are labelled smoke: `verify_run` sets `verify.attribution_grade=true` only when both eval runs carry an asserted `eval.attribution_grade=true` (AC5.9). `lineage_dry_run --eval-smoke` re-run gives `verify.attribution_grade=false` and a label containing "smoke", and `artifacts/s9_summary.json` is regenerated accordingly [analysis-r1 E1]
- **Verify by:** `.venv/bin/python -m vlapoc.verdict --phail OpenPI GR00T` (and the other pairs);
  `--eval-runs <a> <b>`.
- **Demo value:** PhAIL's three resolution outcomes reproduced by our gate on real data. The gate the
  counterfactual loop will use is shown to say "can't tell yet" when it should.

### S8: SmolVLA base-policy training definition with per-subset normalization

- **Why:**
  - Owner decision 3: "Every training run normalizes with mean/std computed only from its own training
    episodes ... Wherever this feature trains or defines training, it applies."
  - Owner decision 2: SmolVLA.
  - PoC §10 Foundation p. 20 "LIBERO ... + SmolVLA".
  - p. 1: data is the only variable.
  - The evaluation layer needs an attribution-grade policy (§9.1 Step 2 input "A policy trained on the
    dataset").
- **Depends on:** S0 (training subsets must come from blind versions)
- **Deliverables:**
  - `vlapoc/subset_stats.py`: subset stats = LeRobot `aggregate_stats` over the subset's per-episode stats
    from `meta/episodes`. No frame reads, no `recompute_stats` copy (P23).
  - A hook in `vlapoc/lerobot_train_entry.py` so `train()` builds its normalizer/unnormalizer from the subset
    stats. Lead: `dataset.meta.stats` feeds the processors at `lerobot_train.py:314, 351, 376` (quoted). The
    mechanism is the executor's choice.
  - `train_run` params: `normalization_stats = "per-subset (aggregate_stats over N episodes)"` plus the
    stats digest.
  - **(r2, F-P4) The SmolVLA training spec fine-tunes from `lerobot/smolvla_base@d9f33c94`**
    (`--policy.path=lerobot/smolvla_base`).
    - Source (quoted, LeRobot SmolVLA docs): "Use `smolvla_base`, our pretrained 450M model, and fine-tune it on
      your data".
    - Measured: S8 loaded 500/500 tensors and the frozen VLM was bit-equal; the download is owner-approved
      (906,712,520-byte weights).
    - Steps and batch size are taken from the LeRobot SmolVLA docs' fine-tuning example and fixed in the runbook
      (to verify during execution: the round-1 plan quoted the LIBERO page's `--steps=100000 --batch_size=64`).
      They are identical across seeds and arms.
  - **Divergence, recorded in the docstring, the runbook and the report.** Round 1 specified
    `--policy.type=smolvla --policy.load_vlm_weights=true` (quoted, LeRobot LIBERO docs example), which
    initializes from the 2.03 GB SmolVLM2 backbone.
    - The owner did not approve that download (brief, owner answers 2026-10-02: approved "the SmolVLA base
      model").
    - PoC §2.3 p. 4 only says "SmolVLA via LeRobot" (quoted), so no PoC page requires the backbone recipe.
    - Both are documented LeRobot recipes.
- **Acceptance criteria:**
  - [ ] AC8.1 `aggregate_stats` over all 1000 v0 episodes reproduces `meta/stats.json` for state and action to ≤ 1e-9 (planning: ≤ 2.4e-10) [P23]
  - [ ] AC8.2 Local 2-step SmolVLA run on MPS over a small subset (e.g. `v0/reference`, 40 episodes, about 280 MB cache measured in R1 of the data+lineage plan): the checkpoint's normalizer and unnormalizer state/action mean and std equal the subset stats to ≤ 1e-6 and differ from the full `stats.json`. If the SmolVLA base download blocks locally, an ACT 2-step run proves the hook and the SmolVLA check moves to pending server (recorded) [owner decision 3]
  - [ ] AC8.3 Two subsets differing by k episodes yield different stats digests, and `diff` reports the normalization difference as a data difference, not a config one [owner decision 3; PoC §2.1 item 1 p. 3 "what changed between the good run and the bad run?"]
  - [ ] AC8.4 Grounding docstring cites p. 1, PoC §2.3 p. 4 and owner decision 3 (brief) [vla-build]
  - [ ] AC8.5 **(r2 restated, F-P4; pending server)** SmolVLA **fine-tuned from `lerobot/smolvla_base@d9f33c94`** on `v0/candidate` (960 episodes):
    - per-subset normalization stats;
    - 2–3 training seeds;
    - fixed config, steps and batch size (from the runbook, identical across seeds);
    - render and train at 128².
    - The train run params record `policy.path=lerobot/smolvla_base`, the revision, and `load_vlm_weights`
      not used.
    - Checkpoints are registered as train runs, are not tagged `smoke`, and feed AC5.8.
    - Sources: [PoC §2.3 p. 4, §10 p. 20; brief owner answers; LeRobot SmolVLA docs]
- **Verify by:** `.venv/bin/python -m vlapoc.subset_stats --check v0`; `scripts/...` 2-step run plus the normalizer
  safetensors comparison.
- **Demo value:** normalization becomes a function of the training data, which keeps data the only variable;
  shown with the measured equality.

### S9: End-to-end evaluation chain locally, and the server runbook

- **Why:**
  - PoC §9.1 Steps 2–4 pp. 18–19 as one recorded chain.
  - §10 p. 20, Evaluation-layer row.
  - Brief constraint: mark server work with "exactly what will be run there".
- **Depends on:** S0–S8
- **Deliverables:**
  - `scripts/lineage_dry_run.py` updated so the eval and verdict stages are real (smoke scale). The chain is:
    train (ACT smoke or the S8 2-step run) → eval (AC5.3) → returns (S6) → verdict (S7) → `trace`.
  - `artifacts/server-runbook.md` with exact commands, expected outputs, disk and time estimates for:
    - AC5.8 (SmolVLA eval at full scale);
    - AC6.7 (returns at full scale);
    - AC7.5 (first LIBERO counterfactual gate);
    - AC8.5 (SmolVLA training).
- **Acceptance criteria:**
  - [ ] AC9.1 One command runs the local chain. `trace(<verdict run>)` reaches v0's raw-file sha256s through eval → train → subset → version. No stage is `stand_in=true` except attribution. **(r2, E1)** The chain's eval and verify runs are tagged smoke (`eval.attribution_grade=false`, `verify.attribution_grade=false`, labels contain "smoke"), and the S9 log line and `s9_summary.json` say so [PoC §2.1 item 1 p. 3, §9.1 p. 18–19; analysis-r1 E1]
  - [ ] AC9.2 The runbook lists every pending-server AC (5.8, 6.7, 7.5, 8.5) with its command, inputs (version/subset ids), outputs, and disk estimate. The disk estimate is inferred: about 37 KB per stored frame (7.36 GB / 200,340 frames, measured-derived) × expected frames. **(r2)** The runbook includes:
    - the fixed batch plan (AC5.8) in every eval command;
    - `attribution_grade=True` asserted only in the AC5.8 / AC7.5 commands;
    - the `smolvla_base@d9f33c94` fine-tuning command (AC8.5);
    - family W as the declared verdict.
    - Sources: [brief constraints; analysis-r1 F-P4, F-P5, E1]
  - [ ] AC9.3 The index spec-coverage rows for §3.2, §3.4, §3.5 A/B, §4.3, §9.1 Step 2 and §10 Evaluation layer are proposed to the summarizer, with built / pending server / not transferable [vla-build step 5]
- **Verify by:** `.venv/bin/python scripts/lineage_dry_run.py --eval-smoke`; then `python -m vlapoc.lineage trace
  <run>`.
- **Demo value:** a lineage graph from raw LIBERO files to a verdict over real rollouts, and a runbook that
  makes the server day mechanical.

---

## Out of scope (deliberately)

- **Attribution itself:** the CUPID fork, TRAK/influence computation, the episode × metric influence matrix
  values, the Kendall's τ agreement check across attribution vectors (§4.4), the four-group classification
  (§4.2), and "same dataset, opposite curations" (§4.5). Those belong to the attribution feature. This feature
  delivers their inputs (S6) and the gate (S7).
- **Ghost events in LIBERO:** LIBERO defines no fitting rule (P13). This is owner decision 6, and the report
  states it. The metric layer supports ghosts and is validated with them on PhAIL data (S1, S2).
- **Wrong-object / missed-grasp / ignored-instruction / idle failure rules:** LIBERO defines none, and the
  catalog's exact-zero idle rule does not transfer to policy actions. Inventing contact
  rules is the same kind of custom rule the owner declined for ghosts. Raw signals are recorded (S5) so they can
  be added later without new rollouts.
- **Metric-targeted and full corruption injections** (§2.5, §4.5, §8.10): the separate, owner-approved injection
  feature.
- **Real-robot execution on the PhAIL station or a positronic replica** (§3.5 B). This feature answers
  downloadability and license (AC2.6) and validates on the released rollouts. Fine-tuning on the 449 episodes
  (13.95 GB) is a later stage.
- **Power analysis / Brownian-bridge model** (PhAIL App. H), Q-Q plots (App. L Fig. 8), and the spatial-
  configuration study (App. G). These are not named by the PoC; simulation makes rollouts cheap (§3.5 A, quoted).
- **A paired-by-init-state bootstrap.** The spec's gate is PhAIL's unpaired pooled resampling; a paired variant
  is a possible later refinement (inferred).
- **Dashboard** (§2.7): figures only.
- **libero_object / libero_goal, SimplerEnv:** §10 asks for one or two suites.
- **(r2) Enforcing the v1 fence at method entry points.** No method under test exists yet. The screening and
  attribution features must call `lineage.assert_blind` on their input version (analysis-r1, judged
  acceptable for this feature).
- **(r2) Per-rollout policy RNG independent of the batch plan.** That would need patching LeRobot policy
  internals (noise sampling). The fixed batch plan (AC5.8, AC7.8) is the declared control instead, following
  LeRobot's own paired-comparison guidance.
- **(r2) The data+lineage regression re-run** (full `lineage_dry_run.py` without `--eval-smoke`, and
  `verify_runs_s5.py`, now under per-subset normalization and the real eval stage). This belongs to the
  refactorer, against a scratch lineage copy (analysis-r1 Non-blocking). It is not a planner AC.

## Open questions

- **Resolved, BLOCKING Q1 (v1's positional leak):** the owner chose A, "Fix code, fence v1 (Recommended)"
  (brief, 2026-10-02). Executed in S0: `check_blind(v1) = ['episode_index']`, fenced (analysis-r1, measured).
- **Resolved, CLAUDE.md reward wording:** the owner said "Yes, correct it"; the orchestrator edits it. The
  current CLAUDE.md already carries the corrected text.
- **Resolved, defaults stand:** T in seconds of simulated time; τ_episode = 60 s; ghost events out of scope;
  S8 in this feature; HRT anchor = v0 originals (brief, owner answers).
- **(r2) No new blocking question.** F-P2's choice of family is a planner decision, declared above. F-P4
  follows the owner's approved downloads.
- **Non-blocking: the external SmolVLA checkpoint** (`HuggingFaceVLA/smolvla_libero`) is used only to validate
  the harness on a real policy locally. Its training data is unknown (card: `datasets: unknown`), so its
  numbers are never attribution evidence.
- **Non-blocking: S8 (training) lives in this feature** because owner decision 3 applies "wherever this feature
  trains or defines training", and Step 2 needs a policy trained on our data. The owner or orchestrator may move
  it to a training feature without changing S0–S7.
- **Non-blocking: the HRT anchor** is all 1000 v0 originals, fixed across arms (inferred). Using the training
  subset's own demos would move HRT's denominator between counterfactual arms.

## Risks / premises to verify during execution

- **R1. Sim stack placement.**
  - The main 3.14 `.venv` is untested; the 3.12 path is proven (P10).
  - LeRobot's `LiberoEnv` + policy stepping on macOS is not yet run. Only robosuite/LIBERO env stepping,
    rendering and replay were (inferred feasible: lerobot env code has no platform gate, grep measured).
- **R2. MPS determinism and throughput.**
  - Throughput: SmolVLA with `n_action_steps: 1` infers every step (P24), and its speed on MPS is unmeasured.
  - Determinism: bit-identical rollouts on MPS are unverified.
- **R3. PhAIL cohort completeness.** 142 cohort episodes sit under `s3://inference/` (403). Point estimates may
  not match Table 2 exactly, which is why AC2.1 (implementation equivalence) carries the pass.
- **R4. Two simulators for one comparison.** The human reference is replayed in the demo XML; rollouts run in
  the BDDL env (P9). This is approximate by design (PoC p. 8), and the report states it.
- **R5. Assets pinning.** hf-libero downloads assets from the HF Hub on demand (P12). Pin the revision and hash
  the assets dir; otherwise rollouts are not reproducible.
- **R6. Disk.** Lossless trajectories cost about 37 KB per frame (inferred from the training set). 1000
  rollouts at τ_episode 60 s could reach about 300k–600k frames ≈ 11–22 GB per evaluated base policy (inferred).
  Only the attribution base policy records trajectories. Keep > 20 GiB free.
- **R7. Python 3.14 / draccus #69** (existing data+lineage R2): `PreTrainedConfig.from_pretrained` fails. For
  the hub checkpoint use the existing `load_checkpoint` path, or the 3.12 env.
- **R8. ~/.libero side effect** (P12): always set `LIBERO_CONFIG_PATH`.
- **R9. Init states.** Resolved in planning: 50 per task for all 20 tasks (measured, P10). AC3.7 re-checks it in
  the project environment.
- **R10. Delta-method form of the HRT return** (S6) is the planner's derivation. AC6.2 tests it numerically, and
  the attribution feature should re-derive it against CUPID §D.2 (variable-length trajectories).
- **R11. HRT CI method** (P19) is unknown until AC2.5.
- **R12. UPH definition** has three forms in PhAIL (Prior art). The adopted one is disclosed.
- **R13. Compute at τ_episode 60 s** (inferred about 2–4× steps on failures only). Measure in AC5.4 and revisit
  if the server budget is tight. Round 1 measured that on the 20 SmolVLA rollouts no success came after
  LeRobot's 280/520 budgets (analysis-r1).
- **R14 (r2). Batch 50 on the server GPU** (AC5.8) is untested. The fallback is one smaller divisor used for
  every arm and seed.
- **R15 (r2). Ties on the 20 Hz grid.** Family-W AUC counts ties as ½, while PhAIL's validation AUC drops
  them. The two are not comparable across the PhAIL and LIBERO reports without saying so. The registry records
  both rules.
- **R16 (r2). `rmst60_op_cap` is pessimistic** relative to the descriptive operation KM whenever an operation
  is censored before 60 s. It is labelled `_cap`, and attribution claims on family O must carry that label.
