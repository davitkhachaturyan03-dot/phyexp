<!-- Saved verbatim by the orchestrator from the analyzer's returned text: the analyzer's own
     write was refused by the harness. -->

# Dispute: Should AC7.1 accept "GR00T vs ACT → (ii) in ≥ 4/5 seeds" at 1000 bootstrap draws, or keep "5/5" and require enough draws for the result not to depend on the seed?

## The decision you need to make

Our verdict code reproduces PhAIL's trade-off result for GR00T vs ACT (outcome (ii): GR00T wins on RMST, ACT
wins on AUC) in 4 of 5 bootstrap seeds at the default 1000 bootstrap draws. In seed 1 the AUC's
Holm-corrected p lands at 0.054 instead of below 0.05, so that seed reports (i). The question is whether this is
a property of the data, so the AC should be weakened, or a precision artifact, so the code should use more draws.

- **Option A (analyzer's position): keep "(ii) in 5/5 seeds"; make the direction bootstrap precise enough that
  the label does not depend on the seed.**
  - Raise the default `n_boot` of the per-metric bootstrap to 10,000, or adopt a precision rule (below). Report
    each p's Monte-Carlo standard error. Re-run the seed sweep and regenerate the S7 artifacts.
  - Measured at 10,000: 5/5 seeds give (ii).
  - Consequence: AC7.1 passes as written. The PhAIL sweep costs about 70 s per pair per seed instead of about
    8 s (measured). The server counterfactual verdict (AC7.5) inherits the precision. The report can state
    GR00T–ACT (ii) as a stable result.
- **Option B (executor's position): restate AC7.1 as "GR00T vs ACT → (ii) in ≥ 4/5 seeds; the gate fires in 5/5
  and RMST favours GR00T in 5/5; the AUC direction (ACT) is at the Holm boundary (Holm p 0.024–0.054)".**
  - n_boot stays 1000.
  - Consequence: AC7.1 passes now with no code change. The report says the GR00T–ACT trade-off is not robust to
    the bootstrap seed. The server verdicts keep n_boot 1000.
- **Option C (middle path): keep n_boot 1000 as the default, but make the verdict report the Monte-Carlo
  standard error of every p and flag any outcome with a Holm p within 3 MC SE of α as "MC-unstable"; state
  AC7.1 at n_boot 10,000 for the reproduction.**
  - Consequence: cheap by default and honest labels. The AC passes at the stated n_boot. A server verdict that
    lands near the boundary is flagged instead of silently flipping.

**Analyzer recommends: A (C is acceptable).** At 10,000 draws the AUC's Holm p is 0.025–0.032 in all 5 seeds,
so (ii) is the data's answer, and only the 1000-draw Monte-Carlo error makes seed 1 flip. You may overrule this.

## Context: what we're building and where this sits

- **Feature:** `tracker/features/2026-10-01-evaluation-layer/`, story S7, the two-step verdict: the macro-KS
  gate, then per-metric bootstrap CIs with Holm.
- **The AC:** AC7.1 (r2) reproduces PhAIL's outcomes on PhAIL's released real-robot data:
  OpenPI–GR00T (iii), OpenPI–ACT (iii), GR00T–ACT (ii), any vs SmolVLA (i).
- **Spec:** PoC §4.4 p. 10:
  - "Two-step verdict per intervention. First the macro-KS gate … Then each metric's bootstrap CI gives the
    direction";
  - "When the gate fires but metrics disagree in sign, the CDFs cross … That is exactly the trade-off episode
    signature";
  - "correct for it (e.g. Bonferroni or Holm across metrics) before labeling an episode as a trade-off" (quoted).
- **PhAIL §5.3 p. 9:** "This is outcome (ii) of §3.3 … their CDFs cross, with ACT winning the early-time region
  (high AUC) and OpenPI/GR00T winning the integrated region (low RMST)" (quoted).
- **Code today:**
  - `vlapoc/verdict.py` `compare_policies(..., n_boot=1000, n_boot_ks=2000, seed=0)`;
  - the two-sided percentile-bootstrap p is `min(1, 2·min(#(Δ*≤0)+1, #(Δ*≥0)+1)/(n_boot+1))`;
  - Holm across the family's 4 metrics (PhAIL family: F(30 s), HRT(240), RMST(240), AUC);
  - `outcome()`: (ii) iff the gate fires and the Holm-significant metrics disagree in sign.
- **Why it matters beyond this AC.** The same verdict, with the same default n_boot, will label the first real
  LIBERO counterfactual (AC7.5, server). A label that flips with the RNG seed would make the PoC's "trade-off
  signature" depend on the seed, not on the data.

## The disagreement, end to end

1. **What the plan says** (`plan.md`, AC7.1 r2): "On the S2 cohort (sorted export, AC2.9), in 5/5 bootstrap seeds
   (0–4): … **GR00T vs ACT → (ii)**, the §5.3 crossing: the released JSON gives p 0.016 `HOLDS`; RMST favours
   GR00T and AUC favours ACT, both Holm-significant (executor measured, 5/5 seeds)."
   - The "5/5" came from the executor's round-1 sweep, which ran on the PhAIL export before it was made
     deterministic (AC2.9). My round-1 F-P1 proposed the outcomes without a seed count; the planner added it.
2. **What was built:** `vlapoc/verdict.py` (above) and `scripts/verify_verdict_s7.py::phail_seeds`, which calls
   `vd.compare_policies(arms[x], arms[y], arms["Teleop"], "PhAIL", n_boot=1000, n_boot_ks=2000, seed=sd, …)`
   for seeds 0–4.
3. **What I flagged, and why.** I reproduced the executor's ❌ exactly, then probed the cause (measured):

   | n_boot | seed | gate p | RMST Holm p | AUC raw p | AUC Holm p | outcome |
   |---|---|---|---|---|---|---|
   | 1000 | 0 | 0.0070 | 0.0240 | 0.0100 | 0.0300 | (ii) |
   | 1000 | 1 | 0.0065 | 0.0080 | 0.0180 | **0.0539** | **(i)** |
   | 1000 | 2 | 0.0100 | 0.0400 | 0.0120 | 0.0400 | (ii) |
   | 1000 | 3 | 0.0085 | 0.0320 | 0.0100 | 0.0320 | (ii) |
   | 1000 | 4 | 0.0085 | 0.0300 | 0.0060 | 0.0240 | (ii) |
   | 10000 | 0 | 0.0070 | 0.0224 | 0.0104 | 0.0312 | (ii) |
   | 10000 | 1 | 0.0065 | 0.0192 | 0.0102 | 0.0306 | (ii) |
   | 10000 | 2 | 0.0100 | 0.0208 | 0.0098 | 0.0294 | (ii) |
   | 10000 | 3 | 0.0085 | 0.0208 | 0.0108 | 0.0324 | (ii) |
   | 10000 | 4 | 0.0085 | 0.0248 | 0.0068 | 0.0248 | (ii) |

   - The 10,000-draw AUC raw p clusters at 0.007–0.011, so the converged Holm-adjusted p is about 0.03.
   - Seed 1's 0.018 at 1000 draws is a Monte-Carlo excursion. The binomial SE of a p near 0.010 at 1000 draws is
     about 0.003 (inferred arithmetic). The Holm multiplier of 3 turns that into about 0.01 on the adjusted p.
   - This is resampling risk: Gandy (2009) defines it as "the probability of a different decision than the one
     based on the theoretical p-value" (quoted, arXiv math/0612488).
4. **The executor's rebuttal** (`execution-log.md`, "Rebuttals to analyzer", verbatim): "The 5/5 came from the
   round-1, **unsorted** export. On the sorted export it is 4/5. The AUC direction sits at the Holm α = 0.05
   boundary, so "(ii)" for this pair is a Monte-Carlo outcome, not a stable property. A guaranteed 5/5 cannot be
   promised for a boundary quantity. Proposed wording: "GR00T vs ACT → (ii) in ≥ 4/5 seeds; the gate fires in
   5/5 and RMST favours GR00T in 5/5; the AUC direction (ACT) is at the Holm boundary (Holm p 0.024–0.054)"."
   Also, from the log: "I did not change n_boot or the rule to make it pass."
5. **Why I'm not convinced, point by point:**
   - *"The 5/5 came from the unsorted export."* True (measured in round 1, finding E3). It shows the planner's
     number was itself a draw, not that 4/5 is the right expectation.
   - *"The AUC direction sits at the Holm boundary."* At 1000 draws its Holm p ranges over 0.024–0.054, yes. At
     10,000 draws it is 0.025–0.032 in all 5 seeds, so the underlying quantity is not at the boundary. The spread
     straddling 0.05 is Monte-Carlo error.
   - *"(ii) is a Monte-Carlo outcome, not a stable property."* With adequate draws it is stable: 5/5 (measured).
     Writing "≥ 4/5" would put Monte-Carlo noise into the spec and keep the same instability in the server
     verdict.
   - *"A guaranteed 5/5 cannot be promised."* At any finite n_boot no MC decision is guaranteed. But resampling
     risk falls as n_boot grows, and it is negligible when the converged p is about 6 MC SE from α at 10,000
     draws (inferred arithmetic: (0.05 − 0.031) / (3 × 0.001)).
   - *"I did not change n_boot to make it pass."* A sound instinct about post-hoc tuning, and I steelman it below.
     But n_boot is a Monte-Carlo precision parameter. Raising it converges to the same bootstrap p; it changes
     neither the test, α, Holm nor the outcome rule. The guard against post-hoc tuning is to fix the precision
     rule for every verdict, LIBERO included, not to accept the noise.

## Evidence on both sides

| Claim | Made by | Evidence | Label | I checked it? | Result |
|---|---|---|---|---|---|
| Seed 1 at n_boot 1000 gives (i), AUC Holm 0.0539 | executor | `verify_verdict_s7.py --phail-seeds 0 1 2 3 4` | measured | yes, re-ran | holds exactly; all other 29 (pair, seed) outcomes match |
| Round-1 5/5 came from the unsorted export | executor | round-1 log; AC2.9 | measured | yes (round 1 E3) | holds |
| AUC is at the Holm boundary, so (ii) is not a stable property | executor | the 1000-draw spread 0.024–0.054 | inferred from the measured spread | yes, at 10,000 draws | does **not** hold: AUC Holm 0.0248–0.0324, (ii) in 5/5 |
| At 10,000 draws GR00T–ACT is (ii) in 5/5 seeds | analyzer | probe calling `compare_policies(…, "PhAIL", n_boot=10000, n_boot_ks=2000, seed=0..4)`; about 70 s each | measured | — | (ii) 5/5 (table above) |
| Near a threshold, MC decisions are unreliable and need more replicates | analyzer | Gandy 2009, resampling risk "the probability of a different decision than the one based on the theoretical p-value" | quoted | — | supports more draws or a sequential rule, not a weaker expectation |
| PhAIL never claims Holm-significant AUC for this pair | executor | PhAIL §5.3 p. 9 | quoted | yes | true: PhAIL applies no Holm there. But PoC §4.4 requires Holm before labelling a trade-off (quoted), so our AC is about our verdict, which must apply it |
| PhAIL's released gate supports the pair | executor | `dashboard_per_obj_macro.json`: GR00T–ACT p 0.016, `HOLDS` | measured | yes (round 1) | holds; our gate fires in 10/10 runs |

## What each option costs

- **A.**
  - Effort: about 1 hour. Change the default `n_boot` (or add a precision rule) in `verdict.compare_policies`,
    `verify_run` and the CLI; re-run the sweep (about 6 min per pair for 5 seeds, measured) and `--phail-all`;
    regenerate the S7 artifacts; update the log.
  - Compute: the PhAIL verdicts take about 9× longer. The LIBERO verdicts on 1000 rollouts are unmeasured
    (inferred minutes on the server).
  - Risk to PoC claims: none. It strengthens the §4.4 trade-off claim and the AC7.5 gate.
  - Reversible.
- **B.**
  - Effort: a plan wording change.
  - Risk: the report would call the GR00T–ACT trade-off seed-dependent, which the 10,000-draw evidence shows is
    an artifact. The AC7.5 server gate keeps resampling risk at 1000 draws, so a real counterfactual could be
    labelled (i) or (ii) depending on the seed.
  - Reversible, but the report text would need correcting later.
- **C.**
  - Effort: about 1–2 hours (MC SE per p, an "MC-unstable" flag, the AC7.1 sweep at 10,000).
  - Compute stays low by default.
  - Risk: low. Unstable labels become visible rather than silent.

## What I'd need to see to change my mind

- A run at n_boot ≥ 10,000 (any seed in 0–4, or another seed set) where the GR00T–ACT AUC Holm p is ≥ 0.05; or
- a PoC or PhAIL source that fixes the bootstrap at 1000 draws; or
- evidence that the AUC Holm p crosses 0.05 at 10,000 draws on a different but equally valid deterministic
  episode order.

## If you rule, the pipeline will

- **A** → route the **executor**: set the per-metric bootstrap default to n_boot = 10,000 (or implement a stated
  precision rule) in `vlapoc/verdict.py` (`compare_policies`, `verify_run`, CLI); report MC SE; re-run
  `verify_verdict_s7.py --phail-seeds 0 1 2 3 4` (expect 30/30) and `--phail-all`; regenerate `s7_*`; log it.
  Route the **planner**: add the precision rule to AC7.1 and to AC7.5's verdict commands in the runbook. Then I
  re-check and expect PASS.
- **B** → I accept it, record it as an owner-approved divergence ("GR00T–ACT (ii) in ≥ 4/5 seeds at
  n_boot 1000"), route the **planner** to adopt the executor's wording, and expect PASS with no code change.
- **C** → route the **executor**: MC SE and the "MC-unstable" flag in `compare_policies` / `describe`, and the
  AC7.1 sweep at n_boot 10,000. Route the **planner**: restate AC7.1 at n_boot 10,000 and note the flag in AC7.5.
  Then PASS on re-check.
