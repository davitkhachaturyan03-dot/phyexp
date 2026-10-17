<!-- Saved verbatim by the orchestrator from the analyzer's returned text: the analyzer's own
     write was refused by the harness. -->

# Dispute: Should the injected corruptions be rebuilt so that one-line statistics cannot find them, before the ground truth is used?

## The decision you need to make
Five of seven corruption types score AUROC 1.0 against a single catalog-level statistic. Which detectability leaks do we remove in the one planned re-derivation?

- **Option A: fix only the defect.** Change the `time_shift` tail (trim), keep everything else, and report recovery per type against the heuristic baselines.
  - Build: one re-derivation, small code change.
  - PoC: action noise and frozen frames stay "easy", matching RoboDrop's own setup. Idle padding stays findable by duplicate frames. Swaps stay findable by the `task_name` cross-check.
  - Report: must say that only `gripper_release`, `detour` and (after the fix) `time_shift` test anything beyond heuristics.
- **Option B: A plus the cheap realism fixes.**
  - **Swaps:** hide provenance from methods. Use opaque episode keys, drop `task_name`, `libero_task_id` and `source_*` from the `MethodView` catalog, and shuffle physical order independently of the labels. This follows RoboDrop p.13, "Operator provenance is used only for evaluation and is unavailable to RoboDrop and the baselines". It touches the score-table key and the row-order blind rule (plan D1).
  - **Action noise (optional):** temporally correlated noise with the same marginal 0.25·σa. RoboDrop does not say the noise is independent per step (inferred), but this departs from the reading every RoboDrop-comparable number would use.
  - Cost: moderate, mostly in the lineage key scheme.
- **Option C: B plus re-rendered idle padding.** Simulate the pauses in LIBERO from the stored state with zero action, so the frames are real and not byte-identical holds. This is the same cost class as the viewpoint corruption the owner already ruled out of scope.

**Analyzer recommends A**, plus logging B and C as open owner questions. The PoC sets no detectability requirement, RoboDrop accepted a trivially separable action corruption, and A removes the one leak that also corrupts the §4.5 test. You may overrule this.

## Context: what we're building and where this sits
- **Feature:** `2026-10-15-ground-truth` (PoC §10 p.20 "Ground truth"; §2.1 item 4 p.3; §2.5 p.5; §8.10 p.17).
- **v2 so far:** derived with 91 of 960 candidates corrupted, 13 per type. Blind, verified, trainable.
- **S5 heuristic recovery:**

| Type | Best one-line statistic | AUROC |
|---|---|---|
| action_noise | smoothness | 1.0 |
| time_shift | idle_steps | 1.0 |
| idle_padding | idle_steps, duplicate frames | 1.0 |
| dropped_frames | duplicate frames | 1.0 |
| instruction_swap | instruction vs `task_name` | 1.0 |
| detour | length | 0.73 |
| gripper_release | any | about 0.5 |

## The disagreement, end to end
This is not an agent-versus-agent dispute. The executor itself flagged it in "Findings for the analyzer (batch 2)": "Recovery on the pooled set therefore overstates how hard the benchmark is for a real method." It is escalated because the trade-off is the owner's and it determines what goes into the single re-derivation.

1. **The plan** (types table): `time_shift` "For the last 40 steps: arm 0, gripper = a[T−1, 6] … tail fill: inferred, since RoboDrop states neither". D1: "Swapped instructions keep true provenance … the cost is that a provenance-vs-instruction cross-check finds every swap." Out of scope: "Sim-rendered idle or hesitant frames."
2. **What was built:** `corruptions.py:271-274`, `out[:t - steps] = a[steps:]; out[t - steps:, :6] = 0.0; out[t - steps:, 6] = a[-1, 6]`.
3. **What I flagged (measured):** all 13 `time_shift` episodes have exactly 40 idle steps, all from the tail. Clean data has at most 21 idle steps per episode and at most 7 in any last-40 window; the final arm action is exactly 0 in only 9/1000. Idle padding has 80 duplicate frame pairs, while clean data has 0 across 199,340 pairs despite 436 zero-action steps.
4. **The executor's position (quoted):** "The H5/H6 findability is a property of how the corruptions are built … fixing them needs re-rendering or opaque keys (out of scope)."
5. **Where I agree and disagree.** I agree that H5/H6 cost what the executor says, so that is the owner's call (B/C). I disagree that everything is out of scope: the `time_shift` tail costs nothing to fix and is not a realism question, because it is an invented idle segment.

## Evidence on both sides
| Claim | Made by | Evidence | Label | Checked? | Result |
|---|---|---|---|---|---|
| The PoC requires no undetectability | analyzer | p.3 "check whether your attribution method flags exactly those"; p.5 "Inject 5–10% … measure how many the method ranks near the top" | quoted | yes | holds |
| RoboDrop accepted a trivially separable corruption | analyzer | RoboDrop p.7 "DataMIL and QoQ perfectly separate the controlled action corruption" (Table 1: 100.0) | quoted | yes | holds |
| Provenance hidden from methods in RoboDrop | analyzer, for B | RoboDrop p.13 (real-robot non-expert labels, not swaps) | quoted | yes | holds, though for a different corruption |
| Attribution's value is where heuristics fail | analyzer | CUPID TuckBox appendix, "attending to heuristic properties … may result in poor curation performance" | quoted | yes | holds |
| `time_shift` detection comes only from the tail | analyzer | v2 idle_steps = 40 for 13/13; clean last-40 maximum 7 | measured | yes | holds |
| Duplicate frames are a construction artifact for idle padding | analyzer | 0 duplicate pairs in clean data across 436 zero-action steps (P4 plus my v0 count) | measured | yes | holds |
| Exact duplicates are realistic for dropped frames | executor / plan | LeRobot `read_latest` sample-and-hold (plan P4, quoted there) | quoted (plan) | not re-fetched | accepted |
| Fixing H5/H6 needs re-render or opaque keys | executor | Plan D1 lists what the alternative touches | inferred | partly | holds for H6; for H5 only on idle_padding |

## What each option costs
- **A:** about 1 hour of code plus one re-derivation (about 7 GB, about 9 min, disk 87 GiB to about 80). No risk to the §1 claims. The report must carry the per-type caveat.
- **B:** a lineage key-scheme change (score keys, row-order blind rule, `MethodView` columns) plus re-running S2–S5 verifiers. Medium risk of regressions in blindness. Correlated noise makes RoboDrop numbers non-comparable unless both variants are kept.
- **C:** a simulator replay pipeline. It is the same class of work as the viewpoint corruption ruled out on 2026-10-15. Highest cost.
- **All options:** reversible later, by deriving another version.

## What I'd need to see to change my mind
- **To accept the `time_shift` tail as it is:** a source showing RoboDrop or another cited paper fills the shifted tail with zero commands, or a measurement that clean LIBERO episodes end with 40 or more idle steps (currently at most 7).
- **To recommend B over A:** a PoC passage requiring the injected corruptions to be undetectable by metadata.

## If you rule, the pipeline will
- **A:** route the planner to amend the `time_shift` row and AC1.5, then the executor to re-derive and re-run S4–S6.
- **B or C:** the planner adds the chosen stories (provenance hiding, correlated noise, re-rendered idle) to the same re-derivation, and the executor implements them.
- **"Keep everything as is":** I record it as an owner-approved divergence (the tail confound is noted in the report) and continue to PASS on the remaining items (the AC1.5 amendment is still needed).
