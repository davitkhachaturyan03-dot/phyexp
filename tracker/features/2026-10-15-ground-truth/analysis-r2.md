VERDICT: REWORK
ROUTE: planner+executor
ROUND: 2

<!-- Saved verbatim by the orchestrator from the analyzer's returned text: the analyzer's own
     write was refused by the harness ("Subagents should return findings as text"). -->

## Summary (3–5 lines)
Owner ruling A is implemented correctly. The trim is `corruptions.py:280-284`: obs[0:T−40] paired with a[40:T], with `length_change = -40` recorded. AC1.8 holds (child idle steps: max 2). v3 = `002a68e4caf2…`, and every AC from AC4.1 to AC4.8 reproduces on stored bytes. v2 is fenced as "superseded by v3". The size reconciliation is verified. The one blocker is AC5.8's per-type statement, which ruling A requires ("saying plainly which types test anything beyond one-line statistics"). It was built on the six raw heuristics only. Simple task-relative statistics, which need no labels, find `gripper_release` (0.96), `detour` (0.96) and `time_shift` (0.86), so the reported bands (c)/(b)/(c) and the 0.68 "residual" overstate how hard the ground truth is.

## Rebuttals received
| From | Point disputed | Their evidence | Conceded? | Why / why not |
|---|---|---|---|---|
| none | – | – | – | Round-1 feedback was accepted and implemented. The owner settled the dispute (ruling A), and I am not re-opening it. |

## Round-1 items
| Item | Status | Evidence |
|---|---|---|
| Planner 1: replace the `time_shift` fill with a trim, add AC1.8, report H1 | ✅ | Diff `766c33c`. S1: AC1.8 child idle max 2, p99 2. S5 v3: idle_steps AUROC 0.445 (was 1.000). |
| Planner 2: amend AC1.5 | ✅ | plan.md l.302. Re-run: 118/118/118 for the stochastic types, detour 94.9%, swap 89.0%, time_shift and gripper_release 0/118, asserted. |
| Planner 3: budget one re-derivation, retire v2 | ✅ | v3 is the only new full-scale derivation. `fence.json` on v2 says "superseded by v3". The v2 copy is kept for the owner to delete. |
| Planner 4: known issue that `train_run` bypasses the loader | ✅ | R11; the runbook has a "Known issue for the attribution feature" section (AC6.6). |
| Executor 1: re-derive, re-run S4–S6 | ✅ | Reproduced below. |
| Executor 2: 7.3G vs 6.8G | ✅ verified | Byte sums are v2 7,346,100,191 and v3 7,326,736,215, exactly as logged. `du -sk` is v2 7,174,088 KiB (6.84 GiB, matching the bytes) and v3 7,456,304 KiB now, down from the logged 7,775,640. So v3's block slack is measurably shrinking over time. That supports the executor's APFS-preallocation explanation, which they had labelled inferred. Today `du -sh v3` prints 7.1G. |
| Non-blocking: P7 numbers | ✅ | plan.md l.174 now gives 907/960 and 884. |
| Non-blocking: stale data+lineage AC4.8 contrast figure | ⏳ not yet | R12 assigns it to the executor or summarizer. There is no `changelog.md` entry yet, and `2026-09-28-data-lineage-layer/report.html:1888` still says "60 files / 663 MB". |

## Acceptance criteria
I re-ran everything into `data/scratch/gt_an3/` (logs in `logs/`). Main `mlflow.db` sha256 was `e5499b65acfc…` before and after. S3 and S6 overwrote `artifacts/s3_verify.json` and `artifacts/s6_v3_verify.json`; I restored both with `git checkout`.

| Story | AC | Claimed | Reproduced? | Evidence |
|---|---|---|---|---|
| S1 | 1.4 (trim) | ✅ | ✅ | 0 invariant failures × 7 on 118 episodes. |
| S1 | 1.5 amended | ✅ | ✅ | Same hash in a second process for all seven types. Seed-sensitivity counts exactly as logged. |
| S1 | 1.6 | ✅ | ✅ | 960×6 and 884; 907/960. |
| S1 | 1.8 | ✅ | ✅ | max 2, p99 2, > 0 in 6 of 118. |
| S2 | regression | ✅ | ✅ | `all S2 checks pass: True`. |
| S3 | regression | not re-run in round 2 | ✅ (me) | `all S3 checks pass: True`. The committed `s3_verify.json` is stale: it carries scratch child id `3f43df489885`, while the trimmed code gives `dbb9f38a473f`. |
| S4 | 4.1–4.8 on v3 | ✅ | ✅ | 91 labels, 13 per type. `check_blind = []`. 1000/1000 hashes re-hash correctly, 909 equal v0, 91/91 match their labels. Clip recount 65 / 139. Trace visits 3 runs. AC4.8: 78 rows keep their hash, 13 `time_shift` rows differ with −40, v2 is refused and v3 opens. |
| S5 | 5.1–5.3, 5.5 | ✅ | ✅ | 26/26 fixtures. Random-ranker AUROC 0.4987. Both key rejections reproduce. Leak check v3 0.099 / 0.505. |
| S5 | 5.4, 5.6, 5.7 | ✅ | ✅ (spot-check) | Raw length, reversed, for `time_shift` = 0.682 (re-measured independently). |
| S5 | **5.8** | ✅ | **❌ in substance** | The AC's mechanics hold, but the statement it produces is wrong for three of seven types. See the next section. |
| S6 | 6.1–6.6 | ✅ | ✅ | Stats difference 2.60e-10. SmolVLA smoke run finished in 16.3 s with matching digest `1d2a5e14…`. Human-reference digest unchanged. Real store unchanged (15 subsets before and after). The runbook names v3. |

## PoC scope assessment
- **Residual length leak: is reporting it openly enough under ruling A?** Yes in principle. Ruling A's meaning, quoted in `brief.md`, is to report recovery per type "saying plainly which types test anything beyond one-line statistics", so disclosure is the ruled remedy and I am not asking for a different corruption. But the number disclosed must be the honest one, and 0.682 is not.
  - Lengths vary strongly by task (75 to 517 steps). Comparing each episode's length with its task's median (computed over all candidates, no labels) gives `time_shift` AUROC **0.848**, and a task z-score gives **0.864**. That is band (b), not (c). `task_name` is a column methods can see. Measured with `data/scratch/gt_an3/lenres.py`.
  - The same statistic, longer = suspicious, gives `detour` **0.96** and `idle_padding` **0.95**, so `detour` moves from (b) to (a).
- **`gripper_release` is not "not found".** Task-relative gripper statistics read from v3's stored actions, with label-free task medians, measured with `data/scratch/gt_an3/grip.py`:

| Statistic (vs task median) | AUROC |
|---|---|
| share of the episode after the final release | **0.964** |
| share of steps with the gripper open | 0.949 |
| steps after the final release | 0.949 |

  The direction follows from the construction: a premature release lengthens the open tail. Caveat: I designed these statistics knowing the corruptions. H1–H6 were designed the same way, so this is the same standard, not a stricter one.
- **Consequence for the report.** Under the plan's own bands, no type is in band (c). Four types are in (a) with raw statistics, `detour` and `gripper_release` reach (a) with task-relative ones, and `time_shift` is in (b). This doesn't break the ground truth. RoboDrop accepted corruptions that are easy to separate (round 1, quoted), and ruling A keeps the corruptions as they are. But the report must not tell stakeholders that two types "test a method beyond heuristics". The correct statement is that every injected type can be found at 0.85 or better by a one- or two-line, task-normalised statistic, so a method's recovery number has to be compared against these baselines per type. This is information for the owner, not a re-opening of ruling A.
- **v2's retirement is sufficient.** It is fenced, the loader refuses it (re-run), it is kept physically as history, and deleting it is the owner's call. One wording nit: the refusal message for a supersession fence reads "not blind to its edits … offending=[]". That is accurate about the fence but misleading about the reason (non-blocking).
- **Over-built:** nothing. `retire_version` is 10 lines over the existing `write_fence`.
- **Silent divergences:** none. The trim's length change is in `params` and in the docstrings.

## Overcomplication & reinvention audit
| Mechanism (file:line) | Existing OSS alternative | Covers our need? | Evidence | Recommendation |
|---|---|---|---|---|
| `time_shift` trim (`corruptions.py:280-284`) | numpy slicing | It already is slicing | quoted (code) | keep |
| `retire_version` (`groundtruth.py:373`) | Existing `lineage.write_fence`, which it wraps; DVC/MLflow have no "fence from method use" | n/a | quoted (code) | keep |
| `s5_recovery_table.py` band table (+28 lines) | pandas | It already uses pandas | inferred | keep |

There are no new mechanisms beyond these. The round-1 audit (corruption injectors, import-linter, sklearn metrics) still applies unchanged.

## Feedback: planner
1. **AC5.8 (plan.md l.496-506).** The six raw heuristics are not enough to say "which types test anything beyond one-line statistics" (ruling A). Add two label-free, task-normalised statistics to the AC5.4 set and require the bands to use the best over all eight:
   - **H7 `length_vs_task`:** `length_steps / median(length_steps of the episode's task_name)`, computed over the candidate subset, both directions reported.
   - **H8 `post_release_share_vs_task`:** the share of steps after the last gripper +1→−1 transition (T if none), minus the task median.

   Expected (measured): `time_shift` about 0.85, `detour` about 0.96, `gripper_release` about 0.96.
2. Update R-items and the report guidance so the headline says no type is in band (c), with `time_shift` the hardest at about 0.86. Record that as the owner-approved limitation alongside B and C.
3. Fix the stale S5 "Verify by" line (l.508-510 still says `v2/candidate`).

## Feedback: executor
1. Implement H7 and H8 in `vlapoc/methods/heuristics.py`, reading only through `open_for_method` (`task_name` and `length_steps` from the catalog; actions through `iter_episodes`). Run `python -m vlapoc.methods.heuristics --subset v3/candidate`, score both through the `score` CLI (scratch lineage copy, or the real store if the AC requires MLflow runs; record the sha before and after), then regenerate `s5_v3_bands.csv` and the AC5.8 table. Expect `time_shift` H7 (lowest) ≈ 0.85, `detour` H7 ≈ 0.96, `gripper_release` H8 ≈ 0.96, within seed-free tolerance because these are deterministic. My probes are `data/scratch/gt_an3/lenres.py` and `grip.py`.
2. Re-run `scripts/verify_method_boundary_s3.py` (the static scan must still pass with H7 and H8 in `heuristics.py`) and commit its refreshed `s3_verify.json`. The committed one predates the trim (child id `3f43df489885` vs `dbb9f38a473f`).
3. Update the runbook's detectability paragraph (AC6.6) to match the new bands.
4. R12, if the summarizer won't: add the data+lineage `changelog.md` entry and correct `report.html:1888`.

## Non-blocking
- Make the refusal wording for supersession fences say "fenced: superseded by v3" instead of "not blind to its edits".
- `du -sh v3` keeps falling as the slack is reclaimed. Quote byte sums (7.33 GB = 6.82 GiB) in the report, not `du`.
