VERDICT: PASS
ROUTE: none
ROUND: 3

<!-- Saved verbatim by the orchestrator from the analyzer's returned text: the analyzer's own
     write was refused by the harness. -->

## Summary (3–5 lines)
Round 3 resolves all of analysis-r2's items:
- H7 (`length_vs_task`, both directions) and H8 (`post_release_share_vs_task`) are label-free. They read only through `open_for_method` (catalog `length_steps` and `task_name`, the gripper dimension via `iter_episodes`), and their directions were fixed in a commit before scoring.
- Their AUROCs reproduce from the stored tables: 0.848, 0.960, 0.964.
- The bands now put six types in (a) and `time_shift` in (b), and the AC5.9 headline states this plainly as an owner-approved limitation under ruling A.
- The S3 scan, the fence wording, the GT-R6 rule and the R12 data+lineage correction are all in place.
- The S2–S5 verifiers pass, and the real `mlflow.db` was untouched by my checks.

## Rebuttals received
| From | Point disputed | Their evidence | Conceded? | Why / why not |
|---|---|---|---|---|
| none | – | – | – | All round-2 feedback was implemented as specified. |

## Round-2 items
| Item | Status | Evidence |
|---|---|---|
| Planner 1: H7/H8 in AC5.4/5.8 | ✅ | `c1768bb`. The AC5.4 set is now H1–H8. |
| Planner 2: honest headline (AC5.9, R13, limitation D) | ✅ | plan.md l.538-546. "No type is in band (c)"; comparison per type against each type's best heuristic. |
| Planner 3: S5 verify line names v3 | ✅ | plan r3 change 3. |
| Executor 1: implement and score H7/H8 | ✅ | `heuristics.py` `compute` and `scan_episodes`. 3 screen + 3 score runs. `s5_v3_r3_bands.csv`. |
| Executor 2: S3 re-run, refreshed `s3_verify.json` | ✅ | My re-run: `all S3 checks pass: True`. The scan covers `heuristics.py` with H7/H8 (0 violations, 6/6 planted violations caught). My run leaves `s3_verify.json` byte-identical to the committed file (git clean). |
| Executor 3: runbook detectability paragraph | ✅ | `server-runbook.md` GT-R6: per-type table to beat, rule "compare per type, not against chance", pooled recall@91 not comparable, limitation under ruling A. |
| Executor 4 / R12: data+lineage correction | ✅ | `2026-09-28-data-lineage-layer/changelog.md` has a 2026-10-17 entry with evidence. `report.html:1888` is corrected and l.1993 has a dated "Changes since first release" line. |
| Non-blocking: fence wording | ✅ | `lineage.py:889-892`. v2 is refused with "is fenced: superseded by v3 (002a68e4caf2); use the superseding version". v1 keeps `fenced=True, offending=['episode_index']`. Reproduced in S3 and S4. |
| Non-blocking: quote byte sums, not `du` | carried to the summarizer | – |

## Acceptance criteria
I re-ran everything into `data/scratch/gt_an4/` (logs in `logs/`). Main `mlflow.db` sha256 was `306af4123fda7e63` before and after, and `git status` on `tracker/` was clean afterwards.

| Story | AC | Claimed | Reproduced? | Evidence |
|---|---|---|---|---|
| S2 | regression (`lineage.py` changed) | – | ✅ | `all S2 checks pass: True`. |
| S3 | 3.1–3.5 (r3, v2 row added) | ✅ | ✅ | v2 row REFUSED with the supersession wording. `all S3 checks pass: True`. |
| S4 | 4.1–4.8 on v3 | ✅ | ✅ | `all S4 checks pass: True`. AC4.8 refusal uses the new wording. |
| S5 | 5.1–5.3, 5.5 | ✅ | ✅ | `all S5 verifier checks pass: True`. |
| S5 | 5.4 (H7/H8 values) | ✅ | ✅ | Recomputed from `data/eval/groundtruth/heuristics_v3_r3/*.parquet` with sklearn (`data/scratch/gt_an4/auroc.py`). Every per-type AUROC and recall@13 in the log's H7/H8 table matches, e.g. `time_shift` 0.848 / 0.462, `detour` 0.960 / 0.154, `idle_padding` 0.950 / 0.308, `gripper_release` 0.964 / 0.000. |
| S5 | 5.6 | ✅ | ✅ (artifact) | `s5_r3_trace_check.txt`: 3 score runs tagged with v3 `002a68e4caf2` and `labels_sha256` `3e14922a63cf`. `git_dirty false`. The trace reaches v3. |
| S5 | 5.8 bands | ✅ | ✅ | `s5_v3_r3_bands.csv`: (a) for action_noise, detour, dropped_frames, gripper_release, idle_padding, instruction_swap; (b) for time_shift at 0.848. This matches my recomputation. |
| S5 | 5.9 headline | ✅ | ✅ | Log and runbook both state no type is in band (c), time_shift hardest at 0.848, compare per type against the best heuristic, limitation under ruling A. |
| S6 | 6.6 | ✅ | ✅ (read) | GT-R6 table and rule present. v3 named throughout. S6 code was not touched in round 3, so it was not re-run (it passed in round 2). |

**Directions fixed a priori:** `DIRECTIONS` and the H7 two-way and H8 highest-first choices are in `b8cc799`, committed before the scoring commit `9778298` (`git log`). Verified.

## PoC scope assessment
- **Real-store score runs.** These are 3 screen and 3 score runs (`e5499b65` → `306af412`). AC5.6 requires the score runs to trace to v3 in the real lineage, and rounds 1 and 2 did the same. Only the three new heuristics were run (`--only`), so no duplicate H1–H6 runs were created. Accepted.
- **Recall@13 vs AUROC.** Reporting both is correct, and the explanation is verified (measured). Under H8 the 13 `gripper_release` episodes rank [16, 17, 20, 23, 26, 27, 28, 29, 39, 48, 50, 78, 97] of 882. About 15 clean episodes sit above them, with task-relative post-release shares of 0.24–0.57. So high AUROC means "ranked high", while recall@13 of 0 means "not in the very top". Both belong in the report. See non-blocking item 1.
- **Covered:** §2.1 item 4, §2.5 (minus viewpoint, which the owner ruled out), §8.7, §8.10/§4.5 hypotheses as labels, the §10 "Ground truth" deliverable, with honest per-type baselines.
- **Missing:** nothing in scope. Policy harm is pending server (runbook GT-R1–R5).
- **Over-built:** nothing. H7/H8 add about 40 lines to an existing module and share one episode pass with H5.
- **Silent divergences:** none.

## Overcomplication & reinvention audit
| Mechanism (file:line) | Existing OSS alternative | Covers our need? | Evidence | Recommendation |
|---|---|---|---|---|
| H7 task-median ratio (`heuristics.py` `compute`) | pandas `groupby().transform("median")` | It already uses it | quoted (code) | keep |
| H8 post-release share and `minus_task_median` | numpy and pandas | It already uses them | quoted (code) | keep |
| Supersession branch in `assert_blind` (`lineage.py:889`) | – | 4 lines on the existing fence | quoted (code) | keep |

There are no new mechanisms beyond these. The round-1 audit still applies.

## Feedback: planner
None.

## Feedback: executor
None.

## Non-blocking (for the summarizer)
1. **Add best recall@13 to the per-type "table to beat".** Recall@k is the PoC §2.1 headline ("recovered 90% of the injected bad episodes"), and the best-AUROC heuristic is not always the best-recall one. The values are already in `artifacts/s5_v3_r3_recovery_heuristics.csv`; for example, `gripper_release`'s best AUROC (0.964) comes with recall@13 of 0.000.
2. Quote dataset sizes as byte sums (v3 7.33 GB = 6.82 GiB), not `du -sh`. On a fresh copy `du` keeps falling as block slack is reclaimed (measured: 7,775,640 → 7,456,304 KiB).
3. The `heuristics.py` module docstring's CLI example still says `--subset v2/candidate`. Cosmetic, since v2 is fenced and would be refused.
4. Remind the owner in the report that v2's physical copy (`data/lerobot/libero_spatial_10__v2`, about 6.8 GiB) is kept for them to delete.
