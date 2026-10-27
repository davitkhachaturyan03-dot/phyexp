VERDICT: REWORK
ROUTE: executor
ROUND: 2

<!-- Saved verbatim by the orchestrator from the analyzer's returned text: the analyzer's own
     write was refused by the harness ("Subagents should return findings as text"). -->

## Summary (3–5 lines)
- **Round 1 fixed:** all 9 items are fixed, and I re-measured each one.
  - Every `verify_dashboard.py` mode passes, including `--render` 4/4. `verify_runbook.py` passes 9/9, and the regression set passes.
  - In a real browser: the strip chart is 463x240 px, a row click changes the selected rollout, and `/clusters` no longer opens a blank page. The banner is one line.
  - The null calibration reproduces: Welch 0.0586 (41/700). My own single-cell simulation gives 0.054.
  - `data/lineage/mlflow.db` sha256 is unchanged (`306af412…bde79`).
- **New, small, executor-side:**
  - The runbook's Step 2 copy list omits `data/runs/train-smolvla-2step-persubset-v0ref-20261005T012854Z/`. Step 3's `--local-smoke` and probe both load that checkpoint, so a fresh server session stops at Step 3.
  - The dashboard's lineage command `python -m vlapoc.lineage trace <id> --lineage-root <store>` is rejected by the CLI for any non-default store. That covers the demo store and the Mac "results back" store.
- **Dedupe side effect:** acceptable and honestly disclosed. Two presentation suggestions are under non-blocking.

## Rebuttals received
None. The executor addressed every round-1 item rather than disputing it.

## Round-1 items
| # | Round-1 item | Status | My evidence |
|---|---|---|---|
| E1 | Bad-vs-harmful strip chart was 18 px; scatter and heatmap squeezed | ✅ fixed | Live, pane width: SVG `path.background` is `M0,0h463v240h-463Z` (measured). `--render` 5.10a: Clusters 394x355, Bad vs harmful 839x240, heatmap axes 566x611. The re-captured `bad_vs_harmful.png` and `clusters.png` show spread points and readable plots (looked at). |
| E2 | Rollouts row click did nothing | ✅ fixed | Live: clicking the `libero_10/8/init001/seed1` row made "Selected rollout" read `libero 10/8/init001/seed1`, failure type `timeout partial 1 of 2` (page text, measured). `--render` 5.10b also passes. |
| E3 | `pages/` auto-nav opened blank pages | ✅ fixed | Folder renamed `dashboard/views/`. The sidebar shows only the radio. `http://localhost:8791/clusters` now renders the app (screenshot). |
| E4 | Runbook Step 2 did not repoint the store on the server | ✅ fixed | Walked it on a scratch copy of the demo store: dry run, then real (`runs rewritten 43, experiments 1`). The read check printed `score-post_release_share_vs_task ['version','version_id','subset']` and `attribute-demo-smoke (40, 30)` (measured). `verify_runbook` 8.7 passes. |
| E5 | Runbook Step 3 probe exited with an error | ⚠ partly fixed | Now `--track --smoke`, and the flags parse (8.1b, 23/23). It is still not runnable on a fresh server: see new item N1. |
| E6 | Hand-rolled Welch | ✅ fixed | `discrepancy.welch` calls `scipy.stats.ttest_ind(...).confidence_interval` (read). 9.7 reports max \|diff\| 0.0. |
| E7 | Re-capture and look at the screens | ✅ fixed | I opened `bad_vs_harmful.png`, `clusters.png` and `rollouts.png`; they are consistent with the live app. |
| P1 | Reading must use Welch, say "uncorrected", add Holm | ✅ fixed | `--discrepancy` 10/10. The sentence carries "Uncorrected across 7 types × 5 metrics. Holm across the 5 metrics: …" (read in `estimated_reading`, seen on the page). Calibration: Welch 0.0586, bootstrap 0.0929, Holm 0.0071 (measured by my rerun). With 700 non-independent cells (SE about 0.009), 0.0586 is consistent with nominal. |
| P2 | min_sup 19 > 13 decision | ✅ fixed (option a) | `--clusters` 2.8: the Round A job passes `--min-sup-fraction 0.01`, giving min_sup 10. The page states the size and the quintile caveat (screenshot). |
| P3 | Plain-language layer, §3.4 reachable | ✅ fixed | One-line banner with an expander. Metric names are plain ("Success within 30 s"), with the functional caption on the same page. Quadrants read "looks bad, hurts" and so on. Recorded synthetic verdicts start "SYNTHETIC:". `--app` 5.4/5.9 report no offending labels. |
| P4 | Real-browser rendering AC | ✅ fixed | `--render` 4/4, reproduced. |
| P5 | Runbook flag parsing, server-side relocation | ✅ fixed | 8.1b and 8.7 pass. |
| NB | F16 shared control, positive-control size, dedupe, `st.metric` truncation | ✅ fixed | The sentence text is in `planted_arms.py:212-233` (read). Dedupe: 2.9 passes. Truncation: 5.10c reports `truncated []`. |

## Acceptance criteria (round 2)
| Story | AC | Claimed | Reproduced? | Evidence |
|---|---|---|---|---|
| S1–S4, S9, S10 | all, plus AC2.8, AC2.9, AC9.7 | ✅ | ✅ | `--results` 12/12, `--clusters` 14/14, `--planted` 7/7, `--discrepancy` 10/10, `--round-b` 13/13, `--planted-arms` 8/8 |
| S5 | 5.1–5.8 and new 5.9, 5.10 | ✅ | ✅ | `--app` 10/10, `--render` 4/4, plus my live-browser checks above |
| S5 | 5.5 "lineage link per cluster" | ✅ | ⚠ | The MLflow link is present. The shown `trace` command fails for non-default stores (N2). |
| S6, S7 | all | ✅ | ✅ | `--demo` 6/6, `--dropin` 4/4 |
| S8 | 8.1–8.7 | ✅ | ✅ by verifier; ⚠ walkthrough | 9/9. Step 3 needs an input that Step 2 does not copy (N1). |
| Regression | — | ✅ | ✅ | `verify_attribution` 32/32. `verify_training_attribution` modes: `--blind`, `--functionals` 18/18, `--pipeline` 9/9, `--step4` 13/13, `--recovery` 11/11, `--names` 13/13, `--groups` 9/9, `--integration` 12/12. mlflow.db sha256 unchanged. |

## Dedupe side effect (asked by the orchestrator)
- **What it does:** the first row in lens order is kept and identical member sets become its `equivalent_definitions` (`clusters.dedupe`, read). Slices are ordered by metric column (f30, rmst60, auc_expected, hrt60_lin, censored60). So when the planted HRT slice has exactly the same members as an AUC slice, the kept id is `slice:auc_expected:harm:k`.
- **Judgement: acceptable, and nothing downstream is wrong.**
  - With identical members, every per-metric mean, CI and reading is identical, so `expected_by_metric` for a Round B `drop_cluster` arm does not depend on which id is kept (inferred from the code path: readings come from the members' scores).
  - The page filters through the equivalents, so the cluster still appears under the HRT metric, with the "also described as" line.
  - In the demo the merge is also the truth: the synthetic generator moves AUC together with HRT (R2-F1).
- **Two presentation weaknesses (non-blocking):**
  - The kept id and `target_metric` name AUC, which a reader takes as "the metric this was found for".
  - When the equivalent has the same definition text, the page prints "also described as: <the same text>" (seen on `clusters.png` for `smoothness > 0.1052 AND gripper toggles (3, 4]`). Showing "also found for: <metric>" would read better.

## PoC scope assessment
- **Covered:** unchanged from round 1. §2.7 p.5 both views, §4.2, §4.5 p.11, §3.4, §9.2 p.19, §10 p.20.
- **Over-built:** the presentation layer added `dashboard/words.py` and DevTools measurement in `dashboard_screenshots.py`. Both are proportionate to ACs the plan now requires.
- **Silent divergences:** none.

## Overcomplication & reinvention audit (round-2 additions)
| Mechanism | OSS alternative | Covers? | Evidence | Recommendation |
|---|---|---|---|---|
| `discrepancy.welch` | scipy (now used) | yes | read; 9.7 diff 0.0 | done |
| Holm (`verdict.holm`, existing) | `statsmodels.multipletests` (not installed) | — | existing repo function, reused | keep |
| `clusters.dedupe` (about 20 lines, byte-key of masks) | none needed | — | inferred | keep |
| `dashboard/words.py` mapping | none (domain names) | — | inferred | keep |
| `dashboard_screenshots.py` DevTools client | Playwright (not installed) | partly | inferred | keep (it now carries the 5.10 measurements) |

## Feedback: planner
None.

## Feedback: executor
1. **N1: runbook Step 2 copy list misses the checkpoint Step 3 uses.**
   - `verify_training_attribution.py:1382` makes `local_smoke` default to `train_ref = "0a33bc75bb71433ca2def5220014dde0"`, and Step 3's probe uses the same run.
   - `runs.eval_run` loads `REPO_ROOT / tags["checkpoint.path"]` (`runs.py:723`). That run's output is `data/runs/train-smolvla-2step-persubset-v0ref-20261005T012854Z` (sqlite params; 1.5 GiB, `du`).
   - `SKILL.md` lines 57-59 do not list `data/runs/`. On a fresh server, both Step 3 commands will fail on a missing checkpoint (inferred from source; the Mac passes only because the directory exists here).
   - Fix: add that directory, by name with its size, to Step 2's copy list. Add an existence check (`ls data/runs/train-smolvla-2step-persubset-v0ref-20261005T012854Z/lerobot/checkpoints/last/pretrained_model`) under the preflight STOP AND ASK.
   - Optionally, make `verify_runbook` check that every input path a "verified locally" command needs appears in the copy list.
2. **N2: the dashboard's lineage command fails on any non-default store.**
   - `dashboard/common.py:229-230` builds `python -m vlapoc.lineage trace <id> --lineage-root <store>`.
   - The CLI rejects it (measured): `python -m vlapoc.lineage: error: unrecognized arguments: --lineage-root data/lineage_demo`. `trace` takes only `run_id`.
   - This affects the demo and Step 14's `data/lineage_from_server`.
   - Fix, either: give the `trace` subparser a `--lineage-root` (if the function already accepts one), or show a command that works. Add a check in `--app` that runs the displayed command and expects rc 0.
3. **Re-run and restore.** Re-run `verify_runbook.py`, `verify_dashboard.py --app` and the regression. Restore the tracked artifacts the verifiers rewrite.
   - Note: `plot_heatmap` now writes `<png>.axes.json` beside every heatmap. `verify_training_attribution.py --groups` therefore leaves an untracked `tracker/features/2026-10-20-training-attribution/artifacts/s7_heatmap_planted_SYNTHETIC.axes.json` (measured; I removed it).
   - Either write the sidecar only when asked (e.g. a flag the render check uses), or commit it. A tracked file should not appear on every run.

Scope of the next round: only items 1–3. I will re-check just these.

## Non-blocking
- Dedupe presentation: see the section above.
- Bad-vs-harmful legend: "looks bad, hurts" and "looks bad, no detectable harm" are two similar blues (screenshot). A more distinct pair would read better.
- Recorded pattern text still mixes raw names: "Estimated helpful on Success within 30 s, rmst60, censored60" (clusters.png). It is recorded output, so the fix belongs in `pattern_of`, not the app.
- R2-F3 (row-click geometry depends on Streamlit 1.66.1): re-check after any Streamlit upgrade.
- Housekeeping: I restored the tracked artifacts the verifiers rewrote (ids and timings only), removed the untracked `axes.json` and my scratch store copy, stopped the port-8791 server, and closed the browser. The tree matches its start (only the owner's 4 files modified).
