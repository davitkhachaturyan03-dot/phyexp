VERDICT: REWORK
ROUTE: planner+executor
ROUND: 1

<!-- Saved verbatim by the orchestrator from the analyzer's returned text: the analyzer's own
     write was refused by the harness ("Subagents should return findings as text"). -->

## Summary (3–5 lines)
- **Reproduced (measured):** all 9 `verify_dashboard.py` modes, `verify_runbook.py` (6/6), `verify_attribution.py` (32/32) and all 8 `verify_training_attribution.py` modes pass. Before and after, `data/lineage/mlflow.db` sha256 = `306af4123fda7e635892d459a2e34008ba1cb62f6c4fce152c546e0da7bdde79`.
- **Defects the verifiers do not catch (measured in a live browser at 1400x1000):**
  - The Bad-vs-harmful strip chart's plot area is 688x18 px, so every point sits on 0.
  - Ticking a Rollouts table row leaves the selected rollout unchanged.
  - The auto-generated sidebar links (`/clusters` etc.) open blank pages.
- **Runbook gaps:**
  - Step 2 never repoints the Mac-made store on the server.
  - Step 3's probe command exits with `use --track for train:<run_id>`.
- **Statistics:** the §9.2 per-type readings use the plan's percentile bootstrap, which over-calls (9.9% vs 5% under the null, my own simulation). The plan must switch to Welch, a one-line scipy call, and say "uncorrected".
- **Holding up:** blindness, answer-key isolation, fail-closed provenance and the "never harmless" wording. Scope is right for the PoC milestone.

## Rebuttals received
None (round 1). Findings F1–F23 are the executor's own; each is judged below.

| From | Point | Their evidence | Conceded? | Why |
|---|---|---|---|---|
| executor F1 | Bootstrap anti-conservative; built as planned, Welch stored beside it | probe: 8.5% vs 4.9% | Agree with the finding | My independent simulation (1000 trials, 13 vs 869, normal null): bootstrap 0.099, Welch 0.054 (measured, `data/scratch/analyzer_dash/calib.py`). The plan's method is the defect, so this routes to the planner. |
| executor F11 | Plan said 33 jobs for `--matched-random`; the chain gives 45 | dry run | Yes | Reproduced via `verify_runbook` AC8.2. Plan arithmetic error, disclosed in the skill. |
| plan S10 / brief | Oracle (91 episodes) counts as the positive control | brief, owner Q2: "all-91 oracle arm as positive control" | Yes (owner ruling) | Non-blocking caveat below: an oracle confirmation does not show power at n = 13. |

## Acceptance criteria
| Story | AC | Claimed | Reproduced? | Evidence |
|---|---|---|---|---|
| S1 | 1.1–1.6 | ✅ | ✅ | `--results` 12/12 (measured). Verifier rewrites of tracked artifacts were run ids and timings only, restored with `git checkout`. |
| S2 | 2.1–2.7 | ✅ | ✅ | `--clusters` 12/12 |
| S3 | 3.1–3.4 | ✅ | ✅ | `--planted` 7/7 |
| S4 | 4.1–4.5 | ✅ (45 not 33) | ✅ | `--round-b` 13/13. The `ceil(2n/3)` rule is guarded by `n >= 2` (`confirm.py:44`, read). |
| S9 | 9.1–9.6 | ✅ | ✅, but method flaw | `--discrepancy` 9/9. Readings over-call (F1, above). |
| S10 | 10.1–10.6 | ✅ | ✅ | `--planted-arms` 8/8. Counts 129/33, 108/27, 162/36 re-measured by `verify_runbook`. |
| S5 | 5.1, 5.3–5.8 | ✅ | ✅ (AppTest) | `--app` 10/10 |
| S5 | 5.2 Rollouts selection | ✅ | ⚠ only through the selectbox | Live: ticking the `libero_10/8/init000/seed0` row left "Rollout (or click a row above)" on `libero_spatial/0/init000/seed0` (page text, measured). Cause: `rollouts.py` builds `st.selectbox(..., index=clicked, key="rollout_pick")`. With a `key`, the widget keeps its session value and `index` is not applied again (inferred from Streamlit widget semantics; the symptom is measured). The page label promises a click that does nothing. |
| S5 | 5.5 heatmap and scatter shown | ✅ | ⚠ shown but squeezed | Scatter plot area is 137x190 px (measured from the SVG `path.background`). The long functional axis titles eat the space. |
| S5 | Bad-vs-harmful chart (page deliverable) | — | ❌ | SVG `path.background` = `M0,0h688v18h-688Z`. The data has spread (`z__f30` std 1.0, max 3.68, measured). Cause: a very long y-axis title (`"harm z-score: " + axis_label`) plus rotated x labels inside a fixed `height=360` with Streamlit's `fit-y` autosize (measured class `fit-x fit-y`). |
| S5 | Navigation | — | ❌ | `dashboard/pages/` sits next to `app.py`. Streamlit docs (quoted): "Streamlit will identify each Python file within it as a page". The sidebar shows `app / bad vs harmful / clusters / ...`, and `http://localhost:8791/clusters` renders blank (measured screenshot). |
| S6 | 6.1–6.5 | ✅ | ✅ | `--demo` 6/6. Store not rebuilt by me (490 s); the existing store was verified. |
| S7 | 7.1–7.3 | ✅ | ✅ | `--dropin` 4/4 |
| S8 | 8.1–8.6 | ✅ | ✅ by verifier; ❌ by walkthrough | AC8.1 only checks `--help`. Two blocking runbook defects are below. |
| AC8.2b | v3 first run | ✅ | ✅ | `--pipeline` 9/9, v3 116 jobs / 24 trainings |

## PoC scope assessment
- **Covered:**
  - §2.7 p.5, both views (failure type, influential episodes with video and source file; ranked list aggregated by catalog column).
  - §4.5 p.11 heatmap and scatter; §4.2 groups; §3.4 declared functionals.
  - §9.2 p.19 discrepancy: estimated half plus causal arms with bounded wording.
  - §10 p.20 "Verification & demo" (removal tests, dashboard linking).
  - I re-read all of these quotes from the PDF; they are verbatim.
- **Missing:** nothing the spec or brief requires. But "present to stakeholders" (brief) is undermined by the broken chart and by raw provenance and functional strings (orchestrator observations 2 and 3, below).
- **Over-built (minor):**
  - `scripts/dashboard_screenshots.py` (hand-rolled DevTools client, not in the plan).
  - Duplicate navigation (the radio plus the accidental auto-nav).
  - Nothing else; the 33-training Round B and the MPS smoke in the demo were owner-requested.
- **Silent divergences:** none found. F2 (min_sup 19 > 13) is disclosed in the log but not in the runbook or the dashboard (planner item 2).

## Overcomplication & reinvention audit
| Mechanism (file:line) | Existing OSS alternative | Covers our need? | Evidence | Recommendation |
|---|---|---|---|---|
| `discrepancy.welch_diff_ci` (`discrepancy.py:98-111`) | `scipy.stats.ttest_ind(a, b, equal_var=False).confidence_interval(0.95)` | Yes | **measured:** identical to 1e-15 on a random draw (-0.42001, 0.74405 both ways) | **replace** (one call, drop about 10 lines) |
| `boot_diff_ci` for the reading (`discrepancy.py:89-95`) | `scipy.stats.bootstrap` / `permutation_test` | Yes, but the method itself is the problem (F1) | measured calibration above | **replace the reading's method** (planner) |
| `repoint_lineage_store.py` (70 lines) | `mlflow-export-import` | No: per the official MLflow docs as relayed in search results, it "does not preserve original IDs or timestamps" (quoted, secondary). Our lineage stores run ids in params (`attribute_run_ids`, `score_run_id`, `cluster_run_id`), so new ids break the graph. MLflow docs show no "move store" command (searched twice). | quoted (search results) + inferred | **keep** (a symlink to the same absolute path is a zero-code alternative, inferred, more fragile) |
| `clusters.bin_spec` (zero-inflated minimum bin, merge below min_sup) | sklearn `KBinsDiscretizer(strategy="quantile")`, `pd.qcut(duplicates="drop")` | Partly: neither isolates the zero mass as its own bin or merges by support (inferred from their documented params) | inferred | keep |
| `results.provenance` | MLflow has no provenance label | — | inferred | keep |
| `vlapoc/video.py` | PyAV (already used), the same tool | — | read | keep |
| `confirm.status_of`, `planted_arms.statuses` | none (spec-specific rules) | — | inferred | keep |
| `dashboard_screenshots.py` (88 lines, DevTools protocol) | `chrome --headless --screenshot`; Playwright (not installed) | Mostly (inferred) | inferred | simplify, non-blocking |
| Radio page switcher + `dashboard/pages/` | `st.navigation` / `st.Page` | Either works; the problem is having both | measured (blank auto pages) | **simplify:** rename the folder |

Checked with nothing better found: Streamlit choice (plan's prior art stands), sliceline adoption (still used through its `Slicefinder` API, `clusters.py:273-287`), Fisher via scipy (`recovery.py`).

Searches run:
- "mlflow move tracking store change artifact_uri existing runs absolute path sqlite"
- "mlflow-export-import copy runs between tracking servers artifacts"
- Streamlit pages-directory doc fetched.

## Feedback: planner
1. **S9 reading method (F1, AC9.1/9.2 wording).**
   - Problem: `estimated_reading` must not use the percentile bootstrap at 13 vs 869. Measured under the null it is 9.9% vs 5% nominal; Welch is 5.4%.
   - Fix: amend the plan so `reading` and `estimated_reading` use the Welch interval (`scipy.stats.ttest_ind(..., equal_var=False).confidence_interval()`).
   - Also require the sentence and the page to say "uncorrected across 7 types x 5 metrics", with about 1.75 false readings expected under the null at 5%. Optionally Holm across the 5 metrics per type, in the spirit of PoC §4.4 p.10, "correct for it (e.g. Bonferroni or Holm across metrics)".
   - AC: under the stored null calibration, the reading's false-call rate is ≤ 6%.
2. **F2 (min_sup 19 > 13): make a decision and record it.**
   - Problem: at the defaults no slice can isolate a single planted type, so S3 precision is ≤ 13/19 by construction.
   - Fix, either: (a) the Round A `clusters` job runs with `--min-sup-fraction 0.01` (min_sup 10); or (b) keep 19 and require the Clusters ground-truth panel and the runbook Step 8 to state the cap.
3. **Plain-language presentation layer (brief: "present the project to any future stakeholders").** Add an AC that keeps §3.4 intact while making pages readable to a non-specialist:
   - a one-line banner, with the reasons in an expander (they already appear on the Overview page);
   - short metric names ("Success within 30 s", "Speed vs human (HRT)") with the declared functional in a tooltip, help text or caption;
   - quadrant display names in words ("looks bad, hurts"), not snake_case;
   - SYNTHETIC in the sentence/status column of recorded synthetic verdicts (generated by `discrepancy_verdict_run`/`confirm_run` when `synthetic=True`, so the app still shows them verbatim).
   - AC5.4's scan should accept "functional reachable from the element" rather than "embedded in the label".
4. **A rendering check, not just AppTest.** Add an AC that each chart's plot area is at least about 200 px tall at 1400 px wide (measurable from the SVG `path.background` in headless Chrome, which the repo already drives). Also require that a real row click on Rollouts changes the selected rollout. AppTest cannot see either defect (F22).
5. **Runbook AC8.1 is too weak.** Add a check that every flag used in a fenced command is accepted by the target parser (argparse `parse_known_args` with no unknown flags). Add a "store relocation" step on the server side (see executor item 4).

## Feedback: executor
1. **`dashboard/pages/bad_vs_harmful.py`, the `alt.Chart(...).properties(height=360)` strip chart.**
   - Problem: the plot area is 18 px (measured SVG `path.background` `M0,0h688v18h-688Z`).
   - Fix: shorten the y title to e.g. "harm z (positive = harmful)", put the functional in the chart title or caption, use short x labels or a horizontal layout (y = group, x = z).
   - Re-measure: plot height ≥ 200 px. Same treatment for `clusters.py` `scatter()` (137x190 px measured) and the matplotlib heatmap, whose long functional tick labels take most of the figure (screenshot).
2. **`dashboard/pages/rollouts.py`, `st.selectbox("Rollout (or click a row above)", ..., index=clicked, key="rollout_pick")`.**
   - Problem: the row click does not take effect (measured).
   - Fix: write the clicked id into `st.session_state["rollout_pick"]` before the selectbox renders (or use an `on_change`/selection callback), or drop the "click a row" promise.
   - Verify in headless Chrome: click a row, then the page text shows that rollout_id under the selectbox.
3. **Auto multipage navigation.** Rename `dashboard/pages/` to e.g. `dashboard/views/` (update the imports in `app.py` and `verify_dashboard.py`). Re-check that the sidebar shows only the radio and that `/clusters` no longer exists.
4. **Runbook Step 2 (`.claude/skills/server-run/SKILL.md`): repoint the store on the server right after copying it.**
   - Measured: a copy of `data/lineage_demo/mlflow.db` with its prefixes moved to a non-existent root gave `MlflowException: Failed to download artifacts from path 'attribution_scores.parquet'` from `results.score_table`.
   - The real store's experiment location is `file:///Users/davit_khachaturyan/.../data/lineage/mlartifacts` (sqlite query, measured), and `runs.experiment_id` (`runs.py:133-142`) reuses an existing experiment's location. So on the server, every new stage run will try to log artifacts under `/Users/...` (inferred from source), and `discrepancy` cannot read the ground-truth score runs' `recovery` JSON.
   - Fix: add `scripts/repoint_lineage_store.py --root data/lineage --old <Mac repo>/data/lineage --new $PWD/data/lineage` (dry run first, then real) in Step 2. Add a read check (e.g. `results.score_table` of an existing attribute run, or `lineage trace` of a GT score run) as a STOP AND ASK point, and correct the sentence "new runs on the server write server paths".
   - Step 14 then repoints once from the server prefix. Its prose names `data/eval/returns-*` and `rollouts.parquet`, but the rsync commands don't copy them. They are not needed, since outputs are MLflow artifacts (`runs.py:250-257` `log_artifact`), so fix the prose.
5. **Runbook Step 3 probe.**
   - Problem: `eval_smoke.py --policy train:<run> ... --out data/scratch/tp` exits with `untracked mode supports noop and hf: policies; use --track for train:<run_id>` (measured).
   - Also, the `--local-smoke` train run lives in the scratch store `data/scratch/training_attribution/smoke/lineage` (`verify_training_attribution.py:1396-1399`), not in `data/lineage`.
   - Fix: give a command that runs, e.g. `--track --smoke --name eval-smoke-tp1 --lineage-root data/scratch/training_attribution/smoke/lineage`, or an `hf:` policy. Mark which form was executed locally.
6. **`discrepancy.py:98-111`:** replace the hand-rolled Welch with the scipy call (measured identical), as part of planner item 1.
7. After the fixes, re-run all `verify_dashboard.py` modes, `verify_runbook.py` and the regression set. Re-capture `artifacts/screens/*.png` and **look at them**; `bad_vs_harmful.png` already showed the flattened chart.

## Non-blocking
- **Orchestrator observations, my verdict:**
  - (1) Flattened chart: confirmed, measured, root cause found; blocking (executor item 1).
  - (2) Raw provenance in the banner: confirmed (page text). The reasons already appear in an Overview table, so a one-line banner loses nothing (planner item 3).
  - (3) Raw functional and snake_case labels: confirmed. Fold into planner item 3; §3.4 is satisfied if the functional stays reachable on the element (caption or tooltip).
- **F16:** one clean random draw is shared by all 7 types and all 3 seeds, so "k of 3 seeds" is not 3 independent replicates of the control (inferred). Say so in the S10 sentence, or recommend `--random-draws 2` in the runbook cost table.
- **Positive control:** the oracle (91 episodes, owner-approved) confirming harm does not show the test has power at 13 episodes. The union-CI bound in the sentence carries that honestly; consider adding "the positive control removed 91 episodes".
- **F14:** uncorrected cluster readings become `expected_by_metric`. That can produce "not confirmed" or "refuted" but not a false "confirmed" (inferred). Acceptable as labelled.
- **Duplicate clusters:** in the demo, `scene = KITCHEN_SCENE3` and `task_name = KITCHEN_SCENE3_turn_on_the_stove...` are the same 48 episodes, listed twice. Consider de-duplicating identical member sets.
- **`st.metric` truncation:** failure type ("timeout_no…") and T ("none (did n…") are cut off on Rollouts (screenshot).
- **F23:** NumPy/Accelerate warnings are noisy; harmless.
- **Housekeeping:** I restored the 21 tracked artifacts the verifiers rewrote (ids and timings only) with `git checkout`. The tree matches its start state (only the owner's 4 files modified), and the port-8791 server is stopped.

Sources:
- [Streamlit pages directory docs](https://docs.streamlit.io/develop/concepts/multipage-apps/pages-directory)
- [mlflow-export-import on PyPI](https://pypi.org/project/mlflow-export-import/)
- [MLflow: Migrate from File Store to Database](https://mlflow.org/docs/latest/self-hosting/migrate-from-file-store/)
