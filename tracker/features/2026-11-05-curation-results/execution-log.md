# Execution log: curation headline results in the dashboard

## Round 1: 2026-11-05

Executor, branch `curation-results`, started at HEAD `4aeff57`. Evidence labels: **measured** (a command run in this session, real output) or **inferred**.
Commits (in order): `d078bfb` log started, S1 `step4` status/headline, S2/S3 loaders and demo Round A, S4/S5 page and verifier. Verifier outputs for this feature are in
`artifacts/verify_dashboard_curation.json`, `verify_dashboard_dropin.json`, `verify_dashboard_render.json`, `render_measurements.json`; the S1 fixtures in
`artifacts/s1_*`; the screenshot in `artifacts/screens/curation_results.png`; the Round A dry run at HEAD in `artifacts/dry_run_head.txt`.

How to view the page: `.venv/bin/streamlit run dashboard/app.py -- --lineage-root data/lineage_demo --scores attribute-demo-ens`, then choose **Curation results** in the sidebar.

### S1: Record Round A statuses and the 4.5 headline in the step4-summary output
**Files:** `vlapoc/step4.py` (+~200 lines: `RULE`, `status_of_calls`, `metric_phrase`, `sentence_of`, `annotate`, `headline`, `treated_advantage`; `collect` carries `k` and the synthetic tag; `summarize` appends `k`, `rollouts_per_task`, `synthetic` and per metric `<m>__status_code/__status/__sentence` as LAST columns; `summary_run` writes `headline`, `rule`, `synthetic` into the JSON and tags the run synthetic when an input is) · `scripts/verify_training_attribution.py` (`--step4` extended with checks 1.1-1.6; new fixture helper `add_eval_outcomes` next to `add_eval`).
**Premises checked:** the chain records counts but no status (plan table) -> read `step4.summarize` source: no status column existed -> holds. Plan's example sentence says it "adds 'this is not evidence that removal has no effect'" while AC1.3 forbids the phrase `no effect` -> **the plan contradicts itself**; I word it "A change that is not detected is not evidence that removal leaves the policy unchanged" (see findings).
**Acceptance criteria:**
- PASS AC1.1: `.venv/bin/python scripts/verify_training_attribution.py --step4` -> `[PASS] 1.1 ... status codes {'case-confirmed': 'confirmed', 'case-refuted': 'refuted', 'case-notconfirmed': 'not confirmed', 'case-inconclusive': 'inconclusive', 'random0-k5': 'no expectation (control arm)'}` (calls T,T,0 / C,C,T / T,0,0 gate 3/3 / T,g,g gate 1/3, hand-computed from the planted verdicts).
- PASS AC1.2: `[PASS] 1.2 ... headline codes [('h-contradicted', 'contradicted'), ('h-inconclusive', 'inconclusive'), ('h-partly', 'partly shown'), ('h-shown', 'shown')]; inconclusive per-seed 'inconclusive (gate did not fire)'`.
- PASS AC1.3: `[PASS] 1.3 ... 138 texts scanned` (case-insensitive scan for the three banned phrases; every f30 sentence contains `Success within 30 s (τ 30 s)`; with the SYNTHETIC fixture every status and sentence begins `SYNTHETIC: `; the pure function called with `synthetic=False` returns unprefixed text). Real sentences, from `artifacts/s1_step4_summary_SYNTHETIC.csv`: "SYNTHETIC: Removing the 5 episodes per task ranked most harmful for Success within 30 s improved Success within 30 s (τ 30 s) compared with the original policy in 2 of 3 seeds (macro-KS gate fired in 3 of 3): confirmed." and "SYNTHETIC: Inconclusive: the macro-KS gate (did the time-to-success distribution change at all?) fired in only 1 of 3 seeds, so no direction is claimed for Success within 30 s (τ 30 s)."
- PASS AC1.4: `[PASS] 1.4 ... 46 old columns equal HEAD: True; 18 new columns (last)`: the old columns of the fixture summary re-serialised are byte-identical to `git show HEAD:.../s4_step4_summary_SYNTHETIC.csv`. `confirm` / `discrepancy-verdict` unchanged: `verify_dashboard.py --round-b` 13/13 and `--planted-arms` 8/8 (they read `collect`).
- PASS AC1.5: `[PASS] 1.5 ... would submit 116 jobs; chain t-...` and the `step4-summary` command equals `python scripts/stage.py --chain t-X step4-summary --tag t --name step4-summary-t --chain-id t-X`; `diff` of the Round A dry run at HEAD (`artifacts/dry_run_head.txt`) against the dry run after S1 differs only in the chain's git sha. `scripts/verify_runbook.py`: 9/10, the one failure (8.5, CLAUDE.md diffstat) is **pre-existing, see findings**.
- PASS AC1.6: `[PASS] 1.6 ... missing: []` (module docstring cites PoC §9.1 Step 4 (p. 19), §4.5 (p. 11), §4.4 (p. 10), §2.3 (p. 4) and labels the rule **inferred**).
**Demo artifacts produced:** `artifacts/s1_step4_summary_SYNTHETIC.csv`, `artifacts/s1_headline_cases_SYNTHETIC.json`, `artifacts/dry_run_head.txt`.
**Notes / divergences:** the training-attribution feature's own artifact `s4_step4_summary_SYNTHETIC.csv` is still written by `--step4`, restricted to its original columns (so it stays as released); the full table is written to this feature's folder. `step4.treated_advantage` is a method-side copy of `planted_arms.treated_advantage` (the latter imports `vlapoc.recovery`, ground-truth side); `--curation` asserts both agree on every recorded verdict row.

### S2: Read the Round A curation results from the lineage store
**Files:** `vlapoc/results.py` (+~55 lines: `CURATION_OPS`, `curation_runs`, `removed_episodes`; docstring cites §9.1 Step 4 (p. 19), §2.6 (p. 5), §4.5 (p. 11)) · `dashboard/common.py` (cached readers `curation_of`, `curation_verdicts`, `removed_of`).
**Premises checked:** (a) `select.dropped` is a truncated prefix for a k5-per-task arm -> **measured** on the demo store: the tag holds 78 of the 100 ids (cut at 7,900 chars) -> holds; (b) random arms record no `score_run_id`, so the chain is found through the ranked arms and the random arms by chain -> measured in `select_run` source and on the demo -> holds.
**Acceptance criteria:** (all from `verify_dashboard.py --curation`, 20/20 PASS, `artifacts/verify_dashboard_curation.json`)
- PASS AC2.1: `[PASS] 2.1 ... summary step4-summary-demo; arms ['drop-random-20','harm-f30-k5','harm-hrt60_lin-k5','help-f30-k5','help-hrt60_lin-k5','random0-k5','random1-k5']; with a newer Round B-like summary in the chain it still returns step4-summary-demo` (the Round B-like summary was added in a scratch copy of the demo store).
- PASS AC2.2: `[PASS] 2.2 ... drop-random-20: 20 (whole tag ids equal); random1-k5: 100 (prefix of 78 tag ids equal) ...` for all 7 arms, 100 = 5 in each of 20 tasks; `[PASS] 2.2b` the `rank` column (1 to 5 within the task) equals a recomputation from the score table.
- PASS AC2.3: `[PASS] 2.3 ... [('summary', True), ('verdict', True)]; None for the smoke run: True` (one byte appended to the artifact in a scratch copy raises the sha error; the smoke attribute run and an unknown id return `None`).
- PASS AC2.4: `[PASS] 2.4` docstring citations present, `forbidden imports []`; `verify_dashboard.py --results` 12/12.
**Notes:** a Round B summary sharing the chain id is skipped by the row-op filter (tested).

### S3: SYNTHETIC Round A in the demo store (real select / verdict / step4 code)
**Files:** `scripts/dashboard_demo.py` (+~170 lines: `round_a`, `synthetic_rollouts`, `ARM_EFFECTS`, `argparse_ns`; manifest gains `round_a`).
**Premises checked:** (a) the 20 v3/candidate task cells are all among the human-anchor cells -> measured (`True`) -> holds; (b) the demo ensemble's `__same_task` score columns are populated -> **FALSE**: the demo's SYNTHETIC rollouts cover 2 of 20 tasks, so `harm_f30__same_task` is NaN for the other 18 tasks and `select_run` refuses (`LineageError: score table has NaN in 'harm_f30__same_task'`, measured on the first build). Resolution (documented divergence): the demo ranks per task by the all-rollout column `harm_<m>`; the select run records that metric and its declaration, and the page's ranking caption says "rollouts: all". The server ranks by `__same_task`.
**Acceptance criteria:**
- PASS AC3.1: `[PASS] 3.1 ... {'select': 7, 'train': 24, 'eval': 24, 'verify': 48, 'summarize': 1}; verify kinds {'curated_vs_random': 24, 'vs_base': 21, 'head_to_head': 3}; names equal the dry run's 48; provenance SYNTHETIC (111 runs walked)`; every run of the chain tagged `vlapoc.synthetic=true`; real `mlflow.db` sha256 unchanged (`306af412...bde79`).
- PASS AC3.2: `[PASS] 3.2 ... gate p 0.00049975 == 0.00049975; f30 diff -0.0590 CI -0.0930..-0.0200` (pair HRT-curated vs F30-curated, seed 1, recomputed with `compare_policies` from the recorded rollouts, equality on gate p and every metric's diff, ci95, p_holm).
- PASS AC3.3: `[PASS] 3.3 ... status codes ['confirmed', 'inconclusive', 'no expectation (control arm)', 'not confirmed']; headline 'SYNTHETIC: inconclusive'`. Printed as produced, not targeted. The recorded headline sentence: "SYNTHETIC: Same dataset, opposite curations: inconclusive. The macro-KS gate fired in 3 of 3 seeds; speed claim on Speed vs human (HRT) (τ 60 s): not confirmed; reliability claim on Success within 30 s (τ 30 s): not confirmed. No winner is stated." Rows of the demo summary (16 = W only: 7 vs_base, 8 curated_vs_random, 1 head_to_head): both curated-for-F30 and curated-for-HRT arms are `confirmed` against base and against both random draws on f30 and hrt60; help-F30 `confirmed` (worse); help-HRT (planted weak) `inconclusive`; random arms `no expectation (control arm)`; the head-to-head pair is `not confirmed` on both claims although the gate fired 3/3 (per seed: one seed each way, otherwise not shown). **My pilots could not plant "each policy wins its own metric"** (the F30-curated policy that completes more attempts also wins on HRT and RMST; the reason, that timeouts count at 60 s in both, is **inferred**; the observation is **measured** in `data/scratch/cur/pilot3.py` runs: f30-vs-hrt diff on hrt60 +0.058, p_holm 0.033 in the F30 arm's favour), so I stopped tuning and kept the honest result, an `inconclusive` headline: this demonstrates the discipline, not a win.
- PASS AC3.4: `verify_dashboard.py --app` 11/11, `--render` 4/4, `--dropin` 6/6 after the rebuild. `--demo` 5/6: `6.5` fails on **free disk 15 GiB < 20 GiB (measured `df`: 17 GiB free)**, an environment fact; the demo store is 17 MiB and the cache 18 MiB (under 2 GiB). Build times (**measured**, manifest): whole build with smoke 921 s, of which Round A 381 s (48 verdicts 377 s, about 7.9 s each; plan inferred 9.3 s each).
- PASS AC3.5: `[PASS] 3.5` (AST of `synthetic_rollouts` names none of groundtruth / recovery / discrepancy / open_for_method; every eval run's params carry `planted_effect` in words, e.g. 'same as the base policy (random removal) (f 3.0, sigma 0.5, p 0.8)').
**Demo artifacts produced:** `data/lineage_demo` (rebuilt with `--with-smoke --force`, manifest `round_a`), `artifacts/verify_dashboard_curation.json`.
**Divergences from plan:** family O not built (as planned); ranked by `harm_<m>` not `__same_task` (above). Planted effects (SYNTHETIC; words in the manifest): base f 3.0, sigma 0.5, completes 80%; HRT-curated: faster but more variable (f 1.9, sigma 0.9, 82%); F30-curated: more reliable, tight spread (f 2.6, sigma 0.15, 88%); help-F30: 68%; help-HRT: f 3.1 (weak); random arms = base; every trained policy gets seed jitter (f 1.5%, p 1 point).

### S4: The Curation results page
**Files:** `dashboard/views/curation.py` (new, ~190 lines) · `dashboard/common.py` (`PAGES`, readers, `condensed_reasons` for the page's banner, `functional_note_text`) · `dashboard/app.py` (`RENDER`).
**Premises checked:** `mark_errorbar` draws in the browser -> **FALSE for this layered chart**: measured in headless Chrome the plot area was 22 px and 74 px tall although `height=220/270` was set; replaced by `mark_rule` + `mark_point` with `alt.Step` heights, measured 222 px and 306 px. (Also noted: Streamlit does not hot-reload without watchdog, so the first measurement needed a server restart.) The provenance of the summary lists 150+ recorded reasons; the page condenses them (grouped, with examples in backticks) so the banner stays one line plus an expander and no snake_case labels leak.
**Acceptance criteria** (`verify_dashboard.py --curation`, 20/20):
- PASS AC4.1: `[PASS] 4.1 ... first load 1.9s; f30 table equal True, hrt60 table equal True` (16 rows: status and sentence equal the recorded text for two metrics; the headline status and sentence appear verbatim in the markdown).
- PASS AC4.2: `[PASS] 4.2` forest data equals `treated_advantage` recomputed from the recorded `verdict.json` (and equals `planted_arms.treated_advantage`, asserted); the F30-curated row is the sign-flipped advantage; per-arm plot 9 rows.
- PASS AC4.3: `[PASS] 4.3 ... 100 rows; trace commands 2 rc [0, 0]`.
- PASS AC4.4: `[PASS] 4.4 ... unreached []; ranking caption has 'F(30) = P(T <= 30 s)'`; the verdict functionals are listed in a caption (`Success within 30 s: F(30) = P(T <= 30 s) [τ 30 s] | ...`) and in the Metric selector's tooltip; the HRT ranked-by-linearized-return vs judged-on-ratio point is in the headline card's caption (PoC §4.3) and in the sentence for the HRT metric.
- PASS AC4.5: `[PASS] 4.5 ... snake labels []; banners 2` (header's + page's, each one line, reasons in an expander); no banned phrase in text or table cells.
- PASS AC4.6: `[PASS] 4.6` banner SYNTHETIC equals `results.provenance(summary)`, every recorded status/sentence begins `SYNTHETIC: `.
- PASS AC4.7: `[PASS] 4.7` on a scratch copy where the head-to-head gate did not fire in any seed: card "inconclusive", "No winner is stated", every seed reads "inconclusive (gate did not fire)", every forest row "did not fire", no "wins its own metric".
- PASS AC4.8: `[PASS] 4.8` info message with the `server_pipeline.py ... --from select` command for a score run without Round A, no exception.
- PASS AC4.9: `[PASS] 4.9 ... 16 rows recomputed` (summary stripped of the new columns and of `headline` in a scratch copy: same text, caption "computed at display time from the recorded per-seed calls").
- PASS AC4.10: screenshots at 1400 px: `artifacts/screens/curation_results.png` (looked at: headline card, per-seed table, forest plot with the two curated-policy rows per seed, arm table, per-arm plot, removed-episode table with video). Page height 4061 px.
**Demo artifacts produced:** `artifacts/screens/curation_results.png`.
**Notes:** first load of the page on the demo store is 1.9 s (**measured**, provenance walks 111 runs): the plan's 10 s risk did not materialise. Nothing is corrected across arms or seeds and the page says so.

### S5: Drop-in proof and render checks
**Files:** `scripts/verify_dashboard.py` (`curation_dropin`, `--curation` mode, PAGES, `--render` extended), `scripts/dashboard_screenshots.py` (PAGES), `scripts/verify_training_attribution.py` (`add_eval_outcomes`).
**Premises checked:** the real eval records `input_entity("checkpoint")` (runs.py l.738) and the columns `T_s`, `event`, `tau_episode_steps`, `suite`, `libero_task_id`, `rollout_id` (read from `eval-smolvla-harness-spatial0` through `verdict.eval_rollouts`, printed dtype table) -> holds; the fixture helper writes exactly those plus `censored`, `pairs`.
**Acceptance criteria:**
- PASS AC5.1: `verify_dashboard.py --dropin` -> `[PASS] 5.1 ... 7 selects (k 5/3), 48 verdicts, 16 summary rows, headline ['inconclusive']; page rows equal recorded: True; porcelain before '' after ''` (the select, W-verify and step4-summary commands parsed from `server_pipeline.py --tag ra ... --dry-run` run through `scripts/stage.py`, 208 s). Divergences: the fixture's own subset is the parent and k is scaled (5 -> 5 per task, 20 -> 3 overall, in the 20-task fixture); family O verifies are not run; the commands are otherwise the dry run's.
- PASS AC5.2: `[PASS] 5.2 ... re-tagged: banners [], REAL captions 2; one synthetic tag added: banner back True`.
- PASS AC5.3: `verify_dashboard.py --render` -> `[PASS] 5.10a ... 'Curation results': [[592, 222], [720, 306]]` (two plot areas, both >= 200 px); `[PASS] 5.10c` no truncated metric, no exception on any page.
- AC5.4: regression below.
**Demo artifacts produced:** `artifacts/render_measurements.json`, `artifacts/verify_dashboard_dropin.json`, `artifacts/verify_dashboard_render.json`.

### Regression (measured)
Run after the final demo rebuild, on the committed tree (logs under `data/scratch/cur/logs/`); the other features' tracked artifacts that these verifiers rewrite (run ids, timings) were restored with `git checkout`.

| Command | Result |
|---|---|
| `verify_dashboard.py --curation` | pass 20/20 |
| `--results` / `--clusters` / `--planted` / `--discrepancy` / `--round-b` / `--planted-arms` | 12/12, 14/14, 7/7, 10/10, 13/13, 8/8 |
| `--app` | 11/11 (the six pages, incl. Curation results, pass the SYNTHETIC-banner, plain-language and declared-functional checks) |
| `--render` (headless Chrome, 1400 px) | 4/4; Curation results plots 222 px and 306 px |
| `--dropin` | 6/6 |
| `--demo` | 5/6; 6.5 fails on free disk (17 GiB < 20 GiB), environment |
| `verify_training_attribution.py` `--blind --functionals --step4 --recovery --names --groups --integration` and `--pipeline data/scratch/dash_exec/dry_pipeline.txt` | all pass: 42s S1, 18/18, 19/19, 11/11, 13/13, 9/9, 12/12, 9/9 |
| `verify_attribution.py` | 32/32 |
| `verify_runbook.py` | 9/10; 8.5 fails at HEAD without my changes (finding 2) |
| `verify_lineage_s3`, `verify_blind_s0`, `verify_method_boundary_s3`, `verify_derive_gt_s2`, `verify_verdict_s7`, `verify_metrics_s1`, `verify_consumers_s6`, `verify_runs_s5 --r2`, `verify_returns_s6 --synthetic` | rc 0 |
| `data/lineage/mlflow.db` sha256 | `306af4123fda7e635892d459a2e34008ba1cb62f6c4fce152c546e0da7bdde79` (unchanged) |
| job counts | Round A dry run 116 jobs; `verify_runbook` counts 116/24, 129/33, 21/6 pass

### Findings for the analyzer
1. **Plan contradiction (S1):** the plan's sentence rule says to add "this is not evidence that removal has no effect" while AC1.3 forbids `no effect`. I used "A change that is not detected is not evidence that removal leaves the policy unchanged".
2. **Pre-existing `verify_runbook.py` failure (AC1.5 / brief "still passes"):** check `8.5` (CLAUDE.md diffstat equals the value recorded before dashboard batch B1) fails at HEAD `4aeff57` without any of my changes (**measured**: with my work stashed, `verify_runbook.py` gives 9/10 with only 8.5 failing; the recorded file `b1_claude_md_diffstat_before.txt` is non-empty and CLAUDE.md is now committed). All counts (116/24, 129/33, 21/6) and the other 9 checks pass. I did not edit the recorded file or CLAUDE.md. Also: `verify_runbook.py` check 8.1c fails when the source tree is dirty (uncommitted edits under `vlapoc/` or `scripts/` make the eval stage refuse), so run it on a committed tree.
3. **`verify_dashboard.py --demo` 6.5 fails on this machine**: free disk is 17 GiB, the check wants > 20 GiB. Not related to the change (store 17 MiB).
4. **Plan divergence (S3):** demo ranks by the all-rollout column because the synthetic rollouts cover 2 tasks. If the owner wants the demo to show per-task ranking by `__same_task`, the demo needs synthetic rollouts for all 20 tasks.
5. **The demo's headline is `inconclusive`, not `shown`.** In my pilots (**measured**, `data/scratch/cur/pilot3.py`) a synthetic policy that completes more attempts won on HRT and RMST as well as F30; why (timeouts count at 60 s in both) is **inferred**. So the two curations did not trade off (`data/scratch/cur/pilot*.py`); I did not keep tuning until the headline looked good. The `shown`, `partly shown` and `contradicted` states are exercised by the S1 fixture (1.2) and the unit-level page test (4.7 shows the inconclusive card).
6. **Scope of the page checks:** AppTest checks 4.1-4.9 run only on the demo and scratch copies; the real server summary is first rendered the day Round A finishes (the drop-in proves the shape, not the numbers).
7. The inferred rules (across-seed rule from `confirm`, headline rule, inconclusive vs not confirmed) are recorded as the `rule` param and in the summary JSON; the owner may override.
8. `step4.collect` reads `k` from the select run's params; summaries recorded before S1 lack the `k` column and arm labels then read "the top most harmful ..." instead of "5 most harmful ...".

## Round 2: 2026-11-05 (rework after analysis-r1 and plan revision r1)

Commits: B1 `step4` and `--step4` checks; B2 runbook check; B3 and N4 page and verifier; two follow-up fixes. Demo rebuilt with `dashboard_demo.py build --with-smoke --force` (881 s, Round A 380 s; **measured**, manifest). Screenshot re-captured and looked at: `artifacts/screens/curation_results.png` (1400 px, page height 4508): the card now reads "SYNTHETIC: not shown" with the bounded sentence, the per-seed table, the forest plot, the arm table with the selected arm's sentences as text below it, and the operation-level section caption.

### B1: headline state `not shown`
**Files:** `vlapoc/step4.py` (`headline()` order: contradicted, shown, partly shown, `inconclusive` only if gates < ceil(2n/3), else `not shown`; `RULE`, module docstring bullet, both sentences), `scripts/verify_training_attribution.py` (`--step4` 1.2 six cases, 1.3 six headline sentences, 1.6 `not shown` in RULE/docstring, new 1.7), `dashboard/views/curation.py` docstring.
- PASS AC1.2: `[PASS] 1.2 ... headline codes [...]` six cases with literal expectations: shown, partly shown, contradicted, `h-inconclusive` gate 0/3 -> inconclusive, `h-notshown` gate 3/3 (hrt60 `C,0,0`, f30 `0,T,0`) -> not shown, `h-notshownb` gate 2/3 (hrt60 and f30 `0,0,g`; the boundary case tests the **inferred** rule) -> not shown; the not-shown and inconclusive sentences contain "No winner is stated" and not "wins its own metric".
- PASS AC1.3: `[PASS] 1.3 ... 162 texts scanned` (all six headline sentences included, 6 asserted).
- PASS AC1.6: `[PASS] 1.6 ... missing: []` including `grep -c "not shown" vlapoc/step4.py` >= 2 (12), the docstring bullet "``not shown`` otherwise" and no "inconclusive otherwise".
- Demo: `--curation` 3.3 prints `headline 'SYNTHETIC: not shown': SYNTHETIC: Same dataset, opposite curations: not shown at this sample size (3 seeds x 50 rollouts per task). The two curated policies' time-to-success distributions differ (macro-KS gate fired in 3 of 3 seeds), but neither curated policy's win on its own metric replicated across seeds (speed claim on Speed vs human (HRT) (τ 60 s): not confirmed, 0 of 3; reliability claim on Success within 30 s (τ 30 s): not confirmed, 0 of 3). A change that is not detected is not evidence that the curations leave the policies unchanged. No winner is stated.`
- PASS AC4.7a/b/c (`--curation`, 23/23): `4.7a` gate-0/3 scratch copy -> card "inconclusive"; `4.7b` demo -> `card '**SYNTHETIC: not shown**'; headline gate 3/3`, no "inconclusive" in the card, no "wins its own metric" on the page; `4.7c` `demo: card 'not shown', row statuses hrt60 'not confirmed' / f30 'not confirmed'; gate-0/3 copy: card 'inconclusive', row 'inconclusive' / 'inconclusive'`.

### B2: runbook check 8.5
**Files:** `scripts/verify_runbook.py` (8.5 re-anchored), `artifacts/claude_md_base_commit.txt` (`bad024135a4e9462bf34a843cc6b061b80dd4c6d`); the old `b1_claude_md_diffstat_before.txt` is kept unchanged as history. CLAUDE.md not touched.
- The check now asserts no commit after the base touches CLAUDE.md and the working tree has no change to it. **Measured:** `git log --oneline bad0241..HEAD -- CLAUDE.md | wc -l` -> `0`.
- PASS: `.venv/bin/python scripts/verify_runbook.py` -> `[PASS] 8.5 ... base bad02413; commits touching CLAUDE.md after it: []; working-tree diff ''` and `runbook: pass=True (10/10)` on a committed tree (all counts 116/24, 129/33, 21/6 pass).

### B3: family O section
**Files:** `dashboard/views/curation.py` (section "Operation-level check (family O), alongside": recorded O rows with arm, metric, compared with, status, calls by seed, gate fired, sentence verbatim; caption with the O functional from the registry and the three required statements; "No operation-level (family O) verdicts are recorded for this chain." when none; O rows stay out of the headline, W table and plots; new `row_keys` because the recorded JSON gives every row every column, NaN for the other family's metrics), `vlapoc/step4.py` (`metric_phrase` takes the registry's plain `metric` name for family O), `scripts/verify_dashboard.py`.
**Premises checked (planner's risks):** (a) code that loops all summary rows and reads `f30__status` breaks once O rows exist -> **true**: the first Round 2 drop-in run raised `KeyError: 'nan'` on the page (the JSON rows carry NaN for the other family's columns) and the 5.1 `want` set needed a family filter; fixed in the page (`row_keys`) and the verifier (filter by family) -> now holds. (b) `compare_policies(..., family="O")` on the drop-in fixture -> **it completes**: 21 O verifies ran through the stage CLI and the recorded `rmst60_op_cap` rows have statuses and sentences; O was not dropped.
- PASS AC1.7: `[PASS] 1.7 ... 'operation RMST, capped (τ 60 s)'`.
- PASS AC4.11a: `[PASS] 4.11a ... W table rows 16; O rows recorded 0` and the caption above is on the page.
- PASS AC5.5 and 4.11b/c: `--dropin` 7/7: `[PASS] 5.5 ... 23 summary rows, O metric 'rmst60_op_cap', O table 7 rows, W table 16; forest rows [6, 9]; unreached []; snake []; 69 verdicts ran in total`; `[PASS] 5.1 ... 7 selects (k 5/3), 69 verdicts, 23 summary rows (7 family O), headline ['not shown']`; `[PASS] 5.2`. Added drop-in time (**measured**): the stage-CLI chain took 236 s (was 208 s); the whole mode 278 s (was 241 s) for the 21 added O verifies. 5.5 is evaluated on the re-tagged (REAL) page, since before the re-tag the header banner lists unconverted smoke reasons.

### Item 4: inferred claims labelled
Finding 5 and S3 AC3.3 above now read: the observation (a policy that completes more attempts also wins on HRT and RMST in the pilots) is **measured** (`data/scratch/cur/pilot3.py`: f30-arm vs hrt-arm, hrt60 diff +0.058, p_holm 0.033), the reason (timeouts count at 60 s in both) is **inferred**.

### N3 and N4 (done)
- N3: `--step4` writes its curation checks to `artifacts/verify_step4_curation.json` (20/20 pass, S1 checks 1.1-1.7) and the training-attribution feature's `s4_summary_and_selects.json` is no longer touched (`git status` clean for that file after the run, **measured**).
- N4: removed-episodes column shows the demo index (`demo 10`) with the full task name in its own column; the span reads "5 in each of 20 tasks"; the selected arm's recorded sentences are shown as text below the arm table (`Sentences for the arm selected under "One arm in detail"`). N1, N2, N5 left to the refactorer.

### Regression (Round 2, measured; final rebuilt demo, committed tree)
| Command | Result |
|---|---|
| `verify_dashboard.py --curation` / `--app` / `--render` / `--dropin` | 23/23, 11/11, 4/4 (Curation plots 222 and 306 px), 7/7 |
| `--results --clusters --planted --discrepancy --round-b --planted-arms` | 12/12, 14/14, 7/7, 10/10, 13/13, 8/8 |
| `--demo` | 5/6; 6.5 fails on free disk only (`free 19 GiB` < 20 GiB; store 17 MiB, cache 19 MiB): environmental |
| `verify_training_attribution.py --step4` | 20/20 |
| `--blind --functionals --recovery --names --groups --integration`, `--pipeline data/scratch/dash_exec/dry_pipeline.txt` | all rc 0 |
| `verify_runbook.py` | **10/10** |
| `verify_attribution.py` | 32/32 |
| `verify_lineage_s3`, `verify_blind_s0`, `verify_method_boundary_s3`, `verify_derive_gt_s2`, `verify_verdict_s7`, `verify_metrics_s1`, `verify_consumers_s6`, `verify_runs_s5 --r2`, `verify_returns_s6 --synthetic` | rc 0 |
| `data/lineage/mlflow.db` sha256 | `306af4123fda7e635892d459a2e34008ba1cb62f6c4fce152c546e0da7bdde79` (unchanged) |
Other features' tracked artifacts rewritten by the verifiers were restored with `git checkout`. No streamlit or Chrome process is left running (`ps` count 0).

### Findings
1. The recorded summary JSON gives every row every column (NaN where the other family has no metric). Any reader that picks metric keys from `rows[0]` must use only the string-valued `__by_seed` columns (`row_keys`).
2. Pre-existing and still open for the refactorer or owner: N1 (free-disk threshold in `--demo` 6.5), N2 (`treated_advantage` duplicate), N5 (`condensed_reasons`).
