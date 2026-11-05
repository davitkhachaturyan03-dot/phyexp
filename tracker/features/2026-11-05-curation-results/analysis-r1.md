VERDICT: REWORK
ROUTE: planner+executor
ROUND: 1

<!-- Saved verbatim by the orchestrator from the analyzer's returned text: the analyzer's own
     write was refused by the harness. -->

## Summary (3–5 lines)
Almost everything reproduces: `--step4` S1 checks pass, `--curation` passes 20/20, `--app` 11/11, `--render` 4/4 with Curation plots at 222 and 306 px, `--dropin` 6/6, `--demo` 6/6 (6.5 passed on my run), and the full regression set passes. The real store hash is unchanged. The page reads the recorded summary verbatim and the drop-in exercises the server's `__same_task` ranking. Two blocking problems remain:
- **B1 (planner+executor), the headline rule.** The §4.5 headline says **"inconclusive" even though "the macro-KS gate fired in 3 of 3 seeds"** (the demo, measured). That contradicts the brief ("'inconclusive' where the gate did not fire"). It also contradicts the plan's own per-metric vocabulary, under which the same head-to-head row reads "not confirmed" in the arm table on the same page.
- **B2 (executor), the runbook check.** `verify_runbook.py` is 9/10. The 8.5 failure is genuinely pre-existing: the owner's commit `bad0241` committed exactly the recorded diffstat. But the brief requires the runbook to pass, and the fix is in the verifier.

There is also one silent divergence to record: the page drops family O rows entirely (B3).

## Rebuttals received
| From | Point disputed | Their evidence | Conceded? | Why / why not |
|---|---|---|---|---|
| (none, round 1) | | | | |

## Acceptance criteria
| Story | AC | Claimed | Reproduced? | Evidence |
|---|---|---|---|---|
| S1 | AC1.1 | PASS | ✅ | **Measured**, `verify_training_attribution.py --step4` rc 0: `[PASS] 1.1 ... {'case-confirmed': 'confirmed', 'case-refuted': 'refuted', 'case-notconfirmed': 'not confirmed', 'case-inconclusive': 'inconclusive', 'random0-k5': 'no expectation (control arm)'}`. I read the fixture: the calls are planted inputs (T,T,0 / C,C,T / T,0,0 at 3/3 / T,g,g) and the expected codes are literals, not code output. |
| S1 | AC1.2 | PASS | ✅ (the check passes), but see B1 | `[PASS] 1.2 ... [('h-contradicted','contradicted'), ('h-inconclusive','inconclusive'), ('h-partly','partly shown'), ('h-shown','shown')]`. The fixture's `h-inconclusive` case has the gate fired in **0/3** seeds. The case "gate fired, no claim replicated" is not tested, and that is exactly what the demo produces. |
| S1 | AC1.3 | PASS | ✅ | `[PASS] 1.3 ... 138 texts scanned`. |
| S1 | AC1.4 | PASS | ✅ | `[PASS] 1.4 ... 46 old columns equal HEAD: True`. `s4_step4_summary_SYNTHETIC.csv` is not in `git diff --stat bad0241..HEAD`, so HEAD equals the base for that file. `--round-b` passes 13/13 and `--planted-arms` 8/8. |
| S1 | AC1.5 | PASS | ⚠️ in part | The job side holds: `[PASS] 1.5 ... would submit 116 jobs`, and the step4-summary command is unchanged. The "`verify_runbook.py` passes" part does **not** reproduce: 9/10, 8.5 fails (B2). |
| S1 | AC1.6 | PASS | ✅ | `[PASS] 1.6 ... missing: []`. I also read the docstring: § and page numbers are present and the rule is labelled inferred. |
| S2 | AC2.1–2.4 | PASS | ✅ | `--curation` 20/20 (`[PASS] 2.1`, `2.2` "prefix of 78 tag ids", `2.2b`, `2.3`, `2.4`). The rank has to be recomputed because `lineage.drop_top_k_per_task` stores `dropped = sorted(dropped)` (vlapoc/lineage.py l.616, read), so the rank is not recoverable from the derivation. |
| S3 | AC3.1–3.3, 3.5 | PASS | ✅ | `[PASS] 3.1 ... {'select': 7, ...}`, `3.2` "gate p 0.00049975 == 0.00049975", `3.3`, `3.5`. I printed the 16 recorded demo rows myself (see B1 for the head-to-head row). |
| S3 | AC3.4 | PASS (with 6.5 failing on disk) | ✅ | **Measured** `--demo` 6/6 on my run: "free 20 GiB", `df -k` 21,230,660 KiB = 20.25 GiB. It passed by a hair where the executor's run failed at 17 GiB, so the result tracks free disk, not the feature (see N1). Store 17 MiB, cache 18 MiB. |
| S4 | AC4.1–4.9 | PASS | ✅ | `--curation` `[PASS] 4.1` through `4.9`, first load 1.8 s. |
| S4 | AC4.10 | PASS | ✅ (looked at) | `artifacts/screens/curation_results.png`, read by me. Problems seen there: N4 and N5. |
| S5 | AC5.1–5.2 | PASS | ✅ | `--dropin` 6/6 (207 s): `[PASS] 5.1 ... 7 selects (k 5/3), 48 verdicts, 16 summary rows ... porcelain before '' after ''`, `[PASS] 5.2`. The dry run's select arms use `--metric harm_f30__same_task` / `harm_hrt60_lin__same_task` (grep of `artifacts/dry_run_head.txt`, measured), so the server's ranking column is exercised here. |
| S5 | AC5.3 | PASS | ✅ | `--render` 4/4: `'Curation results': [[592, 222], [720, 306]]`. No streamlit or headless Chrome process was left behind (ps, measured). |
| S5 | AC5.4 | PASS except runbook and disk | ⚠️ | `--results` 12/12, `--clusters` 14/14, `--planted` 7/7, `--discrepancy` 10/10, `--round-b` 13/13, `--planted-arms` 8/8, `verify_attribution` 32/32, training-attribution `--blind` S1 pass, `--functionals` 18/18, `--recovery` 11, `--names` 13, `--groups` 9, `--integration` 12, `--pipeline` 9 (all PASS, 0 FAIL). `verify_runbook` 9/10 (B2). `shasum` gives `306af4123fda…bde79`. I restored all tracked artifacts the verifiers rewrote, and `git status` is clean apart from `data.zip`. |

## PoC scope assessment
- **Covered.** §4.5 p.11 (headline card and forest plot), §2.6 p.5 (curated vs random, per draw), §9.1 Step 4 p.19 (status per arm, comparison and metric), §4.4 p.10 (gate first, then Holm-corrected direction, metric and τ named), §3.4 p.7 (declared functionals reachable). I re-read every quote the plan cites against the PDF text. All are verbatim and on the stated pages.
- **Missing.** Nothing the brief requires, apart from B1's wording.
- **Over-built.** Nothing material. Minor items are listed in the audit.
- **Silent divergence (B3).** `dashboard/views/curation.py` keeps only `rows = [r for r in rows if r.get("family", "W") == "W"]`, so family O verdicts are never shown. The plan (S3 deliverables and "Out of scope") says "The page shows W (declared) first and O where recorded". The execution log records only "family O not built (as planned)" for the demo. On the server, Round A records 21 family-O vs_base verdicts (plan table, `W && O`), and the page will hide them without saying so.
- **Recorded divergences I accept** (my rulings on open item 3):
  - **Sentence wording.** The plan contradicts itself, and the executor resolved it in the spirit of the plan. Plan l.150–151 asks for the sentence to add "this is not evidence that removal has no effect"; AC1.3 (l.160) bans `no effect`. Accepted (quoted, plan.md).
  - **Demo ranks by the all-rollout column.** Accepted. The cause is measured (`LineageError ... NaN in 'harm_f30__same_task'`). It is labelled on the page ("rollouts: all", visible in the screenshot) and in the manifest. The drop-in exercises the server's `__same_task` column through the real stage CLI (measured above).
  - **Demo headline is not "shown"; other states are covered only by the fixture.** Accepted in principle. The page has no branch that depends on the headline state: it prints `h['status']` and `h['sentence']` verbatim and draws the same forest for any state (curation.py, read). So the S1 fixture is the right place to cover the states. Not tuning the demo until it shows a win is the correct call. But after B1, the demo's state must be the new "not shown", not "inconclusive".
  - **`mark_rule` + `mark_point` instead of `mark_errorbar`.** Accepted. Both are built-in Altair marks; the Altair docs allow ranged `x`/`x2` either way ([Altair error bar docs](https://altair-viz.github.io/user_guide/marks/errorbar.html), [python-altair 5.0.1 doc source](https://sources.debian.org/src/python-altair/5.0.1-2/doc/user_guide/marks/errorbar.rst)). The 222 and 306 px plots reproduce in Chrome (measured). The executor's 22 px measurement for errorbar is theirs; I did not reproduce it, and nothing hinges on it.
  - **`step4.treated_advantage` is a copy.** Accepted for this round; it is non-blocking (see the audit). The reason is real: `vlapoc/planted_arms.py` imports `from vlapoc.recovery import EXPECTED_METRICS` (read).
  - **Drop-in scoped to family W.** Accepted for the proof. But see B3 for what the page does with O.

### The two inferred rules (open item 4)
- **Across-seed "confirmed" = `confirm.status_of`** (≥ ⌈2n/3⌉ seeds as expected, none contradicting). The PoC gives no across-seed aggregation rule. §2.3 p.4 (quoted) "2–3 seeds per data configuration | Prevents seed variance from masquerading as a data effect" motivates replication. §4.4 p.10 (quoted) makes direction conditional on the gate, and `step4.call` returns `g` for that. The rule is consistent with both. It is the rule the dashboard feature already used for Round B, and it is recorded as the `rule` param. **Accept as inferred.** It goes to the owner as a non-blocking question, with no second rule invented.
- **The headline rule.** "Shown iff both claims confirmed" matches §4.5 p.11 (quoted) "show that each curated policy wins on its own metric", and "contradicted if either refuted" is the natural opposite. **But the orchestrator's description of the rule is not what the code does.** The code (`step4.headline`) returns `inconclusive` for every case that is not shown, partly shown or contradicted. That includes the gate firing in 3/3 seeds with no claim replicating, which is exactly the demo. See B1.

## Overcomplication & reinvention audit
| Mechanism (file:line) | Existing OSS alternative | Covers our need? (spec rules checked) | Evidence | Recommendation |
|---|---|---|---|---|
| `step4.status_of_calls` / `annotate` / `sentence_of` / `headline` (vlapoc/step4.py, ~180 lines) | None for bounded wording. The status reuses `vlapoc.confirm.status_of` | n/a | Inferred: no library words a replication status. The statistics are not re-done (it reads the recorded `__by_seed`) | **Keep.** Fix the headline rule (B1). |
| `step4.treated_advantage` (step4.py, 9 lines) | The identical function `vlapoc.planted_arms.treated_advantage` (l.169–177) | Yes, byte-for-byte the same arithmetic | Measured (diff read; `--curation` 4.2 asserts the two agree) | **Simplify (non-blocking, refactorer).** Keep the step4 copy as the only one, and change `planted_arms` to `from vlapoc.step4 import treated_advantage`. That removes the copy and its equality assertion. step4 does not import planted_arms, so there is no cycle (inferred from the import list read). |
| `results.curation_runs`, `results.removed_episodes` (vlapoc/results.py +62) | MLflow `search_runs`, already wrapped by `results.find_runs` / `runs_with_param`, which these reuse | They are thin adapters | Measured (read). The rank recomputation is needed because the derivation stores `dropped` sorted (lineage.py l.616) | **Keep.** |
| Forest plot (curation.py `forest`) | Altair `mark_errorbar`; `forestplot` on PyPI (matplotlib) | Altair built-ins; `forestplot` would add a dependency and lose tooltips | Quoted (Altair docs above). Measured render | **Keep.** |
| `common.condensed_reasons` (dashboard/common.py, regex over formatted reason strings) | None. But it re-parses strings that `results.provenance` itself formats | Works (measured 4.5) | Inferred: parsing your own formatted strings is fragile. Structured reasons from `provenance` would make the regex unnecessary | Simplify (non-blocking). |
| `common.functional_note_text` (a one-line wrapper of `st.caption`) | `st.caption` | Yes | Read | Simplify (non-blocking): call `st.caption` directly. |
| curation.py repeats the `CURATION_OPS` tuple inline (the `rows` filter) | `results.CURATION_OPS` | Yes | Read | Simplify (non-blocking). |
| Demo `round_a` hand-parses the dry-run commands into dicts (`toks[i].lstrip("-")`) twice, and builds `argparse_ns()` by hand | argparse, but `stage.py` / `server_pipeline.py` build their parsers inside `main()` (l.382, l.309), so no parser object is exposed | Partly | Read. The `--curation` check 3.1 checks the run names against the dry run, which guards the retyped Namespace | **Keep** for a demo builder (non-blocking). |
| `curation_dropin` and the `--curation` mode (verify_dashboard.py ~430 lines) | AppTest (Streamlit's own), as on the other pages | Yes, it uses it | Read | Keep. |

**Searches run:** "Altair mark_errorbar pre-aggregated x x2 forest plot layered chart height" and "streamlit st.dataframe column_config TextColumn width wrap long text" (results cited in N4). From the plan I re-checked `forestplot` (a new dependency). I found nothing off the shelf for replication-status wording or for Round-A chain discovery in MLflow beyond `search_runs`, which is already used.

## Feedback: planner
1. **B1: the headline rule needs a "gate fired, not replicated" state distinct from "inconclusive".** In plan S1, "The rule", the headline is `inconclusive` "otherwise". That collides with the per-metric rule two lines above it, under which a row is `inconclusive` only when the gate fired in fewer than ⌈2n/3⌉ seeds. It also collides with the brief ("'inconclusive' where the gate did not fire") and with your own open question 2.
   - **Measured on the demo:** the head-to-head row has `gate_fires 3/3`, hrt60 `C,0,0` → `not confirmed`, f30 `0,T,0` → `not confirmed`. Yet the headline reads "SYNTHETIC: Same dataset, opposite curations: **inconclusive**. The macro-KS gate fired in 3 of 3 seeds; …". So the same page shows "not confirmed" in the arm table and "inconclusive" on the card for the same verdicts.
   - **Specify the rule:** `inconclusive` only when the gate fired in fewer than ⌈2n/3⌉ seeds (then both claims are `inconclusive`, since speed and reliability share one verdict and so one gate count). Use `not shown` when the gate fired in at least ⌈2n/3⌉ seeds and neither claim is confirmed or refuted. The sentence is bounded, e.g. "not shown at this sample size: the two policies differ (gate fired k of n), but neither curated policy's win on its own metric replicated across seeds; no winner is stated."
   - **Extend AC1.2** with a fifth planted case (gate 3/3, calls `C,0,0` / `0,T,0`, expected `not shown`) and require the `h-inconclusive` case to keep gate 0/3.
   - **Update AC4.7 and the page docstring** to the two states.
2. **B3:** decide family O for the page. Either AC: "family O rows are shown in their own section when recorded". Or a recorded divergence with an on-page caption: "family O (operation-level) verdicts are recorded for the vs-base comparisons and are not shown here". The plan currently promises "O where recorded" and the code drops O silently.

## Feedback: executor
1. **B1:** implement the planner's revised headline rule in `vlapoc/step4.py` `headline()`, in the final `else` branch: `code = "inconclusive"` only when `gates < math.ceil(2*n/3)`, else `"not shown"`, with its own sentence. Update `RULE` and the module docstring bullet. Add the fifth case to `verify_training_attribution.py --step4` 1.2. Update the `dashboard/views/curation.py` module docstring item 1. Rebuild the demo (`dashboard_demo.py build --with-smoke --force`).
   - **Expect:** `--curation` 3.3 prints headline `'SYNTHETIC: not shown'`, and 4.7 still shows `inconclusive` on its gate-0/3 scratch copy.
   - Re-capture `artifacts/screens/curation_results.png`.
2. **B2: `scripts/verify_runbook.py` check 8.5** (l.313–315) compares the working-tree `git diff --stat CLAUDE.md` with `tracker/features/2026-10-22-dashboard/artifacts/b1_claude_md_diffstat_before.txt`.
   - **Why it fails (measured):** that file holds ` CLAUDE.md | 5 +++--` / `3 insertions(+), 2 deletions(-)`, which is exactly the diffstat of the owner's commit `bad0241` (`git show --stat bad0241`). The owner's uncommitted edit was recorded and later committed, so the working-tree diff is now empty and the check fails with no agent change involved. Neither the recorded file nor `verify_runbook.py` changed in `bad0241..HEAD` (measured), and dashboard analysis-r3 recorded 10/10 before that commit.
   - **The fix:** re-anchor the check to what it means, "no agent commit touches CLAUDE.md". For example, assert `git log --format=%H <base>..HEAD -- CLAUDE.md` is empty and `git diff --quiet -- CLAUDE.md` holds, with `<base>` = `bad0241` recorded in a new artifact file, and keep the old file as history.
   - Do **not** edit CLAUDE.md. **Expect** `verify_runbook.py` 10/10 on a committed tree.
3. **B3:** implement the planner's O decision. If it is a caption only, add the divergence line to the execution log and to the page's caption, and have `--curation` check the caption is present.
4. **Label the inferred claim in execution-log finding 5 and S3 AC3.3.** "A synthetic policy that completes more attempts wins on every metric ... because timeouts count at 60 s in both" is stated as mechanism. Label it **inferred** (or cite the pilot output under `data/scratch/cur/` as measured).

## Non-blocking
- **N1, `--demo` 6.5 disk threshold (open item 2).** This is environmental (measured). It passes for me at 20.25 GiB free and failed for the executor at 17 GiB. The store plus cache is 35 MiB, so this feature does not move it. The 20 GiB figure comes from the server-run runbook's rule for the server scratch disk (`.claude/skills/server-run/SKILL.md` l.25, l.184, quoted). As a hard pass/fail on a laptop that also holds the 90 GB `data.zip`, it measures the machine, not the dashboard. Suggest (a dashboard-feature change, for the refactorer or the owner): keep `store + cache < 2 GiB` as the check and turn free disk into a printed warning.
- **N2.** Dedupe `treated_advantage` (see the audit).
- **N3.** `--step4` writes this feature's checks 1.1–1.6 into the training-attribution feature's tracked `s4_summary_and_selects.json`, so every run dirties another feature's artifact (measured: +30 lines). Write them to this feature's `artifacts/` instead.
- **N4, page readability (screenshot).**
  - The removed-episodes `episode` column uses `words.plain(e)[-70:]`, which cuts ids mid-word from the left ("NE2 put both the cream cheese…"). It also repeats the task, which is already in the `task` column. Show the demo index, or truncate on the right.
  - "(5 to 5 in each of 20 tasks)" should read "5 in each of 20 tasks" when min equals max.
  - The arm table's `sentence` and `gate fired` columns are off-screen at 1400 px. `st.dataframe` does not wrap text, so long text needs horizontal scrolling ([Streamlit forum](https://discuss.streamlit.io/t/wrap-text-columns-in-dataframe-data-editor/56899/2), quoted: "st.dataframe does not support word wrapping"). The bounded sentence is the page's main plain-language content, so consider showing the selected arm's sentences as text below the table.
- **N5.** `common.condensed_reasons` regex-parses strings that `results.provenance` formats. Structured reasons would be more robust.
- **N6.** Recorded statuses get `SYNTHETIC:` only from verify, select or verdict inputs, not from synthetic train or eval runs. The banner covers those (the full provenance walk), so this is acceptable and matches the plan. Mention it in the report.
- **N7.** The executor's other notes (finding 8, older summaries lacking `k`) are fine as recorded.
