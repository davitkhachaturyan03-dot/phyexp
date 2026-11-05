# Refactor: curation headline results in the dashboard

Refactorer, branch `curation-results`, starting at HEAD `8828a72` (feature code range `bad0241..HEAD`). Four commits: `2317831` (N2, N9), `19c6fb2` (N5), `2ebdc2c` (N6, caption), `dc6f087` (N1, docstrings, `arm_of`). Evidence labels: **measured** = a command I ran, output kept under `data/scratch/refactor_cr/{before,after}/` (gitignored scratch).

Behaviour is unchanged except for the two verifier edits the brief asked for (N1 and the label text of check 4.2), both listed below. No recorded wording changed, and the page shows the same status and sentence texts.

## Changes

| File | Change | Why it helps a reader |
|---|---|---|
| `vlapoc/planted_arms.py` | N2: deleted its copy of `treated_advantage`; it now does `from vlapoc.step4 import ALPHA, call, treated_advantage` at module top (the lazy import inside `statuses` is gone) and drops the unused `import math` | One definition of the advantage arithmetic. The import direction is ground-truth side -> method side, which is the allowed one. No cycle: `vlapoc.step4` imports nothing from `planted_arms` (**measured**: after `import vlapoc.step4`, `'vlapoc.planted_arms' in sys.modules` is `False`; `import vlapoc.planted_arms` works; `planted_arms.treated_advantage is step4.treated_advantage` is `True`). `planted_arms.py` is not in your list of files but N2 required the edit |
| `vlapoc/step4.py` | N2: the `treated_advantage` docstring now says it is the only copy and who imports it. N9: `_name_of` has one `from vlapoc.metrics_registry import SHORT_NAMES, entry` | The docstring no longer describes a duplicate that does not exist |
| `vlapoc/results.py` | N5: `provenance` also returns `reason_items`, a list of `{"kind", "run", "stage", "condition", "text"}` dicts, built once by a small local `reason(...)` helper. `reasons` is still the list of the same strings (`x["text"]`), in the same order, so every existing caller and verifier is untouched. The docstring documents both | `common.condensed_reasons` reads fields instead of parsing text that `provenance` had just formatted |
| `dashboard/common.py` | N5: `condensed_reasons` takes `reason_items` and groups on `(kind, stage, condition)` with no regex; `banner` passes `prov["reason_items"]` when `condense=True`. The one reason that names no run (a trace with no train or eval run) is shown as its recorded text. Audit: removed `functional_note_text` (a one-line wrapper of `st.caption`) | Removes a regex that had to stay in step with the format string in another file, and a wrapper that added nothing |
| `dashboard/views/curation.py` | Audit: `st.caption(metric_caption())` replaces `common.functional_note_text(...)`. N6: in the "Sentences for the arm selected" list a SYNTHETIC input no longer prints `SYNTHETIC:` twice per line (see Behaviour check) | The list reads `(SYNTHETIC: confirmed): Removing ...` instead of `(SYNTHETIC: confirmed): SYNTHETIC: Removing ...` |
| `scripts/dashboard_demo.py` | `arm_of` moved out of the loop in `round_a` to module level (with a docstring); `build` got a docstring | A function defined inside a loop body is hard to find; behaviour identical (**measured**: `arm_of('eval-demo-s1') == 'base'`, `arm_of('eval-demo-harm-f30-k5-s2') == 'harm-f30-k5'`) |
| `scripts/verify_dashboard.py` | N2: `--curation` 4.2 no longer imports `pa_adv` and asserts equality with it; it computes the expected rows straight from `step4.treated_advantage` and the check text says so. N1: `--demo` 6.5, see next section. Docstrings for the nested helpers `table_ok` and `argv` | One fewer thing to read in a 2,265-line file |

### Two changes to what a verifier checks (recorded explicitly)

1. **`--demo` 6.5 (N1).** Before: `store + cache < 2 GiB and free disk > 20 GiB` was pass/fail. Now: `store + cache < 2 GiB` is the check, and a `[WARN] 6.5 free disk is 19.4 GiB, below the 20 GiB ... (not a check here)` line is printed when free disk is under 20 GiB. Reason (analysis-r1 N1 and analysis-r2 N1, **measured** by the analyzer): the result tracked free space on the laptop, not the dashboard. It passed at 20.25 GiB free and failed at 17 to 19 GiB, while the store plus cache is about 36 MiB and did not change. The 20 GiB figure is the server-run runbook's rule for the server scratch disk (`.claude/skills/server-run/SKILL.md` l.25, l.184). Effect: `--demo` goes from 5/6 to **6/6** on this machine with no code change to the demo.
2. **`--curation` 4.2 (N2).** The check name changes from "(and equal planted_arms.treated_advantage, asserted)" to "(step4.treated_advantage is the only copy; planted_arms imports it)". The numerical comparison of the plot data against `treated_advantage` of the recorded verdicts is unchanged and still passes.

N6 is the only change to what the page prints. The recorded status and sentence (`SYNTHETIC: ...`) are untouched in the summary JSON, the table cells and the verifiers; only the bullet list drops the second copy of the prefix with `removeprefix("SYNTHETIC: ")`.

## Behaviour check

Run before any edit (HEAD `8828a72`, clean tree) and after the last commit (HEAD `dc6f087`, clean tree), each command in one sequence, tracked artifacts the verifiers rewrite restored with `git checkout` after every command (`git status` at the end: only `?? data.zip`, which I did not touch). I compared the `[PASS]`/`[FAIL]` lines and the summary line of each output after replacing run ids, chain ids (the chain id contains the git sha, `t-8828a721` vs `t-dc6f087c`), timestamps and durations. Those are the only "Formatting differences in timestamps or paths" I accepted; the one large duration (`--dropin` 8,317 s wall in the after run) is the machine being idle for part of it (user CPU 8m18s), not the code.

| Command | Before == After? | Note |
|---|---|---|
| `verify_training_attribution.py --step4` | Yes, 20 PASS / 0 FAIL both | Only the chain id in 1.5 differs (git sha) |
| `verify_dashboard.py --curation` | Yes, 22 PASS lines + R both, 0 FAIL | Only the 4.2 check text (item 2 above) |
| `verify_dashboard.py --app` | Yes, 11/11 | |
| `verify_dashboard.py --render` | Yes, 4/4 | Plots measured in headless Chrome again |
| `verify_dashboard.py --dropin` | Yes, 7/7 | Includes 5.5 (family O) and 5.2 (REAL re-tag). The drop-in banner path goes through the changed `condensed_reasons` |
| `verify_dashboard.py --demo` | **No, by design**: 5/6 -> 6/6 | 6.5 now a pass with a `[WARN]` (item 1 above). The other 5 checks are the same |
| `verify_dashboard.py --results` | Yes, 12/12 | The provenance case table prints the same label and reasons, only the 8-hex run ids in the scratch store differ |
| `--clusters` | Yes, 14/14 | |
| `--planted` | Yes, 7/7 | |
| `--discrepancy` | Yes, 10/10 | |
| `--round-b` | Yes, 13/13 | |
| `--planted-arms` | Yes, 8/8 | Exercises the `planted_arms` import change |
| `verify_attribution.py` | Yes, 32/32 | |
| `verify_training_attribution.py --blind` | Yes, `S1 pass: True` | |
| `... --functionals` | Yes, 17 PASS, rc 0 | |
| `... --recovery` | Yes, 11/11 | |
| `... --names` | Yes, 13/13 | |
| `... --groups` | Yes, 9/9 | |
| `... --integration` | Yes, 12/12 | Needs a clean tree, which the after run had |
| `... --pipeline data/scratch/refactor_cr/dry_<before|after>.txt` | Yes, 9 PASS | |
| `verify_runbook.py` | Yes, 10/10 | Includes 8.5 "no agent commit touches CLAUDE.md" (`commits touching CLAUDE.md after it: []`), on a committed tree |

Direct equivalence check for N5 (**measured**, `data/scratch/refactor_cr/cond.py`): for the demo's summary run (208 reasons), the ensemble run (9 reasons, including the run-less "trace ... no train run reached" reason) and the smoke attribute run (2 reasons), the `reasons` strings and the output of `condensed_reasons` are byte-identical before and after (`diff` empty).

Other checks: `data/lineage/mlflow.db` sha256 is `306af4123fda7e635892d459a2e34008ba1cb62f6c4fce152c546e0da7bdde79` (unchanged). No streamlit or Chrome process left running. I did not rebuild the demo store; no change touches what the builder writes (`arm_of` is a pure move). `scripts/dashboard_screenshots.py` is only exercised through `--render`.

One slip to report: while checking the tree I ran `git add -N .`, which marked `data.zip` as intent-to-add in the index. I noticed at once and undid it with `git rm --cached data.zip` (index only). The file was never opened, read, moved or committed, and `git status` shows it untracked again.

## Left as-is (and why)

- **N7, the `not shown` headline sentence.** It ends "A change that is not detected is not evidence that the curations leave the policies unchanged" right after saying the distributions differ. Rewording it changes a recorded string (in `step4.headline`, the demo summary and several checks), so it is a behaviour change. **Open suggestion for the owner:** the analyzer proposed "Not replicating a win is not evidence that either curation fails on its own metric."
- **N8, runbook check 8.5 fragility.** It fails again if the owner commits a CLAUDE.md edit after `bad0241`. That is an executor or owner decision about the recorded base commit, not a readability change.
- **`curation.py` `CURATION_OPS`.** Already `res.CURATION_OPS` in the final code (the r2 executor round removed the inline tuple); nothing to change.
- **`scripts/verify_dashboard.py` (2,265 lines) and `verify_training_attribution.py` (1,602 lines).** Large single files with a function per mode. Splitting them would be a wholesale reformat; I kept the structure and only added docstrings to the nested helpers I touched.
- **`dashboard/views/curation.py` `render` (about 150 lines).** It reads top to bottom as the five numbered page sections and shares many locals (`rows`, `verd`, `m`). Cutting it into functions would need a context dict passed around for no gain in understanding; the section comments mark the joins.
- **`app.py`, `verify_runbook.py`, `verify_training_attribution.py`, `dashboard_screenshots.py`.** Already have a module docstring with the exact command; the feature's changes there are a few lines and need no further comment.
- **`planted_arms.treated_advantage` as an importable name.** It still resolves (it is the imported function), so any older caller keeps working.

## Reading guide

Follow the data from the server's output to the page.

1. `tracker/features/2026-11-05-curation-results/brief.md`, then the "Headline" part of `plan.md`: the question ("add the curation headline results to the dashboard") and the rule for each state.
2. `vlapoc/step4.py`: where the words come from. Read the module docstring (PoC §9.1 Step 4, §4.5), then `call` (T/C/0/g per metric and seed), `status_of_calls` and `sentence_of` (the per-arm status and sentence), `headline` (the four-way rule: contradicted, shown, partly shown, `inconclusive` below ceil(2n/3) seeds with the gate, else `not shown`), `treated_advantage` (the single copy of the plot arithmetic), then `summarize` and `summary_run` (the row table recorded as `step4_summary.json`).
3. `vlapoc/results.py`, bottom half: `curation_runs` (finds the Round A chain and its summary run for a score run), `removed_episodes` (the full list of removed episodes, read from the child subset), then `provenance` and its `reason_items` (what makes a run REAL, SMOKE or SYNTHETIC).
4. `dashboard/common.py`: the cached readers `curation_of`, `curation_verdicts`, `removed_of`; `banner` and `condensed_reasons` (the banner and its "why this label" expander).
5. `dashboard/views/curation.py`: the page, in the order it draws: headline and its forest plot, the arm table with the metric picker, the family O section, one arm in detail (forest plot, removed episodes, video, lineage). Helpers first: `arm_label`, `control_label`, `row_keys`, `verdict_index`, `forest`.
6. `dashboard/app.py`: how the sixth page is registered (the `RENDER` table).
7. `scripts/dashboard_demo.py`, `round_a`: how the SYNTHETIC Round A in `data/lineage_demo` is built from the server dry run, using the real `select_run`, `verify_run` and `summary_run`. This is what the page shows in the screenshot.
8. `scripts/verify_dashboard.py --curation`, `--dropin`, `--demo`, `--render` and `scripts/verify_training_attribution.py --step4`: the evidence, in that order (words, page, drop-in on a server-shaped chain, demo store, real browser). `scripts/verify_runbook.py` check 8.5 guards CLAUDE.md.
9. `vlapoc/planted_arms.py`, `statuses`: the Round B planted-arm wording, which reuses `step4.call`, `ALPHA` and `treated_advantage`.

How to see it: `.venv/bin/streamlit run dashboard/app.py -- --lineage-root data/lineage_demo --scores attribute-demo-ens`, then choose **Curation results**.
