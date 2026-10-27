# Refactor: dashboard (demonstration surface), cluster and pattern discovery, server runbook

Stage 4 of the feature pipeline. Branch `dashboard`, started at HEAD `5d278d5` (analyzer round 3, PASS). Rule followed: no behaviour change, small safe edits.
Evidence labels: **measured** (a command run in this session) or **inferred**.

Starting point (measured, with `ast` scripts under `data/scratch/refactor/`): every module in scope already has a module docstring with its place in the
pipeline and its grounding line, and every public function in `vlapoc/` has a docstring. The gaps were in `dashboard/` (page `render` functions, the cached
readers in `common.py`, the empty `views/__init__.py`) and in the `main()` of the scripts. There were no debug prints, no `TODO`/`FIXME`, no unreferenced private
functions. One module had two unused imports and one reading-order line named a function that does not exist. The edits are limited to those.

## What changed
| File | Change | Why it helps a reader |
|---|---|---|
| `dashboard/views/__init__.py` | Added a one-line module docstring: the five pages each expose `render(ctx)`, `app.py` picks one, `common.header` builds `ctx` | The package was empty and undocumented; this is the contract between `app.py` and the pages |
| `dashboard/views/{overview,rollouts,episodes,clusters,bad_vs_harmful}.py` | Docstring on each page's `render(ctx)`; one on `clusters.heatmap_png` (the `.axes.json` beside the PNG is what `verify_dashboard.py --render` reads) | Each page entry point now says what the page shows without reading 70 lines of Streamlit calls |
| `dashboard/views/bad_vs_harmful.py` | `arm_name` moved from a nested function inside `render` to a module-level function with a docstring (body identical) | A reader no longer meets a function definition in the middle of the page's table code, and the arm wording can be found by name |
| `dashboard/common.py` | Docstrings on `Config`, the ten cached readers (`scores_of`, `catalog_of`, `provenance_of`, `run_info`, `runs_from`, `table_of`, `json_of`, `influence_of`, `rollouts_of`), `mlflow_run_url`, `trace_command` | Each reader now names the `vlapoc.results` function it wraps, so the "dashboard reads only through `results.py`" rule can be followed by eye |
| `vlapoc/discrepancy.py` | Removed unused imports `DEFAULT_METRICS` and `REPO_ROOT`; the module docstring's "Reading order" named `quadrants`, which does not exist, now `quadrant_of` | Dead imports hide what the module depends on; the reading order must name real functions |
| `scripts/stage.py`, `dashboard_demo.py`, `dashboard_screenshots.py`, `repoint_lineage_store.py`, `verify_dashboard.py`, `verify_runbook.py` | One-line docstring on `main()` pointing at the module docstring's usage block | These files already open with the exact command; `main()` now says where to find it |

Commit: `543d6cb`. Python bytecode is the only thing that differs at run time (docstrings, two unused imports, one function moved up a level).

## Behaviour check
Method (measured): `data/scratch/refactor/run_all.sh <dir>` runs every command below and saves one log per command (`before/` on `5d278d5`, `after/` on `543d6cb`);
The per-check lines (check id plus PASS/FAIL, the `pass=True (n/n)` summary line, `rc=`) of the two runs are compared with `diff`. The only differences are run-to-run noise: timings (`14.1s` / `14.0s`), MLflow run ids, scratch directory names, the git sha inside a chain name (`t-5d278d55` / `t-543d6cb4`), the throughput of a video conversion. No check line, count or verdict differs. The demo store was not rebuilt
(see "Left as-is"); `--demo` and `--app`, `--render`, `--dropin` read the existing `data/lineage_demo`.

| Command | Before == After? | Note |
|---|---|---|
| `verify_dashboard.py --results` | yes, 12/12 in both | every `[PASS]` check id identical |
| `--clusters` | yes, 14/14 in both | every `[PASS]` check id identical |
| `--planted` | yes, 7/7 in both | every `[PASS]` check id identical |
| `--discrepancy` | yes, 10/10 in both | every `[PASS]` check id identical |
| `--round-b` | yes, 13/13 in both | every `[PASS]` check id identical |
| `--planted-arms` | yes, 8/8 in both | every `[PASS]` check id identical |
| `--app` | yes, 11/11 in both | every `[PASS]` check id identical |
| `--render` | yes, 4/4 (real headless Chrome) in both | every `[PASS]` check id identical |
| `--demo` | yes, 6/6 in both | every `[PASS]` check id identical |
| `--dropin` | yes, 4/4 in both | every `[PASS]` check id identical |
| `verify_runbook.py` | yes, 10/10 in both | every `[PASS]` check id identical |
| `verify_attribution.py` | yes, 32/32 in both | every `[PASS]` check id identical |
| `verify_training_attribution.py --blind` | yes, pass in both | every `[PASS]` check id identical |
| `--functionals` | yes, 18/18 in both | every `[PASS]` check id identical |
| `--pipeline data/scratch/dash_exec/dry_pipeline.txt` | yes, 9/9 in both | every `[PASS]` check id identical |
| `--step4` | yes, 13/13 in both | every `[PASS]` check id identical |
| `--recovery` | yes, 11/11 in both | every `[PASS]` check id identical |
| `--names` | yes, 13/13 in both | every `[PASS]` check id identical |
| `--groups` | yes, 9/9 in both | every `[PASS]` check id identical |
| `--integration` | yes, 12/12 in both | every `[PASS]` check id identical |
| `verify_lineage_s3.py` | yes, rc=0 and the same check lines in both | |
| `verify_runs_s5.py --r2` | yes, rc=0 and the same check lines in both | |
| `verify_blind_s0.py` | yes, rc=0 and the same check lines in both | |
| `verify_method_boundary_s3.py` | yes, rc=0 and the same check lines in both | |
| `verify_derive_gt_s2.py` | yes, rc=0 and the same check lines in both | |
| `verify_verdict_s7.py` | yes, rc=0 and the same check lines in both | |
| `verify_returns_s6.py --synthetic` | yes, rc=0 and the same check lines in both | |
| `verify_metrics_s1.py` | yes, rc=0 and the same check lines in both | |
| `verify_consumers_s6.py` | yes, rc=0 and the same check lines in both | |
| `data/lineage/mlflow.db` sha256 | yes | `306af4123fda7e635892d459a2e34008ba1cb62f6c4fce152c546e0da7bdde79` before and after |
| all 29 commands | yes | rc=0 in both runs; 215 `[PASS]` lines in each, 0 `[FAIL]` |

## Left as-is (and why)
- **`.claude/skills/server-run/SKILL.md`**: not edited. The listed rule is readability only with every command, count, marker and verified/unverified tag kept exactly, and `verify_runbook.py` parses it; the file already has one numbered step per stage with its tag. Nothing in it was misleading enough to justify a text change.
- **Long lines and dense list-comprehensions** in `dashboard/views/*.py` and `scripts/verify_dashboard.py` (1,824 lines, one function per check): kept. They match the surrounding style (inferred: the existing files use long lines throughout), the 14-check verifier is a flat list of independent checks, and reformatting would touch hundreds of lines for no behaviour.
- **`import mlflow  # noqa: F401`** in `scripts/dashboard_demo.py` and `scripts/verify_dashboard.py`: an intentional import side effect (they must fail early when MLflow is missing), not an unused import.
- **Nested helper functions without docstrings** inside the verifier and demo-builder bodies (`check`, `refuse`, `run_case` ...): local to one check, named for what they do.
- **`dashboard/common.py` `label()` ignores its `decls` argument**: kept, because every page calls it with two arguments and the signature is the place the declared functional is meant to come back to (see its docstring).
- **Demo store `data/lineage_demo` and the demo build (`dashboard_demo.py build --with-smoke --force`, about 8 minutes on MPS)**: not rebuilt, because no line the build runs changed except one docstring. `--demo` 6/6 (the build's own provenance and manifest checks on the existing store) is part of the behaviour check.
- **`--local-smoke`** of `verify_training_attribution.py`: not run, as in every earlier round (an 8-minute real MPS chain; the demo build contains it).
- **Restored artifacts**: the verifiers rewrite tracked files (run ids, timings: 23 files under `tracker/features/`, including the data-lineage `s3_summary.json` and `s5_r2_summary.json`); both runs were followed by `git checkout -- tracker/`, so none is in a commit. `data/lineage/mlflow.db` sha256 is `306af4123fda7e635892d459a2e34008ba1cb62f6c4fce152c546e0da7bdde79` before and after.

## How to read the code
Follow the data, not the directory listing. The feature adds a read side (the dashboard), two analysis stages on top of the existing attribution (clusters, the
bad-vs-harmful table), a second retraining round (Round B) and a runbook. `docs/dataflow.md` shows where each sits in the whole pipeline.

**1. The contract: what a run records and how it is read back.**
- `vlapoc/attribution.py` (`attribute_run`, `influence_index`): the attribute stage now also records the rollout x episode influence matrix and its row/column index. That is what makes the first dashboard view possible.
- `vlapoc/results.py`: **start here.** The only reader of the lineage store for the dashboard and the cluster stage. `find_runs` -> `score_table` / `influence` / `rollouts` / `clusters` -> `provenance`. Every artifact is checked against the sha256 its run recorded, and `provenance` decides SYNTHETIC / SMOKE / REAL from the recorded chain (fail-closed: a missing tag counts against REAL).
- `vlapoc/metrics_registry.py`: one table of the PoC metrics (functional, family, verdict metric); `SHORT_NAMES` is the plain-name mapping the dashboard and the cluster pattern text share.

**2. Stages that discover and test (each is one `scripts/stage.py` subcommand, one MLflow run).**
- `vlapoc/clusters.py` (stage `clusters`, method side, never reads the answer key): `descriptor_table` -> `bin_descriptors` -> the three lenses `label_clusters`, `slice_clusters` (sliceline), `group_clusters` -> `dedupe` -> `pattern_of` (the sentence the dashboard shows) -> `clusters_run`. Clusters are only *estimated* (first-order sums, CUPID p. 11).
- `vlapoc/recovery.py` (`cluster_recovery_run`, stage `score-clusters`, ground-truth side): scores each discovered cluster against the planted corruption types, only after the cluster run exists.
- `vlapoc/discrepancy.py` (stage `discrepancy`, ground-truth side, PoC §9.2): `quadrant_of` -> `type_table` (Welch interval, Holm across metrics) -> `filter_flags` / `heuristic_evidence` -> `discrepancy_run`. "Looks bad" against "hurts", estimated.
- `vlapoc/confirm.py` (stage `confirm`): turns Round B verdicts into `confirmed` / `refuted` / `not confirmed` per cluster and metric: `status_of` -> `cluster_status` -> `confirm_run`.
- `vlapoc/planted_arms.py` (ground-truth side): the same question tested causally, by retraining without each planted type: `build_arm` -> `planted_select_run` -> `statuses` -> `discrepancy_verdict_run`. Method entry points refuse these arms.
- `scripts/stage.py`: the module docstring lists every subcommand (`clusters`, `score-clusters`, `discrepancy`, `confirm`, `select-planted`, `discrepancy-verdict` are new). Each `cmd_*` is a thin call into one of the modules above. `scripts/server_pipeline.py`: builds the SLURM chain (`--dry-run` prints the job and training counts; `--round-b` and `--round-b-planted` are the new flags).

**3. The dashboard (read-only; `.venv/bin/streamlit run dashboard/app.py -- --lineage-root data/lineage_demo --scores attribute-demo-ens`).**
- `dashboard/app.py` (33 lines): sets up the page, builds `ctx` through `common.header`, renders the page picked in the sidebar.
- `dashboard/common.py`: configuration, the cached readers (each wraps one `vlapoc.results` function), the provenance banner, the header and run picker.
- `dashboard/words.py`: the plain-language names (metric keys, descriptors, planted types, quadrants).
- `dashboard/views/`: `overview` (what the numbers are, how the label was decided) -> `rollouts` (pick a rollout, see the training episodes that pushed it, with video) -> `episodes` (ranked lists, aggregate by catalog column) -> `clusters` (heatmap, scatter, cluster table with pattern, estimated or confirmed) -> `bad_vs_harmful` (§9.2). Video comes from `vlapoc/video.py` (H.264 from LeRobot frames, cached on disk).
- `scripts/dashboard_demo.py build [--with-smoke] [--force]`: makes `data/lineage_demo`: real videos, catalog and rollouts, SYNTHETIC scores (`vlapoc/synthetic.py`), and the real 2-step SmolVLA chain labelled SMOKE. This is what the dashboard shows when no server run exists. `scripts/dashboard_screenshots.py` drives headless Chrome for screenshots and the rendering measurements.

**4. The runbook (for a Claude session on the GPU server).**
- `.claude/skills/server-run/SKILL.md`: 15 steps, each tagged verified locally or unverified until a server session: preflight, inputs and smoke (1-3), Round A dry run and submit (4-5), monitor and recovery (6-7), clusters and discrepancy by hand (8), the real dashboard (9), Round B for clusters (10-11) and for the planted data (12-13), results back (14), known risks (15). It stops and asks the owner at every marked point.
- `scripts/repoint_lineage_store.py`: used in steps 2 and 14: a copied MLflow store keeps absolute artifact paths, so this rewrites them (with a read check) before the dashboard can open it.
- `scripts/verify_runbook.py`: proves the runbook's commands exist, its dry-run counts match a fresh `server_pipeline.py --dry-run`, and every step is tagged, at least 7 STOP AND ASK markers exist, and `CLAUDE.md` is untouched by the batch.

**5. How it is proven.** `scripts/verify_dashboard.py` has one mode per story group (`--results`, `--clusters`, `--planted`, `--discrepancy`, `--round-b`, `--planted-arms`, `--app`, `--render`, `--demo`, `--dropin`); its module docstring lists each mode and the criterion it checks. `--render` opens the running app in headless Chrome and measures plot areas; `--dropin` shows that a real run (not a synthetic one) is shown by the same code.
