# Execution log: dashboard (demonstration surface), cluster and pattern discovery, server runbook

Evidence labels: **quoted**, **measured** (a command run in this session, output shown), **inferred**.

## Round 1 / Batch B1: S1, S2, S3, S9 (executor, 2026-10-22)

Baseline before any change (measured):
- Branch `dashboard`, HEAD `008405d`.
- `data/lineage/mlflow.db` sha256 `306af4123fda7e635892d459a2e34008ba1cb62f6c4fce152c546e0da7bdde79` (artifacts/b1_mlflow_sha_before.txt).
- `git diff --stat CLAUDE.md` before: `3 insertions(+), 2 deletions(-)` (owner's uncommitted edit; artifacts/b1_claude_md_diffstat_before.txt).
- `sliceline==0.3.0` installed into `.venv` (Python 3.14.0): `pip install sliceline==0.3.0` printed `Successfully installed sliceline-0.3.0`
  (only that one package; deps numpy/scikit-learn/scipy were already satisfied). `numba 0.68.0` and `llvmlite 0.50.0` were already
  in this venv before the install (pip did not install them). sliceline imports numba only as an optional accelerator
  (`sliceline/_numba_ops.py` header: "This module is optional"); S2's verifier also runs the slice finder with numba blocked.

(Per-story sections are appended below as each story lands.)

### S1: Record the influence matrix; `vlapoc/results.py` with a fail-closed provenance label
**Files:** `vlapoc/attribution.py` (+`featurized_rollouts`, `influence_index`; `attribute_run` now also writes `influence_index.json` and records `influence_matrix` and `influence_index` as outputs; the matrix row order is the same function `attribute()` uses) · `vlapoc/results.py` (new, read-only: `find_runs`, `resolve_run`, `score_table`, `functionals`, `seed_runs`, `influence`, `rollouts`, `verdicts`, `provenance`, `NotRecordedError`) · `scripts/verify_dashboard.py` (new; mode `--results`).
**Premises checked:** (1) the matrix was not a lineage output before → read `attribute_run` (only `attribution_scores`, `agreement`, `functionals` were `output_file`d) → holds. (2) row order of the matrix = rollouts with a trajectory sorted by `traj_episode_index`, column order = score table row order → read `attribute()`; the new `featurized_rollouts` is that code factored out, and AC1.1 checks the index against the score table → holds. (3) `ensemble_run` has no matrix → ensemble runs raise `NotRecordedError` and `results.seed_runs` returns the per-seed attribute runs (param `attribute_run_ids`); the dashboard's Rollouts page will use a seed run.
**Acceptance criteria:** (verifier: `.venv/bin/python scripts/verify_dashboard.py --results` → `results: pass=True (12/12, 7s)`, artifact `artifacts/verify_dashboard_results.json`)
- ✅ AC1.1: new `attribute_run` (SYNTHETIC `--integration`-style fixture, scratch store) → `recorded outputs: ['agreement', 'attribution_scores', 'functionals', 'influence_index', 'influence_matrix']`; `matrix (12, 14), tags (12, 14), index 12 rollouts x 14 episodes`; index column order equals the score table's `episode_id`/`content_hash` order; `rollouts()` rows map back to the index (`matrix_row`).
- ✅ AC1.2: a tampered matrix (one value changed in the scratch artifact) → `ValueError: run 96a635dc...: artifact rollout_episode_influence.npy differs from its recorded sha256 (output.influence_matrix.sha256); refusing to read it`. The real store's stand-in run (read in the scratch copy) → `NotRecordedError: influence matrix not recorded (run babbdb329ed6 'attribute-train-full-s0 [stand-in]' predates dashboard S1; re-run the attribute stage to record it)`.
- ✅ AC1.3: provenance cases on fake chains train→eval→attribute in a scratch store (artifacts/s1_provenance_cases.json): all-real → `REAL []`; one `vlapoc.synthetic=true` → `SYNTHETIC`; one `vlapoc.smoke=true` → `SMOKE`; one `eval.attribution_grade=false` → `SMOKE`; one missing tag (train run has no `vlapoc.smoke`) → `SMOKE`; one stand-in → `SMOKE`; plus an extra case, synthetic param `True` → `SYNTHETIC`. 6/6 required cases (7/7 with the extra) name the reason. A fixture attribute run (train/eval without server tags) is `SMOKE`, as the fail-closed rule requires.
- ✅ AC1.4: AST import scan of `vlapoc/results.py` → forbidden imports `[]` (checked against `vlapoc.groundtruth`, `vlapoc.recovery`, `vlapoc.discrepancy`).
- ✅ AC1.5: regression after S1 (logs in `artifacts/regress_s1/`): `verify_attribution.py` `pass: True (32/32)`; `verify_training_attribution.py` `--blind` (S1 pass), `--functionals` (18/18), `--pipeline` (9/9), `--step4` (13/13), `--recovery` (11/11), `--names` (13/13), `--groups` (9/9), `--integration` (12/12) all pass. `data/lineage/mlflow.db` sha256 after: `306af412...bde79`, unchanged. Tracked artifacts of the training-attribution feature that these verifiers rewrite (run ids, chain ids from the git sha, timings only: `git diff --stat` 8 files) were restored with `git checkout`. (`--local-smoke` is not run: it is the 8-minute real MPS chain; the change touches only what `attribute_run` writes after the tables are computed.)
- ✅ AC1.6: docstring check in the verifier finds `§2.7 (p. 5)`, `§3.4 (p. 7)`, `§2.1 item 1 (p. 3)`, `§2.1 item 4 (p. 3)`.
**Demo artifacts produced:** `artifacts/s1_provenance_cases.json` (the label table), `artifacts/verify_dashboard_results.json`.
**Notes / divergences from plan or spec:** REAL requires, besides the plan's conditions, at least one `train` and one `eval` run in the trace (otherwise the chain is not fully recorded, label SMOKE with that reason). This is the fail-closed reading of "a missing tag counts against REAL". `verify.attribution_grade` / `eval.attribution_grade.asserted` is not required for REAL (the plan names `eval.attribution_grade=true`); a possible tightening is listed under findings.
