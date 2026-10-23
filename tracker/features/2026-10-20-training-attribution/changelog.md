# Changelog: training and attribution

## 2026-10-22: the documented first server run is now Round A on `v3/candidate` (dashboard feature, owner answer Q1)
- **Why:** the dashboard feature's owner answer Q1 ("Planted-bad-data v3 (Recommended)", `tracker/features/2026-10-22-dashboard/brief.md`) replaces the first-run ruling of the entry below ("112 jobs, ~96 GPU-h" on `v0/candidate`).
- **Grounding:** PoC §2.1 item 4 p.3 (planted ground truth), §9.1 Steps 3-4 p.19, §9.2 p.19; the dashboard plan (AC8.2b).
- **Prior art:** nothing new: only the documented command and the counts change; the chain code is the same plus three CPU jobs.
- **Files:** `scripts/server_pipeline.py` (module docstring "FIRST SERVER RUN" now `--tag v3cand --subset v3/candidate --curation per_task --metrics f30 hrt60_lin`; the chain also runs `clusters` after the ensemble and, with an answer key, `score-clusters` and `discrepancy`), `scripts/verify_training_attribution.py` (`--pipeline` check 4.7 asserts the v3 count; 4.6/4.7 default count 313 -> 314), `artifacts/s4_dry_run_first_run.txt` (regenerated for v3), `report.html` ("Changes since first release").
- **Evidence:** `.venv/bin/python scripts/server_pipeline.py --tag v3cand --subset v3/candidate --curation per_task --metrics f30 hrt60_lin --dry-run` -> `would submit 116 jobs` (24 training runs; the dashboard plan's measured 113 plus `clusters`, `score-clusters`, `discrepancy`). The `v0/candidate` variant is 113 jobs (112 plus `clusters`), 24 trainings. The script default (`--curation both`, four metrics, v0/candidate) is 314 jobs (313 plus `clusters`). `verify_training_attribution.py --pipeline` check 4.7 passes (see the dashboard execution log, batch B2).
- **Divergence:** the first-run size changes from 112 to 116 jobs and the dataset from v0/candidate to v3/candidate by the owner's answer; training count (24) and GPU-hours (~96, inferred) are unchanged.

## 2026-10-22: owner rulings applied (±1 returns by default, first-run command, group rule approved)
- **Why:** owner rulings after round 2 PASS, recorded in `brief.md` "Owner rulings" (commit `47632aa`): "CUPID's +1/−1" for return weighting; "112 jobs, ~96 GPU-h"
  as the first server run; "Keep 2-of-5, documented" for the group rule. (A fourth, "Merge: not yet", needs no change; nothing merged or pushed.)
- **Grounding:** CUPID App. A.2 p.19 (R = +1 success, −1 otherwise) and the released `error_util.compute_demo_quality_scores` `net` score (`docs/papers/CUPID_Influence_Functions_arXiv-2506.19121.pdf`); PoC §4.2 p.10 (groups), §9.1 Step 4 p.19, §10 p.20 ("Compute budget" open decision).
- **Prior art:** nothing new needed: the ±1 encoding already existed as `return_encoding="pm1"` and the vendored CUPID functions; this change only flips the default.
- **Files:**
  - `vlapoc/attribution.py`: `return_encoding` now defaults to `"pm1"` in `column_plan`, `score_table`, `column_declarations`, `attribute`, `attribute_run` (the primitive `performance_influence` keeps `"raw"` as its own default); yes/no columns are CUPID ±1 uncentred, continuous columns centred (`--no-center` still turns that off); `weighting_label`; every `functionals.json` row has a `weighting` field and the run params an `return_weighting` summary; the `centred_returns` entry of `DIVERGENCES` now says centring applies only to continuous columns; docstrings updated.
  - `--cupid-returns` is **retired** (redundant: it is the default now) and replaced by `--raw-returns` (restores centred yes/no columns), in `vlapoc/attribution.py` and `scripts/stage.py`.
  - `scripts/server_pipeline.py`: module docstring documents the first server run (`--curation per_task --metrics f30 hrt60_lin`, 112 jobs, 24 trainings, ~96 GPU-h inferred) and that global curation runs later; defaults unchanged.
  - `vlapoc/analysis.py`: `DIVERGENCE_NOTE` now says "owner-approved 2026-10-22" (no behaviour change).
  - `scripts/verify_attribution.py` (checks 7e-7g), `scripts/verify_training_attribution.py` (2.5b, 4.7).
  - `tracker/features/2026-10-20-training-attribution/artifacts/s4_dry_run_first_run.txt` (new), `report.html`.
- **Evidence:** see the final run below.
- **Divergence:** none from the owner rulings. Scores computed before this date used centred yes/no columns and are not comparable (the run params' `return_encoding` tells which).
- **Group rule:** now owner-approved as a recorded divergence from §4.2 p.10 ("every metric"); no code change beyond the note.
- **Evidence (HEAD after this entry's code commit):** `verify_attribution.py` 32/32 (new 7e: default `score_table` yes/no column equals the vendored CUPID `net` score, max abs err 4.2e-17;
  7f timed-out column and centred continuous column; 7g `--raw-returns` restores the centred version). `verify_training_attribution.py --functionals` 18/18 (new 2.5b: a default attribute run's params and every `functionals.json` row state
  "CUPID ±1, uncentred" for yes/no columns and "centred (R − mean R)" for continuous ones), `--pipeline` 9/9 (new 4.7: `--curation per_task --metrics f30 hrt60_lin` = 112 jobs, 24 trainings; script default still 313),
  `--blind`, `--step4`, `--recovery`, `--names`, `--groups`, `--integration` all rc=0. Older verifiers (`verify_lineage_s3`, `verify_runs_s5 --r2`, `verify_blind_s0`, `verify_method_boundary_s3`,
  `verify_derive_gt_s2`, `verify_verdict_s7`, `verify_returns_s6 --synthetic`, `verify_metrics_s1`, `verify_consumers_s6`) all rc=0. `data/lineage/mlflow.db` sha256 `306af4123fda7e63…` unchanged.
  Other features' rewritten artifacts (ids/timings; plus the known structural drift of `data-lineage-layer/s3_summary.json`) restored with `git checkout`; this feature's artifacts refreshed by the final run are committed.
