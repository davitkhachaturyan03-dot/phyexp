# Changelog: training and attribution

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
