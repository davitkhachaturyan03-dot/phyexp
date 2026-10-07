# Refactor: evaluation layer

Refactorer: vla-refactorer, 2026-10-07. Inputs: `brief.md`, `plan.md` (r3), `execution-log.md`, `analysis-r1..r3.md`,
`dispute-r2.md`. Labels: **measured** (I ran it; outputs under `data/scratch/refactor_baseline/` and the session
scratchpad), **quoted**, **inferred**.

*Work in progress: this file is written incrementally. Sections below the findings are filled in as the work lands.*

## Findings recorded before any edit

### F1. The committed AC6.3 artifact is stale (round 1); HEAD's code writes a different column order

- **What differs (measured).** The baseline run of `scripts/verify_returns_s6.py --join data/eval/returns-3392bb4d5491/returns.parquet`
  on HEAD `553b3d1` (no edits) rewrote `artifacts/s6_verify_returns.json` with
  `"AC6.3" → "metric_columns": ["ret_f30", "ret_rmst60", "ret_auc_expected", "ret_auc_sampled", "ret_hrt60_lin", "ret_censored60"]`.
  The committed file (`git show HEAD:…/s6_verify_returns.json`) has
  `["ret_f30", "ret_rmst60", "ret_auc_sampled", "ret_auc_expected", "ret_hrt60_lin", "ret_censored60"]`.
- **Every other AC6.3 value is the same:** shape `[960, 8]`, `checkpoint_matches: true`, `check_scores` 960/960 `ok`,
  `stale_rejected: true`, and the stdout line `== RETURN_COLUMNS: True`.
- **Why (read from the source and git history).** Round 2 (`1c1671a`) made `returns.RETURN_COLUMNS` =
  `metrics_registry.return_columns("W")`, which lists `auc_h`'s columns as `ret_auc_expected, ret_auc_sampled`
  (registry entry order). The round-1 list had the two AUC columns the other way round. AC6.3 was not re-run in round 2,
  so the committed artifact still shows the round-1 order. The rest of `s6_verify_returns.json` was regenerated in
  round 2, because the script merges each mode's keys into the existing file.
- **Impact.** The set of columns is unchanged and the AC6.3 claim holds (the frame's columns equal `RETURN_COLUMNS`).
  Only the order in the recorded artifact is stale. This is a pre-existing difference between HEAD's output and the
  committed evidence. The refactor did not cause it. I restored the committed file and kept the regenerated copy at
  `data/scratch/refactor_baseline/changed_artifacts/s6_verify_returns.json`. Whether to refresh the evidence is the
  summarizer's or owner's call.

### Other baseline differences from the committed artifacts (bookkeeping only; committed versions restored)

- `artifacts/s0_verify_blind.json`: `timing_s.hash_and_blind_check` 2.0 → 1.9 and `total` 14.2 → 14.1. Wall-clock only.
- `artifacts/s5_ac5_9_grade_tests.json`: `default_run` id, and `runs_before_after` 38/38 → 43/43. The second is the
  number of runs in the scratch copy of the store. The real store has gained runs since round 2, and before == after
  still holds. Both refusal messages are identical.
- Every other artifact the baseline rewrote is byte-identical to the committed one (`git status` showed only these three),
  including `s7_ac7_1_seed_sweep.json`, `phail_validation.json`, both S2 figures, `s6_metrics_registry.json` and
  `s1_metric_registry.md`.
