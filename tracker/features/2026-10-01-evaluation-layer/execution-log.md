# Execution log: evaluation layer (PhAIL primitives on LIBERO)

Evidence labels as in CLAUDE.md: **measured** = a command was run here and its real output is quoted
(captured stdout under `artifacts/`); **quoted** = from a source, with file/line or page; **inferred** =
reasoning, unverified.

## Round 1: 2026-10-02 / Batch 1

Stories S0–S3 (S4–S6 batch 2, S7–S9 batch 3). Executor: vla-executor. Owner answers of 2026-10-02
(brief.md) applied: v1 leak option A (fix code, fence v1, no re-derivation); downloads approved except
PhAIL's 13.95 GB fine-tuning set. Spec pages re-read from the PDF this session (pypdf dump to the
session scratchpad): p.3 (§2.1, for S0).

### S0: Fix the answer-key leak (method-facing tables must not reveal edited episodes)

**Files:**
- `vlapoc/derive.py`: `derive_version` now writes the child with one LeRobot writer pass in parent
  episode order (new `_write_derived_dataset`, new `iter_stored_episodes`, rewritten
  `_derived_episode_table`). The pass writes to a temp dir and renames it on success. The child table
  has exactly the parent's columns. The pre-edit hashes go to a lineage-only `parents.parquet`.
  `compare_blind` runs before registration and fails the derivation on any offending column.
  `derivation.json` records `mechanism`, `episode_order`, `blind_check`, `parents_sha256`. The module
  docstring explains why `delete_episodes` + `merge_datasets` was replaced. `MECHANISM` is a constant.
- `vlapoc/lineage.py`: new section "blindness of derived versions":
  - `DATA_DERIVED_COLUMNS`, `BOOKKEEPING_COLUMNS`, `META_BOOKKEEPING_COLUMNS`;
  - `compare_blind`, `blind_report`, `check_blind`;
  - `parent_hashes` (documented LINEAGE-ONLY);
  - `fence_record`, `write_fence`, `assert_blind`;
  - CLI `version check-blind <ref>`.

  Also: the `version_episodes` docstring is corrected, and the dead `parent_content_hash` entry is
  removed from `GROUP_COLUMNS_EXCLUDED`.
- `vlapoc/catalog.py`: the comment on `CATALOG_COLUMNS` is corrected (derived tables have exactly these).
- `vlapoc/runs.py`: `derive_run`'s `mechanism` param now comes from the derive summary.
- `scripts/fix_v1_leak_s0.py` (new, idempotent). It moves v1's `parent_content_hash` to
  `parents.parquet` and rewrites v1's table and catalog copy with v0's 21 columns, in v0's row order.
  It then runs `check_blind`, writes `fence.json` (with before/after table sha256), and writes
  `artifacts/s0_fix_v1.json`. Backup of the two pre-fix tables: `data/scratch/s0_v1_tables_backup/`.
- `scripts/verify_blind_s0.py` (new). Scratch derivation (AC0.3) and negative tests (AC0.5).
- `scripts/verify_derive_s4.py`: AC4.1 now reads `parent_hashes(v1)`; AC4.7 now expects exactly
  `CATALOG_COLUMNS`. New `--artifacts DIR`, so a re-run doesn't overwrite the data+lineage feature's
  recorded evidence. Fixed a relative-path `print`.

**Premises checked:**
- P1 (`parent_content_hash` differs for exactly the 100 labelled rows). Checked with pandas over v1's
  version table + `labels.parquet`: `extra in v1 {'parent_content_hash'}`, `P1 changed 100 == labelled True`.
  **Holds** (measured).
- P2 (positional leak). Same script: `labelled episode_index min/max 900 999 contiguous tail True`; also
  `differs episode_index 988` (every row after the first edited one shifts). **Holds** (measured).
  - **New, third channel (measured):** v1's edited episodes also sit in their own data files.
    - v0 has 73 data files, the last holding episodes 991–999.
    - v1 has 81: `data/file_index` 73–80 hold episode_index 900–999.
    - `merge_datasets` keeps each source's files.
  - This channel is closed by the same order-preserving fix, and fenced for v1.
- P3 (`dataset_version_id` is order-independent). Checked by reading `vlapoc/lineage.py` `version_id`:
  `eps = sorted([str(e), str(h)] for e, h in pairs)`. After the rewrite, `version recompute v1` →
  `"match": true`. **Holds** (quoted source + measured).
- P4 (data-derived columns that differ v0→v1). Same comparison: `content_hash 100, smoothness 100,
  idle_fraction 9, idle_steps 9, gripper_toggles 100`. **Holds** (measured).
- LeRobot has no reorder or in-place replace tool. Checked by grepping the
  `lerobot/datasets/dataset_tools.py` defs: `delete_episodes, split_dataset, merge_datasets,
  modify_features, add_features, remove_feature, modify_tasks, recompute_stats,
  convert_image_to_video_dataset, reencode_dataset`. The order-preserving mechanism therefore uses the
  public writer (`LeRobotDataset.create/add_frame/save_episode/finalize`), the same one
  `libero_convert.convert` uses (measured grep).

**Acceptance criteria:**
- ✅ AC0.1: v1's tables have exactly v0's 21 columns, and the version id is unchanged.
  - `scripts/fix_v1_leak_s0.py` → `"equal_v0_columns": true, "n_columns": 21, "version_id_unchanged": true`;
    `unchanged_files: version.json true, files.json true, labels.parquet true`.
  - `python -m vlapoc.lineage version show v1` →
    `"dataset_version_id": "16076b9c5cb2a0e8c5d291fc0766977dc7286c6b9f87df39bd87fe5c17cd85f8"`.
  - `version recompute v1` → `"match": true` (`artifacts/s0_fix_v1_stdout.txt`, `artifacts/s0_v1_after_cli.txt`).
- ✅ AC0.2: the label join and `parent_hashes` work.
  - `fix_v1_leak_s0.py` → `"label_join_rows": 100, "labels": 100, "parent_hashes_rows": 1000,
    "parent_hashes_equal_v0": true, "labels_parent_hash_equal_parents_parquet": true`.
  - Re-checked by the updated `verify_derive_s4.py --v1`: `AC4.1 parent_hashes(v1) == v0 content_hash
    for all 1000 rows: True`; `AC4.2 ... join with v1 episodes on (episode_id, content_hash): 100/100 matched`.
- ✅ AC0.3: order is preserved on a scratch derivation. `scripts/verify_blind_s0.py` (20-episode parent
  converted from raw into a scratch lineage root; 3 edits at parent index 3/11/16, one adding 20 idle
  frames) →
  - `episode_index equal to parent's: 20/20; edited episodes now at [3, 11, 16]; length change on the
    padded episode: 99 -> 119`
  - `meta/episodes tasks per index equal to parent's: True; data-file episode ranges child {0: [0, 14],
    1: [15, 19]} vs parent {0: [0, 14], 1: [15, 19]}`
  - `check_blind(child) = []`; `scratch removed: True` (`artifacts/s0_verify_blind_stdout.txt`, `s0_verify_blind.json`).
- ✅ AC0.4: v1 reports only the positional channel, and is fenced.
  - Before the fix, `version check-blind v1` → `['(row order)', 'episode_index', 'parent_content_hash']`
    (`artifacts/s0_check_blind_v1_before.txt`).
  - After: `check_blind(v1) = ['episode_index']; fenced: True`.
  - `check_blind(v0) = []` (root version) (`artifacts/s0_v1_after_cli.txt`).
  - The fence record is `data/lineage/versions/16076b9c…/fence.json`, copied into `artifacts/s0_fix_v1.json`.
- ✅ AC0.5: the negative tests are rejected (`verify_blind_s0.py`):
  - extra `parent_content_hash` column → `['parent_content_hash']`;
  - episode_index of rows 0 and 5 swapped → `['(meta/episodes does not match the episode table)', 'episode_index']`;
  - `source_demo` changed on one row → `['source_demo']`;
  - rows shuffled → `['(row order)']`;
  - after restore → `[]`;
  - `assert_blind(v1)` refused: `version v1 (16076b9c5cb2) is not blind to its edits: fenced=True,
    offending=['episode_index']`; `assert_blind(scratch child)` passed.
- ✅ AC0.6: records in the data+lineage feature folder.
  - `tracker/features/2026-09-28-data-lineage-layer/changelog.md` has the entry "2026-10-02: Answer-key
    leak closed …". It covers both channels (column; row order + own data files 73–80), the fix, the v1
    disposition, and that its plan's AC4.1 required the column and AC4.7 asserted it.
  - In `report.html`: the S4 figure caption carries a dated correction; Open decisions has a new item 13
    (2026-10-02); "Changes since first release" has its first dated line.
  - In `tracker/index.html`, the §2.5 row's divergence cell notes the order fix, `check_blind` and the
    v1 fence.
- ✅ AC0.7: `grep -n "§2.1 item 4" vlapoc/derive.py vlapoc/lineage.py scripts/fix_v1_leak_s0.py
  scripts/verify_blind_s0.py` → 14 hits.
  - `derive.py`: module docstring l.5 and l.34, `derive_version` l.244, `_derived_episode_table` l.421.
  - `lineage.py`: `version_episodes` l.327, blindness section l.684, `parent_hashes` l.833.
  - Both new scripts' module docstrings.

**Other checks (measured):**
- `verify_derive_s4.py --guard --columns --cache --v1 --artifacts artifacts/s0_verify_derive_s4`
  (first run, 15:07):
  - `AC4.3 ... LineageError raised: True`;
  - `AC4.7 v1 columns == v0 catalog columns (CATALOG_COLUMNS, no parent_content_hash): True`;
  - `AC4.1 v1 loads: episodes=1000 frames=200340 ... recomputed id matches registered: True;
    verify_version files ok: True (85)`;
  - `AC4.8 shared HF_DATASETS_CACHE ... identical listing and bytes: True`.

  That run's summary JSON is `artifacts/s0_verify_derive_s4/s4_summary.json`.
- A second run, made only to capture stdout to a file (`artifacts/s0_verify_derive_s4/stdout.txt`), was
  cut off by the 600 s tool timeout and the session stall.
  - It got through `AC4.8 … isolated temp caches left`. Its scratch derivation logged
    `'hash_and_blind_check': 980.3` s, against 1.8–2.0 s in the two earlier runs.
  - A background re-run of `--cache` alone (`artifacts/s0_verify_derive_s4/cache_recheck_stdout.txt`)
    gave `'hash_and_blind_check': 3.1`, `identical listing and bytes: True`, `0 cache files`, so the
    stall did not reproduce.
  - The cause is unknown. Inferred: the `ProcessPoolExecutor` in `hash_stored_dataset` was starved or
    suspended during the session stall. Logged as a finding.
- Behaviour change (measured): with the new writer, `isolate_hf_cache=False` leaves `0 cache files
  (0 .arrow, 0 parquet builder entries)`. The new mechanism no longer loads datasets through HF
  `datasets`; the old one left 13 entries (data+lineage analysis-r1 E3).
- Cost: the scratch derivation wrote 4,210 frames at `write_frames_per_s: 375.9`. Full scale (200,340
  frames) is therefore ≈ 8.9 min (inferred extrapolation), against about 5 min for the old
  delete + merge mechanism (data+lineage log).

**Demo artifacts produced:** `artifacts/s0_check_blind_v1_before.txt` (before),
`artifacts/s0_fix_v1.json` + `s0_fix_v1_stdout.txt` + `s0_v1_after_cli.txt` (after),
`artifacts/s0_verify_blind.json` + stdout (order preservation and negative tests),
`artifacts/s0_verify_derive_s4/` (amended data+lineage S4 checks).

Two-row demo table:

| v1 leak channel | before | after |
|---|---|---|
| column `parent_content_hash` (differs for exactly the 100 labelled rows) | present in version table + catalog copy | removed; moved to lineage-only `parents.parquet` |
| row order (`episode_index` 900–999 = the 100 labelled; own data files 73–80) | open | fenced for v1 (`fence.json`, `assert_blind` refuses); fixed in `derive_version` for every new version |

**Notes / divergences from plan or spec:**
- `check_blind` returns unprefixed column names, as a union over the three tables; `blind_report` gives
  the per-table detail.
  - Two pseudo-columns exist: `(row order)` (a table's row sequence differs from the parent's) and
    `(meta/episodes does not match the episode table)` (the table's `episode_index`/length does not
    describe the physical copy).
  - Columns that are neither data-derived nor `episode_id` are treated as bookkeeping (strict
    default), so a new column can't pass silently.
- v1's two tables were also re-sorted into v0's row order, besides dropping the column. Without that,
  `check_blind(v1)` would also report `(row order)`. Nothing reads v1 by table row order (`Subset`
  sorts by `episode_id`; `recompute_version_id` uses a map; `version recompute v1` → match).
- In `meta/episodes`, data-file layout columns (`data/chunk_index`, `data/file_index`,
  `dataset_from/to_index`) are exempt as data-derived: a length-changing edit legitimately moves later
  boundaries. With a sequential writer in parent order, edited episodes cannot be grouped into their
  own files. The scratch derivation shows identical file ranges.

### Findings for the analyzer (S0)
- **Third leak channel (measured).** v1's edited episodes are in their own data files (73–80), via
  `merge_datasets`. It is closed for new versions by the order-preserving writer, and fenced for v1.
- **Cross-version channel (inferred, not built).** `parent_hashes(v1)`, v0's own episode table and
  `diff_versions(v0, v1)` all name the edited episodes. That is correct for lineage and evaluation code,
  but a method under test must be confined to the derived version (plus its reference subset). The
  docstrings say so. Nothing enforces it beyond `assert_blind` on the input version: screening and
  attribution don't exist yet, so the entry-point call belongs to those features.
- **`data/lineage_scratch/`** (data+lineage leftover, 2.3 GB) still holds a stale v1 table with the leaked
  column. It is read by nothing. Not touched: the owner decides on cleanup (data+lineage report, Open
  decisions).
- **One unexplained 980 s stall** in `hash_stored_dataset` during a scratch derivation. It did not
  reproduce (3.1 s on re-run).

### S1: Metric layer: PhAIL primitives over (T, E) pairs, with LIBERO-ready cells

Re-read this session: PoC pp. 6–8 (§3.1–3.5) from the PDF. The PhAIL paper pp. 4–6 (§3.1 T pairs,
Table 1, §3.2 HRT / P-P / CIs, §3.3 macro-KS + logrank) was read from
`docs/papers/PhAIL_Real-Robot_VLA_Benchmark_arXiv-2605.29710.pdf`. Also read in the clone: `build/stats.py`
(all of it), `build/explore_pp_plot.py:63-216`, `build/fig_qq_tradeoff.py:40-216`,
`build/tab_leaderboard.py:1-90` and `build/explore_separation.py:61-160`.

**Download (approved, brief 2026-10-02):** `https://github.com/Positronic-Robotics/phail-paper`.
- Revision: `18ce72d5703dcbbbb10a980336aa5a1622601fb4` (2026-05-27, "Add Apache 2.0 LICENSE and drop stale
  operator clip-generation script"; the head of `main`, per the GitHub API).
- Fetched with depth 1 into `data/external/phail-paper/` (gitignored under `data/`).
- Size (measured `du`): 87 MB in total, of which 12 MB is `.git`. The plan said ~52 MB, and the GitHub
  API `size` field says 32,475 KB.

**Files:**
- `vlapoc/third_party/phail_stats.py`: verbatim `build/stats.py`.
- `vlapoc/third_party/LICENSE-phail-paper`: verbatim `LICENSE`.
- `vlapoc/third_party/__init__.py`: the attribution header (repo, commit, sha256, "unmodified", the
  Apache-2.0 §4 reasoning).
- `vlapoc/evaluation.py` (new):
  - `Episode` plus `episode_from_ops`, which goes through PhAIL's `episode_to_T_pairs`;
    `libero_episode`, with threshold 0 and no ghosts.
  - Functionals of the KM F: `F_at`, `median_T`, `rmst`, `completion`, `uph` (+ `uph_conditional`),
    `mtbfa`, `ghost_event_rate` (+ `ghost_fraction`), `censored_rate`.
  - `macro`, `pooled`, `hrt_cells`, `hrt`; `auc_censored`, `auc`, `macro_auc`; `pp_curve`,
    `macro_pp_curve`; `ks`, `macro_ks`; `logrank_stratified`.
  - Bootstrap: `cluster_bootstrap` (episode or operation unit; fixed arms), `hrt_ci` (both variants),
    `F_band`; and `uph_mtbf_sweep`.
  - The `METRICS` registry plus `metric_registry()`, and `plot_km` / `plot_pp`.
- `scripts/verify_metrics_s1.py` (new): the synthetic suite.

**Premises checked:**
- `build/stats.py` imports only numpy. Checked with `grep -n "^import\|^from"` → `from __future__ import
  annotations`, `from typing import Protocol`, `import numpy as np`. **Holds** (measured).
- LICENSE is Apache-2.0, "Copyright 2025-2026 Positronic Robotics". Read the file in the clone; the
  GitHub API says `spdx_id: NOASSERTION`. **Holds** (quoted file).
- Apache-2.0 §4(b): "You must cause any modified files to carry prominent notices stating that You
  changed the files" (quoted LICENSE l.97–98), and the repo has no NOTICE file (`ls`). So an unmodified
  copy plus the license text satisfies §4. **Holds.**
- lifelines and statsmodels are not installed (`importlib` → `MISSING`). scipy 1.18.1 `stats.logrank`
  signature is `(x, y, alternative)`, so it has no strata; `stats.ecdf(sample)` takes `CensoredData`.
  pytest is not installed. **Measured.** As a result, the stratified logrank is written here (cross-checked
  against scipy), and the suite is a script.
- The vendored `_collect_T_E` always uses the default `multi_grip_threshold_s = 0.3`, while
  `episode_to_T_pairs` accepts the threshold. Read in the source. **Holds.** So `evaluation.py` stores
  pairs per episode and re-runs the bootstrap loops itself, in the same RNG order; the equivalence is
  checked below.

**Acceptance criteria:**
- ✅ AC1.1: `verify_metrics_s1.py` → `vendored sha256 802387d343463cd4… == clone build/stats.py: True
  (clone HEAD 18ce72d5703dcbbbb10a980336aa5a1622601fb4); LICENSE identical: True; attribution header:
  True`. `shasum -a 256` gives `802387d343463cd46a56e8d1dc4b4917b90c34b51fc72acce5b6e6cb915b46f2` for both
  files.
- ✅ AC1.2: `AC1.2 SYNTHETIC finite censored data (400 pairs, 256 events, 226 tied values): max
  |S_vendored - S_scipy.ecdf| over 146 event times = 3.331e-16; Greenwood 95% half-width max diff =
  9.714e-17`.
- ✅ AC1.3 (to floating point):
  - `55 operations, 5 ghosts (T=inf), 0 censored: ghost_event_rate = 0.09090909090909088, g/n =
    0.09090909090909091, |diff| = 2.8e-17`.
  - `RMST(30 s) = 14.55964694266428; mean(min(T, 30)) with ghosts -> 30 = 14.55964694266428; |diff| = 0.0e+00`.
  - "Exactly" holds to 2.8e-17: the KM cumulative product rounds in the last bits.
- ✅ AC1.4:
  - `zero-success episode (τ_episode 60 s, 2 ops) -> ((60.0, 0),); all ops done -> ((5.0, 1), (7.0, 1))
    (no censored tail); 1 of 2 done -> ((8.0, 1), (52.0, 0)); LIBERO simultaneous ops (threshold 0) ->
    ((10.0, 1), (0.0, 1))`.
  - `the zero-success episode stays in the risk set: 5 pairs, S after the first event = 0.8 (= 1 - 1/5)`.
- ✅ AC1.5, on 40 episodes × 5 correlated ops (SYNTHETIC):
  - RMST(30) = 6.3719: `clustered 95% CI [5.1479, 7.7031] vs per-operation [5.4706, 7.2974]; width ratio 1.399`.
  - F(10) = 0.8050: `[0.7350, 0.8701] vs [0.7500, 0.8600]; width ratio 1.228`.
- ✅ AC1.6: `registry: 14 entries, every named function exists with a docstring; required missing: []`.
  The table is in `artifacts/s1_metric_registry.md` (name → function → functional of F → τ → source).
- ✅ AC1.7: `citations in vlapoc/evaluation.py: {'PoC §3.2 (p. 6)': 1, 'PoC §10 (p. 20)': 1, '§3.1 p. 4':
  7, 'Table 1 p. 5': 2, '§3.2 p. 5': 6, '§3.3 p. 6': 3}`. The module docstring also cites App. L.

**Equivalence checks (SYNTHETIC, measured), ahead of S2:**
- macro-KS, on 3 cells with ghosts and censoring: `vendored per_obj_macro_bootstrap_ks
  D=0.1902324217651046 p=0.6866267465069861; evaluation.macro_ks D=0.1902324217651046
  p=0.6866267465069861; identical: True`.
- Logrank, one stratum with ties: `ours chi2=0.5102598713 p=0.4750263814; scipy Z^2=0.5102598713
  p=0.4750263814; |dchi2|=5.55e-16`.
- AUC: the hand example gives `0.666667 (expected 2/3)`, and a ghost vs a 5 s event gives `0.0`. On
  uncensored synthetic data, `P-P area 0.35813 vs AUC 0.35817 (|diff| 3.9e-05)`, the P-P area claim of
  PhAIL §3.2 p. 5.
- HRT, 2 cells: `macro 0.6056 (= mean: True)`; `CI resample human [0.5191, 0.7111] vs human fixed
  [0.5486, 0.6708]`.

**Demo artifacts produced:** `artifacts/s1_metric_registry.md` (the registry table),
`artifacts/s1_verify_metrics.json`, `artifacts/s1_verify_metrics_stdout.txt`.

**Notes / divergences from plan or spec:**
- **AC1.1 wording conflict, resolved.** The AC says "byte-identical … and its license header is present".
  A header inside the file would break byte identity, so the header lives in
  `vlapoc/third_party/__init__.py`, next to `LICENSE-phail-paper`. Apache-2.0 does not require a header
  in an unmodified file.
- **Restated formulas, not imported.** `auc_censored`, `pp_curve`, `macro_pp_curve`, `macro_auc`, UPH
  and MTBF/A, and the logrank arithmetic are restated from PhAIL's scripts, with each source function
  cited. Those scripts import PhAIL's loader, `configuronic` and matplotlib at module level, so they
  can't be imported here. S2 checks equality on PhAIL's data.
- **`ghost_event_rate` is the KM mass at T = ∞.** With censoring, KM redistributes censored mass to the
  right, including to ∞. That is PhAIL's "gap between the asymptote and 1". `ghost_fraction` (raw g/n)
  is kept as a diagnostic.
- **`censored_rate` = share of episodes ending in a censored pair.** This makes it the mean of S6's
  per-rollout `ret_censored`, by construction.
- **UPH adopted = 3600 / RMST(τ_episode)** (PhAIL `fig_qq_tradeoff.py`). Table 1's conditional form is
  `uph_conditional`, computed in closed form from the KM: E[T | T ≤ τ] = (RMST − τ(1 − F(τ))) / F(τ).
  The full three-form disclosure is S2 AC2.8.
- **`mtbfa` returns `inf` when F(τ_episode) = 1.** PhAIL's figure code clips the denominator at 1e-9.
- **`uph_mtbf_sweep` uses the exact step-integral RMST.** PhAIL's figure uses a 1500-point Riemann sum,
  so figure-level differences are expected.
- **HRT is returned as a ratio.** PhAIL reports it ×100 as a percentage.
