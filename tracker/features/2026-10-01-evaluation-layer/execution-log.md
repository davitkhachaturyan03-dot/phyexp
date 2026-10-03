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

### S2: Validate the metric layer on PhAIL's released real-robot data, and answer the §3.5 B data question

**Environment created:** `.venv-phail/` (gitignored by the new `.venv-*/` rule in `.gitignore`).
- Python 3.12.13, made with `uv venv --python 3.12`.
- Packages pinned to the clone's `uv.lock`: pos3 0.2.2, configuronic 0.4.0, numpy 2.4.4, scipy 1.17.1,
  matplotlib 3.10.8, pyarrow 23.0.1, boto3 1.42.83; pandas 3.0.6 (unpinned, used for the parquet export).
- 348 MB, plus a 371 MB uv cache under `data/scratch/uv_cache`.
- `positronic` is not installed. The loader and the scripts import only `pos3`/`configuronic`/numpy/
  scipy/matplotlib; "positronic" appears in `common.py`/`loader.py` only in comments and strings (grep).

**Downloads (approved):** PhAIL metadata sidecars from the public bucket `positronic-public`, endpoint
`https://storage.eu-north1.nebius.cloud` (quoted from positronic `cfg/ds/__init__.py` at `498ef5e`),
prefix `phail/v1.0/dataset/`. Everything except `*.parquet` and `*.mp4`, as the phail-paper README
recipe does. Listing (measured, `scripts/phail_fetch.py --list`):
- `human`: 40 episodes, 0.07 GB;
- `rollouts`: 594 episodes, 26.64 GB;
- `teleoperation`: 449 episodes, 13.95 GB;
- total 14,361 objects, 40.66 GB;
- sidecars: 182.5 MB (2,166 files; 1,089 zero-byte directory-marker keys are skipped).

The fetch is complete: 2,166 files, 182.5 MB (`du` 183 MB), in `data/phail/dataset/`. It resumed after one
`RemoteDisconnected`, after which the script got a 4-try retry. The fine-tuning set is listed and counted,
never downloaded.

**Measured so far:**
- Access probe (`scripts/phail_fetch.py --probe`): `s3://positronic-public/phail/v1.0/dataset/: HTTP 200`;
  `s3://inference/phail_act_groot/`, `s3://inference/phail_final/` and `s3://inference/spoons_ablation/`:
  `HTTP 403 Forbidden`.
- Other prefixes: `datasets/phail/v1.0/training/`: 5,841 files, 13.95 GB, 449 episodes (static.json);
  `phail/v1.0/`: 1,083 episodes, 76.99 GB (`dataset` + `models`).
- `dataset/croissant.json` gives license `https://creativecommons.org/licenses/by/4.0/`, and three times
  calls the fine-tuning set "352-episode", against 449 in the bucket and the paper.
- Published values located in the clone (artifacts the paper's numbers come from):
  - `out/paper/tab_headline_ci.json` (Table 2 RMST/HRT with CIs);
  - `out/paper/tab_leaderboard.json` (F(20/30/60 s), pooled %-of-human);
  - `out/explore/pp_summary.json` (per-cell AUC, macro-AUC CIs, n_boot 1000);
  - `out/explore/ks_power/dashboard_per_obj_macro.json` (macro-KS: `OpenPI-GR00T D_macro_per_obj
    0.12391553589925894 p 0.11944027986006997`, n_boot 2000 since p_min = 1/2001; producer
    `build/dashboard.py` H1, `seed=42`).
- `build/dashboard.py` cannot be imported. It does `from build import ... fig_cdf`, and
  `build/fig_cdf.py` is absent from the commit (`ls` → No such file). Its H1 loop (`default_rng(42)`,
  `combinations(POLICY_MODELS, 2)`) is therefore replicated in `scripts/phail_export_cohort.py`, calling
  PhAIL's `stats.per_obj_macro_bootstrap_ks` directly.
- Third UPH form, `build/common.py:_derive_metrics`: `'UPH': 3600 * S / T` per episode (S successful
  items, T duration).

**Files:**
- `scripts/phail_fetch.py` (new; stdlib urllib, `certifi` CA bundle, retry).
- `scripts/phail_export_cohort.py` (new; runs in `.venv-phail`). PhAIL's own `loader.load('mixed')` builds
  the cohort, which is exported to `data/phail/cohort_mixed.parquet`. PhAIL's own functions then compute the
  reference values: `tab_leaderboard._per_model_row`, `stats.*`, `explore_pp_plot._auc_censored` /
  `_macro_auc` / `_pp_curve` / `_macro_pp_curve`, `explore_separation._logrank`, and
  `stats.per_obj_macro_bootstrap_ks` in the `dashboard.py` H1 order.
- `scripts/phail_validate.py` (new; runs in `.venv`). It computes our layer on the same parquet and
  compares it with PhAIL's functions and with the published files; `--hrt-ci-seeds` covers AC2.5.

**Premises checked:**
- The loader and the needed scripts import only `pos3`, `configuronic`, numpy, scipy and matplotlib
  (grep of the import lines). **Holds** (measured).
- The cohort is incomplete: 990 of 995 episodes. **Holds**, and the missing five are identified below.

**Acceptance criteria:**
- ✅ AC2.1: implementation equivalence on identical input (`artifacts/s2_phail_validate_stdout.txt`):
  - `max |ours - PhAIL| pooled_per_model (rmst240, median, F20/30/60, pct_of_human): 0.000e+00`;
    `cells (rmst240, median, F20/30/60): 0.000e+00`; `cells censored AUC vs teleop: 0.000e+00`;
    `cells HRT %: 3.553e-15`; `macro AUC: 0.000e+00`;
    `KS distance per cell (24 = 6 pairs x 4 objects): 0.000e+00`; `P-P curve ACT|Batteries: 0.000e+00`;
    `macro P-P curve ACT: 0.000e+00`; `pooled logrank chi2/p (one stratum): 4.868e-17`.
  - `macro-KS D and p identical for all 6 pairs (same RNG): True`. Example: `OpenPI-GR00T: ours
    D=0.123916 p=0.117941 | PhAIL D=0.123916 p=0.117941`.
- ✅ AC2.2 (cohort incomplete; the missing episodes are named): `OpenPI 163/165, GR00T 164/165, ACT
  151/151, SmolVLA 116/118, Teleop 396/396`. By cohort source: OpenPI 109 manual + 54 auto, GR00T 104 + 60,
  ACT 55 + 96, SmolVLA 53 + 63, Teleop 96 + 300 (`artifacts/s2_phail_export_stdout.txt`).
  - The 5 missing episodes are exactly the 5 `reviewed=True` annotations under the private
    `s3://inference/phail_act_groot/` (HTTP 403). The loader drops them because their `static.json`
    cannot be read.
  - Joined to `out/release_classifier/episode_summary.csv`, all 5 are **USB-C Boxes**: 2 SmolVLA (1 + 1
    placements), 1 GR00T (3), 2 OpenPI (1 + 1). That is 2/1/2 episodes and 2/3/2 events, exactly the
    shortfall in episodes (165 − 163, …) and in events (631 − 629, 620 − 617, 154 − 152)
    (`artifacts/s2_missing_episodes_check.txt`).
  - USB-C Boxes has no teleop reference (PhAIL `explore_pp_plot.py:12`, quoted), so these episodes enter
    only the pooled numbers.
- ✅ AC2.3, under the "incomplete" branch: the deltas are reported and explained, and AC2.1 carries the
  pass. Per-object published numbers are reproduced exactly:
  - Macro HRT `ours 13.754838767437327 pub 13.754838767437327` (OpenPI), |diff| 0–1.8e-15 for all 4 models.
  - All 16 per-cell censored AUCs equal `pp_summary.json`, max |diff| `0.00e+00`.
  - Macro-AUC: ACT 0.1338 [0.111, 0.162] vs pub 0.134 [0.108, 0.162]; OpenPI 0.1004 [0.084, 0.117] vs
    0.100 [0.084, 0.116]; GR00T 0.0949 [0.079, 0.111] vs 0.095 [0.080, 0.110]; SmolVLA 0.0268 [0.017, 0.037]
    vs 0.027 [0.016, 0.036]. CI widths are within +3.6% / −3.7% of `pp_summary`.
  - F(30 s): GR00T 0.5091, OpenPI 0.4741, ACT 0.4409, against the paper's 0.51 / 0.47 / 0.44, all within ±0.005.
  - The pooled RMST points of the three models with USB-C episodes miss ±0.1 s: OpenPI 77.572 vs 77.693
    (Δ −0.121 s), GR00T 77.313 vs 77.205 (+0.107 s), SmolVLA 166.373 vs 165.776 (+0.597 s). ACT
    (100.929, Δ 0.0000) and Teleop (10.516, Δ 0.0000) are exact.
  - RMST CI widths are within ±10% (+1.3%, −2.2%, +0.3%, −5.0%, −0.3%).
  - Macro-KS OpenPI-GR00T: ours `D=0.123916 p=0.117941` vs `dashboard_per_obj_macro.json D=0.123916
    p=0.119440` / paper p = 0.12, within ±0.03.
  - Tolerance checks: 23/26 within. The 3 failures are those three pooled RMST points.
- ✅ AC2.4: the four yardsticks were regenerated from the data:
  - `F(30 s)`: GR00T 0.509 [0.471, 0.547], the winner.
  - `HRT macro (%)`: OpenPI 13.755 [12.344, 15.674], the winner.
  - `global RMST(240 s)`: GR00T 77.313 [69.171, 86.190], the winner.
  - `AUC vs human (macro)`: ACT 0.134 [0.111, 0.162], the winner.
  - Result: `published GR00T/OpenPI/GR00T/ACT reproduced: True; distinct top-1: 3`.
- ✅ AC2.5: the HRT CI variant is decided with Monte-Carlo spread (`--hrt-ci-seeds 8`, n_boot 1000 per
  seed; `artifacts/s2_hrt_ci_variants_seeds_stdout.txt`).
  - Published endpoints as z-scores: human resampled gives OpenPI −1.14/+0.77, GR00T +0.65/+1.40, ACT
    −0.09/+1.02, SmolVLA −2.23/+1.28; human fixed gives −3.14/+1.16, +0.39/+3.18, −1.08/+0.81, −3.44/+2.85.
  - `sum of z^2 over 8 endpoints: human_resampled 11.92, human_fixed 43.35`.
  - Adopted: resample the human anchor too, which is already `hrt_ci(resample_human=True)` by default.
  - The single-seed run pointed the same way (summed endpoint error 0.58 vs 0.67 pp).
  - Not an exact reproduction: PhAIL's Table 2 CI producer is unreleased (P19), so its RNG order is unknown.
- ✅ AC2.6: the p. 8 caveat, answered with measured facts (table in `phail_validation.json` → `AC2.6`):
  - The fine-tuning set is 449 episodes / 13.95 GB at `phail/v1.0/dataset/teleoperation/`, identical in
    size and count to `datasets/phail/v1.0/training/` (5,841 files). It is public, anonymous HTTPS,
    licensed CC BY 4.0 (`croissant.json`), and was not downloaded.
  - The evaluation rollouts are 594 episodes / 26.64 GB; human, 40 episodes.
  - Private: `s3://inference/phail_act_groot/`, `phail_final/`, `spoons_ablation/` (HTTP 403).
  - Discrepancy: `croissant.json` says "352-episode" three times; the bucket and the paper say 449.
  - Table 2's human anchor is the teleop cohort (`REFERENCE_MODEL = 'Robot teleoperated by Human'`, P22);
    in the export, Teleop = 396 episodes, 96 manual + 300 auto-validated.
- ✅ AC2.7: figures on real data:
  - `artifacts/s2_fig_km_pp_ACT_Batteries.png`: ACT vs Teleop KM CDFs with 95% episode-clustered bands,
    and the P-P plot, AUC 0.1575. That AUC equals PhAIL's `pp_summary` cell value 0.15754457702104840.
  - `artifacts/s2_fig_uph_mtbfa_macro.png`: macro UPH vs MTBF/A as τ_episode sweeps 30→240 s.
  - Both were inspected visually. `monotone over τ in [30, 240]`: UPH falls and MTBF/A rises for all four
    models, which reproduces App. L's "UPH falls … MTBF/A rises".
- ✅ AC2.8: the three UPH forms, at τ_episode = 240 s, pooled:
  - OpenPI: 3600/RMST = 46.4; Table 1's 3600/E[T|T<240] = 97.1; `common.py`'s mean 3600·S/T = 63.2.
  - GR00T 46.6 / 90.6 / 59.9; ACT 35.7 / 105.3 / 45.4; SmolVLA 21.6 / 66.9 / 13.1; Teleop 342.3 / 342.3 / 345.3.
  - The forms differ by up to 3×, and they rank ACT differently. The adopted form is 3600/RMST(τ_episode),
    from PhAIL's App. L figure code.

**Demo artifacts produced:** `artifacts/phail_validation.json` (every number above),
`artifacts/s2_phail_reference.json` (PhAIL-side values), the two figures, `artifacts/s2_missing_episodes_check.txt`,
`artifacts/s2_hrt_ci_variants_seeds.json`, `artifacts/s2_phail_listing_summary.json`,
`artifacts/s2_phail_access_probe.json` and `artifacts/s2_phail_other_prefixes.json`. The cohort parquet
(`data/phail/cohort_mixed.parquet`, gitignored) is regenerated by `scripts/phail_export_cohort.py`.

**Notes / divergences:**
- **Two published files disagree on pooled RMST.** `tab_leaderboard.json` gives ACT 100.9451 and OpenPI
  77.7027; `tab_headline_ci.json` (Table 2's source) gives ACT 100.9287 and OpenPI 77.6926. Ours matches
  `tab_headline_ci.json` for ACT exactly. So the two sidecars come from different runs of the cohort;
  the artifact wins, and this is recorded.
- **"% of human" is computed two ways in PhAIL.** In `tab_leaderboard.md` it is the pooled RMST ratio
  (GR00T 13.6%). In Table 2 / `tab_headline_ci.json`, HRT is the per-object macro (GR00T 13.349%). We
  reproduce both: AC2.1 pooled `pct_of_human` |diff| 0, and the macro HRT above.
- **The KM CDFs start at F(0) ≈ 0.05** because PhAIL's multi-grip extras enter as T = 0 events
  (`episode_to_T_pairs`, threshold 0.3 s; the PhAIL default is kept for PhAIL data).
- **Main `.venv` packages:** S2 changed none (the PhAIL env is separate).

### S3: LIBERO simulator stack locally, and the human reference by replaying the demos' goal check

Re-read this session: PoC §3.5 A pp. 7–8, §9.1 Step 2 p. 18 and §10 p. 20 (PDF). Also read:
- LIBERO `scripts/create_dataset.py` at `8f1084e5…` (fetched from GitHub raw; l. 123 `cap_index = 5`,
  l. 130–153 the replay recipe, l. 226–231 `states = states[valid_index]`, `rewards[-1] = 1`, `dones[-1] = 1`).
- The installed hf-libero 0.1.4 source:
  - `libero/libero/__init__.py`: the config prompt, `_assets_path_cache`, HF asset download;
  - `envs/env_wrapper.py` (`ControlEnv`);
  - `envs/problems/*_manipulation.py` (`_check_success`, `_eval_predicate`);
  - `envs/bddl_utils.py` (`robosuite_parse_problem`);
  - `utils/utils.py` (`postprocess_model_xml`);
  - LeRobot `envs/libero.py` (`get_task_init_states`, `num_steps_wait=10`).

**Environment (plan rule: main `.venv` first).** The sim stack went into the main `.venv` (Python 3.14.0).
- Before installing, `uv pip install --dry-run` showed additions only (32 packages, nothing changed).
- It would have added `opencv-python` 5.0 next to the installed `opencv-python-headless` 4.13; both write
  the `cv2` package (inferred risk to LeRobot). So robosuite and hf-libero were installed `--no-deps`, with
  their other dependencies pinned.
- The freeze diff before/after (`data/scratch/logs/venv_freeze_before_s3.txt` vs `_now.txt`) shows **only
  added lines** (32 packages, 0 changed or removed). Among them: `hf-libero==0.1.4`, `robosuite==1.4.0`,
  `mujoco==3.8.1` (hf-libero requires `mujoco<3.9.0,>=3.0.0`; 3.8.1 is proven by P10), `bddl==1.0.1`,
  `numba==0.68.0`, `hydra-core==1.3.7`, `easydict`, `einops`, `cloudpickle`, `future`.
- Skipped hf-libero dependencies: `robomimic`/`hf-egl-probe` (broken on macOS, P11), `wandb`, `thop`,
  `transformers`, `opencv-python`.
- No `.venv-sim` was needed.

**LIBERO config (never `~/.libero`).** `import libero.libero` *prompts on stdin* when
`$LIBERO_CONFIG_PATH/config.yaml` is missing (quoted, `libero/libero/__init__.py`), and the default path is
`~/.libero`. So `vlapoc.libero_goals.setup_libero()` sets `LIBERO_CONFIG_PATH=data/libero_config` and writes
`config.yaml` before the first import. The assets path is pinned through `libero.libero._assets_path_cache`,
which `get_assets_path()` returns when it exists (quoted, l. 68–80). That replaces LIBERO's on-demand
download to `~/.cache/libero/assets`. Measured: `ls ~/.libero` → `No such file or directory`, and
`ls ~/.cache/libero` → `No such file or directory`.

**Download (approved):** LIBERO assets from the HF dataset `lerobot/libero-assets` @
`0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` (last modified 2025-11-03). That is the repo id hard-coded in
hf-libero's `download_utils.py:109`. 586 files, 422,320,936 bytes (HF API), via `snapshot_download(...,
revision=…, local_dir="data/libero_assets")`; 17.5 min; `du` 415M; 586 files on disk.
- The demos' model XML carries two absolute asset roots (measured over all 1000 demos):
  `/Users/yifengz/workspace/libero-dev/chiliocosm/assets/` (14,500 refs) and
  `/home/yifengz/workspace/libero-dev/chiliocosm/assets/` (12,050).
- `demo_model_xml` maps both to `data/libero_assets/` and raises on any other unmapped root. Robosuite
  paths are left to LIBERO's own `postprocess_model_xml`.

**Files (so far):**
- `vlapoc/libero_goals.py` (new): `setup_libero`, `fetch_assets`, `assets_digest`, `bddl_path`,
  `goal_conjuncts` (LIBERO's parser), `make_replay_env` (`ControlEnv` with no renderer and no camera obs),
  `demo_model_xml`, `replay_goal_trace` (LIBERO's `_eval_predicate` per conjunct + `_check_success` per
  stored state), `first_true`, `rle`, `env_versions`. Its docstring covers the AC3.5 alignment conventions.
- `scripts/human_reference.py` (new): replay → parquet + `sim_env.json`; `--bddl-check` (AC3.6);
  `--init-states` (AC3.7); `--summary` (AC3.3/AC3.4 via the S1 metric layer).

**Acceptance criteria so far:**
- ✅ AC3.7: `human_reference.py --init-states` (LeRobot `get_task_init_states`) →
  `libero_spatial: 10 tasks, init-state array shapes [(50, 92)]`; `libero_10: 10 tasks, init-state array
  shapes [(50, 45), (50, 47), (50, 51), (50, 71), (50, 84), (50, 123)]`; `tasks with exactly 50 fixed
  init states: 20/20` (`artifacts/s3_init_states_stdout.txt`, `s3_init_states.json`).
- ✅ AC3.1: the sim stack is installed in the project env (`.venv`) without touching `~/.libero`.
  `artifacts/sim_env.json`:
  - `"python": "3.14.0"`; packages `hf-libero 0.1.4, robosuite 1.4.0, mujoco 3.8.1, bddl 1.0.1, numba
    0.68.0, numpy 2.2.6, lerobot 0.6.1, torch 2.11.0`;
  - `"assets_revision": "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"`; `"assets_digest": {"files": 586,
    "bytes": 422320936, "sha256": "4cb93d37afaa096c3fcd8481995c72a615278b17e6e0feea883ec14940f2c259"}`;
  - `"home_libero_exists_before": false, "home_libero_exists_after": false, "home_cache_libero_exists": false`;
  - `"LIBERO_CONFIG_PATH": ".../vlaExp/data/libero_config"`.
- ✅ AC3.2: `human_reference.py --suites libero_spatial,libero_10 --workers 8` → `replayed 1000 demos from 20
  task files (demo XML) in 10405.7s with 8 workers`; `AC3.2 goal reached at some stored state: 1000/1000;
  replay errors: 0; _check_success == AND(conjuncts) at every state: 1000/1000`
  (`artifacts/s3_replay_stdout.txt`). No demo failed, so none needed listing. `length_steps == n_states`
  for 1000/1000.
- ✅ AC3.3: `T_success_s < duration_s: 1000/1000`.
  - Per task, `gap = length_steps - T_success_steps`: min 9–11, median 11.0–19.5, max 11–82
    (`artifacts/s3_gap_steps_per_task.csv`).
  - Planning measured "min 8–10, median 10–18, max 81". The planner counted the last stored *index*
    minus first success, which is gap − 1. Under that definition ours is 8–10 / 10–18.5 / 81, matching.
  - Success holds over the final stored states with a median tail run of 11 states. But it does **not**
    hold at the last stored state in **12/1000** demos (`artifacts/s3_success_not_at_last_state.txt`):
    e.g. `[[0, 266], [1, 10], [0, 1]]`, the goal holding 10 steps and then dropping in the final 1–2
    stored states. Several demos also toggle before that (libero_spatial task 0 demo_5: on 7, off 44, on 9).
  - T = *first* time the goal holds, which matches LeRobot rollouts terminating on the first
    `check_success()` (P16).
- ✅ AC3.4: human reference per task and per suite (`artifacts/s3_human_reference_table.csv`;
  `artifacts/s3_summary_stdout.txt`). Built with the S1 metric layer: whole-task T, every demo an event,
  no censoring. Per suite:
  - libero_spatial (n 500): median T_success 5.30 s vs duration 6.20 s; RMST(60 s) 5.356 vs 6.225 s;
    F(30 s) 1.0 / 1.0. An HRT anchored on `duration_s` is **1.162×** too high (per task 1.119–1.227).
  - libero_10 (n 500): median 12.45 vs 13.15 s; RMST(60) 13.166 vs 13.809 s; F(30) 1.0 / 1.0. That
    anchor is **1.049×** too high (per task 1.033–1.082).
  - Figure `artifacts/s3_fig_human_reference_cdfs.png`: per-task and pooled KM CDFs of T_success vs
    duration_s, both suites. Inspected visually.
- ✅ AC3.5: the alignment conventions are documented in the `vlapoc/libero_goals.py` module docstring,
  section "Alignment conventions (AC3.5)":
  - human T counts from stored state 0 = original step 5 (`cap_index = 5`, quoted);
  - rollout T counts from the post-settle state (`num_steps_wait = 10`, LeRobot);
  - residual ≤ 5 steps = 0.25 s (inferred);
  - "approximate, since demos and rollouts don't share initial states" (PoC p. 8);
  - the demo-XML (human) vs BDDL-scene (rollouts) split.

  The report text is the summarizer's.
- ✅ AC3.6: the BDDL-scene vs demo-XML discrepancy is reproduced, and found to be wider and less stable
  than P9 said:
  - `--bddl-check --only libero_10:8` → `no success at any state: 2/50; success but different T than
    demo-XML replay: 5` (`artifacts/s3_bddl_check_task8_stdout.txt`).
  - All 1000 → `replayed 1000 demos ... (BDDL scene) in 201.0s`; `no success at any state: 3/1000`
    (`KITCHEN_SCENE8_put_both_moka_pots_on_the_stove/demo_19`, `demo_33`;
    `STUDY_SCENE1_pick_up_the_book_..._caddy/demo_27`); `different T: 9` (BDDL − demo XML from −11 to +23 steps). Only these 2
    libero_10 tasks are affected (`artifacts/s3_bddl_check_all_stdout.txt`, `s3_bddl_vs_demo_xml.csv`).
  - P9's specific demo (task 8 demo_1) succeeds in both of our BDDL runs. The *kind* of discrepancy
    reproduces; the instance does not.
  - **New (measured): the BDDL-scene replay is not deterministic; the demo-XML replay is.** Three repeats
    in one process (`artifacts/s3_replay_determinism.txt`): demo-XML T = `[411, 411, 411]`,
    `[418, 418, 418]`, `[193, 193, 193]`, all equal to the parquet. The BDDL scene gave demo_19 `T = [419,
    -1, 419]`. Across runs, demo_3 gave 409 / 413 / 418 and demo_43 233 / 216.
  - Rollouts will run in the BDDL scene (LeRobot's `LiberoEnv`) and the human reference uses the demo
    XML, as the plan states.
- ✅ AC3.7: as above, 20/20 tasks with exactly 50 init states.
- ✅ AC3.8: records in the data+lineage feature folder.
  - `changelog.md` entry "2026-10-03: Reward evidence corrected; `duration_s` is demo length, not human
    time-to-success". It quotes `create_dataset.py` l. 228–231 (P6) and `task_completion_hold_count = 10`
    (P7), and gives the replay 1000/1000 (P8 re-measured) and the 1.162× / 1.049× bias.
  - `report.html`: dated corrections at the reward paragraph, the S2 durations caption and "Next", plus a
    "Changes since first release" line.
  - `tracker/index.html`: the §3.5 A row is reworded. The CLAUDE.md / vla-build wording was already
    corrected by the orchestrator (brief); not touched here.

**Demo artifacts produced:**
- `artifacts/sim_env.json`; `artifacts/s3_human_reference_table.csv` (20 tasks + 2 suites);
  `artifacts/s3_gap_steps_per_task.csv`; `artifacts/s3_fig_human_reference_cdfs.png`;
  `artifacts/s3_bddl_vs_demo_xml.csv` (+ `_task8`); `artifacts/s3_replay_determinism.txt`;
  `artifacts/s3_success_not_at_last_state.txt`; `artifacts/s3_init_states.json`.
- The full per-demo table, with conjunct first-true steps, truth at t0 and RLE traces for S4, is
  `data/eval/human_reference/human_reference.parquet` (gitignored), regenerated by
  `scripts/human_reference.py`.

**Notes / divergences:**
- **Replay wall time was 10,405.7 s, not the planned "2–3 min".** Per-demo `replay_s`: median 3.9 s, p75
  6.5 s, max 1,990.5 s, sum 81,474 s over 8 workers. The median implies about 9 min. The long tail
  coincides with the session stall (inferred: process suspension or contention).
  - Most of the per-demo cost is `reset_from_xml_string`, the per-demo model compile (inferred): the
    BDDL-scene replay, which has no recompile, did 1000 demos in 201 s.
  - I did not re-run the replay (coordinator instruction).
- **The S4 inputs are stored now** (RLE conjunct and success traces), so S4 needs no second replay.
- **Main `.venv` gained 32 packages** (none changed). LeRobot was not re-tested end to end here; S5 will
  exercise it.

### Findings for the analyzer (S1–S3)
- **S2: the published cohort has 5 private episodes.** They are all USB-C Boxes (no teleop reference)
  under `s3://inference/phail_act_groot/` (HTTP 403). Per-object results reproduce exactly; pooled
  results differ only by those 5.
- **S2: two published PhAIL sidecars disagree** (`tab_leaderboard.json` vs `tab_headline_ci.json`, pooled
  RMST for ACT/OpenPI). Ours matches `tab_headline_ci.json` (Table 2's source).
- **S2: PhAIL's `croissant.json` says the fine-tuning set has 352 episodes;** the bucket and the paper say 449.
- **S2: PhAIL's `build/dashboard.py` can't be imported** (it needs the missing `build/fig_cdf.py`). Its
  H1 loop is replicated in `scripts/phail_export_cohort.py`.
- **S3 → S5 (risk for AC5.2 determinism and for success checks), measured:** the BDDL-built LIBERO scene
  gives a different goal-check outcome for the same stored states across resets (demo_19: 419 / −1 / 419).
  LeRobot rollouts run in that scene, so the same init state could succeed or not depending on reset
  sampling. S5 should check AC5.2 with repeated resets, not only repeated actions. Inferred cause:
  reset-time placement/region sampling that `set_init_state` does not overwrite.
- **S3: in 12/1000 demos the goal stops holding in the last 1–2 stored states.** T = first success is
  unaffected. Rollout success semantics (first `check_success`) match it.
- **Plan cost estimate:** "2–3 min for 1000 demos" was measured in planning with a different procedure.
  The demo-XML replay costs about 4 s per demo here (model compile).

## Round 1 / Batch 2 (S4–S6): 2026-10-03

### S4: Per-sub-task time-to-success T for libero_10 (required story)

Re-read: PoC §3.5 A p. 7, §3.4 p. 7, §3.2 p. 6 (PDF dump); PhAIL §3.1 p. 4; LIBERO `bddl_utils.robosuite_parse_problem`
(returns `initial_state` and `goal_state`) and the `_check_success` lines of the problem classes.

**Files:**
- `vlapoc/libero_goals.py`:
  - `operations_table(bddl)`: static, from LIBERO's parser; maintained = goal ∩ `:init`.
  - `operations_from_trace(truth, success, tau_end_steps, ...)`: the single rule. Its docstring declares
    the rule. It builds the (T, E) pairs through `vlapoc.evaluation.libero_episode`, i.e. PhAIL's
    `episode_to_T_pairs` with threshold 0.
  - `rle_decode`.
- `scripts/human_reference.py --operations`: S4 on the stored S3 traces, with no second replay. It writes
  `data/eval/human_reference/human_operations.parquet`.
- `scripts/verify_operations_s4.py` (new): the synthetic traces and the single-implementation grep.

**Premises checked:**
- The S3 parquet stores per-step conjunct truth. Checked by reading `conjunct_trace_rle` and
  `success_trace_rle` per demo; `rle_decode` gives arrays of length `n_states`. **Holds**, so no re-replay.
- libero_10 task 8's goal includes `(turnon flat_stove_1)`, which is also in `:init`. Checked by parsing
  the installed BDDL: `init [... ['turnon', 'flat_stove_1']]`,
  `goal [['on', 'moka_pot_1', 'flat_stove_1_cook_region'], ['on', 'moka_pot_2', 'flat_stove_1_cook_region'], ['turnon', 'flat_stove_1']]`.
  **Holds** (P14).

**Acceptance criteria:**
- ✅ AC4.1: `libero_10 n_ops per task (BDDL) = [2, 2, 2, 2, 2, 1, 2, 2, 2, 2]; maintained = {8: ['(turnon
  flat_stove_1)']}; demos whose t0 truth gives the same n_ops: 500/500`; `libero_spatial ... = [1, 1, 1,
  1, 1, 1, 1, 1, 1, 1]; ... 500/500` (`artifacts/s4_operations_table.csv`). This matches P14 exactly, and
  the static (BDDL) and dynamic (t0 truth) operation sets agree for every demo.
- ✅ AC4.2: `7/7 synthetic cases pass` (LABELLED-SYNTHETIC; `artifacts/s4_verify_operations_stdout.txt`):
  - two conjuncts in either order → `[(2.0, 1), (3.0, 1)]` both ways;
  - a conjunct that toggles off and on → `[(2.0, 1), (3.0, 1)]`;
  - a maintained conjunct broken over steps 50–110 → `t_ops [40, 110]`, `[(2.0, 1), (3.5, 1)]` (success
    waits for the restore);
  - simultaneous completion → `[(4.0, 1), (0.0, 1)]`;
  - timeout after 1 of 2 ops (τ_end 1200) → `[(15.0, 1), (45.0, 0)]`;
  - zero ops done → `[(60.0, 0)]`.
- ✅ AC4.3: `libero_10: sum_k T_k == T_success_s for 500/500; demos with events == n_ops: 500/500;
  censored pairs: 0; operation samples: 950 from 500 demos`; `libero_spatial: 1 op each: 500/500; T_op ==
  T_success_s: 500/500` (`artifacts/s4_human_operations_stdout.txt`). The 950 equals the plan's inferred
  "≈950" (19 ops × 50).
- ✅ AC4.4: human per-operation CDFs per libero_10 task, KM over all events
  (`artifacts/s4_fig_human_operation_cdfs.png`, inspected), with a per-op-index breakdown and
  episode-clustered bootstrap CIs (n_boot 1000, `artifacts/s4_human_operation_stats.csv`). Examples:
  - task 2: op1 median 4.25 s [4.10, 4.55], op2 8.25 s [7.70, 8.65];
  - task 3: op1 7.15 s [7.05, 7.30], op2 4.30 s [4.10, 4.80];
  - task 8: op1 9.05 s [8.75, 9.35], op2 10.50 s [10.10, 11.05];
  - task 5 (1 op): 8.70 s [8.45, 9.10].
- ⏳→✅ AC4.5 (complete once S5 is in; see S5): `definitions of operations_from_trace in vlapoc/ +
  scripts/: 1 (['vlapoc/libero_goals.py'])`; callers `scripts/human_reference.py: True`.
  `vlapoc/rollouts.py` (S5) is the second caller, and `verify_operations_s4.py` checks both.
- ✅ AC4.6: the `operations_from_trace` docstring cites `PoC §3.5 A (p. 7)`, `PoC §3.4 (p. 7)`,
  `PhAIL §3.1 (p. 4)`, `libero_tabletop_manipulation.py:135-143` and
  `libero_kitchen_tabletop_manipulation.py:136-144`: all `True` (grep).

**Demo artifacts produced:** `artifacts/s4_operations_table.csv`, `s4_human_operation_stats.csv`,
`s4_fig_human_operation_cdfs.png`, `s4_verify_operations.json`; plus
`data/eval/human_reference/human_operations.parquet` (1,450 rows; gitignored, regenerated in seconds).

**Notes / divergences:**
- **The clustered bootstrap can be narrower here.** For the *pooled* per-task median, the
  episode-clustered CI is narrower than the per-operation CI in 6 of 9 two-op tasks (ratio 0.405–0.900),
  and it is wider in 2 (tasks 0 and 7: 1.22, 1.11).
  - The narrowing is strongest where op1 and op2 differ most (task 2: 0.468, task 3: 0.405).
  - Inferred reason: each demo contributes exactly one op1 and one op2, so resampling whole demos acts like
    stratification; per-operation resampling can unbalance the two kinds.
  - The PoC rule (resample whole episodes) is applied regardless. The "clustered is always wider" intuition
    (S1 AC1.5) holds for positively correlated operations, not in general.
- **Threshold 0 vs PhAIL's 0.3 s** (declared in the docstring). With exact sim timestamps, simultaneous
  completion gives T = 0 under both conventions.

### S5: Rollout harness, rollout record and the eval stage (IN PROGRESS, partial entry)

Read: LeRobot `lerobot_eval.rollout()` (`lerobot/scripts/lerobot_eval.py:167-410`), `LiberoEnv`
(`lerobot/envs/libero.py`), `LiberoProcessorStep` (`processor/env_processor.py`), the env processors
(`envs/configs.py:446-450`), `make_pre_post_processors`; robosuite `environments/base.py`; LIBERO
`bddl_base_domain.seed`; SmolVLA `smolvlm_with_expert.py`.

**Premises checked (all measured or quoted, before building):**
- **LeRobot's `LiberoEnv` cannot run a 1200-step (60 s) episode as shipped.** The chain:
  - robosuite: `self.done = (self.timestep >= self.horizon) and not self.ignore_done` (`base.py:429`).
  - LIBERO's `ControlEnv` defaults to `horizon=1000`, and LeRobot's `OffScreenRenderEnv(...)` call doesn't
    pass one.
  - `LiberoEnv.step`: `terminated = done or is_success`.
  - With the 10 settle steps, a policy gets at most 990 steps.
  - The harness raises the inner horizon to τ + settle + 1. Measured: the AC5.1 no-op rollouts ran exactly
    1200 steps. The unpatched check is `--horizon` (below).
- **LeRobot's init-state assignment depends on batching:** `init_state_id = episode_index`, then
  `+= n_envs` per reset (`self._reset_stride = n_envs`). The harness fixes it per env (`_reset_stride = 0`).
- **Seeding** (mechanism of the S3 finding):
  - LIBERO `seed(seed)` is `np.random.seed(seed)` (`bddl_base_domain.py:164-165`).
  - robosuite's placement sampler draws from global `np.random` (`placement_samplers.py:167,183,196`).
  - `LiberoEnv.reset(seed)` seeds and then hard-resets.
  - So an unseeded reset samples the scene afresh. S3's replay called `reset()` unseeded.
- **Image orientation:** our dataset is stored rotated 180° (data+lineage AC1.4, measured there), the
  same flip `LiberoProcessorStep` applies, so LeRobot's processors match our training layout. The state
  layout is 8-d `eef_pos(3) + axis-angle(3) + gripper_qpos(2)`, as in our data.
- **SmolVLA loading:**
  - `load_vlm_weights=True` makes it fetch the full SmolVLM2 weights (`smolvlm_with_expert.py:90-92`).
  - `AutoProcessor.from_pretrained(model_id)` runs either way (l. 101).
  - The checkpoint's `model.safetensors` holds every weight (strict load), so the harness sets
    `load_vlm_weights=False` and points the VLM config and tokenizer at a pinned, repo-local copy of only
    SmolVLM2's non-weight files.

**Downloads (approved):**
- `HuggingFaceVLA/smolvla_libero` @ `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` (Apache-2.0, last modified
  2025-09-17). 8 files, 1,218,055,540 bytes, in `data/hf_models/smolvla_libero` (`du` 1.1G).
  - Its `config.json`: inputs 2 × 256² + state 8, `chunk_size 50`, `n_action_steps 1`, `num_steps 10`,
    `"normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}`,
    `"vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Instruct"`, `"load_vlm_weights": true`.
- A loading dependency: `HuggingFaceTB/SmolVLM2-500M-Instruct` @ `7b375e1b73b11138ff12fe22c8f2822d8fe03467`
  (Apache-2.0). **Only its 11 non-weight files** (config, tokenizer, processor): 4,866,925 bytes, in
  `data/hf_models/SmolVLM2-500M-Instruct`. The 2,029,990,624 bytes of weights are not downloaded.
  - This is a small dependency of the approved checkpoint, recorded here. Flag for the owner, if the
    approval was meant to exclude it.
- Nothing is written to `~/.cache/huggingface` for these: both use `snapshot_download(local_dir=...)`.

**Files:**
- `vlapoc/rollouts.py` (new):
  - `ProbedLiberoEnv`: fixed init state, horizon, per-step goal probe, t0 sim-state sha.
  - `run_rollouts`: LeRobot `rollout()` unchanged, in a `SyncVectorEnv` with NEXT_STEP autoreset; first-
    episode traces are kept across autoresets; operations via `operations_from_trace`; LeRobot's success
    is cross-checked against the probe.
  - `noop_policy`, `checkpoint_policy` (LeRobot's saved processors), `_Capture` + `write_trajectories`
    (lossless LeRobot v3 dataset, training schema).
  - `RECORD_COLUMNS` (32 columns), `validate_record`, `plan_rollouts`.
- `vlapoc/runs.py`: `eval_run` replaces `eval_standin` (single record system); `load_checkpoint` gains
  `config_overrides`.
- `scripts/eval_smoke.py` (CLI); `scripts/verify_rollouts_s5.py` (checks).
- `scripts/lineage_dry_run.py` and `scripts/verify_runs_s5.py` now call `eval_run` at smoke scale. The
  dry run is exercised in S9; `verify_runs_s5.py` has not been re-run (see notes).
- `vlapoc/libero_goals.py`: `setup_libero` now refuses if `libero` was already imported with another
  config path.

**Side effect caught and undone (measured):** the first debug run imported `libero.libero` before
`setup_libero()` (inside `run_rollouts`). LIBERO's import created an empty `~/.libero` and prompted on
stdin (EOFError). The empty directory was removed (`rmdir ~/.libero`; `ls` → No such file). Two fixes:
`run_rollouts` calls `setup_libero()` first, and `setup_libero` now raises if `libero.libero` was imported
earlier with another config path.

**Acceptance criteria so far:**
- ✅ AC5.1: `eval_smoke.py --policy noop --tasks libero_spatial:0,libero_10:8 --n 3 --batch-size 3
  --track --name eval-noop-smoke-a` → MLflow eval run `02bbe9eb1ab64dc2ae03175276101f87`.
  - All 6 rollouts: `censored at 1200 (0/1 ops)` for spatial and `(0/2 ops)` for libero_10 task 8.
  - The record validates against the schema (`schema errors: [] | columns 32 == schema True`); pairs
    `[[60.0, 0]]`; max arm action norm 0.0 (`artifacts/s5_ac5_1_stdout.txt`, `s5_noop_smoke_a_summary.json`).
  - Throughput: about 23 s per 1200-step rollout in batches of 3 (≈ 51 env-steps/s at 128², noop, CPU).
