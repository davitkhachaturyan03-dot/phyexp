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

### S5: Rollout harness, rollout record and the eval stage (COMPLETE locally: AC5.1–AC5.7 ✅; AC5.8 pending server)

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

#### S5 continuation (2026-10-03, replacement executor): the AC5.2 evidence slip, investigated

The previous executor stalled after writing "That stub import was a slip on my part". It left three AC5.2 results
that looked contradictory. I traced each one from the files on disk (birth/modify times by `stat`, the logs under
`data/scratch/logs/`, and the run dirs under `data/eval/`), then recomputed every comparison myself without stubs.

**The three results compare different things, and none of them shows a real difference in values:**

| Artifact | What it compared | Produced by | Verdict |
|---|---|---|---|
| `s5_ac5_2a_rerun.json` as committed in `4400422`: `rows 6, differing {arm_action_norm: 6}` | noop smoke (6 rollouts) re-run in a new process vs `data/eval/eval-noop-smoke-a` | first `--rerun` (log `s5_verify_a.txt`, born 11:15:21) with the comparator from **before** `4400422` | **comparator artifact**, explained below |
| `s5_ac5_2a_rerun.json` uncommitted overwrite (11:25:07): `rows 6, differing {}` | same comparison | second `--rerun`, same log file overwritten in place (`s5_verify_a.txt`, modified 11:25:07). It shows the sim loading, 6 × 1200-step rollouts, the result line and `exit 0` | **valid** (a real process, no stub) |
| `s5_ac5_2_ac5_3_act_stdout.txt` (13:33): "4 rollouts, 30 columns compared; differing: none" | ACT `train-full-s0`, eval runs `3392bb4d` (`-a`) vs `613a53db` (`-b`), 4 rollouts each | a stdin heredoc (its traceback reads `File "<stdin>", line 20`); no script in the repo writes this line, and no copy of the heredoc survives | **invalid: provenance cannot be checked**, and the previous executor said a stub import was involved. Renamed to `s5_ac5_2_ac5_3_act_stdout.INVALID.txt` and kept. Superseded by the clean recheck below, which reaches the same conclusion |

- **Why the committed result said `arm_action_norm: 6` (measured).**
  - `run_rollouts` returns `arm_action_norm` as a Python list (`vlapoc/rollouts.py:364`), but a parquet read returns
    a numpy array.
  - The old comparator ran `json.dumps(x, default=str)` on both. For an array that is `str(ndarray)`, which numpy
    truncates (`"[0. 0. 0. ... 0. 0. 0.]"`). For a list it is the full JSON. So the two never match.
  - Reproduced: the old comparator applied to the noop record against **itself**, with only that column re-typed
    as lists, gives `{'rows': 6, 'columns_compared': 30, 'differing': {'arm_action_norm': 6}}`, exactly the
    committed result. The current comparator (`.tolist()` first) gives `{}` (`artifacts/s5_ac5_2_slip_recheck.txt`).
  - A second flaw in the old comparator: because numpy truncates, it could not see a difference in the middle of an
    array when both sides were arrays. The current comparator compares full lists.
- **Clean recheck of the ACT comparison (measured).** I imported `scripts/verify_rollouts_s5.py` for real; no libero,
  lerobot or `vlapoc.rollouts` module was loaded, because the comparator needs only pandas. Then I compared the two
  stored records (`artifacts/s5_ac5_2_slip_recheck.txt`):
  - `ACT a vs b (current comparator): {'rows': 4, 'columns_compared': 30, 'differing': {}}`
  - An independent exact check (`np.array_equal` per cell, every column) differs only in the two volatile columns:
    `{'eval_run_id': 4, 'wall_seconds': 4}`.
  - Both runs: checkpoint sha `6f49bdae15db…`, `policy_type act`, MLflow param `device cpu`. The arm action norm lies
    in 0.210029–0.282199 over 4 × 1200 steps, so these are real (non-zero) policy actions.
  - The two runs were separate processes. `s5_act.txt` holds two JSON summaries, and `eval_smoke.py` prints exactly
    one per invocation. MLflow start/end times are 1791012278012/1791012379092 and 1791012383802/1791012484064.
- **AC5.3 trace recheck (measured).** `python -m vlapoc.lineage trace 3392bb4d…` (exit 0) is byte-equal to the
  uncommitted `s5_ac5_3_trace.json` (`recheck == s5_ac5_3_trace.json: True`), so that artifact is valid.

#### S5 continuation: AC5.2, AC5.3 and AC5.6

**New file:** `scripts/verify_resets_s5.py` (determinism across resets, separated by mechanism; the AC5.6 pairing
re-checked on a property-initializer task). **Changed:** the `vlapoc/rollouts.py` module docstring now states the
one-env-per-rollout constraint and why (below). `scripts/eval_smoke.py`: a relative `--out` path no longer crashes
the summary print. The rollouts parquet was already written before the crash.

**Premise checked: why the scene can differ across resets with the same seed.**
- Read in LIBERO's source: `_load_model` runs on every hard reset and calls `_setup_placement_initializer`. That
  *appends* an `OpenCloseSampler`/`TurnOnOffSampler` to `self.object_property_initializers`
  (`bddl_base_domain.py:653,674`), and the list is never cleared. `_reset_internal` draws from every sampler in
  the list from the global `np.random` (l. 743), before the placement sampler.
- So in tasks whose BDDL `:init` has open/close/turnon/turnoff, the number of random draws taken before placement
  grows with the reset count. That holds for 4/20 tasks (`artifacts/s5_property_initializer_tasks.txt`): libero_10
  tasks 3, 8, 9 and libero_spatial task 4.
- Inferred: this growth is unintended behaviour in LIBERO, not a designed feature. Not checked upstream.

**Acceptance criteria:**
- ✅ AC5.2 (a) **noop, re-run in a new process:** `verify_rollouts_s5.py --rerun data/eval/eval-noop-smoke-a` →
  `6 rollouts, 30 columns compared (all but ['eval_run_id', 'wall_seconds']); differing columns: none`, `exit 0`.
  - Measured twice: the previous executor's second re-run (11:25, `s5_verify_a.txt`), and mine (log
    `s5_verify_a_r2.txt`, `artifacts/s5_ac5_2a_rerun_r2_stdout.txt`; it rewrote `s5_ac5_2a_rerun.json` with
    identical content).
  - Every column is compared in full, including the 1200-long `arm_action_norm` list and the per-step
    `obj_pos`/conjunct traces.
- ✅ AC5.2 (a) **ACT `train-full-s0` on CPU:** eval runs `3392bb4d` vs `613a53db`, two processes →
  `{'rows': 4, 'columns_compared': 30, 'differing': {}}`. The independent exact check differs only in
  `eval_run_id` and `wall_seconds` (`artifacts/s5_ac5_2_slip_recheck.txt`). Arm action norm range is
  0.210029–0.282199, so the actions are real.
  - MPS determinism for ACT was not tested. The plan asks for CPU, and it holds there. MPS determinism for SmolVLA:
    see AC5.4.
- ✅ AC5.2 (b) **across resets: deterministic for the harness's pattern, not for env reuse (measured,
  `verify_resets_s5.py` → `artifacts/s5_ac5_2b_resets_r2.json`, `s5_ac5_2b_resets_r2_stdout.txt`).**
  - libero_spatial task 0 (no property initializer), one env with `reset(seed=0)` × 3, and 3 fresh envs: all the
    same, state `e7263aae8a32` and body_pos `9f2fca44bea7`. Property initializers `[0, 0, 0]`, `n_moved: 0`.
  - libero_10 task 8, **one env, `reset(seed=0)` × 3**:
    - the flattened state is identical (`54f7df61644e` × 3);
    - but the model's body_pos is `['1240ac26e199', 'd68c1c917901', '06eedc6577de']`;
    - property initializers `[3, 4, 5]`;
    - reset 2 vs 1: `n_moved: 1, max_abs_m: 0.0072, moved_top: [('flat_stove_1_main', 0.0072)]`.
    - So the stove (and its cook region, the goal's target) moves by 7.2 mm between resets of one env.
  - libero_10 task 8, **3 fresh envs, first `reset(seed=0)` each** (what `run_rollouts` does): body_pos
    `['1240ac26e199'] × 3`, property initializers `[3, 3, 3]`, `n_moved: 0`.
  - **This explains the batch-1 S3 finding:** the BDDL replay env was one object, reset repeatedly. In round 1's
    `--resets` (`s5_ac5_2b_resets.json`), demo_19's first success under seed 0 × 3 on one env was `[-1, 420, 418]`.
  - **Consequence for the harness:** every recorded rollout comes from the first seeded reset of a fresh env.
    `run_rollouts` builds one env per rollout, and `first_trace` discards the NEXT_STEP autoreset episodes. So
    records are a function of (task, init state, seed), which the AC5.2 (a) re-runs confirm.
  - Reusing an env object across rollouts would break this for 4/20 tasks. The constraint and its reason are now
    in the `rollouts.py` docstring.
  - `t0_state_sha256` cannot detect this drift, because a fixture's position is in the model, not the state. That
    is a limit of the record's t0 identity (see Findings).
- ✅ AC5.3 **lineage:** eval run `3392bb4d` (`eval-act-train-full-s0-a`), git `44004220`.
  - Its input is `checkpoint:6f49bdae15db…`, produced by train run `ff64512b` (`train-full-s0`), and its param
    `checkpoint_sha256` matches. `stand_in` is false on every run in the trace, and `eval.attribution_grade=true`.
  - `python -m vlapoc.lineage trace 3392bb4d…` (exit 0) visits runs `eval → train-full-s0 → derive-v1 → build-v0,
    split-v0`, with 20 raw files.
  - Edges: the checkpoint ← `ff64512b`. `ff64512b` consumes subset `69684e60…` (**v1/candidate**, 960 episodes),
    which comes from `30f34cd0` (derive-v1). derive-v1 consumes version `3c4dca68…` (**v0**) ← `bc8f7d09`
    (build-v0), which ends at the 20 raw HDF5 files with sha256.
  - The trace is byte-equal to `artifacts/s5_ac5_3_trace.json`. The 5-step smoke checkpoint was trained on v1's
    candidate subset, so the chain reaches v0's raw files through derive-v1's parent edge.
- ✅ AC5.6 **init-state pairing:**
  - Round 1, valid (log `s5_verify_c.txt`, a real run): libero_spatial task 3 × 4 rollouts, batch 1 vs 4 →
    identical `(init_state_index, seed, t0_state_sha256)`: `True` (`artifacts/s5_ac5_6_pairing.json`).
  - That task has no property initializer, and t0_state does not cover fixtures. So I repeated the check on
    libero_10 task 8 (init/seed 0–3) through the harness's `SyncVectorEnv`, comparing the model too.
  - Batch 1 body_pos `['1240ac26e199', '7f7e8f06cae8', 'a6e239b125c4', '1db238954bf8']` equals batch 4. `(state,
    body_pos, site_pos) identical: True`.
  - Different seeds place the stove differently, as LIBERO's sampler intends; the same (init, seed) gives the same
    scene regardless of batching.
- **Horizon premise (measured, round 1, `s5_verify_c.txt`):** unpatched LeRobot `LiberoEnv(episode_length=1200)`
  with noop actions completes `990 policy steps`, then raises `ValueError: executing action in terminated episode`
  at step 991 (inner horizon 1000, 10 settle steps; `artifacts/s5_horizon_check.json`). `ProbedLiberoEnv` raises
  the horizon, and AC5.1's rollouts ran 1200 steps.

### S6: One return per metric per rollout (and per operation): the attribution-ready output (COMPLETE locally: AC6.1–AC6.6 ✅; AC6.7 pending server)

Re-read: PoC §4.1 p. 9 (the four yardsticks), §4.2 p. 9, §4.3 p. 10 ("Success within 30 s → binary return; RMST
→ negative capped time; AUC → win indicator against a sampled human time. HRT is a ratio, so it needs a
first-order (linearized) approximation"), §4.4 p. 10, §3.2 p. 6 (primitives table), §3.4 p. 7; PhAIL Table 1
p. 5 (the six scalars as functionals of F), PhAIL `build/explore_pp_plot.py::_auc_censored`.

**Provenance.** The stalled executor wrote `vlapoc/returns.py`, `scripts/verify_returns_s6.py` and four `s6_*`
artifacts but never logged them. I committed them unchanged as `5f141b1` ("unverified WIP"), so my changes show as a
diff against that commit. Its synthetic run reproduces byte-for-byte: my re-run's stdout equals the WIP
`s6_synthetic_stdout.txt` (`diff` → no difference).

**Files (changes in this session, on top of `5f141b1`):**
- `vlapoc/returns.py`:
  - Renamed `ret_hrt_lin` → `ret_hrt60_lin`, `ret_censored` → `ret_censored60` and `ret_rmst_op` → `ret_rmst60_op`,
    so every return column whose functional has a τ carries it (AC6.6). This diverges from the plan's proposed
    names; the AC wording wins.
  - Registry gains the two §3.2 primitives it lacked: **time-to-success T** and **Kaplan–Meier CDF F(t)**.
    "censored tail" becomes "right-censoring / censored tail".
  - UPH now states Table 1's literal functional, `3600 / E[T | T < τ_episode]` (∝ 1/RMST).
  - The registry has 21 entries.
- `vlapoc/evaluation.py`: one docstring reference follows the rename.
- `scripts/verify_returns_s6.py`:
  - The AC6.4 required list is now the PoC's own list: §3.2's ten primitives, Table 1's six scalars, and §4.1's
    four yardsticks plus the ghost-event rate.
  - The AC6.6 check parses each registry entry's τ and requires it in the column name. Under this check the old
    names `ret_hrt_lin` and `ret_censored` would fail.
  - `--join` now starts from a `returns.parquet`, as AC6.3 says, instead of from a train run id. It also checks
    that the returns' `checkpoint_sha256` equals the train run's recorded weights sha.
  - `AC6.2.pass` is stored as a JSON bool.

**Premises checked:**
- The returns' cell functionals equal S1's. Checked against `vlapoc/evaluation.py`, which defines
  `hrt_cells = rmst(human)/rmst(model)`, macro = the equal-weight mean, `rmst` = ∫₀^τ (1−F), and `auc_censored`
  as PhAIL's verified-pairs rule. **Holds** for the identities AC6.1 asks for.
- **Ties diverge.** PhAIL's `_auc_censored` drops tied pairs from the denominator. `auc_expected` counts a tie as
  ½. They agree when there are no ties, so the synthetic data is jittered off the 0.05 s grid.
  - On real rollouts, model and human T share the 20 Hz grid, so ties can occur. There, mean(`ret_auc_expected`)
    is the Mann–Whitney (ties-½) AUC, not PhAIL's ties-dropped AUC.
  - The difference is recorded, not hidden. AC6.1 only requires the AUC identity on uncensored synthetic data.

**Acceptance criteria:**
- ✅ AC6.1 (synthetic part) `verify_returns_s6.py --synthetic` (LABELLED SYNTHETIC; `artifacts/s6_verify_stdout_r2.txt` with the current column names, and
  `s6_verify_returns.json`. `s6_synthetic_stdout.txt` is the WIP run: same numbers, old names):
  - 10% censoring, at τ_episode only: `max |mean(ret_rmst60) + RMST60_KM| = 1.07e-14`, `max |mean(ret_f30) - F30_KM| = 1.11e-16`.
  - No censoring: `7.11e-15`, `1.11e-16`, and `max |mean(ret_auc_expected) - AUC_censored| = 2.78e-17`.
  - All are ≤ 1e-9. The sampled AUC indicator differs from its expectation by sampling noise only, e.g. `syn/0`:
    0.167 vs 0.1453 over 60 rollouts.
- ⏳ AC6.1 (real part): on the AC5.4 SmolVLA rollouts, pending (see S5 AC5.4).
- ✅ AC6.2 HRT linearization (LABELLED SYNTHETIC; `artifacts/s6_hrt_linearization_curve.csv`):
  - HRT = 0.3796, and `sum_cells mean(ret_hrt60_lin) = -0.3796` (= −HRT at the expansion point).
  - Worst relative error over the 12 perturbations with |ΔHRT|/HRT ≤ 5% is **9.51%** (criterion ≤ 10%): it passes,
    narrowly.
  - The curve shows relative error ≈ the per-cell relative change in T (δ = ±0.01 → 0.93–0.95%, ±0.03 → 2.8%,
    ±0.06 → 5.6–5.7%, ±0.10 → 9.3–9.5%). That is the first-order behaviour of 1/RMST.
  - The worst in-scope case is a 10% slowdown of one cell out of three. It moves macro HRT by only 3.55%, so the
    "|ΔHRT|/HRT ≤ 5%" scope admits it.
  - Inferred: the error bound that matters for attribution is the per-cell perturbation size, not the macro change.
- ✅ AC6.3 join contract: `python -m vlapoc.returns --eval-run 3392bb4d…` (the ACT smoke eval; exit 0) wrote
  `data/eval/returns-3392bb4d5491/returns.parquet` (4 rows) and `operation_returns.parquet` (4 rows). Its
  `hrt_macro_at_policy` is 0.1518 (`artifacts/s6_returns_act_example.txt`). Then
  `verify_returns_s6.py --join data/eval/returns-3392bb4d5491/returns.parquet`:
  - `train run ff64512b4097; returns' checkpoint_sha256 == train run's weights sha: True`.
  - `attribution frame: shape (960, 8)`. The rows are the training subset's (`episode_id`, `content_hash`), i.e.
    v1/candidate. The metric columns are the 6 `RETURN_COLUMNS` (`== RETURN_COLUMNS: True`).
  - `check_scores: {'rows': 960, 'training_episodes': 960, 'covered': 960, 'ok': True}`.
  - `a stale content_hash is rejected: True` ("1/960 score rows are not episode versions trained on by run ff64512b4097").
  - No influence is computed.
- ✅ AC6.4 registry: 21 entries, `required ... missing: []`, `entries without status/reason: []`
  (`artifacts/s6_metrics_registry.json`). Each entry has a transfer status and a reason:
  - ghost-event rate: "not available";
  - intervention/safety-stop: "not transferable";
  - blind scheduling: "replaced";
  - UPH: "projection only";
  - MTBF/A: "adapted";
  - wall-clock T: "adapted";
  - the four PoC-named failure types without a LIBERO rule: "not built".
- ⏳ AC6.5 failure-type subsets: the code path exists (`--real`); its demonstration on AC5.4 is pending.
- ✅ AC6.6: every column carries its metric, and its τ where the functional has one:
  `ret_f30` (30), `ret_rmst60`, `ret_hrt60_lin`, `ret_censored60`, `ret_rmst60_op` (60);
  `ret_auc_sampled`/`ret_auc_expected` have no τ in the functional. Per-rollout registry columns ==
  `RETURN_COLUMNS: True`.
- ⏳ AC6.7 (pending server): returns for every AC5.8 rollout. That means
  `python -m vlapoc.returns --eval-run <id>` per seed's eval run.

#### S5 continuation: AC5.4, AC5.5, AC5.7 and AC4.5 (S5 complete locally)

Placed under S5, although it comes after the S6 entry in the file: S6's first part was logged while these runs
were in flight.

**New dependency (logged per the executor rules).** `HuggingFaceVLA/smolvla_libero` could not load:
`ImportError: 'transformers' is required but not installed. Install it with: pip install 'lerobot[smolvla]'`
(`modeling_smolvla.py:162`).
- What the extra needs, read from installed lerobot's `METADATA`: `lerobot[transformers-dep]` →
  `transformers<5.6.0,>=5.4.0`; `num2words<0.6.0,>=0.5.14`; and `lerobot[accelerate-dep]` (accelerate 1.15.0 was
  already installed).
- Installed in `.venv`: `pip install "transformers>=5.4.0,<5.6.0" "num2words>=0.5.14,<0.6.0"` (exit 0).
- The freeze diff is exactly `docopt 0.6.2, num2words 0.5.14, regex 2026.9.29, rich 15.0.0, shellingham 1.5.4,
  tokenizers 0.22.2, transformers 5.5.4, typer 0.27.2` (`data/scratch/logs/venv_freeze_{before,after}_smolvla.txt`;
  the after list is in `artifacts/s5_venv_freeze_after_smolvla.txt`).
- No other way exists: SmolVLA's VLM backbone is a `transformers` model. S8 (SmolVLA training) needs the same package.
- All SmolVLA runs set `HF_HUB_OFFLINE=1`, so nothing was downloaded at load time.

**Throughput on MPS, measured before sizing AC5.4** (`artifacts/s5_smolvla_throughput.txt`). The checkpoint's
config has `n_action_steps: 1`, so the policy runs a full flow-matching inference (10 steps) at every env step.
- batch 1: about 2.4–2.5 policy steps/s at steady state, ≈ 0.4 s per env-step (40 steps in 21 s including warm-up).
- batch 10: 2.69 s per vector step, ≈ 0.27 s per env-step.
- batch 5 (the AC5.4 runs): 1.45 s per vector step, ≈ 0.29 s per env-step.
- A failed rollout runs the full 1200 steps, so a batch of 5 with one failure costs ≈ 29 min.
- I kept **10 init states per task, as the plan asks**: the count was not reduced. Each task ran as its own
  tracked eval run, so an interruption loses at most one task.

**MPS determinism (measured, `artifacts/s5_smolvla_mps_determinism.txt`):** libero_spatial:0, init/seed 0–9,
batch 10, 30 steps, two processes → `{'rows': 10, 'columns_compared': 30, 'differing': {}}`. Arm action norms
(0.3666–1.3040) agree to the record's 1e-6 rounding. So at a fixed batch plan, SmolVLA on MPS is reproducible at
record precision.

**An interrupted run, recorded and closed.** The first AC5.4 attempt was eval run `8fae29c4`
(`eval-smolvla-libero-harness`, both tasks in one run). It died when the app session ended (2026-10-03 20:38), at
policy step 311/1200 of its first batch, with no rollouts and no outputs. I set it to MLflow status `KILLED`, with
tag `eval.interrupted` saying why, and removed its empty output dir. The rerun went through `screen -dmS …
caffeinate -i` so it could outlive the session (`data/scratch/run_ac5_4.sh`).

**Acceptance criteria:**
- ✅ AC5.4 **real-policy harness validation.** External checkpoint (not trained on our data), harness validation,
  not attribution-grade. `HuggingFaceVLA/smolvla_libero@6721902bc4d61e50a3bfdb11dfb4cb626f05d102`, weights sha256
  `71d9563c8295…`.
  - Settings: MPS, render 256² (the checkpoint's input size), τ_episode 1200 steps (60 s), `control_mode=relative`,
    batch 5, seed_base 0, `record_trajectories=True`.
  - Code at git `2dc6dcb`. MLflow: `e138fec9…` (`eval-smolvla-harness-spatial0`, 1902 s) and `e92604a8…`
    (`eval-smolvla-harness-l10t8`, 3529 s). Both are `FINISHED`, `eval.attribution_grade=false`, and
    `eval.external_checkpoint` names the hub id@rev (`artifacts/s5_ac5_4_mlflow_runs.txt`).
  - Outcomes, from `verify_rollouts_s5.py --summary` (`artifacts/s5_ac5_4_summary.json`, `s5_ac5_4_5_verify_stdout.txt`,
    `s5_ac5_4_eval_stdout.txt`); KM via S1:

    | Task | Success (≤ 1200 steps) | Within LeRobot's budget | T of successes (steps) | Ops done | KM F(30 s) | KM median | KM RMST(60) | Operation pairs |
    |---|---|---|---|---|---|---|---|---|
    | libero_spatial:0 | 6/10 | 6/10 (280 steps) | 67, 71, 73, 74, 75, 77 | {0: 4, 1: 6} of 1 | 0.60 | 3.75 s | 26.19 s | 10 |
    | libero_10:8 (two moka pots on the stove) | 3/10 | 3/10 (520 steps) | 404, 407, 469 | {0: 5, 1: 2, 2: 3} of 2 | 0.30 | ∞ (F < 0.5) | 48.40 s | 15 |

  - **T, ops and censoring are recorded per rollout.** Every failure is censored at exactly 1200 steps. The two
    partial rollouts on task 8 (init 1 and 5) did 1 of 2 operations, giving one event plus a censored tail.
  - LeRobot's own success flag agrees with the probe's goal check on every rollout: `run_rollouts` raises if they
    disagree.
  - No success came after LeRobot's budget, so on these 20 rollouts τ = 1200 changes no outcome against LeRobot's
    280/520-step protocol.
  - The published LIBERO success rates for this checkpoint are not given on its model card or in the LeRobot LIBERO
    docs (both fetched 2026-10-04). So **these numbers are not compared against a reference, and are not evidence
    about the checkpoint's quality.** They show only that the harness runs a real policy and records real outcomes.
- ✅ AC5.5 **trajectory store** (`verify_rollouts_s5.py --trajectories`; `artifacts/s5_ac5_5_trajectories_*.json`):
  - spatial: `LeRobotDataset loads: 10 episodes, 5237 frames; sum(sim_steps) = 5237`.
  - libero_10:8: `10 episodes, 9680 frames; sum(sim_steps) = 9680`.
  - Both: `episodes == rollouts' traj_episode_index and per-episode length == sim_steps: True`;
    `frame_index == policy step 0..sim_steps-1 in every episode: True`.
  - `video keys: []`, so the images are lossless PNG, not video. Frame image is (3, 256, 256), state (8,), action (7,).
  - Each frame is the observation the policy acted on at that step plus the action executed, so (s′, a′) per
    CUPID Prop. 1 is recoverable as (`rollout_id` via `traj_episode_index`, policy step).
  - Size on disk: 569 MB + 1.0 GB (gitignored). The dataset digests are logged as each run's
    `output.trajectories` (`9ab3d9ad…`, `197a7f88…`).
- ✅ AC5.6 **addendum: init state yes, policy noise no (measured, `artifacts/s5_smolvla_batch_effect.txt`).**
  - Rollouts with the same (init j, seed j) of spatial:0 at batch 1 vs batch 5 start from identical t0 sim states
    (`t0_state_sha256 equal: True` × 5).
  - The policy's outcomes differ:
    - init 3 succeeds at step 71 (batch 5) vs 78 (batch 1);
    - init 4 never succeeds within 1200 steps at batch 5, but succeeds at step 71 at batch 1;
    - inits 0–2 fail both ways.
  - Inferred cause: SmolVLA samples flow-matching noise from torch's global RNG for the whole batch. `run_rollouts`
    seeds that RNG once per batch with the batch's first seed, as its docstring already says. So a stochastic
    policy's outcome is a function of (checkpoint, init, seed, **batch plan**).
  - AC5.6 as written, init-state pairing independent of batch size, holds. Reproducing a stochastic policy's
    outcomes additionally needs the same `batch_size`, which every record and MLflow run carries (see Findings).
- ✅ AC5.7 **grounding** (grep, `artifacts/s5_ac5_7_grounding.txt`): `vlapoc/rollouts.py` + `runs.py` cite
  `PoC §9.1 Step 2 (p. 18)`, `§2.1 item 1 (p. 3)`, `§2.1 item 3 (p. 3)`, `§3.5 A (pp. 7-8)`,
  `lerobot.scripts.lerobot_eval.rollout`, and `CUPID Prop. 1 (p. 5)`: all `True`.
- ✅ AC4.5 (S4, now complete): `verify_operations_s4.py` → `definitions of operations_from_trace in vlapoc/ +
  scripts/: 1 (['vlapoc/libero_goals.py']); callers: {'scripts/human_reference.py': True, 'vlapoc/rollouts.py': True}`.
  Synthetic cases: `7/7` (`artifacts/s5_ac4_5_single_impl_r2.txt`).
- ⏳ AC5.8 (pending server), unchanged: the S8 SmolVLA base policy × 50 init states × 20 tasks, τ 60 s, 128²,
  `record_trajectories=True`, for each training seed.
  - Add: use a **fixed `--batch-size`** across seeds and re-runs. The batch-effect addendum shows outcomes depend on
    it.
  - Budget for the cost: at the measured ≈ 0.29 s per env-step on MPS, one failed 1200-step rollout is ≈ 6 min of
    policy compute. The server's GPU throughput is unmeasured.

**Files:** `scripts/verify_rollouts_s5.py`:
- `--summary` (AC5.4: outcomes, the LeRobot-budget column, per-task whole-task and operation KM via S1).
- `--trajectories` now takes several dirs. It checks `frame_index` = policy step and the episode ↔ rollout mapping,
  and writes one JSON per run.
- Data (gitignored): `data/eval/eval-smolvla-harness-{spatial0-20261004T150110Z,l10t8-20261004T153306Z}/`.

#### S6 continuation: AC6.1 (real) and AC6.5 on the AC5.4 rollouts (S6 complete locally)

- ✅ AC6.1 (real part): `python -m vlapoc.returns --eval-run e138fec9…` and `… e92604a8…` both exit 0. They write
  `data/eval/returns-e138fec97361/` (10 rows, 10 operation rows; `hrt_macro_at_policy` 0.1577) and
  `data/eval/returns-e92604a8b995/` (10 rows, 15 operation rows; 0.4161). Then
  `verify_returns_s6.py --real <spatial dir> --real <l10t8 dir>` (`artifacts/s6_real_stdout.txt`):
  - spatial:0: `max |mean(ret_rmst60) + RMST60| = 0.00e+00; max |mean(ret_f30) - F30| = 0.00e+00;
    max |mean(ret_censored60) - censored_rate| = 0.00e+00`.
  - libero_10:8: `7.11e-15`, `5.55e-17`, `0.00e+00`.
  - All are ≤ 1e-9. Censoring occurs only at τ_episode = 60 s, which is the AC's condition.
- ✅ AC6.5 failure-type subsets (`artifacts/s6_real_stdout.txt`). Rollouts per cell × `failure_type`:
  - spatial:0: `success 6, timeout_no_progress 4`.
  - libero_10:8: `success 3, timeout_no_progress 5, timeout_partial_1_of_2 2`.
  - S1 metrics recomputed per subset:
    - spatial success: F(30 s) 1.000, RMST(60) 3.64 s, mean `ret_auc_expected` 0.800;
    - libero_10:8 success: F(30 s) 1.000, RMST(60) 21.33 s, mean `ret_auc_expected` 0.283;
    - the timeout subsets: F(30 s) 0, RMST(60) 60 s.
  - Each eval run here is one task, so the task filter is the run. `returns.parquet` carries `suite`,
    `libero_task_id` and `cell` for multi-task runs.
- **Example tables for the report:** `artifacts/s6_returns_smolvla_example.txt`, both runs' returns and libero_10:8's
  15 operation rows. Libero_10:8 init 3 shows the structure:
  - op 1 covers steps 0–159 (7.95 s) and op 2 covers steps 159–469 (15.50 s); Σ = T = 23.45 s;
  - a partial rollout (init 1) has op 1 = 39.80 s (event) and a censored tail of 20.20 s.

**Notes / divergences (S6):**
- **Operation-level returns with an early-censored tail are lower bounds (measured).** `ret_rmst60_op = −min(T_op,
  60 s)` is the plan's definition. For a censored tail shorter than 60 s (after 1 of 2 operations), it is not the
  per-op term of an RMST.
  - On libero_10:8's 15 operation pairs (2 such tails): mean(−`ret_rmst60_op`) = 32.27 s, but KM RMST(60) over the
    same pairs = 36.99 s.
  - The `event` column marks these rows. Using operation returns with censoring (e.g. §3.4's "negative
    time-to-success with a penalty for ghost events") is the attribution feature's decision. Not changed here.
- **AUC ties:** see the S6 premise above. On the 20 real rollouts, mean(`ret_auc_expected`) is the ties-½ AUC.
- **`ret_hrt60_lin` weights are evaluated at each eval run's own policy, and J = the cells in that run.** For a
  run per task (as here) J = 1. A multi-task eval run gets one J across its cells. The attribution feature must
  compute the weights over the cells it attributes against (inferred; the form is still to be confirmed there, as
  the plan says).
- Renamed columns, as above: `ret_hrt60_lin`, `ret_censored60`, `ret_rmst60_op`.

### Findings for the analyzer (batch 2: S4–S6)
- **Evidence slip (S5 AC5.2), resolved.**
  - The committed `{arm_action_norm: 6}` was a comparator artifact: `str(ndarray)` vs a JSON list. I reproduced it
    by comparing a record against itself.
  - The ACT stdout from a stdin heredoc, where the previous executor reported a stub import, is kept as
    `s5_ac5_2_ac5_3_act_stdout.INVALID.txt`.
  - Every AC5.2 claim now rests on comparisons I re-ran without stubs.
- **LIBERO scene drift with reset count (measured, inferred to be unintended upstream).** `object_property_initializers`
  grows by one on every hard reset (`bddl_base_domain.py:653,674`), so in 4/20 tasks a reused env object places
  fixtures differently (task 8 stove, 7.2 mm).
  - The harness is immune because it uses one fresh env per rollout. The S3 human replay (one env, many resets) is
    not; this is the mechanism behind the S3 demo_19 419/−1/419 finding.
  - S3's 1000/1000 human reference used the demos' own model XML (`reset_from_xml_string`), so it is unaffected. S3
    measured that replay as deterministic, e.g. demo-XML T = `[411, 411, 411]` (`artifacts/s3_replay_determinism.txt`);
    only its BDDL-scene variant drifts.
  - `t0_state_sha256` cannot see fixture placement. A `t0_model_sha256` column would close that gap. **Not built:
    it is a schema change the plan does not ask for.** Planner decision.
- **Stochastic-policy outcomes depend on the batch plan (measured on SmolVLA).** Same init/seed, batch 1 vs 5:
  init 4 succeeds at step 71 vs never. AC5.6 as worded (init states) passes.
  - Per-rollout policy RNG would need patching LeRobot policy internals (noise sampling), which is out of scope.
  - AC5.8 should fix `batch_size` (it is recorded). Planner: decide whether attribution needs batch-independent
    rollouts.
- **New dependency:** `transformers 5.5.4` + `num2words 0.5.14` (the `lerobot[smolvla]` extra; S8 needs it too).
- **SmolVLA cost on this Mac:** ≈ 0.29–0.4 s per env-step on MPS. The checkpoint replans every step
  (`n_action_steps 1`), so one failed 1200-step rollout is ≈ 6 min. AC5.8 (1000 rollouts × seeds) is server-only.
- **Not re-run in this batch:** `scripts/verify_runs_s5.py` (the data+lineage S5 checks, switched to `eval_run` in
  `4448ed2`) and `scripts/lineage_dry_run.py`. Both train tiny ACT runs and belong to S9's end-to-end check.
- **AC6.2 passes narrowly (9.51% ≤ 10%).** The linearization error tracks the per-cell perturbation size, not the
  macro |ΔHRT|.

## Round 1 / Batch 3 (S7–S9): 2026-10-04

### S7: The verification gate: the two-step verdict (macro-KS, then per-metric CIs) (AC7.1 ❌ in part; AC7.2–AC7.4 ✅; AC7.5 pending server)

Re-read: PoC §4.4 p. 10, §3.2 p. 6, §3.4 p. 7, §2.6 p. 5, §9.1 Step 4 p. 19 (PDF dump); PhAIL §3.3 p. 6 (outcomes
(i)/(ii)/(iii), the gate is "regimes (i)+(ii) vs (iii)", "Direction ... comes from a scalar"), §5.1 Table 3 p. 8, §5.3
pp. 9–10, App. I p. 18, App. J and Table 8 pp. 18–19; PhAIL `build/dashboard.py` (`KS_HOLDS = 0.05`,
`KS_WEAK = 0.20`) and `out/explore/ks_power/dashboard_per_obj_macro.json`.

**Prior art:**
- The gate, the logrank sanity line and the four scalars already exist in S1 (`evaluation.macro_ks`, equal to
  PhAIL's `per_obj_macro_bootstrap_ks` on PhAIL's data, S2 AC2.1). S7 adds only the wrapper.
- Holm: statsmodels is not installed, and scipy 1.18.1 has no Holm. `scipy.stats.false_discovery_control` is
  Benjamini–Hochberg only. So `holm()` is a 6-line function, checked against a hand-computed example.

**Files:**
- `vlapoc/verdict.py` (new):
  - `compare_policies`: step 1 is the macro-KS gate. Step 2 bootstraps the four PoC §4.1 yardsticks jointly,
    episode-clustered, with the human anchor fixed:
    - F(30 s) pooled;
    - HRT(τ) per-cell macro;
    - RMST(τ) pooled;
    - AUC vs human, macro.
  - Each metric gets a 95% CI, a two-sided percentile-bootstrap p, and a Holm-adjusted p.
  - `outcome` gives (i), (ii) or (iii), or `gate_only` (see the divergence below). The direction comes only from
    scalars, and the KS value is exposed unsigned. The stratified logrank is the sanity line.
  - `libero_arm` builds an arm from rollout records (operation pairs); `libero_human_arm` uses S4's
    `human_operations.parquet` (all 1000 v0 originals).
  - `null_calibration` implements App. J setup (i). `verify_run` is a new **`verify` stage** MLflow run whose
    inputs are both eval runs' rollouts entities.
  - CLI: `--phail A B`, `--phail-all`, `--null-calibration M`, `--eval-runs A B`.
- `vlapoc/runs.py`: `STAGES` gains `verify`.
- `vlapoc/evaluation.py`: `phail_cohort_arms()`. The S2 loader moved here, so S2 and S7 share one loader.
  `scripts/phail_validate.py::load_arms` now delegates to it; the old and new loaders return identical arms
  (`True`; counts Teleop 396, OpenPI 163, GR00T 164, ACT 151, SmolVLA 116).
- `scripts/verify_verdict_s7.py` (new): Holm, the outcome rule, and the App. I test. It runs as a script or under
  pytest (`3 passed`).

**Premise checked before AC7.1, and FALSE for one pair.** The plan expects "ACT vs OpenPI → (ii) (published: KS
rejects ...)". PhAIL's own released macro-KS for that pair does not reject at α = 0.05:
- `dashboard_per_obj_macro.json`: `['OpenPI', 'ACT'] D 0.141798 p 0.102949 WEAK`. Under `dashboard.py`, `HOLDS`
  means p < 0.05 and `WEAK` means p < 0.20.
- S2's re-run of PhAIL's own function (dashboard H1 order, rng 42) gives p 0.122939.
- In the same file, GR00T vs ACT is `p 0.015992 HOLDS`.
- Inferred: §5.3's "ACT and OpenPI/GR00T are different (KS rejects)" holds for GR00T–ACT in PhAIL's released
  numbers, not for OpenPI–ACT. For OpenPI–ACT only the pooled logrank rejects (Table 3, 7.2e-4).
- I built the gate as the PoC specifies (macro-KS is the gate; logrank is only the sanity line) and report what it
  gives.

**Acceptance criteria:**
- AC7.1, PhAIL S2 cohort, τ 240 s, n_boot 1000, n_boot_ks 2000, seed 0, α 0.05 (`python -m vlapoc.verdict --phail-all`;
  `artifacts/s7_phail_verdicts_stdout.txt`, `s7_phail_verdicts.json`):
  - ✅ **OpenPI vs GR00T → (iii):** macro-KS D̄ 0.1239, **p 0.1099** (published 0.12; PhAIL's JSON 0.1194). No
    metric is significant after Holm. Logrank p 0.803.
  - ❌ **ACT vs OpenPI → (iii), not the plan's (ii):** macro-KS D̄ 0.1418, p 0.1109, so the gate does not fire.
    - The scalars do show the crossing: RMST favours OpenPI (Holm 0.040), and AUC favours ACT (raw 0.036, Holm 0.090).
    - Logrank p 6.4e-4, in line with Table 3.
    - (iii) in all 5 seeds (gate p 0.108–0.120; `artifacts/s7_ac7_1_seed_sweep_stdout.txt`).
    - This is the false premise above. **Planner decision:** restate the expectation as GR00T vs ACT → (ii), or
      state that PhAIL's prose and its released macro-KS disagree.
  - ✅ **(ii) is reproduced on GR00T vs ACT, the pair PhAIL's released macro-KS rejects:**
    - gate D̄ 0.1754, p 0.0130, fires;
    - **RMST favours GR00T** (Δ −23.6 s [−42.9, −6.4], Holm 0.024);
    - **AUC favours ACT** (Δ −0.039 [−0.069, −0.009], Holm 0.042);
    - F(30 s) and HRT favour GR00T but are not significant after Holm (0.068).
    - This is §5.3's crossing: ACT wins the early-time AUC, and GR00T wins integrated RMST.
    - (ii) in 5/5 seeds; AUC Holm p 0.016–0.042, RMST 0.008–0.032.
  - ✅ **Any model vs SmolVLA → (i):** OpenPI, GR00T and ACT each beat SmolVLA. Gate p 0.0005 in each pair, all four
    metrics significant after Holm (each 0.008), and every metric favours the non-SmolVLA model.
- ✅ AC7.2 null calibration, App. J setup (i): same-model halves, 200 trials × 500 inner resamples
  (`artifacts/s7_ac7_2_null_calibration_stdout.txt`, `s7_ac7_2_null_*.json`). Rejection rate at α = 0.05, ±1.96 SE:

  | Arm | Ours: rejection rate | Ours: mean p | PhAIL Table 8 (500 trials): rate | PhAIL: mean p |
  |---|---|---|---|---|
  | OpenPI | 0.040 ± 0.027 | 0.438 | 0.072 ± 0.023 | 0.441 |
  | GR00T | 0.040 ± 0.027 | 0.480 | 0.050 | 0.455 |
  | ACT | 0.035 ± 0.025 | 0.405 | 0.042 | 0.416 |
  | SmolVLA | 0.035 ± 0.025 | 0.380 | 0.070 | 0.408 |
  | Teleop | 0.070 ± 0.035 | 0.490 | 0.050 | 0.479 |

  - All five are inside the plan's [0.02, 0.10] band.
  - Mean p below 0.5 matches PhAIL's own "mild upward bias".
- ✅ AC7.3 KS never ranks (LABELLED SYNTHETIC, App. I shape; `artifacts/s7_verify_stdout.txt`, `s7_ac7_3_intransitive.json`):
  - KS sup-sign winners: `A vs B: sup 0.40 -> A; B vs C: sup 0.50 -> B; C vs A: sup 0.50 -> C`, a cycle.
  - RMST(70 s): A 35.50, B 21.10, C 35.30, which is transitive.
  - The verdict for A vs B: gate D̄ 0.40 (unsigned), p 0.0025 → (i), **direction B**, from RMST Δ +14.40 s
    [+9.03, +19.84]. The KS sup-sign favours A.
  - The verdict dict's gate carries only `{test, D, p, D_cells, n_boot, fires}`; there is no sign field.
- ✅ AC7.4 LIBERO smoke, recorded as a run. Labelled smoke: the policies differ in architecture and training data,
  and the SmolVLA arm is an external checkpoint.
  - ACT `train-full-s0` was evaluated on libero_spatial:0 init 0–9 (eval run `fe507fa3`, 128², CPU, batch 5,
    τ 1200 steps): **0/10 succeeded**, all censored at 1200, as expected for a 5-step smoke checkpoint.
  - The SmolVLA arm is AC5.4's `e138fec9`: the same task and init/seed 0–9, **6/10 succeeded**.
  - `python -m vlapoc.verdict --eval-runs fe507fa3… e138fec9… --name verify-smoke-act-vs-smolvla-spatial0` produced
    **verify run `5bb1615b`**:
    - gate D̄ 0.60, p 0.0090, fires → (i), direction SmolVLA;
    - all four metrics are significant after Holm (each 0.008): F(30 s) 0.0 vs 0.6, RMST 60.0 vs 26.19 s, HRT
      0.069 vs 0.158, AUC 0.000 vs 0.480;
    - logrank p 0.004.
  - The run's tags: `verify.outcome=i`, `verify.attribution_grade=false`. The label says "smoke ... exercises the
    code only" (`artifacts/s7_ac7_4_libero_smoke_stdout.txt`, `s7_ac7_4_libero_smoke_verdict.json`).
  - `trace(5bb1615b)` (`artifacts/s7_ac7_4_trace_summary.txt`): the verify run consumes `rollouts:b2844f80…` ←
    `fe507fa3`, which comes from checkpoint ← `train-full-s0` ← v1/candidate ← derive-v1 ← v0 ← build-v0, ending
    at **20 raw files**. It also consumes `rollouts:a820e989…` ← `e138fec9` ← checkpoint `71d9563c…` ← **no
    producer**. That dead end is visible, as an external checkpoint should be.
- ⏳ AC7.5 (pending server): full-data SmolVLA vs a drop-20-random retrain, each evaluated per AC5.8. Then
  `python -m vlapoc.verdict --eval-runs <full> <drop20> --name verify-full-vs-drop20-seed<s>`. See the runbook (S9).

**Divergences / notes (S7):**
- **`gate_only` outcome.** When the gate fires but no metric is significant after Holm, the verdict says so instead
  of forcing (i) or (ii). PhAIL's taxonomy has no name for this case. It did not occur on PhAIL's pairs.
- **The human anchor is held fixed in the difference bootstrap.** It is the common yardstick of both arms (plan:
  the HRT anchor is fixed across arms). S2's single-policy HRT CI resamples the human, matching PhAIL's Table 2.
  For a difference of two policies on one anchor, holding it fixed removes common-mode noise (inferred).
- **The p-value floor.** With n_boot 1000 the smallest bootstrap p is 2/1001 ≈ 0.002, and Holm × 4 gives 0.008.
  The macro-KS floor at 2000 is 0.0005.
- **LIBERO arms use operation pairs (S4 rule), as S1 does.** This is the same granularity as PhAIL's per-operation
  CDFs. libero_spatial's single operation equals whole-task T.

### S8: SmolVLA base-policy training definition with per-subset normalization (AC8.1–AC8.4 ✅; AC8.5 pending server)

Re-read: brief owner decision 3; PoC p. 1, §2.3 p. 4, §10 p. 20; LeRobot `lerobot_train.py:212-400` (where
`dataset.meta.stats` feeds `make_policy`, the processors' `dataset_stats`, and, with a pretrained policy, the
normalizer/unnormalizer overrides at l. 376-377), `datasets/factory.py` (ImageNet stats on camera keys),
`datasets/compute_stats.py:582-640` (`aggregate_feature_stats`, `aggregate_stats`), `datasets/dataset_tools.py:852-950`
(LeRobot's own subset metadata path), `policies/factory.py:240-345` (`make_policy` with `pretrained_path`).

**Prior art (measured from LeRobot's source):**
- LeRobot already computes a subset's stats the way owner decision 3 needs. When its dataset tools copy a subset
  of episodes, they `aggregate_stats` the kept episodes' stored `stats/<feature>/<stat>` columns from
  `meta/episodes` and write that as the new dataset's `stats.json` (`dataset_tools.py:885-950`).
- What LeRobot lacks is a way to train on `episodes=[...]` of an existing dataset with those stats: `DatasetConfig`
  has no stats field, and `train()` always uses the full `meta/stats.json`.
- So `vlapoc/subset_stats.py` calls LeRobot's `aggregate_stats` on the stored per-episode stats. There are no frame
  reads, and no 7 GB `recompute_stats` copy. `lerobot_train_entry.install_subset_stats` puts the result into
  `dataset.meta.stats`, which every downstream consumer in `train()` reads.

**Files:**
- `vlapoc/subset_stats.py` (new): `episode_stats`, `subset_stats`, `to_json`, `stats_digest`, `check`, and a CLI.
- `vlapoc/lerobot_train_entry.py`:
  - `install_subset_stats` wraps `make_train_eval_datasets` in the training subprocess. After LeRobot builds the
    dataset and puts ImageNet stats on the cameras, it replaces every non-camera feature's stats.
  - `policy_pretrained_path`: the config is read via `draccus.load` from the pretrained dir, its features are
    cleared so they come from our dataset, overrides are applied, and `pretrained_path` is set. This is what
    `lerobot-train --policy.path=...` does in `TrainPipelineConfig.validate`.
- `vlapoc/runs.py`, `train_run`:
  - `normalization="per_subset"` is now the **default** (owner decision 3); `"full"` reproduces LeRobot's default.
  - It writes `<out_dir>/subset_stats.json`, logs it as an artifact, and sets tags `train.normalization`,
    `train.subset_stats.sha256` and `train.subset_stats.n_episodes`. The param `normalization_stats` holds a
    constant string.
  - `pretrained_path` adds the tag `train.pretrained.weights.sha256`.
  - `diff()` reports `data.normalization` (both digests and `differ`).
- `scripts/verify_subset_stats_s8.py` (new): `--check`, `--train-smolvla`, `--train-act`, `--normalizer`,
  `--drop-subset`, `--diff`.

**Downloads (owner-approved; source, revision, size recorded):**
- `lerobot/smolvla_base` @ `d9f33c94a60fb382c90dea2164c96845bd955e28` (Apache-2.0, last modified 2026-09-17), into
  `data/hf_models/smolvla_base`. I fetched only `model.safetensors` (906,712,520 bytes, sha256 `7cd549ac2351…`),
  `config.json`, `README.md` and the four `policy_*` processor files. `du` 865M. I skipped the 8 MB gif and the
  notebook. Log: `data/scratch/logs/smolvla_base_download.txt`, exit 0.
- **No VLM weights downloaded.**
  - smolvla_base's config names `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`. The HF API shows that
    `HuggingFaceTB/SmolVLM2-500M-Instruct` (batch 2's local copy) **redirects to the same repo at the same sha
    `7b375e1b…`** (`id HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, measured). So batch 2's 11 non-weight files are
    the right ones (`cmp`: `tokenizer.json`, `config.json` identical).
  - For SmolVLA training, the same 11 files (4.7 MB) are placed in a repo-local HF cache,
    `data/hf_home/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct`, with `refs/main = 7b375e1b…`. This is
    needed because smolvla_base's saved preprocessor names the tokenizer by hub id. The run uses
    `HF_HOME=data/hf_home HF_HUB_OFFLINE=1` (offline `AutoConfig` and `AutoProcessor` load: measured OK).
  - The VLM weights come from smolvla_base's own `model.safetensors` (`load_vlm_weights=false`).
  - Nothing is written to `~/.cache/huggingface`.

**Acceptance criteria:**
- ✅ AC8.1 `python -m vlapoc.subset_stats --check v0` / `verify_subset_stats_s8.py --check v0`
  (`artifacts/s8_ac8_1_check.json`): `aggregate_stats` over all 1000 episodes vs `meta/stats.json`:
  - `observation.state`: mean 4.16e-17, **std 2.43e-10**, min/max/count 0;
  - `action`: mean 5.55e-17, std 1.93e-10, min/max/count 0;
  - all ≤ 1e-9, matching P23's 2.4e-10.
- ✅ AC8.2 **SmolVLA, 2 steps, MPS, per-subset stats, fine-tuned from smolvla_base on `v0/reference` (40 episodes).**
  - Train run `0a33bc75` (`train-smolvla-2step-persubset-v0ref`), 18.0 s. Losses 1.594 and 2.749; `episodes_match: true`.
  - Its `train_config.json`: `policy.type smolvla`, `pretrained_path data/hf_models/smolvla_base`, `device mps`,
    `load_vlm_weights False`, input features `observation.images.image`, `observation.images.image2`,
    `observation.state` (taken from our dataset), 40 episodes.
  - `--normalizer 0a33bc75` (`artifacts/s8_ac8_2_normalizer_smolvla_0a33bc75.json`, `s8_ac8_2_3_stdout.txt`), for
    the normalizer **and** the unnormalizer, max |saved − subset stats|:
    - `observation.state.mean` 4.54e-08, `.std` 2.77e-08;
    - `action.mean` 1.71e-09, `.std` 2.80e-08.
    - All are ≤ 1e-6 (the saved tensors are float32).
  - The same tensors vs the full `stats.json`: 1.09e-02, 4.95e-02, 9.00e-03, 6.26e-03, so they differ.
  - Tag `train.subset_stats.sha256` == recomputed digest: `True` (`e9e51a0cf12a…`).
  - **The pretrained weights were really loaded** (`artifacts/s8_ac8_2_pretrained_weights_check.txt`). 500/500
    tensor names match smolvla_base. **378 tensors are bit-equal to smolvla_base after 2 steps**, i.e. the frozen
    VLM (0 changed tensors under `.vlm.`). The 122 that changed are the trainable action expert and its projections
    (`train_expert_only=true`).
  - One failed attempt first: train run FAILED, dir `data/runs/train-smolvla-2step-persubset-v0ref-20261005T012655Z`.
    It died offline on `AutoConfig.from_pretrained(<hub id>)`, because the repo-local cache lacked `refs/main`.
    Fixed by recording `refs/main` (same sha). FAILED runs form no lineage edges.
  - The same check on the two ACT runs below passes too: per-subset mean/std are within ≤ 6e-8 of the subset
    stats and 1e-2 away from `stats.json`.
- ✅ AC8.3 two subsets differing by k = 5 episodes. `v0/reference` (40) and `drop_random_k(v0/reference, 5, seed 0)` =
  subset `e470d671…` (35). Two ACT 2-step per-subset runs: `a8647af8` and `fea7ac62`. `--diff`
  (`artifacts/s8_ac8_3_diff.json`):
  - `data.n_removed 5, n_added 0`;
  - `data.normalization {"stats_sha256_a": "e9e51a0c…", "stats_sha256_b": "ac940b92…", "differ": true}`;
  - `params that differ: []; config keys that differ: []; git same: True`.
  - So the stats difference appears with the data, and nothing in params or config differs.
- ✅ AC8.4 grounding (grep, `artifacts/s8_ac8_4_grounding.txt`): `PoC p. 1`, `§2.3 (p. 4)` and "owner decision 3"
  (brief) are cited in `subset_stats.py` and `lerobot_train_entry.py`. All `True`.
- ⏳ AC8.5 (pending server): SmolVLA on `v0/candidate` (960 episodes), per-subset stats, 2–3 seeds, fixed config
  and steps. Command and estimates are in the runbook (S9).

**Divergences / notes (S8):**
- **Which initialization the server recipe uses is open.** The plan's training spec is LeRobot's LIBERO docs example
  (quoted, fetched 2026-10-04): `--policy.type=smolvla --policy.load_vlm_weights=true --steps=100000
  --batch_size=64`. That initializes from the SmolVLM2 backbone and needs its 2.03 GB weights on the server.
  - The local check fine-tuned from `lerobot/smolvla_base`, the approved download.
  - Both keep data as the only variable if fixed across seeds. **Planner/owner: pick one for AC8.5.**
  - The hook is the same either way. The runbook lists both commands.
- **Default changed:** `train_run` now normalizes per subset unless told `normalization="full"`. The data+lineage
  smoke runs (`train-full-s0` etc.) were trained with full stats; their MLflow params say so and are unchanged.
- **Quantile stats** (q01…q99) are aggregated by LeRobot as count-weighted means of per-episode quantiles, which is
  an approximation. Neither ACT nor SmolVLA uses them (MEAN_STD); a QUANTILES-normalized policy (pi05) would.

### S9: End-to-end evaluation chain locally, and the server runbook (AC9.1–AC9.3 ✅)

Re-read: PoC §9.1 Steps 2–4 pp. 18–19, §2.1 item 1 p. 3, §10 p. 20; the brief's constraint "mark every
server-dependent AC explicitly as 'pending server', with exactly what will be run there".

**Files:**
- `scripts/lineage_dry_run.py`: new `--eval-smoke` mode (`eval_smoke()`). It writes this feature's `artifacts/s9_*`,
  never the data+lineage feature's S6 files. The chain:
  - `build(v0)` and `split(v0)`, reused;
  - `select` drop-random-5 of `v0/reference`;
  - `train` ×2 (ACT, 2 steps, MPS, per-subset stats from S8);
  - `eval` ×2 (real LIBERO rollouts: libero_spatial:0, 2 init states, τ 1200 steps, CPU, batch 2);
  - `returns` ×2 (S6, now a tracked stage);
  - `verify` (S7);
  - then `trace(verify)`.
- `vlapoc/returns.py`: `returns_run()` and `--track`. This is a **`returns` stage** MLflow run whose input is the eval
  run's rollouts entity and whose outputs are `returns.parquet`, `operation_returns.parquet`, `returns_meta.json` and
  `metrics_registry.json` by sha256. `vlapoc/runs.py` `STAGES` gains `returns`.
- `artifacts/server-runbook.md` (new).

**Acceptance criteria:**
- ✅ AC9.1 **one command runs the local chain:** `.venv/bin/python scripts/lineage_dry_run.py --eval-smoke` → exit 0
  in **122.6 s**, chain `eval-smoke-k5-s0-20261005T033607Z`, git `b59c3d59`, `dirty=False`
  (`artifacts/s9_eval_smoke_stdout.txt`, `s9_chain_runs.csv`, `s9_summary.json`).
  - 10 runs, all `FINISHED`, **`stand_in false` for every one**:
    - build-v0 and split-v0 (reused);
    - `select-drop-random-5-s0` (child `e470d671…`, 35 of 40);
    - `train-ref` `8e61cf2e` (40 episodes) and `train-ref-drop-random-5` `4e0c4115` (35);
    - `eval` `12bd8940` (0/2 succeeded) and `c03b9928` (0/2);
    - two `returns` runs;
    - `verify` `7080353a`: macro-KS D̄ 0.0000, p 1.0000 → **(iii)**, labelled smoke.
  - `trace(verify 7080353a)` (`artifacts/s9_trace_verify.json`): `10 runs, stages ['build', 'eval', 'select',
    'split', 'train', 'verify']; consumed kinds ['checkpoint', 'rollouts', 'subset']; dead ends []; stand-in runs in
    the trace: []; raw files reached: 20` (first `KITCHEN_SCENE3_…`, sha256 `6b30906a52a5741e`).
  - The path: verify ← rollouts ← eval ← checkpoint ← train ← subset (`v0/reference` ← split-v0 ← build-v0, and
    `e470d671` ← select ← `v0/reference`) ← v0 ← raw files.
  - The returns runs sit beside the verdict (both consume the same rollouts entities). They appear in the chain
    graph `artifacts/s9_lineage_graph.mmd`, not on the verdict's upstream path. The verdict needs operation-level
    (T, E) pairs, which are in the rollouts.
  - **New measured fact from the trace: ACT training on MPS is bit-reproducible** (`artifacts/s9_train_determinism.txt`).
    The chain's `train-ref` and S8's `train-act-2step-persubset-v0ref` (same subset, seed 0, 2 steps, separate
    processes, different commits) have the same weights sha256 `cdee42fbf84f56ec…` and normalizer `084a3540…`.
    The drop-5 pair is also identical (`fac999b0…`). `trace` therefore lists both runs as producers of each
    checkpoint, which is correct for content-addressed entities.
- ✅ AC9.2 the runbook (`artifacts/server-runbook.md`) covers AC8.5, AC5.8, AC6.7 and AC7.5, in dependency order. For
  each it gives:
  - the exact commands (the existing CLIs: `eval_smoke.py --track`, `vlapoc.returns --track`,
    `vlapoc.verdict --eval-runs`; plus `train_run`/`select_run` one-liners, since there is no train CLI);
  - the inputs by id: v0 `3c4dca68…`, `v0/candidate` `8ee096b7…` with 960 episodes / 192,425 frames (measured);
  - outputs, checks, and disk estimates labelled measured or inferred:
    - 37 KB per stored frame at 128² (6.9 GiB / 200,340 frames, measured);
    - 108 KB per frame at 256² (measured on the AC5.4 trajectories);
    - 1.2 GB per SmolVLA checkpoint (measured: 1,197,789,224 B);
    - 11–18 KB per rollout row (measured);
    - ≈ 11–22 GB of trajectories per seed (inferred);
    - total ≈ 60–95 GB plus 20 GiB headroom (inferred).
  - It also records the batch-2 operational rules: fixed batch size; one env per rollout; a job runner; closing
    interrupted MLflow runs; a throughput probe first; a GPU determinism probe; a clean git tree.
  - It names the open initialization choice for AC8.5 (VLM backbone vs smolvla_base).
- ✅ AC9.3 **proposed spec-coverage rows for the summarizer** (`tracker/index.html` is the summarizer's to edit):

| PoC § | Status | Where / why |
|---|---|---|
| §3.2 primitives (T, ghost events, right-censoring, KM CDF, Success@τ, median, RMST, UPH, completion, MTBF/A, HRT, clustered bootstrap, macro-KS, P-P) | **built** (S1), validated on PhAIL's real data (S2) | Ghost events are built and validated on PhAIL. For LIBERO they are **not transferable**: LIBERO defines no unrecoverable-state rule (P13, owner decision 6) |
| §3.4 PhAIL fit (verification gate, richer targets, declared targets, strata, operation-level returns, failure taxonomy, fair comparisons) | **built** (S6 returns, S7 gate), except: ghost-event-type taxonomy **not transferable**; blinded scheduling **replaced** by fixed init states + seeds | `vlapoc/returns.py`, `vlapoc/verdict.py` |
| §3.5 A (LIBERO methodology: T per episode / per sub-task, censoring, human reference from demos) | **built** (S3, S4, S5) | Approximate by design: demos and rollouts do not share initial states, and the demo-XML vs BDDL scenes differ (R4, S3) |
| §3.5 B (PhAIL data and license) | **answered** (S2: CC BY 4.0, public bucket, 449-episode fine-tuning set, 13.95 GB, not downloaded) | real-robot stage out of scope |
| §4.3 (one return per metric) | **built** (S6), **pending server** at scale (AC6.7) | `ret_hrt60_lin` is a first-order form; the attribution feature should confirm it (R10) |
| §9.1 Step 2 (rollouts with fixed init states, goal check, T) | **built** and validated on a real policy locally (S5, AC5.4), **pending server** at full scale (AC5.8) | |
| §9.1 Step 4 (the gate on a counterfactual pair) | **built** (S7), **pending server** (AC7.5) | AC7.1 expectation for ACT vs OpenPI is disputed (S7 premise) |
| §10 Evaluation layer (KM CDFs with ghost and censoring rules, HRT vs LIBERO demos, clustered bootstrap, macro-KS, P-P) | **built** locally; attribution-grade policy training and the 1000-rollout evals **pending server** (AC8.5, AC5.8) | |

### Findings for the analyzer (batch 3: S7–S9)
- **S7, false premise in AC7.1 (for the planner).** The plan expects ACT vs OpenPI → (ii) because "KS rejects".
  - PhAIL's own released macro-KS for that pair is p = 0.103, "WEAK" under its `KS_HOLDS = 0.05`
    (`dashboard_per_obj_macro.json`). Re-running PhAIL's function gives 0.123 (S2). Our gate gives 0.111, i.e. (iii),
    in 5/5 seeds.
  - The (ii) trade-off is real and reproduced on **GR00T vs ACT** (gate p 0.013; RMST favours GR00T, AUC favours
    ACT, both significant after Holm; 5/5 seeds).
  - Restate AC7.1, or record the paper-vs-JSON disagreement. I did not change the gate, since macro-KS is the PoC's
    gate and logrank is only the sanity line.
- **S7, a fourth verdict state, `gate_only`:** the gate fires but no metric is significant after Holm. It is reported
  rather than forced into (i)/(ii). Not seen on PhAIL's pairs.
- **S7, the human anchor is held fixed** in the difference bootstrap (inferred: removes common-mode noise). S2's
  single-policy HRT CI resamples it, to match PhAIL's Table 2.
- **S8, open choice for AC8.5 (planner/owner).** One option initializes from the SmolVLM2 backbone, as the plan and
  LeRobot docs do (needs a 2.03 GB unapproved download). The other fine-tunes from `lerobot/smolvla_base`, which is
  approved and verified locally. The hook is identical either way.
- **S8, default changed.** `train_run` now normalizes per subset (owner decision 3). Older runs used full stats, and
  their params say so.
- **S8, external facts measured:**
  - `HuggingFaceTB/SmolVLM2-500M-Instruct` is a redirect to `…-Video-Instruct` (same sha `7b375e1b`), so batch 2's
    files were already correct.
  - `refs/main` must exist in a repo-local HF cache for offline hub-id resolution (the first SmolVLA attempt FAILED
    on it).
- **S9, determinism:** ACT 2-step training on MPS reproduced bit-identical weights across separate runs.
  `trace` lists both runs as producers of the one checkpoint entity. That is the right semantics, but a reader of the
  graph should expect it.
- **Not re-run in batch 3:** the data+lineage feature's full dry run (`lineage_dry_run.py` without `--eval-smoke`) and
  `verify_runs_s5.py`. Both now exercise the real eval stage and per-subset normalization by default, but they train
  four ACT runs on 960-episode subsets (≈ 7 GB of HF cache per new subset). Their regression run is left to the
  refactorer or analyzer. The eval-smoke chain covers the same stage code at smaller scale.
- **Downloads in batch 3:**
  - `lerobot/smolvla_base@d9f33c94` (906,712,520-byte weights + config/processor files, 865M on disk; approved);
  - the 4.7 MB VLM config/tokenizer cache (same files as batch 2's copy, a repo-local cache at `data/hf_home`).

## Round 2: 2026-10-05

Inputs: `analysis-r1.md` (REWORK, planner+executor: E1–E5, F-P1–F-P5) and `plan.md` r2 ("Changes in this
revision (r2)", "Declared functionals (r2)", and every AC marked (r2)). Scope: exactly the (r2) ACs. No rebuttals:
every analyzer item routed to the executor is accepted.

**Lineage stores used.**
- MLflow-writing tests go to a scratch copy of the lineage store, with the copy's experiment `artifact_location`
  repointed (the copied `mlflow.db` stores the real `file:///…/data/lineage/mlartifacts`, measured with sqlite).
- The real store is used only for the corrected `--eval-smoke` chain (AC9.1/AC7.9). Each AC says which it used.

### AC2.9 (r2, E3): deterministic PhAIL cohort export ✅

- **Premise checked (quoted, the clone):** `build/markup/loader.py:316-317` reads
  `candidate_urls = all_candidate_urls | _walk_dataset_inference_urls()` / `for url in candidate_urls:`. It iterates
  a Python set, so episode order follows hash randomization.
- **Change:** `scripts/phail_export_cohort.py` now sorts the loaded episodes by `s3_url`
  (`eps = sorted(loader.load(cohort="mixed"), key=lambda e: e.s3_url)`) before writing the parquet *and* before
  computing PhAIL's reference numbers. So both sides of the AC2.1 equivalence see the same order.
- **Byte identity (measured, `artifacts/s2_ac2_9_export_determinism.txt`):** three exports in fresh processes, with
  `PYTHONHASHSEED=0`, `=1` and unset (`env -u`), all `exit 0`:
  - `cohort_mixed.parquet` sha256 `35b7181bc92c42c5a7175cb36e1acb9b93f2167c71388418506cc5c9754418ad`, identical ×3;
  - `s2_phail_reference.json` sha256 `22549627c8b4bc170b3351b96bf27a5b7e0b5870678f1c7d24c221474438b707`, identical ×3;
  - 990 rows, `s3_url` monotonic increasing and unique.
- **S2 regenerated on the sorted export** (`scripts/phail_validate.py`, exit 0; `artifacts/s2_phail_validate_stdout.txt`,
  `phail_validation.json`, `s2_phail_reference.json`, `s2_fig_km_pp_ACT_Batteries.png`):
  - AC2.1 still holds: max |ours − PhAIL| ≤ 3.553e-15; `macro-KS D and p identical for all 6 pairs (same RNG): True`.
  - **Updated p-values** (PhAIL's function and ours, dashboard H1 order, rng 42, n_boot 2000):

    | Pair | p (round 1) | p (sorted export) | D (unchanged) |
    |---|---|---|---|
    | OpenPI–GR00T | 0.117941 | **0.125437** | 0.123916 |
    | OpenPI–ACT | 0.122939 | **0.110445** | 0.141798 |
    | GR00T–ACT | 0.010495 | **0.008996** | 0.175409 |
    | OpenPI–SmolVLA, GR00T–SmolVLA, ACT–SmolVLA | 0.000500 | 0.000500 | 0.465455 / 0.434398 / 0.360733 |

  - Every D, point estimate and conclusion is unchanged:
    - AC2.3 23/26 within tolerance; the same 3 pooled-RMST misses, caused by the 5 private episodes;
    - AC2.4 winners GR00T / OpenPI / GR00T / ACT, 3 distinct;
    - AC2.5 human-resampled still closer (0.628 vs 0.808 pp).
  - The CIs move slightly (e.g. OpenPI HRT [12.402, 15.684]) because the bootstrap draws follow episode order.
- **Disclosure, for the report:**
  - Bootstrap p-values and CI endpoints are Monte-Carlo draws, conditional on episode order and seed.
  - D and the point estimates do not depend on order.
  - The fetch's reproduction route is pos3 (`pos3.download(..., profile=PUBLIC)`, the phail-paper README recipe),
    installed in `.venv-phail`.
- The S7 artifacts are regenerated from this export under AC7.6 (below).

### AC6.8 (r2, E2): one metric registry ⏳ in progress (not yet verified)

- `vlapoc/metrics_registry.py` (new) is a data-only module: `ENTRIES` (29), `VERDICT_FAMILIES`, `FAMILY_TAU_S`,
  `entry`, `return_columns`, `verdict_metrics`, `to_json`. Each entry has the fields the plan lists: `key`, `metric`,
  `family` (W / O / O-descriptive / PhAIL / scoring-only / primitive / procedure / protocol / not-transferable), `T`,
  `tau`, `censoring`, `functional`, `functions`, `return_columns`, `verdict`, `identity`, `transfer_status`, `reason`,
  `source`.
  - UPH is declared as `3600 / RMST(τ_episode)`. Table 1's form and common.py's form are recorded as notes.
  - `rmst60_op` and `hrt60_op` are labelled `O-descriptive` (S4 AC4.4 r2 note).
- `vlapoc/evaluation.py`: the local `METRICS` table is removed. `metric_registry()` now reads the registry. New
  functions: `auc_ties_half`, `macro_auc_ties_half`, `rmst_capped`, and `libero_arm` / `libero_human_arm` (families
  W and O, moved here from `verdict.py` per S1 r2).
- `vlapoc/returns.py`: `REGISTRY` removed; `metrics_registry()` = `to_json()`; `RETURN_COLUMNS` =
  `return_columns("W")`. Also changed (verified under AC6.1/6.11/6.9):
  - `ret_auc_sampled` counts ties ½;
  - `operation_returns` gives `ret_rmst60_op_cap` (−60 for the unfinished attempt);
  - `load_eval_rollouts` uses a `TemporaryDirectory`.
- **Pending:** `vlapoc/verdict.py` still has its `METRICS` table (grep: `verdict.py:71`). It moves to the registry
  with the family-W/O verdict rework, after which the AC6.8 checks run.
