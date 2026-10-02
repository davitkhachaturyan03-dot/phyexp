# Changelog: data layer + lineage layer

Changes after the first release (2026-10-01), one dated entry per change, added per vla-build step 5.

## 2026-10-02: Answer-key leak closed: v1's `parent_content_hash` removed, `derive_version` keeps parent order, v1 fenced
- **Why:** owner request, 2026-10-01: "also fix the leak now as a separate story inside the feature".
  It is implemented as evaluation-layer S0 (`tracker/features/2026-10-01-evaluation-layer/`), under owner
  ruling A of 2026-10-02: fix the code, fence v1, don't re-derive.
- **Grounding:** PoC §2.1 item 4 p. 3: corrupt known subsets, "Then check whether your attribution method
  flags exactly those". The method under test must not be able to read the answer off the tables it reads.
- **Leak channels found in v1 (measured, evaluation-layer execution log S0):**
  1. **Column.** v1's episode table and catalog copy carried `parent_content_hash`. It differs from
     `content_hash` for exactly 100 rows, and that set equals the labelled set. This feature's plan
     **required** the column (AC4.1: "a `parent_content_hash` column"; AC4.7 asserted the column list
     with it), and neither analyzer round caught it.
  2. **Row order.** `delete_episodes` + `merge_datasets` appended the edited episodes, so v1's
     `episode_index` 900–999 are exactly the 100 corrupted episodes. They also sit in their own data
     files, `data/file_index` 73–80. v0 has 73 data files; v1 has 81.
- **Fix:**
  - `derive_version` now writes the child in one LeRobot writer pass in parent order, so every episode
    keeps its parent's `episode_index`.
  - Pre-edit hashes go to the lineage-only `versions/<id>/parents.parquet`, read through
    `vlapoc.lineage.parent_hashes`.
  - `compare_blind` / `check_blind` / `assert_blind` compare a derived version's method-facing tables
    (episode table, catalog copy, `meta/episodes`) with the parent's, and every derivation runs that
    check before registering.
  - v1's two tables were rewritten with exactly v0's 21 columns, in v0's row order. The physical copy is
    unchanged, and v1 is fenced from method use: `versions/16076b9c…/fence.json` records the before/after
    table sha256s, which also leaves a trace of this rewrite in the store.
- **v1 disposition:**
  - `dataset_version_id` is unchanged (`16076b9c…`; `version recompute v1` → match).
  - `labels.parquet`, `version.json` and `files.json` are byte-unchanged.
  - The S6 checkpoints' episode lists stay valid, because the physical copy was not touched.
  - `check_blind(v1)` = `['episode_index']`, the positional channel, fenced.
  - The injection feature derives its real corrupted versions with the fixed code.
- **Prior art:** LeRobot 0.6.1 `dataset_tools` has no reorder or in-place replace
  (`delete_episodes, split_dataset, merge_datasets, modify_features, add_features, remove_feature,
  modify_tasks, recompute_stats, …`). The public writer `libero_convert.convert` uses was the only
  order-preserving option without parquet surgery.
- **Files:**
  - `vlapoc/derive.py` (writer pass, `iter_stored_episodes`, blind check before registering, `parents.parquet`);
  - `vlapoc/lineage.py` (blindness section, `version check-blind` CLI, `version_episodes` docstring);
  - `vlapoc/catalog.py` (comment); `vlapoc/runs.py` (mechanism param);
  - `scripts/verify_derive_s4.py` (AC4.1 reads `parent_hashes`; AC4.7 now expects exactly
    `CATALOG_COLUMNS`; `--artifacts`);
  - new `scripts/fix_v1_leak_s0.py` and `scripts/verify_blind_s0.py`.
- **Evidence:**
  - `version check-blind v1`, before the fix → `['(row order)', 'episode_index', 'parent_content_hash']`.
  - `scripts/fix_v1_leak_s0.py` → `"equal_v0_columns": true, "n_columns": 21, "version_id_unchanged": true`,
    `"label_join_rows": 100`, `"parent_hashes_equal_v0": true`, `check_blind(v1) = ['episode_index']; fenced: True`.
  - `scripts/verify_blind_s0.py`: 20-episode scratch parent, 3 edits incl. +20 frames →
    `episode_index equal to parent's: 20/20; edited episodes now at [3, 11, 16]`, `check_blind(child) = []`.
    All 4 tampered tables are rejected.
  - `verify_derive_s4.py --v1 --columns` → `AC4.1 parent_hashes(v1) == v0 content_hash for all 1000 rows:
    True`, `AC4.7 v1 columns == v0 catalog columns (CATALOG_COLUMNS, no parent_content_hash): True`.
- **Divergence:**
  - AC4.1 and AC4.7 of this feature's plan are amended: no `parent_content_hash` in a method-facing table.
  - Derivation cost rises from about 5 min to about 8.9 min at full scale (inferred from 375.9 frames/s
    measured on the scratch derivation), because every frame is re-encoded (lossless; content hashes
    unchanged).
  - `isolate_hf_cache=False` now leaves 0 cache files, since the writer pass loads no HF dataset.
