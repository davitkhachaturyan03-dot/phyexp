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
- ⏳ AC0.6: pending. The data+lineage `changelog.md` entry and the report note are not yet written
  (next step).
- ⏳ AC0.7: pending a grep check. The docstrings citing PoC §2.1 item 4 p. 3 are in `derive.py` (module,
  `derive_version`, `_derived_episode_table`), `lineage.py` (blindness section, `version_episodes`,
  `parent_hashes`, `assert_blind`) and both new scripts.

**Other checks (measured):**
- `verify_derive_s4.py --guard --columns --cache --v1 --artifacts artifacts/s0_verify_derive_s4`
  (first run, 15:07):
  - `AC4.3 ... LineageError raised: True`;
  - `AC4.7 v1 columns == v0 catalog columns (CATALOG_COLUMNS, no parent_content_hash): True`;
  - `AC4.1 v1 loads: episodes=1000 frames=200340 ... recomputed id matches registered: True;
    verify_version files ok: True (85)`;
  - `AC4.8 shared HF_DATASETS_CACHE ... identical listing and bytes: True`.

  That run's summary JSON is `artifacts/s0_verify_derive_s4/s4_summary.json`. A second run, made only to
  capture stdout to a file, was cut off by the 600 s tool timeout (status in the next entry).
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
