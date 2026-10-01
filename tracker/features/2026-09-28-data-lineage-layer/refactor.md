# Refactor: data layer + lineage layer

Refactorer, 2026-10-01, after `analysis-r2.md` (PASS). Source commit **`a8ee9e3`** (parent `97fb70b`).
Evidence labels as in CLAUDE.md: **measured** (a command was run; harness and outputs were kept in the
session scratchpad), **quoted** (from a source, with the quote), **inferred** (reasoning).

Scope: the 13 files in the execution log, `vlapoc/` (7) and `scripts/` (6). No other code was touched.

## What changed

| File | Change | Why it helps a reader |
|---|---|---|
| `vlapoc/__init__.py` | Package docstring lists the six modules in data-flow order, each mapped to its story (S1–S6) and its box in `docs/dataflow.md`. | One place to see the whole package before opening any file. |
| `vlapoc/libero_convert.py` | Docstrings only. Added "Where it sits" and runnable usage (full run and a 20-episode smoke run). Corrected the `RawEpisodeRef.source_file` docstring, which said "relative to the raw root's parent's parent", but `_rel` returns a repo-relative path, or an absolute one outside the repo. Documented that `libero_task_id` is suite-relative, and that `iter_raw_episodes` order *is* the dataset's `episode_index` order. | The one thing the catalog silently relies on (row order = episode order) is now stated where the order is produced. |
| `vlapoc/catalog.py` | `_decode_png` → public `decode_png`, now shared with `derive.py`. Docstrings on every public function, with input shapes and units (`action` (T, 7), `state` (T, 8), images (T, 128, 128, 3) uint8). Usage with `--out data/scratch/...` so a rebuild can avoid overwriting the stored catalog. `read_tasks` explains the two pandas views of `tasks.parquet`, checked on v0 (measured). | Makes visible that a single decoder feeds every stored-side hash. A newcomer can see what goes into `content_hash` without reading the loop. |
| `vlapoc/lineage.py` | **CLI:** one `_cmd_<command>` handler per subcommand, attached with argparse's `set_defaults(func=...)`, which is the standard library's documented subcommand idiom. It replaces a 14-branch `if/elif` chain with compound `a.cmd == … and a.vcmd == …` tests and a `version diff` parser defined out of place. The two copies of the "first n + '… k more'" truncation are now one `_head()`. **`split make --track` now calls `make_splits` once.** **Resolvers:** `resolve_version` and `Subset.load` share one `_resolve_ref()`, with the same name / full id / unique ≥8-char prefix rules. **`drop_group` docstring:** use `task_name` for leave-one-task-out, because `libero_task_id` is suite-relative. The lazy `CATALOG_COLUMNS` import moved to the top. Module docstring: the full CLI usage (all 14 commands), what a `ref` may be, and "Where it sits". | Every command is a named 2–8 line function you can jump to. The module docstring is a CLI manual. The resolver rule lives in one place. |
| `vlapoc/derive.py` | **`derive_version` split** into three parts: the guards and top-level flow; `_write_derived_dataset` (the LeRobot `delete_episodes` → writer → `merge_datasets` mechanism inside the isolated cache); and `_derived_episode_table` (identity re-established from the stored bytes). The registration tail stays inline. The function went from ~160 lines to ~115, docstring included, and reads as four labelled steps. `read_stored_episode` uses `catalog.decode_png` and `catalog.IMAGE_KEYS` instead of local copies. Removed the unused `load_names` import and `__all__` entry. Docstrings: the `EpisodeEdit.transform` contract, the `clip: None` effect, and the fact that `idle_padding_edit(at=…)` is only recorded (always inserts at the middle). Usage and "Where it sits". | S4 is the hardest file. Its main function now shows the steps by name, and the decoder shared with the catalog makes "expected hash == stored hash" obviously like-for-like. |
| `vlapoc/runs.py` | Inlined the identity function `_local_source`. Its reason, that MLflow's `LocalArtifactDatasetSource` can't be imported by name, is now a comment at the call site, and the warning filter at the top says which MLflow warning it silences (checked in `mlflow/data/dataset_source_registry.py:77`). Removed the dead `_sha256_bytes`. Hoisted the redundant lazy imports of already-imported modules (`build_catalog`, `raw_file_hashes`, `make_splits`, `recompute_version_id`). `split_run` returns `(run_id, record)`, like `derive_run` and `select_run`. Docstrings on every public function. "Where it sits" maps each `stage` tag to a dataflow.md step. | The stage ↔ diagram mapping is spelled out. Every function says what it returns, e.g. the shape of `trace` and what an empty `produced_by` edge means. |
| `vlapoc/lerobot_train_entry.py` | Docstrings only. The spec keys `build_config` reads, "Where it sits", and a usage line runnable from the repo root. | The spec ↔ `train_run` contract is in one place. |
| `scripts/verify_runs_s5.py`, `scripts/lineage_dry_run.py` | The two `split_run` call sites are adapted to its tuple return (`sp, _ = …`, `…[0]`). Usage lines now state cost and side effects: full chain vs `--tamper-only`, round-1 vs `--r2`, and the clean-tree requirement. `log()` moved above its first use. Docstrings on `reuse_or`, `mermaid` and the `main`s. | A reader knows which invocations write real runs before running one. |
| `scripts/verify_libero_convert.py`, `orientation_figure.py`, `verify_lineage_s3.py`, `verify_derive_s4.py` | Docstrings and usage only. Each flag's cost is stated (e.g. `--padding` = a 7 GB full-scale copy), and so is which runs rewrite files in `artifacts/`. | Same as above. |

Code size: 2,984 → 3,001 non-blank, non-comment, non-docstring lines across the 13 files (measured).
The CLI is **longer**: 119 → 140 logical lines, because 14 handler `def`s, 14 `set_defaults` calls and
the `_head` helper replace 14 `elif` branches (measured). Its 14 commands can't be removed (see "Left as-is"), so the simplification is
structural.

## Behaviour check

How it was proven:

- **AST equivalence.** Every file's HEAD AST was compared with its refactored AST, docstrings stripped.
- **AST-transform proofs.** For the mechanical edits, the intended transformation was applied to the
  *old* function's AST and the result compared with the *new* one.
- **Runtime replays.** Everything was run before (`97fb70b`) and after (`0c72136`, whose code is
  AST-identical to `a8ee9e3`), with `HF_DATASETS_CACHE=$PWD/data/hf_datasets_cache`. All scratch
  stores were under `data/scratch/` and deleted. Any script that rewrites files in `artifacts/` had
  its output saved aside, and the committed artifacts were restored from git.

| Command | Before == After? | Note |
|---|---|---|
| AST equivalence, 13 files | 7/13 code-identical | `__init__`, `libero_convert`, `lerobot_train_entry`, `verify_libero_convert`, `orientation_figure`, `verify_lineage_s3` and `verify_derive_s4` changed only in docstrings and comments (measured). The other 6 change exactly the functions listed above. `train_run`, `derive_run`, the stand-ins, `load_checkpoint`, `select_run`, `trace`, `consumers`, `diff` and `check_scores` are AST-identical. |
| AST-transform proofs (9) | Yes, 9/9 | `input_subset`/`input_version` = old with `_local_source(x)` → `x`. `build_run`, `split_run` and `drop_group` = old minus their lazy imports (plus the tuple return for `split_run`). `_hash_data_file` = old with the rename. `decode_png` = old `_decode_png`. `read_stored_episode` = old with its nested `dec()` replaced by `decode_png`, whose body is identical to `dec()`'s (measured). |
| `python -m vlapoc.lineage …` read-only replay against the real store: 23 commands (`version show/verify/recompute/diff [--ids]` on v0/v1, prefix refs, `names`, `subset show [--ids]`, `trace` ×2, `consumers` ×2, `diff` ×4, `check-scores`), 6 error cases, and `--help` of all 19 parsers | **Yes, 54/54 files byte-identical** | stdout and exit codes. Error cases are compared on the last stderr line. stderr otherwise differs only in MLflow's tqdm download rate (`it/s`). Real `mlflow.db` sha256 `c5faec81…` before and after. |
| Same replay vs the stored S6 artifacts | Yes | `diff` (top, seed, rand), `trace(train-drop-top-20)` and `consumers(probe@v1)` equal `s6_diff_full_s0_vs_*.json`, `s6_trace_train_drop_top.json` and `s6_consumers_probe.json` (measured). |
| Writing commands in a scratch copy of the store (manifests + `mlflow.db`, artifact location repointed): `split make`, `split make --track`, `build verify v0`, `version register` | Yes | Identical once the 2 new MLflow run ids and `registered_utc` are normalised. The store afterwards is identical: names, subset files, splits, version records, and the new runs' params, tags, dataset inputs (source, digest, tags) and artifact lists. `--track` prints the same record from one `make_splits` call. |
| `derive` / `derive --no-track` CLI wiring (old and new `main` with stubbed `derive_run` / `derive_demo_action_noise`) | Yes | Same arguments forwarded, same stdout. The full-scale derivation itself was not run. |
| Small-scale end-to-end driver, 60 episodes (3 per task) in `data/scratch/`. Steps: `convert`, `hash_stored_dataset`, `register_version`, `make_splits` (1 ref/task), recompute and verify, prefix and error resolution, the reference guard plus 3 other refusals, `derive_demo_action_noise` (20 edits), a chained 20-frame padding derivation on the child, `read_stored_episode` → hash, and every selection op with its 5 refusals | **Yes**: 16/16 output files byte-identical old vs new with `PYTHONHASHSEED=0` | This replaces the full-scale derivation, which was not re-run. Without a fixed seed, the derived datasets' `meta/tasks.parquet` and `meta/stats.json` (`task_index` stats only) vary **between two runs of the unchanged code too**: the control pair of HEAD runs differed in `tasks.parquet` size (measured). Cause: LeRobot `dataset_tools.py:549` `dst_meta.save_episode_tasks(list(set(tasks)))` (quoted source). The set order depends on Python's per-process string hashing, so `task_index` assignment changes run to run. Fixing the hash seed removes all differences (measured). Content hashes, version ids, labels and `derivation.json` are identical either way. |
| `scripts/verify_lineage_s3.py --summary data/scratch/…` | Yes | Only the random scratch-dir name differs; the summary JSON is identical. |
| `scripts/verify_derive_s4.py --guard --columns --cache --v1` | Yes | Only timings, frames/s and the scratch-dir name differ. `s4_summary.json` and `s4_labels_head10.csv` are identical. Derivation id `6cbeb214cd4c`, the unisolated contrast (60 files / 662,758,780 B), and v1 recompute/verify/labels/noise statistics are all unchanged. |
| `scripts/verify_runs_s5.py --r2` | Yes | Only new MLflow run ids and the scratch-dir name differ. Subset ids `c598f5692b40`, `ab03e001…`, `5eafbb8e…` and `d03a5cf5…`, the membership matrix and the summary are identical, as are the real DB and `mlartifacts` (63 paths). |
| `scripts/lineage_dry_run.py --tamper-only` | Yes | Only the scratch-dir name differs. Refused; 15/15 runs. |
| `scripts/orientation_figure.py` | Yes | PNG and stats byte-identical, both to each other and to the committed artifacts. |
| `scripts/verify_libero_convert.py --summary data/scratch/…` | Yes | stdout and summary identical. |
| `python -m vlapoc.catalog --version v0 --verify-raw --out data/scratch/…` | Yes | Only the build seconds differ. 1000/1000 hashes match raw, and the rebuilt tables equal the stored catalog. |
| Real `data/lineage/mlflow.db` sha256 | Unchanged | `c5faec81f235ebf866b05980f3d72c069e1601c55b7874e3b055233e4b0f2ada` at start, after every batch, and at the end. |
| Shared HF cache `data/hf_datasets_cache` | Unchanged | 124 files / 28,643,158,514 B before and after every script run. |
| `grep -rnE '/private/tmp\|/tmp/claude\|/Users/' vlapoc scripts` (AC3.9) | Still empty | exit 1. |

**Not re-run, per the disk and lineage constraints:**
- The S1 conversion: `libero_convert.py` is AST-identical, and the 60-episode driver runs `convert()`.
- A full-scale derivation (`lineage derive`, `verify_derive_s4.py --padding`): covered by the 60-episode
  driver and the `--cache` 20-episode derivations.
- Training (`verify_runs_s5.py` round-1 mode, the full `lineage_dry_run.py` chain): `train_run`, the
  stand-ins and `load_checkpoint` are AST-identical. `input_subset` is proven by AST transform and also
  ran in `--r2` and `split make --track`. In those two scripts, the only code change is the one-token
  `split_run` call-site edit.

**Known measurement-only difference:** the LeRobot imports moved from the top of `derive_version` into
`_write_derived_dataset`, after the clock starts. In a fresh process the recorded `derive.timing.delete`
therefore now includes the one-time import: 1.3–1.9 s (measured), against 166–181 s for the full-scale
delete phase. It's documented in the helper's docstring. In exchange, a refused derivation (guard
error) no longer imports LeRobot at all.

## Left as-is (and why)

**Behaviour changes, out of scope here; the orchestrator lists them as follow-ups:**
- Validating `drop_group`'s `column`/`value` before `stage_run`. Today a refused `drop_group` inside
  `select_run` leaves an inert FAILED run.
- The `select.dropped` tag, which is truncated at 7,900 characters. Now documented in `select_run`'s
  docstring.
- A digest of the episode table in `version.json`.
- Clipping of noised actions. The `clip: None` effect is now documented in `robodrop_action_noise`.

**Kept deliberately:**
- **Prefix matching in the resolvers.** I deduplicated it but didn't remove it. Removing it would change
  the result for any prefix ref: the replay shows `version show 16076b9c`, `subset show 8ee096b7` and
  `version diff 3c4dca68 16076b9c` all resolve today (measured). No logged or scripted command uses a
  prefix (grep), so dropping it is safe to *decide*, but that decision is the owner's.
- **All 14 CLI commands.** Each appears in the execution log, the plan's verify-by lines, or an artifact
  (`s3_*_stdout.txt`, `s4_build_verify_v0_stdout.txt`, `s4_split_track_stdout.txt`). The orchestrator's
  rule was identical output for every one of them.
- **`isolate_hf_cache=False`.** The analyzer called it a test hook, but `verify_derive_s4.py --cache`
  needs it for the AC4.8 contrast measurement.
- **Cross-module private helpers** (`_root`, `_bind_name`, `_write_json`, `_demo_number` imported into
  `derive`/`runs`). Making them public touches many lines for no reader gain beyond the name.
- **The long, AC-sectioned `main()`s in the verification scripts** (`lineage_dry_run.main` 160 lines).
  The full dry run can't be re-run cheaply to prove a restructure.
- **`orientation_figure.stored_frame`, which duplicates part of `read_stored_episode`.** It's a
  standalone figure script.

**Not fixed, because the fix changes exit status:**
- `verify_libero_convert.py` crashes on its final print when `--summary` is relative or outside the
  repo.
- `catalog.py` crashes when `--out` is outside the repo.

Both write their files first, and both are now documented in the usage. The fix is the one
`verify_lineage_s3.py` already uses: print the path as given when `relative_to(REPO_ROOT)` fails.

**Observations for the owner, found while documenting, not changed:**
- **`register_version` on already-registered content rewrites that version's `version.json`.** After
  registering v0's content as `v0copy` in a scratch store, `resolve_version("v0")` returned a record
  named `v0copy` (measured). Documented in the docstring.
- **Derived datasets' `task_index` order isn't reproducible across processes** (LeRobot
  `dataset_tools.py:549`, above). Our ids are unaffected, because the content hash excludes `task_index`.
  `files.json` digests and `meta/stats.json` `task_index` stats of a re-derivation will differ.
- **`python -m vlapoc.lineage` runs the module as `__main__`, and `vlapoc.runs` imports a second copy**
  as `vlapoc.lineage`. That's why an error raised in `runs` prints as `vlapoc.lineage.LineageError`
  and one raised in `lineage` as `LineageError` (measured). It's harmless today and noted in the CLI
  comment.
- **Untouched files:** `tracker/features/…/analysis-r2.md` (untracked, saved by the orchestrator) and
  `.claude/skills/feature-pipeline/SKILL.md` were not committed by me. `data/lineage_scratch/` and
  `data/scratch/r2_strip_label_column.py` are still on disk (analyzer item 5, the owner's call).

## Reading guide

Read in this order. Each step names the functions worth reading closely; the rest of each file is
tables or plumbing. Commands marked *read-only* are safe to run while reading.

**Context (10 min)**
- **`docs/dataflow.md`**, the "Full flow" diagram. Every module below implements one of its boxes:
  the LIBERO demos, the clean reference set, "corrupt 5–10% of a copy", the training dataset, and
  Steps 1–4 with the implicit training run.
- **`vlapoc/__init__.py`**: the module map, in data-flow order.

**Data layer (S1, S2)**
- **`vlapoc/libero_convert.py`**: raw LIBERO HDF5 → one LeRobot v3 dataset.
  - Read the module docstring (the 4 deliberate divergences from the hub datasets).
  - `load_raw_episode` is *the* raw → canonical mapping: rotate 180°, cast to float32, state =
    `ee_pos ‖ ee_ori ‖ gripper`.
  - `convert` is LeRobot's own writer loop.
- **`vlapoc/catalog.py`**: episode identity and the §2.2 catalog.
  - The module docstring gives the exact `content_hash` byte layout and the columns LIBERO can't supply.
  - `episode_content_hash`: the only hash function in the system.
  - `hash_stored_dataset`: the stored side.
  - `scan_raw` / `_raw_file_worker`: the raw side.
  - `build_catalog`: joins the two and runs the 1000/1000 checks.

**Lineage layer (S3)**
- **`vlapoc/lineage.py`**, read top to bottom:
  1. `version_id`, then `register_version`: a version is identified by its content.
  2. `Subset`: a selection of episode versions; `subset_id` includes how the selection was made, and
     `resolve()` gives LeRobot's `episodes=` list.
  3. `make_splits`: the held-out clean reference set (2 per task) and the candidate set.
  4. The selection ops `drop_top_k`, `drop_random_k` (also the datamodels random subsets) and
     `drop_group` (leave-one-group-out; use `task_name`).
  5. `verify_version` (tamper check) and `isolated_hf_datasets_cache` (why loads cost disk; plan R1).
  6. The CLI at the bottom: `_build_parser` lists every command; each `_cmd_*` is a few lines.
  - Try (*read-only*): `.venv/bin/python -m vlapoc.lineage version show v0`, `… subset show v0/reference`,
    `… names`.

**Ground-truth injection (S4)**
- **`vlapoc/derive.py`**:
  1. `EpisodeEdit`: what an edit must return.
  2. `derive_version`, read as four steps: guards (`protected_episode_ids`: never edit a reference
     episode) → `_write_derived_dataset` (LeRobot `delete_episodes` → writer → `merge_datasets`) →
     `_derived_episode_table` (every kept episode must still hash to its parent hash) → register +
     `labels.parquet` + inherited splits.
  3. `robodrop_action_noise` and `derive_demo_action_noise`: how v1 was made.
  - Try (*read-only*): `… version diff v0 v1` → 100 changed, 900 identical.

**Run lineage (S5)**
- **`vlapoc/runs.py`**:
  1. Module docstring, "How lineage is encoded in MLflow", then "Where it sits" (stage tag ↔ diagram step).
  2. `stage_run` + `StageRun`: inputs as MLflow dataset inputs, outputs as `produced.<id>` tags, and the
     clean-git-tree rule.
  3. One simple stage, `select_run`, then `train_run`.
  4. The queries: `producers` → `trace` → `consumers` → `diff` → `check_scores`.
  - Try (*read-only*):
    - `… trace 9339d4685a8e4d009c191fa4641ccf3a`: from `train-drop-top-20` back to the 20 raw files.
    - `… diff ff64512b40974587a9d711cd8e4f8b04 9339d4685a8e4d009c191fa4641ccf3a`: exactly 20 episodes
      removed, nothing else.
- **`vlapoc/lerobot_train_entry.py`**: why LeRobot's CLI is bypassed on Python 3.14 (draccus #69), and
  the spec `train_run` hands to `build_config`.

**End to end (S6)**
- **`scripts/lineage_dry_run.py`**: read `main()` top to bottom. It is every stage call in order
  (build → split → derive → screen → train ×2 → eval → attribute → select ×2 → train ×2 → eval ×2),
  followed by the AC6.x checks. `graph` and `mermaid` render `artifacts/s6_lineage_graph.mmd`.

**Worked examples (optional)**
- `scripts/verify_lineage_s3.py` (the S3 API in a scratch store).
- `scripts/verify_runs_s5.py --r2` (the three `select_run` ops).
- `scripts/verify_derive_s4.py`.
- `scripts/verify_libero_convert.py` and `scripts/orientation_figure.py` (the S1 evidence).
