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

## 2026-10-03: Reward evidence corrected; `duration_s` is demo length, not human time-to-success
- **Why:** the evaluation-layer plan found two premises false: P6 (the stored reward is not a goal check)
  and P8 (demos end after the first success). The owner approved correcting the wording on 2026-10-02
  (brief "Owner answers": "Yes, correct it"). The orchestrator edited CLAUDE.md and vla-build; this entry
  records the correction for this feature (evaluation-layer S3, AC3.8).
- **Grounding:** PoC §3.5 A pp. 7–8 ("Human reference = durations of LIBERO's original
  human-teleoperated demonstrations"), §9.1 Step 2.2 p. 18 ("using the benchmark's goal check").
- **What was wrong, and the evidence:**
  1. *"Every demo ends with its only `reward == 1` … so we keep them all"* (report, S1; catalog
     docstring) used the reward as proof of success. It isn't one. LIBERO's `scripts/create_dataset.py`
     (GitHub `8f1084e`, l. 228–231) sets `rewards = np.zeros(len(actions)); rewards[-1] = 1`, and the same
     for `dones`, unconditionally (quoted). The conclusion still holds on a real check: replaying each
     demo's stored states through LIBERO's own `_check_success` (the demo's XML, hf-libero 0.1.4) reaches
     the goal in **1000/1000** demos, with 0 replay errors, and `_check_success == AND(conjuncts)` at every
     state in 1000/1000 (measured, `scripts/human_reference.py`).
  2. *"`duration_s` = human time-to-success … valid because every demo succeeds at its last step"*
     (report S2 figure caption, `tracker/index.html` §3.5 A row) is false. The goal first holds before the
     end in every demo. Per task, `length_steps − T_success_steps` is min 9–11, median 11.0–19.5, max
     11–82 steps (measured). Collection stops only after the goal holds for 10 consecutive steps
     (`collect_demonstration.py`: `task_completion_hold_count = 10`, quoted, P7). In 12/1000 demos the
     goal no longer holds at the last stored state.
- **Effect on the HRT anchor (measured, KM of the 500 demos per suite):**
  - libero_spatial: median 5.30 s (T_success) vs 6.20 s (duration), RMST(60 s) 5.356 vs 6.225 s.
  - libero_10: median 12.45 vs 13.15 s, RMST(60 s) 13.166 vs 13.809 s.
  - An HRT anchored on `duration_s` would be **1.162×** too high on libero_spatial and **1.049×** on
    libero_10 (per task, 1.119–1.227 and 1.033–1.082).
- **Prior art:** LIBERO's own goal check and replay recipe (`create_dataset.py` l. 130–153), used as is.
- **Files:** `report.html` (dated corrections at the reward paragraph, the S2 figure caption and "Next";
  a "Changes since first release" line); `tracker/index.html` (§3.5 A row). The catalog's `duration_s`
  column is unchanged and keeps its meaning (length / 20 Hz). The new anchor is
  `data/eval/human_reference/human_reference.parquet` → `T_success_s` (evaluation layer).
- **Evidence:**
  - `scripts/human_reference.py --suites libero_spatial,libero_10` → `AC3.2 goal reached at some stored
    state: 1000/1000; replay errors: 0; _check_success == AND(conjuncts) at every state: 1000/1000`.
  - `--summary` → `T_success_s < duration_s: 1000/1000`; `an HRT anchored on duration_s is 1.162x too
    high` (libero_spatial), `1.049x` (libero_10).
- **Divergence:** none from the spec. This corrects this feature's own wording.

## 2026-10-08: `verify_runs_s5.py` AC5.3 checks the real eval stage's record (round-1 mode failed with `KeyError: 'suites'`)
- **Why:** orchestrator request, from the evaluation-layer refactor record (`tracker/features/2026-10-01-evaluation-layer/refactor.md`
  "N5"): `scripts/verify_runs_s5.py` in round-1 mode fails at AC5.3 with `KeyError: 'suites'`, because the block reads
  params of the retired `eval_standin`. "Update only the AC5.3 block … so it checks the same AC5.3 requirements against
  the real eval stage's recorded params and tags. Don't weaken the AC: every AC5.3 field must still be checked."
- **Cause (measured):** the round-1 `eval_standin` (`git show 4448ed2~1:vlapoc/runs.py`) logged params `suites`,
  `libero_task_ids`, `init_state_indices` and `seeds`, tags `eval.checkpoint.loaded` / `eval.checkpoint.n_params`, and
  a plan column `outcome`. `runs.eval_run` replaced it in evaluation-layer S5 (`4448ed2`, single record system) and
  writes none of those keys. The verifier still read `suites` at `4448ed2` and at the pre-refactor baseline `553b3d1`
  (`grep -F -c "suites="` = 1 at each), so the break dates to `4448ed2`, not to the refactor.
- **Grounding:** this feature's plan AC5.3 ("An eval-run record contains the sha256 of the checkpoint it loaded, the
  producing train run ID, suite, `libero_task_id`s, init-state indices and seeds"); PoC §2.1 item 1 p. 3 ("Every eval
  rollout should be tied back to that checkpoint") and item 3 p. 3 ("fixed seeds and initial states"). Both are cited
  in the block's comment.
- **Prior art:** none needed. This is a verifier fix: it reads MLflow params and tags through the existing `vlapoc.runs`
  client and reuses `vlapoc.rollouts.validate_record`.
- **Where each AC5.3 field is now checked** (read from `eval_run` and `rollouts.RECORD_COLUMNS`/`plan_rollouts`/`checkpoint_policy`):
  every field is compared in the run's params/tags **and** in the rollout record the run logged, against the eval
  request in the script.

  | AC5.3 field | `eval_run` params / tags | rollout record column |
  |---|---|---|
  | checkpoint sha256 it loaded | `checkpoint_sha256`; tag `input.checkpoint` | `checkpoint_sha256` |
  | producing train run | `train_run_id`; `policy` = `train:<id>` | `train_run_id` |
  | suite | `tasks` = `[[suite, libero_task_id], …]` | `suite` |
  | `libero_task_id`s | `tasks` | `libero_task_id` |
  | init-state indices | `n_init`, `init_offset`, `init_state_rule` | `init_state_index` |
  | seeds | `seed_base`, `init_state_rule` | `seed` |

  The checkpoint sha256 must also equal train-A's `checkpoint.weights.sha256` and a fresh sha256 of
  `<checkpoint.path>/model.safetensors`, the file `checkpoint_policy` hashes and loads. The record must be the run's
  own logged output (its sha256 equals the `output.rollouts.sha256` tag). The stand-in's `eval.checkpoint.loaded` /
  `n_params` tags were not AC5.3 fields. Their real-stage counterpart is checked instead: the rollout ran on the loaded
  policy (`policy_type` param and record = the checkpoint's `config.json` type), with `vlapoc.stand_in=false`.
  No AC5.3 field is missing from the real record.
- **Files:** `scripts/verify_runs_s5.py`, the AC5.3 block only (commit `91f0902`; the rest of the file is unchanged).
- **Evidence** (`HF_DATASETS_CACHE=$PWD/data/hf_datasets_cache LIBERO_CONFIG_PATH=$PWD/data/libero_config HF_HUB_OFFLINE=1
  MUJOCO_GL=cgl`; full stdout in `data/scratch/ac53_fix/`):
  - `.venv/bin/python scripts/verify_runs_s5.py` (round-1, scratch store `data/lineage_scratch/s5` built by
    `runs.scratch_lineage_copy`, HEAD `91f0902`) → `exit=0 seconds=76`:
    - `AC5.1 runs: 7, stages ['build', 'eval', 'select', 'split', 'train'] (all in (…): True)`; digest search →
      `['s5-select-random5', 's5-train-A', 's5-train-A-seed1']`.
    - `AC5.2 git_sha_is_HEAD: True`, `weights_sha256_matches_disk: True`, `normalizer_sha256_matches_disk: True`,
      `normalization_stats: per-subset: …`.
    - `AC5.3 checkpoint_sha256: {'param': 'cdee42fbf84f56ec', 'tag input.checkpoint': 'checkpoint:cdee42fbf84f56ec',
      'record': ['cdee42fbf84f56ec']}; … True`; `AC5.3 train_run_id: {'param': 'a9cc11041b4a', 'param policy':
      'train:a9cc11041b4a', 'record': ['a9cc11041b4a']}; … True`; suite, `libero_task_ids` (`[['libero_spatial', 0]]`),
      `init_state_indices` and `seeds` (`[['libero_spatial', 0, 0]]` from params and from the record) → `True` each.
    - `AC5.3 eval run: rollout record is its logged output (…): True; policy_type param/record == checkpoint config type
      'act': True; stand_in=false; rows=1 (eval.n_rollouts tag 1); events=[0] censored=[True]; record schema errors: []`.
    - `AC5.3 every field recorded (params/tags and rollout record) and equal to the request: True`.
    - `AC5.4 trace(s5-train-B) … raw files reached: 20`; `consumers` lists train-A `True`, train-B `False`;
      `diff(A, B): removed 5, added 0, changed_version 0`; `diff(A, A-seed1)` → `params {'seed': ('0', '1')}`.
    - `AC5.5` v1-keyed table `'ok': True`; v0-keyed → `rejected: 6/10 score rows are not episode versions trained on …`.
    - `AC5.6 tracking uri scheme=sqlite; HF_HUB_OFFLINE=1; wandb CLI installed=False`.
  - Negative control: `data/scratch/ac53_fix/ac53_block_check.py` is a read-only harness that executes the committed
    AC5.3 block against this run's store. With train-A it prints the same lines as above, ending `… equal to the
    request: True`. With `--wrong-train` (train-A-seed1 passed as train-A) it prints `AC5.3 checkpoint_sha256: … False`,
    `AC5.3 train_run_id: … False`, `… equal to the request: False`, while suite, task ids, init states and seeds stay
    `True`. The check is not vacuous (outputs `ac53_block_check.txt`, `ac53_block_check_wrong_train.txt`).
  - `.venv/bin/python scripts/verify_runs_s5.py --r2` → `exit=0`: AC5.8 lowest/highest `True`, child == S6 child
    `True`; AC5.9 `child n=912`, `raw files reached: 20`; AC5.10 `distinct subset ids: 3/3`, all reproducible;
    `real mlflow.db sha256 unchanged: True (7845d8d7634e); real mlartifacts tree unchanged: True (203 paths); scratch
    removed: True`.
  - `shasum -a 256 data/lineage/mlflow.db` before and after both runs → `7845d8d7634e2757…` both times;
    `find data/lineage -newer <start marker>` → nothing; no verifier processes left.
- **Left as committed (outside this change's scope, for the owner or the summarizer):**
  - Both modes rewrite this feature's `artifacts/s5_summary.json`, `s5_diff_A_B.json` and `s5_r2_summary.json`. The
    regenerated files are kept in `data/scratch/ac53_fix/data_lineage_artifacts/` and the committed ones were restored.
    `s5_r2_membership_matrix.csv` regenerated byte-identical. The committed `s5_summary.json` AC5.3 entry is still the
    round-1 stand-in evidence (`"plan_rows": 1000, "outcomes": ["not_run"]`).
  - `S5-mlflow-run-lineage.html` (AC5.3 row: "loaded=true (51600263 params); stand_in=true"; deliverables list
    `eval_standin`) still describes round 1. It has been out of date since `4448ed2`, not because of this change.
  - The refactorer's N5 store was moved aside to `data/lineage_scratch/s5.refactor-n5-diagnostic` (2.3 GB, nothing
    deleted). `data/lineage_scratch/s5` (2.3 GB) is now this run's store.
- **Divergence:** none. AC5.3 is checked as written; the fields are read from where the real eval stage records them.

## 2026-10-11: S5 page AC5.3 row and `s5_summary.json` AC5.3 refreshed to the real eval stage's evidence
- **Why:** the 2026-10-08 entry left these two records describing round 1's `eval_standin` ("loaded=true (51600263 params);
  stand_in=true"; `"plan_rows": 1000, "outcomes": ["not_run"]`). The evaluation-layer summarizer was asked to refresh them.
- **Grounding:** plan AC5.3; PoC §2.1 items 1 and 3, p. 3 (as in the 2026-10-08 entry).
- **Prior art:** none needed (record update).
- **Files:**
  - `S5-mlflow-run-lineage.html`: the AC5.3 row keeps the round-1 evidence, marked as such, and adds the 2026-10-08 re-run
    lines (checkpoint sha, train run, suite, task ids, init states, seeds all `True`; `stand_in=false`; negative control
    `False`). The deliverables row notes that `eval_standin` was replaced by `eval_run`.
  - `artifacts/s5_summary.json`: the `AC5.3` entry is replaced with the regenerated one from
    `data/scratch/ac53_fix/data_lineage_artifacts/s5_summary.json`, and an `AC5.3_provenance` key records where it came from.
    The other entries and `run_ids` are the original 2026-09-29 run's.
  - `report.html`: a dated line in "Changes since first release", and a dated note on the "stand-ins only" known issue
    (evaluation is now real).
- **Evidence:** the 2026-10-08 entry's re-run (`verify_runs_s5.py`, exit 0, HEAD `91f0902`); nothing re-run for this record update.
- **Divergence:** none.

## 2026-10-17: AC4.8 "contrast" figure corrected (60 files / 663 MB no longer reproduces; now 0 files)
- **Why:** the ground-truth feature's analyzer (analysis-r1/r2, plan R12) found the data-lineage record still quoting the old
  un-isolated-cache figure. Request: add the changelog entry and correct `report.html`.
- **Grounding:** none needed (record correction). Context: analysis-r1 E3 and AC4.8 (r2) of this feature: a derivation leaves the
  shared HF datasets cache unchanged.
- **Prior art:** none (our own measurement).
- **Finding (measured):** `scripts/verify_derive_s4.py --cache` (20-episode parent, one edit with `isolate_hf_cache=False`, cache
  pointed at a scratch dir) used to leave `60 files / 12 .arrow / 662,758,780 bytes` (2026-09-29, round-1 writer:
  `delete_episodes` + `merge_datasets`). On 2026-10-15 and 2026-10-17 it leaves `0 files / 0 bytes`, on the current code and on the
  code just before the ground-truth change, so the cause is the order-preserving one-pass LeRobot writer of 2026-10-02
  (evaluation-layer S0), not a change of ours today. The isolated path is unchanged: the shared cache listing is identical
  before and after (134 files / 28,952,405,234 bytes in the 2026-10-15 run).
- **Files:**
  - `report.html`: the round-2 analyzer-table sentence now says the 60 files / 663 MB were the round-1 writer's behaviour and are
    corrected (0 files); a dated line in "Changes since first release".
  - `S4-derived-versions-labels.html`: the AC4.8 row marks 60 files as the 2026-09-29 round-1 measurement and adds the re-measured 0.
  - `artifacts/s4_summary.json`: an `AC4.8_note_2026-10-17` key; the old `contrast_unisolated_*` values stay as the historical record.
- **Evidence:** `.venv/bin/python scripts/verify_derive_s4.py --cache --artifacts <scratch>` → `AC4.8 contrast, isolate_hf_cache=False:
  0 cache files (0 .arrow, 0 parquet builder entries), 0 bytes left behind by one 20-episode derivation`
  (`tracker/features/2026-10-15-ground-truth/artifacts/s2_regress_verify_derive_s4.txt`).
- **Divergence:** none; the isolation guarantee (AC4.8) is met as before. The `isolate_hf_cache` code path itself is kept.
