# Execution log: ground truth (injection experiment)

## Round 1: 2026-10-15 / Batch 1 (S1-S3)

### S1: Implement the seven corruption transforms with invariant checks on real episodes
**Files:**
- `vlapoc/corruptions.py` (new, ~520 lines): `action_noise`, `time_shift`, `instruction_swap`, `dropped_frames`, `detour`, `idle_padding`, `gripper_release` constructors, each returning a `derive.EpisodeEdit`; `eligible(type, episode)`, `clip_actions`, `final_release`, `constant_gripper_windows`, `hypothesis_for`, `make_edit`. Grounding docstring with PoC §2.5 p.5, §8.10 p.17, §4.5 p.11, RoboDrop App. A.1 p.13.
- `vlapoc/derive.py` (+2 dataclass fields on `EpisodeEdit`, defaults make it backward compatible): `new_task: str | None = None`, `hypothesis: str = "none stated"`. Declared here because the S1 constructors set them; `derive_version` starts using them in S2.
- `scripts/verify_corruptions_s1.py` (new): in-memory invariant runner, eligibility over the 960 candidates, hash dump/check, figures.

**Premises checked:**
- P1/P2 (clean range ±0.9375 on dims 0-2, gripper exactly {-1,+1}) → `meta/stats.json` action min/max = ±0.9375 on dims 0-2, gripper ±1.0 (printed); gripper binarity re-asserted on every output → holds.
- P7 (release geometry) → recomputed on the 960 candidates: final release in 907/960; carry span median 63, ≥40 in 95.7%, ≥30 in 97.5% → holds (plan's 944/1000 was over all 1000).
- Catalog `language_instruction` strings equal `tasks.parquet` strings (20 = 20, asserted in the script) → holds.
- RoboDrop App. A.1 quote ("zero-mean Gaussian noise with standard deviation 0.25σa", "shifts the action sequence by two seconds") → read from PDF p.13 via pypdf this session → holds, and it states no direction, tail fill, gripper handling or clipping.

**Acceptance criteria:**
- ✅ AC1.1: seven names as in the plan. Every label's `params` has `source` (spec quote and page), `parameters` (each `{value, basis, from}` with basis in quoted/measured/inferred), `corrupted_frames` ([start, end) child ranges), plus flat clip counts. Checked by `check_params` on all 118 × 7 outputs, 0 failures. Evidence: `.venv/bin/python scripts/verify_corruptions_s1.py --version v0 --per-type 100 --seed 0 ...` → `AC1.1 example label params` block (artifacts/s1_run1.txt), e.g. `detour ... corrupted_frames=[[51, 112]]`, `idle_padding ... [[51, 71], [86, 106]]`.
- ✅ AC1.2: 118 candidate episodes (eligible for all seven types, seeded) per type, all outputs in [-1, 1] with gripper in {-1,+1} → `invariant_failures` = 0 for all seven rows (artifacts/s1_run1.txt, s1_invariants.json).
- ✅ AC1.3: `action_noise` totals over 118 episodes: n_values 167,846, n_clipped 521 (frac_clipped 0.31% of all values), dims 0-2: 521 clipped = 0.72% of dims-0-2 values (inside the expected 0.5-1.5%), `n_beyond_clean_range` 1,215 = 1.69% of dims-0-2 values lie in (0.9375, 1]. The other six types record `n_clipped = 0`, `frac_beyond_clean_range = 0`, asserted per episode → 0 failures. Note: `frac_clipped_dims0_2` is the recorded fraction over the three translation dims; all clipped values are on dims 0-2 (n_clipped_dims0_2 == n_clipped).
- ✅ AC1.4: lengths (T for five types, T+40 for `detour` and `idle_padding`), image identity (every output image of detour/idle_padding/dropped_frames is byte-identical to a parent image: sha256 frame digests), no consecutive identical frame pair in `detour`, swap targets from the parent's tasks, same suite and ≠ original, and only declared arrays change (dropped_frames leaves action and state bit-identical; noise changes only dims 0-5 of action; swap changes only the task string; gripper_release changes only gripper frames inside its declared range) → 0 failures on 118 episodes per type.
- ❌ AC1.5 (second clause only): same (seed, episode_id) gives the same hash in two fresh processes for all seven types (✅, `AC1.5 same (seed, episode_id) ... {all True}` in artifacts/s1_run2_check_hashes.txt). A different seed (0 vs 1) changes the hash for: action_noise 118/118, dropped_frames 118/118, idle_padding 118/118, detour 112/118, instruction_swap 105/118, time_shift 0/118, gripper_release 0/118. Reasons: `time_shift` and `gripper_release` are deterministic by their parameters (40-step shift, 20-step release; no random draw exists), so no seed can change them; `detour` and `instruction_swap` draw from a finite set (window start; 9 alternative tasks), so chance collisions occur (13/118 ≈ the 1/9 expectation for swap). The plan's wording assumed a seed dependence for every type but swap. Fix if wanted: amend the AC to "same-seed determinism for all; seed-sensitivity for the five stochastic types", or give the two deterministic types a seeded parameter (not done: parameters are quoted/inferred as fixed in the plan).
- ✅ AC1.6: eligibility over 960 candidates (all in 20 tasks, 2 suites): action_noise, time_shift, instruction_swap, dropped_frames, detour, idle_padding 960 each; gripper_release 884. Each type has ≥13 eligible in 20 tasks. Gripper_release 884 matches the P7 re-measurement (907 with a final release, 97.5% of those with carry ≥ 30 → expected ≈ 884). Evidence: `AC1.6 eligibility` table in artifacts/s1_run1.txt.
- ✅ AC1.7: module docstring cites PoC §2.5 p.5, §8.10 p.17, §4.5 p.11, RoboDrop App. A.1 p.13 for the two types it defines, lists each inferred parameter, and states the dims-0-5 divergence from v1.

**Demo artifacts produced:** `artifacts/s1_invariants.json` (all numbers), `artifacts/s1_run1.txt` (printed table, run 1 with the hash dump), `artifacts/s1_run2_check_hashes.txt` (second process), `artifacts/s1_traces.png` (seven panels), `artifacts/s1_filmstrip.png` (dropped_frames, detour, idle_padding, corrupted vs clean).

**Notes / divergences from plan or spec:**
- Detour window: 21 frames [t, t+20] inclusive; child order 0..t+19, t+20 down to t, t+1..t+20, t+21..; a naive "forward, backward t+19..t, forward" repeats the turn frame and creates consecutive duplicates, so the turn frames are not repeated. `corrupted_frames` = [t, t+61).
- RNG key: `derive._episode_rng(seed, "<type>:<episode_id>")` (salted with the type so streams differ per type); the plan said "the existing generator", which it still is.
- `EpisodeEdit.params` is completed by the transform at run time (corrupted frames, clip counts, draws depend on the episode). `derive_version` serialises it after its writer pass, so this works unchanged; a caller must run the transform before reading those keys.
- `frac_clipped` is defined over all T×7 values; `frac_clipped_dims0_2` over the 3T translation values (that is the number comparable to the P3 estimate).
- The sample for the in-memory test is 118 candidates eligible for all seven types (a seeded draw), covering 19 of 20 tasks.

### S2: Extend `derive_version` for instruction edits and count the instruction as content in the blind check
**Files:**
- `vlapoc/derive.py`: `EpisodeEdit.new_task` is honoured (guard: target must be in the parent's `tasks.parquet`, else `LineageError` before any write; a transform may change the task only to `new_task`); the writer pre-registers the parent's tasks in parent order (`ds.meta.save_episode_tasks`) before the first episode; `_derived_episode_table` refreshes `language_instruction` from the stored task (and requires every kept episode's stored task to equal the parent's); `labels.parquet` gains the `hypothesis` column. Docstrings of `EpisodeEdit`, `derive_version`, `_derived_episode_table`, `load_labels` updated.
- `vlapoc/lineage.py`: `language_instruction` moved into `DATA_DERIVED_COLUMNS`; `tasks` removed from `META_BOOKKEEPING_COLUMNS`; new consistency rule in `_compare_meta`: `meta/episodes.tasks` must equal the child table's `language_instruction` per episode (else `META_MISMATCH`), so the instruction is not a free channel. `compare_blind` docstring and the section comment state the reclassification.
- `scripts/verify_derive_gt_s2.py` (new).

**Premises checked:**
- R1 (pre-registering tasks on LeRobot's writer metadata object works) → AC2.3 on stored bytes: `tasks.parquet` identical to the parent's → holds.
- P10 (hash includes the instruction) → read `vlapoc/catalog.py:127-148` → holds.
- Scratch parent's stored content hashes equal v0's for the same `episode_id` (20/20) → holds, so the scratch parent is a faithful miniature.
- P12 (hash-seed quirk) → measured in AC2.4: the task order is NOT affected on the current writer path.

**Acceptance criteria** (`.venv/bin/python scripts/verify_derive_gt_s2.py`, output `artifacts/s2_run.txt`, numbers `artifacts/s2_verify.json`; 20-episode scratch parent with v0's full catalog columns):
- ✅ AC2.1: one edit of each of the seven types registers (`3f43df489885`, 13.4 s) with `check_blind = []` and `assert_blind` passing. `labels.parquet` has 7 rows and columns `['episode_id','parent_content_hash','content_hash','corruption_type','params','seed','hypothesis']`; hypothesis is set only for `idle_padding` (speed) and `gripper_release` (reliability), `none stated` otherwise.
- ✅ AC2.2: a swap to a string not in the parent's tasks → `LineageError: instruction edit(s) to task string(s) not in small20's tasks.parquet ...`; out_root absent, no temp dir, versions in store 2 -> 2. A swap of reference episode `.../demo_14` on real v0 (scratch lineage copy) → `LineageError: refusing to edit 1 reference-split episode(s) ...`, nothing written.
- ✅ AC2.3: child `tasks.parquet` equals the parent's (raw pandas frames equal, 20 tasks, same order). The swapped episode's 137 data rows carry task_index 7 (the target's parent index; original 5). Only that one episode's task_index differs from the parent's (`[5]`); its meta `tasks` is the target string.
- ✅ AC2.4: `PYTHONHASHSEED=0` vs `1`, one subprocess each: identical `tasks_map`, `tasks.parquet` sha256, `labels_rows_digest`, labels.parquet bytes, all content hashes and `dataset_version_id` (`3f43df4898854eed`). Of the 6 child files exactly one differs by bytes: `meta/stats.json`, only in dictionary key order (parsed JSON equal, 0 numeric entries differ; first difference at char 7: `observation.images.image2` first vs `task_index` first). It is not part of the version digest. This is LeRobot's own stats writer, not our code.
- ✅ AC2.5: `check_blind(v1) = ['episode_index']`, `check_blind(v0) = []`, `assert_blind(v1)` refused (fenced). Existing verifiers: `verify_blind_s0.py` passes unchanged (AC0.3 `check_blind(child) = []`, the four AC0.5 tampers rejected, v1 refused; `artifacts/s2_regress_verify_blind_s0.txt`); it rewrote the evaluation feature's `s0_verify_blind.json` and a diff against HEAD showed only timings (write 11.2 -> 10.8 s, total 14.2 -> 13.7 s, frames/s), so I restored it with `git checkout`. `verify_derive_s4.py --guard --columns --cache` pass (`artifacts/s2_regress_verify_derive_s4.txt`; summary written to scratch). Not run: `--padding` (full-scale 7 GB derivation, out of scope for this batch) and `--v1` (a 7 GB transient load, and nothing in S2 touches what it checks).
  - One expectation differs, NOT caused by S2: AC4.8's "contrast" (`isolate_hf_cache=False` left 60 files / 663 MB in data-lineage round 1) now leaves 0 files. I re-ran `--cache` on the pre-S2 code (`git stash`) and got the same 0, so the order-preserving writer introduced in evaluation S0 no longer produces that cache. The isolation guarantee (shared cache listing identical before/after, 134 files both times) still holds.
- ✅ AC2.6: blind-check table (scratch child, tamper applied to the stored table and undone): as written `[]`; `task_name` changed on the swapped episode `['task_name']`; `libero_task_id` changed `['libero_task_id']`; `source_demo` changed on an edited episode `['source_demo']`; `language_instruction` of an unedited episode changed with meta untouched `['(meta/episodes does not match the episode table)']`; swapped row's instruction put back to the parent's while meta keeps the swap: same META_MISMATCH.
- ✅ AC2.7: docstrings of `derive_version` and `compare_blind` (and the lineage section comment) state the reclassification, the hashed-content reason (`catalog.py:127-148`) and that provenance stays bookkeeping (D1).

**Demo artifacts produced:** `artifacts/s2_run.txt` (blind-check table), `artifacts/s2_verify.json`, `artifacts/s2_regress_verify_blind_s0.txt`, `artifacts/s2_regress_verify_derive_s4.txt`.

**Notes / divergences from plan or spec:**
- Plan said "`tasks` out of META_BOOKKEEPING into the data-derived set". Doing only that would let `meta/episodes.tasks` drift from the catalog instruction unnoticed, so I added the tasks-equals-language_instruction consistency rule (extra, tested in AC2.6).
- `EpisodeEdit` gained `new_task` and `hypothesis` (the latter declared in the S1 commit).
- v1's labels have no `hypothesis` column (v1 is not rebuilt).
- Main `data/lineage/mlflow.db` sha256 unchanged (`7845d8d7...`); free disk 92 GiB; scratch dirs deleted.

### S3: Add the method-facing loader and the ground-truth boundary (owner decision 3)
**Files:**
- `vlapoc/method_data.py` (new): `open_for_method(subset_ref, lineage_root=None) -> MethodView` (frozen: `version_id`, `subset_id`, `lerobot_root`, `episode_indices`, `catalog`, `reference_ids`, `iter_episodes()`). Every call: `Subset.load` → `verify_version` → `assert_blind` → ground-truth-chain refusal → diff guard. Module docstring lists the catalog columns (grounding: PoC §2.1 item 4 p.3, owner decision 3, evaluation S0 fence).
- `vlapoc/groundtruth.py` (new, ground-truth side): `load_answer_key(version)` wraps `derive.load_labels` and cross-checks against `lineage.parent_hashes` and the version's episode table.
- `vlapoc/methods/__init__.py` (new, empty package) so the static scan has a `vlapoc/methods/` target; S5 adds the heuristics there.
- `scripts/verify_method_boundary_s3.py` (new).

**Premises checked:** there is no `oracle_drop` op in `lineage.py` today (grep), so ground-truth-derived is defined by op name `oracle_drop`, any op containing `groundtruth`, or a `groundtruth` tag, followed through `parent_subset_id` and `from_subset_id`. → documented in the module docstring; later features that create ground-truth subsets must use one of those markers.

**Acceptance criteria** (`.venv/bin/python scripts/verify_method_boundary_s3.py`, output `artifacts/s3_run.txt`, `artifacts/s3_verify.json`; 47 s):
- ✅ AC3.1: `open_for_method("v1/candidate")` → `LineageError ... fenced=True, offending=['episode_index']`. `v0/candidate` opens (960 episodes, 40 reference ids, catalog columns == `CATALOG_COLUMNS`, first episode decodes: action (98, 7), image (98, 128, 128, 3)). The S2 scratch derived version opens (20 episodes). One byte flipped in a data file of the scratch child → `VersionIntegrityError: dataset copy at ... does not match version small20_gt` (file restored afterwards).
- ✅ AC3.2: scratch subset with op `oracle_drop` (labelled episodes removed) → `LineageError ... is ground-truth-derived (op 'oracle_drop')`; a `drop_random_k` subset of it is refused too (chain recursion).
- ✅ AC3.3: in one process, child then its parent → `LineageError: refusing to open a second version of one lineage ...`; parent then its child → same; reopening the same version is allowed. Caveat: the real-store analogue (v0 then v1) is refused too, but there by the fence check, which runs first, so it does not demonstrate the guard; the scratch cases do.
- ✅ AC3.4: AST scan of `vlapoc/method_data.py` and `vlapoc/methods/*.py`: no violations. Six planted violations (from-import and module import of groundtruth, `load_labels` import, `lineage.parent_hashes` attribute, `labels.parquet` string, `fence.json` string) are each detected. Docstrings are exempt from the string scan; the forbidden-name set is `load_labels`, `parent_hashes`, `diff_versions`, `fence_record`, `load_answer_key`.
- ✅ AC3.5: all 21 `CATALOG_COLUMNS` are named in the module docstring; the view's columns equal that list and lie within the tables `compare_blind` checks; `check_blind(scratch derived child) = []`. Only the one scratch derived version exists so far; the full-scale v2 (S4) will repeat this.
- ✅ AC3.6: docstrings cite PoC §2.1 item 4 p.3, owner decision 3 (brief 2026-10-15) and the evaluation S0 fence.

**Demo artifacts produced:** the refusal table is in `artifacts/s3_run.txt` (rows: v1 fenced, v0 open, scratch derived open, tampered refused, oracle refused, oracle-chain refused, diff-guard cases); `artifacts/s3_verify.json`.

**Notes / divergences from plan or spec:**
- `open_for_method` runs `verify_version` on every call (hashes the whole version's files; v0 about 20 s). That is what the AC asks for; a method that opens a version many times pays for it each time.
- The diff guard registers a version only after every other check passes, and keys on (lineage root, version id), so scratch stores do not interfere with the real one.
- The catalog still exposes provenance (`task_name`, `episode_id`, position) next to the instruction (design D1), so a swapped instruction is findable by cross-check; S5 measures it.
- `MethodView.iter_episodes` reads each data file once and imports the private `derive._canonical_episode` (same package).
- Main `data/lineage/mlflow.db` sha256 unchanged (`7845d8d7...`); S3 writes only under `data/scratch/` (deleted); free disk 92 GiB.

### Findings for the analyzer
- AC1.5 (second clause) as written cannot hold for `time_shift` and `gripper_release` (deterministic by their quoted/inferred parameters) and cannot hold for every episode of `instruction_swap` / `detour` (finite draw sets, chance collisions: 105/118 and 112/118 differ). Same-seed determinism across processes holds for all seven. Suggest amending the AC.
- `meta/stats.json` of a derived copy differs between `PYTHONHASHSEED` values only in dict key order; content and `dataset_version_id` are identical. The plan's P12 quirk does not affect task order on the current writer path (measured, AC2.3/2.4).
- data+lineage AC4.8's "contrast" number (60 files left by `isolate_hf_cache=False`) no longer reproduces: 0 files, also on pre-S2 code. The order-preserving writer from evaluation S0 no longer creates that cache; the isolation guarantee still holds. The data-lineage report quotes the old number.
- Plan P7: over the 960 candidates 907 have a final release (the plan's 944 was over all 1000); 884 are eligible for `gripper_release` (carry ≥ 30).
- `action_noise` clip: 0.72% of translation values clipped, but 1.69% of translation values end up in (0.9375, 1], a range clean data never has (S4 will recount on stored bytes). A noise-level detector on max |a| could use that; S5's heuristics should include it if the owner wants the clipped-range leak measured.
- Not run in this batch: `verify_derive_s4.py --padding` (full-scale derivation) and `--v1` (7 GB transient load).

## Round 1: 2026-10-15 / Batch 2 (S4-S6)

### S4: Derive the full-scale injected version `v2` from v0
**Files:**
- `vlapoc/groundtruth.py` (+): `select_injection` (seeded, stratified), `build_edits`, `derive_injection`, `derive_injection_run` (MLflow `derive` stage run, tags `labels_sha256`), CLI `python -m vlapoc.groundtruth derive`. Grounding docstring: PoC §2.5 p.5, §10 p.20, §8.7 p.15, RoboDrop App. A.1 p.13, plan D2/D3.
- `scripts/verify_injection_s4.py` (new): AC4.1-4.7 verifier that recomputes everything from stored bytes.
- Committed first (`d0a0a14`), because the CLI refuses a dirty `vlapoc/`; the MLflow run records that commit (`vlapoc.git_dirty=false`).

**Premises checked:** the selection constraints (13 per type over 13 distinct tasks, 4 or 5 per task, 6/7 suite split, no repeated type within a task) are satisfiable → dry run of `select_injection` before deriving (window offsets `[0, 3, 6, 9, 12, 15, 18]` over the suite-interleaved task order) → holds. Eligible counts over the 960 candidates: 960 for six types, 884 for `gripper_release`.

**Run:** `HF_DATASETS_CACHE=$PWD/data/hf_datasets_cache .venv/bin/python -m vlapoc.groundtruth derive --parent v0 --name v2 --per-type 13 --seed 0` in the background, log `data/scratch/logs/gt_s4_derive.log` (copy of the JSON output: `artifacts/s4_derive_cli_output.txt`). No process left running.

**Acceptance criteria** (`.venv/bin/python scripts/verify_injection_s4.py --version v2 --parent v0 --run-id fece666aea1b44a08c2ff1da315d5701`, output `artifacts/s4_run.txt`, numbers `artifacts/s4_verify.json`, table `artifacts/s4_allocation.csv`):
- ✅ AC4.1: `v2` = `eaeed043455a...`, 1000 episodes, `labels.parquet` 91 rows, 13 per type for each of the 7 types; all in `v0/candidate`, 0 in `v0/reference`.
- ✅ AC4.2: allocation table printed (artifacts/s4_run.txt). Each type spans 13 distinct tasks, suite split 6/7 or 7/6 for every type, corrupted per task {4, 5} (11 tasks with 5, 9 with 4), no type repeated within a task; 91/960 = 9.48% of candidates.
- ✅ AC4.3: `check_blind(v2) = []`; `verify_version` ok (77 files); `v2/reference` 40 (same ids as v0's) and `v2/candidate` 960 inherited; the 40 reference content hashes equal v0's.
- ✅ AC4.4: all 1000 episodes re-hashed from the stored bytes equal the episode table; the 909 unedited ones equal v0's, both in the table and in the stored bytes; the 91 labelled episodes read back with `read_stored_episode` hash to their label's `content_hash` (91/91); S1's range, gripper-binary, length, content and image-provenance invariants hold on the stored data (0 failing episodes).
- ✅ AC4.5: the 13 `action_noise` labels: `n_values` 16,912; `n_clipped` 65 (0.38% of all values, 0.90% of dims-0-2 values); `n_beyond_clean_range` 139 (1.92% of dims-0-2 values). A recount from the stored bytes gives the same: values exactly ±1 on dims 0-2 = 65 (clean data has none: clean max 0.9375) and values with 0.9375 < |a| ≤ 1 = 139; the same count in the parents' episodes is 0.
- ✅ AC4.6: `du -sh data/lerobot/libero_spatial_10__v2` = 7.3G; free disk 92 GiB before, 83 GiB after (floor 20); `timing_s` guards 2.5 / write 532.7 / hash+blind check 18.1 / total 555.4 s; MLflow derive run `fece666aea1b44a08c2ff1da315d5701`.
- ✅ AC4.7: `trace(fece666a...)` visits 3 runs (build, derive, split), reaches the v0 version entity and lists 20 raw files; the run's tag `labels_sha256` (`1505920498116c77...`) equals `derivation.json`.

**Demo artifacts produced:** `artifacts/s4_run.txt`, `artifacts/s4_verify.json`, `artifacts/s4_allocation.csv`, `artifacts/s4_filmstrip.png` (one stored episode per type), `artifacts/s4_derive_cli_output.txt`.

**Notes / divergences from plan or spec:**
- The real `data/lineage/mlflow.db` changed as the AC requires (one `derive` run); sha256 before `7845d8d7...`, after `801a09ac...`. The lineage store also gained v2's version record, three subsets and a split under `data/lineage/` (labels and parents are lineage-only files there).
- Gripper-release (R10): 2 of the 13 selected episodes have their final release before 0.8T.
- The v2 copy is the one full-scale derived version of the feature; nothing else was derived at scale in this batch.

### S5: Score recovery on the ground-truth side and run the heuristic baselines through the loader
**Files:**
- `vlapoc/groundtruth.py` (+): `recovery_metrics`, `score_recovery` (key check via `lineage.check_score_keys`), `leak_check`, `score_run` (MLflow `score` stage), CLI `python -m vlapoc.groundtruth score --version ... --scores ... --column ...`. Grounding docstrings: PoC §2.1 item 4 p.3, §2.5 p.5, RoboDrop §5.1 p.7 and Table 7 (oracle); the tie rule is stated as inferred.
- `vlapoc/methods/heuristics.py` (new): H1-H6; reads only through `open_for_method`; records one `screen` stage run per heuristic.
- `vlapoc/lineage.py`: `_check_score_keys` made public as `check_score_keys` (old name kept as an alias).
- `vlapoc/runs.py`: `score` added to `STAGES`.
- `scripts/verify_recovery_s5.py`, `scripts/s5_recovery_table.py` (new).

**Premises checked:** sklearn 1.9.1 is installed (`roc_auc_score`, `roc_curve`, `average_precision_score` import fine) → holds. `smoothness` direction: `catalog.py` line 56 "lower is smoother", so a high value is the suspicious one → used a priori. Directions for all six heuristics were fixed in the module docstring before scoring ("high = more suspicious"); nothing was tuned afterwards.

**Acceptance criteria:**
- ✅ AC5.1: `.venv/bin/python scripts/verify_recovery_s5.py` → 26 of 26 comparisons equal hand-computed values: perfect ranker (recall@k 1, precision@k 1, AUROC 1, AP 1, balanced accuracy 1), the same ranking given as `direction="lowest"`, reversed ranker (recall 0, AUROC 0, AP (1/9 + 2/10)/2 = 0.155556, balanced accuracy 0.5), constant score (recall@2 = 0.2 and recall@4 = 0.4 = k/n, AUROC 0.5, AP 0.2), 3-tie (recall@2 = 2/3, recall@4 = 1, AUROC 7/8, balanced accuracy 0.75, AP 0.75). Derivations are in the script's comments. Output `artifacts/s5_verify_run.txt`.
- ✅ AC5.2: per type (13 injected vs 869 clean, n = 882) and pooled (91 vs 869, n = 960), each with `random_recall_at_npos` (13/882 = 0.0147 per type; 0.0948 pooled), `random_auroc` 0.5 and n. Sanity check on 200 seeded random rankers: mean pooled AUROC 0.4987, mean recall@k 0.0931 (expected 0.0948).
- ✅ AC5.3: a v0-keyed table (same episode_ids, v0 hashes; 91 of 960 hashes differ) → `LineageError: score table keys != parent subset's episode versions: 91 unknown/stale ...`; a table missing one candidate → `LineageError ... 1 unscored`.
- ✅ AC5.4: `.venv/bin/python -m vlapoc.methods.heuristics --subset v2/candidate --out data/eval/groundtruth/heuristics/` (90 s), then `python -m vlapoc.groundtruth score --version v2 --scores <h>.parquet --column <h> --out ...` for each of the six; table assembled by `scripts/s5_recovery_table.py` into `artifacts/s5_recovery_heuristics.csv` (and `s5_recovery_table.txt`; per-heuristic JSON in `artifacts/s5_recovery_json/`). No expected values were set. Results, AUROC (recall@k, k = 13 per type; pooled k = 91):

  | heuristic | action_noise | time_shift | instruction_swap | dropped_frames | detour | idle_padding | gripper_release | pooled |
  |---|---|---|---|---|---|---|---|---|
  | H1 length_steps | 0.446 | 0.527 | 0.537 | 0.476 | 0.730 | 0.681 | 0.479 | 0.554 |
  | H2 idle_steps | 0.445 | **1.000** | 0.485 | 0.520 | 0.445 | **1.000** | 0.481 | 0.625 |
  | H3 gripper_toggles | 0.445 | 0.400 | 0.495 | 0.481 | 0.584 | 0.654 | 0.505 | 0.509 |
  | H4 smoothness | **1.000** | 0.337 | 0.512 | 0.508 | 0.543 | 0.502 | 0.449 | 0.550 |
  | H5 duplicate_frame_pairs | 0.500 | 0.500 | 0.500 | **1.000** | 0.500 | **1.000** | 0.500 | 0.643 |
  | H6 instruction_mismatch | 0.500 | 0.500 | **1.000** | 0.500 | 0.500 | 0.500 | 0.500 | 0.571 |

  Recall@13 is 1.000 in exactly those bold cells and at or near the random 0.015 elsewhere (largest other values: detour 0.105 by H3, 0.077 by H1/H4). Pooled recall@91: H5 0.335, H2 0.308, H4 0.220, H6 0.213, H1 0.143, H3 0.125 (random 0.095).
  **What this says (measured, not set in advance):**
  - Findable by a trivial heuristic (AUROC 1.0): `action_noise` (roughness H4, because clipped-noise raises the step change), `time_shift` (the 40-step zero tail is idle, H2; H4 is *below* 0.5, so the opposite direction would also separate it), `idle_padding` (H2 and H5, because pauses repeat frames and zero the arm), `dropped_frames` (H5, because a held frame is an exact duplicate; clean data has none), and `instruction_swap` (H6, the instruction vs `task_name` cross-check; plan design D1 predicted this).
  - Not found by these six: `gripper_release` (best AUROC 0.505, no heuristic above chance) and `detour` (best 0.730 by length, recall 0.077, a weak signal because it adds 40 frames without changing the other statistics).
  - Consequence for later features: five of the seven types are saturated by one-line statistics, so a screening method's recovery number is only informative on `gripper_release` and `detour` and on the pooled score minus those five. A gradient-based method should be compared against these baselines per type, not pooled only.
  - H5 and H6 are also leaks of the *construction*, not of the loader: a held or paused frame is an exact duplicate (P4) and a swap leaves `task_name`/`episode_id` true (D1). They are measured here, not removed.
- ✅ AC5.5: leak check (ground-truth side, ranking by `episode_index`): v1 recall@100 = 1.000 (AUROC 1.000, random 0.104); v2 recall@91 = 0.099 (AUROC 0.505, random 0.095). Output in `artifacts/s5_verify_run.txt`.
- ✅ AC5.6: each of the six scoring calls is an MLflow `score` run (ids in `artifacts/s5_score_runs.txt`) whose tags include `score_table_digest`, `scored_version_id` (v2 `eaeed043455a...`) and `labels_sha256` (`150592049811...`), with `vlapoc.git_dirty=false`. `lineage trace` on each reaches v2 and its producing `screen` run (5 runs: score, screen, derive, build, split). Output `artifacts/s5_trace_check.txt`.
- ✅ AC5.7: docstrings of `groundtruth.py` (scoring section) and `heuristics.py` cite PoC §2.1 item 4 p.3, §2.5 p.5, §5 p.11-12, §2.2 p.4, RoboDrop §5.1 p.7 and Table 7; the tie rule is marked inferred.

**Demo artifacts produced:** `artifacts/s5_recovery_heuristics.csv`, `artifacts/s5_recovery_table.txt`, `artifacts/s5_recovery_json/`, `artifacts/s5_verify_run.txt`/`s5_verify.json`, `artifacts/s5_score_runs.txt`, `artifacts/s5_trace_check.txt`, `artifacts/s5_boundary_rescan.txt` (the S3 static scan re-run with `heuristics.py` present: no violations).

**Notes / divergences from plan or spec:**
- The score tables themselves are under `data/eval/groundtruth/heuristics/` (not in git, per the repo rule).
- The real `mlflow.db` changed as the ACs require: 6 `screen` runs and 6 `score` runs (plus the S4 derive run); sha256 after S5 is `see S6 entry` below.
- `smoothness` AUROC for `time_shift` is 0.337 (< 0.5) because the zero tail lowers the mean step change; I report it as scored, not flipped.

### S6: Prove v2 is trainable and evaluable locally, and write the server runbook
**Files:**
- `scripts/verify_consumers_s6.py` (new): per-subset stats checks, the SmolVLA smoke on a scratch label-side subset, the task-string and human-reference checks, and a before/after check of the real store.
- `tracker/features/2026-10-15-ground-truth/artifacts/server-runbook.md` (new). **It is a ground-truth runbook in this feature's `artifacts/`; I did not edit the evaluation feature's `server-runbook.md`** (it reuses that file's environment section and AC8.5 / AC5.8 / AC6.7 commands by reference).

**Premises checked:** `data/hf_models/smolvla_base` and `data/hf_home` exist (the evaluation S8 configuration needs them) → holds, so the ACT fallback was not needed. The evaluation feature's human-reference files were last modified 2026-10-07, before this feature started → holds.

**Acceptance criteria** (`.venv/bin/python scripts/verify_consumers_s6.py`, output `artifacts/s6_run.txt`, `artifacts/s6_verify.json`; 33 s):
- ✅ AC6.1: `python -m vlapoc.subset_stats --check v2` → aggregate_stats over all 1000 episodes equals v2's `stats.json`, worst max |diff| 2.70e-10 (tolerance 1e-6; action mean 1.1e-16, std 2.1e-10). `v2/candidate` stats digest `b2909c49...` differs from `v0/candidate`'s `2c62ec3f...`. Per-dim action mean/std of v2/candidate vs v0/candidate are in `artifacts/s6_action_stats_v2_vs_v0.csv`: the std ratio is 0.9973-1.0000 (largest shift: dim 0 mean -0.0012, std x0.9973), i.e. the injection moves the normalization by under 0.3%.
- ✅ AC6.2: SmolVLA 2-step fine-tune from `smolvla_base` on MPS (`policy_overrides load_vlm_weights=False`, batch 2, seed 0, tagged smoke) on a 14-episode subset of v2 holding one labelled episode of each of the seven types plus seven clean candidates from distinct tasks. The subset was built label-side in a **scratch** lineage copy (`data/scratch/gt_s6/lineage`, op `gt_smoke_subset`, `groundtruth: true`); `data/lineage/subsets` has 12 files before and after. The run finished in 18.5 s with `episodes_match` true; tag `train.subset_stats.sha256` `c60d2183e24e139a...` equals the recomputed digest, and the saved normalizer and unnormalizer mean/std equal the subset stats (max |diff| ≤ 5.7e-8) while differing from v2's full `stats.json` (up to 0.81 for state mean). Scratch run id `a9986f2d670e...` lives in the scratch MLflow DB, deleted afterwards. The scratch dir (lineage copy, HF cache, 1.2 GB checkpoint) was removed.
- ✅ AC6.3: v2's 20 task strings (task_index to string) equal v0's. The `data/eval/human_reference/` tree digest is `68c888b1930d66de...` before and after the script; its files were last modified 2026-10-07, before this feature began, so the HRT anchor is still the 1000 v0 originals.
- ✅ AC6.4: `artifacts/server-runbook.md` lists GT-R1 to GT-R6, each with command, inputs (version and subset ids), outputs, disk estimate (labelled inferred or measured) and order; GT-R1-GT-R5 are pending server, GT-R6 pending later features. Checked by reading it against the AC's list.
- ✅ AC6.5: the runbook ("What cannot be shown here") states that LIBERO has no ghost-event rule (evaluation brief decision 6), so the gripper-release "drops" hypothesis can only appear as success, F(tau) or timeouts.

**Demo artifacts produced:** `artifacts/s6_run.txt`, `artifacts/s6_verify.json`, `artifacts/s6_action_stats_v2_vs_v0.csv`, `artifacts/server-runbook.md`.

**Notes / divergences from plan or spec:**
- AC6.2's run is tagged `smoke` (the name contains "smoke"), as `train_run` requires for smoke runs, and it never entered the real MLflow store.
- Real store: `data/lineage/mlflow.db` sha256 was `ecd5b4b4...` before and after S6 (it already included the S4/S5 runs); the batch as a whole changed it from `7845d8d7...` (batch-1 end) to `ecd5b4b4...`, through 1 derive run (S4) plus 6 screen and 6 score runs (S5), as those ACs require.
- `train_run` and the eval path do not go through `open_for_method`. That is intended for the injected training arm (runbook says so) but means the loader's protection covers methods only, not trainers; a trainer given `v1/candidate` would still run.

### Findings for the analyzer (batch 2)
- Five of seven corruption types are found with AUROC 1.0 by a single one-line statistic on stored data (noise by roughness, time shift and idle padding by idle steps, dropped and idle frames by duplicate-frame count, swaps by the instruction vs `task_name` cross-check); `gripper_release` is at chance for all six heuristics and `detour` is weak (best AUROC 0.73). Recovery on the pooled set therefore overstates how hard the benchmark is for a real method.
- The H5/H6 findability is a property of how the corruptions are built (exact duplicates, true provenance, plan D1/P4), not of the loader. Both are measured; fixing them needs re-rendering or opaque keys (out of scope).
- `action_noise` leaves 1.92% of translation values in (0.9375, 1], a range clean data never has (measured, S4). A `max |a|` statistic would likely separate it; it was not among H1-H6 and was not added.
- `EpisodeEdit.params` is completed by the transform at run time (S1 note); the labels written by `derive_version` are therefore only complete after the writer pass.
- Open from the plan, unchanged: clip bound [-1, 1] and swap provenance (D1) kept as defaults.
- Disk after the batch: 83 GiB free; v2 copy 7.3 GB; real `data/lineage` grew by v2's version record, three subsets, a split and 13 runs.

## Round 2: 2026-10-17

Owner ruling A (brief.md, 2026-10-17; `dispute-r1.md`): fix `time_shift` only. Plan r2 (`62f80a5`). Work order: (1) the trim in `corruptions.py` and S1's verifier, (2) one re-derivation as v3 and v2's retirement, (3) S4-S6 on v3, (4) the 7.3G vs 6.8G reconciliation.

### R2 step 1: the `time_shift` trim (S1)
**Files:**
- `vlapoc/corruptions.py`: `time_shift` now returns obs[0:T-40] (state, image, image2) paired with a[40:T]; no fill; `params["length_change"] = -40`; `vacated_steps` parameter tagged inferred (owner ruling A); docstring and module docstring state the change and the residual length signal.
- `scripts/verify_corruptions_s1.py`: `time_shift` expectations (T-40, parent frames 0..T-41, parent actions 40..T-1, `length_change == -40`), AC1.8 check and distribution, AC1.5 amended thresholds asserted.

**Acceptance criteria** (`artifacts/s1_r2_run1.txt`, `s1_r2_run2_check_hashes.txt`; r1 numbers kept in `artifacts/s1_invariants_r1.json`; same 118-episode seeded sample as round 1):
- ✅ AC1.4: `time_shift` gives T-40 with state and images equal to parent frames 0..T-41 and actions equal to parent actions 40..T-1 bit for bit; all other type checks unchanged; 0 invariant failures on 118 episodes per type (`invariant_failures` 0 in all seven rows).
- ✅ AC1.5 (amended): same (seed, episode_id) gives the same hash in two fresh processes for all seven types (all True). A different seed (0 vs 1) changes the hash for 118/118 `action_noise`, `dropped_frames`, `idle_padding`; 112/118 = 94.9% of `detour`; 105/118 = 89.0% of `instruction_swap` (both ≥ 80%); 0/118 for `time_shift` and `gripper_release`, asserted. The round-1 ❌ on this AC is resolved by the amendment, not by a code change.
- ✅ AC1.6: eligibility unchanged: 960 for six types (`time_shift` needs T > 40, all 960), 884 for `gripper_release`; 907/960 have a final release; all types ≥ 13 eligible in 20 tasks.
- ✅ AC1.8: for every `time_shift` output the child's idle_steps equals the parent's all-zero-arm steps within a[40:T]. Distribution over the 118 episodes: max 2, p99 2, > 0 in 6 episodes (the plan's 960-candidate measurement: max 7, p99 2); the clean maximum run is 21.
- ✅ AC1.1-AC1.3, AC1.7 hold as in round 1 (params carry `source`, tagged parameters and `corrupted_frames`; `time_shift` records clip counts 0; the module docstring was updated for the trim).

**Demo artifacts produced:** `artifacts/s1_traces.png`, `artifacts/s1_filmstrip.png` regenerated (the `time_shift` panel now shows a 40-step shorter child), `artifacts/s1_invariants.json` (r2), `artifacts/s1_invariants_r1.json` (r1 for comparison).

**Notes:** `eligible("time_shift")` stays "T > 40" (shortest child 35 frames, measured in the plan).

### R2 step 2: re-derivation as v3, v2 retired (S4, AC4.1-AC4.8 on v3)
**Files:** `vlapoc/groundtruth.py` (+ `retire_version`, CLI `retire`: writes `fence.json` through `lineage.write_fence`), `scripts/verify_injection_s4.py` (generalised: `--prefix`, `--prior`; trim checks for `time_shift`; AC4.8 section). Committed before the run (`vlapoc.git_dirty=false`).

**Run:** `.venv/bin/python -m vlapoc.groundtruth derive --parent v0 --name v3 --per-type 13 --seed 0` in the background (`data/scratch/logs/gt_s4_v3_derive.log`, copy of its JSON `artifacts/s4_v3_derive_cli_output.txt`); free disk 86 GiB before, 78 GiB after; no process left running. Then `python -m vlapoc.groundtruth retire --version v2 --superseded-by v3`, then `.venv/bin/python scripts/verify_injection_s4.py --version v3 --parent v0 --prefix s4_v3 --prior v2 --run-id d3dd9fd4dff84d37b5be21fa563fcc57` (`artifacts/s4_v3_run.txt`, `s4_v3_verify.json`, `s4_v3_allocation.csv`, `s4_v3_filmstrip.png`). The round-1 v2 artifacts (`s4_run.txt`, `s4_verify.json`, ...) are kept as the record of what v2 was.

**Acceptance criteria (v3 = `002a68e4caf2...`, MLflow derive run `d3dd9fd4dff84d37b5be21fa563fcc57`):**
- ✅ AC4.1: 1000 episodes; 91 labels, 13 per type; all in `v0/candidate`, none in `v0/reference`.
- ✅ AC4.2: allocation identical to v2's (same selection, seed 0): 13 distinct tasks per type, 6/7 suite split per type, 4 or 5 per task (11 tasks with 5, 9 with 4), no type repeated within a task, 91/960 = 9.48%.
- ✅ AC4.3: `check_blind(v3) = []`; `verify_version` ok (77 files); `v3/reference` 40 and `v3/candidate` 960 inherited; the 40 reference hashes equal v0's.
- ✅ AC4.4: all 1000 episodes re-hashed from stored bytes equal the episode table; the 909 unedited ones equal v0's (table and bytes); 91/91 labelled episodes hash to their label; S1 invariants (with the trim: T-40, state and images equal parent frames 0..T-41, actions equal parent actions 40..T-1, `length_change == -40`) hold on stored data (0 failing).
- ✅ AC4.5: identical to v2 (the `action_noise` rows did not change): n_values 16,912, n_clipped 65 (0.90% of translation values), n_beyond_clean_range 139 (1.92%); recount from stored bytes 65 and 139; the parent has 0.
- ✅ AC4.6: derive timing guards 2.4 / write 538.5 / hash+blind 19.1 / total 562.2 s; v3 has 200,860 frames (v2: 201,380; 13 trimmed episodes × -40 minus the 0 lengthened-difference = -520, as expected); free disk 86 -> 78 GiB. **Size: see the reconciliation below.**
- ✅ AC4.7: `trace(d3dd9fd4...)` visits 3 runs (build, derive, split), reaches the v0 version entity and lists 20 raw files; tag `labels_sha256` (`3e14922a63cf...`) equals `derivation.json`.
- ✅ AC4.8: v3 labels have the same 91 `episode_id`s and types as v2's; the 78 non-`time_shift` rows keep their `content_hash`; the 13 `time_shift` rows differ, with `length_change: -40` (v2's had none). `fence_record("v2")` has reason "superseded by v3" and `superseded_by` = v3's id `002a68e4...`; `open_for_method("v2/candidate")` raises `LineageError: ... fenced=True`; `open_for_method("v3/candidate")` opens 960 episodes. AC1.8 holds on v3's stored bytes for the 13 `time_shift` episodes. v2's physical copy (`data/lerobot/libero_spatial_10__v2`) was NOT deleted (owner's cleanup). Free disk after the derivation 78 GiB (> 20).

**Reconciliation of 7.3G vs 6.8G (step 4; all measured today):**
- Round 1's "7.3G" was `du -sh data/lerobot/libero_spatial_10__v2` run minutes after the copy was written. Today the same command on v2 prints 6.8G. The analyzer's 6.8G is the same command, run later.
- `du` counts allocated disk blocks, not bytes. Per-file `st_blocks * 512` versus `st_size`: v2 (two days old) size 6.842 GiB, allocated 6.842 GiB, slack 0.2 MiB total. v3 (written minutes ago) size 6.824 GiB, allocated 7.415 GiB, slack 606 MiB (median 6.3 MiB per file, up to 16 MiB), so `du -sh` prints 7.4G and `du -sk` 7,775,640. The freshly written parquet files carry allocated-but-unused tail blocks, which v2 no longer has.
- So the sum of file sizes is the stable figure: v2 7,346,100,191 bytes = 7.346 GB = 6.842 GiB; v3 7,326,736,215 bytes = 7.327 GB = 6.824 GiB (v0 6.851 GiB, v1 6.849 GiB). "7.3 GB" (decimal) and "6.8 GiB" are the same number; the 7.3G from `du -sh` in GiB units was block slack on a fresh copy. Why the slack disappears with time (APFS reclaiming speculative preallocation) is **inferred**, not tested: I did not observe v3's slack shrink within this session. Disk budgeting should use `df` free-space change instead: v3 cost 86 -> 78 GiB = 8 GiB while the slack was present.

**Notes:** the real `mlflow.db` changed (one more derive run): sha256 before this step `ecd5b4b4...`, see the final entry for the value at the end of round 2. `labels.parquet` of v3 lives in the lineage store as for v2.

### R2 step 3a: S5 on v3 (AC5.1-AC5.8)
**Files:** `scripts/verify_recovery_s5.py` (`--version`, `--prefix`), `scripts/s5_recovery_table.py` (`--prior-dir`, `--bands-out`, AC5.8 band table and the v2-vs-v3 `time_shift` rows). Commands, all on v3: `scripts/verify_recovery_s5.py --version v3 --prefix s5_v3`; `python -m vlapoc.methods.heuristics --subset v3/candidate --out data/eval/groundtruth/heuristics_v3/`; `python -m vlapoc.groundtruth score --version v3 --scores <h>.parquet --column <h> --out ...` for each of the six; `scripts/s5_recovery_table.py --dir ... --prior-dir artifacts/s5_recovery_json --bands-out artifacts/s5_v3_bands.csv`. Outputs: `artifacts/s5_v3_verify_run.txt`, `s5_v3_recovery_heuristics.csv`, `s5_v3_recovery_table.txt`, `s5_v3_bands.csv`, `s5_v3_recovery_heuristics_time_shift_v2_vs_v3.csv`, `s5_v3_recovery_json/`, `s5_v3_score_runs.txt`, `s5_v3_trace_check.txt`, `s5_v3_boundary_rescan.txt`. The round-1 (v2) S5 files are unchanged.

**Acceptance criteria:**
- ✅ AC5.1: 26 of 26 fixture comparisons equal the hand-computed values (unchanged code).
- ✅ AC5.2: per type n = 882 (13 vs 869) and pooled n = 960 (91 vs 869), each with random expectations; 200 random rankers give mean pooled AUROC 0.4987 and recall@k 0.0931 (expected 0.0948).
- ✅ AC5.3: a v0-keyed table (91 of 960 hashes differ from v3's) and a table missing one candidate are both rejected with `LineageError`.
- ✅ AC5.4 (regenerated on v3, no expected values set): table below; heuristics read only through `open_for_method` (the S3 static scan re-run: no violations).
- ✅ AC5.5: leak check: v1 recall@100 = 1.000 (AUROC 1.000); v3 recall@91 = 0.099 (AUROC 0.505; random 0.095).
- ✅ AC5.6: six `score` runs (ids in `s5_v3_score_runs.txt`), tags `score_table_digest`, `scored_version_id` (v3 `002a68e4caf2...`) and `labels_sha256` (`3e14922a63cf...`); `lineage trace` of each reaches v3 and its screen run (5 runs).
- ✅ AC5.7: docstrings unchanged and still cite PoC §2.1 item 4 p.3, §2.5 p.5, RoboDrop §5.1 p.7 and Table 7.
- ✅ AC5.8: per-type statement on v3, with the bands as inferred reporting bands (best of the six heuristics at their a-priori direction; the direction-free value is shown because AUROC below 0.5 means the opposite direction would work):

| type | best heuristic (as scored) | AUROC | band | direction-free best |
|---|---|---|---|---|
| `action_noise` | smoothness | 1.000 | (a) | 1.000 |
| `dropped_frames` | duplicate_frame_pairs | 1.000 | (a) | 1.000 |
| `idle_padding` | idle_steps | 1.000 | (a) | 1.000 |
| `instruction_swap` | instruction_mismatch | 1.000 | (a) | 1.000 |
| `detour` | length_steps | 0.730 | (b) | 0.730 |
| `time_shift` | smoothness | 0.519 | (c) | length_steps 0.682 (inverted direction) |
| `gripper_release` | gripper_toggles | 0.505 | (c) | smoothness 0.551 |

  - (a) found by a one-line statistic. `action_noise` (roughness) and `dropped_frames` / `idle_padding` (exact duplicate frames; idle_steps for idle_padding) are **partly inherent, partly construction**: noise raises the step-to-step change by definition; a held or paused frame is a duplicate because we cannot re-render (plan P4, out of scope), so that part is a construction artifact. `instruction_swap` is a **construction artifact of design D1**: the instruction is checked against `task_name` / `episode_id`, which stay true.
  - (b) `detour` is only partly found (length 0.730, recall@13 0.077 per `s5_v3_recovery_heuristics.csv`): +40 frames with real frames, no other statistic changes.
  - (c) `gripper_release` is not found by any of the six (best 0.505 as scored). `time_shift` is not found at its a-priori directions.
  - **`time_shift`, reported openly (owner ruling A):** idle_steps AUROC **0.445** (r1 with the idle tail: **1.000**), i.e. about chance, as AC1.8 predicted (the trimmed child's idle count equals the parent's within a[40:T]). Length (H1) AUROC is **0.318** as scored (long = suspicious) because the trim makes the child 40 steps *shorter*; read in the opposite direction it is 0.682, a partly-found residual signal, **not** hidden here. Smoothness 0.519 (r1 0.337), duplicate frames 0.500, instruction mismatch 0.500, gripper toggles 0.400 (unchanged: the gripper is untouched by the shift apart from the trimmed ends). Side by side (one type vs 869 clean; `..._time_shift_v2_vs_v3.csv`):

| heuristic | AUROC v2 (idle tail) | AUROC v3 (trim) | recall@13 v2 | recall@13 v3 |
|---|---|---|---|---|
| length_steps | 0.527 | 0.318 | 0.000 | 0.000 |
| idle_steps | 1.000 | 0.445 | 1.000 | 0.000 |
| gripper_toggles | 0.400 | 0.400 | 0.000 | 0.000 |
| smoothness | 0.337 | 0.519 | 0.000 | 0.077 |
| duplicate_frame_pairs | 0.500 | 0.500 | 0.015 | 0.015 |
| instruction_mismatch | 0.500 | 0.500 | 0.015 | 0.015 |

  - Which types test a method beyond one-line statistics: `gripper_release` and `time_shift` (band (c) at the a-priori direction; the length residual makes `time_shift` detectable at 0.68 by a statistic that knows the direction), with `detour` partly. The other four (`action_noise`, `dropped_frames`, `idle_padding`, `instruction_swap`) are saturated by one statistic. Pooled recall@91 on v3: duplicate_frame_pairs 0.335, smoothness 0.242, instruction_mismatch 0.213, idle_steps 0.172, length_steps 0.132, gripper_toggles 0.125 (random 0.095).
  - Known limitations recorded by the owner (ruling A, not fixed): provenance stays visible (B) and idle/duplicate frames are exact duplicates because they are not re-rendered (C).

**Notes:** the real `mlflow.db` gained 6 `screen` and 6 `score` runs for v3 (sha256 `e5499b65acfc...` after this step).

### R2 step 3b: S6 on v3 and the runbook (AC6.1-AC6.6)
**Files:** `scripts/verify_consumers_s6.py` (`--version`, `--prefix`), `artifacts/server-runbook.md` rewritten for v3. Command: `.venv/bin/python scripts/verify_consumers_s6.py --version v3 --prefix s6_v3` (`artifacts/s6_v3_run.txt`, `s6_v3_verify.json`, `s6_v3_action_stats_vs_v0.csv`). The round-1 S6 files are kept.

**Acceptance criteria:**
- ✅ AC6.1: `python -m vlapoc.subset_stats --check v3`: aggregate over all 1000 episodes equals v3's `stats.json`, worst max |diff| 2.60e-10 (tolerance 1e-6). `v3/candidate` digest `168667c2365bf7dd` differs from `v0/candidate`'s `2c62ec3fec3e078f`. Action std ratios v3/candidate over v0/candidate: 0.9986-1.0003; gripper mean shifts from -0.0446 to -0.0423 (the trim removes the episode ends, where the gripper is open: inferred from the construction, the shift itself is measured).
- ✅ AC6.2: SmolVLA 2-step fine-tune (MPS, smoke) on a 14-episode label-side subset of v3 (one labelled episode of each of the seven types plus seven clean candidates), built in a scratch lineage copy; `data/lineage/subsets` stayed at 15 files; run finished in 15.1 s, `episodes_match` true, `train.subset_stats.sha256` tag `1d2a5e14c68d77c7...` equals the recomputed digest, and the checkpoint's normalizer/unnormalizer equal the subset stats (max |diff| 3.5e-08) while differing from v3's full `stats.json` (up to 0.78). Scratch (lineage copy, HF cache, 1.2 GB checkpoint) removed.
- ✅ AC6.3: v3's 20 task strings equal v0's; the `data/eval/human_reference/` digest `68c888b1930d66de...` is unchanged before and after (its files date from 2026-10-07).
- ✅ AC6.4: runbook lists GT-R1 to GT-R6 for v3 with commands, inputs (v3 `002a68e4...`, `v3/candidate` `8d80a5d4731e...` with 192,945 frames), outputs, disk estimates and order.
- ✅ AC6.5: runbook still states that LIBERO has no ghost-event rule, so the gripper-release "drops" hypothesis can only show up as success, F(tau) or timeouts.
- ✅ AC6.6: the runbook names v3 in every GT-R command (a text search for "v2" finds only the header sentence that says v2 is superseded), and carries a "Known issue for the attribution feature" section: `train_run` and `eval_run` do not call `open_for_method` or `assert_blind`; correct for this feature's training arms; a later method that trains policies could pass a ground-truth-derived subset without being refused; the attribution feature must route its training inputs through the loader. It also reports the size reconciliation and the detectability bands (four types saturated; `detour` partly; `gripper_release` and `time_shift` not found at the a-priori direction, with the length residual stated).

### R2 step 4 and regression: reconciliation and S2 re-run
- 7.3G vs 6.8G: reconciled in step 2 above (block slack on a fresh copy, measured per file).
- Regression: `scripts/verify_derive_gt_s2.py` re-run with the trim in the seven-type derivation (`artifacts/s2_r2_run.txt`): AC2.1-AC2.6 hold (scratch derivation registered `check_blind = []`; tasks.parquet equal to the parent's; `PYTHONHASHSEED` 0 vs 1 identical apart from `meta/stats.json` key order; v1 still `['episode_index']`, fenced). Its `s2_verify.json` was overwritten by that run and restored from git, the r2 evidence being the run text.

### Round 2 end state
- Real `data/lineage/mlflow.db` sha256: `ecd5b4b4...` at the start of round 2 (end of round 1), `803a24d3...` after the v3 derive run, `e5499b65acfc9cb9` after the 12 S5 runs on v3 (6 screen, 6 score); unchanged by S6 and by the S2 re-run. The real store gained those 13 runs (as the ACs require) and v3's version record, three subsets, a split, and v2's `fence.json`.
- Free disk 78 GiB (v3 copy 6.82 GiB of bytes, written as 7.4 GiB allocated); v2's physical copy is untouched (owner cleanup). No process left running. Scratch tidied (`data/scratch/gt_s1*` removed).
- v2 is fenced in the lineage store: `versions/eaeed043455a.../fence.json`, reason "superseded by v3".

## Round 3: 2026-10-17

Plan r3 (`c1768bb`), analysis-r2 feedback to the executor. Steps: (1) H7/H8, (2) S3 boundary re-run, (3) runbook, (4) R12, (5) fence wording.

### R3 step 1: H7 and H8 (S5, AC5.4, AC5.8, AC5.9)
**Files:** `vlapoc/methods/heuristics.py` (+H7 `length_vs_task`, ranked both directions as `length_vs_task` / `length_vs_task_lowest`; +H8 `post_release_share_vs_task`; `DIRECTIONS`; one shared pass `scan_episodes` for H5 and H8; `directions.json` written next to the tables), `scripts/s5_recovery_table.py` (the nine score files). Committed before the run.
- H7 = `length_steps` / median `length_steps` of the same `task_name` in the subset (catalog columns only). H8 = share of the episode after the final gripper release, (T - i)/T for the last i with gripper > 0 then < 0 (0 if none), minus the task median (actions via `iter_episodes`). Both are label-free; they read only through `open_for_method` (the S3 static scan still passes: `artifacts/s5_r3_boundary_scan.txt`).
- Directions fixed a priori: H7 both ways (a trim makes episodes shorter than their task's norm, detours and padding longer), H8 highest-first (a premature release moves the final release earlier, so the post-release share grows).

**Commands:** `python -m vlapoc.methods.heuristics --subset v3/candidate --out data/eval/groundtruth/heuristics_v3_r3/ --only length_vs_task length_vs_task_lowest post_release_share_vs_task` (only the three new ones, so H1-H6 are not re-run and no duplicate screen runs are created), then `python -m vlapoc.groundtruth score --version v3 --scores ... --column ... --direction ...` for each, then `scripts/s5_recovery_table.py --dir artifacts/s5_v3_recovery_json --out artifacts/s5_v3_r3_recovery_heuristics.csv --prior-dir artifacts/s5_recovery_json --bands-out artifacts/s5_v3_r3_bands.csv`.

**Real store, not a scratch copy:** the score runs must reach v3 through `lineage trace` (AC5.6), so they were written to the real `data/lineage/mlflow.db`. sha256 before `e5499b65acfc9cb9`, after `306af4123fda7e63` (3 screen + 3 score runs; `artifacts/s5_r3_mlflow_sha.txt`).

**Acceptance criteria:**
- ✅ AC5.4 (r3): on `v3/candidate` the set is H1-H8 with H7 in both directions; table `artifacts/s5_v3_r3_recovery_heuristics.csv` (per type 13 vs 869, pooled 91 vs 869), command above. H7 and H8 AUROC (recall@13):

| heuristic | action_noise | time_shift | instruction_swap | dropped_frames | detour | idle_padding | gripper_release | pooled |
|---|---|---|---|---|---|---|---|---|
| H7 length_vs_task, longest first | 0.385 | 0.152 | 0.534 | 0.418 | **0.960** (0.154) | **0.950** (0.308) | 0.504 | 0.557 |
| H7 length_vs_task, shortest first | 0.615 | **0.848** (0.462) | 0.466 | 0.582 | 0.040 | 0.050 | 0.496 | 0.443 |
| H8 post_release_share_vs_task | 0.517 | 0.768 (0.000) | 0.583 | 0.484 | 0.370 | 0.279 | **0.964** (0.000) | 0.566 |

  Recall@13 is low even where AUROC is high (e.g. H8 on `gripper_release`: AUROC 0.964, recall@13 0.000): the injected episodes rank high but not in the very top 13 of 882, because clean episodes with a naturally long post-release tail sit above them.
- ✅ AC5.6: three more `score` runs (ids in `s5_r3_score_runs.txt`) with tags `score_table_digest`, `scored_version_id` (v3), `labels_sha256`; `lineage trace` reaches v3 and the producing screen run (`s5_r3_trace_check.txt`).
- ✅ AC5.8 (r3), best of H1-H8 per type (a-priori direction; `artifacts/s5_v3_r3_bands.csv`):

| type | best heuristic | AUROC | band |
|---|---|---|---|
| `action_noise` | H4 smoothness | 1.000 | (a) |
| `dropped_frames` | H5 duplicate_frame_pairs | 1.000 | (a) |
| `idle_padding` | H2 idle_steps | 1.000 | (a) (H7 longest-first 0.950 also) |
| `instruction_swap` | H6 instruction_mismatch | 1.000 | (a) |
| `gripper_release` | H8 post_release_share_vs_task | 0.964 | (a) |
| `detour` | H7 length_vs_task | 0.960 | (a) |
| `time_shift` | H7 length_vs_task, shortest first | 0.848 | (b) |

  The analyzer's expected values (about 0.85, 0.96, 0.96) are reproduced exactly: 0.848, 0.960, 0.964. The r2 statement that `gripper_release` and `time_shift` are "not found" (band c) is superseded: they were missed because H1-H6 were not task-normalised.
- ✅ AC5.9 (headline, written honestly): **with H1-H8 no type is in band (c).** Six of seven types are found at AUROC ≥ 0.95 by a label-free one-line statistic; `time_shift` is the hardest at 0.848 (a residual of the 40-step trim: the child is shorter than its task's norm). Any later method's recovery on this ground truth means something only per type, against that type's best heuristic AUROC in `s5_v3_r3_bands.csv`, and not against chance (0.5). That is the owner-approved limitation under ruling A (with B, provenance stays visible, and C, duplicates not re-rendered), plus D: the length and release-position signals are inherent to how the corruptions change episode structure, not defects fixed here. The three expected-value checks above were not tuned: the directions were fixed before scoring and the numbers match the analyzer's probes (`data/scratch/gt_an3/lenres.py`, `grip.py`).
- Pooled recall@91 of the new heuristics: H7 longest-first 0.286, H8 0.220, H7 shortest-first 0.176 (random 0.095; best old one H5 0.335).

**Demo artifacts produced:** `artifacts/s5_v3_r3_recovery_heuristics.csv`, `s5_v3_r3_recovery_table.txt`, `s5_v3_r3_bands.csv`, `s5_v3_recovery_json/` (now holding all nine), `s5_r3_score_runs.txt`, `s5_r3_trace_check.txt`, `s5_r3_mlflow_sha.txt`.

### R3 steps 2 and 5: S3 boundary re-run and the supersession wording
**Files:** `vlapoc/lineage.py` (`assert_blind`: a version whose fence carries `superseded_by` and that has no offending columns is refused with "version v2 (...) is fenced: superseded by v3 (002a68e4caf2); use the superseding version"; every other fence or leak keeps the old wording, so v1 still says `fenced=True, offending=['episode_index']`), `scripts/verify_method_boundary_s3.py` (the refusal table gains a `v2/candidate` row and asserts the new wording).

**Acceptance criteria** (`.venv/bin/python scripts/verify_method_boundary_s3.py`, 48 s; `artifacts/s3_r3_run.txt`, refreshed `artifacts/s3_verify.json`; the r1 run text is `s3_run.txt`):
- ✅ AC3.1: `v1/candidate` refused (`fenced=True`), `v2/candidate` refused with "is fenced: superseded by v3", `v0/candidate` opens (960 episodes, 40 reference ids), the scratch derived version opens (its 7-edit derivation now includes the trimmed `time_shift`; child id differs from the committed round-1 one, as the analyzer noted), a one-byte change raises `VersionIntegrityError`.
- ✅ AC3.2, AC3.3, AC3.5: unchanged results (oracle and oracle-chain subsets refused; diff guard in both orders; catalog columns documented and `check_blind(scratch child) = []`).
- ✅ AC3.4: the AST scan now covers `vlapoc/method_data.py`, `vlapoc/methods/__init__.py` and `vlapoc/methods/heuristics.py` (with H7 and H8): 0 violations; all six planted violations are still detected.
- Real store: this script only reads it; `mlflow.db` stays `306af4123fda7e63`.

### R3 step 3: runbook detectability paragraph and GT-R6
**Files:** `artifacts/server-runbook.md` (GT-R6 and the limitation line in "What cannot be shown here").
- GT-R6 now states the comparison rule of plan AC5.9: compare per type against that type's best heuristic, not against chance; the table to beat (type, best of H1-H8, AUROC): `action_noise` H4 1.000, `dropped_frames` H5 1.000, `idle_padding` H2 1.000, `instruction_swap` H6 1.000, `gripper_release` H8 0.964, `detour` H7 0.960, `time_shift` H7 shortest-first 0.848; pooled recall is not comparable (best single heuristic 0.335). It says the earlier "no heuristic finds `gripper_release` / `time_shift`" statement was an artifact of H1-H6 lacking task normalisation, and records the limitation as owner-approved under ruling A (with B and C).
- ✅ AC6.6 still holds (v3 in every GT-R command; known-issue section unchanged).

### R3 step 4: R12, the stale data+lineage AC4.8 contrast figure
**Files (data+lineage feature, per vla-build step 5):** `tracker/features/2026-09-28-data-lineage-layer/changelog.md` (new dated entry 2026-10-17), `report.html` (the round-2 analyzer-table sentence that said the shared cache "used to gain 60 files / 663 MB" now says that was the round-1 writer's behaviour, corrected to 0 files; plus a dated line in "Changes since first release"), `S4-derived-versions-labels.html` (AC4.8 row annotated), `artifacts/s4_summary.json` (an `AC4.8_note_2026-10-17` key; the old `contrast_unisolated_*` values are kept as the historical record).
- Evidence quoted in the entry: `verify_derive_s4.py --cache` leaves `0 cache files (0 .arrow, 0 parquet builder entries), 0 bytes` on the current code and on the pre-S2 code (`artifacts/s2_regress_verify_derive_s4.txt`), so the cause is the order-preserving writer of 2026-10-02, not a ground-truth change; the isolation guarantee itself still holds.
- The `report.html` figure the analyzer cited (line 1888) was the only report occurrence; the S4 page row was the other.
