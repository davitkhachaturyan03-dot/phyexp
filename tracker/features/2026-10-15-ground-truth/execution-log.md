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
