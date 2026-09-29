# Execution log: data layer + lineage layer

## Round 1: 2026-09-28

Batch 1 of 2 (S1, S2, S3). S4–S6 come in batch 2. Executor: vla-executor. Evidence labels as in
CLAUDE.md: **measured** means a command was run and its real output is quoted below (all captured
stdout is under `artifacts/`). Wall-clock note: the session ran from 2026-09-28 ~23:55 to
2026-09-29 ~10:00 local time (+04). The machine clock shows a gap of about 9.5 h between the catalog
build (00:20) and version registration (09:51), probably the machine sleeping. Timestamps in
records are genuine.

Spec pages re-read from the PDF this session (pypdf dump): p.1–5 (Goal, Key takeaways, §1, §2.1–2.7),
p.7–8 (§3.5), p.14–15 (§8.1–8.7), p.18–19 (§9.1 Step 4, §9.2), p.20 (§10). RoboDrop App. A.1 read from
`docs/papers/RoboDrop_Gradient_Compatibility_arXiv-2609.10021.pdf` p.13.

### Setup: git (owner answer BLOCKING Q2)
**Files:** `.gitignore` (+32 lines: `data/`, `.venv/`, `__pycache__/`, `outputs/`, `checkpoints/`,
`mlruns/`, `mlartifacts/`, `mlflow.db`, `wandb/`, `*.ckpt`, `*.safetensors`, `docs/papers/`, `.idea/`,
`.claude/settings.local.json`, `.DS_Store`).
- Repo root before init: `.claude/ .idea/ .venv/ CLAUDE.md data/ docs/ main.py tracker/` (measured, `ls -la`).
  `docs/papers/` = 94 MB of third-party PDFs, `docs/VLA_Data_Attribution_PoC.pdf` = 131 KB.
- `git init -b main` → `Initialized empty Git repository in …/vlaExp/.git/`; `git remote -v` → 0 remotes.
- Local commit (source + tracker record + small artifacts): `git rev-parse HEAD` →
  **`0fdb923209aff09902634ffc818c5148536d0a3f`**. `git ls-files | grep -cE '^(data|\.venv|docs/papers|\.idea)/'` → `0`.
  Largest tracked file: `artifacts/s1_orientation.png`, 412 KB. No push and no remote.
- Decisions to note: `docs/papers/` is ignored because it holds large, re-downloadable third-party
  binaries (the spec PDF is versioned). `.idea/` and `.claude/settings.local.json` are ignored as
  machine-local. The owner can reverse any of these by editing `.gitignore`.

### S1: Convert raw LIBERO (libero_spatial + libero_10) into a LeRobot v3 dataset, unfiltered
**Files:**
- `vlapoc/__init__.py` (+6, package docstring with grounding).
- `vlapoc/libero_convert.py` (+277, of which 59 are docstring lines by `ast` count, plus the 20-name pinned task order).
  - `LIBERO_TASK_ORDER` holds the LIBERO benchmark order.
  - `FEATURES` holds the LeRobot LIBERO schema.
  - `load_raw_episode()` is the single raw-to-canonical mapping: rotate 180° by flipping H and W, cast
    to float32, and state = `ee_pos ‖ ee_ori ‖ gripper_states`.
  - `convert()` uses LeRobot's own `LeRobotDataset.create(fps=20, use_videos=False, robot_type="panda",
    image_writer_threads=4)` → `add_frame` → `save_episode` → `finalize`.
- `scripts/verify_libero_convert.py` (+205, AC1.1–1.6 evidence + `artifacts/s1_summary.json`).
- `scripts/orientation_figure.py` (+105, AC1.4 figure + mean-abs-diff numbers).
- `artifacts/reference/` holds pinned copies of the reference info.json files, the LIBERO task map and
  the hub reference frames.

**Premises checked:**
- LeRobot 0.6.1 writer API (`create(repo_id, fps, features, root, robot_type, use_videos,
  image_writer_threads, …)`; root "Must not already exist"; image mode writes PNG, compress_level 6,
  then `embed_images` into parquet). Checked in `lerobot/datasets/lerobot_dataset.py:684-790`,
  `dataset_writer.py:202-270,441-505`, `dataset_metadata.py:813,832`. **Holds.**
- Camera mapping `agentview→image`, `eye_in_hand→image2`, from `lerobot/envs/libero.py:156-159`
  `{"agentview_image": "image", "robot0_eye_in_hand_image": "image2"}`. **Holds.**
- State layout, from `lerobot/processor/env_processor.py:75` `torch.cat((eef_pos, eef_axisangle,
  gripper_qpos))`, with the 180° flip at `:45,59`. Raw `obs/ee_pos == robot_states[:,2:5]` and
  `obs/gripper_states == robot_states[:,0:2]` hold for 9 of 9 checked demos (3 files × demo_0/17/49,
  `np.array_equal` → True). The planner already measured `ee_ori` as axis-angle. **Holds.**
- `fps: int = 20  # Must match robosuite's default control_freq (20 Hz)` at
  `lerobot/envs/configs.py:325`; raw `env_args.control_freq` = `[20]` for all 1000 demos (catalog
  check `control_freq_values: [20]`). **Holds.**
- LIBERO task order: `libero_suite_task_map.py` was last changed in LIBERO commit `4f0dd188` ("init",
  2023-06-07, GitHub commits API) and is identical at master `8f1084e3` (fetched; copy in
  `artifacts/reference/`). **Holds**, and it matches the order in the plan.
- Reference schemas at the pinned revisions. The full SHAs resolved via `HfApi.list_repo_commits`:
  `lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` and
  `HuggingFaceVLA/libero@86958911c0f959db2bbbdb107eb3e17c5f9c798e`. The **premise is partly false as
  worded in AC1.2**: `lerobot/libero` stores images as dtype `video` (AV1), so a lossless conversion
  (`use_videos=False`, the plan's own AC1.6 decision) necessarily has dtype `image` there as well. See
  AC1.2 and the findings.
- Demo keys are contiguous `demo_0..demo_49` in every file: `True` (verify script). **Holds.**

**Acceptance criteria:**
- ✅ AC1.1: `.venv/bin/python scripts/verify_libero_convert.py` →
  `AC1.1 info.json: {'codebase_version': 'v3.0', 'fps': 20, 'total_episodes': 1000, 'total_frames': 200340, 'total_tasks': 20, 'robot_type': 'panda'}`
  and `AC1.1 LeRobotDataset loads: num_episodes=1000 num_frames=200340`.
- ⚠️ AC1.2 (met against the PNG variant; the literal wording does not hold against `lerobot/libero`).
  Evidence from the same script:
  ```
  AC1.2 feature diff vs lerobot/libero@a1aaacb7 (fps ours=20 ref=10.0):
      observation.images.image.dtype: ours=image ref=video
      observation.images.image.shape: ours=[128, 128, 3] ref=[256, 256, 3]
      observation.images.image2.dtype: ours=image ref=video
      observation.images.image2.shape: ours=[128, 128, 3] ref=[256, 256, 3]
  AC1.2 feature diff vs HuggingFaceVLA/libero@86958911 (fps ours=20 ref=10.0):
      observation.images.image.shape: ours=[128, 128, 3] ref=[256, 256, 3]
      observation.images.image2.shape: ours=[128, 128, 3] ref=[256, 256, 3]
  AC1.2 LeRobotDataset(episodes=[0, 999]): 393 frames; item[0] shapes {'observation.images.image': [3, 128, 128], 'observation.images.image2': [3, 128, 128], 'observation.state': [8], 'action': [7], ...}
  ```
  - Keys, names and the state/action dtypes and shapes equal both references. Against
    `HuggingFaceVLA/libero` (the PNG variant) the only differences are image shape and fps, exactly
    as AC1.2 states. Against `lerobot/libero` the image dtype also differs (`image` vs `video`).
  - That difference follows from the plan's lossless decision, so it is not a defect of the build.
    It is logged under findings for the planner to reword AC1.2.
- ✅ AC1.3: the same script, every frame of 6 episodes (indices 0, 250, 499 in libero_spatial and 500,
  750, 999 in libero_10), read through `LeRobotDataset(..., episodes=[e])[j]`. Each returned
  `'state_equal': True, 'action_equal': True` against `concat(ee_pos, ee_ori, gripper_states).astype(float32)`
  and `actions.astype(float32)` read directly from the HDF5, e.g.
  `episode_index 999: {'episode': 'libero_10/KITCHEN_SCENE6_…/demo_49', 'frames': 295, 'state_equal': True, 'action_equal': True, ...}`.
- ✅ AC1.4: the same 6 episodes, every frame gave `'image_eq_raw_rot180': True, 'image_eq_raw_unrotated': False, 'image2_eq_raw_rot180': True`.
  The comparison uses `np.flip(raw, axis=(1,2))`, computed independently of the converter.
  - Figure `artifacts/s1_orientation.png` shows raw HDF5 | ours | `HuggingFaceVLA/libero` for two tasks
    and both cameras. Hub frames are frame 0 of hub episodes 1272 and 6, fetched as JPEG previews from
    datasets-server rows API offsets 221834 and 1646.
  - Visual check: our frames share the hub layout (robot base at top, same object arrangement) and
    raw is upside-down.
  - `scripts/orientation_figure.py` gives the mean absolute pixel difference to the hub frame for ours
    vs unrotated raw: agentview 7.44 vs 71.22 (spatial task 0) and 11.30 vs 102.49 (KITCHEN_SCENE3);
    wrist 13.96 vs 75.48 and 27.58 vs 74.02. The hub frame is a different demo re-rendered at 256²,
    so it is not expected to be 0.
- ✅ AC1.5: `AC1.5 per-task episode counts: 20 tasks, counts=[50]; demo keys contiguous demo_0..demo_49 in every file: True`;
  `AC1.5 episode length == raw num_samples: 1000/1000; task string == raw instruction: 1000/1000`;
  `AC1.5 raw rewards: exactly one reward==1, at the last step: 1000/1000; dones likewise: 1000/1000`.
- ✅ AC1.6: `use_videos=False`, image dtype `image`, no `videos/` dir.
  - Real conversion (`/usr/bin/time -l .venv/bin/python -m vlapoc.libero_convert --out data/lerobot/libero_spatial_10 --threads 4`):
    `{"episodes": 1000, "frames": 200340, "wall_time_s": 468.5, ...}` and `471.25 real  652.73 user  80.69 sys`,
    `peak memory footprint 764970688` (0.76 GB). Throughput 428 frames/s.
  - Log start `2026-09-28T20:09:34Z`, end `2026-09-28T20:17:25Z`, `exit=0`.
  - On disk: `AC1.6 on-disk: du=7.44 GiB, sum(file sizes)=7.356 GB (36719 B/frame), data parquet files=73`.
  - The planner projected about 18 min and 8.0 GB. Real: **7.9 min wall, 7.36 GB (7.44 GiB by du)**.
  - Smoke comparison over 20 episodes (4,190 frames): 449 frames/s with 4 writer threads vs 296 with 0,
    which is why 4 threads were used.
- ✅ AC1.7: the `vlapoc/libero_convert.py` module docstring cites PoC §10 p.20, §2.2 p.4 and §2.1 item 2
  p.3, plus the LeRobot LIBERO schema sources (hub info.json at pinned SHAs, `envs/libero.py`,
  `processor/env_processor.py`, `envs/configs.py`). It lists 4 divergences from the hub datasets: all
  1000 demos, 128², fps 20, and lossless PNG instead of AV1.

**Demo artifacts produced:** `artifacts/s1_convert_stdout.txt` (tqdm "Map:" progress bars stripped,
everything else verbatim), `artifacts/s1_verify_stdout.txt`, `artifacts/s1_summary.json`,
`artifacts/s1_orientation.png`, `artifacts/s1_orientation_stats.json`, `artifacts/s1_orientation_stdout.txt`,
`artifacts/reference/*` (pinned reference info.json ×2, LIBERO task map, hub frames ×4 + row provenance).
Dataset: `data/lerobot/libero_spatial_10/` (gitignored).

**Notes / divergences from plan or spec:**
- LeRobot's writer leaves an empty `images/observation.images.image{,2}/` staging tree behind (0 files,
  measured). I removed it from this dataset with `find … -type d -empty -delete`. I then added the
  same cleanup to `convert()` after `finalize()`, removing the tree only if it holds no files. So the
  committed converter differs from the code that ran by those 4 lines.
- The PyAV/Homebrew `libavdevice` duplicate-class warning the plan flagged appears in the log, and the
  run finished with no crash.
- Line count: the plan estimated a converter of about 100 lines. Mine is 277: 59 docstring lines, the
  20-name task table, the schema dict, the CLI, and the loader shared with S2.

### S2: Give every episode a stable ID, a content hash and a §2.2 catalog row
**Files:** `vlapoc/catalog.py` (+411, of which 79 are docstring lines). Its functions:
- `episode_content_hash()`: domain-separated sha256 over task, frame count, then per frame f32 action,
  f32 state, uint8 image and uint8 image2.
- `hash_stored_dataset()`: decodes the stored PNGs from parquet, 8 processes.
- `scan_raw()`: re-hashes raw data through `load_raw_episode`, plus metadata and reward checks.
- `raw_file_hashes()`, `action_stats()`, `scene_of()`, `build_catalog()`, `write_catalog()`, and a CLI.

Output: `data/lineage/catalog/libero_spatial_10/{episodes.parquet, raw_files.parquet, catalog.json}`.

**Premises checked:**
- Gripper action values are exactly ±1, so toggles need no threshold. Scanning all 1000 demos gives
  `libero_10 unique gripper action values: [-1.0, 1.0]` and `libero_spatial … [-1.0, 1.0]`. **Holds.**
- `scene`: libero_10 task names carry `<ROOM>_SCENE<n>`. libero_spatial names do not; its raw
  `env_args.problem_name` = `libero_tabletop_manipulation`, while libero_10's is
  `libero_kitchen_tabletop_manipulation` etc. So `scene` = the prefix where present, else
  `problem_name` (documented in the docstring). **Resolved, see findings.**
- `objects`: model-XML `*_main` bodies. For example, spatial task 0 gives `akita_black_bowl_1, akita_black_bowl_2,
  cookies_1, glazed_rim_porcelain_ramekin_1, plate_1, wooden_cabinet_1, flat_stove_1`. The plan
  expected robot, mount and table to be excluded, and they are not `*_main` bodies. The list includes
  **distractors** (LIVING_ROOM_SCENE2 lists ketchup, orange_juice and milk) and is constant within
  each task (`True`). **Holds**, with the docstring corrected to say "every object in the scene".
- `init_state == states[0]` for 1000/1000 demos (catalog check). **Holds.**
- File-level vs content-level hashing. Two conversions of the same 20 demos (smoke, threads 4 vs 0)
  wrote **different parquet bytes but identical tables**. The measured reasons:
  - `pq.read_table(...).equals(...)` → `True`, and all image bytes are equal.
  - Only the schema metadata differs: the HF `datasets` `"fingerprint": "e863380b6dfc304e"` vs `"09caca26c6c318b8"`.
  - `meta/stats.json` has identical values but a different key order.

  This supports hashing decoded content rather than files. See findings.

**Acceptance criteria:**
- ✅ AC2.1: `episode_id = "<suite>/<task_name>/demo_<k>"`, built from the raw path and demo key,
  e.g. `libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate/demo_0`.
  - `.venv/bin/python -m vlapoc.catalog --root data/lerobot/libero_spatial_10 --verify-raw` → `unique episode_id: 1000/1000`.
  - `source_file` (repo-relative, e.g. `data/raw/libero/libero_spatial/…_demo.hdf5`) and `source_demo`
    (`demo_0`) are columns of `episodes.parquet` (schema printed in `artifacts/s2_catalog_inspect_stdout.txt`).
- ✅ AC2.2: the same command → `1000/1000 hashes match raw`. The stored side decodes the PNGs from the
  LeRobot parquet and reads the task via `task_index`. The raw side goes through `load_raw_episode`.
  - Sensitivity (`artifacts/s2_hash_sensitivity_stdout.txt`): a recompute equals the catalog hash
    `3dd60ba6…`. A 1-pixel LSB flip, a 1-ulp change of one action value, one space appended to the task,
    and dropping the last frame each change the hash (`True` ×4).
  - The plan's exact command `.venv/bin/python -m vlapoc.catalog --version v0 --verify-raw` (runnable
    after S3) → `1000/1000 hashes match raw`, exit 0. The rebuilt catalog is identical to the first
    build (`True`).
- ✅ AC2.3: `raw files: 20 sha256 recorded; 19.97 GB`, in `raw_files.parquet` (source_file, sha256,
  size_bytes) and `artifacts/s2_raw_files_sha256.csv`. Copied into the v0 version dir by S3.
- ✅ AC2.4: `episodes.parquet` schema (from `artifacts/s2_catalog_inspect_stdout.txt`):
  - Columns: `episode_id, episode_index, content_hash, suite, task_name, libero_task_id,
    language_instruction, scene, objects: list<string>, source, operator: null, collection_date: null,
    length_steps, duration_s, init_state_sha256, smoothness, idle_fraction, idle_steps,
    gripper_toggles, source_file, source_demo`. That is the AC list plus `idle_steps`, the integer
    count behind `idle_fraction`, which AC2.5 needs.
  - `libero_task_id per suite: {'libero_spatial': [0..9], 'libero_10': [0..9]}`, in the LIBERO map order
    pinned in the `libero_convert` docstring (commit `4f0dd188`).
  - `operator all null: True | collection_date all null: True`.
  - `source` = "LIBERO human teleoperation (3Dconnexion SpaceMouse), 50 demos per task" (LIBERO p.6).
  - `init_state_sha256 unique: 1000`.
- ✅ AC2.5: the per-suite summary from the build matches every planning measurement.
  - libero_spatial: `duration_s [3.75, 6.2, 9.85]` (min, median, max), `idle_steps_total 97`,
    `episodes_with_idle 28`, `gripper_toggles [1, 7]`.
  - libero_10: `duration_s [7.5, 13.15, 25.85]`, `idle_steps_total 339`, `episodes_with_idle 85`,
    `gripper_toggles [1, 8]`.
- ✅ AC2.6: checks from the build: `reward_sum_eq_1: 1000`, `first_success_is_last_step: 1000`,
  `dones_last_only: 1000`, and `duration_s == length_steps/20: True`. The docstring defines
  `duration_s` as the human time-to-success T used as the HRT reference, citing PoC §3.5 pp.7–8 and its
  caveat about differing initial states. The per-task table is in `artifacts/s2_human_duration_per_task.csv`,
  e.g. KITCHEN_SCENE8 has median 20.42 s (max 25.85) and spatial task 0 has median 4.75 s.
- ✅ AC2.7: the `vlapoc/catalog.py` docstring cites §2.1 p.3, §2.2 p.4, §3.5 pp.7–8 and §1 p.3. A
  section "Catalog columns that LIBERO cannot supply" covers operator, collection_date, scene and
  objects, with reasons and the LIBERO p.6 quote. Identity cites CUPID's `demo_{i}` keys and LeRobot's
  reindexing (`_copy_and_reindex_data`).

**Demo artifacts produced:** `artifacts/s2_catalog_stdout.txt`, `artifacts/s2_catalog_version_v0_stdout.txt`,
`artifacts/s2_catalog_inspect_stdout.txt` (schema + 5 transposed example rows + per-task durations + raw
sha256s), `artifacts/s2_catalog_excerpt_first20.csv`, `artifacts/s2_catalog_excerpt_spot.csv` (episode
indices 0, 1, 250, 499, 500, 750, 999), `artifacts/s2_human_duration_per_task.csv`,
`artifacts/s2_raw_files_sha256.csv`, `artifacts/s2_hash_sensitivity_stdout.txt`.
The full catalog is `data/lineage/catalog/libero_spatial_10/episodes.parquet` (gitignored, 180 KB).

**Notes / divergences from plan or spec:**
- The catalog build takes 30 s (hashing 200,340 frames from parquet and from raw, plus sha256 of
  19.97 GB of raw files, 8 processes): `30.33 real`.
- The raw re-hash reuses the converter's `load_raw_episode`. It therefore proves storage fidelity
  (write then read is bit-exact for all 1000 episodes). The mapping itself (rotation, cast, state
  layout) is proven independently, against direct HDF5 reads, only for the 6 AC1.3/1.4 episodes, every
  frame.
- Not tested: hash determinism on another OS/CPU (x86 Linux). Little-endian float32 and decoded pixels
  are platform-independent by construction (inferred). The first run on the Linux GPU box should run
  `python -m vlapoc.lineage version recompute v0`.

### S3: Define dataset versions, named splits and subsets that resolve to LeRobot `episodes=`
**Files:** `vlapoc/lineage.py` (+534, S3 part only; 79 docstring lines). It contains:
- `canonical_json`/`sha256_json`, `schema_of`, `version_id` and `file_digests`.
- `register_version`, which writes `versions/<id>/{version.json, episodes.parquet, files.json,
  raw_files.parquet}`, binds `names.json` and creates `<name>/all`.
- `resolve_version`, `version_episodes`, `recompute_version_id` and `verify_version` (raises
  `VersionIntegrityError`).
- The `Subset` dataclass (`subset_id`, `resolve()`, `keys()`, `save()`, `load()`), `make_splits`,
  `drop_top_k` (strict key check, digest, tie-break), `drop_random_k` and `score_table_digest`.
- A CLI: `version register|show|verify|recompute`, `split make`, `subset show`, `names`.

`scripts/verify_lineage_s3.py` (+183) covers AC3.1 and AC3.3–3.6.

**Premises checked:**
- `LeRobotDataset(episodes=…)` serves exactly the listed episodes. `io_utils.load_nested_dataset` uses
  `filters = pa_ds.field("episode_index").isin(episodes)`, and `dataset_reader._build_index_mapping`
  maps absolute to relative indices, so `len(ds) == len(hf_dataset)`. Measured below. **Holds.**
- Each episode lies in exactly one data file, which the stored-side hashing assumes.
  `hash_stored_dataset` asserts this and did not raise across 1000 episodes and 73 files. **Holds.**
- APFS clone for the tamper test: `cp -c -R` completed in 0.2 s and the original still verifies after
  the clone was modified. **Holds.**

**Acceptance criteria:**
- ✅ AC3.1: `python -m vlapoc.lineage version register --root data/lerobot/libero_spatial_10 --name v0`
  → `"dataset_version_id": "3c4dca68d6f196c10d249b43bd94ae32f51b905735f25c7865e3c943caf553b7"`,
  `"lerobot_root": "data/lerobot/libero_spatial_10"`, `"n_episodes": 1000`, `"n_frames": 200340`.
  - `scripts/verify_lineage_s3.py`:
    `AC3.1 recomputed from dataset = 3c4dca68…53b7 match=True (19s)`;
    `version dir contents: ['episodes.parquet', 'files.json', 'raw_files.parquet', 'version.json']; files.json entries=77 (data parquet=73)`.
  - The id is sha256 over canonical JSON of {scheme, episode_hash_scheme, schema (fps + every feature's
    dtype/shape/names), sorted [episode_id, content_hash]}.
- ✅ AC3.2: `python -m vlapoc.lineage split make --version v0 --reference-per-task 2 --seed 0` →
  `"sizes": {"candidate": 960, "reference": 40}`, `split_id 6c3c195f…`, reference subset `b6a72d5e…`,
  candidate `8ee096b7…`, with the rule and seed persisted in `data/lineage/splits/<split_id>.json` and
  in each subset's derivation. The verify script gives `reference=40 candidate=960 overlap=0
  union==version: True; reference per task: [2] over 20 tasks`.
- ✅ AC3.3: both subsets served through LeRobot are exact.
  - `v0/reference`: `resolve()=40 indices; served episodes == subset: True; frames served=7915 sum(length_steps)=7915 (every frame via __getitem__) -> OK`.
  - `v0/candidate`: `resolve()=960 indices; served episodes == subset: True; frames served=192425 sum(length_steps)=192425 -> OK`.
    For the candidate, the served set is read from `ds.hf_dataset`'s `episode_index` column, which is
    what `__getitem__` indexes. Per-frame `__getitem__` was done in full for the reference subset.
- ✅ AC3.4: `drop_top_k(cand, scores, "score", k=20, "highest")`, with the real catalog column
  `smoothness` as a stand-in score, labelled in `score_source`.
  - Result: `child c68eacb518e1 n=940 (parent 960 - 20); dropped == 20 highest scores: True`.
  - Derivation keys: `['direction', 'dropped', 'k', 'metric', 'op', 'parent_subset_id', 'score_source', 'score_table_digest', 'tie_break']`.
  - `drop_random_k(seed=0) twice -> same subset_id: True (4a06edb41e77); seed=1 differs: True`, with
    derivation keys `['dropped', 'k', 'numpy_version', 'op', 'parent_subset_id', 'rng', 'seed']`.
  - A score table carrying one wrong `content_hash` is rejected with `LineageError: score table keys != parent subset's episode versions: 1 unknown/stale …`.
  - The drop-top-k child round-trips through LeRobot: `True, frames=190269`.
- ✅ AC3.5: `AC3.5 untouched clone verified first: True (77 files)`.
  - After flipping 1 bit at byte 50660612 of `data/chunk-000/file-017.parquet` in the clone:
    `VersionIntegrityError: dataset copy at <scratch>/v0_tamper_copy does not match version v0 (3c4dca68d6f1): changed=['data/chunk-000/file-017.parquet'] missing=[] extra=[]`.
  - `original still verifies: True`.
- ✅ AC3.6: `AC3.6 scanned 15 lineage files (json + parquet) under data/lineage/: absolute-path hits=[]`.
  `rel()` raises `LineageError` if asked to record a path outside the repository.
- ✅ AC3.7: the `vlapoc/lineage.py` module docstring cites PoC §2.1 p.3, §8.3 p.14, §8.7 p.15, §2.6 p.5,
  §9.1 Step 4 p.19 and RoboDrop App. A.1 p.13 (quoted), plus the LeRobot `load_nested_dataset`
  predicate-pushdown source.

**Demo artifacts produced:** `artifacts/s3_version_register_stdout.txt`, `artifacts/s3_version_show_v0_stdout.txt`,
`artifacts/s3_split_make_stdout.txt`, `artifacts/s3_names_stdout.txt`, `artifacts/s3_subset_reference_stdout.txt`
(the 40 reference episode IDs), `artifacts/s3_verify_stdout.txt`, `artifacts/s3_summary.json`.
Lineage store: `data/lineage/` (944 KB, gitignored). `names.json` = `{"versions": {"v0": "3c4dca68…"},
"subsets": {"v0/all": "58b3c415…", "v0/candidate": "8ee096b7…", "v0/reference": "b6a72d5e…"}}`.

**Notes / divergences from plan or spec:**
- The verify script left three **unnamed** demo subsets in `data/lineage/subsets/`: the drop-top-k by
  `smoothness` child `c68eacb5…` and the random-k children `4a06edb4…` (seed 0) and `3ab14b12…`
  (seed 1). Each derivation says what it is (`score_source` names the catalog column). They are not
  attribution results. S6 can delete them or leave them.
- `register_version` also registers `<name>/all` (the version's full episode set) as a subset, so
  that split derivations have a parent subset id. The plan did not list this. It adds no data.
- `drop_top_k` requires the score table to key **exactly** the parent's episode versions: no stale
  hashes, no unscored parent episodes, no extras. That is stricter than the plan wording and is the
  same contract AC5.5 `check_scores` will enforce.
- Line count: the plan estimated a lineage module of about 300 lines for S3–S5. The S3 part alone is
  534 including docstrings and CLI.

### Findings for the analyzer
1. **AC1.2 wording conflicts with the plan's own lossless decision (planner).** `lerobot/libero` stores
   images as dtype `video` (AV1). A `use_videos=False` dataset necessarily has dtype `image`, the same
   as the PNG variant `HuggingFaceVLA/libero`. Our schema equals `HuggingFaceVLA/libero`'s except image
   shape and fps. Suggested rewording: "equal `HuggingFaceVLA/libero`@86958911; vs `lerobot/libero` the
   image dtype is also `image` rather than `video` (lossless storage)". There is no training impact:
   LeRobot decodes both into the same tensors (inferred from `hf_transform_to_torch` and the
   video-decoding path; not tested with a policy here).
2. **LeRobot data files are not byte-reproducible (measured, new information).** Two conversions of the
   same demos give equal tables and image bytes but different parquet sha256s. The cause is the random
   HF `datasets` `fingerprint` in each file's schema metadata, plus a different key order in
   `meta/stats.json`. This supports the plan's content-hash design: DVC-style file md5 would report a
   changed dataset on every rebuild, while `dataset_version_id` reproduces (AC3.1 `match=True`). It also
   means per-file digests (`files.json`) identify one physical copy, not the content. `verify_version`
   uses them only for tamper detection, which is how the plan intended them.
3. **Planner projections vs real (measured).** Conversion took 7.9 min wall (the plan said about 18) and
   7.36 GB (the plan said about 8.0). Catalog build plus raw re-hash takes 30 s. Version recompute takes
   19 s. `python -m vlapoc.lineage version verify v0` (sha256 of all 77 files, 7.36 GB) takes
   `2.59 real`.
4. **`scene` for libero_spatial (spec ambiguity, resolved).** LIBERO gives libero_spatial no scene
   name. I used the raw `env_args.problem_name` (`libero_tabletop_manipulation`, one value for all 10
   spatial tasks) rather than inventing a label. For aggregation by scene, libero_spatial is one group;
   `task_name`/`libero_task_id` distinguish its layouts.
5. **`objects` includes distractors and fixtures (resolved, documented).** The model XML's `*_main`
   bodies include objects the task never touches. §2.2 says "objects" without qualification, so all
   scene objects are listed. A task-relevant-objects column would need BDDL `(:obj_of_interest …)`
   parsing, which is out of scope and not built.
6. **Normalization-stats risk (plan's Risks list), still open, and relevant to S5 and S6.** v0's
   `meta/stats.json` covers all 1000 episodes, including the 40 reference episodes. A train run on the
   960-episode candidate subset would therefore normalize with statistics that include held-out
   episodes. S5 records the stats hash, as the plan says. The choice itself belongs to the training
   feature.
7. **Git decisions the owner may want to review:** `docs/papers/` (94 MB) and `.idea/` are gitignored.
   The commit author is set per-commit (`-c user.name/-c user.email`); no global or repo git config
   was changed.

---

### Batch 2 (S4, S5, S6): 2026-09-29

Re-read before starting: `brief.md`, including the new owner answer (perturbation scope: the §2.5/§8.10
corruption suite is a separate later feature; S4 ships the mechanism plus the one RoboDrop
action-noise demo). PoC pages re-read from the PDF: p.5 (§2.5, §2.6), p.15 (§8.7), p.17–18 (§8.10),
p.18–19 (§9.1 Steps 1–4, §9.2).

**Commits (local, no remote; nothing under `data/` or `.venv/`).** Every recorded run's
`mlflow.source.git.commit` is a commit whose `vlapoc/` and `scripts/` match the tree that ran:
`stage_run` refuses to start while those paths differ from HEAD.
- `5d9d272`: S4–S6 code. The build-v0 run and the FAILED split attempt were recorded at this commit.
- `398e037`: fix, dataset source passed as a string; only FINISHED runs form lineage edges. The
  split-v0 and derive-v1 runs were recorded here.
- `9a754f3`: fix, LeRobot needs a fresh `output_dir`. S5 scratch runs were recorded here.
- `7effd25`: docstring note on the HF cache. All 11 S6 chain runs were recorded here.
- `67bbd41`: Mermaid label fix. No runs were recorded at this commit; the graph was regenerated from recorded runs.

**Dependencies added (both named with reasons):**
- `mlflow==3.16.1` (owner answer Q1). Measured:
  - `pip install --dry-run mlflow` resolved with wheels on Python 3.14, so the plan's
    mlflow-skinny fallback was not needed.
  - The install took `38.79 real`. Packages went from 69 to 128; none were removed or upgraded (a
    diff of `pip list` before and after showed only additions). It pulls SQLAlchemy 2.1.1 and
    alembic 1.20.0.
  - The full package was chosen because the S6 verify step uses `mlflow ui`.
  - On import, mlflow prints an "agent hint" suggesting that a bundled tracing skill be loaded. That
    is tool output, not an instruction. This feature does not use tracing, so the hint was ignored
    and silenced with its documented env var `MLFLOW_DISABLE_AGENT_HINT=1`.
- `accelerate==1.15.0` (+ `psutil==7.2.2`). **Not in the plan: a premise gap.** LeRobot's `train()`
  calls `require_package("accelerate", extra="training")` and raised
  `ImportError: 'accelerate' is required but not installed`. The `lerobot[training]` extra would also
  add wandb and sentry-sdk. Only accelerate is needed with wandb off, so only accelerate was
  installed, within lerobot's pin `accelerate<2.0.0,>=1.14.0`.

**Housekeeping requested by the orchestrator:** the three unnamed S3 demo subsets (`c68eacb5…`
drop-top-k by `smoothness`, `4a06edb4…` / `3ab14b12…` random-k seeds 0/1) were moved from
`data/lineage/subsets/` to `data/lineage/quarantine/s3_verify_demo_subsets/`, with a `README.json`
saying what they are, why they were moved, and that `scripts/verify_lineage_s3.py` re-creates them.
No run ever consumed them.

### S4: Derive new dataset versions with ground-truth labels (the injection hook)
**Files:**
- `vlapoc/derive.py` (+429). Functions:
  - `derive_version()`: the mechanism, then content re-verification, catalog, registration, labels,
    `derivation.json` and split inheritance.
  - `protected_episode_ids()`, which walks the ancestor chain of reference splits, and `ancestry()`.
  - `inherit_splits()`, `read_stored_episode()` and `select_per_task()`.
  - `robodrop_action_noise()`, `idle_padding_edit()` (the AC4.4 test edit) and `derive_demo_action_noise()`.
- `vlapoc/lineage.py` (+`diff_versions`, CLI `derive`, `version diff`).
- `scripts/verify_derive_s4.py` (+195).

**Premises checked:**
- LeRobot 0.6.1 v3 edit tools (plan risk: upstream issue #2283 crash). Tested on the 20-episode
  smoke dataset with `delete_episodes([3, 7])`, then the writer for 2 edited episodes (one padded by
  20 frames), then `merge_datasets`. Output: `delete ok`, `merge ok 20 4210`; `kept hashes unchanged
  in order: True`; `edited appended: ep18 len 119 orig 99`; `meta lengths == stored lengths: True
  from/to contiguous: True`. **Holds: no crash on 0.6.1.** The plan's first-choice route (tools)
  was used, and the full-rewrite fallback was not needed.
- `delete_episodes` copies parquet rows without re-encoding PNGs (`dataset_tools.py:513-608`
  `_copy_and_reindex_data`). **Holds.**
- σ_a: v0 `meta/stats.json` `action.std` equals a recomputation over all 200,340 frames (max abs diff
  6.8e-9). **Holds.** σ_a = [0.3334, 0.3531, 0.4134, 0.0393, 0.0612, 0.0811, 0.9990].

**Acceptance criteria:**
- ✅ AC4.1: the v1 derivation, run as the plan's verify-by command:
  `python -m vlapoc.lineage derive --parent v0 --demo-action-noise --per-task 5 --seed 0`.
  - It produced `"dataset_version_id": "16076b9c5cb2a0e8c5d291fc0766977dc7286c6b9f87df39bd87fe5c17cd85f8"`,
    `"derive_run_id": "30f34cd056ab44e69dcf606f067d8871"`, `"n_edits": 100` and `"n_frames": 200340`.
    Timing: `delete 166.0, write_edits 55.4, merge 74.6, total 316.8` s.
  - `python -m vlapoc.lineage version diff v0 v1` → **`changed 100, identical 900, added 0, removed 0`**.
  - `scripts/verify_derive_s4.py --v1` gives `v1 record parent_version_id == v0: True; changed ids == label ids: True`,
    and `parent_content_hash == v0 content_hash for all 1000 rows: True`.
  - It also gives `v1 loads: episodes=1000 frames=200340; info total_episodes=1000 total_frames=200340 total_tasks=20 fps=20;
    recomputed id matches registered: True; verify_version files ok: True (85)`.
  - Kept-episode identity is re-proven inside `derive_version`: every kept episode must hash to its
    parent hash and every edited one to its in-memory transformed hash, or the derivation raises.
- ✅ AC4.2: `labels.parquet: 100 rows, columns ['episode_id', 'parent_content_hash', 'content_hash',
  'corruption_type', 'params', 'seed']`. The join with v1 episodes on (episode_id, content_hash) gives
  `100/100 matched, 1:1 (unique ids True)`, and `label parent hash == v0 hash: True`.
  Labels sha256 is `0915f276…`, also recorded as the derive run's `output.labels.sha256` and in `derivation.json`.
- ✅ AC4.3: `scripts/verify_derive_s4.py --guard` gives
  `LineageError raised: True`: `refusing to edit 1 reference-split episode(s) (PoC §8.7 p.15: corrupt only the training copy): ['libero_10/KITCHEN_SCENE3_…/demo_14']`,
  and `nothing written: True; protected ids for v0: 40`. The guard runs before `verify_version` or any write.
- ✅ AC4.4: `scripts/verify_derive_s4.py --padding`, run in an isolated scratch lineage root. Output:
  `derive_version(v0, idle_padding x20 on libero_10/KITCHEN_SCENE3_…/demo_0) -> f4b0ef06952b in 278s`, then
  `length 272 -> 292 (catalog); meta/episodes length=292; LeRobotDataset(episodes=[999]) frames=292; info total_frames=200360 (v0 200340 + 20); full load frames=200360; from/to contiguous=True; inserted frames idle=True`,
  then `diff v0 -> v0pad: changed 1, identical 999; recomputed id matches: True; labels rows: 1`.
  The scratch dataset (7.4 GB) and scratch lineage root were removed afterwards (`True`); the test
  re-creates them.
- ✅ AC4.5: `corrupted per task: [5] over 20 tasks; in reference split: 0; corruption_type ['action_gaussian_noise']; seed [0]`.
  - `params` record std_scale 0.25, σ_a (7 values), source "v0 meta/stats.json action.std (1000
    episodes)", `dims` "all 7 action dims, including the gripper (literal reading …)", `clip: null`,
    the per-episode RNG rule, the selection rule and "lineage demo" labelling.
  - Measured on 10 corrupted episodes (2,014 frames), frame by frame against v0: `same length 10/10,
    observations bit-identical 10/10`. Per dim, (a_v1 − a_v0)/σ_a has mean `[-0.004 … -0.001]` and
    std `[0.253, 0.246, 0.254, 0.255, 0.247, 0.251, 0.251]`; expected 0 and 0.25.
- ✅ AC4.6: the `vlapoc/derive.py` docstring cites §2.1 item 4 p.3, §2.5 p.5, §10 p.20, §8.7 p.15,
  §8.3 p.14, §8.10 p.17 and RoboDrop App. A.1 p.13 (quoted). It records the gripper ambiguity (all 7
  dims, stored in `params`) and the owner-approved scope of the corruption suite.

**Demo artifacts produced:** `artifacts/s4_guard_padding_stdout.txt`, `s4_derive_v1_stdout.txt`,
`s4_version_diff_v0_v1_stdout.txt`, `s4_verify_v1_stdout.txt`, `s4_summary.json`, `s4_labels_head10.csv`,
`s4_build_verify_v0_stdout.txt`, `s4_split_track_stdout.txt`. The v1 dataset is at
`data/lerobot/libero_spatial_10__v1/` (6.8 GB); v1's version dir holds `labels.parquet` and `derivation.json`.

**Notes / divergences:**
- **Derived episodes move position.** Edited episodes are appended after the kept ones, so v1's
  `episode_index` for the 100 edited episodes is 900–999. Identity is carried by `episode_id` in
  v1's episodes table, and subsets resolve through it.
- `derive_version` also carries the parent's reference/candidate split over to the child
  (`v1/reference`, `v1/candidate`, same episode_ids, derivation `inherit_split`). S6 needs these; the
  plan did not list the step.
- The v1 files differ from v0's in number (85 vs 77) because `merge_datasets` repacks the parquet files.
- A derivation costs about 5 min and a full 7 GB copy. `delete_episodes`' per-episode metadata loop
  alone takes 166–181 s.

### S5: Record every stage as an MLflow run and answer trace / consumers / diff
**Files:**
- `vlapoc/runs.py` (+694, of which the module docstring is about 40 lines). Contents:
  - Backend helpers: `tracking_uri`, `client`, `experiment_id` (experiment `vlapoc-lineage`, artifacts
    in `data/lineage/mlartifacts`), and `git_state`.
  - The `StageRun` handle (`input_subset`, `input_version`, `input_entity`, `produced`,
    `output_file`) and the `stage_run` context manager.
  - Stages: `build_run`, `split_run`, `derive_run`, `train_run`, `screen_standin`, `eval_standin`,
    `attribute_standin` and `select_run`.
  - Queries and helpers: `load_checkpoint`, `producers`, `trace`, `consumers`, `diff` and `check_scores`.
- `vlapoc/lerobot_train_entry.py` (+60).
- `vlapoc/lineage.py` CLI: `build verify`, `split make --track`, `trace`, `consumers`, `diff`, `check-scores`.
- `scripts/verify_runs_s5.py` (+169).

**Premises checked:**
- **`lerobot-train` works on Py 3.14: FALSE.** Every invocation, including `lerobot-train --help`,
  fails while building the parser: `TypeError: str | None is not callable`. The failing field is
  `dataset.root`, with `type(type) = <class 'typing.Union'>`.
  - Cause: on 3.14, `str | None` is a `typing.Union`, which draccus 0.11.6 does not recognise as Optional.
  - draccus 0.11.6 is the newest release (`pip index versions draccus`), and lerobot 0.6.1 pins
    `draccus<0.12.0,>=0.11.6`.
  - Fallback used: build `TrainPipelineConfig` in Python and call
    `lerobot.scripts.lerobot_train.train(cfg)`. LeRobot's own `parser.wrap` skips parsing when given a
    config object (`if len(args) > 0 and type(args[0]) is argtype: cfg = args[0]`).
  - Measured: 2 ACT steps on MPS in 10.2 s; the checkpoint was saved with `train_config.json`.
  - The planner had verified only that the module imports.
- **`PreTrainedConfig.from_pretrained` on Py 3.14: FALSE** (same bug: `typing.Dict[str, …PolicyFeature] | None is not callable`).
  Fallback in `load_checkpoint`: `draccus.load(PreTrainedConfig, f)` (a decoder with no argparse),
  then `get_policy_class(cfg.type)(cfg)` and `safetensors.torch.load_model(strict=True)`. Measured:
  `weights loaded strict=True; params 51600263`.
- **Missing dependency `accelerate`: FALSE premise.** See dependencies above.
- **MLflow honours a caller digest: holds, but the length is capped.** `mlflow/utils/validation.py:790`
  raises if `len(dataset.digest) > MAX_DATASET_DIGEST_SIZE` (36), and the DB column is `String(36)`.
  Our ids are 64 hex. Adaptation: `digest = id[:32]` (128 bits), with the full id stored as the input
  tag `vlapoc.id`; every query confirms the full id.
- `search_runs(filter_string="datasets.digest IN (...)")` and tag searches work on 3.16.1, tested in
  a scratch DB. The comparators are in `search_utils.py:183` `VALID_DATASET_COMPARATORS = {"!=", "=", LIKE, ILIKE, "IN", "NOT IN"}`.
- **Normalization-stats leak (plan risk): confirmed.** A checkpoint trained on the 40-episode
  `v0/reference` subset saved normalizer stats equal to v0's full `meta/stats.json`: `action.mean ==
  v0 stats.json action mean: True | state.std equal: True`. Camera stats are ImageNet's
  (`[0.485 0.456 0.406]`). The source is `lerobot_train.py:314 dataset_stats=dataset.meta.stats`.
  `train_run` records `dataset.stats_json.sha256`, the normalizer file's sha256 and a
  `normalization_stats` param saying so.
- LeRobot refuses an existing `output_dir` (`FileExistsError … resume is False`). `train_run` first
  created the dir to hold its spec, so the first S5 attempt failed there (recorded as FAILED in the
  scratch DB). Fixed in `9a754f3`: LeRobot now writes to `<run dir>/lerobot/`.

**Acceptance criteria.** Evidence is from `scripts/verify_runs_s5.py`, run in the isolated scratch
lineage root `data/lineage_scratch/s5` (its own `mlflow.db`) at HEAD `9a754f3c8d47`, **under
`sandbox-exec` with outbound network denied** (see AC5.6).
- ✅ AC5.1:
  - `runs: 7, stages ['build', 'eval', 'select', 'split', 'train'] (all in STAGES: True)`.
  - The train-A input: `name=v0/reference digest=b6a72d5e7f313bff3690caea2b3b2bef
    source={"uri": "data/lineage_scratch/s5/subsets/b6a72d5e….json"}`, with tags `mlflow.data.context:
    training, vlapoc.id, vlapoc.kind: subset, vlapoc.dataset_version_id`.
  - `search_runs(datasets.digest = 'b6a72d5e…') -> ['s5-select-random5', 's5-train-A', 's5-train-A-seed1']`.
  - Eval output tag: `output.rollouts.sha256 == sha256(file): True`.
- ✅ AC5.2: every field of the train-A record:
  - `dataset_version_id 3c4dca68d6f1` and `subset_id b6a72d5e7f31`.
  - Episode-ID list artifact: `subset/b6a72d5e….json`.
  - Mixture weights: "uniform over the subset's frames (LeRobot default sampler; sample_weighting=None)".
  - Augmentation: "none (dataset.image_transforms.enable=false)".
  - Config artifacts: `vlapoc_train_spec.json`, `checkpoint/config.json`, `checkpoint/train_config.json`.
  - Git: `git_sha 9a754f3c8d47`, `git_sha_is_HEAD: True`, `git_dirty: false`. Seed `0`.
  - Checkpoint: `checkpoint.path data/lineage_scratch/s5/runs/…/lerobot/checkpoints/000002/pretrained_model`;
    `weights_sha256_matches_disk: True`; `normalizer_sha256_matches_disk: True`;
    `stats_json_sha256 e7881df78102`; `episodes_match: true`.
- ✅ AC5.3: the eval run shows `checkpoint_sha256 == train-A weights: True; train_run_id == A: True`,
  `suites=["libero_10", "libero_spatial"]`, `libero_task_ids={"libero_10": [0..9], "libero_spatial": [0..9]}`,
  `init_state_indices=0..49 (LIBERO fixed init states)`, `seeds=seed_base + init_state_index, seed_base=0`,
  `loaded=true (51600263 params)` and `stand_in=true`. The rollout plan has
  `rows=1000 outcomes=['not_run']`. The checkpoint is actually loaded (strict), not only hashed.
- ✅ AC5.4 (CLI functions `trace`, `consumers`, `diff`):
  - `trace(s5-train-B) runs: ['train:s5-train-B', 'select:s5-select-random5', 'split:split-v0', 'build:build-v0']; raw files reached: 20`.
  - `consumers(LIVING_ROOM_SCENE2_…/demo_41@b5442591b156): ['s5-select-random5', 's5-train-A', 's5-train-A-seed1', 'split-v0']; lists train-A: True; lists train-B (dropped it): False`.
  - `diff(A, B): removed 5, added 0, changed_version 0; params []; config []; git same True`.
  - `diff(A, A-seed1): data removed/added/changed 0/0/0; params {'seed': ('0', '1')}; config {'seed': (0, 1)}`.
  - Full JSON in `artifacts/s5_diff_A_B.json`.
- ✅ AC5.5: a tiny run trained on a v1 subset (6 corrupted + 4 reference episodes). The v1-keyed
  table is accepted (`{'rows': 10, 'training_episodes': 10, 'covered': 10, 'ok': True}`). The same
  episodes keyed by clean v0 hashes are rejected: `6/10 score rows are not episode versions trained on
  by run f8fc1a9eae50 (e.g. libero_spatial/…/demo_12@7707f6ecd8f9)`. Exactly the 6 corrupted rows are
  rejected; the 4 reference episodes have identical hashes in v0 and v1, so they are correctly accepted.
- ✅ AC5.6: the whole S5 script (MLflow, 4 real LeRobot trainings, checkpoint loading, queries) ran
  under `sandbox-exec -p '(version 1)(allow default)(deny network-outbound (remote ip))'` and exited 0
  in `55.09 real`.
  - Control, measured: under the same sandbox `curl https://huggingface.co` returns `000`, exit 7;
    without it, `200`.
  - Tracking URI scheme is `sqlite`; `HF_HUB_OFFLINE=1`; no wandb installed; no account anywhere.
- ✅ AC5.7: the `vlapoc/runs.py` docstring cites §2.1 p.3, §2.3 p.4 and §9.1–9.2 pp.18–19 (quoted),
  and documents the MLflow encoding and the 36-character digest limit. `lerobot_train_entry.py`
  documents the draccus/3.14 fallback.

**Demo artifacts produced:** `artifacts/s5_verify_stdout.txt`, `s5_summary.json`, `s5_diff_A_B.json`.
The scratch lineage root `data/lineage_scratch/s5/` (2.3 GB, mostly 4 tiny-run checkpoints) was left
for inspection; the script deletes and re-creates it.

**Notes / divergences:**
- AC6.2's wording ("real `lerobot-train` runs") is met by LeRobot's `train()` function rather than
  the `lerobot-train` CLI binary, which cannot parse arguments on Python 3.14 (above). The training
  loop, checkpointing and `train_config.json` are LeRobot's own.
- MLflow stores some absolute paths in its own records: the experiment `artifact_location` and the
  auto tag `mlflow.source.name`. Our manifests and the MLflow artifact copies were rescanned after
  S6, and none holds an absolute path: `lineage manifests scanned: 31 absolute-path hits: []`,
  `mlflow artifact copies scanned: 27 hits: []`. Moving `data/lineage` to another machine would need
  the artifact location remapped (inferred; not tested).
- The first `split make --track` attempt failed (`ImportError: cannot import name
  'LocalArtifactDatasetSource'`, a class MLflow generates dynamically). It is left in the main DB as a
  FAILED `split-v0` run with no inputs or products. Queries ignore non-FINISHED runs (`398e037`).

### S6: Run every downstream stage type end to end on the real data (lineage dry run)
**Files:** `scripts/lineage_dry_run.py` (+297: the chain, the AC checks, `graph()`, `mermaid()`).

**Command:** `HF_DATASETS_CACHE=<repo>/data/hf_datasets_cache .venv/bin/python scripts/lineage_dry_run.py --k 20 --seed 0`
→ `done in 83.2s`, exit 0, chain `dry-run-k20-s0-20260929T074215Z`, HEAD `7effd257e5f1`, `dirty=False`.

**Acceptance criteria:**
- ✅ AC6.1: 14 runs, all FINISHED (`artifacts/s6_chain_runs.csv`):
  - build-v0 (`5d9d272`) and split-v0 (`398e037`) were recorded by the S4 commands and reused via
    their `produced.*` tags (`[reuse] build(v0): run bc8f7d09a7bf`, `[reuse] split(v0): run 30411f67f697`).
  - derive-v1 (`398e037`) was likewise recorded by S4 and reused (`[reuse] derive(v1): run 30f34cd056ab`).
  - Recorded at `7effd25`:
    - screen-v1 [stand-in]
    - train-full-s0 and train-full-s1
    - eval-train-full-s0 [stand-in] and attribute-train-full-s0 [stand-in]
    - select-drop-top-20 and select-drop-random-20-s0
    - train-drop-top-20 and train-drop-random-20
    - eval-train-drop-top-20 [stand-in] and eval-train-drop-random-20 [stand-in]
  - Stand-ins have `stand_in=true`. Their payloads are placeholders, and no outcome numbers were invented:
    - screening: `suspicion_score` = NaN;
    - eval: a rollout plan with `outcome = not_run`;
    - attribution: `placeholder_score` = sha256 of the key mapped to [0,1), labelled "not an influence estimate".
  - Order: attribute precedes select (select consumes its score table), matching AC6.4's trace path.
- ✅ AC6.2: `train_config.json episodes=960 == subset.resolve(): True` for train-full-s0 and s1, and
  `940 … True` for drop-top-20 and drop-random-20. All use `policy=act steps=5 device=mps wandb=False`.
  Seeds are 0/1/0/0; training took 16.3 / 7.5 / 12.2 / 25.8 s.
- ✅ AC6.3: `diff(train-full-s0, train-drop-top-20): removed 20, added 0, changed_version 0; params {}; config {}; git same True`
  and `diff(train-full-s0, train-full-s1): removed/added/changed 0/0/0; params {'seed': ('0', '1')}; config {'seed': (0, 1)}; git same True`.
  The diff lists its excluded bookkeeping keys explicitly: `checkpoint_path, dataset.episodes (reported
  as data), job_name, n_episodes, n_frames, output_dir, run_name, subset_id, wandb.run_id`.
- ✅ AC6.4: `trace(train-drop-top-20)` reaches 8 runs: `train-drop-top-20, select-drop-top-20,
  derive-v1, attribute-train-full-s0 [stand-in], build-v0, split-v0, train-full-s0, eval-train-full-s0
  [stand-in]`. All 11 consumed-entity edges resolve to exactly 1 producer, and `raw files reached: 20`
  (e.g. `KITCHEN_SCENE3_…` sha256 `6b30906a52a5741e`). JSON is in `artifacts/s6_trace_train_drop_top.json`.
  - `corrupted episodes among the 20 dropped by top-k: 3`. The placeholder scores are a key hash,
    so this is chance, not attribution.
  - Probe `libero_10/KITCHEN_SCENE3_…/demo_17` (dropped: True):
    `consumers(demo_17@v1 6cfad44e5243): ['attribute-train-full-s0 [stand-in]', 'screen-v1 [stand-in]',
    'select-drop-random-20-s0', 'select-drop-top-20', 'train-drop-random-20', 'train-full-s0',
    'train-full-s1']`, so it `lists train-full-s0: True; lists train-drop-top-20: False`.
  - The same episode at its clean v0 hash `3456348a0125` gives `['derive-v1', 'split-v0']`.
- ✅ AC6.5: `flipped byte 5552926 of data/chunk-000/file-080.parquet in a scratch clone of v1; train_run refused: True`:
  `dataset copy at <scratch>/v1_tamper_copy does not match version v1 (16076b9c5cb2): changed=['data/chunk-000/file-080.parquet'] missing=[] extra=[]`.
  `MLflow runs before/after the refused launch: 15/15`, and no run dir was created for it.
- ✅ AC6.6: `screening stand-in rows 960; keyed to v1/candidate episode versions: 960/960; left join
  with labels.parquet on (episode_id, content_hash): 100 labelled corrupt, 860 unlabelled; labels found
  in screening table: 100/100`. The attribution stand-in against train-full-s0 passes `check_scores`:
  `{'rows': 960, 'training_episodes': 960, 'covered': 960, 'ok': True}`.
- ✅ AC6.7: the lineage graph has `36 nodes (14 runs), 40 edges`, written to
  `artifacts/s6_lineage_graph.json` and `artifacts/s6_lineage_graph.mmd` (Mermaid flowchart: runs,
  entities, and the 20 raw files as the source node). It was regenerated from the recorded run ids
  after a label-only fix (`67bbd41`), with the same 36/14/40.
- Plan's visual check (`mlflow ui`): the server was started on 127.0.0.1:5055 against
  `data/lineage/mlflow.db` and queried over HTTP (`artifacts/s6_mlflow_ui_rest_stdout.txt`):
  `search_runs over HTTP, datasets.digest = v1/candidate[:32] (69684e609529...): 6 runs` (attribute,
  screen, both selects, train-full-s0/s1), `all runs in experiment: 15`, `UI root HTTP 200`. Then
  stopped. No screenshot was taken.

**Demo artifacts produced:** `artifacts/s6_dry_run_stdout.txt`, `s6_summary.json`, `s6_chain_runs.csv`,
`s6_diff_full_s0_vs_top.json`, `s6_diff_full_s0_vs_seed.json`, `s6_diff_full_s0_vs_rand.json`,
`s6_trace_train_drop_top.json`, `s6_consumers_probe.json`, `s6_lineage_graph.json`,
`s6_lineage_graph.mmd`, `s6_mlflow_ui_rest_stdout.txt`. Checkpoints for the 4 train runs are under
`data/runs/` (2.3 GB).

**Notes / divergences:** the train runs are 5-step ACT smoke runs. They prove the data → trainer →
checkpoint path, not training quality (as AC6.2 says). The chain's build, split and derive runs were
recorded by the S4 commands, not by this script; the script found and reused them through their
`produced.*` tags because the entities are content-addressed.

### Findings for the analyzer (batch 2)
1. **`lerobot-train` CLI and `from_pretrained` are unusable on Python 3.14 (FALSE premise; routes to
   planner/owner).** draccus 0.11.6 (the latest, and pinned by lerobot) cannot build a parser for
   `X | None` fields on 3.14. The workarounds are in place: programmatic `train(cfg)` and
   `draccus.load`, then a strict safetensors load. They depend on LeRobot internals (`parser.wrap`'s
   bypass). The SmolVLA / GPU training feature should either run on Python ≤ 3.13, where the CLI
   presumably works (inferred, untested), or keep this entry point.
2. **Missing training dependency.** `accelerate` was installed; it was not in the plan's
   environment table.
3. **MLflow's 36-character digest cap.** Handled with 32-hex prefixes plus the full id in tags.
   The plan's AC5.1 wording `digest=<version_id | subset_id>` is therefore satisfied only by prefix.
4. **HF `datasets` Arrow cache per subset (FALSE premise "training subsets never copy data", in
   practice; routes to planner and the training feature).**
   - Mechanism: `LeRobotDataset(episodes=…)` → `Dataset.from_parquet(filters=…)` writes a filtered
     Arrow copy per distinct episode list, about 7 GB for a 960-episode subset of this dataset.
   - Measured: this session created about 50 GB of such caches in
     `~/.cache/huggingface/datasets/parquet/` (7 entries of 6.5–7.2 GB, all dated 2026-09-29; the
     whole dir is 62 GB). S6's three distinct subsets used 20 GB of the project-local
     `data/hf_datasets_cache/`.
   - Free disk went from 162 GiB (planning) to 53 GiB. The biggest users are the v1 copy (6.8 GB),
     checkpoints (2.3 + 2.3 GB) and these caches.
   - **I did not delete anything in `~/.cache`.** The owner may want to clear the 2026-09-29
     `parquet/default-*` entries. They are regenerable, and an unrelated `sentence-transformers___gooaq`
     entry sits in the same directory.
   - Counterfactual experiments with many subsets need `HF_DATASETS_CACHE` on scratch plus cleanup,
     or a loader that avoids the materialisation.
5. **Normalization-stats leak confirmed (measured).** A subset-trained checkpoint normalizes with
   the whole version's `meta/stats.json`, including dropped and reference episodes. It is recorded per
   run (stats sha256 plus normalizer sha256), not solved. The training/verification feature owns the choice (§1 p.1 "data is the only variable").
6. **Derivation cost.** Each derived version is a full dataset copy (6.8 GB) and takes about 5 min,
   mostly LeRobot's `delete_episodes` metadata loop. That is acceptable for the planned handful of
   injection versions.
7. **Main lineage DB contents:** 15 runs. They are the 14 S6 chain runs (build/split/derive from the
   S4 commands, 11 from the dry run) plus 1 FAILED split attempt from the `LocalArtifactDatasetSource`
   bug. The failed run is inert (no inputs or products; queries use FINISHED runs only).
8. **Plan wording to revisit (not changed by me):** AC6.2 says "real `lerobot-train` runs". See
   finding 1: it is LeRobot's `train()` without the CLI.
