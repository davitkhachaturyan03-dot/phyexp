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
