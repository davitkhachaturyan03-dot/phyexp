# Plan: Data layer + lineage layer (LIBERO → LeRobot dataset, episode versions, run lineage)

Planner round 1 · 2026-09-28 · evidence labels: **quoted** (source + quote), **measured** (command run
in this planning session, reproducible), **inferred** (reasoning, unverified).

## Request

The owner asked (verbatim, [brief.md](brief.md)): *"build the data layer and the lineage layer FEATURE.
The output must be a dataset and a lineage layer that will enable all downstream tasks inside the
poc"*. Concretely this means: (1) a training-ready dataset built from the LIBERO demonstrations on disk,
and (2) a lineage layer such that every downstream PoC stage (injection with ground-truth labels,
screening against a clean reference, training on a subset, rollout evaluation, per-episode
attribution, drop-top-k counterfactual retraining) can name **exactly which episodes, and which
version of each episode**, it consumed and produced. PoC-sized; standard tools plus thin adapters.

## Spec grounding

- **Key takeaways (p. 2):** "Lineage (exact data versions, episode IDs, configs and checkpoints tied to
  eval results) is the foundation. Attribution is impossible without it." (quoted)
- **Goal (p. 1):** "Scope is deliberately limited to data: model, config, training schedule and
  evaluation protocol are held fixed so that data is the only variable." (quoted)
- **§2.1 "Implementation roadmap", item 1 (p. 3):** "Build lineage before attribution. You can't
  attribute anything you haven't recorded. Every training run should be tied to exact dataset versions
  (content hashes), episode IDs, mixture weights, augmentation settings, the config, the git SHA, and
  the resulting checkpoints. Every eval rollout should be tied back to that checkpoint. W&B Artifacts,
  MLflow, or DVC handle this well. This part is unglamorous, but it's the backbone. It is also useful
  on its own, because it lets you answer "what changed between the good run and the bad run?"" (quoted)
- **§2.1 items 2–4 (p. 3):** "LIBERO is a good default because it has distinct task suites …
  SmolVLA via LeRobot is small enough for a single GPU."; "Use enough rollouts per task (roughly 50 or
  more), fixed seeds and initial states"; "Deliberately corrupt known subsets of training data … Then
  check whether your attribution method flags exactly those." (quoted)
- **§2.2 "Episode catalog" (p. 4):** "Each episode gets a stable ID and a metadata row: source, task,
  language instruction, scene, objects, operator, collection date, episode length, and basic action
  statistics (smoothness, idle time, gripper toggles). LeRobot's dataset format plus a Parquet
  metadata table works well. Per-episode attribution scores become most valuable when aggregated by
  these columns." (quoted)
- **§2.3 "Experimental setup" (p. 4), Controls:** "Fixed config and steps; 2–3 seeds per data
  configuration". (quoted)
- **§2.5 "Corruption types for the injection test" (p. 5):** noisy/time-shifted actions, swapped
  instructions, altered viewpoint/dropped frames, low-skill/hesitant demos; "Inject 5–10% corrupted
  episodes and measure how many the method ranks near the top." (quoted)
- **§2.6 "Counterfactual verification" (p. 5):** "Remove the top-k “harmful” episodes and retrain: the
  policy should improve more than after removing k random episodes." (quoted)
- **§3.5 A (pp. 7–8):** "T = time (or steps) to task success per episode; for long-horizon LIBERO-10
  tasks, per sub-task." … "Human reference = durations of LIBERO's original human-teleoperated
  demonstrations on the same tasks." (quoted)
- **§8.3 (p. 14):** references "are a small, held-out set of clean demonstrations for the target task,
  separate from the candidate data being audited." (quoted)
- **§8.7 (p. 15), row "Benchmark demos (PoC)":** "Treat LIBERO's original demos as clean; corrupt only
  the training copy" → "Free references plus ground truth". (quoted)
- **§8.10 (p. 17):** "pad some with idle or hesitant segments (should hurt speed, not reliability) and
  add gripper-release noise to others" (quoted), so derived episodes may change length.
- **§9.1 (pp. 18–19):** Step 1 screening verdicts "are stored rather than acted on, so they can later
  be compared with what actually happened"; Step 3 input "The training dataset, the trained policy, and
  the rollouts from Step 2", output "An episode × metric table of influence scores"; Step 4 "Remove the
  top-k most harmful episodes and retrain. As a control, remove k random episodes and retrain."
  (quoted). **§9.2 (p. 19):** "The most compelling output is where screening and outcome attribution
  disagree" (quoted), so screening and attribution outputs must share episode-version keys.
- **§10 "First milestone" (p. 20):** "Foundation | Lineage tracking; episode catalog; LIBERO (one or
  two suites) + SmolVLA"; "Ground truth | Injection experiment with 5–10% corrupted episodes, using
  LIBERO's original demos as clean references". (quoted)

**Rules the build must follow (derived from the above):**
1. Identity is content-based: a run names episodes by a stable `episode_id` **and** a
   `content_hash`; a corrupted copy is a new version of the same `episode_id` (§2.1 p. 3).
2. The clean reference set is held out from the candidate/training set and is never corrupted
   (§8.3 p. 14, §8.7 p. 15, §10 p. 20).
3. Training subsets are selections, not copies; counterfactual subsets record how they were derived
   (drop top-k by which score/metric; random-k with which seed) (§2.6 p. 5, §9.1 Step 4 p. 19).
4. Every train run records dataset version, episode IDs, mixture weights, augmentation, config, git
   SHA, seed, checkpoints; every eval run records the checkpoint it loaded (§2.1 p. 3, §2.3 p. 4).
5. Do not filter the corpus on an unverified premise (CLAUDE.md / vla-build: the OpenVLA 189-demo
   filter).

**Papers:**
- `docs/papers/RoboDrop_Gradient_Compatibility_arXiv-2609.10021.pdf` App. A.1 (p. 13): "corrupting
  10% of the episodes for every task. Action corruption adds zero-mean Gaussian noise with standard
  deviation 0.25σa to the demonstrated actions, where σa is the empirical action standard deviation.
  Temporal corruption shifts the action sequence by two seconds relative to the observations. For
  validation, we reserve two uncorrupted episodes per task, giving 20 clean validation episodes for
  each suite." (quoted) → the reference-split size (2/task) and the demo derivation transform in S4.
  App. p. 20: the final policy "is post-trained on the retained candidate episodes together with the
  clean validation episodes" (quoted). Whether our references join training is left to the
  screening/training features.
- `docs/papers/LIBERO_Benchmark_arXiv-2306.03310.pdf` p. 6: "50 trajectories of high-quality
  demonstrations for every single task … collected by human experts through teleoperation with
  3Dconnexion Spacemouse." (quoted) → catalog `source`; operator identity and dates are not published.
- `docs/papers/CUPID_Influence_Functions_arXiv-2506.19121.pdf` released code
  (`diffusion_policy/dataset/robomimic_replay_image_dataset.py`) identifies demos by `demo_{i}` HDF5
  keys and selects subsets via `get_dataset_masks(...)` train/val/holdout masks (quoted, source via
  fetch) → keep `(source_file, demo_key)` on every episode so a CUPID fork can map back.

## Premises verified against the artifacts (planning session)

| Claim | Evidence | Label |
|---|---|---|
| 20 raw files (10 per suite), 50 demos each → **1000 demos**; **200,340 steps** (62,250 libero_spatial + 138,090 libero_10) | h5py scan of `data/raw/libero/*/*.hdf5` | measured |
| Lengths: libero_spatial 75–197 (median 124), libero_10 150–517 (median 263) steps | same scan | measured |
| Every demo has `rewards` summing to exactly 1 and `dones` summing to exactly 1, both at the **final** step (1000/1000); no reward earlier | same scan | measured |
| No `mask/` (robomimic filter-key) group in any file; datasets uncompressed | same scan | measured |
| Images `agentview_rgb`, `eye_in_hand_rgb` are 128×128×3 uint8; `env_args.control_freq = 20`; `camera_heights = 128`; `macros_image_convention = opengl` | `data` attrs | measured |
| `obs/ee_ori` is axis-angle of the eef quaternion (max abs err 4.4e-16 vs `quat2axisangle(robot_states[5:9])`, 8 demos) | computed | measured |
| `init_state` attr == `states[0]` | computed | measured |
| Model XML lists scene objects as bodies (e.g. `moka_pot_1_main`, `flat_stove_1_main`, `chefmate_8_frypan_1_main`) | regex over `demo_0.attrs['model_file']` | measured |
| Exactly-zero non-gripper action steps: 97 (0.2%, 28 episodes) libero_spatial; 339 (85 episodes) libero_10. Gripper toggles per episode 1–7 / 1–8 | computed | measured |
| Raw frames rotated 180° match the `HuggingFaceVLA/libero` layout for the same task (both cameras) | frame fetched via datasets-server rows API, stacked against raw and rotated raw, inspected | measured (visual) |
| LIBERO benchmark task order ≠ alphabetical; task names == HDF5 stem minus `_demo` | `libero/libero/benchmark/libero_suite_task_map.py` (LIBERO and huggingface/lerobot-libero agree) | quoted |
| LeRobot 0.6.1 writer works on this env (Py 3.14) in PNG mode: 2 demos / 182 frames written in 1.0 s, **39,888 B/frame** (2 cams, parquet) → ≈ **8.0 GB** for 200,340 frames; `LeRobotDataset(root, episodes=[1])` returns exactly that episode's 84 frames; image/state/action read back bit-exact | scratchpad smoke test | measured (size and time projected, inferred) |
| `lerobot.scripts.lerobot_train` imports; torch 2.11.0 with MPS available | import check | measured |
| Environment: Python 3.14.0, lerobot 0.6.1, pyarrow 25, pandas 2.3, h5py 3.16 installed; **mlflow, wandb, dvc not installed**; **repo is not a git repository**; 162 GiB free; Apple M4 Pro, 24 GB, no CUDA | `pip list`, `df`, `sysctl`, env | measured |
| PyAV's bundled libavdevice clashes with Homebrew ffmpeg ("may cause spurious casting failures and mysterious crashes") | objc warning when importing lerobot | measured |

## Prior art

| Candidate | What it covers | What it misses | Evidence |
|---|---|---|---|
| **`lerobot/libero`** (HF, rev `a1aaacb7`) | LeRobot v3 LIBERO, 4 suites, schema the LeRobot LIBERO env expects, 1.94 GB, MP4/AV1 | Only **811 of our 1000 demos** (432/500 spatial, 379/500 libero_10) = OpenVLA no-noops corpus; 154,439 of 200,340 frames; **68/811 episodes have a length no raw demo of that task has** (steps removed); `fps: 10` although the data is 20 Hz control; 256² re-render; `meta/episodes` has 14 columns and **no mapping back to raw demo IDs** | `info.json`/`meta/episodes` downloaded at pinned rev; per-task counts and frame totals via `HuggingFaceVLA/libero` meta (same 1,693/273,465); LeRobot docs: "episodes / frames / tasks: 1,693 / 273,465 / 40", "download size 1.9 GB" (quoted); measured |
| **`HuggingFaceVLA/libero`** (rev `86958911`) | Same episodes, PNG in parquet | Same gaps as above; **34.93 GB** on the Hub (LeRobot docs say "69.9 GB", but the Hub file sizes win, discrepancy recorded) | `HfApi.dataset_info(files_metadata=True)`; measured |
| **openpi `convert_libero_data_to_lerobot.py`** / `physical-intelligence/libero` | Converter template | Reads OpenVLA's `modified_libero_rlds` (filtered), sets `fps=10` (likely the origin of the hub's fps label, inferred), v2.x format | script source (quoted) |
| **`openvla/modified_libero_rlds`** | RLDS no-noops LIBERO | `libero_spatial_no_noops` = 432 episodes / 52,970 frames: identical to the hub's spatial slice, so the hub is this filtered corpus. Adopting it adopts the 189-demo filter (68 + 121 = 189) that CLAUDE.md records as premise-false | HF page (quoted, third-party) + our meta arithmetic (measured) |
| **any4lerobot `libero2lerobot/libero_h5.py`** (MIT) | Raw-HDF5 → LeRobot v3 adapter | Declares 256² shapes; **does not rotate frames** (rotation only in its sim `regenerate_libero_dataset.py`), so raw 128² frames would be upside-down vs the eval env; key `wrist_image` ≠ env's `image2`; needs `datatrove` + its `generic_converter`; no identity/hashes | source read (171 lines), measured; README (quoted) |
| **LeRobot v3 format + writer** (lerobot 0.6.1, installed) | Storage format SmolVLA trains on; `LeRobotDataset.create/add_frame/save_episode/finalize`; `episodes=` subset loading (`DatasetConfig.episodes`); `episode_filter`; `delete_episodes`, `split_dataset`, `merge_datasets`, `modify_tasks`, `recompute_stats`; per-dataset `stats.json`; checkpoints save `train_config.json` + preprocessor (normalizer) | **Episode identity is positional**: edit tools reindex (`_copy_and_reindex_data(... episode_mapping: Mapping from old episode indices to new indices)`); no per-episode content hash, no parent links, no labels; **normalization stats always come from full `meta/stats.json` even with `episodes=`** (`dataset_metadata.py:219 self.stats = load_stats(self.root)`; `lerobot_train.py:314 dataset_stats=dataset.meta.stats`; no subset recompute) | source (quoted), smoke test (measured) |
| **robomimic filter keys** | Named subsets in HDF5 (`mask/<key>`), used by CUPID's code | Not consumed by LeRobot/SmolVLA; no versioning; our files have no `mask` group | robomimic docs (quoted); measured |
| **RLDS / TFDS** | Episode-of-steps spec, `episode_metadata` (e.g. DROID `file_path`), semantic dataset versions | No per-episode content hash or derivation links; not LeRobot's training format | web docs (quoted) |
| **MLflow 3.16.1** | Runs, params, tags incl. auto `mlflow.source.git.commit`; artifacts; **dataset inputs** `mlflow.log_input(dataset, context=…)`; `mlflow.data.from_pandas(df, source, name, digest)` / `MetaDataset(source, name, digest)` where "If unspecified, a digest is computed automatically" (so a caller digest is honoured); `search_runs(filter_string="datasets.digest IN (...)")`; local SQLite default since 3.7 (FS store deprecated, issue #18534); no account | No output-dataset lineage (outputs only as artifacts/tags; Databricks KB: "no built-in method … to log output tables"); the pandas digest hashes only `df.head(MAX_ROWS)`, `MAX_ROWS = 10000` (`mlflow/data/digest_utils.py`), so it is not a content hash of frames; no episode concept; not integrated in LeRobot (grep: 0 hits) | docs + source (quoted); `pip index versions` (measured: resolves on Py 3.14) |
| **W&B Artifacts** | Best lineage DAG (`use_artifact`/`log_artifact`), reference artifacts ("The files themselves never leave your system"; 10,000-file limit per directory reference); **LeRobot built-in** `--wandb.enable` logs `cfg.to_dict()` (incl. `dataset.episodes`) and checkpoint artifacts | Needs a W&B account or self-hosted server: **`use_artifact` raises `TypeError("Cannot use artifact when in offline mode.")`** (`wandb/sdk/wandb_run.py:3104-3105`, main); no episode concept | source (measured by fetch); docs (quoted); `lerobot/common/wandb_utils.py:107,160` (quoted) |
| **DVC 3.67** | md5 per file/dir in `dvc.lock`, pipelines, remotes | File-level: v3 packs many episodes per parquet file, so a changed episode changes a whole file hash and doesn't say which episode; `dvc diff` needs Git (`--no-scm`: "dvc diff and dvc metrics diff are unavailable"); another tool plus Git | docs (quoted) |
| **HF Hub revisions** | Commit-SHA pinning; LeRobot docs: "Pin `--dataset.revision=<commit-sha>` when reporting results" | Needs account and upload; dataset-level only. Useful later for moving versions to a Linux GPU box | LeRobot docs (quoted) |
| **RoboLineage** (arXiv 2606.22142, github.com/robolineage/robolineage) | Typed lineage graph incl. "dataset locks freezing accepted episodes with adapter context and hashes" | "agents act as artifact-producing lifecycle workers": requires LLM/VLM agents; real-robot collection governance; research prototype; no MLflow/LeRobot integration | arXiv HTML + project page (quoted) |

**Verdict: adapter over LeRobot v3 (dataset) + MLflow (run lineage).** The only custom code is the
one thing none of the tools provide: a stable `episode_id` + per-episode `content_hash`, version and
subset manifests, derivation labels, and three lineage queries. Estimated size: converter ~100 lines,
lineage ~300 lines (inferred).

Decisions and why:
- **Convert our raw HDF5 with LeRobot's own writer; do not download the hub LIBERO datasets.** The hub
  corpus drops 189 of our 1000 demos and shortens 68 others (measured). That is the filter CLAUDE.md
  rules out, and it would bias the HRT human-reference durations (§3.5 p. 8) and break traceability to
  `demo_N` (needed for §8.7 references and a CUPID fork). Nothing to download: 0 GB instead of 1.9 GB.
- **Match the LeRobot LIBERO schema exactly** (keys `observation.images.image` / `.image2`,
  `observation.state` = eef pos ‖ axis-angle ‖ gripper qpos (8), `action` (7), frames rotated 180°) so
  `lerobot-train`/`lerobot-eval` and SmolVLA need no remapping. LeRobot docs: "If your data contains
  different keys, you must rename the observations … since naming keys are encoded inside the
  normalization statistics layer" (quoted); `processor/env_processor.py`: "Rotates images by 180
  degrees by flipping both height and width dimensions" and `state = torch.cat((eef_pos,
  eef_axisangle, gripper_qpos), dim=-1)` (quoted).
- **Declare fps 20**, the truthful control rate (`env_args.control_freq: 20`, measured; LeRobot's own
  env config: `fps: int = 20  # Must match robosuite's default control_freq (20 Hz)`, quoted). This
  matters for time-based corruptions: RoboDrop's "two seconds" is 40 steps, and for HRT in seconds.
- **Keep native 128×128**: no upsampling, no simulator re-render (the simulator requires Linux: LeRobot
  docs "LIBERO requires Linux (`sys_platform == 'linux'`)", quoted).
- **Store images losslessly (PNG in parquet, `use_videos=False`)**, about 8 GB (projection). Lossless
  storage lets S2 prove bit-exact fidelity to raw for every episode and makes content hashes
  decoder-independent. AV1 would be ~1 GB but lossy (inferred), and PyAV/ffmpeg clash on this machine
  (measured). LeRobot docs report "equivalent loading speed" between the video and PNG variants (quoted).
- **MLflow local** for run lineage: it is named by the PoC, needs no account, works offline, and
  records dataset digests with native search. **W&B is the alternative, and that is the owner's call
  (BLOCKING Q1).** DVC was not chosen: it is file-level, Git-bound and a second tool, and it adds
  nothing MLflow plus our manifests lack for a PoC.

## Stories

Proposed layout (the executor may refine): `vlapoc/libero_convert.py` (S1), `vlapoc/catalog.py` (S2),
`vlapoc/lineage.py` (S3–S5, CLI via `python -m vlapoc.lineage …`), `scripts/lineage_dry_run.py` (S6).
Data: `data/lerobot/<dataset_name>/` (LeRobot roots), `data/lineage/versions/<version_id>/`,
`data/lineage/subsets/<subset_id>.json`, `data/lineage/mlflow.db` + `data/lineage/mlartifacts/`.
Put evidence outputs (figures, tables, command logs) in `tracker/features/2026-09-28-data-lineage-layer/artifacts/`.

### S1: Convert raw LIBERO (libero_spatial + libero_10) into a LeRobot v3 dataset, unfiltered
- **Why:** §10 p. 20 "LIBERO (one or two suites) + SmolVLA"; §2.2 p. 4 "LeRobot's dataset format";
  user ask (1): "the output must be a dataset".
- **Depends on:** —
- **Deliverables:** `vlapoc/libero_convert.py` (h5py read → rotate 180° → `LeRobotDataset.create(fps=20,
  use_videos=False, robot_type="panda")` / `add_frame` / `save_episode` / `finalize`); dataset root
  `data/lerobot/libero_spatial_10/` (v0); a verify script; an orientation figure.
- **Acceptance criteria:**
  - [ ] AC1.1 `meta/info.json` reports `codebase_version` v3.0, `fps` 20, `total_episodes` 1000,
    `total_frames` 200340, `total_tasks` 20; `LeRobotDataset(repo_id, root=…)` loads. [PoC §10 p.20; measured raw counts]
  - [ ] AC1.2 Feature keys, dtypes and non-image shapes equal `lerobot/libero`@`a1aaacb7`'s
    `info.json` (`observation.images.image`, `observation.images.image2`, `observation.state` float32[8],
    `action` float32[7]); the only difference is image shape 128×128×3 and `fps`; a script prints the diff. [LeRobot LIBERO docs, quoted; env_processor.py]
  - [ ] AC1.3 `observation.state` == `ee_pos ‖ ee_ori ‖ gripper_states` cast to float32 and `action` ==
    raw actions cast to float32, for every frame of 3 spot-checked episodes per suite. [LeRobot docs "8-dim … (eef position, axis-angle orientation, gripper qpos)"; measured axis-angle]
  - [ ] AC1.4 Frames are the raw frames rotated 180°; orientation figure in `artifacts/` shows raw vs
    converted vs a `HuggingFaceVLA/libero` reference frame for the same task. [env_processor.py, quoted; planning measurement]
  - [ ] AC1.5 No filtering: per-task episode count is 50 for all 20 tasks, and each episode's length
    equals its raw `num_samples` (1000/1000). The log re-verifies that every demo ends with its only
    `reward == 1` at the last step. [CLAUDE.md / vla-build rule; measured]
  - [ ] AC1.6 Storage is lossless (`use_videos=False`); the actual on-disk size and conversion
    wall-time are reported (planning projection ≈ 8.0 GB, ≈ 18 min, inferred). [inferred design choice, needed by AC2.2]
  - [ ] AC1.7 Grounding docstring cites PoC §2.2 p.4, §10 p.20 and the LeRobot LIBERO schema source, and
    states the divergences from the hub datasets (all 1000 demos, 128², fps 20). [vla-build]
- **Verify by:** `.venv/bin/python -m vlapoc.libero_convert --out data/lerobot/libero_spatial_10` then
  the verify script, which prints the info.json fields, the per-task counts, the length check,
  `LeRobotDataset(..., episodes=[0, 999])[0]` shapes and the feature diff vs `lerobot/libero`.
- **Demo value:** "1000 demos / 200,340 frames converted, 0 dropped, against 811 / 154,439 in the hub
  `lerobot/libero` for the same two suites", plus the orientation figure.

### S2: Give every episode a stable ID, a content hash and a §2.2 catalog row
- **Why:** §2.1 p. 3 "exact dataset versions (content hashes), episode IDs"; §2.2 p. 4 catalog; §3.5
  p. 8 human-reference durations for HRT; §1 p. 3 aggregation by metadata.
- **Depends on:** S1
- **Deliverables:** `vlapoc/catalog.py`: `episode_content_hash(...)`, `build_catalog(version)` →
  `episodes.parquet`; raw-file sha256 list.
- **Acceptance criteria:**
  - [ ] AC2.1 `episode_id = "<suite>/<task_name>/demo_<k>"`, taken from the raw source rather than from
    LeRobot's positional `episode_index`; 1000 unique IDs; each row keeps `source_file` (relative path)
    and `source_demo` (`demo_<k>`). [PoC §2.2 p.4 "stable ID"; CUPID code `demo_{i}` keys]
  - [ ] AC2.2 `content_hash` = sha256 over the canonical episode content (task string, then per frame:
    `action` f32, `observation.state` f32, `image` uint8 HWC, `image2` uint8 HWC, in frame order;
    bookkeeping columns `index`/`episode_index`/`timestamp` excluded, since they change on reindex).
    The hash computed from the stored LeRobot dataset **equals** the hash recomputed directly from raw
    HDF5 (rotate + cast) for **1000/1000** episodes. [PoC §2.1 p.3 "content hashes"]
  - [ ] AC2.3 sha256 of each of the 20 raw HDF5 files is recorded (dataset provenance root). [PoC §2.1 p.3]
  - [ ] AC2.4 `episodes.parquet` has one row per episode with: `episode_id`, `episode_index`,
    `content_hash`, `suite`, `task_name`, `libero_task_id` (LIBERO `libero_suite_task_map.py` order,
    commit pinned in the docstring), `language_instruction`, `scene`, `objects` (model-XML bodies,
    robot/mount/table excluded), `source` (per LIBERO paper p.6), `operator` and `collection_date`
    (null, with a note that they are not in the source), `length_steps`, `duration_s` (= length/20),
    `init_state_sha256`, `smoothness` (mean ‖Δa[:6]‖₂), `idle_fraction` (share of steps with
    a[:6] == 0 exactly), `gripper_toggles` (sign changes of a[:,6]). [PoC §2.2 p.4; §3.5 p.8; LIBERO p.6]
  - [ ] AC2.5 Catalog values match the planning measurements: `duration_s` range 3.75–9.85 s
    (libero_spatial) and 7.5–25.85 s (libero_10); total zero-action steps 97 and 339; toggles 1–7 and
    1–8. Idle and toggles need no invented threshold (exact zeros and sign changes). [measured]
  - [ ] AC2.6 `duration_s` is documented as the human time-to-success T (last step = first `reward == 1`,
    re-verified 1000/1000), which the evaluation feature consumes as the HRT reference. [PoC §3.5 pp.7–8; measured]
  - [ ] AC2.7 Grounding docstrings cite §2.1 p.3, §2.2 p.4, §3.5 p.8 and state which §2.2 columns are
    unavailable in LIBERO and why. [vla-build]
- **Verify by:** `.venv/bin/python -m vlapoc.catalog --version v0 --verify-raw`, which prints
  `1000/1000 hashes match raw`, the uniqueness count and the per-suite column summaries.
- **Demo value:** the catalog table (one row per episode, the AC2.4 columns) and a per-task
  human-duration summary; the sentence "every stored episode is bit-exact to its raw demo (1000/1000)".

### S3: Define dataset versions, named splits and subsets that resolve to LeRobot `episodes=`
- **Why:** §2.1 p. 3 exact dataset versions; §8.3 p. 14 / §8.7 p. 15 held-out clean references; §2.6
  p. 5 and §9.1 Step 4 p. 19 drop-top-k and random-k subsets; training on a subset (user request).
- **Depends on:** S2
- **Deliverables:** in `vlapoc/lineage.py`: `version_id(...)`, `register_version(...)`,
  `make_splits(...)`, `Subset` (+ `resolve()` → LeRobot indices), `drop_top_k(...)`,
  `drop_random_k(...)`, `verify_version(...)`.
- **Acceptance criteria:**
  - [ ] AC3.1 `dataset_version_id` = sha256 over the feature schema plus the sorted
    (`episode_id`, `content_hash`) pairs. v0 is registered under `data/lineage/versions/<id>/` with its
    episodes table, a per-data-file sha256 list and the LeRobot root (relative). Recomputing the id from
    the dataset reproduces it. [PoC §2.1 p.3]
  - [ ] AC3.2 Named splits are persisted as episode-ID lists with rule and seed: `reference` = 2 episodes
    per task, stratified and seeded (**40**); `candidate` = the rest (**960**); the two are disjoint.
    [RoboDrop App. A.1 p.13 "reserve two uncorrupted episodes per task"; PoC §8.3 p.14]
  - [ ] AC3.3 A subset is (`dataset_version_id`, sorted `episode_id`s, derivation record), with
    `subset_id` = sha256 of those. `LeRobotDataset(root, episodes=subset.resolve())` serves exactly the
    subset: reading every served frame's `episode_index` and mapping it to `episode_id` gives set
    equality, and the frame count equals the sum of `length_steps`. [user request; LeRobot `DatasetConfig.episodes`, quoted]
  - [ ] AC3.4 `drop_top_k(parent, score_table, metric, k, direction)` and `drop_random_k(parent, k, seed)`
    return child subsets whose derivation record names the parent, the score-table digest, the metric,
    k and the seed. The random control is reproducible from its seed. [PoC §2.6 p.5; §9.1 Step 4 p.19]
  - [ ] AC3.5 `verify_version(id)` recomputes the file digests and fails with a clear error when one byte
    of a data file (in a scratch copy) is changed. [PoC §2.1 p.3 "exact dataset versions"]
  - [ ] AC3.6 No manifest, split, subset or version file contains an absolute path. [inferred: training and eval will run on Linux, per LeRobot docs "LIBERO requires Linux"]
  - [ ] AC3.7 Grounding docstrings cite §2.1 p.3, §2.6 p.5, §8.3 p.14 and RoboDrop App. A.1 p.13. [vla-build]
- **Verify by:** `python -m vlapoc.lineage version show v0`; `python -m vlapoc.lineage split make
  --version v0 --reference-per-task 2 --seed 0`; a verify script for AC3.3–3.5.
- **Demo value:** v0's version ID, split sizes 40/960, and a subset that round-trips through LeRobot
  exactly.

### S4: Derive new dataset versions with ground-truth labels (the injection hook)
- **Why:** §2.1 p. 3 item 4 "corrupt known subsets"; §2.5 p. 5; §10 p. 20 "Ground truth". Injection
  must produce new episode versions whose parentage and corruption are recorded.
- **Depends on:** S3
- **Deliverables:** `derive_version(parent_id, edits, name)` in `vlapoc/lineage.py`. It uses LeRobot
  tools where they fit (`delete_episodes` + writer + `merge_datasets`; `modify_tasks` is the documented
  route for later language edits on a copy, since it works "in-place"), otherwise a full rewrite
  through the writer. It writes a `labels.parquet` and one demo derived version `v1`.
- **Acceptance criteria:**
  - [ ] AC4.1 The derived version's `content_hash` changes for exactly the edited episodes and is
    identical for all others (count of changed hashes == count of edits). Its version record stores
    `parent_version_id`, and each edited row stores `parent_content_hash`. The derived dataset passes
    the S1/S2 load and hash checks. [PoC §2.1 p.3]
  - [ ] AC4.2 `labels.parquet` has one row per edited episode (`episode_id`, `parent_content_hash`,
    `content_hash`, `corruption_type`, `params` JSON, `seed`) and joins 1:1 with the version's episodes
    table. [PoC §2.1 p.3 item 4; §10 p.20]
  - [ ] AC4.3 Editing any `reference`-split episode raises an error. [PoC §8.7 p.15 "corrupt only the training copy"; RoboDrop A.1 p.13]
  - [ ] AC4.4 Length-changing edits work: a test inserts 20 idle frames into one episode; the new
    length is recorded, `meta/episodes` is consistent, and the dataset loads. [PoC §8.10 p.17 "pad some with idle or hesitant segments"]
  - [ ] AC4.5 Demo derivation `v1` applies RoboDrop's action corruption literally (additive zero-mean
    Gaussian noise, std = 0.25·σ_a per action dim, σ_a from v0's `meta/stats.json`) to **5 candidate
    episodes per task (100 = 10% of 1000)**, seeded. Result: exactly 100 changed hashes, 900 identical,
    100 label rows. It is labelled as a lineage demo; the injection feature owns the corruption suite.
    [RoboDrop App. A.1 p.13 "corrupting 10% of the episodes for every task … 0.25σa"; PoC §2.5 p.5 "5–10%"]
  - [ ] AC4.6 Grounding docstring cites §2.5 p.5, §8.7 p.15, §8.10 p.17 and RoboDrop A.1 p.13, and notes
    the ambiguity of whether RoboDrop noises the gripper dim (applied to all 7 dims here, literal
    reading, recorded in `params`). [vla-build]
- **Verify by:** `python -m vlapoc.lineage derive --parent v0 --demo-action-noise --per-task 5 --seed 0`,
  then `python -m vlapoc.lineage version diff v0 v1` → `changed 100, identical 900`; the padding test script.
- **Demo value:** "v1 differs from v0 in exactly 100 episode versions, each with a ground-truth label;
  the other 900 hashes are unchanged."

### S5: Record every stage as an MLflow run and answer trace / consumers / diff
- **Why:** §2.1 p. 3 (the tie-in list, and "what changed between the good run and the bad run?");
  §9.1–9.2 pp. 18–19 (screening vs attribution comparison, eval tied to checkpoint).
- **Depends on:** S3 (S4 for derive runs). **BLOCKING Q1 and Q2.**
- **Deliverables:** in `vlapoc/lineage.py`: `stage_run(stage, inputs, params)` context manager over
  MLflow (SQLite at `data/lineage/mlflow.db`); `train_run(...)`, a thin wrapper that runs
  `verify_version`, writes the LeRobot train config JSON, launches `lerobot-train` with
  `--dataset.episodes=<resolved subset>` and logs the checkpoint hashes; `check_scores(...)`; CLI
  `trace`, `consumers`, `diff`. New dependency: `mlflow` (named in the report).
- **Acceptance criteria:**
  - [ ] AC5.1 Each stage execution is one MLflow run tagged `stage` ∈ {build, split, derive, screen,
    select, train, eval, attribute}. Inputs are logged with
    `mlflow.log_input(mlflow.data.from_pandas(<episode_id, content_hash table>, source=<relative path>,
    name=…, digest=<version_id | subset_id>), context=<role>)`. Outputs are artifacts plus
    `output.<name>.sha256` tags. `search_runs(filter_string="datasets.digest = '<id>'")` finds the
    consumers of a version. [PoC §2.1 p.3 "MLflow … handle this well"; MLflow docs, quoted]
  - [ ] AC5.2 A train-run record contains every §2.1 field: `dataset_version_id`, `subset_id` + the
    episode-ID list artifact, mixture weights (uniform over the subset, or LeRobot `sample_weighting`
    config), augmentation (`dataset.image_transforms`), the full LeRobot train config JSON, git SHA
    (`mlflow.source.git.commit`, needs Q2), seed, checkpoint path + sha256 of the weights **and** of the
    saved normalizer stats actually used. [PoC §2.1 p.3; LeRobot `save_checkpoint` saves `train_config.json` + preprocessor, quoted]
  - [ ] AC5.3 An eval-run record contains the sha256 of the checkpoint it loaded, the producing train
    run ID, suite, `libero_task_id`s, init-state indices and seeds. [PoC §2.1 p.3 "Every eval rollout should be tied back to that checkpoint"; "fixed seeds and initial states"]
  - [ ] AC5.4 CLI: `trace <run>` walks upstream to v0 and the raw-file sha256s; `consumers
    <episode_id>@<hash>` lists runs whose input subset contained that episode version; `diff <run_a>
    <run_b>` lists episodes added / removed / changed version plus param, config and git differences.
    [PoC §2.1 p.3 "what changed between the good run and the bad run?"]
  - [ ] AC5.5 `check_scores(table, train_run)` rejects any score row whose (`episode_id`,
    `content_hash`) was not in that run's training subset (e.g. clean-v0 hashes against a run trained
    on v1). [PoC §9.1 Step 3 p.19; §9.2 p.19]
  - [ ] AC5.6 Everything runs offline without an account. [owner constraint; BLOCKING Q1]
  - [ ] AC5.7 Grounding docstrings cite §2.1 p.3, §2.3 p.4 and §9.1–9.2 pp.18–19. [vla-build]
- **Verify by:** a unit-style script that creates two tiny runs and checks each AC5.x field via
  `mlflow.get_run` / `search_runs`; `python -m vlapoc.lineage diff <a> <b>`.
- **Demo value:** a `diff` printout between two runs, the answer to "what changed between the good run
  and the bad run?" at episode-version resolution.

### S6: Run every downstream stage type end to end on the real data (lineage dry run)
- **Why:** the owner's requirement that the layer "enable ALL downstream tasks": each stage must name
  the episode versions it consumed and produced (§9 pp. 18–19, §2.5–2.6 p. 5, §10 p. 20).
- **Depends on:** S4, S5
- **Deliverables:** `scripts/lineage_dry_run.py`; a lineage graph (JSON + Mermaid/text) and CLI outputs
  in `artifacts/`.
- **Acceptance criteria:**
  - [ ] AC6.1 One script records this chain as MLflow runs on the real v0/v1 data: build(v0) →
    split → derive(v1) → screen (stand-in) → train-full (v1 `candidate`, seed 0) and train-full
    (seed 1) → select drop-top-k and select random-k (same k, from the train-full subset) → train on
    each → eval (stand-in) → attribute (stand-in). Stand-ins carry `stand_in=true` and write only
    lineage keys plus clearly marked placeholder payloads. **No outcome numbers are invented**: the
    eval stand-in records the rollout plan (checkpoint sha256, `libero_task_id`, init-state index,
    seed) with outcomes marked not run. [user request; PoC §9 pp.18–19; §2.6 p.5]
  - [ ] AC6.2 Train steps are real `lerobot-train` runs (a small policy such as ACT, a few steps,
    CPU/MPS, `--wandb.enable=false`) on `--dataset.episodes=<resolved subset>`. The checkpoint's saved
    `train_config.json` lists exactly the subset's episode indices (read back and compared). The
    smoke run proves the path from data to trainer to checkpoint, not training quality; the SmolVLA
    feature swaps only the policy. [PoC §2.1 p.3; LeRobot `DatasetConfig.episodes`]
  - [ ] AC6.3 `diff(train-full-s0, train-drop-top-k)` reports exactly k removed episode versions and no
    other data difference; `diff(train-full-s0, train-full-s1)` reports only the seed.
    [PoC §2.1 p.3; §2.3 p.4 "2–3 seeds per data configuration"]
  - [ ] AC6.4 `trace(train-drop-top-k)` reaches v0 and the 20 raw-file sha256s through select →
    attribute → train-full → derive(v1). `consumers(<a corrupted episode>@<v1 hash>)` lists train-full
    and does not list train-drop-top-k when that episode was dropped. [PoC §2.1 p.3]
  - [ ] AC6.5 Tamper test: after one byte of a v1 data file (in a scratch copy) is modified,
    `train_run` refuses to launch. [PoC §2.1 p.3]
  - [ ] AC6.6 The screening stand-in's table joins v1's `labels.parquet` on (`episode_id`,
    `content_hash`) for 960/960 candidate episodes, and the attribution stand-in's table passes
    `check_scores` against train-full: the key contract §9.2's comparison needs. [PoC §9.2 p.19]
  - [ ] AC6.7 A lineage-graph artifact (JSON + rendering) of the whole chain is written for the report. [demo value; PoC §2.1 p.3]
- **Verify by:** `.venv/bin/python scripts/lineage_dry_run.py --k 20 --seed 0`, then the three CLI
  queries; `mlflow ui --backend-store-uri sqlite:///data/lineage/mlflow.db` for a visual check.
- **Demo value:** the lineage graph from raw files to a drop-top-k checkpoint, the `diff` showing
  exactly k removed episode versions, and the tamper refusal.

## Out of scope (deliberately)
- **The corruption suite** (§2.5: time shift, instruction swap, viewpoint/dropped frames, hesitant
  demos, gripper-release noise §8.10). The injection feature owns it; S4 ships the mechanism and one
  demo transform.
- **SmolVLA training at scale, GPU selection, rollout evaluation in the LIBERO simulator, PhAIL
  metrics.** Separate features. The simulator needs Linux (LeRobot docs, quoted); S6 uses stand-ins
  and a tiny `lerobot-train` smoke run.
- **Screening (DemInf, RoboDrop) and attribution (CUPID).** Only their lineage contracts (stand-ins) here.
- **Re-rendering at 256² or from new camera viewpoints** via simulator replay of `states` (Linux + sim).
- **Per-sub-task T for LIBERO-10** (§3.5 p. 7). Needs BDDL predicate replay; the evaluation feature owns it.
- **Dashboard** (§2.7 p. 5); only a lineage graph artifact here.
- **Pushing versions to HF Hub / DVC remotes** for moving data to a GPU box (needs an account; later).
- **robomimic-HDF5 export of derived versions** for CUPID's original codebase (`source_file`/`demo_<k>`
  kept, so this is a thin later step).
- **libero_object / libero_goal**: not on disk; §10 asks for "one or two suites".
- **Video encoding** of the dataset (lossy; see decisions).

## Open questions
- **BLOCKING Q1: Tracking backend.** (a) **MLflow, local SQLite** (recommended; the plan is written for
  it): no account, offline, dataset digests searchable, git SHA auto-tagged, `mlflow` 3.16.1 resolves
  for Py 3.14. (b) **W&B**: best lineage-graph UI, and LeRobot's `--wandb.enable` already logs train
  config and checkpoints. But it needs a W&B account (or self-hosted server), and `use_artifact` fails
  offline (`wandb_run.py:3104`). Choosing W&B changes S5/S6 only.
- **BLOCKING Q2: Put the repo under Git.** §2.1 p. 3 requires "the git SHA" per run; the repo is not a
  Git repository (measured). Recommended: `git init` with `data/`, `.venv/`, `outputs/` ignored (no
  remote needed). If declined, runs record `git_sha = null` and the report flags the gap (AC5.2 partial).
- Non-blocking, raw vs hub data: converting raw (see Prior art). No download.
- Non-blocking, resolution 128² (native): the evaluation feature must render at 128
  (`--env.observation_height=128 --env.observation_width=128`; LeRobot's `LiberoEnv` config default is
  360, quoted). Published SmolVLA/π0 LIBERO results used 256² hub data, so absolute success rates may
  differ (inferred).
- Non-blocking, reference split = 2 per task (RoboDrop A.1). Whether references also join final
  training (RoboDrop does) is for the screening/training features.
- Non-blocking, `operator` / `collection_date` are null: LIBERO publishes neither (LIBERO p. 6).
- Non-blocking, derived versions are full dataset copies (~8 GB each, inferred), acceptable for about
  10 versions on 162 GiB free. Training subsets never copy data.

## Risks / premises to verify during execution
- **Normalization-stats leak in counterfactual runs.** LeRobot uses the full dataset's
  `meta/stats.json` even with `--dataset.episodes` (source, quoted), so episodes dropped by drop-top-k
  still shape normalization. That conflicts with "data is the only variable" (p. 1) unless declared.
  This feature **records** the stats hash per train run (AC5.2); the training/verification feature must
  choose fixed-reference stats or per-subset stats (`recompute_stats` on a copy) and disclose it.
- **`mlflow` install on Py 3.14** (full package pulls Flask, SQLAlchemy, scikit-learn and others).
  Measure it; fall back to `mlflow-skinny` + `sqlalchemy` + `alembic` if a dependency lacks a wheel.
- **`lerobot-train` on macOS / Py 3.14 / MPS** for the S6 smoke run. Fall back to `--policy.device=cpu`;
  ACT's ImageNet backbone download can be avoided with `pretrained_backbone_weights=null`.
- **LeRobot v3 edit tools**: `delete_episodes`/`split` crashed on v3 in upstream issue #2283. Check it
  on 0.6.1; the fallback for S4 is a full rewrite through the writer.
- **PyAV/Homebrew ffmpeg duplicate `libavdevice`** (measured warning). PNG mode avoids decoding video,
  but lerobot imports `av`; watch for crashes.
- **Hash determinism across machines**: PNG is lossless and float32 bytes are little-endian on both
  arm64 and x86_64 (inferred). S2's raw re-hash check is the guard.
- **Task-ID mapping**: `libero_task_id` follows `libero_suite_task_map.py` (default task order). The
  evaluation feature must confirm that LeRobot's `--env.task_ids` uses the same order once LIBERO is
  installed.
- **Time-based corruptions on the hub datasets would be mis-scaled** (declared fps 10 vs 20 Hz data).
  On our fps-20 dataset, RoboDrop's "two seconds" = 40 steps; the injection feature should compute
  offsets from `info.json` fps.
- Conversion time (≈ 18 min) and size (≈ 8 GB) are projections from a 2-demo smoke test (inferred);
  S1 reports the real numbers.
