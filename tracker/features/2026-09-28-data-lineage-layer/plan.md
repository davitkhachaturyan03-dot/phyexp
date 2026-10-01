# Plan: Data layer + lineage layer (LIBERO → LeRobot dataset, episode versions, run lineage)

Planner round 2 (rework after `analysis-r1.md`) · 2026-09-29 · first version 2026-09-28.
Evidence labels: **quoted** (source + quote), **measured** (a command run and reproducible), **inferred**
(reasoning, unverified).

## Changes in this revision

Revision r2 addresses every planner item in [analysis-r1.md](analysis-r1.md) and turns its executor items
into checkable ACs. New or changed ACs are marked **(r2 new)** or **(r2 reworded)**. All other ACs are
unchanged and already reproduced by the analyzer.

1. **P1: false premise removed.** The old Open-questions line (r1 `plan.md:406`) said "Training
   subsets never copy data" and projected "about 10 versions on 162 GiB free". Both are withdrawn.
   - Replaced by new risk **R1** with the measured facts:
     - a ~6.4–6.6 GB HF `datasets` Arrow cache per distinct training subset;
     - streaming ignores `episodes`;
     - `derive_version` leaves orphaned caches;
     - free disk is now 108 GiB.
   - R1 also gives a corrected disk projection, the consequence for datamodels-style many-subset
     training (§2.1 item 5 p.4), and the sampler-level lead for the training feature.
   - Rule 3 is reworded: a subset is recorded as a selection, not a rewrite, but the loader still
     materialises a cache. AC3.3 adds the matching docstring fix.
2. **P2: AC1.2, AC6.2 and AC1.6 reworded to match what was verifiably built.**
   - **AC1.2** now compares against `HuggingFaceVLA/libero@86958911`. The `image` vs `video` difference
     against `lerobot/libero` is recorded as a consequence of AC1.6.
   - **AC6.2** now reads "LeRobot `train()` via `vlapoc/lerobot_train_entry.py`", because the CLI is
     broken on Python 3.14 (draccus #69).
   - **AC1.6** reports sizes as file bytes.
   - New risk **R2**: lerobot 0.6.1 declares only Python 3.12 and 3.13.
3. **P3: counterfactual and causal-baseline selections.**
   - **AC5.8** (r2 new): `select_run` accepts and records `direction`, for the top-k helpful control
     (§2.6 p.5, §9.1 Step 4 p.19).
   - **Group leave-one-out: added, not deferred.** It comes as **AC3.8** (`drop_group`) and **AC5.9**
     (recorded `select` op).
   - **Random subsets: covered by the existing `drop_random_k` op**, with no new operation. **AC5.10**
     demonstrates the datamodels use.
   - The decision and its reasons are under "Decisions and why".
4. **Executor items E1–E4 turned into ACs:**
   - E1 → AC5.8;
   - E2 → **AC4.7** (no ground-truth column in a derived version's episode table);
   - E3 → **AC4.8** (no orphaned derivation cache, or measured disclosure);
   - E4 → **AC3.9** (no session-bound or absolute paths in code; scripts re-run in a fresh shell).
5. **Accepted executor findings folded in:**
   - **AC5.1 reworded:** MLflow caps dataset digests at 36 characters, so the digest is a 32-hex
     prefix and the full id sits in a tag.
   - **S5 deliverables** name the `accelerate` dependency.
   - The normalization-stats risk now carries the measurement.
   - The prior-art line-count estimate is corrected against the measured size.
6. **Owner answers recorded** (brief.md, 2026-09-28/29): MLflow local, `git init` local only, and the
   corruption suite as a separate feature. The BLOCKING questions are closed.
7. **New section "Premises corrected by execution (round 1)",** plus a disposition list for the
   analyzer's non-blocking items.

**Rebuttals to analyzer: none.** I agree with P1–P3 and E1–E4 on the evidence.

One precision, not a rebuttal: the r1 plan already had `direction` at the function level (AC3.4:
`drop_top_k(parent, score_table, metric, k, direction)`). That is implemented at
`vlapoc/lineage.py:424-435` (measured). The gap the analyzer found is real: the tracked `select` stage
hard-codes `"highest"` (`vlapoc/runs.py:519,522`, measured), so the helpful control could not be
recorded as a lineage stage.

## Request

The owner asked (verbatim, [brief.md](brief.md)): *"build the data layer and the lineage layer FEATURE.
The output must be a dataset and a lineage layer that will enable all downstream tasks inside the
poc"*. Concretely:

1. a training-ready dataset built from the LIBERO demonstrations on disk, and
2. a lineage layer such that every downstream PoC stage can name **exactly which episodes, and which
   version of each episode**, it consumed and produced. The stages are: injection with ground-truth
   labels, screening against a clean reference, training on a subset, rollout evaluation, per-episode
   attribution, drop-top-k counterfactual retraining, and the group leave-one-out causal baseline.

It should be PoC-sized, using standard tools plus thin adapters.

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
- **§2.1 items 2–4 (p. 3):**
  - "LIBERO is a good default because it has distinct task suites … SmolVLA via LeRobot is small
    enough for a single GPU." (quoted)
  - "Use enough rollouts per task (roughly 50 or more), fixed seeds and initial states" (quoted)
  - "Deliberately corrupt known subsets of training data … Then check whether your attribution method
    flags exactly those." (quoted)
- **§2.1 item 5 (pp. 3–4):** "Leave-one-group-out ablations. Drop one data source or task group,
  retrain, and measure the change. This is coarse but unambiguous, and it gives you a reference result
  for everything else." (p. 3) · "Randomized subset training with regression (datamodels-style). Train
  many runs on random data subsets, then fit the effect of each group on success." (p. 4) (quoted)
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
  policy should improve more than after removing k random episodes." · "Remove the top-k “helpful”
  episodes and retrain: the policy should degrade more than after random removal." (quoted)
- **§3.5 A (pp. 7–8):** "T = time (or steps) to task success per episode; for long-horizon LIBERO-10
  tasks, per sub-task." … "Human reference = durations of LIBERO's original human-teleoperated
  demonstrations on the same tasks." (quoted)
- **§8.3 (p. 14):** references "are a small, held-out set of clean demonstrations for the target task,
  separate from the candidate data being audited." (quoted)
- **§8.7 (p. 15), row "Benchmark demos (PoC)":** "Treat LIBERO's original demos as clean; corrupt only
  the training copy" → "Free references plus ground truth". (quoted)
- **§8.10 (p. 17):** "pad some with idle or hesitant segments (should hurt speed, not reliability) and
  add gripper-release noise to others" (quoted). Derived episodes may therefore change length.
- **§9.1 (pp. 18–19):**
  - Step 1: screening input is "The training dataset and a small set of trusted clean demos". Its
    verdicts "are stored rather than acted on, so they can later be compared with what actually
    happened" (quoted).
  - Step 3: input is "The training dataset, the trained policy, and the rollouts from Step 2"; output
    is "An episode × metric table of influence scores" (quoted).
  - Step 4: "Remove the top-k most harmful episodes and retrain. As a control, remove k random
    episodes and retrain." … "Repeat for the top-k most helpful episodes (removing them should make
    things worse)." (quoted)
- **§9.2 (p. 19):** "The most compelling output is where screening and outcome attribution
  disagree" (quoted). Screening and attribution outputs must therefore share episode-version keys.
- **§10 "First milestone" (p. 20):** "Foundation | Lineage tracking; episode catalog; LIBERO (one or
  two suites) + SmolVLA"; "Ground truth | Injection experiment with 5–10% corrupted episodes, using
  LIBERO's original demos as clean references"; "Causal baseline | Group-level leave-one-out
  ablations, judged with the macro-KS gate". (quoted)

**Rules the build must follow (derived from the above):**
1. **Identity is content-based.** A run names episodes by a stable `episode_id` **and** a
   `content_hash`. A corrupted copy is a new version of the same `episode_id` (§2.1 p. 3).
2. **References stay clean.** The clean reference set is held out from the candidate/training set and
   is never corrupted (§8.3 p. 14, §8.7 p. 15, §10 p. 20).
3. **(r2 reworded) Subsets are selections, and their derivation is recorded.** A training subset is
   recorded as a selection of episode versions of one dataset version: no dataset rewrite, no
   reindexing. Counterfactual and ablation subsets record how they were derived: drop top-k by which
   score, metric and direction; random-k with which seed; drop which group (§2.6 p. 5, §9.1 Step 4
   p. 19, §2.1 item 5 p. 3). LeRobot's loader still materialises an on-disk cache per distinct
   episode list (R1). "Selection" describes the lineage record, not the disk cost.
4. **Train and eval runs record their provenance.** Every train run records dataset version, episode
   IDs, mixture weights, augmentation, config, git SHA, seed and checkpoints. Every eval run records
   the checkpoint it loaded (§2.1 p. 3, §2.3 p. 4).
5. **The method under test never sees the ground truth.** Labels live apart from the catalog that
   screening and attribution read (§2.1 item 4 p. 3; r2, from analysis-r1 E2).
6. **No filtering on an unverified premise** (CLAUDE.md / vla-build: the OpenVLA 189-demo filter).

**Papers:**
- **RoboDrop** (`docs/papers/RoboDrop_Gradient_Compatibility_arXiv-2609.10021.pdf`):
  - App. A.1 (p. 13): "corrupting 10% of the episodes for every task. Action corruption adds zero-mean
    Gaussian noise with standard deviation 0.25σa to the demonstrated actions, where σa is the
    empirical action standard deviation. Temporal corruption shifts the action sequence by two seconds
    relative to the observations. For validation, we reserve two uncorrupted episodes per task, giving
    20 clean validation episodes for each suite." (quoted) This sets the reference-split size (2 per
    task) and the demo derivation transform in S4.
  - App. p. 20: the final policy "is post-trained on the retained candidate episodes together with the
    clean validation episodes" (quoted). Whether our references join training is left to the
    screening/training features.
- **LIBERO** (`docs/papers/LIBERO_Benchmark_arXiv-2306.03310.pdf`) p. 6: "50 trajectories of high-quality
  demonstrations for every single task … collected by human experts through teleoperation with
  3Dconnexion Spacemouse." (quoted) This fills the catalog's `source`. Operator identity and dates are
  not published.
- **CUPID** (`docs/papers/CUPID_Influence_Functions_arXiv-2506.19121.pdf`), released code
  (`diffusion_policy/dataset/robomimic_replay_image_dataset.py`): identifies demos by `demo_{i}` HDF5
  keys and selects subsets via `get_dataset_masks(...)` train/val/holdout masks (quoted, source via
  fetch). So `(source_file, demo_key)` stays on every episode, letting a CUPID fork map back.
- **(r2) DataMIL** (`docs/papers/DataMIL_Datamodels_arXiv-2505.09603.pdf`):
  - p. 4: "we first sample m random subsets of the prior dataset D_j ⊂ D; for each of these datasets,
    we train a policy A(D_j), and evaluate the target metric M" (quoted).
  - App. G (p. 25): "In our MetaWorld experiments, we found m = 10^4 sufficient to obtain high-quality
    datamodels." (quoted)
  - Uses here: AC5.10 records random subsets; R1 sizes the disk consequence.

## Premises verified against the artifacts (planning session, 2026-09-28)

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
| LeRobot 0.6.1 writer works on this env in PNG mode (2-demo smoke test, bit-exact read-back) | scratchpad smoke test | measured |
| Environment at planning time: Python 3.14.0, lerobot 0.6.1; mlflow/wandb/dvc not installed; repo not a git repository; 162 GiB free; Apple M4 Pro, 24 GB, no CUDA | `pip list`, `df`, `sysctl` | measured |
| PyAV's bundled libavdevice clashes with Homebrew ffmpeg | objc warning on importing lerobot | measured |

## Premises corrected by execution (round 1) (r2 new)

| Planning premise | What execution and the analyzer measured | Label | Plan change |
|---|---|---|---|
| "Training subsets never copy data" | `LeRobotDataset(episodes=…)` → `Dataset.from_parquet(filters=…)` (`lerobot/datasets/io_utils.py:79-82`) writes an Arrow cache per distinct episode list: 6.6 / 6.4 / 6.5 GB for the three S6 subsets, 6.9 GB for a full-version load, 280 MB for `v0/reference`. Checked today in `data/hf_datasets_cache/parquet/` with `du -sh` | measured (sizes) + quoted (source) | R1; Rule 3; AC3.3 addendum |
| Streaming could avoid it | `StreamingLeRobotDataset` stores `self.episodes = episodes` (`streaming_dataset.py:308`), and grep finds no other read of it | quoted (source); analyzer measured | R1 |
| Derivations cost one full copy (~7 GB) | Plus orphaned caches of the temporary datasets: 13 entries from a 20-episode derivation (analyzer); ≈ 2 × 6.3 GB per full-scale derivation | measured (small scale) / inferred (full scale) | AC4.8; R1 |
| `lerobot-train` CLI usable (planning checked only that the module imports) | `lerobot-train --help` → `TypeError: str \| None is not callable` (re-run today). draccus issue #69 "Python 3.14: any `Optional[X]` / union field raises `TypeError: ... is not callable` at parser construction", open since 2026-07-28. `PreTrainedConfig.from_pretrained` fails the same way | measured + quoted | AC6.2 reworded; R2 |
| lerobot supports Py 3.14 | `lerobot-0.6.1.dist-info/METADATA`: `Classifier: … Python :: 3.12`, `… :: 3.13`, `Requires-Python: >=3.12`, `Requires-Dist: draccus<0.12.0,>=0.11.6` | measured | R2 |
| MLflow honours the caller digest as-is | Honoured, but capped at 36 characters (`MAX_DATASET_DIGEST_SIZE = 36`, DB column `String(36)`) | measured by the executor, reproduced by the analyzer | AC5.1 reworded |
| LeRobot `train()` runs with the installed packages | Needs `accelerate` (`lerobot_train.py:231 require_package("accelerate")`) | measured | S5 deliverables |
| AC1.2 comparable against `lerobot/libero` | `lerobot/libero` stores images as `video` (AV1), so a lossless dataset necessarily differs in image dtype | measured | AC1.2 reworded |
| Conversion ≈ 18 min / 8.0 GB | 7.9 min wall, 7.36 GB file bytes | measured | AC1.6 wording |
| Lineage code ≈ 400 lines | 1,724 non-docstring code lines in `vlapoc/` (analyzer count); no verified off-the-shelf replacement (analysis-r1 audit) | measured | Prior-art verdict updated |
| Free disk 162 GiB | 108 GiB today (`df -h`) | measured | R1 projection |

## Prior art

| Candidate | What it covers | What it misses | Evidence |
|---|---|---|---|
| **`lerobot/libero`** (HF, rev `a1aaacb7`) | LeRobot v3 LIBERO, 4 suites, schema the LeRobot LIBERO env expects, 1.94 GB, MP4/AV1 | Only **811 of our 1000 demos** (432/500 spatial, 379/500 libero_10) = OpenVLA no-noops corpus; 154,439 of 200,340 frames; **68/811 episodes have a length no raw demo of that task has** (steps removed); `fps: 10` although the data is 20 Hz control; 256² re-render; `meta/episodes` has 14 columns and **no mapping back to raw demo IDs** | `info.json`/`meta/episodes` at pinned rev; per-task counts via `HuggingFaceVLA/libero` meta (same 1,693/273,465); LeRobot docs: "episodes / frames / tasks: 1,693 / 273,465 / 40" (quoted); measured |
| **`HuggingFaceVLA/libero`** (rev `86958911`) | Same episodes, PNG in parquet; **the schema reference for AC1.2** (docs: "They contain the same demonstrations with the same schema and differ in how camera frames are stored", quoted) | Same gaps as above; **34.93 GB** on the Hub (LeRobot docs say "69.9 GB"; the Hub file sizes win, discrepancy recorded) | `HfApi.dataset_info(files_metadata=True)`; measured |
| **openpi `convert_libero_data_to_lerobot.py`** / `physical-intelligence/libero` | Converter template | Reads OpenVLA's `modified_libero_rlds` (filtered), sets `fps=10`, v2.x format | script source (quoted) |
| **`openvla/modified_libero_rlds`** | RLDS no-noops LIBERO | `libero_spatial_no_noops` = 432 episodes / 52,970 frames, identical to the hub's spatial slice. Adopting it adopts the 189-demo filter CLAUDE.md records as premise-false | HF page (quoted) + meta arithmetic (measured) |
| **any4lerobot `libero2lerobot/libero_h5.py`** (MIT) | Raw-HDF5 → LeRobot v3 adapter | Declares 256² shapes; **does not rotate frames**; key `wrist_image` ≠ env's `image2`; needs `datatrove`; no identity/hashes | source read (measured); README (quoted) |
| **LeRobot v3 format + writer** (lerobot 0.6.1) | Storage format SmolVLA trains on; writer; `episodes=` subset loading; `episode_filter`; `delete_episodes`, `split_dataset`, `merge_datasets`, `modify_tasks`, `recompute_stats`; `EpisodeAwareSampler(episode_indices_to_use=…)`; checkpoints save `train_config.json` + normalizer | **Episode identity is positional** (edit tools reindex); no per-episode content hash, parent links or labels; **normalization stats always come from full `meta/stats.json`**; **(r2) `episodes=` materialises an Arrow cache per distinct episode list (R1)** | source (quoted); smoke test + S1–S6 (measured) |
| **robomimic filter keys** | Named subsets in HDF5 (`mask/<key>`), used by CUPID's code | Not consumed by LeRobot/SmolVLA; no versioning; our files have no `mask` group | robomimic docs (quoted); measured |
| **RLDS / TFDS** | Episode-of-steps spec, `episode_metadata`, semantic dataset versions | No per-episode content hash or derivation links; not LeRobot's training format | web docs (quoted) |
| **MLflow 3.16.1** (chosen, owner answer Q1) | Runs, params, tags incl. auto `mlflow.source.git.commit`; artifacts; dataset inputs `mlflow.log_input`; `search_runs(filter_string="datasets.digest IN (...)")`; local SQLite default since 3.7; no account | No output-dataset lineage; digest ≤ 36 chars; the pandas digest hashes only `df.head(MAX_ROWS)`, `MAX_ROWS = 10000`; no episode concept | docs + source (quoted); installed and used in S5/S6 (measured) |
| **W&B Artifacts** | Best lineage DAG; reference artifacts; LeRobot built-in `--wandb.enable` | Needs an account or server: `use_artifact` raises `TypeError("Cannot use artifact when in offline mode.")` (`wandb/sdk/wandb_run.py:3104-3105`) | source (measured by fetch); docs (quoted) |
| **DVC 3.67** | md5 per file/dir, pipelines, remotes | File-level: v3 packs many episodes per parquet file. **Also (measured, round 1):** LeRobot parquet files are not byte-reproducible (random HF fingerprint in the schema metadata), so file md5 changes on every rebuild. `dvc diff` needs Git | docs (quoted); execution finding 2 (measured) |
| **HF Hub revisions** | Commit-SHA pinning | Needs account and upload; dataset-level only | LeRobot docs (quoted) |
| **RoboLineage** (arXiv 2606.22142) | Typed lineage graph incl. "dataset locks … with adapter context and hashes" | Requires LLM/VLM agents; research prototype; no MLflow/LeRobot integration | arXiv HTML + project page (quoted) |
| **(r2) DataChain / lakeFS / Deep Lake** (analyzer's audit) | Record-level dataset versioning | Auto-increment (not content-addressed) versions; don't feed LeRobot's `episodes=`; a second lineage system beside MLflow | analysis-r1 audit (quoted docs) |

**Verdict: adapter over LeRobot v3 (dataset) plus MLflow (run lineage).** The only custom code is what
none of the tools provide:
- a stable `episode_id` and per-episode `content_hash`;
- version and subset manifests, and derivation labels;
- the selection operations;
- the lineage queries.

The r1 size estimate (~400 lines) was low. The build is 1,724 non-docstring lines (measured by the
analyzer), and the analyzer's independent audit found no replacement that keeps the spec rules. The
refactorer is asked only to simplify (CLI, resolvers).

Decisions and why:
- **Convert our raw HDF5 with LeRobot's own writer; don't download the hub LIBERO datasets.** The hub
  corpus drops 189 of our 1000 demos and shortens 68 others (measured). That is the filter CLAUDE.md
  rules out. It would also bias the HRT human-reference durations (§3.5 p. 8) and break traceability
  to `demo_N`.
- **Match the LeRobot LIBERO schema exactly** (keys `observation.images.image` / `.image2`;
  `observation.state` = eef pos ‖ axis-angle ‖ gripper qpos (8); `action` (7); frames rotated 180°),
  so LeRobot's trainer and LIBERO env need no remapping. From `processor/env_processor.py`: "Rotates
  images by 180 degrees by flipping both height and width dimensions"; `state = torch.cat((eef_pos,
  eef_axisangle, gripper_qpos), dim=-1)` (quoted).
- **Declare fps 20**, the true control rate. From `lerobot/envs/configs.py`: `fps: int = 20  # Must
  match robosuite's default control_freq (20 Hz)` (quoted).
- **Keep native 128×128, stored losslessly** (PNG in parquet). Measured: 7.36 GB, bit-exact to raw for
  1000/1000 episodes.
- **MLflow local** for run lineage (owner answer Q1). No account, works offline, records dataset digests
  with native search.
- **(r2) Group leave-one-out is added as a recorded `select` operation (AC3.8, AC5.9), not deferred.**
  - §10 p. 20 puts "Causal baseline | Group-level leave-one-out ablations" in the first milestone.
  - §2.1 item 5 p. 3 calls it "a reference result for everything else".
  - The owner asked that this layer "enable all downstream tasks inside the poc".
  - Without the op, a hand-made group subset has no producer and `trace` dead-ends (analysis-r1,
    measured: `select_run` raises on other ops, `runs.py:526-527`).
  - Cost: one ~20-line function over the existing `Subset` machinery, plus one branch in `select_run`
    (inferred). It needs no data access or training: a group is a filter on catalog columns that
    already exist (`suite`, `task_name`, `scene`).
- **(r2) Random-subset selection reuses `drop_random_k`; no new op (AC5.10).**
  - A uniformly random subset of size n is exactly `drop_random_k(parent, k = N − n, seed)`.
  - The record already holds the seed, the RNG rule and the dropped IDs, and the membership vector
    DataMIL's regression needs is derivable from the subset's `episode_ids`.
  - A second op would duplicate code for no new information (inferred).
  - Training many such subsets is a disk problem (R1) owned by the training feature.

## Stories

**Layout as built:**
- **Code:** `vlapoc/libero_convert.py` (S1), `vlapoc/catalog.py` (S2), `vlapoc/lineage.py` (S3, CLI),
  `vlapoc/derive.py` (S4), `vlapoc/runs.py` and `vlapoc/lerobot_train_entry.py` (S5),
  `scripts/lineage_dry_run.py` (S6).
- **Data:** `data/lerobot/<dataset_name>/`, `data/lineage/versions/<version_id>/`,
  `data/lineage/subsets/<subset_id>.json`, `data/lineage/mlflow.db` + `data/lineage/mlartifacts/`.
  The HF cache is at `data/hf_datasets_cache/` (gitignored).
- **Evidence:** in `tracker/features/2026-09-28-data-lineage-layer/artifacts/`.

**Round-2 work for the executor:** AC3.3 addendum, AC3.8, AC3.9, AC4.7, AC4.8, AC5.8, AC5.9 and
AC5.10, plus re-running the checks the analyzer names under E2. Reworded ACs (AC1.2, AC1.6, AC5.1,
AC6.2) describe what is already built and need no new code.

### S1: Convert raw LIBERO (libero_spatial + libero_10) into a LeRobot v3 dataset, unfiltered
- **Why:** §10 p. 20 "LIBERO (one or two suites) + SmolVLA"; §2.2 p. 4 "LeRobot's dataset format";
  user ask (1): "the output must be a dataset".
- **Depends on:** —
- **Deliverables:**
  - `vlapoc/libero_convert.py`: h5py read → rotate 180° → `LeRobotDataset.create(fps=20,
    use_videos=False, robot_type="panda")` → `add_frame` → `save_episode` → `finalize`.
  - The dataset root `data/lerobot/libero_spatial_10/` (v0).
  - A verify script and an orientation figure.
- **Acceptance criteria:**
  - [ ] AC1.1 `meta/info.json` reports `codebase_version` v3.0, `fps` 20, `total_episodes` 1000,
    `total_frames` 200340, `total_tasks` 20; `LeRobotDataset(repo_id, root=…)` loads. [PoC §10 p.20; measured raw counts]
  - [ ] AC1.2 **(r2 reworded)** Schema equals the PNG-in-parquet reference, except image shape and fps.
    - Feature keys and names, and the dtype and shape of `observation.state` (float32[8]) and
      `action` (float32[7]), equal `HuggingFaceVLA/libero`@`86958911`'s `info.json`.
    - The only differences from that reference are image shape (128×128×3 vs 256×256×3) and `fps`
      (20 vs 10).
    - Against `lerobot/libero`@`a1aaacb7` the image dtype also differs (`image` vs `video`). That
      follows from AC1.6's lossless storage and is recorded, not treated as a defect.
    - The verify script prints both diffs.
    [LeRobot LIBERO docs: "They contain the same demonstrations with the same schema and differ in how camera frames are stored" (quoted); AC1.6]
  - [ ] AC1.3 `observation.state` == `ee_pos ‖ ee_ori ‖ gripper_states` cast to float32, and `action` ==
    raw actions cast to float32, for every frame of 3 spot-checked episodes per suite. [LeRobot docs "8-dim … (eef position, axis-angle orientation, gripper qpos)"; measured axis-angle]
  - [ ] AC1.4 Frames are the raw frames rotated 180°. The orientation figure in `artifacts/` shows raw vs
    converted vs a `HuggingFaceVLA/libero` reference frame for the same task. [env_processor.py, quoted]
  - [ ] AC1.5 No filtering. Every one of the 20 tasks has 50 episodes, and each episode's length
    equals its raw `num_samples` (1000/1000). The log re-verifies that every demo ends with its only
    `reward == 1` at the last step. [CLAUDE.md / vla-build rule; measured]
  - [ ] AC1.6 **(r2 reworded)** Storage is lossless (`use_videos=False`). The on-disk size is reported
    as the sum of file bytes (not a `du` figure), and the conversion wall time is reported. [analysis-r1 non-blocking note; needed by AC2.2]
  - [ ] AC1.7 Grounding docstring cites PoC §2.2 p.4, §10 p.20 and the LeRobot LIBERO schema source, and
    states the divergences from the hub datasets (all 1000 demos, 128², fps 20, lossless). [vla-build]
- **Verify by:** `.venv/bin/python -m vlapoc.libero_convert --out data/lerobot/libero_spatial_10`, then
  `scripts/verify_libero_convert.py`.
- **Demo value:** "1000 demos / 200,340 frames converted, 0 dropped, against 811 / 154,439 in the hub
  `lerobot/libero` for the same two suites", plus the orientation figure.

### S2: Give every episode a stable ID, a content hash and a §2.2 catalog row
- **Why:** §2.1 p. 3 "exact dataset versions (content hashes), episode IDs"; §2.2 p. 4 catalog; §3.5
  p. 8 human-reference durations for HRT; §1 p. 3 aggregation by metadata.
- **Depends on:** S1
- **Deliverables:** `vlapoc/catalog.py`: `episode_content_hash(...)` and `build_catalog(version)` →
  `episodes.parquet`, plus the raw-file sha256 list.
- **Acceptance criteria:**
  - [ ] AC2.1 `episode_id = "<suite>/<task_name>/demo_<k>"`, taken from the raw source rather than
    LeRobot's positional `episode_index`. There are 1000 unique IDs, and each row keeps `source_file`
    (relative path) and `source_demo` (`demo_<k>`). [PoC §2.2 p.4 "stable ID"; CUPID code `demo_{i}` keys]
  - [ ] AC2.2 `content_hash` = sha256 over the canonical episode content: the task string, then per
    frame `action` f32, `observation.state` f32, `image` uint8 HWC and `image2` uint8 HWC, in frame
    order. Bookkeeping columns are excluded. The hash computed from the stored LeRobot dataset
    **equals** the hash recomputed directly from raw HDF5 for **1000/1000** episodes. [PoC §2.1 p.3 "content hashes"]
  - [ ] AC2.3 The sha256 of each of the 20 raw HDF5 files is recorded. [PoC §2.1 p.3]
  - [ ] AC2.4 `episodes.parquet` has one row per episode with these columns:
    - identity and indexing: `episode_id`, `episode_index`, `content_hash`, `suite`, `task_name`,
      `libero_task_id`;
    - description: `language_instruction`, `scene`, `objects`, `source`, and `operator` and
      `collection_date` (null, with a note that they are not in the source);
    - measurements: `length_steps`, `duration_s`, `init_state_sha256`, `smoothness`, `idle_fraction`,
      `gripper_toggles`.
    [PoC §2.2 p.4; §3.5 p.8; LIBERO p.6]
  - [ ] AC2.5 Catalog values match the planning measurements:
    - `duration_s` ranges 3.75–9.85 s (libero_spatial) and 7.5–25.85 s (libero_10);
    - zero-action step totals are 97 and 339;
    - gripper toggles are 1–7 and 1–8.
    [measured]
  - [ ] AC2.6 `duration_s` is documented as the human time-to-success T (the last step is the first
    `reward == 1`, re-verified 1000/1000). [PoC §3.5 pp.7–8; measured]
  - [ ] AC2.7 Grounding docstrings cite §2.1 p.3, §2.2 p.4 and §3.5 p.8, and state which §2.2 columns
    LIBERO doesn't provide and why. [vla-build]
- **Verify by:** `.venv/bin/python -m vlapoc.catalog --version v0 --verify-raw`.
- **Demo value:** the catalog table and a per-task human-duration summary; "every stored episode is
  bit-exact to its raw demo (1000/1000)".

### S3: Define dataset versions, named splits and subsets that resolve to LeRobot `episodes=`
- **Why:** §2.1 p. 3 exact dataset versions; §8.3 p. 14 / §8.7 p. 15 held-out clean references; §2.6
  p. 5 and §9.1 Step 4 p. 19 counterfactual subsets; §2.1 item 5 p. 3 / §10 p. 20 group leave-one-out;
  training on a subset (user request).
- **Depends on:** S2
- **Deliverables:** in `vlapoc/lineage.py`:
  - `version_id`, `register_version`, `make_splits`;
  - `Subset` (+ `resolve()`);
  - `drop_top_k`, `drop_random_k`, **`drop_group` (r2)**;
  - `verify_version`.
- **Acceptance criteria:**
  - [ ] AC3.1 `dataset_version_id` = sha256 over the feature schema plus the sorted
    (`episode_id`, `content_hash`) pairs. Recomputing it from the dataset reproduces it. [PoC §2.1 p.3]
  - [ ] AC3.2 Named splits are persisted as episode-ID lists with their rule and seed:
    - `reference` = 2 per task, stratified and seeded (**40**);
    - `candidate` = the rest (**960**);
    - the two are disjoint.
    [RoboDrop App. A.1 p.13; PoC §8.3 p.14]
  - [ ] AC3.3 A subset is (`dataset_version_id`, sorted `episode_id`s, derivation record), with
    `subset_id` = sha256 of those. `LeRobotDataset(root, episodes=subset.resolve())` serves exactly the
    subset (set equality of served `episode_id`s; frame count = sum of `length_steps`).
    - **(r2 addendum)** The `Subset` docstring (`lineage.py:283`, currently "Never a data copy") says
      the record is a selection (no rewrite, no reindex), and that LeRobot's loader materialises an
      Arrow cache per distinct episode list (R1).
    [user request; LeRobot `DatasetConfig.episodes`; analysis-r1 P1]
  - [ ] AC3.4 `drop_top_k(parent, score_table, metric, k, direction)` and `drop_random_k(parent, k, seed)`
    return child subsets. Each derivation record names the parent, the score-table digest, the metric,
    the direction, k and the seed, and the random control is reproducible from its seed. [PoC §2.6 p.5; §9.1 Step 4 p.19]
  - [ ] AC3.5 `verify_version(id)` recomputes the file digests, and fails with a clear error when one
    byte of a data file (in a scratch copy) is changed. [PoC §2.1 p.3]
  - [ ] AC3.6 No manifest, split, subset or version file contains an absolute path. [inferred: training and eval will run on Linux]
  - [ ] AC3.7 Grounding docstrings cite §2.1 p.3, §2.6 p.5, §8.3 p.14 and RoboDrop App. A.1 p.13. [vla-build]
  - [ ] AC3.8 **(r2 new)** `drop_group(parent, column, value)` returns the group-ablation child.
    - It drops every parent episode whose row in the version's episode table has `column == value`.
    - `column` must be a catalog column (e.g. `suite`, `task_name`, `scene`). Otherwise it raises;
      after AC4.7 no ground-truth column exists to group by.
    - It raises when `value` matches no parent episode, or every one.
    - The derivation records `op`, `column`, `value`, the parent subset id and the dropped IDs.
    - Check on `v1/candidate`, in a scratch lineage root:
      - `task_name = <one libero_10 task>` → 912 episodes (960 − 48, since the task keeps 2 of 50 as
        references);
      - `suite = libero_10` → 480 episodes;
      - for both children, `resolve()` returns exactly the `episode_index` set of the remaining
        episodes. This is checked against the version's episode table, without loading through
        LeRobot, which would write a ~6 GB cache per child (R1).
    [PoC §2.1 item 5 p.3 "Drop one data source or task group, retrain, and measure the change"; §10 p.20 "Causal baseline | Group-level leave-one-out ablations"]
  - [ ] AC3.9 **(r2 new, E4)** No module or script hard-codes an absolute or session-bound path.
    - `grep -rnE '/private/tmp|/tmp/claude|/Users/' vlapoc scripts` prints nothing.
    - Scratch space comes from `tempfile.mkdtemp()` or a gitignored repo-relative directory.
    - `scripts/verify_lineage_s3.py`, and the tamper step of `scripts/lineage_dry_run.py` (AC6.5),
      complete in a fresh shell where the round-1 session scratchpad does not exist.
    [analysis-r1 E4 (measured: `cp -c -R` "No such file or directory", exit 1, after 11 runs recorded); AC3.6 extends the rule to code]
- **Verify by:**
  - `python -m vlapoc.lineage version show v0`;
  - `python -m vlapoc.lineage split make --version v0 --reference-per-task 2 --seed 0`;
  - `scripts/verify_lineage_s3.py`, extended for AC3.8 and run from a fresh shell for AC3.9;
  - `grep -rnE '/private/tmp|/tmp/claude|/Users/' vlapoc scripts`.
- **Demo value:** v0's version ID, split sizes 40/960, a subset that round-trips through LeRobot exactly,
  and the leave-one-task-out and leave-one-suite-out children (912 / 480).

### S4: Derive new dataset versions with ground-truth labels (the injection hook)
- **Why:** §2.1 p. 3 item 4 "corrupt known subsets"; §2.5 p. 5; §10 p. 20 "Ground truth". Injection
  must produce new episode versions whose parentage and corruption are recorded, while the method
  under test stays blind to the labels.
- **Depends on:** S3
- **Deliverables:** `derive_version(parent_id, edits, name)` in `vlapoc/derive.py`, built over LeRobot
  `delete_episodes` + writer + `merge_datasets`. It writes a `labels.parquet` and one demo derived
  version `v1`.
- **Acceptance criteria:**
  - [ ] AC4.1 The derived version's `content_hash` changes for exactly the edited episodes, and stays
    identical for all others.
    - The version record stores `parent_version_id`; each row stores `parent_content_hash`.
    - The derived dataset passes the S1/S2 load and hash checks.
    [PoC §2.1 p.3]
  - [ ] AC4.2 `labels.parquet` has one row per edited episode (`episode_id`, `parent_content_hash`,
    `content_hash`, `corruption_type`, `params` JSON, `seed`), and joins 1:1 with the version's episodes
    table. [PoC §2.1 p.3 item 4; §10 p.20]
  - [ ] AC4.3 Editing any `reference`-split episode raises an error. [PoC §8.7 p.15; RoboDrop A.1 p.13]
  - [ ] AC4.4 Length-changing edits work: inserting 20 idle frames into one episode records the new
    length, keeps `meta/episodes` consistent, and the dataset loads. [PoC §8.10 p.17]
  - [ ] AC4.5 Demo derivation `v1` applies RoboDrop's action corruption literally to **5 candidate
    episodes per task (100 = 10% of 1000)**, seeded.
    - The corruption is additive zero-mean Gaussian noise with std = 0.25·σ_a per action dim, with σ_a
      from v0's `meta/stats.json`.
    - Result: exactly 100 changed hashes, 900 identical, and 100 label rows.
    - It is labelled as a lineage demo; the injection feature owns the corruption suite (owner answer,
      2026-09-29).
    [RoboDrop App. A.1 p.13; PoC §2.5 p.5]
  - [ ] AC4.6 Grounding docstring cites §2.5 p.5, §8.7 p.15, §8.10 p.17 and RoboDrop A.1 p.13, and
    notes the gripper-dim ambiguity (all 7 dims noised, recorded in `params`). [vla-build]
  - [ ] AC4.7 **(r2 new, E2)** A derived version's episode table carries no ground truth.
    - This covers `data/lineage/versions/<id>/episodes.parquet` and its catalog copy. Neither has a
      `corruption_type`, `params` or `seed` column; the columns equal the parent's catalog columns
      plus `parent_content_hash`.
    - Ground truth lives only in `labels.parquet`.
    - Remove the column from `derive.py:265-266`, and rewrite v1's two existing tables without it.
    - Expected afterwards:
      - v1's id recomputes unchanged (`match: true`);
      - the labels join is still 100/100;
      - `version diff v0 v1` still reports 100 changed / 900 identical;
      - a column listing of both tables shows no label column.
    [PoC §2.1 item 4 p.3 "check whether your attribution method flags exactly those", so the method must not read the answer; §9.1 Step 1 p.18 screening input is "The training dataset and a small set of trusted clean demos"]
  - [ ] AC4.8 **(r2 new, E3)** `derive_version` leaves no orphaned HF `datasets` cache.
    - Check: in a scratch lineage root, run `derive_version` on a small registered parent (e.g. 20
      episodes). The project cache's entry list and byte total are the same before and after.
    - Lead for the fix (inferred, untested): `datasets/builder.py:365` resolves the cache root when each
      builder is constructed (`self._cache_dir_root = str(cache_dir or config.HF_DATASETS_CACHE)`,
      quoted). Pointing `datasets.config.HF_DATASETS_CACHE` at a temp dir for the duration of the
      derivation, then deleting it with `.tmp_kept`/`.tmp_edit`, should route every derivation-time
      cache.
    - Fallback if that proves infeasible: disclose instead.
      - The `derive_version` docstring and the derive run's params record the measured orphaned bytes
        per derivation.
      - The execution log says why the fix failed.
      - Round-1 finding 6 ("a full dataset copy (6.8 GB)") is corrected to include the cache cost.
    [analysis-r1 E3 (measured: 13 cache entries from a 20-episode derivation); disk budget R1]
- **Verify by:**
  - `python -m vlapoc.lineage version diff v0 v1` → `changed 100, identical 900`;
  - `scripts/verify_derive_s4.py --v1 --guard --padding`;
  - for AC4.7: a column listing of v1's tables plus `version recompute v1`;
  - for AC4.8: a before/after listing of `data/hf_datasets_cache/` (or the scratch cache) around the
    small derivation.
- **Demo value:** "v1 differs from v0 in exactly 100 episode versions, each with a ground-truth label held
  outside the catalog the method reads; the other 900 hashes are unchanged."

### S5: Record every stage as an MLflow run and answer trace / consumers / diff
- **Why:** §2.1 p. 3 (the tie-in list, and "what changed between the good run and the bad run?");
  §9.1–9.2 pp. 18–19 (screening vs attribution comparison, eval tied to checkpoint, top-k helpful
  control); §2.1 item 5 / §10 p. 20 (group leave-one-out, random subsets).
- **Depends on:** S3 (S4 for derive runs). Owner answers Q1 (MLflow local) and Q2 (git local) are in
  brief.md.
- **Deliverables:** in `vlapoc/runs.py`:
  - the `stage_run` context manager over MLflow (SQLite at `data/lineage/mlflow.db`);
  - `train_run` (verifies the version, calls LeRobot `train()` through
    `vlapoc/lerobot_train_entry.py`, logs checkpoint hashes);
  - `select_run` (**r2: `direction`, `drop_group`**);
  - `check_scores`;
  - the `trace`, `consumers` and `diff` queries and their CLI.

  New dependencies, each named in the report: `mlflow` (owner answer Q1), and `accelerate`, which
  LeRobot `train()` requires (`require_package("accelerate")`).
- **Acceptance criteria:**
  - [ ] AC5.1 **(r2 reworded digest)** Each stage execution is one MLflow run, tagged `stage` ∈ {build,
    split, derive, screen, select, train, eval, attribute}.
    - Inputs are logged with `mlflow.log_input(mlflow.data.from_pandas(<episode_id, content_hash
      table>, source=<relative path>, name=…, digest=<first 32 hex chars of version_id |
      subset_id>), context=<role>)`. The full 64-hex id goes in the input tag `vlapoc.id`, because
      MLflow caps dataset digests at 36 characters (`MAX_DATASET_DIGEST_SIZE = 36`). Queries confirm
      the full id.
    - Outputs are artifacts plus `output.<name>.sha256` tags.
    - `search_runs(filter_string="datasets.digest = '<prefix>'")` finds the consumers of a version.
    [PoC §2.1 p.3; MLflow docs, quoted; execution finding 3]
  - [ ] AC5.2 A train-run record contains every §2.1 field:
    - `dataset_version_id`, and `subset_id` plus the episode-ID list artifact;
    - mixture weights and augmentation;
    - the full LeRobot train config JSON;
    - the git SHA (`mlflow.source.git.commit`, equal to HEAD with a clean tree) and the seed;
    - the checkpoint path, with sha256 of the weights **and** of the saved normalizer stats.
    [PoC §2.1 p.3]
  - [ ] AC5.3 An eval-run record contains the sha256 of the checkpoint it loaded, the producing train
    run ID, suite, `libero_task_id`s, init-state indices and seeds. [PoC §2.1 p.3]
  - [ ] AC5.4 The CLI answers three queries:
    - `trace <run>` walks upstream to v0 and the raw-file sha256s;
    - `consumers <episode_id>@<hash>` lists runs whose input subset contained that episode version;
    - `diff <run_a> <run_b>` lists episodes added, removed or changed in version, plus param, config
      and git differences.
    [PoC §2.1 p.3]
  - [ ] AC5.5 `check_scores(table, train_run)` rejects any score row whose (`episode_id`,
    `content_hash`) was not in that run's training subset. [PoC §9.1 Step 3 p.19; §9.2 p.19]
  - [ ] AC5.6 Everything runs offline without an account. [owner answer Q1]
  - [ ] AC5.7 Grounding docstrings cite §2.1 p.3, §2.3 p.4 and §9.1–9.2 pp.18–19. [vla-build]
  - [ ] AC5.8 **(r2 new, P3/E1)** `select_run(op="drop_top_k", …, direction=…)` records the direction.
    - It accepts `direction` ∈ {`"highest"`, `"lowest"`}, default `"highest"`, and passes it to
      `drop_top_k` (today it is hard-coded at `runs.py:519,522`).
    - It logs `direction` as an MLflow param; the child's derivation records it.
    - Check in a scratch lineage root: a `"lowest"` select with k = 20 drops exactly the 20 lowest
      scores (tie-break `episode_id` ascending). `derivation["direction"] == "lowest"`, and the run's
      params show it.
    - The docstring says that which direction is "harmful" or "helpful" depends on the score's sign
      convention. The record therefore always carries both `metric` and `direction`.
    [PoC §2.6 p.5 "Remove the top-k “helpful” episodes and retrain"; §9.1 Step 4 item 5 p.19 "Repeat for the top-k most helpful episodes"]
  - [ ] AC5.9 **(r2 new, P3)** `select_run(op="drop_group", column=…, value=…)` records a group
    leave-one-out selection as a `select` stage.
    - The run records params `op`/`column`/`value`, a `select.dropped` tag and the produced subset.
    - Check in a scratch lineage root: leave-one-task-out on `v1/candidate` gives a 912-episode child.
      `trace(<that select run>)` reaches v0 and the 20 raw-file sha256s.
    - No training is required for this AC.
    [PoC §10 p.20 "Causal baseline"; §2.1 item 5 p.3]
  - [ ] AC5.10 **(r2 new, P3)** Random-subset selection for datamodels-style regression is recorded
    with the existing op.
    - A random subset of size n is `select_run(op="drop_random_k", k=N−n, seed=s)`; no new
      operation.
    - Check in a scratch lineage root: 3 selects on `v1/candidate` with k = 480 (α = 0.5) and seeds
      0/1/2 give 3 distinct subset ids of 480 episodes each, each reproducible from its seed.
    - A 3 × 960 binary membership matrix is built from the subset records alone (row sums 480), the
      input the regression estimator needs.
    - No training is run. The docstring states the k = N − n mapping.
    [PoC §2.1 item 5 p.4 "Train many runs on random data subsets, then fit the effect of each group on success"; DataMIL p.4 "we first sample m random subsets of the prior dataset"]
- **Verify by:**
  - `scripts/verify_runs_s5.py` (extended for AC5.8–5.10, in its scratch lineage root);
  - `python -m vlapoc.lineage diff <a> <b>`;
  - `python -m vlapoc.lineage trace <select run>`.
- **Demo value:** a `diff` printout between two runs, answering "what changed between the good run
  and the bad run?" at episode-version resolution. Also the recorded helpful-control and
  leave-one-task-out selections.

### S6: Run every downstream stage type end to end on the real data (lineage dry run)
- **Why:** the owner's requirement that the layer "enable ALL downstream tasks": each stage must name
  the episode versions it consumed and produced (§9 pp. 18–19, §2.5–2.6 p. 5, §10 p. 20).
- **Depends on:** S4, S5
- **Deliverables:** `scripts/lineage_dry_run.py`; a lineage graph (JSON + Mermaid) and CLI outputs in
  `artifacts/`.
- **Acceptance criteria:**
  - [ ] AC6.1 One script records this chain as MLflow runs on the real v0/v1 data:
    1. build(v0) → split → derive(v1) → screen (stand-in);
    2. train-full (v1 `candidate`, seed 0) and train-full (seed 1);
    3. eval (stand-in) → attribute (stand-in);
    4. select drop-top-k and select random-k (same k);
    5. train on each selection → eval (stand-in).

    Stand-ins carry `stand_in=true`, and **no outcome numbers are invented**. [user request; PoC §9 pp.18–19; §2.6 p.5]
  - [ ] AC6.2 **(r2 reworded)** Train steps are real LeRobot training runs.
    - They call LeRobot's own `lerobot.scripts.lerobot_train.train(cfg)` through
      `vlapoc/lerobot_train_entry.py`. The `lerobot-train` CLI cannot build its argument parser on
      Python 3.14 (`TypeError: str | None is not callable`, draccus issue #69, open).
    - Setup: a small policy (ACT), a few steps, CPU/MPS, wandb disabled, on
      `dataset.episodes=<resolved subset>`.
    - The sampler, training loop, checkpointing and `train_config.json` are LeRobot's.
    - The checkpoint's saved `train_config.json` lists exactly the subset's episode indices (read back
      and compared).
    - The smoke run proves the path from data to trainer to checkpoint, not training quality.
    [PoC §2.1 p.3; LeRobot `DatasetConfig.episodes`; R2]
  - [ ] AC6.3 `diff(train-full-s0, train-drop-top-k)` reports exactly k removed episode versions and no
    other data difference; `diff(train-full-s0, train-full-s1)` reports only the seed. [PoC §2.1 p.3; §2.3 p.4]
  - [ ] AC6.4 `trace(train-drop-top-k)` reaches v0 and the 20 raw-file sha256s.
    `consumers(<a corrupted episode>@<v1 hash>)` lists train-full, and does not list train-drop-top-k
    when that episode was dropped. [PoC §2.1 p.3]
  - [ ] AC6.5 Tamper test: after one byte of a v1 data file (in a scratch copy under a temp dir, per
    AC3.9) is modified, `train_run` refuses to launch. [PoC §2.1 p.3]
  - [ ] AC6.6 The screening stand-in's table joins v1's `labels.parquet` on (`episode_id`,
    `content_hash`) for 960/960 candidate episodes. The attribution stand-in's table passes
    `check_scores` against train-full. [PoC §9.2 p.19]
  - [ ] AC6.7 A lineage-graph artifact (JSON + rendering) of the whole chain is written for the report. [PoC §2.1 p.3]
- **Verify by:** `.venv/bin/python scripts/lineage_dry_run.py --k 20 --seed 0`, then the three CLI
  queries. Round 2 needs no new chain runs in the main DB. AC5.8–5.10 are verified in a scratch
  lineage root, and AC3.9 only needs the tamper step to run from a fresh shell.
- **Demo value:** the lineage graph from raw files to a drop-top-k checkpoint, the `diff` showing
  exactly k removed episode versions, and the tamper refusal.

## Out of scope (deliberately)
- **The corruption suite** (§2.5: time shift, instruction swap, viewpoint/dropped frames, hesitant
  demos, gripper-release noise §8.10). **Owner-approved separate feature** (brief.md, 2026-09-29).
  S4 ships the mechanism and one demo transform.
- **SmolVLA training at scale, GPU selection, rollout evaluation in the LIBERO simulator, PhAIL
  metrics.** Separate features. S6 uses stand-ins and 5-step ACT runs.
- **Screening (DemInf, RoboDrop) and attribution (CUPID).** Only their lineage contracts (stand-ins).
- **(r2) Training the leave-one-out and random subsets, and the datamodels regression itself.** This
  feature records the selections (AC5.9, AC5.10). Training many subsets is blocked by the per-subset
  cache (R1) and owned by the training feature.
- **(r2) Loader changes that avoid the per-subset cache** (the sampler-level lead in R1). That is a
  training-pipeline change; here it is documented, not built.
- **(r2) Oracle ablations by corruption label** (e.g. "drop the injected group"). They would read
  `labels.parquet` and belong to the injection feature's evaluation, not to the method-facing catalog
  (Rule 5).
- **Re-rendering at 256² or from new camera viewpoints** via simulator replay (Linux + sim).
- **Per-sub-task T for LIBERO-10** (§3.5 p. 7). Needs BDDL predicate replay; the evaluation feature
  owns it.
- **Dashboard** (§2.7 p. 5). Only a lineage graph artifact here.
- **Pushing versions to HF Hub / DVC remotes** (needs an account; later).
- **robomimic-HDF5 export of derived versions** for CUPID's original codebase (`source_file`/`demo_<k>`
  kept, so this is a thin later step).
- **libero_object / libero_goal**: not on disk; §10 asks for "one or two suites".
- **Video encoding** of the dataset (lossy).

## Open questions
- **Resolved, Q1 tracking backend:** MLflow, local SQLite (owner answer, brief.md 2026-09-28).
- **Resolved, Q2 git:** `git init`, local only; the executor makes local source commits so runs record
  an exact SHA (owner answer).
- **Resolved, perturbation scope:** the corruption suite is a separate feature (owner answer,
  2026-09-29).
- Non-blocking, resolution 128² (native): the evaluation feature must render at 128. LeRobot's
  `LiberoEnv` config defaults to 360 (quoted). Published SmolVLA/π0 LIBERO results used 256² hub data,
  so absolute success rates may differ (inferred).
- Non-blocking, reference split = 2 per task (RoboDrop A.1). Whether references also join final
  training is for the screening/training features.
- Non-blocking, `operator` / `collection_date` are null: LIBERO publishes neither (LIBERO p. 6).
- Non-blocking, normalization-stats policy (fixed reference stats vs per-subset stats): owned by the
  training feature, and to be listed in the report as an open decision (see R3).

## Risks / premises to verify during execution
- **R1 (r2 new, P1): each training subset and each derivation costs disk.**
  - **The mechanism (quoted source, measured sizes).**
    - `LeRobotDataset(root, episodes=…)` calls `load_nested_dataset`, which runs
      `Dataset.from_parquet(paths, filters=pa_ds.field("episode_index").isin(episodes))`
      (`lerobot/datasets/io_utils.py:79-82`). HF `datasets` then writes a memory-mapped Arrow copy of
      the matching rows into its cache, one per distinct (files, episode list).
    - Measured today in `data/hf_datasets_cache/parquet/`: 6.6, 6.4 and 6.5 GB for the three S6
      training subsets (960/940/940 episodes); 6.9 GB for a full-version load; 280 MB for
      `v0/reference` (40 episodes).
    - The same episode list reuses its entry: 4 train runs produced 3 entries (measured).
    - The size is roughly the subset's frame share of the 7.36 GB dataset, e.g. 7,915/200,340 × 7.36
      GB ≈ 0.29 GB (inferred from these points).
  - **Streaming is not an escape.** `StreamingLeRobotDataset` stores `episodes`
    (`streaming_dataset.py:308`) but never filters on it, so it would silently train on every episode
    (quoted source; analyzer measured).
  - **Derivations leave orphaned caches** for their temporary datasets (AC4.8): 13 entries from a
    20-episode derivation (analyzer, measured); ≈ 2 × 6.3 GB per full-scale derivation (inferred).
  - **Corrected projection** (replaces r1's "about 10 versions on 162 GiB"; inferred arithmetic from
    the measured parts). Free disk is **108 GiB** today (`df -h`, measured).
    - One injection experiment is a derived version (6.8 GB) plus its three §2.6 training subsets
      (full, drop-top-k, random-k, ≈ 6.5 GB each). That is ≈ 26 GB with AC4.8 fixed, or ≈ 39 GB with
      orphaned derivation caches.
    - That allows **about 4 such experiments (2–3 without the fix)** before the disk fills, unless
      caches are cleared between runs.
    - Caches are regenerable: `HF_DATASETS_CACHE` points at gitignored `data/hf_datasets_cache/`.
  - **Consequence for many-subset training.** This covers §2.1 item 5 p. 4 datamodels, and §10
    leave-one-out across 20 tasks.
    - At α = 0.5 each distinct random subset costs ≈ 3.7 GB (inferred).
    - DataMIL used m = 10^4 subsets on MetaWorld (App. G p. 25, quoted). Even m = 30 would need
      ≈ 110 GB of cache, and 20 leave-one-task-out subsets ≈ 20 × 7 GB.
    - With the current loader, many-subset training is feasible only if each run deletes its cache
      after training, or if loading stops materialising per subset.
  - **Lead for the training feature** (inferred from source, untested).
    - LeRobot's trainer already restricts episodes at the sampler:
      `EpisodeAwareSampler(dataset.meta.episodes["dataset_from_index"], …,
      episode_indices_to_use=dataset.episodes, …)` (`lerobot_train.py:457-465`, quoted).
    - Loading each version once without `episodes=` would give one ≈ 6.9 GB cache per version,
      shared by all subsets. Passing the subset only to the sampler would then make the per-subset
      cost ≈ 0.
    - This needs a hook around `train()`, because `dataset.episodes` also drives the reader.
    - `train_config.json` (or the run record) must still list the subset, which AC6.2 checks.
- **R2 (r2 new, P2): lerobot 0.6.1 declares only Python 3.12 and 3.13; the venv is 3.14.**
  - Evidence: `METADATA` classifiers `3.12`/`3.13`, `Requires-Python: >=3.12` (measured). The
    `lerobot-train` CLI and `PreTrainedConfig.from_pretrained` both fail on 3.14 (draccus #69, open,
    quoted; reproduced).
  - Workarounds in place: `vlapoc/lerobot_train_entry.py` calls `train(cfg)` directly, and
    `load_checkpoint` uses `draccus.load` plus a strict safetensors load. Both depend on LeRobot
    internals (`parser.wrap`'s bypass when handed a config object).
  - Recommendation for the SmolVLA/GPU feature: a Python 3.12/3.13 venv (the CLI presumably works
    there; inferred, untested), or keep the entry point. The LIBERO simulator needs Linux anyway
    (LeRobot docs, quoted).
- **R3: the normalization-stats leak in counterfactual runs (measured in round 1).**
  - All four S6 checkpoints normalize with v1's full `meta/stats.json`. That covers all 1000 episodes,
    including the 40 references, the 100 corrupted and the 20 dropped (analysis-r1).
  - It conflicts with "data is the only variable" (p. 1) unless declared.
  - This feature records `stats_json.sha256`, `normalizer.sha256` and a `normalization_stats` param
    per train run (AC5.2). The training/verification feature must choose fixed-reference or
    per-subset stats (`recompute_stats` on a copy) and disclose the choice.
- **R4: MLflow records some absolute paths** in its own records (experiment `artifact_location`,
  `mlflow.source.name`). Our manifests and artifact copies hold none (measured, round 1). Moving
  `data/lineage` to another machine needs the artifact location remapped (inferred; not tested).
- **R5: time-based corruptions on the hub datasets would be mis-scaled** (declared fps 10 vs 20 Hz
  data). On our fps-20 dataset, RoboDrop's "two seconds" = 40 steps. The injection feature should
  compute offsets from `info.json` fps.
- **R6: task-ID mapping.** `libero_task_id` follows `libero_suite_task_map.py`. The evaluation feature
  must confirm that LeRobot's `--env.task_ids` uses the same order once LIBERO is installed.
- **R7: PyAV/Homebrew ffmpeg duplicate `libavdevice`** (measured warning). No crash observed in round 1.

## Non-blocking analyzer items: disposition (r2)
- **Noise leaves `Box(-1, 1)`, and the noised gripper becomes continuous** (v1 `gripper_toggles`
  84–428 on corrupted episodes). Recommended to the executor: record this effect of `clip: null` in
  v1's `params` note. The clipping decision belongs to the injection feature. No AC, since it doesn't
  affect lineage correctness.
- **The episode table is not integrity-checked.** `Subset.resolve()` depends on the
  `episode_id` → `episode_index` map in the episode table, which `version.json` does not digest.
  Recommended: add a digest and check it in `verify_version`. It is a cheap hardening of "name exactly
  which episodes" and left optional for this round. If the executor does it, it should land together
  with the AC4.7 rewrite of v1's tables.
- **`catalog.py:398` and `verify_libero_convert.py:200` crash on out-of-repo `--out`/`--summary`;
  `split make --track` runs `make_splits` twice; the CLI and resolvers are over-built.** For the
  refactorer.
- **Checkpoints as MLflow `LoggedModel`s:** optional, for the training/eval features.
- **Git author:** commits carry the owner's name and email. It is disclosed in the log, and the owner
  may prefer a distinct author.
