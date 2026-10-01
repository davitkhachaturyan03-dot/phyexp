VERDICT: REWORK
ROUTE: planner+executor
ROUND: 1

<!-- Saved verbatim by the orchestrator from the analyzer's returned text: the analyzer's own
     write was refused by the harness ("Subagents should return findings as text, not write
     report files"). -->

## Summary (3–5 lines)
I re-ran all 38 ACs (S1–S6) against the stored datasets, the main MLflow DB, the S5 scratch DB, and live runs of my own (a real `train_run` under a network-deny sandbox, a tamper refusal, and a real `derive_version` with a length-changing edit on a 20-episode parent). Every AC reproduces in substance; nothing was fabricated. The data and lineage layer works, and no off-the-shelf tool would replace the custom code without breaking a spec rule.
REWORK is for a short list of cheap but real problems:
- The plan states a false premise: training subsets do copy data on disk (about 6.5 GB each).
- The executor silently put the ground-truth `corruption_type` into v1's general episode table.
- `derive_version` leaves orphaned HF caches that are not disclosed.
- The tracked `select` stage cannot record the spec's "remove top-k helpful" control.
- Two verification scripts hard-code this session's `/private/tmp` path.
- Two AC wordings (AC1.2, AC6.2) don't match what was verifiably built.

## Rebuttals received
| From | Point disputed | Their evidence | Conceded? | Why / why not |
|---|---|---|---|---|
| — | none (round 1) | — | — | — |

### Points the executor raised, judged on evidence
| Point | What I checked | Verdict |
|---|---|---|
| AC1.2 wording | Re-ran the diff: 4 differing lines vs `lerobot/libero`, 2 vs `HuggingFaceVLA/libero`. Fetched both pinned `info.json` files from the Hub; they match `artifacts/reference/`. LeRobot LIBERO docs (quoted): "They contain the same demonstrations with the same schema and differ in how camera frames are stored" | **Accepted.** Met in substance. AC1.2 contradicts the plan's own AC1.6 lossless choice, so the planner must reword it (P2) |
| `lerobot-train` CLI broken on 3.14 | `lerobot-train --help` gives `TypeError: str \| None is not callable` (measured). draccus 0.11.6 is the newest release and lerobot pins `draccus<0.12.0,>=0.11.6`. The bypass exists at `parser.py:398`. Upstream draccus issue #69 is open since 2026-07-28 with this exact error (quoted). **lerobot 0.6.1 classifiers list only Python 3.12 and 3.13** (measured) | **Accepted.** AC6.2 is met in substance: the sampler, training loop, checkpointing and `train_config.json` are all LeRobot's own |
| `from_pretrained` broken on 3.14 | Reproduced: `TypeError typing.Dict[str, …PolicyFeature] \| None is not callable`. The live eval stand-in loads the checkpoint strictly (51,600,263 params) | **Accepted** |
| MLflow 36-character digest cap | `validation.py:81 MAX_DATASET_DIGEST_SIZE = 36`; the DB column is `String(36)`. `producers()` and `consumers()` re-check the full id | **Accepted** |
| `accelerate` added outside the plan | `lerobot_train.py:231 require_package("accelerate")`. lerobot's own `training` and `smolvla` extras both pull it | **Accepted** |
| Per-subset HF cache | `io_utils.py:79-82`. The three S6 subsets have cache entries of 6.6, 6.4 and 6.5 GB. `StreamingLeRobotDataset` stores `episodes` (`streaming_dataset.py:308`) but never filters on it, so it would silently train on all 1000 episodes and is not an escape. `keep_in_memory` isn't exposed by LeRobot | **Acceptable at this PoC's handful of subsets.** Disclosed in the log and the docstring, but the plan still states the false premise (P1) and the derive-side cost is undisclosed (E3) |
| Normalization-stats leak | All four S6 checkpoints normalize with v1's full `meta/stats.json` (all 1000 episodes, including the 40 references, the 100 corrupted and the dropped 20). The stats are identical across the full and drop-top-k runs. Each run records `stats_json.sha256`, `normalizer.sha256` and a `normalization_stats` param | **Recording is sufficient here.** Fixed vs per-subset stats is a choice for the training feature and belongs in the report as an open decision |
| Size of custom code | 1,724 non-docstring code lines in `vlapoc/`, against the plan's ~400-line estimate | **Keep.** No verified replacement; only simplification nits |

## Acceptance criteria
All commands run by me on 2026-09-29, with `HF_DATASETS_CACHE=data/hf_datasets_cache` unless noted.

| Story | AC | Claimed | Reproduced? | Evidence |
|---|---|---|---|---|
| S1 | 1.1 | ✅ | ✅ | `verify_libero_convert.py`: v3.0, fps 20, 1000 episodes / 200,340 frames / 20 tasks; the full dataset loads |
| S1 | 1.2 | ⚠️ | ✅ substance / ❌ wording | Against `lerobot/libero` the image dtype (`image` vs `video`) and shape differ. Against `HuggingFaceVLA/libero` only the shape differs. fps 20 vs 10. Keys, state and action match |
| S1 | 1.3 | ✅ | ✅ | The script's 6 episodes, plus 10 random episodes read with my own pyarrow/h5py code (not `load_raw_episode`). State and action are bit-equal; `ee_pos` and gripper slices match `robot_states` |
| S1 | 1.4 | ✅ | ✅ | 180° rotation matches, unrotated doesn't, on all 16 episodes. The orientation figure rules out a vertical-only flip |
| S1 | 1.5 | ✅ | ✅ | 50 episodes × 20 tasks. Lengths match raw 1000/1000. Exactly one reward at the last step, 1000/1000 |
| S1 | 1.6 | ✅ | ✅ size; wall time not re-run | No videos, 7.356 GB of file bytes. `du` now reports 6.85 GiB vs 7.44 GiB in the log (same byte sum) |
| S1 | 1.7 | ✅ | ✅ | Docstring read |
| S2 | 2.1–2.7 | ✅ | ✅ | `build_catalog`: 1000/1000 hashes match raw. The rebuilt catalog equals both the stored catalog and v0's version table. All summary numbers are identical. My own re-implementation of the hash layout matches 10/10 episodes. The 20 raw-file sha256s match what was registered |
| S3 | 3.1 | ✅ | ✅ | Recompute for v0 and v1 both give `match: true` |
| S3 | 3.2 | ✅ | ✅ | 40 reference / 960 candidate, disjoint, union equals v0, 2 per task |
| S3 | 3.3 | ✅ | ✅ | `v0/reference` checked for every frame via `__getitem__` (7,915). `v1/candidate` (192,425), drop-top (188,212) and drop-random (188,875) all serve exactly the subset |
| S3 | 3.4 | ✅ | ✅ | In a scratch lineage root: highest and lowest drops correct, random-k reproducible, stale key rejected. My seed-0 random child's id equals the executor's quarantined `4a06edb4…` |
| S3 | 3.5 | ✅ | ✅ | One bit flipped in a clone of v1 raises `VersionIntegrityError` naming the changed file |
| S3 | 3.6 | ✅ | ✅ | 38 manifests, 0 absolute paths. Only the training stdout logs contain absolute paths (library warnings) |
| S3 | 3.7 | ✅ | ✅ | RoboDrop App. A.1 quote verified against the PDF |
| S4 | 4.1 | ✅ | ✅ | `diff v0 v1`: 100 changed, 900 identical. Parent links correct; v1 id recomputes. v1 loads via `v1/candidate` (no new full-v1 cache) |
| S4 | 4.2 | ✅ | ✅ | 100 label rows, all hashes consistent |
| S4 | 4.3 | ✅ | ✅ | The guard raises for v0 and for v1 (via ancestry); nothing is written |
| S4 | 4.4 | ✅ | ✅ small scale | Real `derive_version` on a registered 20-episode parent: length 272 → 292, metadata consistent, 1 changed / 19 identical, id recomputes |
| S4 | 4.5 | ✅ | ✅ | 5 per task, none in reference. Noise std ≈ 0.25σ on 10 other corrupted episodes (0.243–0.258). But `clip: null` puts actions outside `Box(-1, 1)`: the gripper reaches −2.07 / +1.96 |
| S4 | 4.6 | ✅ | ✅ | Docstring read |
| S5 | 5.1–5.7 | ✅ | ✅ | Queries on both DBs reproduce. My live `train_run` (v1/candidate, 7.7 s) records git SHA == HEAD and a clean tree. Live eval loads the checkpoint strictly. `check_scores` rejects 20/960 and 100/960 correctly. Offline confirmed under a network-deny sandbox (control curl fails with 000, exit 7) |
| S6 | 6.1 | ✅ | ✅ | 15 runs: 14 finished, 1 inert failed run |
| S6 | 6.2 | ✅ | ✅ substance / ❌ wording | Episode lists in all 4 `train_config.json` equal the subsets (960/960/940/940), via `train()` rather than the CLI |
| S6 | 6.3 | ✅ | ✅ | Diffs identical to the stored JSON artifacts |
| S6 | 6.4 | ✅ | ✅ | Trace reaches 8 runs over 11 edges, each with exactly one producer, and all 20 raw files. The consumers probe matches the log |
| S6 | 6.5 | ✅ | ✅ live, in scratch | Refused launch, run count 0 → 0. The dry-run script itself was not re-run |
| S6 | 6.6 | ✅ | ✅ | Screening table 960/960, labels 100/100 |
| S6 | 6.7 | ✅ | ✅ | Graph: 36 nodes, 14 runs, 40 edges |

**My footprint:** about 7.2 GB of regenerable cache added to `data/hf_datasets_cache` (the full-v0 load for AC1.1, plus `v0/reference`). The main `mlflow.db` sha256 is `c5faec81…` before and after. `git status` is clean. My scratch under `artifacts/_analyzer_r1_scratch/` (1.5 GB) is deleted.

## PoC scope assessment
**Covered:**
- §10 p.20 "Foundation": lineage tracking, episode catalog, two LIBERO suites. SmolVLA training is deliberately deferred; ACT smoke runs prove the path.
- The §10 "Ground truth" mechanism: derived versions, labels, the reference guard, length-changing edits.
- The §3.5 human-reference durations.
- The §9.2 key contract (scores keyed by episode version).
- The corruption suite's absence is owner-approved scope, not a gap.

**Missing (spec-grounded, cheap to add):**
1. "Remove the top-k helpful episodes" (§2.6 p.5; §9.1 Step 4 item 5 p.19). `select_run` hard-codes `"highest"` (`runs.py:519,522`), so this control can't be recorded as a lineage stage. The plan never mentions "helpful".
2. Group-level leave-one-out ablations (§10 p.20 "Causal baseline"; §2.1 item 5) and random-subset selection. `select_run` rejects other operations (`runs.py:526-527`), so a hand-made subset has no producer and `trace` dead-ends. The plan neither covers nor defers these.

**Over-built (mild):**
- a 126-line CLI with 14 subcommands (`lineage.py:491-616`)
- prefix-matching resolvers
- `_local_source` is an identity function (`runs.py:187-191`)
- `split make --track` runs `make_splits` twice

**Silent divergences:**
- `derive.py:265-266` adds a `corruption_type` column to the derived version's episode table. This is in neither the plan nor the log.
- The derive-side cache cost is undisclosed, and `plan.md:405`'s "about 10 versions" projection no longer holds.

## Overcomplication & reinvention audit
| Mechanism (file:line) | Existing OSS alternative | Covers our need? | Evidence | Recommendation |
|---|---|---|---|---|
| Per-episode `content_hash` (`catalog.py:109-131`) | LeRobot metadata, HF `datasets` fingerprint, DVC | No. LeRobot has only 9 non-stats `meta/episodes` columns and no hashing, and its edit tools reindex. DVC hashes "the file or directory" | measured (schema, grep); quoted (DVC docs) | Keep |
| Version registry + `names.json` (`lineage.py:104-273`) | MLflow datasets, DataChain, HF Hub revisions | No. MLflow logs datasets as inputs only. DataChain auto-increments versions (not content-addressed), doesn't feed LeRobot's `episodes=`, and would be a second lineage system beside MLflow. Hub revisions need an account | quoted (DataChain docs, MLflow source) | Keep |
| `Subset` + drop-top-k / random-k (`lineage.py:281-464`) | LeRobot `episodes=` (used), `split_dataset` / `delete_episodes`, robomimic masks | The LeRobot edit tools copy and reindex data, which breaks the plan's "selection, not copy" rule. LeRobot doesn't read robomimic masks | quoted (source), measured | Keep; add `direction` to `select_run` |
| `verify_version` (`lineage.py:255-273`) | `dvc status` | Would work, but adds a second tool against the owner's MLflow ruling, for 19 lines | inferred | Keep |
| `derive_version` (`derive.py:160-306`) | LeRobot `delete_episodes` + `merge_datasets` | Already reuses them. The custom part (identity re-check, labels, guard) isn't provided anywhere | measured | Keep; fix E2 and E3 |
| MLflow tag encoding (`runs.py:129-212`) | MLflow 3 `log_outputs` / `LoggedModel` | Partly. `RunOutputs` holds only `model_outputs`, so 6 of 7 entity kinds (versions, subsets, labels, scores, rollouts, splits) still need tags. Checkpoints could become `LoggedModel`s | quoted (source, Databricks KB) | Keep; LoggedModel optional later |
| `trace` / `consumers` (`runs.py:556-627`) | None in MLflow OSS | n/a | quoted | Keep |
| `diff` (`runs.py:648-681`) | MLflow's run comparison (params only) | Episode-level data diff has no equivalent | inferred | Keep (34 lines) |
| Train entry point + `load_checkpoint` | `lerobot-train`, `from_pretrained` | Both broken on 3.14 (reproduced; draccus #69) | measured + quoted | Keep until the venv is 3.12/3.13 |
| Parallel hashing | none needed | Measured need: 29 s wall vs ~2.5 min single-core | measured | Keep |
| CLI and resolvers | argparse | Convenience beyond the plan | inferred | Simplify (refactorer) |

Searches with nothing that replaces the custom code: the MLflow 3 `log_outputs`/LoggedModel/dataset-output query, the per-record dataset versioning query (DataChain / lakeFS / Deep Lake), the LeRobot versioning/lineage query, and two draccus + Python 3.14 queries. I also read the source of LeRobot `io_utils.py`, `streaming_dataset.py`, `sampler.py`, `lerobot_train.py` and `parser.py`, and the CUPID repo file (confirms `demo_{i}` keys and `get_dataset_masks`).

A lead for the training feature (inferred): LeRobot's trainer already restricts episodes at the sampler (`EpisodeAwareSampler(episode_indices_to_use=…)`, `lerobot_train.py:457-465`). Loading a version once and passing the subset only to the sampler would avoid the per-subset copy.

## Feedback: planner
1. **P1 (blocking): false premise at `plan.md:406`.** "Training subsets never copy data" is false on disk. Replace it with the measured facts:
   - each distinct subset writes a ~6.4–6.6 GB cache (source `io_utils.py:79-82`);
   - the streaming dataset is not an escape, because it ignores `episodes`;
   - each derivation also leaves orphaned caches (E3), so `plan.md:405`'s "about 10 versions" no longer holds at 109 GiB free.

   Add this to Risks, with the consequence for datamodels-style many-subset training (§2.1 item 5) and the sampler-level lead above.
2. **P2 (blocking): reword AC1.2 and AC6.2.**
   - AC1.2: compare against `HuggingFaceVLA/libero@86958911` (shape and fps only), and record `image` vs `video` against `lerobot/libero` as a consequence of AC1.6. Cite the docs' "same schema … differ in how camera frames are stored".
   - AC6.2: say "LeRobot `train()` via `vlapoc/lerobot_train_entry.py`, because the CLI can't parse arguments on 3.14 (draccus #69)".
   - Add a risk: lerobot 0.6.1 declares only Python 3.12 and 3.13. The SmolVLA feature should use a 3.12/3.13 venv or keep the entry point.
3. **P3 (blocking): counterfactual controls.** Add an AC that `select_run` accepts and records `direction` (§2.6 p.5; §9.1 Step 4 item 5 p.19). For group leave-one-out (§10 p.20; §2.1 item 5) and random-subset selection, either add recorded `select` operations (e.g. `drop_group(column, value)`, `random_subset(n, seed)`) or list them explicitly as out of scope with a reason.

## Feedback: executor
1. **E1 (blocking): `runs.py:510-531`.** Add a `direction` parameter to `select_run` (default `"highest"`), pass it through at `:519`, and log it at `:522`. Verify in a scratch lineage root that a `"lowest"` select records `derivation["direction"] == "lowest"`, drops the 20 lowest scores, and logs the MLflow param.
2. **E2 (blocking): `derive.py:265-266` leaks the ground-truth label into v1's episode table.** That table is the catalog every downstream stage reads through `Subset.episodes()` / `version_episodes()` (e.g. `runs.py:462`), and §2.1 item 4 needs the method under test to be blind to the labels. Labels already live, and should only live, in `labels.parquet`.
   - Remove the two lines.
   - Rewrite `data/lineage/versions/16076b9c…/episodes.parquet` and `data/lineage/catalog/libero_spatial_10__v1/episodes.parquet` without the column. The version id doesn't depend on it, so it stays the same.
   - Expected afterwards: the column is absent, `recompute v1` still gives `match: true`, the labels join is still 100/100, and `diff v0 v1` is still 100/900.
3. **E3 (blocking to disclose; fixing recommended): orphaned caches from `derive_version`.** `derive.py:214,240` load `LeRobotDataset` for the parent, `.tmp_kept` and `.tmp_edit`. The temp caches are orphaned once `:242-243` delete those directories.
   - Measured on a 20-episode parent: 13 cache entries, including a 3,918-row entry for the deleted kept set plus per-file entries of 2,453 and 1,465 rows (3,918 again). At full scale that is roughly 2 × 6.3 GB orphaned per derivation (inferred).
   - Preferred fix: point the derivation's cache at a temp directory and delete it with the temp datasets.
   - Minimum: disclose the cost in the docstring and the log, and correct finding #6 ("a full dataset copy (6.8 GB)").
4. **E4 (blocking, cheap): session-bound paths.** `scripts/lineage_dry_run.py:50` and `scripts/verify_lineage_s3.py:42` hard-code this session's `/private/tmp/…/scratchpad`. When it's gone, `cp -c -R` fails (measured: "No such file or directory", exit 1), and with `check=True` the dry run crashes after it has already recorded 11 runs. Use `tempfile.mkdtemp()` or a gitignored repo-relative scratch directory.

## Non-blocking
- **Noise out of range:** v1's noised actions leave LIBERO's documented `Box(-1, 1)` action space, and the noised gripper becomes continuous. v1's `gripper_toggles` for the corrupted episodes are 84–428 (clean: 1–8), so the column is meaningless there. Record this effect of `clip: null` in `params`, and leave the clipping decision to the injection feature.
- **Episode table not integrity-checked:** `version.json` holds no digest of the index-to-id map that `Subset.resolve()` depends on, so an edit to it would go undetected at training time. Add a digest and check it in `verify_version`.
- **Crash on out-of-repo paths:** `catalog.py:398` and `verify_libero_convert.py:200` crash when `--out` / `--summary` point outside the repo, after the files are written.
- **Duplicate split:** `split make --track` runs `make_splits` twice (harmless, idempotent).
- **Size wording:** for AC1.6, report "7.36 GB (file bytes)" rather than a `du` figure.
- **Optional:** represent checkpoints as MLflow `LoggedModel`s in the training/eval features, for native lineage in the MLflow UI.
- **Git author:** commits carry the owner's name and email (set per commit). Disclosed in the log; the owner may prefer a distinct author.

Sources: [draccus issue #69](https://github.com/dlwh/draccus/issues?q=3.14), [LeRobot LIBERO docs](https://huggingface.co/docs/lerobot/libero), [lerobot/libero info.json @a1aaacb7](https://huggingface.co/datasets/lerobot/libero/raw/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4/meta/info.json), [HuggingFaceVLA/libero info.json @86958911](https://huggingface.co/datasets/HuggingFaceVLA/libero/raw/86958911c0f959db2bbbdb107eb3e17c5f9c798e/meta/info.json), [DataChain datasets guide](https://docs.datachain.ai/guide/datasets/), [DVC .dvc files](https://doc.dvc.org/user-guide/project-structure/dvc-files), [Databricks KB on output lineage](https://kb.databricks.com/machine-learning/lineage-for-the-output-table-with-predictions-is-not-tracked-in-mlflow-when-training-a-model), [CUPID robomimic dataset](https://raw.githubusercontent.com/agiachris/cupid/main/diffusion_policy/dataset/robomimic_replay_image_dataset.py)
