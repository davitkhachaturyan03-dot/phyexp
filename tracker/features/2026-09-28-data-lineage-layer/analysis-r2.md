VERDICT: PASS
ROUTE: none
ROUND: 2

<!-- Saved verbatim by the orchestrator from the analyzer's returned text: the analyzer's own
     write was refused by the harness ("Subagents should return findings as text"). -->

## Summary (3–5 lines)
- All 7 round-1 items (P1–P3, E1–E4) are resolved. I re-ran every new or changed AC myself (AC1.2/1.6/6.2 wording, the AC3.3 addendum, AC3.8, AC3.9, AC4.7, AC4.8, AC5.8–5.10) and all reproduce, several on paths the executor didn't use:
  - a *fresh* derivation through the new code;
  - AC4.4 under the new code;
  - a fresh-shell (`env -i`) run of both scripts;
  - adversarial `select_run` cases.
- Round-1 ACs touched by the changes still hold: S6 diff/trace/consumers/check_scores are identical to the stored artifacts, the guard still raises, and v0/v1 verify.
- The reinvention audit of `drop_group` and the random-subset approach found nothing to swap in.
- Four small issues are non-blocking: `libero_task_id` is suite-relative in `drop_group`; a refused `drop_group` leaves a FAILED run; the `select.dropped` tag is truncated; the episode table still has no digest.
- My footprint: the main `mlflow.db` sha256 is `c5faec81…` before and after, the real `data/lineage` store (63 files) is byte-identical, and my scratch under `data/scratch/` is gone.

## Rebuttals received
| From | Point disputed | Their evidence | Conceded? | Why / why not |
|---|---|---|---|---|
| planner | None ("I agree with P1–P3 and E1–E4 on the evidence"). One precision: `direction` already existed in `drop_top_k`; only `select_run` hard-coded it | `lineage.py:424-435`, `runs.py:519,522` | n/a (not a dispute) | Correct, and it matches what I flagged in round 1 |
| executor | none | — | — | — |

### Executor notes, judged on evidence
| Note | What I checked | Verdict |
|---|---|---|
| AC4.4 padding not re-run (needs a 7 GB temp copy) | I re-ran AC4.4 **through the round-2 `derive_version`** on a registered 20-episode parent in `data/scratch/`: `len 261 -> 281; meta 281; served 281; total 4210 (parent 4190); idle True; diff 1/19; recompute True; labels 1` | **Accepted.** The new code only wraps the loads in a cache context. The full-scale LeRobot tools were already exercised by the v1 derivation |
| Stale v1 table with `corruption_type` in `data/lineage_scratch/s5/` | Confirmed: `…/versions/16076b9c…/episodes.parquet` still has the column. `grep lineage_scratch vlapoc/` finds nothing. Only `scripts/verify_runs_s5.py` (round-1 mode) uses that root, and it deletes and re-copies it from the real store | **Accepted as disclosed.** No library path reads it. The owner may delete `data/lineage_scratch/` (non-blocking) |
| Optional items left out (clip note, episode-table digest, refactor nits) | Plan r2 lists them as recommendations, not ACs. The clip-note reason is sound: editing `labels.parquet` would break the `labels_sha256` in `derivation.json` and the derive run's `output.labels.sha256` tag | **Accepted.** The report must still state the effect of `clip: null` (it's in the log) |
| Free disk 107 GiB | Measured 119 GiB on 2026-10-01; my runs left it unchanged | Fine |

## Round-1 items: resolution
| Item | Resolved? | Evidence (mine) |
|---|---|---|
| P1 false premise | ✅ | "never copy" now appears only in the change list and the "Premises corrected" table (`plan.md:14,204`). R1 (`plan.md:656-696`) has the measured mechanism, the streaming non-option, the derive cache, a corrected projection and the sampler lead. Arithmetic checks: 6.8 + 3×6.5 ≈ 26 GB; 0.5×7.36 ≈ 3.7 GB |
| P2 AC wording | ✅ | AC1.2 (`:305-313`), AC1.6 (`:321`) and AC6.2 (`:587-597`) reworded. R2 (`:697`) cites the METADATA classifiers and draccus #69 |
| P3 controls | ✅ | AC5.8 (direction), AC3.8 + AC5.9 (group leave-one-out, added rather than deferred), AC5.10 (random subsets via the existing op). New out-of-scope items are explicit (`:622-629`) |
| E1 `select_run` direction | ✅ | See AC5.8 |
| E2 label leak | ✅ | See AC4.7 (code, data, and a fresh derivation) |
| E3 orphaned caches | ✅ fixed, not just disclosed | See AC4.8 |
| E4 session paths | ✅ | See AC3.9 |

## Acceptance criteria
All commands run by me on 2026-10-01 with `HF_DATASETS_CACHE=$PWD/data/hf_datasets_cache`. Scratch was under `data/scratch/` and deleted.

| Story | AC | Claimed | Reproduced? | Evidence |
|---|---|---|---|---|
| S1 | AC1.2 (reworded) | ✅ | ✅ | The S1 converter and verify script are unchanged since round 1 (`git diff --stat 41f7c11..HEAD` empty). The round-1 reproduction stands: against `HuggingFaceVLA/libero` only image shape and fps differ; against `lerobot/libero` image dtype differs too. The reworded AC now matches |
| S1 | AC1.6 (reworded) | ✅ | ✅ | File bytes 7.356 GB (round-1 reproduction). Wall time `"wall_time_s": 468.5` is in `s1_convert_stdout.txt`; the conversion was not re-run, per the disk rule |
| S6 | AC6.2 (reworded) | ✅ | ✅ | The wording now matches what I verified in round 1: `train()` via the entry point; episode lists 960/960/940/940; live training run |
| S3 | AC3.3 addendum | ✅ | ✅ | The `Subset` docstring (`lineage.py:286-296`) now says it's a selection (no rewrite or reindex), that the loader writes an Arrow cache per distinct episode list, and cites `io_utils.py:79-82`. "Never a data copy" is gone. Behaviour re-verified (next row) |
| S3 | AC3.1–3.5, 3.8 (fresh shell) | ✅ | ✅ | `env -i … .venv/bin/python scripts/verify_lineage_s3.py --summary data/scratch/…` → exit 0 in 52.9 s. AC3.1 `match=True`. AC3.2 40/960 disjoint. AC3.3: `v0/reference` 7,915 frames (every frame read) and `v0/candidate` 192,425, both served == subset. AC3.4: highest and lowest correct, random child `4a06edb41e77` reproducible, stale key rejected. AC3.5: tamper caught in a `data/scratch/…` clone. **Shared cache 124 files / 28,643,158,514 B before and after** |
| S3 | AC3.6 | ✅ | ✅ | Same run: `scanned 58 lineage files … absolute-path hits=[]` |
| S3 | AC3.8 `drop_group` | ✅ | ✅ | Script run: dropping one task → 912 episodes (48 dropped); dropping a suite → 480; `resolve()` equals the remaining `episode_index` set. Raises on a non-catalog column (`corruption_type`), a list column (`objects`), an identity column (`content_hash`), no match, and an all-match. My own driver: the 48 dropped are exactly the task's candidate episodes, and the suite child is all `libero_spatial` |
| S3 | AC3.9 | ✅ | ✅ | `grep -rnE '/private/tmp\|/tmp/claude\|/Users/' vlapoc scripts` → no output, exit 1. Scratch comes from `tempfile.mkdtemp(dir=data/scratch)`. Tamper step under `env -i` (my driver calling `lineage_dry_run.tamper_check()`, so the artifact wasn't overwritten): `train_run refused: True; changed=['data/chunk-000/file-080.parquet']; runs 15/15; new data/runs dirs: []`, clone removed, main DB unchanged |
| S4 | AC4.7 | ✅ | ✅ | v1's version table columns equal `CATALOG_COLUMNS + ['parent_content_hash']`, with no label columns, and the catalog copy equals the version table. Recompute v1 matches; labels join 100/100; diff 100/900. **The rewrite left other values intact (checked against v0):** static columns equal v0 for all 1000; the 900 unedited episodes have the same hash and stats as v0; `parent_content_hash` equals the v0 hash; label `content_hash` equals the v1 table. **A fresh derivation through the new code** writes parent columns + `parent_content_hash` and no label columns |
| S4 | AC4.8 | ✅ | ✅ | My driver ran an isolated `derive_version` on a 20-episode parent. The shared cache was 124 files / 28,643,158,514 B before and after. The `~/.cache/huggingface/datasets` file count was 53/53 (read-only listing). No temp caches left behind, and `derivation.json` records the policy. **Source check of the fix:** every `datasets` call site reads `config.HF_DATASETS_CACHE` at call time (`builder.py:365`, `load.py:807`, `packaged_modules/cache/cache.py:38`, `utils/file_utils.py:360`). The only import-time binding (`config.py:163`) is the downloads path, which local parquet never uses |
| S4 | AC4.1, 4.3, 4.4 (regression) | ✅ | ✅ | `version verify v1` 85 files ok, v0 77 ok. The reference guard raises for both v0 and v1 and writes nothing. AC4.4 under the new code: see the notes table |
| S5 | AC5.8 | ✅ | ✅ | My driver ran in a scratch root holding copies of the manifests and `mlflow.db`, with the artifact location repointed. `lowest` drops exactly the 20 lowest and records `direction=lowest` in both the derivation and the run params. The default `highest` is correct, and **its child id equals S6's `c598f5692b40`** (round-1 identity preserved). `direction="up"` raises `ValueError` with 0 runs added |
| S5 | AC5.9 | ✅ | ✅ | `select_run(op="drop_group", column="task_name", …)` gives n=912, records the `op`/`column`/`n_dropped` params, and its produced tag equals the child. `resolve()` matches the remaining indices. `trace` reaches `select, derive-v1, build-v0, split-v0` and all 20 raw files, each edge with a single producer |
| S5 | AC5.10 | ✅ | ✅ | Seeds 0/1/2 with k=480 each give n=480, record `subset_size=480`, and reproduce from the seed. **The ids `ab03e001`, `5eafbb8e`, `d03a5cf5` match the executor's.** The membership matrix is 3 × 960 with row sums [480, 480, 480]; episodes appear in 0/1/2/3 subsets as [120, 355, 370, 115] |
| S6 | AC6.1–6.7 (regression) | ✅ | ✅ | After the v1 table rewrite: all three diffs equal the stored JSON; trace still reaches 8 runs over 11 edges and 20 raw files; the consumers probe lists train-full-s0 and not drop-top-20; `check_scores` 960/960 ok; screening × labels 100/100 |

## PoC scope assessment
**Covered:**
- everything from round 1;
- the "remove the top-k helpful" control (§2.6 p.5; §9.1 Step 4 item 5 p.19) as a recorded stage;
- group leave-one-out (§2.1 item 5 p.3; §10 p.20) as a recorded stage;
- random subsets (§2.1 item 5 p.4) as recorded selections, with a membership matrix built from the records alone.

Quotes checked in the PDFs:
- PoC p.3 "Drop one data source or task group"; p.4 "Train many runs on random data subsets"; p.20 "Group-level leave-one-out ablations".
- DataMIL p.4 "we first sample m random subsets of the prior dataset", and App. G p.25 "m = 10^4 sufficient". The latter refers to the regression estimator's random-subset trainings, so R1's use of it is accurate.

**Missing:** nothing required. Training many subsets is explicitly out of scope (`plan.md:622`).

**Over-built (nits):** `isolate_hf_cache=False` exists only to measure the old behaviour, so it's a test hook in the library API. `drop_group` allows equality grouping on continuous columns (`smoothness`, `duration_s`), which is harmless but meaningless.

**Silent divergences:** none. The in-place rewrite of v1's two tables is plan-mandated (AC4.7) and logged with before/after sha256 (`135c0655…` → `3fcca3a4…`).

## Overcomplication & reinvention audit
| Mechanism (file:line) | Existing OSS alternative | Covers our need? | Evidence | Recommendation |
|---|---|---|---|---|
| `drop_group` (`lineage.py:490-523`) | LeRobot `LeRobotDataset(episode_filter=…)` / `meta.filter_episodes(predicate)` | No. The predicate is "Evaluated against `meta/` without `stats` keys (e.g. `task_index`, `episode_index`, `length`…)", so it can't see our catalog columns (suite, scene, task_name). It isn't exposed in `DatasetConfig` or the dataset factory (grep: none), records no selection, and as a load-time filter still writes a cache per list | quoted (source `lerobot_dataset.py:161-164`); measured (grep) | **Keep** (about 25 lines of pandas plus the record) |
| Random subsets = `drop_random_k(k=N−n)` plus a membership matrix | MadryLab `datamodels`; DataMIL released code | No new mechanism was built. The matrix uses MadryLab's own representation (`masks.npy`, shape `(num_models, n)`, bool), and that library leaves mask generation to the user's training script (README). DataMIL's released code uses a metagradient estimator over RLDS/MDS and has no subset sampler to adopt | quoted (READMEs via fetch) | **Keep**; the matrix can feed `fast_l1` or scikit-learn later |
| `isolated_hf_datasets_cache` (`lineage.py:526-550`) | `datasets`' `cache_dir=` argument; the `HF_DATASETS_CACHE` env var | LeRobot never passes `cache_dir` (`io_utils.py:82`), and the env var is read once at import (`config.py:157`), so per-call isolation needs the config swap | quoted (source); measured (cache byte-identical) | **Keep** (about 10 lines) |
| `select_run` op dispatch (`runs.py:511-555`) | — | Thin wrapper over the three lineage ops | — | **Keep** |

Searched with nothing that replaces these: "MadryLab datamodels library subset masks …" and "DataMIL datamodels robot imitation learning code github subset sampling". I also read the MadryLab and DataMIL READMEs and LeRobot's `filter_episodes` / `episode_filter` source.

## Feedback: planner
None blocking.

## Feedback: executor
None blocking.

## Non-blocking
1. **`libero_task_id` is suite-relative.** `drop_group(cand, "libero_task_id", 2)` dropped 96 episodes from two tasks, one per suite (`KITCHEN_SCENE3_…` and `pick_up_the_black_bowl_from_ta…`; measured), yet the `drop_group` docstring lists `libero_task_id` as an example column. Either remove it from the groupable set or document "use `task_name` for leave-one-task-out".
2. **A refused `drop_group` inside `select_run` leaves a FAILED run** (measured: 1 FAILED run for `column="corruption_type"`). `direction` is validated before the run starts, but `column` and `value` are not. The run is inert, since only FINISHED runs form lineage edges. Validating before `stage_run` would make the two cases consistent.
3. **The `select.dropped` tag is truncated to 7,900 characters,** which leaves an invalid-JSON prefix for large drops: 7,900 vs about 47,400 characters at k=480 (measured). This is disclosed, and the subset file is the record of truth. Consider storing a count and a digest in the tag instead.
4. **The episode table still has no digest.** `version.json` doesn't digest the `episode_index → episode_id` table. This round's plan-mandated in-place rewrite (sha `135c0655…` → `3fcca3a4…`) left no trace in the lineage store, which shows the gap. Still recommended (plan r2 disposition).
5. **Cleanup the owner may want:** `data/lineage_scratch/` (the round-1 S5 root, 2.3 GB, still holding the stale labelled v1 table) and `data/scratch/r2_strip_label_column.py` (the one-off migration script) remain on disk.
6. **Scale note for the datamodels feature (inferred):** each random-subset record stores both `episode_ids` and `derivation.dropped` (about 95 KB at n=480) plus one MLflow run. At DataMIL's m = 10^4 that is roughly 1 GB of JSON and 10^4 runs; a mask-matrix artifact would be the scalable record.
7. **Ground truth is reconstructible from `derivation.json`:** its `selection_rule`, `seed` and `selected_from_subset` determine which episodes were corrupted. That's acceptable because the file sits beside `labels.parquet` in the version directory, not in the catalog. The injection feature's method code should read episodes only through `version_episodes` / `Subset` (plan Rule 5).

Sources: [MadryLab datamodels](https://github.com/MadryLab/datamodels), [DataMIL code release](https://github.com/UT-Austin-RobIn/datamil)
