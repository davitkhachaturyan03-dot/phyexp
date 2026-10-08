# Refactor: evaluation layer

Refactorer: vla-refactorer, 2026-10-07. Inputs: `brief.md`, `plan.md` (r3), `execution-log.md`, `analysis-r1..r3.md`,
`dispute-r2.md`. Labels: **measured** (I ran it; outputs are under `data/scratch/refactor_baseline/`, gitignored),
**quoted**, **inferred**.

Base commit: `553b3d1` (analyzer round 3 PASS). Refactor commits:
- `a3b57fc`: the findings, recorded before any edit;
- `a3757de`, `7697000`, `bf74b87`, `84e166a`: no behaviour change;
- `285bb62`: N6, a deliberate verifier change;
- the commit that records this file and the N6 artifact.

## Findings recorded before any edit

### F1. The committed AC6.3 artifact is stale (round 1); HEAD's code writes a different column order

- **What differs (measured).** The baseline run of `scripts/verify_returns_s6.py --join data/eval/returns-3392bb4d5491/returns.parquet`
  on HEAD `553b3d1` (no edits) rewrote `artifacts/s6_verify_returns.json` with
  `"AC6.3" → "metric_columns": ["ret_f30", "ret_rmst60", "ret_auc_expected", "ret_auc_sampled", "ret_hrt60_lin", "ret_censored60"]`.
  The committed file (`git show HEAD:…/s6_verify_returns.json`) has
  `["ret_f30", "ret_rmst60", "ret_auc_sampled", "ret_auc_expected", "ret_hrt60_lin", "ret_censored60"]`.
- **Every other AC6.3 value is the same:** shape `[960, 8]`, `checkpoint_matches: true`, `check_scores` 960/960 `ok`,
  `stale_rejected: true`, and the stdout line `== RETURN_COLUMNS: True`.
- **Why (read from the source and git history).** Round 2 (`1c1671a`) made `returns.RETURN_COLUMNS` =
  `metrics_registry.return_columns("W")`, which lists `auc_h`'s columns as `ret_auc_expected, ret_auc_sampled`
  (registry entry order). The round-1 list had the two AUC columns the other way round. AC6.3 was not re-run in round 2,
  so the committed artifact still shows the round-1 order. The rest of `s6_verify_returns.json` was regenerated in
  round 2, because the script merges each mode's keys into the existing file.
- **Impact.** The set of columns is unchanged and the AC6.3 claim holds (the frame's columns equal `RETURN_COLUMNS`).
  Only the order in the recorded artifact is stale. This is a pre-existing difference between HEAD's output and the
  committed evidence. The refactor did not cause it. I restored the committed file and kept the regenerated copy at
  `data/scratch/refactor_baseline/changed_artifacts/s6_verify_returns.json`. Whether to refresh the evidence is the
  summarizer's or owner's call.

### Other baseline differences from the committed artifacts (bookkeeping only; committed versions restored)

- `artifacts/s0_verify_blind.json`: `timing_s.hash_and_blind_check` 2.0 → 1.9 and `total` 14.2 → 14.1. Wall-clock only.
- `artifacts/s5_ac5_9_grade_tests.json`: `default_run` id, and `runs_before_after` 38/38 → 43/43. The second is the
  number of runs in the scratch copy of the store. The real store has gained runs since round 2, and before == after
  still holds. Both refusal messages are identical.
- Every other artifact the baseline rewrote is byte-identical to the committed one (`git status` showed only these three),
  including `s7_ac7_1_seed_sweep.json`, `phail_validation.json`, both S2 figures, `s6_metrics_registry.json` and
  `s1_metric_registry.md`.

## Finding made during the refactor

### F2. An eval run's `trajectories` digest names a physical copy, not the content (measured)

- **Test.** A noop eval with `--record-trajectories` (libero_spatial:0 and libero_10:8 × 2 init states, τ 30 steps)
  was run three times: once on HEAD's code and twice on the refactored code.
- **Result.** All three runs produced the same rollout record and the same frames: every image byte, state, action and
  index is equal. `meta/stats.json` lists its features in a different order in each run (`index, …` / `action, …` /
  `observation.state, …`), even between the two runs of the same code. The values are equal. The data parquet bytes
  differ too, with the same contents.
- **Consequence.** `vlapoc.runs.eval_run` records `sha256_json(file_digests(trajectories))` as the trajectories entity
  id. A bit-identical re-run therefore gets a different id. `lineage.file_digests` already says it identifies "one
  physical copy, not the content". I added a one-line comment at the call site. No behaviour change.
- **Follow-up (inferred).** If attribution needs a content-stable trajectory id, it should hash the frames, as
  `catalog.episode_content_hash` does for training data. That decision belongs to the attribution feature.

## Refactor

| File | Change | Why it helps a reader |
|---|---|---|
| `vlapoc/metrics_registry.py` | Module docstring says where the registry sits (Steps 2–4) and to read it first. Docstrings for `entry`, `verdict_metrics`, `to_json`; comments on the `T_W/T_O/T_P` constants and `FAMILY_TAU_S` | It is the vocabulary every return column and verdict metric comes from. The reader now knows that before opening anything else |
| `vlapoc/evaluation.py` | Module docstring: a pipeline position (Step 2 item 4) and a reading order. `libero_arm` and `libero_human_arm` moved next to `phail_cohort_arms`, so all arm builders sit in one section; `json` imported at module level | The three ways an arm (`{cell: [Episode]}`) is built are now side by side, ahead of the metrics that consume them |
| `vlapoc/libero_goals.py` | Module docstring: a pipeline position (Step 2 item 2, the human reference). `rle` now uses `itertools.groupby` instead of a hand-written loop (checked equal on 2,004 arrays) | One line in place of eight; the stdlib does the run-length grouping |
| `vlapoc/rollouts.py` | Module docstring: a pipeline position and a reading order. `run_rollouts`' 60-line loop body split into `_first_episode_steps`, `_rollout_row`, `_check_against_lerobot` and `_trajectory_episode`. Docstrings on the `ProbedLiberoEnv` methods and `NoopPolicy`; comments on the horizon patch and the arm-action-norm column. `conjunct_labels` reuses `libero_goals.conjunct_label`; `_Capture` imports torch normally instead of `__import__("torch")` | The loop now reads as its steps: run a batch, measure each rollout's length, build its record row, cross-check it with LeRobot, keep its frames |
| `vlapoc/returns.py` | Module docstring: a pipeline position (Step 3 item 1, and why returns come after rollouts). `returns_run` and `main` both wrote the same four files with duplicated code; that code is now `write_returns`. `FPS = 20` names the two literal `20`s. Docstrings on `_sample_rng` and `main` | One place says what a returns run writes. The 20 Hz conversion is named |
| `vlapoc/verdict.py` | Module docstring: a pipeline position (Step 4 items 3–4), a reading order and two usage lines. `compare_policies` split into `_check_batch_plans`, `_bootstrap_differences`, `_bootstrap_p` and `_metric_result`, keeping the RNG draw order and the result's key order. Docstrings on `describe`, `main`, `_resample`, `_plan_key`; a comment on why the gate and the metric bootstrap use separate generators | `compare_policies` now reads as the two steps the PoC names: gate, then per-metric CIs with Holm |
| `vlapoc/runs.py` | **Docstring corrected**: it said `eval` is a stand-in, but it has been real since S5. The stage map now places `returns` (Step 3 input) and `verify` (Step 4 gate). The eval-stage code (`CANONICAL_TASKS` … `eval_run`) gets its own section; it had sat under the "stand-in stages" header. `attribute_standin` sits next to `screen_standin`. `scratch_lineage_copy` moved to the backend section and gained two keyword options, `dirs` and `mlflow_db`; with the defaults it does exactly what it did (N5). A comment at the trajectories digest (F2) | The file's map of stages matched the code again; a newcomer finds the eval stage under its own header |
| `vlapoc/derive.py` | `read_stored_episode` and `iter_stored_episodes` built the canonical episode dict with the same five lines; now `_canonical_episode` does it once | One definition of "the canonical episode" |
| `vlapoc/subset_stats.py`, `vlapoc/lerobot_train_entry.py` | Pipeline position (inside the implicit training run); docstrings on `main` and the `with_subset_stats` wrapper | Says why a training-time module belongs to the evaluation layer |
| `scripts/verify_runs_s5.py`, `verify_lineage_s3.py`, `verify_derive_s4.py` | **N5**: each builds its scratch store with `runs.scratch_lineage_copy(…, dirs=…, mlflow_db=…)` instead of its own copy loop (four call sites) | One scratch-store implementation, the one that repoints the artifact location |
| `scripts/verify_verdict_s7.py` | **N6**: `--phail-seeds` applies the r3 AC7.1 criterion (`ac7_1_r3`) and prints it. The superseded all-seeds rule is still printed, labelled as such | The verifier's pass line agrees with the owner-approved AC (details below) |
| `scripts/lineage_dry_run.py` | **Docstring corrected**: it called the evals stand-ins and spoke of an `outcome not_run` rollout plan; both are gone since S5. `EVAL_FEATURE_ART` moved up with the other constants; comments on `--eval-smoke`'s default overrides | The docstring describes what the script does today |
| 13 other scripts (`eval_smoke`, `human_reference`, `phail_*`, `fix_v1_leak_s0`, `verify_*`) | One-line docstrings on 52 functions that had none (the `acN_M` checks, CLI `main`s, helpers) | Each check says which AC it covers |

Not touched: `vlapoc/third_party/` (byte-identical: `phail_stats.py` sha256 `802387d3…`, as recorded in AC1.1), and
`vlapoc/lineage.py`, `catalog.py` and `libero_convert.py` (`git diff 553b3d1 -- …` is empty). Their S0 docstrings
already match the code.

**Static check (measured).** `ast_docs_only.py` (in `data/scratch/refactor_baseline/harness/`) compares each changed
file with `553b3d1` after stripping docstrings. Fifteen files are the same program; the only differences are
docstrings and comments: `eval_smoke`, `fix_v1_leak_s0`, `human_reference`, `phail_export_cohort`, `phail_fetch`,
`phail_validate`, `verify_blind_s0`, `verify_metrics_s1`, `verify_operations_s4`, `verify_resets_s5`, `verify_returns_s6`,
`verify_rollouts_s5`, `verify_subset_stats_s8`, `lerobot_train_entry`, `metrics_registry`. The rest changed code and are
covered by the behaviour check below.

## Behaviour check

Baseline: every command below, run on `553b3d1` before any edit (`data/scratch/refactor_baseline/outputs/`). After:
the same commands on the refactored commits. Environment: `HF_DATASETS_CACHE=$PWD/data/hf_datasets_cache`,
`LIBERO_CONFIG_PATH=$PWD/data/libero_config`, `HF_HUB_OFFLINE=1`, `MUJOCO_GL=cgl`. Stdout is compared after dropping
progress bars and timings (`compare_outputs.py`). Every MLflow write went to a fresh `runs.scratch_lineage_copy`.

| Command | Before == After? | Note |
|---|---|---|
| `scripts/verify_metrics_s1.py` (S1) | Yes | stdout identical; `s1_*` artifacts byte-identical |
| `scripts/phail_validate.py` (S2, stored cohort parquet) | Yes | stdout identical; `phail_validation.json` and both figures byte-identical |
| `scripts/human_reference.py --summary` (S3) | Yes | identical; tables and figure byte-identical |
| `scripts/human_reference.py --operations` (S4) | Yes | identical; it regenerates `human_operations.parquet` byte-identical (sha `9326e751…`) |
| `scripts/verify_operations_s4.py` (S4) | Yes | 7/7 synthetic cases, one implementation, two callers |
| `scripts/verify_rollouts_s5.py --summary <AC5.4 dirs>` (S5) | Yes | identical |
| `scripts/verify_rollouts_s5.py --grade-tests <scratch copy>` (S5 r2) | Yes | refusals identical; only the default eval's run id and the store's run count differ |
| `scripts/eval_smoke.py --policy noop … --record-trajectories` (2 tasks × 2 inits, τ 30; HEAD vs refactor, untracked) | Yes, content | records equal in every non-volatile column; all 120 trajectory frames equal (images, state, action, indices). `stats.json` key order and parquet bytes differ even between two runs of the same code (F2) |
| `scripts/verify_returns_s6.py --synthetic` / `--real <AC5.4 dirs> --against-verdict` / `--join …` (S6) | Yes | identical (AC6.3's column order differs from the *committed* artifact on HEAD too: F1) |
| `python -m vlapoc.returns --eval-run e92604a8…` (HEAD vs refactor) | Yes | all four output files byte-identical, and identical to the stored round-2 files |
| `scripts/verify_verdict_s7.py`, `pytest scripts/verify_verdict_s7.py`, `--family-consistency` (S7) | Yes | identical; 4 passed |
| `scripts/verify_verdict_s7.py --phail-seeds 0 1 2 3 4` (S7 sweep) | Yes, on the refactored `vlapoc` | byte-identical stdout before N6. After N6, the 30 per-seed lines and the stored `results` are identical; only the pass line changed, deliberately (N6 below) |
| `scripts/verify_verdict_s7.py --plan-guard … --lineage-root <scratch copy>` (S7 AC7.8) | Yes | refused / override label / same-plan label identical; run ids differ |
| `scripts/verify_subset_stats_s8.py --check v0`, `--normalizer` ×2, `--diff` (S8) | Yes | identical |
| `python -m vlapoc.lineage version check-blind v1`; `scripts/verify_blind_s0.py` (S0) | Yes | identical apart from timings (`write` 11.2 → 11.1 s) |
| `scripts/lineage_dry_run.py --eval-smoke` on a `scratch_lineage_copy` (S9; run twice after: once after the `vlapoc` commits, once after the script commits) | Yes | same verdict ((iii), D̄ 0, p 1.0, smoke label, `attribution_grade` false), same 16-run trace reaching the 20 raw files, same rollout records (only `train_run_id` differs), same `returns`/`operation_returns`/`verdict.json`, and the same checkpoint weights `cdee42fb…`/`fac999b0…` (bit-identical ACT training on MPS, as in rounds 1 and 2) |
| `scripts/verify_lineage_s3.py --summary <scratch>` (moved script; HEAD vs refactor) | Yes | identical apart from random temp-dir names; the summary JSONs are equal |
| Old copy code vs `scratch_lineage_copy(dirs=…, mlflow_db=…)` at the four moved call sites | Yes | the same files, bytes and sqlite dump (`n5_setup_equivalence.py`) |
| `read_stored_episode` (v0 episodes 0, 517, 999) and the default `scratch_lineage_copy` (HEAD vs refactor) | Yes | identical array digests; identical tree and DB dump |
| Main store `data/lineage/mlflow.db` sha256 | Unchanged | `7845d8d7…` before and after every run. `find data/lineage -newer <baseline>` finds nothing |

**Verdict: the behaviour check is clean.** Every difference is a timing, a run id, a temp-dir name, or LeRobot's
run-to-run key order in `stats.json` (F2, present between two runs of the same code). The one intended output change
is N6. The files no command exercised are docs-only by the AST check above.

## N5: data+lineage regression

Asked for by analysis-r3 N5: the data+lineage feature's checks had not been run since per-subset normalization (S8)
and the real eval stage (S5) became the defaults. Every MLflow write went to a scratch copy, and the main `mlflow.db`
stayed `7845d8d7…`. Outputs are in `data/scratch/refactor_baseline/n5/`. The data+lineage artifacts these scripts
rewrite were saved there, and the committed versions were restored.

| Run | Result | Detail (measured) |
|---|---|---|
| `scripts/lineage_dry_run.py --k 20 --seed 0` (full chain, no `--eval-smoke`), on `scratch_lineage_copy` via `harness/dry_run_scratch.py` | **PASS** | exit 0 in 77 s; 14 runs FINISHED; all four training runs list exactly their subset (960, 960, 940, 940 episodes) with per-subset stats; three real smoke evals; `diff` shows removed 20 / seed only / removed 20 with no params or config difference; the trace of `train-drop-top-20` reaches the 20 raw files; the tamper copy is refused with 54/54 runs before/after; screening and attribution stand-ins keyed 960/960 with `check_scores ok`. AC6.2, AC6.3 and AC6.6 equal the committed `s6_summary.json` |
| — compared with the committed data+lineage summary | expected differences only | AC6.4's trace and AC6.7's graph (42 nodes, 18 runs, against 36/14) also reach the original chain's runs. The scratch store is a copy of the real one, and content-addressed entities (the placeholder score table, `v1/candidate`) now have two producers. AC6.5 records a newer key, `new_run_dirs: []` |
| `scripts/verify_runs_s5.py --r2` (now on `scratch_lineage_copy`) | **PASS** | AC5.8: lowest/highest drops correct and the child equals the S6 child. AC5.9: leave-one-task-out 912 episodes, trace reaches 20 raw files. AC5.10: three reproducible random subsets. Real DB and real mlartifacts tree unchanged; scratch removed. Same membership matrix as committed; only run ids differ in `s5_r2_summary.json` |
| `scripts/verify_runs_s5.py` (round-1 mode; now on `scratch_lineage_copy`) | **FAIL** (pre-existing; not caused by this refactor) | AC5.1 (7 runs, every stage known, digest search) and AC5.2 (every §2.1 field recorded, with `per-subset` normalization) pass. Then **`KeyError: 'suites'` at AC5.3**: its checks still read the round-1 `eval_standin` params (`suites`, `libero_task_ids`, `init_state_indices`, `seeds`), tags (`eval.checkpoint.loaded`, `eval.checkpoint.n_params`) and plan column `outcome`. The real `eval_run` (S5, `4448ed2`) records none of them. The same lines are in HEAD's version, so the refactor did not cause it |
| — the same run with only its AC5.3 block reading what `eval_run` records (`harness/verify_runs_s5_ac53_patched.py`; a diagnostic, not committed) | AC5.3–AC5.6 pass | AC5.3: `checkpoint_sha256` = train-A weights, `train_run_id` = A, `input.checkpoint` tag = weights, `stand_in=false`, one censored rollout, 33 columns. AC5.4: trace reaches 20 raw files; `consumers` lists train-A and not train-B; `diff` A/B removed 5, A/A-seed1 seed only. AC5.5: v1-keyed table accepted, v0-keyed rejected (6/10). AC5.6: sqlite, offline |

**Follow-up for the report (not done here: it changes a data+lineage verifier):** update `verify_runs_s5.py`'s AC5.3
to the real eval stage's fields, as in the diagnostic. **Moved onto `scratch_lineage_copy`:** `verify_runs_s5.py`
(both modes), `verify_lineage_s3.py` and `verify_derive_s4.py --padding`, with no behaviour change (store equivalence
and the `verify_lineage_s3` before/after above). Still owed to later features: enforcing the v1 fence at the method
entry points of screening and attribution.

**One incident, contained.** My first attempt at the diagnostic exec'd the script without a `__main__` guard, and the
spawn-based process pool re-ran it in every worker. The workers raced on the scratch dir `data/lineage_scratch/s5`.
Nothing outside it changed: `find data/lineage -newer <baseline>` finds nothing and `mlflow.db` is still `7845d8d7…`.
The guarded re-run is the result reported above.

## N6: the r3 pass line in `verify_verdict_s7.py --phail-seeds`

Asked for by analysis-r3 N6. AC7.1 as restated in plan r3, after owner ruling B (`dispute-r2.md`, brief "Owner
rulings"): "GR00T vs ACT → (ii) in ≥ 4/5 seeds; the gate fires in 5/5 and RMST favours GR00T in 5/5", other pairs
5/5. The old pass line encoded the superseded r2 rule (every pair in every seed) and printed `False`.

- **Change (`285bb62`).** A new `ac7_1_r3(out, seeds)` evaluates the r3 criterion from the per-seed results.
  - GR00T–ACT: (ii) in ≥ 4/5 of the seeds, gate p < 0.05 in every seed, and `phail_rmst240` favouring GR00T in every
    seed.
  - Every other pair: its expected outcome in every seed.
  - The superseded rule is still printed, labelled "(the superseded r2 rule)".
  - `s7_ac7_1_seed_sweep.json` gains `all_seeds_match_expected` and an `r3` block, and its `pass` is now the r3
    result. The per-seed computation is untouched.
- **Result (measured, seeds 0–4).** The 30 per-seed lines are byte-identical to the baseline's; only the final lines
  changed. The new ones are:
  - `AC7.1 every (pair, seed) outcome equals the pair's expected outcome: False (the superseded r2 rule)`
  - `AC7.1 (r3) GR00T vs ACT: (ii) in 4/5 seeds (needs >= 80%), gate fires 5/5, RMST favours GR00T 5/5: True; other pairs match in every seed: True (OpenPI-GR00T 5/5, OpenPI-ACT 5/5, OpenPI-SmolVLA 5/5, GR00T-SmolVLA 5/5, ACT-SmolVLA 5/5)`
  - `AC7.1 (r3, owner ruling B, dispute-r2.md) pass: True`
- **Artifact.** The regenerated `artifacts/s7_ac7_1_seed_sweep.json` is committed: `results` is identical to the
  previous file, and `pass` went from `false` to `true` under the r3 criterion.

## Left as-is (and why)

- **N1 (scratch `out_dir` defaults), N2 (legacy record schema), N3 (batch-plan key fields).** These change behaviour.
  They are follow-ups for the report, per the dispatch. N1 was visible during this work: `verify_run` (the plan-guard
  test) still writes `verdict.json` into the real `data/eval/verdict-W-…/` even with a scratch `lineage_root`. The
  regenerated files were byte-identical to the stored ones (backup in `data/scratch/refactor_baseline/`).
- **`verify_runs_s5.py`'s stale AC5.3 checks** (see N5). The script still reads the round-1 stand-in's params, which the
  real eval stage does not record. Correcting the checks would change a data+lineage verifier, and the dispatch did not
  ask for that, so this is a follow-up.
- **F1 and F2.** Both are recorded above. Refreshing the AC6.3 evidence, or making the trajectories id content-based, is
  for the summarizer, the owner or the attribution feature.
- **Not re-run, by instruction or because a run would change state:**
  - The full-size SmolVLA evals (AC5.4: about 1.5 h on MPS) and the 2.9 h human replay (`human_reference.py` without a
    flag): the dispatch said not to. The rollout record path and `operations_from_trace` were re-run at smoke scale
    instead (S9 chain, noop trajectory test). The replay code (`_replay_task`, `replay`, `cmd_replay`) changed only in
    docstrings (AST check above).
  - `fix_v1_leak_s0.py`: idempotent, but it rewrites v1's `fence.json` with a new timestamp. Docstrings only.
  - `phail_fetch.py` would download from the network; `phail_export_cohort.py` runs in `.venv-phail` and rewrites the
    stored cohort parquet. Both are docstrings only.
  - `verify_resets_s5.py` and the other `verify_rollouts_s5.py` modes (`--rerun`, `--pairing`, `--horizon`,
    `--trajectories`, `--t0-model`): docstrings only. The rollout code they drive was checked through the S9 chain and
    the trajectory test.
  - `verify_derive_s4.py --padding` writes a 7 GB full-scale derivation. Its moved setup was proven equivalent by
    comparing the stores (N5); the rest of the function is unchanged.
- **Long verification scripts were not restructured.** `phail_validate.py`, `verify_returns_s6.py` and
  `human_reference.py cmd_operations` are 60–90-line report functions. They print and record evidence that is already
  in the artifacts, so splitting them would change no reading path. They got docstrings instead.
- **Leftovers on disk, for owner cleanup (N4 style; nothing was deleted):**
  - `data/lineage_scratch/refactor-*`: the scratch stores made by this work, a few MB each plus their artifacts.
  - `data/lineage_scratch/s5.pre-refactor-n5`: the 2.3 GB round-1 store N4 lists, renamed aside because
    `verify_runs_s5.py` replaces `data/lineage_scratch/s5`.
  - The new `data/lineage_scratch/s5` written by that run.
  - The `data/runs/eval-smoke-k5-s0-*` and `dry-run-k20-s0-*` chains, the `data/eval/returns-*` dirs of the scratch
    eval runs, and `data/scratch/refactor_baseline/` (the baselines and the harness).

## Reading guide

Read in this order. Each line says what the file gives you on that path. The pipeline position of every module
is in its docstring ("Where it sits").

1. **`docs/dataflow.md`**: Steps 2–4 of the PoC flow. This feature implements Step 2 (Evaluation), the input of Step 3
   (one return per metric per rollout) and the gate of Step 4 (Verification).
2. **`vlapoc/metrics_registry.py`**: the vocabulary. Every PoC metric, its declared functional, its family (W
   whole-task, O per-operation, PhAIL), its return column and how the verdict tests it. Every later file looks names up
   here.
3. **`vlapoc/third_party/__init__.py`, then a skim of `phail_stats.py`**: PhAIL's own code, vendored unmodified. Look at
   `episode_to_T_pairs` (how an episode becomes (T, E) pairs, with censored tails and T = ∞ ghosts) and
   `kaplan_meier`.
4. **`vlapoc/evaluation.py`**: the metric layer. `Episode` and arms, then every scalar as a functional of the KM CDF
   (F(τ), RMST, HRT, AUC, UPH, MTBF/A), macro-KS, and the episode-clustered bootstrap.
5. **`scripts/phail_validate.py`**: the evidence that step 4 is right. It reproduces PhAIL's published numbers on
   PhAIL's released data (S2).
6. **`vlapoc/libero_goals.py`**: LIBERO's own goal check, the demo replay, and `operations_from_trace`, the single
   rule that turns a goal-conjunct trace into per-sub-task T.
7. **`scripts/human_reference.py`**: builds the fixed human anchor (`human_reference.parquet`,
   `human_operations.parquet`) from the 1000 v0 demos with that rule.
8. **`vlapoc/rollouts.py`**: the rollout harness. Start at `RECORD_COLUMNS` (what one rollout record is), then
   `run_rollouts` (LeRobot's `rollout()` plus a probe), then `ProbedLiberoEnv` (what is added to LeRobot's env, and why
   there is one env per rollout).
9. **`vlapoc/runs.py`, section "the eval stage"**: `eval_run` makes rollouts a lineage-tracked MLflow run, with the
   batch plan and the attribution-grade protocol check. The module docstring explains how lineage is encoded in MLflow.
10. **`vlapoc/returns.py`**: one return per metric per rollout (family W) and per operation (family O), plus the
    empty episode × metric frame the attribution feature will fill.
11. **`vlapoc/verdict.py`**: the two-step gate. Macro-KS decides whether two policies differ; per-metric bootstrap CIs
    with Holm give the direction. `verify_run` records the verdict as a run.
12. **`scripts/lineage_dry_run.py`, function `eval_smoke`**: the whole chain in one place. select → train ×2 → eval
    ×2 → returns ×2 → verify → trace back to the raw files. `scripts/eval_smoke.py` is the CLI for one eval run.
13. **`vlapoc/subset_stats.py` and `vlapoc/lerobot_train_entry.py`**: per-subset normalization (owner decision 3),
    i.e. what a trained policy is a function of.
14. **The S0 leak fix, when it matters**: `vlapoc/lineage.py` section "blindness of derived versions"
    (`compare_blind`, `assert_blind`, the v1 fence) and `vlapoc/derive.py`'s module docstring (why derivations keep the
    parent's order).
15. **`scripts/verify_*_s0..s8.py`**: one script per story. Each function names the AC it checks; the artifacts they
    write are the evidence in `execution-log.md`.
