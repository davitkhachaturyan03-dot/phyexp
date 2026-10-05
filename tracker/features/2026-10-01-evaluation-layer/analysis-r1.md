VERDICT: REWORK
ROUTE: planner+executor
ROUND: 1

<!-- Saved verbatim by the orchestrator from the analyzer's returned text: the analyzer's own
     write was refused by the harness ("Subagents should return findings as text, not write
     report files"). -->

# Analysis, round 1: evaluation layer (PhAIL primitives on LIBERO)

Analyzer: vla-analyzer, 2026-10-04/05. Evidence labels: **measured** (I ran it this session; logs were under
`data/scratch/analyzer_logs/`, now deleted, and every number is quoted here), **quoted** (source line, doc or
PDF page), **inferred** (my judgement).

Constraints kept (measured):
- `data/lineage/mlflow.db` sha256 was `02890131…` before and after.
- The sha256-of-all-files of `data/lineage/` was `b9405be1…` before and after.
- Every run that writes records ran against a scratch copy of the lineage store.
- All executor artifacts I regenerated were restored to their committed state (`git status` clean).
- `~/.libero` is absent and no processes are left running. Free disk is 113 GiB.

## Summary (3–5 lines)

**Reproduction.** Every local AC except the ACT-vs-OpenPI part of AC7.1 reproduces, most of them bit for bit:
- the metric layer, the PhAIL validation, the six PhAIL verdicts and the null calibration;
- the human reference (a 20-demo re-replay matches 20/20) and the per-sub-task T;
- the rollout determinism, a SmolVLA batch-1 slice (steps 78/71, as logged) and the reset-drift evidence;
- the return identities, the subset stats and the one-command chain.

**Coverage.** The work covers the §10 p. 20 "Evaluation layer" row, validated on PhAIL's real data.

**Three defects block per-metric attribution readiness (the owner's crucial emphasis):**
1. **Two functionals per metric on libero_10.** The S6 per-rollout returns use whole-task T. The S7 verdict
   and S1's LIBERO arms use per-operation T. On libero_10:8 they disagree: RMST(60) 48.4 vs 37.0 s, F(30 s)
   0.30 vs 0.41, AUC 0.085 vs 0.185, HRT 0.416 vs 0.272. So the gate would verify a different functional
   from the one attributed.
2. **The operation return ranks wrongly.** `ret_rmst60_op` scores a never-completed operation above completed
   ones on real data.
3. **Smoke runs tagged attribution-grade.** Smoke runs are tagged and labelled "attribution-grade".

**AC7.1 and housekeeping.** AC7.1's ACT-vs-OpenPI expectation is a faithful quote of PhAIL's prose, which
PhAIL's own released macro-KS contradicts, so the planner must restate it. I also found that PhAIL's cohort
export is not deterministic (Python set order), and that three metric registries disagree on UPH.

## Rebuttals received

| From | Point disputed | Their evidence | Conceded? | Why / why not |
|---|---|---|---|---|
| none (round 1) | | | | |

## Acceptance criteria

| Story | AC | Claimed | Reproduced? | Evidence |
|---|---|---|---|---|
| S0 | AC0.1 | ✅ | ✅ | measured: v1 `version_episodes` has 21 columns == v0's, `parent_content_hash` absent; catalog copy `libero_spatial_10__v1` 21 columns == v0's; `version show v1` → `16076b9c5cb2…` |
| S0 | AC0.2 | ✅ | ✅ | measured: `labels.parquet` ⋈ v1 on (`episode_id`, `content_hash`) = 100/100; `parent_hashes(v1)` == v0 `content_hash` for 1000/1000 |
| S0 | AC0.3 | ✅ | ✅ | measured `verify_blind_s0.py`: `episode_index equal to parent's: 20/20; edited episodes now at [3, 11, 16]; 99 -> 119`; identical file ranges; `check_blind(child) = []` |
| S0 | AC0.4 | ✅ | ✅ | measured: `version check-blind v1` → `['episode_index']; fenced: True`; v0 → `[]`. `fence.json` read |
| S0 | AC0.5 | ✅ | ✅ | measured: extra column, swapped index, changed `source_demo` and shuffled rows are all rejected; `assert_blind(v1)` refused |
| S0 | AC0.6 | ✅ | ✅ | read: data+lineage `changelog.md` 2026-10-02 entry matches the measured facts (both channels, data files 73–80, AC4.1/AC4.7 named); `report.html` has dated lines; `index.html` §2.5 row names `check_blind` and the fence |
| S0 | AC0.7 | ✅ | ✅ | read: `lineage.py` `version_episodes` and blindness section, `derive.py` docstrings cite §2.1 item 4 p. 3 |
| S1 | AC1.1–1.7 | ✅ | ✅ | measured `verify_metrics_s1.py`, identical to the log: sha `802387d3…` == clone; KM vs scipy 3.331e-16; ghost rate vs g/n 2.8e-17; RMST identity 0.0; CI width ratios 1.399 / 1.228; registry 14, missing [] |
| S2 | AC2.1 | ✅ | ✅ (see finding E3) | measured `phail_validate.py` on the stored parquet: max \|ours − PhAIL\| 0 to 3.553e-15; macro-KS identical for all 6 pairs; regenerated artifacts byte-identical. But a fresh export reorders rows and changes every bootstrap p (E3) |
| S2 | AC2.2 | ✅ | ✅ | measured: OpenPI 163/165, GR00T 164/165, ACT 151/151, SmolVLA 116/118, Teleop 396/396. My re-export has the same 990 `s3_url`s with identical contents |
| S2 | AC2.3 | ✅ (incomplete branch) | ✅ | measured: 23/26 within tolerance; the 3 misses are the pooled RMST of the 3 models with private USB-C episodes; macro HRT exact; macro-AUC within ±0.005 |
| S2 | AC2.4 | ✅ | ✅ | measured: winners GR00T / OpenPI / GR00T / ACT, 3 distinct top-1 |
| S2 | AC2.5 | ✅ | ✅ (single seed) | measured: endpoint error 0.582 (human resampled) vs 0.667 pp (fixed). The 8-seed z-score run was not re-run |
| S2 | AC2.6 | ✅ | ✅ (stored listing) | measured: printed from the stored listing; consistent with the export (5 episodes behind HTTP 403). The live bucket probe was not re-run |
| S2 | AC2.7, AC2.8 | ✅ | ✅ | measured: figures regenerate byte-identical; UPH 3 forms 46.4 / 97.1 / 63.2 (OpenPI) |
| S3 | AC3.1 | ✅ | ✅ | measured: `~/.libero` absent after every sim run of mine; `sim_env.json` read |
| S3 | AC3.2 | ✅ | ✅ | measured: stored parquet 1000 rows, `success_any` 1000, errors 0, consistent 1000. **Re-replay of 20 demos (1 per task, incl. moka-pot demo_19):** `T_success_steps`, `conjunct_first_true` and `success_trace_rle` equal 20/20 |
| S3 | AC3.3, AC3.4 | ✅ | ✅ | measured: T < duration 1000/1000; `--summary` reproduces 1.162× / 1.049×, artifacts byte-identical |
| S3 | AC3.5 | ✅ | ✅ | read: `libero_goals.py` "Alignment conventions"; the 0.25 s residual is labelled inferred |
| S3 | AC3.6 | ✅ | ✅ (in kind) | measured `--bddl-check --only libero_10:8`: this run gave no success 5/50 and different T 4, the log gave 2/50 and 5. The instances differ by design, since the BDDL scene is not deterministic across resets (mechanism reproduced under AC5.2b) |
| S3 | AC3.7, AC3.8 | ✅ | ✅ | measured: 20/20 tasks with 50 init states. Changelog entry read |
| S4 | AC4.1–4.6 | ✅ | ✅ | measured: n_ops `[2,2,2,2,2,1,2,2,2,2]`, maintained `{8: (turnon flat_stove_1)}`, 500/500; Σ T_k == T 500/500, 950 op samples; 7/7 synthetic; 1 definition, 2 callers; artifacts byte-identical |
| S5 | AC5.1 | ✅ | ✅ | measured: `validate_record` → [] on all 6 stored records; noop 6/6 censored at 1200 |
| S5 | AC5.2 | ✅ | ✅ | measured: `--rerun eval-noop-smoke-a` in a new process → `differing columns: none`; ACT a vs b → `{'rows': 4, 'columns_compared': 30, 'differing': {}}`; resets reproduce exactly (below) |
| S5 | AC5.3 | ✅ | ✅ | measured: `trace 3392bb4d…` JSON-equal to `s5_ac5_3_trace.json` (formatting differs, content equal) |
| S5 | AC5.4 | ✅ | ✅ | measured `--summary`: 6/10 (spatial:0), 3/10 (libero_10:8), RMST(60) 26.19 / 48.40 s. **Slice re-run:** SmolVLA MPS batch 1, τ 280, init 3 → success at step 78, init 4 → step 71, exactly `s5_smolvla_batch_effect.txt` |
| S5 | AC5.5 | ✅ | ✅ (spatial store) | measured: 10 episodes, 5237 frames = Σ `sim_steps`; `frame_index` = policy step; no video keys; image (3, 256, 256) |
| S5 | AC5.6 | ✅ | ✅ | measured `verify_resets_s5.py`: libero_10:8 batch 1 vs 4 (state, body_pos, site_pos) identical; body_pos hashes equal the log's |
| S5 | AC5.7 | ✅ | ✅ | read: grounding present in `rollouts.py` |
| S6 | AC6.1 | ✅ | ✅ (but see F1) | measured: synthetic 1.07e-14 / 1.11e-16 / 2.78e-17; real 0.00e+00 and 7.11e-15; regenerated return columns == stored. **The identity holds against whole-task KM, which is not the functional the S7 verdict uses on libero_10 (F1)** |
| S6 | AC6.2 | ✅ | ✅ | measured: worst 9.51% ≤ 10% |
| S6 | AC6.3 | ✅ | ✅ | measured: frame (960, 8), `check_scores ok`, stale hash rejected |
| S6 | AC6.4 | ✅ | ✅ as worded (see E2) | measured: 21 entries, missing [] |
| S6 | AC6.5, AC6.6 | ✅ | ✅ | measured: failure-type subsets and column/τ check as logged |
| S7 | AC7.1 | ❌ in part | ❌ reproduced | measured `--phail-all`: stdout identical to `s7_phail_verdicts_stdout.txt`; OpenPI–ACT → (iii), p 0.1109. Robust: seeds 0 on a reordered cohort p 0.1049; PhAIL's own function on 4 loader orders p 0.109–0.123; PhAIL's released JSON p 0.103 "WEAK" |
| S7 | AC7.2 | ✅ | ✅ | measured: OpenPI 0.040 ± 0.027, Teleop 0.070 ± 0.035, identical to the log |
| S7 | AC7.3 | ✅ | ✅ | measured: sup-sign cycle A≻B≻C≻A; verdict direction B from RMST |
| S7 | AC7.4 | ✅ | ✅ | measured: `compare_policies` on the stored rollouts → D̄ 0.60, p 0.0090, (i), SmolVLA; verify run `5bb1615b` tagged smoke |
| S8 | AC8.1–8.4 | ✅ | ✅ | measured: std diff 2.43e-10; saved normalizer/unnormalizer vs subset ≤ 4.54e-08 and vs full 1.09e-02; `diff` → normalization differ true, params [] config []; tag digest matches |
| S9 | AC9.1 | ✅ | ✅ (but see E1) | measured on a scratch lineage copy: exit 0, 124.0 s, 10 stage runs FINISHED, `stand_in false`, trace reaches 20 raw files, verdict (iii). ACT MPS training bit-identical a third time (weights `cdee42fb…`, `fac999b0…`). **The verdict is labelled "attribution-grade counterfactual verdict" (E1)** |
| S9 | AC9.2, AC9.3 | ✅ | ✅ | read: runbook covers AC8.5 / 5.8 / 6.7 / 7.5 with commands, ids, disk estimates; coverage rows proposed |
| — | AC5.8, 6.7, 7.5, 8.5 | ⏳ pending server | n/a | correctly marked; runbook present |

## Points the orchestrator asked me to judge

- **S0 leak fix.**
  - Derived versions now pass `check_blind` (measured, scratch derivation).
  - v1 is fenced in the sense of a lineage record plus `assert_blind`, which refuses it. Nothing calls
    `assert_blind` at an entry point yet, because no method under test exists (quoted, the grep shows callers
    only in the lineage CLI and scripts). That is acceptable for this feature. The screening and attribution
    features must call it.
  - The changelog and report notes are accurate against what I measured.
- **AC7.1 ❌.**
  - The executor's evidence is right (measured, robust across seeds and loader orders).
  - The plan did not invent the expectation. PhAIL p. 9 says "This is outcome (ii) of §3.3: ACT and
    OpenPI/GR00T are different (KS rejects)" (quoted). PhAIL's released `dashboard_per_obj_macro.json` gives
    OpenPI–ACT `p 0.10294852573713144, status WEAK` (measured).
  - PoC §3.3 p. 7's "Macro-KS resolved … OpenPI vs. ACT at 30 rollouts per … cell" is a detection-rate
    statement. It comes from PhAIL §5.2's subsampling experiment at τ = 120 s (quoted p. 8), not a
    full-cohort test.
  - Artifact over prose, so the planner restates (F-P1).
- **The batch-2 slip.**
  - The `…INVALID.txt` rename is correct.
  - The comparator explanation (`str(ndarray)` truncation vs JSON list) is consistent with the code.
  - The clean recheck reproduces: ACT a vs b `{}` (measured).
- **LIBERO fixture drift and the one-env-per-rollout rule.**
  - Reproduced exactly (measured). One env reset ×3 moves `flat_stove_1_main` 7.2 mm with property
    initializers [3, 4, 5]. Three fresh envs give identical body_pos with initializers [3, 3, 3].
  - The harness builds a fresh env per rollout and keeps `first_trace()` (read, `rollouts.py:338, 161-163`).
  - It is immune for recorded rollouts. The record cannot detect a violation: `t0_state_sha256` misses
    fixture placement (non-blocking).
- **SmolVLA outcomes depend on batch size.**
  - Reproduced (measured: batch-1 outcomes 78 / 71 match; the log's batch 5 gives 71 / never).
  - LeRobot's own docs ask for a fixed batch plan in paired comparisons: "use the same `--seed`, keep
    `--env.init_states=true`, and run each task in a single batch" (quoted, LeRobot LIBERO docs).
  - Not a defect. The fixed batch plan must be written into AC5.8 and AC7.5 (F-P5).
- **`gate_only`.**
  - Accepted as a recorded divergence. PoC §4.4 p. 10 says "each metric's bootstrap CI gives the direction"
    (quoted). When no Holm-corrected CI excludes 0, no direction is established, and saying so is more
    faithful than forcing (i) or (ii).
  - It is one branch (`verdict.py:101-102`), so not over-built.
- **Per-subset normalization as the default.**
  - Correct under owner decision 3, and proven (measured, AC8.2 and AC8.3).
  - The data+lineage full dry run and `verify_runs_s5.py` now run with the new default and were not re-run,
    by the executor or by me. That is non-blocking, and the refactorer must run them.
- **AC8.5 starting point.**
  - Both are documented LeRobot recipes. The SmolVLA docs say "Use `smolvla_base`, our pretrained 450M model,
    and fine-tune it on your data" with `--policy.path=lerobot/smolvla_base` (quoted). The LIBERO page shows
    `--policy.type=smolvla --policy.load_vlm_weights=true` (quoted).
  - The brief withholds the 2.03 GB backbone. The planner should restate AC8.5 on `smolvla_base` (F-P4).
- **Ghost events out of scope.** Confirmed.
  - LIBERO's `step` sets `done = self._check_success()` and nothing else (quoted,
    `bddl_base_domain.py:802-811`).
  - The predicate registry is `true, false, in, on, up, printjointstate, open, close, turnon, turnoff`
    (quoted, `predicates/__init__.py`).
  - There is no off-table, out-of-reach or failure rule (measured grep).
  - Owner decision 6 applies.
- **τ = 60 s.**
  - It is an owner-accepted default (brief, "Non-blocking defaults stand"), so I don't re-open it.
  - On the 20 SmolVLA rollouts no success came after LeRobot's 280/520 budgets (measured `--summary`). τ
    therefore changed no outcome there, at the cost of 1200-step failures. The runbook prices that cost.
- **Registry transfer statuses.**
  - Every PoC metric is present with a status (measured, 21 entries).
  - The registries disagree with each other and with the code on UPH (E2).
  - The registry does not say which T each metric uses on libero_10 (F1).
- **Operation-level return for timed-out rollouts.**
  - This is not only an understatement. It ranks in the wrong direction (F-P3, measured).
- **New dependencies.**
  - `transformers 5.5.4`, `num2words 0.5.14` (the `lerobot[smolvla]` extra, required); and 32 sim-stack
    packages, all additions, including `pytest 9.1.1`, `pluggy`, `iniconfig` (measured freeze diff).
  - `.venv-phail` is separate (Python 3.12).
  - All are logged. The main `.venv` still runs LeRobot training, conversion and eval (measured, the S0
    scratch conversion and the S9 chain).

## PoC scope assessment

**Covered.**
- §3.2 p. 6: every primitive, with ghost events validated on PhAIL's data.
- §3.3 p. 7: the "four scalars, three winners" result reproduced on real data.
- §3.4 p. 7: gate, declared targets, strata, operation-level pairs, failure taxonomy (timeouts only).
- §3.5 A pp. 7–8: T per episode and per sub-task, censoring, the replayed human reference.
- §3.5 B / p. 8 caveat: license CC BY 4.0, 449 episodes / 13.95 GB, private prefixes.
- §4.1 p. 9: four yardsticks and the UPH vs MTBF/A sweep. §4.3 p. 10: one return per metric.
- §4.4 p. 10: two-step verdict, Holm, τ labels.
- §9.1 Steps 2–4 pp. 18–19: as a recorded chain.
- §10 p. 20, Evaluation layer row: built locally. Full scale is pending server.

**Missing.**
- On libero_10, the declared functional of each per-rollout return is not the functional the gate verifies
  (F1). This is the one gap against §3.4 "Declared targets" and §4.4 "for that metric".

**Over-built.**
- Three metric definitions where one would do (E2).
- Otherwise the build is proportionate. The null calibration, figures and sweep are all plan ACs. No caching or
  parallelism was added without a measured need.

**Silent divergences.**
1. The libero_10 whole-task vs per-operation split between S6 and S7 (F1). Neither the log nor a docstring
   flags it. `returns.py` says "T here is whole-task", while `verdict.libero_arm` uses operation pairs.
2. The registry UPH functional contradicts the code (E2).
3. `eval.attribution_grade` means "has a train run", not attribution grade (E1).

## Overcomplication & reinvention audit

| Mechanism (file:line) | Existing OSS alternative | Covers our need? (spec rules checked) | Evidence | Recommendation |
|---|---|---|---|---|
| KM / RMST / KS / macro-KS core (`third_party/phail_stats.py`, vendored) | PhAIL `build/stats.py` itself | yes, it is the defining code | measured: sha `802387d3…` == clone; GitHub API: `18ce72d` is still HEAD (2026-05-27) | keep (reuse) |
| KM via lifelines or scipy instead of vendored | `lifelines.KaplanMeierFitter`, `scipy.stats.ecdf` | **no**: §3.2 ghost events need T = ∞ | quoted lifelines `utils`: `raise TypeError("Infs were detected in the dataset…")`; measured scipy 1.18.1: `` `uncensored` must contain only finite values `` | keep vendored |
| `evaluation.macro_ks` loop (`evaluation.py:317-358`) | vendored `per_obj_macro_bootstrap_ks` | no for LIBERO: `_collect_T_E` re-derives pairs with the default 0.3 s threshold (quoted `phail_stats.py:171-178`); LIBERO needs threshold 0 | measured: identical D and p to PhAIL's on PhAIL data (6 pairs) and on synthetic data | keep |
| `logrank_stratified` (`evaluation.py:361-396`) | `statsmodels.duration.survfunc.survdiff(time, status, group, strata=…)`; `scipy.stats.logrank` | statsmodels yes but not installed; scipy no strata | quoted statsmodels docs "strata: Optional stratum indicators for a stratified test"; measured scipy signature `(x, y, alternative)` | keep (a sanity line; a new dependency is not justified: inferred) |
| `holm` (`verdict.py:79-88`) | `statsmodels.stats.multitest.multipletests(method="holm")` | yes but not installed; scipy has only `false_discovery_control` (BH/BY) | measured: `scipy.stats` names matching holm/bonf/fdr → `['false_discovery_control']`; hand example passes | keep (6 lines) |
| `cluster_bootstrap` (`evaluation.py:410-440`) | `scipy.stats.bootstrap` | partly: independent per-sample resampling exists, but S2's exact equality with PhAIL needs PhAIL's draw order | inferred | keep |
| `auc_censored`, `pp_curve`, `macro_*` (restated) | PhAIL `explore_pp_plot` | not importable (loader import at module level) | measured: equality 0.0 on PhAIL data | keep |
| **Three metric registries**: `evaluation.METRICS` (14), `returns.REGISTRY` (21), `verdict.METRICS` (4) | one registry | n/a | measured: UPH declared `3600 / RMST(τ_episode)` in `evaluation.METRICS` but `3600 / E[T \| T < τ_episode]` in `returns.REGISTRY` (shipped as `metrics_registry.json` with every returns run); `evaluation.uph` computes 3600/RMST; S2 measured the two forms differ up to 3× | **simplify (E2)**: one registry, consumed by S1, S6 and S7 |
| Rollout harness (`rollouts.py`: `ProbedLiberoEnv`, `run_rollouts`) | `lerobot-eval` / `rollout()` | `rollout()` reused unchanged; `eval_policy` keeps per-episode aggregates only | quoted `lerobot_eval.py:317-399`; `libero.py:173, 345-346` (`_reset_stride`); `env_wrapper.py:28` `horizon=1000` | keep |
| Trajectory store (`write_trajectories`) | `LeRobotDataset.create/add_frame/save_episode` | reused | read | keep |
| Subset stats hook (`subset_stats.py`, `lerobot_train_entry.install_subset_stats`) | LeRobot `aggregate_stats`; a `DatasetConfig` stats field | `aggregate_stats` reused; `DatasetConfig` has no stats field | measured: fields `repo_id, root, episodes, image_transforms, revision, use_imagenet_stats, …`; no stats | keep (minimal wrap of `make_train_eval_datasets`) |
| Order-preserving derive + `check_blind` (`derive.py`, `lineage.py`) | LeRobot `dataset_tools`; MLflow / DVC | no reorder or in-place replace; no leak check in tracking tools | measured `dataset_tools` defs: `delete_episodes, split_dataset, merge_datasets, modify_features, add_features, remove_feature, modify_tasks, recompute_stats, convert_image_to_video_dataset, reencode_dataset` | keep |
| `scripts/phail_fetch.py` (161 lines, stdlib S3 client) | `pos3.download(remote, local, exclude, profile)` + `pos3.Profile(local_name, endpoint, public=True)`, installed in `.venv-phail`; the phail-paper README recipe | covers `--fetch`. Not `--list` sizes: `pos3.ls` returns `list[str]` | measured signatures; quoted README `pos3.download('s3://positronic-public/phail/v1.0/dataset/', …, exclude=['*.parquet', '*.mp4'], profile=PUBLIC)` | non-blocking simplify: a one-off already executed, and replacing it now buys nothing for the PoC. Name pos3 as the reproduction path in the report |
| JSON-in-string record columns (`ops`, `pairs`, `conjunct_trace_rle`, `obj_pos`) | native parquet nested lists (pyarrow) | yes | inferred | non-blocking |
| Verify scripts as plain scripts | pytest (now installed, 9.1.1) | either works | measured | keep |

Mechanisms checked with nothing found, and the searches run:
- Per-rollout, per-metric return table with an HRT delta-method term: nothing ships one. CUPID's released code
  is binary (`success_mask`, plan P25, quoted by the planner; I did not re-open CUPID because it doesn't change
  the need for a return table).
- The LIBERO operation rule (`operations_from_trace`).
- The goal-check replay (it uses LIBERO's own recipe).

Searches run:
- WebSearch "statsmodels survdiff strata stratified logrank test python"
- WebSearch "lifelines KaplanMeierFitter infinite durations 'Infs were detected'", then WebFetch of lifelines
  `utils/__init__.py`
- WebFetch statsmodels `survdiff` docs
- WebFetch GitHub API `phail-paper/commits`
- WebFetch LeRobot `smolvla` and `libero` docs
- Local source reads: pos3, LeRobot `dataset_tools` / `configs/default.py` / `lerobot_eval.py` / `envs/libero.py`,
  hf-libero `envs/`

## Feedback: planner

1. **F-P1, restate AC7.1 (S7).**
   - PhAIL p. 9 claims "(KS rejects)" for ACT vs OpenPI/GR00T. PhAIL's released
     `out/explore/ks_power/dashboard_per_obj_macro.json` gives OpenPI–ACT `p 0.10294852573713144`, status
     `WEAK`, and GR00T–ACT `p 0.015992`, `HOLDS` (measured).
   - Our gate gives OpenPI–ACT p 0.105–0.123 across seeds and loader orders (measured).
   - New AC7.1: "OpenPI vs GR00T → (iii); OpenPI vs ACT → (iii) at this N, with the PhAIL prose-vs-artifact
     discrepancy recorded; GR00T vs ACT → (ii), the §5.3 crossing; any model vs SmolVLA → (i)."
   - Note also that PoC §3.3 p. 7's "resolved … OpenPI vs. ACT at 30 rollouts" is a detection rate from
     PhAIL §5.2 (τ = 120 s subsampling, p. 8), not a full-cohort rejection.
2. **F-P2, declare one functional per metric on libero_10, and make S6 and S7 use it (new AC).**
   - Today `returns.rollout_returns` uses whole-task `T_s` and the whole-task human anchor
     (`human_reference.parquet`, `returns.py:50-55, 107-130`). `verdict.libero_arm` / `libero_human_arm` use
     operation pairs and `human_operations.parquet` (`verdict.py:178-196`).
   - Measured on eval run `e92604a8` (libero_10:8):

     | Metric | Returns mean | Verdict functional |
     |---|---|---|
     | RMST(60) | 48.400 s | 36.986 s |
     | F(30 s) | 0.300 | 0.413 |
     | AUC | 0.085 | 0.185 |
     | HRT(60) | 0.4161 | 0.2723 |

     On libero_spatial (1 operation) they agree exactly.
   - PoC §3.4 p. 7: "Every attribution result states which functional it explains". §4.4 p. 10: "each metric's
     bootstrap CI gives the direction for that metric". The gate must verify the attributed functional.
   - Pick one per metric and record it in the registry:
     - (a) whole-task T for the per-rollout returns and for the verdict, with the operation-level functionals
       as a separately named family (`*_op`); or
     - (b) operation-level for both, with per-rollout returns defined as sums of per-operation terms, so that
       cell means equal the operation-level KM functionals.
   - PoC §3.5 A p. 7 ("for long-horizon LIBERO-10 tasks, per sub-task") leans to (b) for T. The PoC §4.3 return
     list is per rollout. Your call; state it.
   - Add an AC: "on libero_10 rollouts, mean(return) equals the verdict's metric for the same cell to ≤ 1e-9
     (or the documented weighting)".
3. **F-P3, fix the definition of `ret_rmst60_op` for censored operations (S6).**
   - The plan's `−min(T_op, 60 s)` gives a censored tail its length as if it had completed. For a tail
     `τ − t_last` the return rises toward 0 the later the policy stalls.
   - Measured on `returns-e92604a8b995/operation_returns.parquet`: `init005` op 2 never completed (event 0,
     11.60 s), yet its return −11.60 outranks the completed op-2s of `init003` (−15.50) and `init006`
     (−12.10).
   - Define the censored-operation return so that it cannot beat a completed operation. For example:
     - −τ for an incomplete operation, as `ret_rmst60` does for a censored rollout; or
     - drop censored tails from the declared operation returns and keep them as diagnostics.
   - Restate the S6 AC accordingly.
4. **F-P4, settle AC8.5's initialization.**
   - The brief withholds the 2.03 GB SmolVLM2 weights. LeRobot's SmolVLA docs prescribe fine-tuning
     `smolvla_base` (quoted above). S8 proved that path locally (500/500 tensors loaded, frozen VLM bit-equal).
   - Restate AC8.5 as fine-tuning from `lerobot/smolvla_base@d9f33c94`, and record the plan's original
     `load_vlm_weights=true` spec as a divergence. Escalate only if you believe the backbone recipe is
     required by the spec. I found no PoC page that requires it: §2.3 p. 4 says only "SmolVLA via LeRobot".
5. **F-P5, write the fixed batch plan into AC5.8 and AC7.5.**
   - Both arms and all seeds use the same `--batch-size` and the same task/init ordering.
   - Evidence: measured batch-1 vs batch-5 outcome difference; quoted LeRobot docs on paired comparisons.

## Feedback: executor

1. **E1, `attribution_grade` mislabels smoke runs (S5/S7/S9).**
   - `vlapoc/runs.py:594` sets `eval.attribution_grade = str(bool(train_run_id))`. `verify_run` labels a
     verdict "attribution-grade counterfactual verdict" when both eval runs carry that tag
     (`verdict.py:255-258`).
   - Measured, read-only from the main DB: `verify-ref-vs-drop-random-5 [eval-smoke]` →
     `verify.attribution_grade=true`; `eval-act-train-full-s0-a` and `eval-act-train-full-s0-spatial0-n10` →
     `eval.attribution_grade=true`. Both are 5-step or 2-step smoke policies on 2–10 rollouts.
   - `artifacts/s9_summary.json` has `"label": "attribution-grade counterfactual verdict"`, while the
     execution log says the S9 verdict is "labelled smoke".
   - Fix: attribution grade must be asserted by the caller. For example `eval_run(..., attribution_grade=False)`
     by default, set true only by the AC5.8 runbook commands; or derive it from the protocol (≥ 50 init states
     per task and τ 1200, from a train run that is not a smoke run).
   - Make `lineage_dry_run --eval-smoke` produce a label containing "smoke" and
     `verify.attribution_grade=false`.
   - Correct the S9 log line and the AC5.3 line ("`eval.attribution_grade=true`" reads as a positive). Record
     in the log how the existing mis-tagged runs are to be read.
   - Expected after the fix: a re-run of `--eval-smoke` shows `verify.attribution_grade=false`.
2. **E2, one metric registry (single record system).**
   - Merge `evaluation.METRICS`, `returns.REGISTRY` and `verdict.METRICS` into one source that S1, S6 and S7
     read.
   - Make the UPH functional match `evaluation.uph`. That means `3600 / RMST(τ_episode)`, as adopted in S2
     AC2.8, with Table 1's form named as a note. Today `metrics_registry.json` declares
     `3600 / E[T | T < τ_episode]`.
   - Include F-P2's per-metric T declaration once the planner states it.
3. **E3, make PhAIL's cohort export deterministic (S2).**
   - Cause, quoted: PhAIL `build/markup/loader.py` builds `candidate_urls = all_candidate_urls |
     _walk_dataset_inference_urls()` and iterates `for url in candidate_urls:` over a Python set. The order
     therefore depends on hash randomization.
   - Measured: two exports with `PYTHONHASHSEED=0` are byte-identical (`1d5e084c…`). Unseeded and other seeds
     give different row orders, with the same 990 episodes and identical contents.
   - Every bootstrap p moves while D and point estimates stay fixed: PhAIL's own OpenPI–ACT p 0.1229 → 0.1169 /
     0.1234 / 0.1089, and GR00T–ACT 0.0105 → 0.0070 / 0.0100.
   - Fix `scripts/phail_export_cohort.py` to sort rows by `s3_url` before writing. Then regenerate
     `s2_phail_reference.json`, `phail_validation.json` and the S7 artifacts, and update the logged p-values.
   - No conclusion changes (measured: (iii) / (ii) / (i) hold on a reordered cohort, OpenPI–ACT p 0.1049,
     GR00T–ACT p 0.0065). Record that p-values are Monte-Carlo draws conditional on episode order.
4. **E4, implement F-P2 and F-P3 as the planner restates them.**
   - Re-run `verify_returns_s6.py --real` on both SmolVLA eval runs with a libero_10 identity against the
     verdict's functional, and `python -m vlapoc.verdict --eval-runs …` (scratch lineage), so the two agree.
5. **E5 (small), temp dirs leak.**
   - `verdict._eval_rollouts` and `returns.load_eval_rollouts` create `data/scratch/verdict_*` / `returns_*`
     with `mkdtemp` and never delete them. Eleven such dirs were left by earlier runs (measured `ls`).
   - Use a `TemporaryDirectory`, or delete after reading.

## Non-blocking

- **Data+lineage regression not re-run.** The full dry run (`lineage_dry_run.py` without `--eval-smoke`) and
  `verify_runs_s5.py` now run with per-subset normalization and the real eval stage by default. Neither the
  executor nor I re-ran them. The refactorer must, against a scratch lineage copy.
- **`t0_model_sha256`.** `t0_state_sha256` cannot see fixture placement (measured, AC5.2b). A model-hash column
  would let the record detect the one-env-per-rollout violation. That is a schema change for the planner to
  weigh.
- **Fence enforcement.** The screening and attribution features must call `lineage.assert_blind` on their
  input version. Nothing enforces it today.
- **`scripts/phail_fetch.py`.** pos3 (installed in `.venv-phail`) does the sidecar fetch. Name it in the report
  as the reproduction route.
- **AC5.3 "byte-equal" wording.** The CLI trace output is JSON-equal, not byte-equal, to `s5_ac5_3_trace.json`
  (the formatting differs). Reword.
- **Unexplained stalls.** The 980 s stall in `hash_stored_dataset` (S0) and the 10,405 s replay wall time (S3)
  remain unexplained. Neither reproduced: my S0 scratch derivation took `hash_and_blind_check: 1.8` s, and my
  20-demo re-replay took 0.6–4.3 s per demo.
- **PoC §2.1 item 3 failure types.** Missed grasp, wrong object and ignored instruction are recorded as
  "not built" with raw signals stored. That is consistent with owner decision 6's reasoning. The report should
  carry it explicitly next to the ghost-event statement.
