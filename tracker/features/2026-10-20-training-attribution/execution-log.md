# Execution log: training and attribution

## Round 1: 2026-10-20 / Batch 1 (S1-S4)

Evidence labels: **measured** (command run in this session), **quoted**, **inferred**. All MLflow-writing checks ran
on scratch lineage stores under `data/scratch/training_attribution/`; the real `data/lineage/mlflow.db` sha256 was
`306af4123fda7e635892d459a2e34008ba1cb62f6c4fce152c546e0da7bdde79` at the start of the batch.

### S1: Route every method input through `open_for_method`
**Files:**
- `vlapoc/attribution.py` (`attribute_run`: `assert_blind` replaced by `open_for_method(subset.subset_id)`, called right
  after the train run's subset is known and before the eval/returns runs are even fetched; docstrings cite PoC §2.1
  item 4 p.3 and owner decision 3).
- `scripts/stage.py` (new `method_view()` helper; `cmd_train`, `cmd_features --which train` and `cmd_select` call it;
  `features` now takes root and episodes from the returned `MethodView`).
- `vlapoc/method_data.py` (check order only: the cheap refusals, fence/leak, ground-truth chain, diff guard, now run
  before `verify_version`'s whole-version hash; same checks, same errors, docstring updated).
- `scripts/verify_method_boundary_s3.py` (`method_sources()` now also covers `vlapoc/influence.py` and
  `vlapoc/attribution.py`).
- `scripts/verify_training_attribution.py` (new; `--blind`).

**Premises checked:**
- "`attribute_run` calls `assert_blind` only (C5)" → read `vlapoc/attribution.py` before the edit → holds. Measured
  consequence: `assert_blind('v0')` does not raise, so a subset of the clean v0 built with `op=oracle_drop` passed the old
  gate.
- "`open_for_method` hashes the whole version, about 20 s (P3)" → timed in the `--blind` run: `v3/candidate` opens in
  **2.5 s** including the integrity hash (960 episodes) → the quoted 20 s does not hold on this Mac (file cache warm).
  Acceptable per job either way.
- "`train_run` must stay unchanged because oracle arms train on `oracle_drop` subsets" → the check lives in
  `stage.py`, `train_run` untouched → holds.

**Acceptance criteria:**
- ✅ AC1.1 `attribute_run` on a train run whose subset is on fenced v1 or v2 raises `LineageError` naming the fence, run
  count unchanged: `.venv/bin/python scripts/verify_training_attribution.py --blind` → rows `attribute | fenced v1 |
  REFUSED LineageError | runs 79 -> 79` and `attribute | fenced v2 (superseded) | REFUSED LineageError | 79 -> 79`
  (message contains `fenced=True` / `fenced: superseded by v3`).
- ✅ AC1.2 same for a ground-truth-derived subset: `attribute | ground-truth-derived (oracle_drop) | REFUSED LineageError
  | 79 -> 79`, and for a `drop_random_k` descendant of it (`descendant of oracle_drop`), message contains
  `ground-truth-derived`.
- ✅ AC1.3 `features --which train`, `train`, `select (parent)` refuse the same two cases (v1, v2, oracle, oracle
  descendant: 12 rows, all `REFUSED LineageError`). Tested by calling `stage.method_view`, the helper the three commands
  use; the AST check in the same run confirms `cmd_train`, `cmd_features`, `cmd_select` each call `method_view`
  (`{'cmd_train': True, 'cmd_features': True, 'cmd_select': True, ...}`).
- ✅ AC1.4 one flipped byte in a scratch version's data file → `VersionIntegrityError` from `attribute_run` (and from the
  three stage helpers), run count `1 -> 1`; with the byte restored the same version opens (20 episodes).
- ✅ AC1.5 `.venv/bin/python scripts/verify_method_boundary_s3.py` → `all S3 checks pass: True (38s)`; boundary scan now
  over `['method_data.py', '__init__.py', 'heuristics.py', 'influence.py', 'attribution.py']` → violations `[]`.
- ✅ AC1.6 grounding docstrings: `attribute_run`, `attribution.py` module docstring, `stage.method_view` and the
  `stage.py` module docstring cite PoC §2.1 item 4 p.3 and owner decision 3 (static read).
- ✅ Real `mlflow.db` sha256 before and after the `--blind` run: `306af4123fda7e63…` both, unchanged.

**Demo artifacts produced:** `tracker/features/2026-10-20-training-attribution/artifacts/s1_refusal_table.csv` (the
refusal table: 4 entry points x {fenced v1, fenced v2, oracle subset, oracle descendant, tampered} plus two controls),
`.../artifacts/s1_blind.json`.

**Notes / divergences from plan or spec:**
- The plan's demo lists `select` as a fourth entry point but its deliverables did not name it; `cmd_select` now opens the
  parent through `method_view` too (a select on a leaky parent is a method-chain read). Recorded here as a small
  addition, not an expansion of scope.
- `open_for_method` check order changed (cheap refusals before the hash). Only the error precedence changes when an input
  is both fenced and tampered.
- `attribute_run` fetches the train run first, opens the subset, then fetches the eval and returns runs, so a blind
  refusal does not depend on those runs existing.

### S2: Make the estimator CUPID's (output function), and record every remaining divergence
**Files:**
- `vlapoc/influence.py` (`OutputCapture`, `square_objective`, `batch_features` added; `featurize(output_fn=...)` with
  `"square"` default and `"loss"` = previous behaviour; manifest has `output_fn`, scheme `vlapoc.features.v2`; meta has an
  `objective` array; CLI `--output-fn`; docstring cites CUPID App. A.1 pp. 18-19 and ATHENA App. A.1 Eq. 14-15 p. 14).
- `vlapoc/attribution.py` (`DIVERGENCES` dict, `column_plan`, `is_binary`, `output_fn_of`; `performance_influence(...,
  return_encoding="pm1")`; `check_compatible` returns `output_fns` and refuses shards mixing output functions;
  `score_table`/`attribute`/`attribute_run` take `return_encoding`; params now carry `train_output_fn`,
  `rollout_output_fn`, `return_encoding`, `center_returns`, `divergences_from_cupid`,
  `min_failure_type_rollouts_inferred`; module docstring: "1/H dropped" corrected to CUPID Prop. 2 §D.2 p. 27, and the
  centring note rewritten with the correct algebra; CLI `--cupid-returns`).
- `scripts/stage.py` (`--preset {cupid,athena,branch}` and `--output-fn` on `features`; `--cupid-returns` on `attribute`).
- `vlapoc/third_party/cupid_quality.py`, `LICENSE-cupid`, `third_party/__init__.py` (vendored `sum_of_sum_influence` and
  `compute_demo_quality_scores`, bodies verbatim from `agiachris/cupid` @ `2941eba9`, lines 17-19 and 299-335, MIT).
- `scripts/verify_attribution.py` (checks 1b, 6, 7, 8 and `--cpu-smoke`).

**Premises checked:**
- P1 "`action_out_proj` runs once per training forward" → `OutputCapture.pop()` raises unless the module ran exactly once;
  the real-checkpoint smoke (13 samples x K=4 x 3 calls, MPS) never raised → holds. The first toy run also showed the guard
  firing when a test called the module extra times (it counted 17 calls), which is what it is for.
- P2 "`KroneckerProjector` assumes batch is dim 0 of every hooked Linear input" → real checkpoint: `117 layers, dim 7488`,
  `collect()` found a gradient for every layer, finite features → holds on CPU/MPS, not only the cluster.
- P5 "numpy matmul RuntimeWarnings on this Mac may hide an overflow" → new check 1b: `np.isfinite` on the brute-force H, its
  inverse and both matrices → all finite, `cond(H) = 2.0` → the warnings are spurious (BLAS flags; inferred cause), not overflow.
- CUPID's `ℓ_square` and `net` score (plan C4) → read `modelout_functions.py` l.116-119 (`pred**2` over unmasked entries) and
  `error_util.py` l.299-335 (`net` = Σ_succ − Σ_fail) at `2941eba9` → hold as the plan states.
- "centred = CUPID contrast scaled by p(1−p)" (branch docstring) → derived and checked numerically (7b, 7c): true for a
  difference of *means* only; CUPID's `net` is a difference of *sums* → the branch's note was wrong, corrected.

**Acceptance criteria:**
- ✅ AC2.1 `.venv/bin/python scripts/verify_attribution.py` → `[PASS] 6a toy module: 'square' features == brute-force
  projected per-sample gradient (masked, K=2, B=5)  max per-sample rel err 8.58e-08` and the same for `'loss'`
  (`1.18e-07`), tolerance 1e-5; `6b square_objective == the definition ... max abs diff 0.0e+00`.
- ✅ AC2.2 `[PASS] 7a CUPID parity: performance_influence(M, ±1, center=False)·m == compute_demo_quality_scores(.., 'net')
  (vendored 2941eba9)  max abs err 2.84e-14` (tolerance 1e-9). M is built from random pair scores by group sums, CUPID's
  side from the vendored `sum_of_sum_influence` per (rollout, episode) block; both `return_encoding="pm1"` and an explicit
  2·r01−1 are compared.
- ✅ AC2.3 `[PASS] 7b ... p = 1/2: centred = p(1-p)·m·(ms-mf), net = m·[p·ms-(1-p)·mf], same ranking  p=0.5, identity err
  2.7e-15, Kendall tau 1.000` and `[PASS] 7c ... at p = 0.3 the two rankings differ (counterexample)  p=0.3, identity err
  3.6e-15, Kendall tau 0.667`.
- ✅ AC2.4 `.venv/bin/python scripts/verify_attribution.py --cpu-smoke data/runs/train-smolvla-2step-persubset-v0ref-
  20261005T012854Z/lerobot/checkpoints/000002/pretrained_model data/lerobot/libero_spatial_10 --device mps` → episodes
  `[31, 41]` of `v0/reference` (opened through `open_for_method`), stride 20, 13 samples, `shape (13, 7488)`, `n_layers 117`,
  manifest `output_fn` = square / loss, all finite, two `square` calls bit-for-bit equal (`max abs diff 0.0e+00`), wall time
  307.3 s / 301.0 s / 302.7 s per call on MPS including the model load (the three calls are the first, the `loss` one, and
  the repeat). Cosine between `square` and `loss` features of the same samples: min 0.402, median 0.870, max 0.953;
  median norm ratio 0.976. So the output-function choice changes the direction of the projected gradient noticeably for
  some samples (cos 0.40) and little for others.
- ✅ AC2.5 (two parts). `8a`/`8b` pass: `check_compatible` accepts one output function per side (ATHENA's mixed
  loss/square, and legacy manifests without the key as `loss`) and refuses `['loss', 'square']` shards within one side
  (`manifest list 0: shards mix output functions ['loss', 'square']`); `8c` `DIVERGENCES` has every key the plan lists.
  **The `attribute_run` *params* part (the run actually carrying `train_output_fn`, `rollout_output_fn`,
  `return_encoding`, `center_returns`, `DIVERGENCES`) needs a tracked run; it is checked in S3's synthetic integration
  fixture** (S3 AC2.5 line below: ✅, added when S3 landed; at the S2 commit this half was by static read only).
- ✅ AC2.6 `.venv/bin/python scripts/verify_attribution.py` → `pass: True (29/29)`; the original checks 1-5 account for 16 of
  them (all PASS), 13 are new (1b, 6a x2, 6b, 6c, 7a-7c, 7d x2, 8a-8c).

**Demo artifacts produced:** `artifacts/s2_verify_attribution.json` (29 checks), `artifacts/s2_cpu_smoke_mps.json` (per-sample
cosines, objectives, timings). The divergence table is `vlapoc.attribution.DIVERGENCES` (9 entries, each with its source).

**Notes / divergences from plan or spec:**
- Default output function is now `square` on both sides (non-blocking default N1 kept by the owner). **This changes what the
  next server run computes**: features regenerated with `--preset cupid` are not comparable with the October `loss/loss`
  features (`--preset branch` reproduces them).
- `rank`/`K`/strides are unchanged (S2 records them as divergences, it does not change them).
- `DIVERGENCES` adds two entries beyond the plan's list, `continuous_returns` (pm1 applies to binary columns only) and
  `model_family` (SmolVLA flow matching vs CUPID's diffusion ε-prediction), and a `failure_type_returns` entry: PoC §9.1
  Step 3 says "only rollouts with a specific failure type", which this code realises as an indicator return over all
  rollouts (C2 in the plan), a standing interpretation, now disclosed in every run's params.
- `return_encoding="pm1"` is applied to *binary* return columns only (`ret_f30`, `ret_censored60`, failure-type
  indicators); continuous returns (RMST, AUC, HRT) stay raw and are centred iff `center`. CUPID has only binary returns.
- Square-objective averaging follows SmolVLA's own loss reduction (valid steps x action dims); CUPID's code averages over all
  entries including masked ones. Equal when no step is padded (documented in `square_objective`).

### S3: Declare the functional of every column, and close the attribute run's provenance gaps
**Files:**
- `vlapoc/attribution.py` (`returns_of`, `score_columns`, `n_models_per_column`, `column_declarations`, `declaration_of`,
  `feature_file_digests`, `_artifact_json`; `attribute_run` records `functionals.json`, one `features` input per feature
  `.npy`/`.meta.npz`, `allow_partial` and `n_rollouts_without_trajectory`, `n_feature_files`; `ensemble_run` refuses
  duplicates, ensembles-of-ensembles, differing `damping`/`center_returns`/`return_encoding`/output functions, inputs
  without declarations and inputs that declare a column differently, and records `n_models_per_column` and its own
  `functionals.json`; `zscore_ensemble` now takes the *union* of columns; CLI `--allow-partial`).
- `vlapoc/runs.py` (`select_run(..., require_declaration=False)`: ranking ops record param `metric_declaration`).
- `vlapoc/influence.py` (manifest `checkpoint` and `dataset_root` via `lineage.rel()`).
- `scripts/stage.py` (`attribute --allow-partial`; `select` passes `require_declaration` for ranking ops).
- `scripts/verify_training_attribution.py` (`--functionals`; the synthetic fixture `make_fixture`/`add_eval`/`run_attribute`,
  reused by S4 and later).

**Premises checked:**
- "failure-type columns can exist in only some seeds" (plan S3) → true, and worse: `zscore_ensemble` built its column list
  from `tables[0]` only, so a failure-type column that table 0 lacked was **silently dropped** from the ensemble. Measured in
  the new check 3.5c (a `drop` failure type present in one seed's returns): fixed by taking the union, and `n_models` records
  it (`columns with n_models=1: ['harm_ft_drop', 'harm_ft_drop__same_task', 'infl_ft_drop', 'infl_ft_drop__same_task']`).
- "rollouts that lack trajectories are dropped silently (C15 e)" → read `attribute()`: `rollouts.dropna(subset=
  ["traj_episode_index"])` with no count → holds; now refused without `allow_partial`.
- "`MIN_FAILURE_TYPE_ROLLOUTS` is absent from params" → `grep` of the old `attribute_run` params → holds; now logged as
  `min_failure_type_rollouts_inferred` (done with S2's params block).

**Acceptance criteria:**
- ✅ AC3.1 `.venv/bin/python scripts/verify_training_attribution.py --functionals` → `[PASS] 3.1 every infl_/harm_ column has
  exactly one functionals row, and its tau is the metrics_registry tau  32 columns, 32 rows; tau_s by return: {'ret_f30':
  30.0, 'ret_rmst60': 60.0, 'ret_auc_expected': None, 'ret_auc_sampled': None, 'ret_hrt60_lin': 60.0, 'ret_censored60': 60.0,
  'ft_drop': 60.0, 'ft_timeout': 60.0}`; 3.1b (declared encoding/centring equals what `column_plan` applied), 3.1c (sign and
  subset), 3.1d (a real attribute run's `functionals.json` has one row per column: 28 rows) also PASS.
- ✅ AC3.2 `--manifest <a manifest written by a new featurize call on the real SmolVLA checkpoint, MPS, 3 samples>` →
  `{'checkpoint': 'data/runs/train-smolvla-2step-persubset-v0ref-20261005T012854Z/lerobot/checkpoints/000002/pretrained_model',
  'dataset_root': 'data/lerobot/libero_spatial_10'}`: no leading `/`, both exist under the repo root.
- ✅ AC3.3 `[PASS] 3.3 ... 6 features inputs for 6 files: ['features.rollouts.meta.npz', 'features.rollouts.npy',
  'features.train.shard0.meta.npz', 'features.train.shard0.npy', 'features.train.shard1.meta.npz', 'features.train.shard1.npy']`
  and the digests equal `sha256_file` of each file; 3.3b: flipping one byte of `train.shard1.npy` changes exactly
  `['input.features.train.shard1.npy']`. Run on a scratch copy of the real store with synthetic features (the real SmolVLA
  chain is S8).
- ✅ AC3.4 `[PASS] 3.4 attribute raises when one rollout lacks a trajectory  1 of 30 rollouts ... have no recorded trajectory`
  and `[PASS] 3.4b --allow-partial passes and records n_rollouts_without_trajectory  n_rollouts 29 of 30, recorded 1`.
- ✅ AC3.5 `3.5` `ensemble inputs contain duplicate runs`, `3.5b` `ensemble inputs differ in 'center_returns': ['False', 'True']`,
  `3.5c` n_models per column recorded in params and in the ensemble's `functionals.json`.
- ✅ AC3.6 `[PASS] 3.6 the select run's params carry the ranked column's declaration  harm_f30__same_task: F(30) = P(T <= 30 s)
  | tau 30 s | subset same_task | good_sign 1.0`; 3.6b refuses (with `require_declaration`) a score run without
  `functionals.json`; 3.6c refuses a column the score run does not declare.
- ✅ AC2.5 (params half, deferred from S2): `[PASS] 2.5 attribute run params carry output functions, encoding, centring and
  DIVERGENCES  {'train_output_fn': 'square', 'rollout_output_fn': 'square', 'return_encoding': 'raw', 'center_returns': 'True',
  'min_failure_type_rollouts_inferred': '5'}, divergences keys [centred_returns, continuous_returns, damping, draws,
  failure_type_returns, model_family, numerics, projection, strides]`.
- ✅ Real `mlflow.db` sha256 before and after the run: `306af4123fda7e63…` both. Total: 17/17 checks.

**Demo artifacts produced:** `artifacts/s3_functionals_example.json` (one line per attribution column stating which functional
it explains; **SYNTHETIC** fixture data, labelled in the file), `artifacts/s3_ensemble_functionals_example.json` (with
`n_models`), `artifacts/s3_manifest_example.json`, `artifacts/s3_functionals.json` (the 17 checks).

**Notes / divergences from plan or spec:**
- `rollout_subset` takes the values `all` / `same_task` only. The plan listed `failure_type=<t>` as a third value; that would
  misdescribe the columns: a failure-type column is the indicator return 1[failure_type = t] over *all* (or the same-task)
  rollouts (plan C2). The indicator is stated in `return_definition` instead.
- `select_run` records `"undeclared"` rather than failing for score runs without declarations (the stand-in attribute runs that
  `verify_runs_s5.py` and `lineage_dry_run.py` use); only the attribution chain (`stage.py select`) passes
  `require_declaration`, which refuses.
- A run produced before S3 has no `functionals.json`, so `ensemble_run` refuses it with a message to re-run the attribute stage.
  October's cluster attribute runs are not in this repo's MLflow store, so nothing local is affected.
- `n_feature_files` and `allow_partial` are also logged as params.

### S4: Complete Step 4 (helpful arms, curated-vs-random verdicts, opposite curations, curation switch, seed summary)
**Files:**
- `scripts/server_pipeline.py` (arms: `help-*` (the k most helpful, `--direction lowest`), `hrt60_lin` in the default metrics,
  `--curation {per_task,global,both}` (default `both`, owner answer Q1) with global arms `gharm-/ghelp-/grandom<r>-n<N>`
  where N = k x `--n-tasks` (20) = 100; per-arm `controls`; verify jobs `curated_vs_random` (each harm/help arm vs each
  random control of its family, family W) and `head_to_head` (harm-hrt60_lin as B vs harm-f30 as A); every verify is tagged;
  a final `step4-summary` job depending on all verifies; docstring rewritten with the PoC sentence each arm tests).
- `vlapoc/step4.py` (new: `collect`, `summarize`, `summary_run`, `call`, `expected_direction`; stage `summarize`).
- `scripts/stage.py` (`verify --comparison/--treated/--control/--seed-id`; `step4-summary`; `select --headline
  --pipeline-curation`; `select_params()`).
- `vlapoc/verdict.py` (`verify_run(extra_tags=)`), `vlapoc/runs.py` (`select_run(extra_params=)`, stage `summarize`).
- `scripts/verify_training_attribution.py` (`--pipeline`, `--step4`).

**Premises checked:**
- "the default dry-run gives 74 jobs and arms harm-{f30,rmst60,auc_expected}-k5, random{0,1}-k5, drop-random-20" (plan C8) → ran
  the unchanged pipeline at S3's commit: `would submit 74 jobs` (planner's measure, `data/scratch/dryrun.txt`) → holds; after S4
  the default dry run is 312 jobs.
- "`__same_task` columns are on per-task scales and cannot be ranked globally" (plan S4, inferred) → the global arms rank the
  all-rollout `harm_<m>` column instead, and are size-matched to the per-task arms (N = 5 x 20 = 100, so 860 of 960 kept in
  both families). Not a measured claim about the scales; it is a design reason, kept as inferred.
- "v0/candidate has 20 tasks x 48 episodes" → measured in 4.5: `20 tasks, counts [48]` → holds.

**Acceptance criteria:**
- ✅ AC4.1 `.venv/bin/python scripts/verify_training_attribution.py --pipeline` (parses `server_pipeline.py --tag t --dry-run`) →
  `[PASS] 4.1 every harm and help arm has a verify job vs each random control of its family, per seed, depending on exactly both
  arms' eval jobs  96 jobs checked, 8 help arms, mismatches []`; `4.1b` both `harm-f30-k5` and `help-f30-k5` have their
  `verify-t-<arm>-vs-random0-k5-s{0,1,2}` jobs.
- ✅ AC4.2 `[PASS] 4.2 a harm-hrt60_lin arm and per seed a head-to-head verify (hrt60_lin arm B vs f30 arm A), depending on both evals`.
- ✅ AC4.3 `4.3` (global: `8 drop_top_k arms (k ['100']), 2 random controls (k ['100'])`, metrics `harm_<m>` without `__same_task`);
  `4.3b` per_task only per-task ops (plus AC7.5's k=20 gate); `4.3c` `--curation both` emits `{'drop_random_k': 3,
  'drop_random_k_per_task': 2, 'drop_top_k': 8, 'drop_top_k_per_task': 8}` (21 selects), is identical to the default, carries
  `--pipeline-curation both` on each, `--headline` only on per-task arms. The default follows the owner's answer (Q1: both,
  per-task headline); 4.5b shows the select run's params record it.
- ✅ AC4.4 `.venv/bin/python scripts/verify_training_attribution.py --step4` (SYNTHETIC verdicts, labelled; 12 verify runs in a
  scratch store, 3 seeds): `4.4b ... T,T,0 | treated better in 2/3 seeds, control better in 0/3; expectation (treated_better) met
  in 2/3`; `4.4c` helpful removal `C,C,0`; `4.4d` head-to-head `hrt60 T,T,T` and `f30 C,C,C`; `4.4e` each row carries the declared
  functional; `4.4g` the summary is a tracked `summarize` run. Chain refusal: `4.4 ... tag 't' matches verify runs of several
  chains ['t-aaaa1111', 't-bbbb2222']; pass the chain id to summarize one`; with a chain id only that chain's verdicts are used
  (4.4f).
- ✅ AC4.5 `4.5 every per-task arm keeps exactly 43 of 48 episodes in each of the 20 tasks (860 total); global arms keep 860 too
  (960 - 100)  21 arms: [('ac75', 940), ('global', 860), ('per_task', 860)]`, run through `select_run` with the arms'
  parsed arguments on a scratch copy of the real store with a **synthetic** score table; `4.5b` the select run records
  `curation=per_task headline=True pipeline_curation=both`.
- ✅ AC4.6 default dry run: **312 jobs** (before S4: 74): `{'train': 66, 'eval': 66, 'returns': 3, 'feat': 6, 'attribute': 3,
  'ensemble': 1, 'select': 1, 'verify': 165, 'step4': 1}`. 21 arms, 63 retraining runs (21 arms x 3 seeds):
  per-task `harm-/help-{f30,rmst60,auc_expected,hrt60_lin}-k5` (8), `random0-k5`, `random1-k5`; global `gharm-/ghelp-...-n100`
  (8), `grandom0-n100`, `grandom1-n100`; `drop-random-20`. The full list is in `artifacts/s4_pipeline.json` and the parsed
  dry run in `artifacts/s4_dry_run_default.txt`.
- ✅ Real `mlflow.db` sha256 before and after `--step4`: `306af4123fda7e63…` both.

**Demo artifacts produced:** `artifacts/s4_dry_run_default.txt` (every job with its dependencies and command),
`artifacts/s4_pipeline.json`, `artifacts/s4_step4_summary_SYNTHETIC.csv` (the table layout, from planted verdicts),
`artifacts/s4_select_counts_SYNTHETIC_SCORES.csv` (episodes kept per arm on the real v0/candidate manifests),
`artifacts/s4_summary_and_selects.json`.

**Notes / divergences from plan or spec:**
- **The server's next run is much bigger**: 312 jobs and 63 retraining runs (it was 74 jobs and 18 retrains: 6 arms x 3
  seeds), because of `--curation both` (owner accepted, Q1), the help arms and `hrt60_lin` (non-blocking N3). `--curation
  per_task` plus `--metrics f30 rmst60 auc_expected` reproduces a smaller chain.
- `curated_vs_random` compares each treated arm against **every** random draw of its family (2 draws), not a single
  "matched" one: one random draw would make the control's own variability invisible. Each is a separate row in the summary.
- The `vs_base` verifies keep both families W and O (as before); the new comparisons use family W only (the declared family).
- The per-metric call (`T`/`C`/`0`) uses the verdict's own Holm-corrected p and α = 0.05; `step4-summary` aggregates and does
  not re-test (the two-step verdict is `vlapoc.verdict`'s). Expected directions are written in `vlapoc/step4.py` before any real
  verdict exists (harm removed: treated better; help removed: treated worse; HRT arm wins hrt60, F30 arm wins f30).
- The headline flag is set only on per-task arms and only when `--curation both`; with a single mode nothing is flagged.
- `docs/dataflow.md` was not changed (no stage was added to the diagram; `summarize` is bookkeeping after Step 4).

### Batch 1 regression run (after S4's commit, clean tree)
`verify_runs_s5.py --r2` rc=0 (3 s), `verify_verdict_s7.py` rc=0 (1 s), `verify_lineage_s3.py` rc=0 (66 s),
`verify_consumers_s6.py` rc=0 (30 s), `verify_method_boundary_s3.py` rc=0 (40 s), `verify_blind_s0.py` rc=0 (31 s). Also
`verify_attribution.py` 29/29 and the S1/S3/S4 verifiers above. Real `data/lineage/mlflow.db` sha256 `306af4123fda7e63…` unchanged.
Tracked artifacts these verifiers rewrote, diffed and restored with `git checkout` (not committed):
- `2026-09-28-data-lineage-layer/artifacts/s5_r2_summary.json`, `2026-10-01-evaluation-layer/artifacts/s0_verify_blind.json`,
  `2026-10-15-ground-truth/artifacts/s6_v3_verify.json`: only run ids and timings differ.
- `2026-09-28-data-lineage-layer/artifacts/s3_summary.json`: **structural** difference (new keys such as `isolated_cache_gb`,
  `AC3.8`, `files_scanned` 15 -> 193, a `names` block removed). Not caused by this batch: `verify_lineage_s3.py` was last
  changed by earlier features (`84e166a`, `91f0902`), the planner measured the same rewrite before any change here
  (plan risk P4), and the script exits 0. The tracked file is stale relative to its own verifier; restored, not committed.

## Round 1 / Batch 2 (S5–S8): 2026-10-20

Real `data/lineage/mlflow.db` sha256 at the start of batch 2: `306af4123fda7e635892d459a2e34008ba1cb62f6c4fce152c546e0da7bdde79`.

### S5: Score attribution against the planted ground truth (recovery per corruption type x metric)
**Files:** `vlapoc/recovery.py` (new; ground-truth side: `EXPECTED_METRICS`, `recovery_matrix`, `recovery_run`), `scripts/stage.py`
(`score` subcommand), `scripts/server_pipeline.py` (`has_answer_key`, a `score` job after the ensemble when the base subset's version
has labels; docstring recommends `--subset v3/candidate` for the injection run), `scripts/verify_training_attribution.py`
(`--recovery`).

**Order of work (as the coordinator asked):** the PoC §4.5 type-to-metric hypotheses were committed alone in `b4790ee`
(`vlapoc/recovery.py` containing only `EXPECTED_METRICS` and its grounding), before any scoring code or any score table existed;
check 5.3b reads that commit back from git.

**Premises checked:**
- "the PoC states the expected metric per type" → quoted PoC §4.5 p.11: only two mappings are named ("idle or hesitant segments aimed at
  time-based metrics", "noisy gripper release near placement aimed at drops"). The plan's map "idle_padding → rmst60/hrt60_lin;
  gripper_release → f30/censored60" holds; the other five types have no PoC mapping, so none was invented (`None`).
- "`groundtruth.score_run` exists but is not wired" → it scores one column per run; for a whole table it would create one MLflow run per
  column (28+), so `recovery_run` wraps `groundtruth.score_recovery` and records one `score` run per table instead (a design choice).
- Heuristic numbers: `s5_v3_r3_bands.csv` best-heuristic names all exist in `s5_v3_r3_recovery_heuristics.csv` (7/7 looked up, incl.
  `length_vs_task_lowest`) → holds.

**Acceptance criteria** (all on SYNTHETIC planted score tables over the real v3 answer key; nothing here is a result about any policy):
- ✅ AC5.1 `.venv/bin/python scripts/verify_training_attribution.py --recovery` → `idle_padding AUROC 1.0 in harm_rmst60 and
  gripper_release 1.0 in harm_f30  1.0, 1.0`; a random column's pooled AUROC `0.527` (band [0.4, 0.6] is the plan's inferred band for 91 of
  960; per type, with 13 positives, the random AUROC's s.d. is about 0.08, so the band is applied to the pooled value only); real db unchanged.
- ✅ AC5.2 `48 rows = 8 x 6 columns` (7 types + pooled); `best_heuristic_auroc` equals the bands CSV `7/7`.
- ✅ AC5.3 `matches_hypothesis` True for idle_padding (best `harm_rmst60`) and gripper_release (best `harm_f30`), `None` for the five
  types without a PoC mapping; 5.3b hypotheses committed first (`b4790ee`); 5.3c run records `synthetic=True`.
- ✅ AC5.4 dry run `--subset v3/candidate`: one job `score-t` with dependency `ensemble-t`
  (`python scripts/stage.py --chain t-... score --scores attribute-t-ens --version v3 --subset candidate --name score-t`); with the default
  `v0/candidate` no score job.
- ✅ AC5.5 a table keyed to v0/candidate is refused: `score table keys != parent subset's episode versions: 91 unknown/stale ...`.
- ✅ AC5.6 `scan_boundary` clean on `influence.py`/`attribution.py`; no mention of `vlapoc.recovery` there; the only import of the scoring
  module is inside `stage.py`'s `cmd_score` (0 top-level imports).
- ✅ real `mlflow.db` sha256 unchanged (`306af4123fda7e63…` before and after).

**Demo artifacts:** `artifacts/s5_recovery.json` (11 checks). The recovery table layout (type x harm column, AUROC and recall@13 beside the
best heuristic) is `recovery_matrix.csv` as written by the check; it is deleted with the scratch store, because synthetic numbers must not
be kept as if they were results.

**Notes / divergences:** recall@k uses k = number injected per type (13), as in the ground-truth feature. `recovery_run` skips (and names)
harm columns containing NaN, since `check_score_keys` refuses NaN scores.

### S6: Make the chain testable and portable (`--lineage-root`, chain-scoped names, site config, scratch dirs)
**Files:** `scripts/stage.py` (global `--lineage-root` and `--latest`; `run_id(ref, stage, chain_prefix, latest, lineage_root)`;
`chain_prefix_of`; every tracked-stage call passes `lineage_root=LR`), `slurm/site.env.example` (new, tracked),
`docs/site-profile-cluster.env` (new: the owner's cluster values, kept as a documented profile), `slurm/env.sh` (sources
`site.env`, fails naming `site.env.example`), `slurm/gpu.sbatch`/`cpu.sbatch` (no absolute path, no `#SBATCH --partition/--output`),
`scripts/server_pipeline.py` (`load_site`, `--site`, `--partition` override, `--output=<LOG_DIR>/...` and `--partition=<PARTITION>`
on every sbatch, the dry run prints the full sbatch argv), `.gitignore` (`slurm/site.env`), `scripts/verify_attribution.py` (cluster
paths became `--old-python/--old-src/--old-hf-home`, temp dirs under `data/scratch/`), `scripts/verify_training_attribution.py`
(`--names`).

**Premises checked:** "`#SBATCH --output` cannot expand variables" → true of sbatch (directive lines are not shell-expanded; inferred
from the sbatch manual, not re-read here), so the pipeline passes the flags on the command line. "`env.sh` can read the site file when
run inside a spooled sbatch script" → the pipeline exports `PHYEXP_DIR` (already did), so `env.sh` does not rely on its own location;
`SITE_ENV` is exported when `--site` is given.

**Acceptance criteria** (`.venv/bin/python scripts/verify_training_attribution.py --names`, 12/12):
- ✅ AC6.1 `git grep -n -E '/raid|/auto/home' -- slurm scripts vlapoc` → no output (rc=1), also checked inside the verifier.
- ✅ AC6.2 `bash -n` passes on the 4 slurm files; `env.sh` without a site file: `rc=2: [env.sh] missing .../does_not_exist.env: copy
  slurm/site.env.example to slurm/site.env and edit it ...` (run in a subshell).
- ✅ AC6.3 scratch store with `train-t-s0` in chains `t-aaaa1111`, `u-bbbb2222`, `t-cccc3333`: scope `u-` → the `u-bbbb2222` run;
  scope `t-` raises `2 FINISHED train runs named 'train-t-s0' in chains ['t-aaaa1111', 't-cccc3333']; pass --latest ...`; `--latest` →
  the `t-cccc3333` run; with no chain (hand use) the most recent wins as before.
- ✅ AC6.4 (first half) static: global `--lineage-root`; no `client()` without it in `stage.py`; every tracked-stage call
  (`train_run … summary_run`) passes `lineage_root`. The second half, the whole chain on a scratch store, is S8.
- ✅ AC6.5 the dry run prints 59 sbatch lines (per_task, 1 seed), each with `--partition=gpu` and
  `--output=/path/to/scratch/phyexp/logs/%x-%j.out` (`%x-%A_%a.out` for array jobs), the values of `site.env.example`.
- ✅ AC6.6 `TMPDIR` set to an empty dir under `data/scratch/`: `verify_attribution.py` → `pass: True (29/29)`, directory still empty,
  `tempfile.TemporaryDirectory()` without `dir=` no longer in the file.
- ✅ real `mlflow.db` sha256 unchanged.

**Notes:** the dry run now needs a site file; without `slurm/site.env` it falls back to `site.env.example` (dry run only), a real
submission exits with an instruction. The owner's cluster values are `docs/site-profile-cluster.env` (documented, outside the
grep scope on purpose: the AC asks for no cluster path in code).

#### Owner-visible cost of the Step 4 chain (inferred; flags that shrink it)
Counts are measured from `server_pipeline.py --tag t --dry-run <flags>` (jobs, and train jobs = base trains + one train per arm and
seed). GPU-hours use the only figure in the repo, runbook §AC8.5: "≈ 4 h per seed on one A100 (quoted, LeRobot SmolVLA docs, 20k
steps); unmeasured on our server". Eval, feature and attribute GPU time are **not derivable** from anything recorded, so they are not
included; training alone is a lower bound.

| Configuration | Arms | Jobs | Train jobs | Training GPU-h (4 h each, inferred) |
|---|---|---|---|---|
| Branch before S4 (6 arms: 3 harm, 2 random, AC7.5 gate; 3 seeds) | 6 | 74 | 21 | about 84 |
| **Default now** (`--curation both`, 4 metrics, 3 seeds, 2 random draws, AC7.5) | 21 | **312** | **66** | about 264 |
| `--curation per_task` | 11 | 171 | 36 | about 144 |
| `--curation per_task --metrics f30 hrt60_lin` (keeps the §4.5 head-to-head) | 7 | 111 | 24 | about 96 |
| the same with `--seeds 0 1` | 7 | 75 | 16 | about 64 |
| the same with `--seeds 0 1 --random-draws 1 --no-ac75` | 5 | 55 | 12 | about 48 |
| `--curation per_task --metrics f30 rmst60 auc_expected --no-ac75` (the old metric set plus help arms) | 8 | 129 | 27 | about 108 |

Flags that shrink it: `--curation per_task|global` (halves the arms), `--metrics` (each metric adds a harm and a help arm per mode),
`--seeds` (the PoC §2.3 allows 2–3), `--random-draws`, `--no-ac75`, `--k-per-task` (each value multiplies arms); `--from select` reuses a
finished base. The 63-versus-18 retrain count comes from `--curation both` (owner Q1), the help arms (PoC §9.1 Step 4.5) and
`hrt60_lin` (non-blocking default N3).

### S7: Metric-resolved outputs (four groups, heatmap, throughput-vs-reliability scatter, aggregation by catalog column)
**Files:** `vlapoc/analysis.py` (new, method side: `metric_groups`, `aggregate_by`, `plot_heatmap`, `plot_tradeoff`, `axis_label`,
`heatmap_text`/`tradeoff_text`, `analyze_run`), `scripts/stage.py` (`analyze`), `scripts/server_pipeline.py` (an `analyze` job after the
ensemble), `scripts/verify_training_attribution.py` (`--groups`).

**Divergences / decisions:**
- The plan placed `metric_groups`, `aggregate_by` and the plots in `attribution.py`; they live in `vlapoc/analysis.py` instead, so the
  estimator module stays about estimation (same grounding, same boundary scan).
- Group labels are **descriptive and unverified** (PoC §4.4 asks for Holm before calling an episode a trade-off; per-episode tests are not
  available from 2-3 seeds). The threshold |z| >= 1 is an **inferred**, declared parameter. Both are in the analyze run's params and on
  every figure title; the verified claim stays at intervention level (`vlapoc.step4`).
- The analyze run is recorded under stage `summarize`; its outputs include `figure_labels.json` (the exact titles and axis labels drawn).
- Adding the `analyze` job makes the default dry run 313 jobs (S4 and S6 logged 312 before it existed).

**Acceptance criteria** (`.venv/bin/python scripts/verify_training_attribution.py --groups`; SYNTHETIC inputs, 9/9):
- ✅ AC7.1 planted table (240 episodes, 5 per group): `20/20 planted correct; counts {'neutral': 220, 'consistent_helper': 5,
  'consistent_harmer': 5, 'trade_off': 5, 'metric_specific': 5}`; the 220 noise episodes are all neutral.
- ✅ AC7.2 `aggregate_by(task_name)`: 6 rows, n sums to 240, each mean inside its bootstrap CI; refuses `not_a_column` and
  `episode_id`.
- ✅ AC7.3 heatmap and scatter PNGs written (55-99 kB); the axis labels name the declared functional and τ, e.g. `harm_rmst60: RMST(60) =
  ∫_0^60 (1 - F(t)) dt [τ 60 s = τ_episode, all, good_sign +1]` and `harm_f30: F(30) = P(T <= 30 s) [τ 30 s, all, good_sign +1]`; both titles
  start `SYNTHETIC DATA — not a result about any policy`. Run on a fixture attribute run on a scratch store.
- ✅ AC7.4 analyze run params: `group_z_threshold 1.0`, `group_status descriptive, unverified ...`, `score_run_id` = the attribute run,
  `synthetic True`.
- ✅ boundary scan clean on `vlapoc/analysis.py`; real `mlflow.db` sha256 unchanged.

**Demo artifacts (all labelled SYNTHETIC):** `artifacts/s7_heatmap_planted_SYNTHETIC.png`, `artifacts/s7_tradeoff_planted_SYNTHETIC.png` (planted
240-episode table; viewed after rendering: legend and group labels do not overlap the data), `artifacts/s7_heatmap_SYNTHETIC.png`,
`artifacts/s7_tradeoff_SYNTHETIC.png` (15-episode fixture run through the stage), `artifacts/s7_figure_labels_SYNTHETIC.json`,
`artifacts/s7_groups.json`.

### S8: Run the chain end to end on this Mac (synthetic integration and a real toy-scale smoke)
**Files:** `scripts/verify_training_attribution.py` (`--integration`, `--local-smoke`; the fixture `make_fixture`/`make_train`/`add_eval`/
`run_attribute`/`pick_covering`), `scripts/stage.py` (`--feats-root`, `--synthetic` on `score`/`analyze`), `vlapoc/runs.py`
(`select_run`: validates the key contract and k's bounds before any run exists, and links the score run's own output entity).

**Defects found by running the chain (fixed here):**
1. `trace` from a select run dead-ended at the score table: `select_run` consumed the table's *column digest*, while the ensemble run
   *produced* the file's sha256, so no edge connected them. `select_run` now also records `input.score_run_table` = the score run's
   `output.attribution_scores.sha256`; `trace` then reaches the ensemble, both attribute runs, returns, eval and train runs.
2. A select refused for a stale score table or an emptying k still left a FAILED MLflow run (the check ran inside the run). The key contract
   and k bounds are now checked before `stage_run`, so a refusal writes nothing (AC8.4).
3. S3's check 3.6c expected `ValueError` from `declaration_of` for an unknown column; after (2) the key check refuses first with
   `LineageError` ("score table lacks column"). The check now asserts a refusal with no run created, whichever check fires.

**Acceptance criteria:**
- ✅ AC8.1 (a) `.venv/bin/python scripts/verify_training_attribution.py --integration` (SYNTHETIC features, rollouts and returns over a real
  v3 training subset of 14 episodes in 2 tasks, 10 of them injected, 2 seeds; scratch copy of the store; every stage through
  `stage.py --lineage-root`): 2 x attribute, ensemble, 5 selects (per-task harm / help / random with `--headline`, global harm / random),
  score (v3), analyze, all 10 stage runs succeeded (`attribute table 14 episodes x 12 rollouts`; per-task arms keep 12 of 14, global arms 12).
  `trace` from the harm select run: stages `['attribute', 'eval', 'returns', 'select', 'train']`, 10 runs including both seeds' returns, eval
  and train runs, **12 `features` inputs (2 seeds x 6 files)**, the training subset in the entities. Dead ends (inputs with no producing run,
  by design): `features` (intermediate files), `subset` (the training subset's own producer is not in the fixture) and `score_table` (the
  select's column digest).
- ✅ AC8.2 four negatives, each refused with the run count unchanged: features from another checkpoint (`features ... were computed from another
  checkpoint`), train features on a different episode list (`train features were not computed on exactly the training subset's episodes`),
  a score table from another subset at select (`score table keys != parent subset's episode versions ...`), an ensemble of two runs on different
  subsets (`ensemble inputs were trained on different subsets`).
- ✅ AC6.4 second half: the whole chain above ran on the scratch store; real db untouched.
- ✅ AC8.3 (b) `.venv/bin/python scripts/verify_training_attribution.py --local-smoke --device mps` completed. Tracked train run `0a33bc75bb…`
  (2-step SmolVLA on v0/reference), through the stage CLI on a scratch copy: eval `--smoke` 1 task x 2 init states 88 s (n_rollouts 2, 0
  successes), returns 2 s, train features (1 sample per episode, 40 samples, dim 7,488, output_fn `square`) 343 s, rollout features (16 samples)
  32 s, attribute 6 s, select 5 s. **smoke (not attribution-grade):** n_rollouts 2, n_episodes 40, score table shape [40, 30], all harm columns
  finite, 4 feature files recorded as inputs, select declared functional `F(30) = P(T <= 30 s)`, child 38 episodes. τ agreement: 15 column pairs
  with n = 2 episodes each (only one task has rollouts), so Kendall τ is undefined (NaN): nothing about agreement can be read from this smoke.
  No stage failed (no MPS problem, premise P6 did not bite). The eval and attribute output directories were deleted afterwards.
- ✅ AC8.4 both runs leave the real `data/lineage/mlflow.db` sha256 unchanged (`306af4123fda7e63…`, checked inside each run and at the end of the batch).
- ✅ AC8.5 final batch run on the committed code: `verify_lineage_s3`, `verify_runs_s5 --r2`, `verify_blind_s0`, `verify_method_boundary_s3`,
  `verify_derive_gt_s2` (full, with the hash-seed part), `verify_verdict_s7`, `verify_returns_s6 --synthetic`, `verify_metrics_s1`,
  `verify_consumers_s6` all rc=0; `verify_attribution` 29/29; this feature's `--blind`, `--functionals`, `--pipeline`, `--step4`, `--recovery`,
  `--names`, `--groups`, `--integration` all rc=0.

**Artifact decisions after the final run (diffed one by one):**
- Committed as legitimate refreshes of this feature's own artifacts from the final run: `s1_blind.json`, `s1_refusal_table.csv` (timing 2.5 s ->
  2.4 s), `s3_functionals.json` (3.6c wording), `s4_dry_run_default.txt` and `s4_pipeline.json` (now 313 jobs: the `analyze` job from S7 and the
  sbatch argv lines), `s5_recovery.json`, `s6_names.json` (chain id, run ids, 60 sbatch lines), `s7_groups.json`, `s8_integration.json`
  (timings/ids), `s7_heatmap_SYNTHETIC.png` and `s7_tradeoff_SYNTHETIC.png` (re-rendered with the final layout; both viewed).
- Restored with `git checkout` (other features' artifacts rewritten by their own verifiers): `data-lineage-layer/s5_r2_summary.json`,
  `evaluation-layer/s0_verify_blind.json`, `ground-truth/s6_v3_verify.json` (ids and timings only) and `data-lineage-layer/s3_summary.json`
  (the same structural drift as batch 1, from the verifier having changed in earlier features; restored, not committed).

**Demo artifacts:** `artifacts/s8_integration.json`, `artifacts/s8_trace_select_SYNTHETIC.json` (every edge of the trace), `artifacts/s8_local_smoke.json`
(per-stage seconds and outputs of the real-checkpoint smoke, labelled).

#### Owner-visible cost of the Step 4 chain (final counts, replaces the S6 table: the `analyze` job adds one job everywhere)
Counts are measured from `server_pipeline.py --tag t --dry-run <flags>`. Training GPU-hours use the only figure in the repo (runbook:
"≈ 4 h per seed on one A100, unmeasured on our server"), labelled **inferred**; eval, feature and attribute GPU time are not derivable, so the
GPU column is a lower bound. The smoke gives one real number for scale: 40 train samples took 343 s on MPS, dominated by opening the dataset.

| Configuration | Arms | Jobs | Train jobs | Training GPU-h (inferred) |
|---|---|---|---|---|
| Before this feature (6 arms, 3 seeds) | 6 | 74 | 21 | about 84 |
| **Default now** (`--curation both`, 4 metrics, 3 seeds, 2 random draws, AC7.5 gate) | 21 | **313** | 66 | about 264 |
| `--curation per_task` | 11 | 172 | 36 | about 144 |
| `--curation per_task --metrics f30 hrt60_lin` (keeps the §4.5 head-to-head) | 7 | 112 | 24 | about 96 |
| the same, `--seeds 0 1` | 7 | 76 | 16 | about 64 |
| the same, `--seeds 0 1 --random-draws 1 --no-ac75` | 5 | 56 | 12 | about 48 |
| `--curation per_task --metrics f30 rmst60 auc_expected --no-ac75` | 8 | 130 | 27 | about 108 |

Shrinking flags: `--curation per_task|global`, `--metrics`, `--seeds` (PoC §2.3 allows 2–3), `--random-draws`, `--no-ac75`, `--k-per-task`,
`--from select|arms` (reuse a finished base). (74 became 312 after S4; the S7 `analyze` job makes it 313; with `--subset v3/candidate` a `score` job
is added.)

### Findings for the analyzer (batch 2)
- **Real defects found and fixed by integration (S8):** the select-to-attribute lineage gap, and refused selects leaving FAILED runs (above).
  Both were invisible to the unit tests that monkeypatch `resolve_version`.
- **Plan deviations:** `metric_groups`, `aggregate_by` and the plots live in `vlapoc/analysis.py`, not `attribution.py` (S7); `recovery_run` records one
  `score` run per table instead of one per column (S5); `rollout_subset` has no `failure_type=<t>` value (S3, batch 1); the S5 random-column AUROC band
  is applied to the pooled value only (per type the s.d. is about 0.08 with 13 positives).
- **Site config:** the owner's cluster values moved to `docs/site-profile-cluster.env` (outside the "no cluster path in code" grep scope on purpose);
  `slurm/site.env` is gitignored; a real submission now needs it.
- **Inferred, not measured:** the 4 h/train GPU estimate; `#SBATCH` lines not expanding variables (from the sbatch manual, not re-read); the group
  threshold |z| >= 1; the global arms' size N = k x 20 = 100.
- **Not verifiable here (server):** real attribution scores, retrained-arm verdicts, the same-dataset-opposite-curations result, real v3 recovery,
  SQLite behaviour under concurrent jobs. The smoke's τ agreement is undefined (2 rollouts) and says nothing.
- **Known limits:** the `score` stage needs a table keyed exactly to the named subset (full `v3/candidate` on the server; the integration uses a named
  20-ish-episode subset); the smoke's per-task `__same_task` columns are NaN for tasks without rollouts, so it selects on the global column.
- **Debris to know about:** `data/scratch/` still holds `dry.txt`, `dry_parsed.txt`, `local_smoke.log`, `reg_b2/` (gitignored, safe to delete);
  `data/scratch/cupid_src/` is the planner's CUPID clone.
