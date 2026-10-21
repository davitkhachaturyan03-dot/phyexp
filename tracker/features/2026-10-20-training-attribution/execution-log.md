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
