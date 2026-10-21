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
  fixture, and the result is logged there.** Until then this half is by static read only (`grep -n train_output_fn
  vlapoc/attribution.py`).
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
