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
