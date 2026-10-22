# Refactor: training and attribution

Stage 4 of the feature pipeline, after analyzer round 2 (PASS). Code commit: `6dce7d2` (on top of `aa47526`).
Evidence labels: **measured** (command run in this session), **inferred** (reasoning only).
Scope was kept small on purpose: the modules were already documented to the standard asked for (module docstring with
grounding, docstrings on public functions, usage examples on every entry point), so the edits are the gaps plus the two
doc-only items from analysis-r2.

## What changed

| File | Change | Why it helps a reader |
|---|---|---|
| `vlapoc/attribution.py` | Migration note in the `attribute_run` docstring and appended to its two data-binding refusal messages (`regen` string). The train manifest is read once (`tm0`) instead of loading the same file a second time as `tman`. | A server user who hits "features were computed on ..." is told what to do (regenerate with `stage.py features`; hand-editing `dataset_root` defeats the check). One read, one name. |
| `vlapoc/analysis.py` | `metric_groups` docstring and `DIVERGENCE_NOTE` reworded (analysis-r2 Non-blocking 2). The rule is unchanged. Added a "Where this sits" paragraph. | The rationale no longer claims redundancy justifies >= 2-of-5. It now says the rule is a judgement call (inferred, nothing measured) and names the real risk: `harm_f30` and `harm_censored60`, two near-redundant reliability readouts, can alone make an episode a "consistent harmer". The owner can see the choice is theirs. |
| `scripts/server_pipeline.py` | Migration note in the module docstring (the runbook for this chain); docstrings on `Chain` and `main`. | The chain's runbook says what to do with pre-round-2 feature files. |
| `scripts/stage.py` | One-line docstrings on `run_name` and every `cmd_*`; `cmd_features` and `cmd_attribute` carry the migration note. | Each stage subcommand says what it does and which `vlapoc` function it wraps. |
| `scripts/verify_training_attribution.py` | Module docstring lists all nine modes with the exact command (it listed three and one line of prose). | The owner can run any check from the docstring. |
| `vlapoc/step4.py`, `vlapoc/recovery.py` | "Where this sits" paragraph (docs/dataflow.md position and reading order). | Pipeline position and the order to read functions in. |
| `vlapoc/recovery.py` | Removed unused `argparse`, `json`, `sys` imports and `score_table_digest` (a function-local import). | No dead imports. |
| `vlapoc/method_data.py` | Removed unused `numpy` import and `version_episodes` from the `lineage` import. | Same. |

Behavioural footprint, stated plainly: the text of two `ValueError` messages in `attribute_run` now has a suffix (the original
text, including the substrings the verifier matches, is intact), and the `group_rule_divergence` param recorded by the `analyze`
stage now carries the reworded `DIVERGENCE_NOTE`. Neither changes any computed value, any control flow or any result.

## Behaviour check

Baseline run on `aa47526` (clean tree, before any edit) and a re-run on `6dce7d2`, same script, logs under
`data/scratch/refactor_before/` and `data/scratch/refactor_after/`. All 18 commands exit 0 both times.
The first "after" attempt on a staged-but-uncommitted tree had three verifiers (`--integration`, `verify_runs_s5 --r2`,
`verify_consumers_s6`) refuse with "source under ('vlapoc', 'scripts') differs from HEAD"; that is the repo's own
commit-first guard, not a behaviour change. After committing, all three pass.

| Command | Before == After? | Note |
|---|---|---|
| `verify_attribution.py` | yes | `pass: True (29/29)` both |
| `verify_training_attribution.py --blind` | yes | `S1 pass: True` |
| `... --functionals` | yes | 17/17; 3.2 is `[SKIP]` without `--manifest`, both times |
| `... --pipeline` | yes | 8/8 |
| `... --step4` | yes | 13/13 |
| `... --recovery` | yes (ids) | 11/11; only the chain sha inside check 5.4's printed command differs |
| `... --names` | yes (ids) | 13/13; only a run id in check 6.3's detail differs |
| `... --groups` | yes | 9/9 |
| `... --integration` | yes | 12/12 (including both new refusal negatives, whose substrings the verifier matches) |
| `verify_lineage_s3.py`, `verify_runs_s5.py --r2`, `verify_blind_s0.py`, `verify_method_boundary_s3.py`, `verify_derive_gt_s2.py`, `verify_verdict_s7.py`, `verify_returns_s6.py --synthetic`, `verify_metrics_s1.py`, `verify_consumers_s6.py` | yes | all rc=0 both times |
| `data/lineage/mlflow.db` sha256 | yes | `306af4123fda7e635892d459a2e34008ba1cb62f6c4fce152c546e0da7bdde79` before and after |

Per-check `[PASS]/[FAIL]/[SKIP]` lines were diffed between the two runs: identical except the two lines named above (chain sha and
run id, which change every run). Tracked artifacts the verifiers rewrite were compared before vs after: only run ids, chain sha
(628 lines of `s4_dry_run_default.txt`, 0 after normalising the 8-hex chain id) and timings differ; `data-lineage-layer/s3_summary.json`
shows its known drift against HEAD in both runs and is identical between them. Every rewritten artifact was restored with
`git checkout`; the tree shows only the owner's four uncommitted files.

Not re-run: `--local-smoke` (about 8 minutes, real SmolVLA on MPS) and `verify_attribution.py --cpu-smoke`. Neither exercises any
changed line (docstrings, two imports, a suffix on two error messages, one duplicate file read removed).

## Left as-is (and why)

- `vlapoc/third_party/cupid_quality.py`: vendored verbatim from `agiachris/cupid` @ `2941eba9`; the missing docstring on
  `sum_of_sum_influence` is part of the verbatim body and check 7a pins it to CUPID's own output.
- Nested `fwd`/`bwd`/`forward` hooks in `influence.py` and the verifier-local helpers without docstrings (`check`, `row`, ...):
  private closures, named for what they do, documented by the check that calls them.
- `runs.py`, `lineage.py`, `verdict.py`: the feature-touched parts (`select_run(require_declaration, extra_params)`, `verify_run(extra_tags)`,
  manifest paths via `rel`) already carry docstrings naming the story and the PoC section. Reformatting them would blur the diff to earlier features.
- `slurm/*`: `site.env.example` and the sbatch scripts already say what each variable and flag is for.
- `attribution.py` is 750 lines. Splitting it (estimator / provenance / ensemble) is a real simplification but touches imports across
  verifiers and is not "small, clearly safe"; left for the owner to decide.
- The `>= 2`-of-5 rule itself, the cost default (313 jobs) and centred-returns default: owner decisions (analysis-r2 Non-blocking 2, 3).
- `# noqa: E402` on the imports in the middle of `recovery.py`: deliberate; the file's first half was committed alone before scoring
  code existed (S5 ordering), so the imports sit under that boundary.

## How to read the code

Follow one attribution run through the chain, then the checks. Spec: PoC §9.1 Steps 3-4 (p. 19), §4.3-§4.5 (pp. 10-11).

1. `docs/dataflow.md`, Step 3 and Step 4: the picture the code implements.
2. `scripts/server_pipeline.py` docstring: the whole experiment as one SLURM dependency chain (base, rank, select, arms, analyze, score, summary),
   with the PoC sentence each arm tests. Run `server_pipeline.py --tag demo --dry-run` to see every job and its command.
3. `scripts/stage.py` docstring and `cmd_*`: one subcommand per tracked stage; shows which `vlapoc` function each wraps. `method_view` is the
   one door through which `train`, `features --which train` and `select` read a dataset (blindness).
4. `vlapoc/method_data.py` (`open_for_method`): the four refusals (fenced, ground-truth-derived, tampered, not blind) every method input passes.
5. `vlapoc/influence.py`: `featurize` turns a checkpoint plus data into projected per-sample gradient features. Read the module docstring for
   the output function (`square` is CUPID's, `loss` the old behaviour), then `OutputCapture`, `square_objective`, `batch_features`.
6. `vlapoc/attribution.py`: module docstring and `DIVERGENCES` first (every difference from CUPID, with its source). Then `performance_influence`
   (CUPID's `net` score), `column_plan` / `column_declarations` (which functional each column explains), `attribute_run` (opens the subset
   blind, checks features against their data, writes scores plus `functionals.json`), then `ensemble_run`.
7. `vlapoc/third_party/cupid_quality.py`: CUPID's own two functions, used only as the parity reference in `verify_attribution.py` check 7a.
8. `vlapoc/analysis.py`: `metric_groups` (four groups, with the named divergence), `aggregate_by`, the two figures, `analyze_run`.
9. `vlapoc/step4.py`: `call` (T/C/0/g per metric and seed, gated on macro-KS), `summarize`, `summary_run`. The verdicts it reads come from
   `vlapoc/verdict.py` `verify_run`.
10. `vlapoc/recovery.py` (ground-truth side only): `EXPECTED_METRICS` (committed before any scoring), `recovery_matrix`, `recovery_run`.
11. Verification: `scripts/verify_attribution.py` (estimator maths, parity with CUPID) and `scripts/verify_training_attribution.py`
    (nine modes, one per story; the module docstring lists the commands).
