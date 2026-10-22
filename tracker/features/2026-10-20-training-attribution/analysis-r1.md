VERDICT: REWORK
ROUTE: executor
ROUND: 1

<!-- Saved verbatim by the orchestrator from the analyzer's returned text: the analyzer's own
     write was refused by the harness ("Subagents should return findings as text"). -->

## Summary (3–5 lines)
Every claimed AC reproduces. I re-ran `verify_attribution.py` (29/29), all eight `verify_training_attribution.py` modes, and nine regression verifiers (all rc=0). `--functionals` needs `--manifest` for 3.2, as the log says. The real `mlflow.db` sha256 was unchanged. CUPID fidelity holds on the output function (ℓ_square on both sides), on the aggregation (sum over steps, Prop. 2 §D.2 p.27), and on parity with the vendored `net` score (2.84e-14). The other differences from CUPID are recorded in `DIVERGENCES`. Three defects need executor rework: the Step 4 summary ignores the macro-KS gate (measured), the SLURM log directory is never created before submission (quoted), and `attribute_run` does not bind feature files to their source data (source read). The default chain size (313 jobs, about 264 GPU-h of training alone) is the owner's call; I flag it but do not block on it.

## Rebuttals received
| From | Point disputed | Their evidence | Conceded? | Why / why not |
|---|---|---|---|---|
| — | none (round 1) | — | — | — |

## Acceptance criteria
All verifier runs are on HEAD `139c32d`. Logs are in `data/scratch/analyzer_r1/`. The other features' artifacts that the verifiers rewrote were diffed: only ids and timings changed, apart from the known `s3_summary.json` drift. All of them were restored with `git checkout`.

| Story | AC | Claimed | Reproduced? | Evidence |
|---|---|---|---|---|
| S1 | 1.1–1.4, 1.6 | ✅ | ✅ | `--blind` → `S1 pass: True (37s)`. Code read: `attribute_run` calls `open_for_method` before any run exists (attribution.py l.564); `cmd_train`, `cmd_features --which train` and `cmd_select` call `method_view` |
| S1 | 1.5 | ✅ | ✅ | `verify_method_boundary_s3.py` → `all S3 checks pass: True (39s)` |
| S2 | 2.1 | ✅ | ✅ | `verify_attribution.py` 6a PASS. I also measured per-sample separation on the real SmolVLA (MPS, 3 samples, one of them with 44 padded chunk steps): a batch of 3 vs three batches of 1 differ by 0.27–0.64% relative (both square and loss). There is no O(1) cross-sample leakage, so P2 holds on the real model and not only on the toy module. `data/scratch/analyzer_r1/batch_sep.json` |
| S2 | 2.2 | ✅ | ✅ | 7a `max abs err 2.84e-14`. The vendored bodies match `error_util.py` @ `2941eba9` (I diffed them by eye against the clone) |
| S2 | 2.3 | ✅ | ✅ | 7b (p=½, τ 1.000) and 7c (p=0.3, τ 0.667) PASS |
| S2 | 2.4 | ✅ | ✅ partly | Not re-run in full (300 s per call). My own MPS run on the same checkpoint loaded it, produced finite 7,488-dim features for both output functions, and `OutputCapture` never raised |
| S2 | 2.5, 2.6 | ✅ | ✅ | 8a–8c PASS; `--functionals` 2.5 PASS (params carry output functions, encoding and DIVERGENCES) |
| S3 | 3.1, 3.3–3.6 | ✅ | ✅ | `--functionals` 16/16 of those checks PASS |
| S3 | 3.2 | ✅ | ✅ with caveat | Without `--manifest` the check fails (`S3 pass: False (16/17)`), so the default invocation exits 1. The committed artifact shows PASS with a real manifest. The code writes `rel()` paths (influence.py l.326–328) |
| S4 | 4.1–4.3, 4.6 | ✅ | ✅ | `--pipeline` 8/8; default dry run `would submit 313 jobs`, 66 trains |
| S4 | 4.4 | ✅ | ❌ **substance** | `--step4` passes, but the summary it tests violates §4.4 (Feedback: executor 1) |
| S4 | 4.5 | ✅ | ✅ | `--step4` 4.5: 43/48 per task, 860 kept |
| S5 | 5.1–5.6 | ✅ | ✅ | `--recovery` 11/11. Hypotheses commit `b4790ee` is in the history before `4965f22` |
| S6 | 6.1–6.6 | ✅ | ✅ | `--names` 12/12. `git grep -E '/raid\|/auto/home'` outside `tracker`/`calls` hits only `docs/site-profile-cluster.env`, by design. Logging defect in Feedback: executor 2 |
| S7 | 7.1–7.4 | ✅ | ✅ | `--groups` 9/9 |
| S8 | 8.1, 8.2, 8.4 | ✅ | ✅ | `--integration` 10/10 (47 s) |
| S8 | 8.3 | ✅ | not re-run | The local smoke takes about 8 min. The MPS feature path is independently exercised by my batch check above |
| S8 | 8.5 | ✅ | ✅ | `lineage_s3`, `runs_s5 --r2`, `blind_s0`, `method_boundary_s3`, `derive_gt_s2`, `verdict_s7`, `returns_s6 --synthetic`, `metrics_s1`, `consumers_s6`: all rc=0 |
| Cost | the cost table | — | ✅ | Measured: `--curation per_task` 172 jobs / 36 trains; `+ --metrics f30 hrt60_lin` 112 / 24; `+ --seeds 0 1 --random-draws 1 --no-ac75` 56 / 12. All match the log |

## PoC scope assessment
- **Covered:**
  - §4.3: one return per metric, CUPID estimator.
  - §3.4: `functionals.json` per column.
  - §4.4: Kendall's τ, and labels carrying metric and τ.
  - §4.5: metric-targeted hypotheses declared before scoring, the HRT-vs-F30 head-to-head, heatmap and scatter.
  - §9.1 Step 4: harmful and helpful removal, random controls, per-task and global modes (owner Q1).
  - §9.2: key contract at attribute, select, ensemble and score.
  - Blindness: all four method entry points go through `open_for_method`.
- **Missing:**
  - The two-step gate in the Step 4 aggregate (executor 1).
  - Binding of feature files to the data they were computed from (executor 3).
- **Over-built:** nothing I would remove. The default chain size is a cost question, not over-building of code (Non-blocking 1).
- **Silent divergences:**
  - The ensemble's z-score averaging differs from TRAK's averaging of raw scores across models, and the ensemble run does not carry `DIVERGENCES` in its params. It is stated only in the docstring and in `functionals.json`'s `aggregation` field (Non-blocking 3).
  - `metric_groups` labels an episode `consistent_*` with only 2 of 5 metrics past the threshold. The PoC says "Improve every metric" / "Degrade every metric" (§4.2 p.10). This is not recorded as a divergence (Non-blocking 4).

### CUPID fidelity, point by point (priority 1)
| Element | CUPID (paper / code) | Default here | Status |
|---|---|---|---|
| Output function | ℓ_square on both sides, Eq. 9 (App. A.1 p.18); Eq. 8, the loss form, "work[s] poorly in practice" (quoted) | `square`/`square` (influence.py l.83; stage.py l.72) | **CUPID's** |
| Hessian | H_square = Gauss-Newton, 1/(\|D\|H) Σ ggᵀ (quoted p.18) | ΦᵀΦ/N of train features (attribution.py `gram`) | CUPID's |
| TRAK Q term | `get_out_to_loss_grad` = ones (modelout_functions.py, read) | none | equivalent |
| Regularization | `lambda_reg=0.0` (train_trak.sh l.311) | λ = 1e-3·mean diag | recorded (`damping`) |
| Projection | dense, d = 4000 (train_trak.sh l.309) | Kronecker r = 8 per layer, 7,488 dims, Linear weights of the trainable set | recorded (`projection`) |
| Draws | B = 64 (A.2 p.19) | K = 4 shared | recorded (`draws`) |
| Return weighting | R = ±1, uncentred (A.2 p.19; `net`) | centred raw by default; `--cupid-returns` gives ±1 | recorded (`centred_returns`), with corrected algebra and a counterexample (7c). This is the planner's N2 default, not owner-approved, and labelled inferred |
| Per-demo aggregation | Eq. 3 sum over (s,a)∈ξ and (s',a')∈τ; `sum_of_sum` + `net` | group sums, `rollout_mean=False` | CUPID's (parity 2.84e-14) |
| Strides, numerics, model family, failure-type returns | — | — | recorded |

## Overcomplication & reinvention audit
| Mechanism (file:line) | Existing OSS alternative | Covers our need? (spec rules checked) | Evidence | Recommendation |
|---|---|---|---|---|
| `KroneckerProjector` + `influence_matrix` (influence.py l.86–141; attribution.py l.150–210) | `trak` projectors | Partly. TRAK projects the whole flattened gradient densely (`BasicProjector`, `CudaProjector`, which needs CUDA via `fast_jl`; `ChunkedCudaProjector` chunks by parameter count, not per layer). That is CUPID's own route and would remove the `projection` divergence, but it is CUDA-only and cannot be tested on this Mac | quoted: `trak/projectors.py` (GitHub main); memory and cost argument inferred | **Keep.** The math core is about 40 lines and brute-force checked. The swap to TRAK is inferred-only, so not blocking |
| same | `kronfluence` | No. Its Hessians are `identity`/`diagonal`/`kfac`/`ekfac`, which is not CUPID's projected Gauss-Newton. It has no random projection, only a `query_gradient_low_rank` option. `compute_measurement` could hold ℓ_square, but the docs do not cover return weighting | quoted: kronfluence `DOCUMENTATION.md` ("identity, diagonal, kfac, and ekfac"; low-rank query batching) and README ("nn.Linear and nn.Conv2d") | **Keep** (adopting it would change the estimator) |
| same | pytorch-influence-functions / dattri | Not CUPID's estimator. dattri was also checked by the planner | inferred | Keep |
| `server_pipeline.py` (`sbatch --dependency=afterok`) | submitit / Snakemake | submitit's examples doc shows only `update_parameters` (partition, arrays), with no dependency chaining. Snakemake would be a second record system next to MLflow | quoted: `submitit/docs/examples.md` | **Keep** |
| `vlapoc/step4.py` (summary) | — (aggregates `verdict.json`; reuses `vlapoc.verdict`) | Small, reuses existing code | inferred | Keep, but fix the gate (executor 1) |
| `vlapoc/recovery.py` | Wraps the existing `groundtruth.score_recovery` | Reuse, not reinvention | source read | Keep |
| `vlapoc/analysis.py` `aggregate_by` (hand percentile bootstrap) | `scipy.stats.bootstrap` | Yes, a drop-in | inferred | Simplify (non-blocking) |
| `load_site` (`server_pipeline.py` l.71–87, `.env` parser) | `python-dotenv` | Would add a dependency for about 15 lines | inferred | Keep |

Mechanisms checked with nothing found: the per-task top-k selection ops (an earlier feature's `lineage` ops) and `column_declarations`. Searches: TRAK projector source; kronfluence README and DOCUMENTATION; submitit examples; a dependency search for "submitit slurm afterok slurm_additional_parameters" (no submitit-specific result).

## Feedback: planner
(none blocking)

## Feedback: executor
1. **`vlapoc/step4.py` l.66–70 (`call`) and l.97–108 ignore the macro-KS gate.**
   - **Spec:** PoC §4.4 p.10 says "First the macro-KS gate ... Then each metric's bootstrap CI gives the direction", and §9.1 Step 4.4 p.19 says "If it did, check each metric's confidence interval".
   - **Measured:** three verdicts with `gate.fires=False`, outcome `iii`, f30 `p_holm=0.01` produced `gate_fires 0/3, outcome_by_seed iii,iii,iii, f30__by_seed T,T,T, f30__supports 3`, i.e. "expectation met in 3/3". Command: inline `summarize([...])`, see `data/scratch/analyzer_r1`.
   - **Fix:** make `call()` take the verdict and return `0` (or a distinct `g`, gate not fired) when `not v["gate"]["fires"]`.
   - **Test:** add a check to `verify_training_attribution.py --step4` that plants gate-not-fired, Holm-significant verdicts and expects `__supports 0`.
2. **The SLURM log directory is never created before submission (`scripts/server_pipeline.py` l.99–100; `slurm/env.sh` `mkdir -p "$LOG_DIR"`).**
   - SLURM opens `--output` before the script runs and does not create missing directories; the job "fails silently" (quoted: SchedMD tickets 8895 and 14956, and the slurm-users 2021-07 thread "incorrect output directory fails silently"). The old sbatch files pointed at an existing `/raid/.../logs`.
   - On a new site from `site.env.example`, every job dies before `env.sh`'s `mkdir` runs, and the `afterok` dependents stay pending.
   - **Fix:** in `main()`, outside `--dry-run`, run `Path(site["LOG_DIR"]).mkdir(parents=True, exist_ok=True)` before the first `submit`. Note in `site.env.example` that `LOG_DIR` must be on a filesystem the compute nodes share.
   - **Test:** extend `--names` to check that a non-dry `main` with a stubbed `subprocess.run` creates `LOG_DIR`.
3. **`attribute_run` (`vlapoc/attribution.py` l.571–577 and `attribute()` l.512–513) does not bind feature files to their source data.**
   - **Train side:** it checks only the checkpoint sha and that the episode indices equal `subset.resolve()`. Train features computed on another version's root with the same indices (for example clean v0 for a v3 subset) pass, and their scores are keyed to v3's content hashes. That is a §9.2 key-contract hole.
   - **Rollout side:** it checks only that the set of `traj_episode_index` values matches. Features from another eval of the same checkpoint with the same rollout count pass. The docstring's "the rollout features exactly the recorded trajectories" overclaims.
   - The stage CLI happens to build the right paths, but `python -m vlapoc.attribution` takes arbitrary ones.
   - **Fix:** compare every manifest's `dataset_root` with `rel(open_for_method(subset).lerobot_root)` on the train side. On the rollout side, recompute `sha256_json(file_digests(root))` and compare it with the eval's `output.trajectories`, as `stage.cmd_features` already does (stage.py l.196–199).
   - **Test:** add two AC8.2-style negatives to `--integration` (train features from another version root, rollout features from another eval), each refused with no run written.
4. **Stale commands in docstrings.** `scripts/verify_training_attribution.py` l.7 and `vlapoc/third_party/cupid_quality.py` l.8 cite `--estimator`, which does not exist. The parity check lives in `scripts/verify_attribution.py` (check 7a). Fix the text.
5. **`--functionals` without `--manifest` exits 1** (`S3 pass: False (16/17)`). Make 3.2 a SKIP that does not fail the run when no manifest is given, or default `--manifest` to `artifacts/s3_manifest_example.json`'s source. Today the documented invocation fails.

## Non-blocking
1. **Cost (owner's call; PoC §10 p.20 lists "Compute budget" as an open decision).**
   - The default is 313 jobs and 66 trainings, about 264 GPU-h of training alone. That figure is inferred from the runbook's "≈ 4 h per seed", so it is a lower bound. Before this feature it was 74 jobs.
   - The shrinking flags are correct and documented in the module docstring and the execution log. I re-measured them: `--curation per_task --metrics f30 hrt60_lin` gives 112 jobs / 24 trains, which still covers the §4.5 headline and §9.1 Step 4.1 and 4.5.
   - I would suggest that as the first server run, with the global mode second. The owner accepted the doubling under Q1, so this is not blocking.
2. **Centred returns are the default for binary columns too** (planner N2, labelled inferred, not owner-approved). The "unbiased baseline" argument assumes ∇log π, and it does not obviously carry over to the ℓ_square surrogate (inferred). Consider making `pm1` the default for binary columns so the F(30) column is CUPID's own, or put N2 to the owner.
3. **The ensemble run's params lack `DIVERGENCES`.** The z-score averaging (versus TRAK's averaging of raw scores across models) is stated only in its docstring and in the `aggregation` field. The ensemble is the table Step 4 ranks on, so add both to its params.
4. **`analysis.metric_groups` thresholds.**
   - It calls ≥ 2 harmful with 0 helpful "consistent_harmer", but §4.2 p.10 says "Degrade every metric". Record this as a divergence or require all non-neutral calls.
   - `harm_f30` and `harm_censored60` are two nearly redundant reliability readouts (inferred), so two of the five "metrics" can make an episode "consistent".
5. **`server_pipeline.py` l.275** calls `step4-summary --tag {t}` without `--chain-id {ch.chain}`. With any older chain under the same tag, the final job refuses ("several chains"). Pass the chain id.
6. **`aggregate_by`** could use `scipy.stats.bootstrap` instead of the hand-written percentile bootstrap.
7. **`analyze_run` reads catalog rows** through `Subset.load(...).episodes()` rather than `open_for_method`. These are metadata only, already opened through the loader by the attribute run, and there is no label access. Fine, but worth one line in the docstring.
