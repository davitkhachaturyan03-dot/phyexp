# Plan: training and attribution (compliance review and gap-closing for the ported branch)

Round 1 (planner, 2026-10-20). Evidence labels: **quoted** (source and page), **measured** (a command run in this
session, reproducible; outputs kept under `data/scratch/`), **inferred** (my reasoning, unverified).

## Request

The owner asked that the code which arrived on the branch (`attribution-curation-merged`, ported from the cluster
project `VLA-data-curation`) be treated as a feature: check it against a training-and-attribution feature and the
whole PoC, and have the pipeline agents fix every gap ([brief.md](brief.md), verbatim request there). **Owner scope
ruling (brief.md, "Owner answers", 2026-10-20): "Just dismiss the parts that need the server and fix everything
else".** Anything that only the server can check or produce is listed once under "Out of scope (needs the server)"
and gets no story. Everything else that is wrong or missing gets a story that can be verified on this Mac.

What is under review: `vlapoc/influence.py` (255 lines), `vlapoc/attribution.py` (444), the per-task ops in
`vlapoc/lineage.py` (+84), `vlapoc/runs.py` (+32), `vlapoc/catalog.py` (path recording), `scripts/stage.py` (268),
`scripts/server_pipeline.py` (161), `scripts/verify_attribution.py` (324), `slurm/*` (66). Measured:
`git diff main...HEAD --stat` gives 23 files, +2,136/−22.

## Spec grounding

PoC = `docs/VLA_Data_Attribution_PoC.pdf`, read in full this session from the PDF (`data/scratch/poc.txt`, 21 pages).

- **§2.1 item 1 (p. 3)** (quoted): "Every training run should be tied to exact dataset versions (content hashes),
  episode IDs, ... the config, the git SHA, and the resulting checkpoints. Every eval rollout should be tied back to
  that checkpoint."
- **§2.1 item 4 (p. 3)** (quoted): "Deliberately corrupt known subsets of training data ... Then check whether your
  attribution method flags exactly those. This turns 'the method says X' into 'the method recovered 90% of the
  injected bad episodes.'"
- **§2.3 (p. 4)** (quoted): "Fixed config and steps; 2–3 seeds per data configuration".
- **§2.6 (p. 5)** (quoted): "Remove the top-k 'harmful' episodes and retrain: the policy should improve more than
  after removing k random episodes. Remove the top-k 'helpful' episodes and retrain: the policy should degrade more
  than after random removal. ... use the PhAIL significance gate (Section 3.4)".
- **§3.4 (p. 7)** (quoted): "Every attribution result states which functional it explains, following PhAIL's
  disclosure principle." and "Strata. Task-level breakdowns. Per-object cells and macro-aggregation map directly to
  per-task attribution."
- **§4.2 (pp. 9–10)** (quoted): "Results are reported as an episode × metric matrix." The four groups: "Consistent
  helpers / Consistent harmers / Trade-off episodes / Metric-specific".
- **§4.3 (p. 10)** (quoted): "CUPID-style influence on returns | Yes, with one return definition per metric".
  "The estimator is CUPID's; the way it is applied is not."
- **§4.4 (pp. 10–11)** (quoted): "Two-step verdict per intervention. First the macro-KS gate ... Then each metric's
  bootstrap CI gives the direction"; "correct for it (e.g. Bonferroni or Holm across metrics) before labeling an
  episode as a trade-off"; "Label every claim with its metric and time cap τ"; "Agreement check. Report rank
  correlation (e.g. Kendall's τ)".
- **§4.5 (p. 11)** (quoted): "Metric-targeted injections ... Check whether attribution assigns each corrupted group to
  the expected metric"; "'Same dataset, opposite curations.' Curate the dataset twice — once for HRT, once for
  reliability (e.g. F(30 s) or hard-failure rate) — retrain on each subset, and show that each curated policy wins on
  its own metric"; "Visualisation. An episode × metric heatmap, and a scatter of throughput influence against
  reliability influence with the four groups as quadrants."
- **§6 (p. 12)** (quoted): CUPID "Relevance: Closest match to the PoC; best starting point to fork." ATHENA:
  "Kronecker-structured gradient projection and a low-rank Hessian approximation".
- **§9.1 Step 3 (p. 19)** (quoted): "For each metric, CUPID-style influence estimates how much each training episode
  pushed that return up or down. Repeat the analysis for subsets, such as a specific task or only rollouts with a
  specific failure type."
- **§9.1 Step 4 (p. 19)** (quoted): "1 Remove the top-k most harmful episodes and retrain. As a control, remove k
  random episodes and retrain. 2 Evaluate both policies as in Step 2. 3 Apply the macro-KS test ... 4 If it did,
  check each metric's confidence interval for the direction of the change. 5 Repeat for the top-k most helpful
  episodes (removing them should make things worse)."
- **§9.2 (p. 19)** (quoted): "which kind of harm each data group causes — drops and safety stops versus slowness".
- **§10 (p. 20)** rows "Fine-grained attribution: Fork CUPID; attribute against ghost-event rate and RMST separately;
  add TRAK or RoboDrop-style scores; compare against DemInf", "Metric-resolved attribution", "Verification & demo:
  Counterfactual removal tests; dashboard linking failed rollouts to influential episodes" (quoted).
- **§1 (p. 3)** (quoted): "Per-episode scores become most valuable when aggregated by metadata."

**Rules the build must follow**
- R1. A method under test reads data only through `vlapoc.method_data.open_for_method` (ground-truth feature, owner
  decision 3). v1 and v2 are fenced; v3 is the blind injected version.
- R2. Scores are keyed by (`episode_id`, `content_hash`) and are refused for versions the run did not train on (§9.2;
  `runs.check_scores`).
- R3. Every attribution column declares its functional: return column, family, τ, rollout subset, sign (§3.4, §4.4).
- R4. Step 4 compares the curated policy against the original policy and against a random-k control, with the
  two-step verdict (macro-KS gate, then per-metric CIs, Holm), n_boot 1000 (owner ruling B).
- R5. Every stage is a tracked MLflow run; paths repo-relative; the guard refuses dirty code or another commit.
- R6. Data is the only variable (PoC p. 1): the AC8.5 recipe unchanged, per-subset normalization (owner decision).
- R7. Anything that writes MLflow runs in a test uses `runs.scratch_lineage_copy`; the real `data/lineage/mlflow.db`
  sha256 is checked before and after (`306af4123fda7e63…` at the start of this session, measured).

**Papers**
- `docs/papers/CUPID_Influence_Functions_arXiv-2506.19121.pdf` (text in `data/scratch/papers/CUPID.txt`):
  - §5.1 p. 5, Prop. 1 / Alg. 1 (quoted): Ψ̂(ξ) = (1/m) Σ_i R(τ_i)/H Σ_{(s',a')∈τ_i} Σ_{(s,a)∈ξ} Ψ_a-inf. Footnote:
    "the fraction 1/H appears from the assumption that all trajectories have equal length".
  - §D.2 p. 27, Prop. 2 (quoted), the variable-length case: Ψ_π-inf(ξ) = E_τ[R(τ) Σ_{(s',a')∈τ} Σ_{(s,a)∈ξ} Ψ_a-inf].
    There is no 1/H, so the branch's "1/H dropped" is CUPID's own variable-length form, not a divergence.
  - App. A.1 pp. 18–19 (quoted): approximating with the denoising-loss gradient (Eq. 8) "we find this approximation
    to work poorly in practice"; CUPID instead uses "a surrogate, label-agnostic output function ℓ_square(s,a;π_θ) :=
    E[‖ε_θ(...)‖²]" on both sides, with H_square the Gauss-Newton Hessian of ℓ_square (Eq. 9).
  - App. A.2 p. 19 (quoted): "R(τ) = 1 if τ completes the task and R(τ) = −1 otherwise"; "batch of B = 64 samples
    (ε, i)"; TRAK "projection dimension d=4000", "single policy checkpoint C=1".
- CUPID released code, `github.com/agiachris/cupid` @ `2941eba9` (cloned to `data/scratch/cupid_src/repo`), measured:
  `scripts/train/train_trak.sh` l.309–316 sets `proj_dim=4000`, `lambda_reg=0.0`, `loss_fn="square"`,
  `num_timesteps=64`; `diffusion_policy/data_attribution/modelout_functions.py` implements `"square"` as
  `pred**2` averaged over unmasked entries; `diffusion_policy/common/error_util.py::compute_demo_quality_scores`
  with `metric="net"` = Σ_{successful τ} M[τ] − Σ_{failed τ} M[τ], which is R = ±1, no centring and no 1/m.
  Pairwise aggregation is `mean_of_mean` or `sum_of_sum` (`error_util.py` l.7–24).
- `docs/papers/ATHENA_Influence_Functions_arXiv-2606.16208.pdf`:
  - Eq. 2 p. 3 (quoted): 1/H_i normalisation by demo length, R(τ_j) ∈ {1, −1}.
  - App. A.1 Eq. 14–15 p. 14, for flow-matching VLAs (π0, the same family as SmolVLA; quoted): "square-flow attribution
    surrogate f_sf(s,a;θ) = E_{t,ε}[‖v_θ(x_t,s,t)‖²]" on the rollout side and the supervised flow-matching loss
    ∇L(z) on the training side.
  - §3.2 p. 3 (quoted): "imbalanced greedy selection under heterogeneous task distributions" is a named bottleneck.
  - §4.2 Eq. 7 p. 5 (quoted): task-local influence over E_c(i), the rollouts of demo i's own task. This is the
    paper ground for the `__same_task` columns and for balancing curation per task.
- `docs/papers/QoQ_Quality_over_Quantity_arXiv-2603.09056.pdf` p. 1 (quoted): "aggregating influence scores of
  state-action pairs within the same trajectory". This is per-trajectory aggregation; QoQ's max over validation
  samples does not apply to return-weighted influence.

## Compliance matrix (every PoC requirement the branch's code touches)

| # | Requirement (source) | Status | Evidence |
|---|---|---|---|
| C1 | Episode × metric matrix, one return per metric (§4.2 p.9, §4.3 p.10) | **Met** | `attribution.score_table` has one `infl_`/`harm_` column per family-W return (`metrics_registry.return_columns("W")`); measured: `verify_attribution.py` checks 3a–3d PASS |
| C2 | Same-task and per-failure-type subsets (§9.1 Step 3 p.19) | **Met, interpretation recorded** | `__same_task` = ATHENA Eq. 7's task-local term (quoted). Failure types are an indicator return over all rollouts, not "only rollouts with a failure type" (inferred design choice, not recorded in run params). `MIN_FAILURE_TYPE_ROLLOUTS = 5` is an invented threshold, also absent from params (static read, l.73, l.364–370) → S3 |
| C3 | Kendall's τ agreement (§4.4 p.11) | **Met** | `attribution.agreement` (scipy `kendalltau`); measured: check 3d PASS |
| C4 | Estimator is CUPID's (§4.3 p.10, §6 p.12, §10 p.20 "Fork CUPID") | **Partly met** | Aggregation matches CUPID Prop. 2 (sum over steps, no 1/H; quoted §D.2). TRAK Gauss-Newton H = ΦᵀΦ/N matches Eq. 9's 1/(\|D\|H) normalisation (quoted). **Gap:** the per-sample gradient is the *training loss* on both sides (`influence.py` l.205–207, `policy.forward(..., reduction="none")`; measured in lerobot 0.6.1 `modeling_smolvla.py` l.315–321, the per-sample mean MSE). That is CUPID's Eq. 8, which the paper says "work[s] poorly in practice"; CUPID uses ℓ_square on both sides and ATHENA uses ‖v_θ‖² on the rollout side → S2. **Divergences, unrecorded or under-recorded:** centred returns vs CUPID's R = ±1 (recorded, but the "scaled by p(1−p)" claim is only true for a difference of *means*, while CUPID's code takes a difference of *sums*; derivation in S2); K = 4 shared (noise, time) draws vs B = 64; Kronecker r = 8 per layer (7,488 dims) over the trainable expert layers only vs dense d = 4000 over all parameters; damping λ = 1e-3·mean diag vs CUPID `lambda_reg=0.0`; strides 2 (train) / 4 (rollouts) vs every sample → S2 records all of them |
| C5 | Methods stay blind (§2.1 item 4 p.3; owner decision 3) | **Partly met** | `attribute_run` calls `assert_blind` only (l.329). It skips `open_for_method`'s integrity check, its refusal of ground-truth-derived subsets and its diff guard. `stage.py features --which train` opens the version with `LeRobotDataset` and no check at all (l.129–132). `stage.py train` calls `train_run` with no check (the ground-truth feature's known issue) → S1 |
| C6 | Key contract (§9.2 p.19) | **Met** | `attribute_run` calls `check_scores` (l.349); `drop_top_k_per_task` calls `_check_score_keys` (exact parent keys); ensemble requires identical keys. Measured: checks 5e and 4b PASS. Only covered by unit tests with `resolve_version` monkeypatched (`verify_attribution.py` l.137); no test runs the stages end to end → S8 |
| C7 | Declared functional per result (§3.4 p.7; §4.4 "label every claim with its metric and τ") | **Partly met** | Run params carry `return_columns` and `good_sign`. No per-column record of family / τ / rollout subset / centring. The ensemble run declares nothing about its columns. `select` does not record which functional its metric explains → S3 |
| C8 | Step 4 harmful removal vs random-k control (§9.1 Step 4.1–4.4 p.19; §2.6 p.5) | **Partly met** | Measured: dry-run gives 74 jobs, arms `harm-{f30,rmst60,auc_expected}-k5`, `random{0,1}-k5`, `drop-random-20`. Every arm is verified only against the base policy. There is no curated-vs-random verdict, so "improve more than after removing k random" (§2.6) is never tested, and no step aggregates over seeds → S4 |
| C9 | Helpful removal (§9.1 Step 4.5 p.19; §2.6 bullet 2) | **Missing** | No `direction lowest` / helpful arm in `server_pipeline.arms()` (static read l.76–87; dry-run arm list above) → S4 |
| C10 | "Same dataset, opposite curations": HRT vs reliability (§4.5 p.11) | **Missing** | Default metrics `f30 rmst60 auc_expected`; no `hrt60_lin` arm and no harm-HRT vs harm-F30 head-to-head verdict → S4 |
| C11 | Global top-k (§9.1 Step 4.1 "Remove the top-k most harmful") | **Divergence, not owner-approved** | Per-task top-k with matched per-task random controls (`lineage.drop_top_k_per_task`). Grounded in ATHENA §3.2/§4.2 (quoted above) but not in the PoC; recorded as "inferred design choice" in the docstring, never put to the owner → BLOCKING Q1, S4 |
| C12 | Two-step verdict, Holm, n_boot 1000 (§4.4; owner ruling B) | **Met** | `stage.py verify` → `verdict.verify_run` (defaults `n_boot=1000`, `multiplicity: holm`, plan guard on). Measured: `verify_verdict_s7.py` PASS |
| C13 | Attribution-grade eval and fixed batch plan (eval layer AC5.8/AC5.9/AC7.8) | **Met** | `cmd_eval` passes `attribution_grade=not a.smoke`, batch 50; `verify_run` refuses mismatched plans (static read) |
| C14 | Recipe and normalization unchanged (R6; owner decision) | **Met** | `cmd_train` = runbook AC8.5 command (20,000 steps, batch 64, `load_vlm_weights=False`, smolvla_base @ `d9f33c94`, `train_run` default `normalization="per_subset"`); static comparison with `server-runbook.md` l.108–113 |
| C15 | Lineage: every stage tracked, relative paths, guard (§2.1 item 1; R5) | **Partly met** | Stage runs and guard are fine (static read). **Gaps:** (a) feature manifests write absolute `checkpoint` and `dataset_root` (`influence.py` l.219–221) and are logged as attribute-run artifacts; (b) the attribute run logs feature *manifests* but no sha256 of the feature `.npy`/`.meta.npz` it consumed, so a regenerated feature file is indistinguishable; (c) `stage.run_id` resolves a name to "the latest FINISHED run" across all chains (l.66), so a re-submitted or partially re-run chain can silently pick up an older chain's run; (d) `stage.py` cannot target a scratch store (no `--lineage-root`); (e) rollouts that lack trajectories are dropped silently (`attribute` l.290 checks returns ⊇ features, not features = all rollouts) → S3, S6 |
| C16 | Older verifiers still pass after the branch's shared-file edits (R5) | **Met** | Measured: `verify_lineage_s3`, `verify_runs_s5 --r2`, `verify_blind_s0`, `verify_method_boundary_s3`, `verify_derive_gt_s2`, `verify_verdict_s7`, `verify_returns_s6 --synthetic`, `verify_metrics_s1`, `verify_consumers_s6` all rc=0 (`data/scratch/regress/summary.txt`); real `mlflow.db` sha256 unchanged (`306af412…`) |
| C17 | Injection recovery measured for the attribution method (§2.1 item 4; §4.5 metric-targeted injections; ground-truth limitation D) | **Missing** | No stage scores an attribution table against the answer key. `groundtruth.score_run` exists but is not wired; the chain's default base subset is clean `v0/candidate`. Nothing compares against each type's best heuristic (`ground-truth/artifacts/s5_v3_r3_bands.csv`, measured: 6 of 7 types at AUROC ≥ 0.95, `time_shift` 0.848) → S5 |
| C18 | Four groups, heatmap, throughput-vs-reliability scatter, aggregation by catalog column (§4.2, §4.5, §1 p.3, §9.2) | **Missing** | No code produces them (measured: grep of `vlapoc/` and `scripts/` for heatmap/quadrant/consistent_helper returns nothing) → S7 |
| C19 | Server config separable (brief lead 10) | **Partly met** | `slurm/env.sh`, `gpu.sbatch`, `cpu.sbatch` hardcode `/raid/chem/gor/...`, `/auto/home/gor/...`, partition `a100`, an FFmpeg lib path; `verify_attribution.py` l.186–205 hardcodes cluster paths → S6 |
| C20 | Scratch discipline (CLAUDE.md) | **Partly met** | `verify_attribution.py` uses `tempfile.TemporaryDirectory()` with no `dir=`, i.e. `/tmp` (l.122) → S6 |

Headline (evidence as tagged per row), over 20 rows:
- **Met, 8:** C1, C2 (one recording gap, fixed in S3), C3, C6, C12, C13, C14, C16.
- **Partly met, 7:** C4, C5, C7, C8, C15, C19, C20.
- **Divergence awaiting the owner, 1:** C11.
- **Missing, 4:** C9, C10, C17, C18.

Measured for C18: `grep -rn -i 'heatmap\|quadrant\|consistent_helper' vlapoc/ scripts/` returns nothing. Measured
for AC4.5: `v0/candidate` has 20 tasks × 48 episodes.

## Prior art

| Candidate | What it covers | What it misses | Evidence |
|---|---|---|---|
| CUPID released code (`agiachris/cupid` @ `2941eba9`, MIT) | Return-weighted aggregation, TRAK featurization with a pluggable model-output function (`"square"`/`"ddpm"`), ±1 returns | Built on the diffusion-policy codebase (`DiffusionUnet*Policy` only, `modelout_functions.py`); uses `trak`'s dense projector over full parameters; no SmolVLA or flow matching; no metric-resolved returns | measured: repo tree and files read in this session |
| `trak` (MadryLab) | TRAK scores; custom output via `AbstractModelOutput.get_output` / `get_out_to_loss_grad` | Dense JL projection over all parameters (CUPID's d = 4000 route); no Kronecker per-layer projection; no return weighting | quoted: trak docs "Defining a custom model output" page |
| `kronfluence` | EK-FAC/KFAC influence on `nn.Linear`/`nn.Conv2d` via hooks; pairwise scores; `compute_measurement` hook | Different Hessian (EK-FAC, not CUPID's projected Gauss-Newton); the docs do not cover random projection or return-weighted measurements; adopting it would change the estimator away from CUPID's | quoted: `DOCUMENTATION.md` "we support `nn.Linear` and `nn.Conv2d`"; nothing on projection |
| LogIX / LoGra (Choe et al. 2024) | Kronecker-structured per-layer projection P_i ⊗ P_o, which is exactly what `KroneckerProjector` does | The LogIX README never names LoGra; "under active development", no releases; HF/Lightning trainer patching, not LeRobot | quoted: arXiv 2405.13954 abstract; LogIX README |
| `dattri` | TRAK/TracIn/IF building blocks (iHVP, random projection) | No Kronecker projection, no return weighting | quoted: arXiv 2410.04555 |
| submitit | Python submission of Slurm jobs and arrays | The examples doc does not cover `afterok` chains; site options via `slurm_*` parameters | quoted: `submitit/docs/examples.md` |
| Snakemake + `snakemake-executor-plugin-slurm` | DAG workflows on Slurm | A second workflow system next to MLflow stage runs; not verified further | search results only (lead, not fact) |
| LeRobot `lerobot-train` / `lerobot-eval` | Training and eval entry points | Already used: `vlapoc.lerobot_train_entry` calls LeRobot's `train()`; the eval layer owns rollouts | quoted: LeRobot SmolVLA/LIBERO docs; repo code |

**Verdict: keep the custom estimator; make it a faithful CUPID port.** No library implements CUPID's
return-weighted influence on a flow-matching VLA. The two that come closest would each change the estimator, either
in its Hessian (kronfluence) or in its projection cost (trak's dense projection over ~450M parameters vs Kronecker
r² per layer, the ATHENA-cited route; inferred cost argument). The custom code's math core is small
(`influence_matrix` and `performance_influence`, about 40 lines) and is already checked against a brute-force sum
(measured 7.4e-16). The defect is not that the code is custom. It deviates from CUPID's output function, and the
divergences are under-recorded (S2). For orchestration, the hand-written `server_pipeline.py` (161 lines over
`sbatch --dependency=afterok`) is kept. submitit would replace roughly 20 lines of `subprocess.run(["sbatch", ...])`
and does not document dependencies, and Snakemake would add a second workflow system beside MLflow stage runs, which
conflicts with the single-record-system rule (inferred). Recorded as the trade: no retries or resubmission, which
those tools provide.

## Stories

Every story below can be verified on this Mac (CPU or MPS, synthetic data, scratch lineage copies, `--dry-run`).
None of them needs the server.

### S1: Route every method input through `open_for_method`
- **Why:** C5; rule R1; brief lead 1 (the ground-truth feature's known issue that `train_run`/`eval_run` bypass the
  loader).
- **Depends on:** —
- **Deliverables:**
  - `attribution.attribute_run` replaces `assert_blind` with `open_for_method(<training subset id>)`.
  - `stage.py features --which train` takes its root and episodes from the returned `MethodView`.
  - `stage.py train` (the attribution chain's trainer) calls `open_for_method` on the subset before `train_run`.
    `train_run` itself is left alone, because the ground-truth feature's oracle arms legitimately train on
    `oracle_drop` subsets; the check sits at the method-chain entry.
- **Acceptance criteria:**
  - [ ] AC1.1 On a scratch lineage copy, `attribute_run` with a train run whose training subset is on fenced v1 (or
    v2) raises `LineageError` naming the fence before any MLflow run is created (run count unchanged) [PoC §2.1 item 4 p.3; owner decision 3]
  - [ ] AC1.2 Same for a training subset whose derivation chain contains `oracle_drop` ("ground-truth-derived") [owner decision 3]
  - [ ] AC1.3 `stage.py features --which train` and `stage.py train` refuse the same two cases. Test with a
    `--dry-check` flag, or by calling the helpers they use, without loading a model [PoC §2.1 item 4 p.3]
  - [ ] AC1.4 A version whose file was tampered with (one flipped byte in a scratch copy) raises `VersionIntegrityError` from the attribute path [PoC §2.1 item 1 p.3]
  - [ ] AC1.5 `verify_method_boundary_s3.py` still passes. The boundary scan (no `groundtruth` import in method code) also covers `vlapoc/influence.py` and `vlapoc/attribution.py` [ground-truth feature S3]
  - [ ] AC1.6 Grounding docstrings cite PoC §2.1 item 4 p.3 and owner decision 3 [vla-build]
- **Verify by:** a new `scripts/verify_training_attribution.py --blind` run on `runs.scratch_lineage_copy` (v1 fence
  and a 20-episode oracle subset, built the way `verify_method_boundary_s3.py` builds them), plus
  `shasum -a 256 data/lineage/mlflow.db` before and after.
- **Demo value:** a refusal table: fenced, ground-truth-derived and tampered inputs, each refused at four entry
  points (attribute, features, train, select).

### S2: Make the estimator CUPID's (output function), and record every remaining divergence
- **Why:** C4; brief lead 2; PoC §4.3 p.10 "The estimator is CUPID's"; §6 p.12; CUPID App. A.1–A.2.
- **Depends on:** —
- **Deliverables:**
  - `influence.featurize(..., output_fn=...)` with `"square"` = E‖v_θ(x_t, s, t)‖² over the unpadded action entries
    of the first `original_action_dim` dims. Take v_θ from a forward hook on `policy.model.action_out_proj`
    (measured: `modeling_smolvla.py` l.722 `v_t = self.action_out_proj(suffix_out)`), averaged over the K shared
    draws. `"loss"` is the current behaviour.
  - Two parameters, `train_output_fn` and `rollout_output_fn`:
    - Default: CUPID (`square`/`square`).
    - `--athena` preset: `loss`/`square`, ATHENA Eq. 15.
    - `loss`/`loss` reproduces the branch.
  - The manifest records the output function. `check_compatible` refuses mixing two train-side output functions.
    The Hessian is always the Gram matrix of the train-side features (Eq. 9's H_square when `square`).
  - `performance_influence` gains `return_encoding="pm1"` for binary returns (CUPID A.2), with `center=False`.
  - A `DIVERGENCES` constant (dict) in `attribution.py`, logged as a run param, lists each remaining departure from
    CUPID with its source:
    - centring;
    - K = 4 shared draws vs B = 64;
    - Kronecker r = 8 over trainable expert layers vs dense d = 4000 over all parameters (ATHENA §4.1);
    - λ = 1e-3·mean diag vs `lambda_reg=0.0`;
    - strides;
    - fp32 eval-loader numerics (G3b);
    - a corrected centring note: centred = p(1−p)·m·(mean_succ M − mean_fail M), whereas CUPID net = Σ_succ M − Σ_fail M
      = m·[p·mean_succ − (1−p)·mean_fail]. The two agree in ranking only when p = ½ (inferred algebra; AC2.3 checks it
      numerically).
  - The module docstring corrects "1/H dropped" to cite CUPID Prop. 2 (§D.2 p.27).
- **Acceptance criteria:**
  - [ ] AC2.1 On a toy torch module with a `nn.Linear` head on CPU, the `square` features equal the brute-force
    per-sample gradient of ‖head output‖² (masked) projected with the same P_in, P_out: max rel err < 1e-5 (float32
    tolerance, **inferred**) [CUPID App. A.1 Eq. 9]
  - [ ] AC2.2 CUPID parity: on a synthetic pairwise score matrix and success mask, `performance_influence(M,
    2·r01−1, center=False)·m` equals CUPID's `compute_demo_quality_scores(sum_of_sum traj scores, mask, "net")`.
    Test against a verbatim vendored copy of those two functions with the commit hash `2941eba9` and the MIT notice.
    Max abs err < 1e-9 [CUPID released code `error_util.py`]
  - [ ] AC2.3 The centring note's algebra holds numerically: centred = p(1−p)·m·(mean_succ − mean_fail) on synthetic data, and the rankings agree with CUPID's at p = ½ but not in general (a demonstrated counterexample) [CUPID App. A.2; inferred derivation]
  - [ ] AC2.4 CPU or MPS smoke on the local SmolVLA checkpoint
    `data/runs/train-smolvla-2step-persubset-v0ref-20261005T012854Z/.../last/pretrained_model` (measured present,
    1.1 GB):
    - `featurize` with `output_fn=square` and `loss` on ≤ 2 episodes of `v0/reference` (stride ≥ 20) writes features
      of dim 7,488, a manifest recording `output_fn`, and no NaN;
    - two calls agree bit-for-bit (seeded draws);
    - wall time recorded [CUPID App. A.1; brief constraint "CPU/MPS smoke"]
  - [ ] AC2.5 `attribute_run` params include `train_output_fn`, `rollout_output_fn`, `return_encoding`, `center_returns`, `DIVERGENCES`; `check_compatible` raises on mixed train-side output functions [PoC §3.4 p.7 disclosure; vla-build step 4]
  - [ ] AC2.6 Existing checks 1–5 of `verify_attribution.py` still pass (16/16) [regression]
- **Verify by:** `.venv/bin/python scripts/verify_attribution.py` (extended with checks 6–8), and
  `.venv/bin/python scripts/verify_attribution.py --cpu-smoke <ckpt> data/lerobot/libero_spatial_10 --device mps`.
- **Demo value:** a table of every departure from CUPID with its source, and the real-checkpoint feature smoke
  (dims, wall time, the cosine between `square` and `loss` features on the same samples). That cosine measures how
  much the output-function choice changes the gradients.

### S3: Declare the functional of every column, and close the attribute run's provenance gaps
- **Why:** C7, C2 (threshold), C15 (a, b, e); PoC §3.4 p.7, §4.4 p.10–11; §2.1 item 1 p.3.
- **Depends on:** S2 (the declaration includes the output function).
- **Deliverables:**
  - `attribution.column_declarations(table, ret_meta)` → one row per score column: `column`, `return_column`,
    `family` (W), `tau_s`, `rollout_subset` (`all` / `same_task` / `failure_type=<t>`), `good_sign`, `centred`,
    `return_encoding`, `output_fns`.
  - Written as `functionals.json` and logged by both `attribute_run` and `ensemble_run`.
  - `select_run` records the declaration of the metric it ranked on.
  - Feature manifests use `lineage.rel()` paths.
  - `attribute_run` records sha256 of every feature `.npy` and `.meta.npz` it read, as input entities (`kind=features`).
  - It refuses an eval run in which any rollout lacks a trajectory, unless `--allow-partial` is passed, which records the count.
  - `MIN_FAILURE_TYPE_ROLLOUTS` goes into params, labelled inferred.
  - `ensemble_run`:
    - refuses duplicate train runs and inputs with differing `damping`/`center_returns`/output functions;
    - records `n_models` per column (failure-type columns can exist in only some seeds).
- **Acceptance criteria:**
  - [ ] AC3.1 Every `infl_`/`harm_` column of a synthetic score table has exactly one row in `functionals.json`, and the τ in each row equals `metrics_registry`'s τ for that return [PoC §3.4 p.7; §4.4 "label every claim with its metric and time cap τ"]
  - [ ] AC3.2 Every `"checkpoint"`/`"dataset_root"` value in the manifests written by the S2 smoke is repo-relative (`rel()` round-trip; no leading `/`) [PoC §2.1 item 1 p.3; R5]
  - [ ] AC3.3 In the S8 integration run, the attribute run's inputs list one `features` entity per shard file, and changing one byte of a feature file changes the recorded digest [PoC §2.1 item 1 p.3]
  - [ ] AC3.4 Dropping one rollout's trajectory makes `attribute` raise; `--allow-partial` passes and records `n_rollouts_without_trajectory` [PoC §2.1 item 1 "every eval rollout tied back"]
  - [ ] AC3.5 `ensemble_run` raises on [id, id] duplicates and on mixed `center_returns`; `n_models` per column is recorded [PoC §2.3 p.4 "2–3 seeds per data configuration"]
  - [ ] AC3.6 The select run's params carry the ranked column's declaration [PoC §3.4 p.7]
- **Verify by:** `scripts/verify_training_attribution.py --functionals` (synthetic), plus the S8 integration run.
- **Demo value:** `functionals.json`: one line per attribution column stating which functional it explains (metric,
  τ, rollout subset, sign).

### S4: Complete Step 4: helpful arms, curated-vs-random verdicts, opposite curations, curation-mode switch, seed summary
- **Why:** C8, C9, C10, C11; PoC §9.1 Step 4.1–4.5 p.19, §2.6 p.5, §4.5 p.11.
- **Depends on:** S3 (declarations travel with each arm), and the owner's answer to Q1 for the default mode.
- **Deliverables:**
  - In `server_pipeline.arms()`:
    - helpful arms: `help-<m>-k<k>` = `drop_top_k_per_task` on `harm_<m>__same_task` with `--direction lowest`;
    - `hrt60_lin` added to the default metrics.
  - `--curation {per_task,global,both}`. `global` uses `drop_top_k` on the all-rollout `harm_<m>` column with a
    size-matched `drop_random_k` control, because `__same_task` columns are on per-task scales and cannot be ranked
    globally (inferred).
  - Extra verify jobs per seed:
    - `harm` arm vs its matched random arm;
    - `help` arm vs its matched random arm;
    - `harm-hrt60_lin` vs `harm-f30`, the §4.5 head-to-head.
  - A new CPU stage `stage.py step4-summary --tag <t>` reads every verify run of the chain and writes one table. It
    has one row per (arm, comparison, family), with:
    - the outcome and Holm-corrected direction per metric, per seed;
    - a seed-consistency count, e.g. "harmful removal beat random on F30 in 2/3 seeds";
    - the declared functional (S3).
  - It is recorded as a tracked run.
- **Acceptance criteria:**
  - [ ] AC4.1 `server_pipeline.py --tag t --dry-run` lists `help-*` arms and, per seed, a harm-vs-random and a help-vs-random verify job whose dependencies are both arms' eval jobs (parsed from the dry-run output, not eyeballed) [PoC §9.1 Step 4.1 and 4.5 p.19; §2.6 p.5]
  - [ ] AC4.2 The dry-run contains an `hrt60_lin` harm arm and a verify job `harm-hrt60_lin` vs `harm-f30` per seed [PoC §4.5 p.11 "same dataset, opposite curations"]
  - [ ] AC4.3 `--curation global` emits `drop_top_k --metric harm_<m>` (no `__same_task`) with `drop_random_k` of the same k. `--curation both` emits both families. The default follows the owner's answer to Q1 and is recorded in each select run's params [PoC §9.1 Step 4.1 p.19; ATHENA §3.2 p.3]
  - [ ] AC4.4 `step4-summary` on synthetic `verdict.json` files (three seeds, labelled synthetic) produces the expected per-arm table, including a planted case where curated beats random in 2/3 seeds; it refuses verdicts from another chain [PoC §2.6 p.5; §4.4 p.10]
  - [ ] AC4.5 All selects of one chain keep the parent's per-task counts where the mode is per-task (re-checked on the real `v0/candidate` manifests in a scratch store with a synthetic score table: 48 → 43 per task for k = 5) [PoC p.1 "data is the only variable"]
  - [ ] AC4.6 The total job count and arm list from the default dry-run are recorded in the execution log [user request: honest record]
- **Verify by:** `.venv/bin/python scripts/server_pipeline.py --tag t --dry-run --curation both > data/scratch/dry.txt`
  parsed by `scripts/verify_training_attribution.py --pipeline data/scratch/dry.txt`; the synthetic
  `step4-summary` run on a scratch lineage copy.
- **Demo value:** the full Step 4 experiment design as the chain will run it. It shows every arm, every comparison,
  and which PoC sentence each one tests.

### S5: Score attribution against the planted ground truth (recovery per corruption type × metric)
- **Why:** C17; PoC §2.1 item 4 p.3, §4.5 p.11 (metric-targeted injections), §10 "Ground truth"; ground-truth
  limitation D (beat each type's best simple heuristic).
- **Depends on:** S3.
- **Deliverables:**
  - New subcommand `stage.py score --scores <attribute run> --version v3`, wrapping `groundtruth.score_run` for every
    `harm_` column (direction `highest`).
  - It writes `recovery_matrix.csv`: corruption type × harm column → AUROC, recall@k (k = number injected, as in S5
    of the ground-truth feature).
  - Each row adds that type's best heuristic AUROC from
    `tracker/features/2026-10-15-ground-truth/artifacts/s5_v3_r3_bands.csv` and the difference.
  - The expected-metric map of §4.5 (idle_padding → rmst60/hrt60_lin; gripper_release → f30/censored60) is declared
    in advance in code as a hypothesis, and the output reports whether each type's best column matches it.
  - `server_pipeline.py` adds the score job after the ensemble when the base version has an answer key.
  - The runbook text (the docstring) recommends `--subset v3/candidate` for the injection run.
- **Acceptance criteria:**
  - [ ] AC5.1 On a scratch lineage copy with the real v3 labels and a **synthetic** score table in which planted
    columns rank one type's episodes first:
    - that type's AUROC = 1.0 in that column;
    - a random column gives AUROC in [0.4, 0.6] (**inferred** band for 91 positives of 960);
    - the real `mlflow.db` is unchanged [PoC §2.1 item 4 p.3]
  - [ ] AC5.2 `recovery_matrix.csv` has one row per (type, column), with `best_heuristic_auroc` equal to the bands CSV for every type (7/7) [ground-truth limitation D]
  - [ ] AC5.3 The expected-metric map lives in code before any real scores exist, and the output marks `matches_hypothesis` per type [PoC §4.5 p.11 "These mappings are hypotheses the experiment should confirm"]
  - [ ] AC5.4 The dry-run with `--subset v3/candidate` contains one `score` job depending on the ensemble job; with `v0/candidate` it contains none [PoC §2.1 item 4 p.3]
  - [ ] AC5.5 `score` refuses a score table keyed to another version (`check_score_keys`) [PoC §9.2 p.19]
  - [ ] AC5.6 The scoring code imports `vlapoc.groundtruth` only in the `score` stage path, never in `influence.py`/`attribution.py` (boundary scan) [owner decision 3]
- **Verify by:** `scripts/verify_training_attribution.py --recovery` on `runs.scratch_lineage_copy`, with the
  mlflow.db sha256 checked before and after.
- **Demo value:** the recovery table layout the server run will fill in (type × metric AUROC next to the heuristic to
  beat), shown on labelled synthetic scores.

### S6: Make the chain testable and portable: `--lineage-root`, chain-scoped names, site config, scratch dirs
- **Why:** C15 (c, d), C19, C20; brief leads 9 and 10; R7.
- **Depends on:** —
- **Deliverables:**
  - `stage.py --lineage-root` (global) threaded to every call.
  - `run_id(ref, stage, chain_prefix)` resolves names only among runs whose `vlapoc.chain` starts with `<tag>-`. It
    raises if two FINISHED runs share the name in that scope, unless `--latest` is passed.
  - `slurm/site.env.example` (tracked) holds `PHYEXP_DIR`, `SCRATCH_ROOT`, `PARTITION`, `LOG_DIR`, `FFMPEG_LIB`.
    `slurm/site.env` is the gitignored per-machine copy.
  - `env.sh` sources it and fails with a clear message when it is missing.
  - The sbatch files drop every absolute path. `#SBATCH --output` cannot expand variables, so `server_pipeline.py`
    passes `--output`/`--partition` from the site config.
  - `--dry-run` prints the full `sbatch` argv.
  - `verify_attribution.py` moves cluster paths to CLI arguments and uses `dir=REPO_ROOT/"data"/"scratch"` for temp dirs.
- **Acceptance criteria:**
  - [ ] AC6.1 `git grep -n -E '/raid|/auto/home' -- slurm scripts vlapoc` returns nothing [brief lead 10]
  - [ ] AC6.2 `bash -n` passes on every `slurm/*.sh` and `*.sbatch`. `env.sh` without a `site.env` exits non-zero with a message naming `site.env.example` (run in a subshell) [brief lead 10]
  - [ ] AC6.3 On a scratch lineage store with two FINISHED train runs named `train-t-s0` (chains `t-aaaa` and `u-bbbb`), `run_id('train-t-s0', 'train', 't-')` returns the `t-` one. Two in scope raise unless `--latest` [PoC §2.1 item 1 p.3]
  - [ ] AC6.4 Every `stage.py` subcommand accepts `--lineage-root`, and S8 runs entirely on a scratch store with the real mlflow.db sha256 unchanged [R7]
  - [ ] AC6.5 The dry-run output shows `--output=<LOG_DIR>/...` and `--partition=<PARTITION>` taken from `slurm/site.env.example` values [brief lead 10]
  - [ ] AC6.6 `verify_attribution.py` creates no file outside `data/scratch/` (check: `TMPDIR` set to an empty dir under `data/scratch/`, which is still empty afterwards; `tempfile` calls use explicit `dir=`) [CLAUDE.md scratch rule]
- **Verify by:** the commands in the ACs; `scripts/verify_training_attribution.py --names`.
- **Demo value:** the chain can be submitted from any cluster account by copying one example file, and it cannot pick
  up another chain's runs.

### S7: Metric-resolved outputs: four groups, heatmap, throughput-vs-reliability scatter, aggregation by catalog column
- **Why:** C18; PoC §4.2 p.9–10 (four groups), §4.5 p.11 (visualisation), §1 p.3 and §2.2 p.4 (aggregate by
  metadata), §9.2 p.19.
- **Depends on:** S3.
- **Deliverables:**
  - `attribution.metric_groups(table, columns, z=...)` labels each episode `consistent_helper` / `consistent_harmer`
    / `trade_off` / `metric_specific` / `neutral` from the ensemble's z-scored `harm_` columns. The threshold is a
    declared parameter (**inferred**, default |z| ≥ 1). Labels are "descriptive, unverified": §4.4 requires Holm
    before calling an episode a trade-off, and per-episode significance is not available from 3 seeds. The verified
    claim lives at intervention level (S4's verdicts). This is recorded as a divergence.
  - `aggregate_by(table, catalog_column)` gives the mean, CI over episodes, and n per group, on any `CATALOG_COLUMNS`
    entry.
  - `plot_heatmap` (episode × metric, sorted by group) and `plot_tradeoff` (x = throughput influence `harm_rmst60`
    or `harm_hrt60_lin`, y = reliability influence `harm_f30`, quadrants labelled with the four groups).
  - An `analyze` stage in `stage.py` writes these as attribute-run-linked artifacts.
- **Acceptance criteria:**
  - [ ] AC7.1 On a planted synthetic table, every planted helper, harmer, trade-off and metric-specific episode gets its label (4/4 groups, all planted episodes correct) [PoC §4.2 p.10]
  - [ ] AC7.2 `aggregate_by(…, "task_name")` returns one row per task with n summing to the table's rows; it refuses a non-catalog column [PoC §1 p.3; §2.2 p.4]
  - [ ] AC7.3 The heatmap and scatter PNGs are written. Each axis label names the column's functional and τ from `functionals.json`, and the titles say "SYNTHETIC" when the input is synthetic [PoC §4.4 "label every claim with its metric and τ"; §4.5 p.11]
  - [ ] AC7.4 The group threshold and the "descriptive, unverified" status are in the analyze run's params [PoC §4.4 p.10 multiple comparisons]
- **Verify by:** `scripts/verify_training_attribution.py --groups` (synthetic, labelled).
- **Demo value:** the §4.5 figures (heatmap and quadrant scatter), clearly labelled synthetic until the server run
  fills them.

### S8: Run the chain end to end on this Mac (synthetic integration and a real toy-scale smoke)
- **Why:** C6 (the contracts are only unit-tested with monkeypatching today), C15; brief constraint ("separate what
  can be re-verified here"); demonstrates S1–S7 working together on real stage runs.
- **Depends on:** S1–S7.
- **Deliverables:**
  - (a) A synthetic integration test in `scripts/verify_training_attribution.py --integration`, on
    `runs.scratch_lineage_copy`:
    - fabricate a train run with `stage_run("train")`, a 20-episode training subset of v3/candidate and a fake
      checkpoint sha;
    - fabricate an eval run with trajectories tags and a returns run;
    - write synthetic feature shards with matching manifests;
    - then run `stage.py attribute` → `ensemble` (two fabricated seeds) → `select` (harm, help, random) → `score`
      → `analyze` through the CLI with `--lineage-root`.
  - (b) A real toy-scale smoke `--local-smoke` on a scratch store:
    - reuse the tracked SmolVLA train run `0a33bc75bb…` (`train-smolvla-2step-persubset-v0ref`, measured in the DB
      copy; checkpoint present locally);
    - `stage.py eval --smoke` with trajectories on 1 task × 2 init states on MPS;
    - then `returns`, `features` (train stride ≥ 20, rollouts stride ≥ 10, `--device mps|cpu`), `attribute`, `select`.
- **Acceptance criteria:**
  - [ ] AC8.1 (a) passes. `trace` on the select run reaches the attribute run, both feature digests, the returns run, the eval run and the training subset [PoC §2.1 item 1 p.3]
  - [ ] AC8.2 (a) negative cases, each refused with nothing written:
    - features from another checkpoint;
    - train features on a different episode list;
    - a score table from another subset at `select`;
    - an ensemble of two runs on different subsets [PoC §9.2 p.19]
  - [ ] AC8.3 (b) completes, or the first failing stage and its error are recorded verbatim. When it completes, it reports n_rollouts, n_episodes, the score table shape, and τ agreement, all labelled "smoke (not attribution-grade)" [brief constraint: CPU/MPS smoke]
  - [ ] AC8.4 Both runs leave the real `data/lineage/mlflow.db` sha256 unchanged (`306af4123fda7e63…`) [R7]
  - [ ] AC8.5 The older verifiers listed in C16 still exit 0 after all stories land. Tracked artifacts in other feature folders that those verifiers rewrite are restored, not committed (see Risks) [regression; R5]
- **Verify by:** `.venv/bin/python scripts/verify_training_attribution.py --integration` and `--local-smoke --device mps`.
- **Demo value:** the first time Step 3 → Step 4 selection runs on real recorded rollouts and a real SmolVLA
  checkpoint in this repo, with the lineage trace from a curated subset back to the rollouts that justified it.

## Out of scope (needs the server)

Dismissed by the owner's ruling. Listed once, so the report stays honest:
- AC8.5 training at 20k steps.
- AC5.8 attribution-grade evals and AC6.7 returns at full scale.
- Feature extraction on real 20k-step checkpoints.
- The October cluster regressions (A6 3.2e-13, A7 GPU smoke), which stay as the branch's server-side evidence only.
- Re-computing the October scores after the time-seeding fix.
- Running the SLURM chain for real.
- Verdicts on retrained arms; the "same dataset, opposite curations" result; real recovery numbers on v3.
- SQLite-lock behaviour under concurrent SLURM jobs.
- The `smolvla_base` revision discrepancy on the cluster cache (`5e8d12a6` vs `d9f33c94`).

## Out of this feature's domain (owner-visible PoC gaps)

| PoC item | Why not here |
|---|---|
| §9.1 Step 1 screening (DemInf + RoboDrop warm-up gradients), §9.2 screening-vs-attribution disagreement, §8.10 RoboDrop comparison, §10 "add TRAK or RoboDrop-style scores; compare against DemInf" | A screening feature. Nothing on the branch touches it; `screen` is still a stand-in (`runs.py` docstring) |
| §10 "Causal baseline: group-level leave-one-out ablations" and §2.1 item 5 datamodels | `drop_group` exists (lineage feature), but LOGO arms and a datamodels regression are a separate causal-baseline feature |
| §10 "attribute against ghost-event rate" | Ghost events are out of scope by owner decision |
| §2.7 / §10 dashboard linking failed rollouts to influential episodes | A demo-surface feature. S7's tables and figures are its inputs |
| §3.5 per-sub-task T for LIBERO-10 | Evaluation-layer owner story T, not attribution |
| §10 Real-robot stage (PhAIL station), Scale-up (π0.5/GR00T with ATHENA) | Need hardware or compute beyond the PoC milestone |

## Open questions

- **BLOCKING Q1 (owner): the curation unit for Step 4.**
  - The facts: the PoC says "Remove the top-k most harmful episodes" (§9.1 Step 4.1, p.19), which is global. The
    branch drops k per task with per-task matched random controls, grounded in ATHENA §3.2/§4.2 (quoted above) but
    never approved.
  - The options:
    - **(a)** keep per-task as the declared default, recorded as a divergence;
    - **(b)** switch to the PoC's global top-k;
    - **(c)** run both (`--curation both`). This roughly doubles the arms on the server (dry-run counts in S4).
  - Planner recommendation: **(c) for the first server run, with (a) as the headline**. Per-task keeps the task
    mixture fixed, so the arm differs from the original only in which episodes it holds. Global is the PoC's literal
    test. S4 implements all three; only the default and the report's headline depend on the answer.
- Non-blocking assumptions (the executor proceeds on these; the owner may override):
  - N1. Default output function = CUPID's `square`/`square`, with ATHENA's `loss`/`square` and the branch's
    `loss`/`loss` selectable. Why: PoC §4.3 p.10 "The estimator is CUPID's"; CUPID App. A.1 rejects the loss form.
    This changes what the server will compute next, so the report flags it.
  - N2. Centred returns stay the default (an unbiased baseline subtraction: E_τ[b·Σ∇log π] = 0, inferred). CUPID's
    ±1 is available, and the divergence and its algebra are recorded (S2).
  - N3. `hrt60_lin` joins the default curated metrics so the §4.5 head-to-head exists; this adds arms (cost lands on
    the server run).
  - N4. The injection run's recommended base subset is `v3/candidate` (blind, measured: no fence file for v3; v1 and
    v2 fenced). `v0/candidate` stays the AC8.5 default.
  - N5. The four-group labels are descriptive (S7). Verified claims come only from S4's intervention-level verdicts.

## Risks / premises to verify during execution

- P1. The `action_out_proj` hook captures v_t in the K-draw loop. Premise: SmolVLA 0.6.1 calls `action_out_proj`
  once per training forward (measured: l.722 in `forward`; l.809 sits in `denoise_step`, defined at l.775, the
  inference path, which `policy.forward` does not hit). Verify with a call counter in AC2.1/AC2.4.
- P2. `KroneckerProjector` assumes every hooked `Linear` input has batch as dim 0. The cluster run's 7,488-dim
  features imply it held (quoted from the branch log, not re-measured). AC2.4 re-checks it on CPU.
- P3. `open_for_method` hashes the whole version (`verify_version`, about 20 s for v0, quoted from the boundary
  verifier docstring). Acceptable per job; measure it in S1.
- P4. **Several existing verifiers rewrite tracked artifacts in other feature folders** (measured this session:
  `s3_summary.json`, `s0_verify_blind.json`, `s6_v3_verify.json`, `s5_r2_summary.json`). The executor must restore
  them (`git checkout -- <file>`) and must not commit them under this feature. Copies are in
  `data/scratch/regress/artifacts_rewritten/`.
- P5. `verify_attribution.py` prints numpy `RuntimeWarning: divide by zero/overflow in matmul` on this Mac while its
  results pass at 7.4e-16 (measured). The cause may be spurious FP flags in the system BLAS (inferred, unverified).
  The executor checks that the brute-force H is finite (`np.isfinite(h).all()`) so the warning is not hiding a real
  overflow.
- P6. MPS eval of SmolVLA for S8(b) may be slow or unsupported in places. AC8.3 accepts a recorded failure point
  rather than a fabricated success.
- P7. The local `.venv` is Python 3.14, while the runbook pins 3.12/3.13 for `lerobot-train` (quoted, runbook §0).
  S8(b) reuses an existing train run, so no `lerobot-train` CLI call is needed.

## Commands run by the planner this session (reproducible)

- `.venv/bin/python scripts/verify_attribution.py` gives `pass: True (16/16)`, check 1 max rel err 7.42e-16, with
  numpy matmul RuntimeWarnings (P5).
- `.venv/bin/python scripts/server_pipeline.py --tag v0cand --dry-run` gives "would submit 74 jobs"; arms
  `harm-f30-k5, harm-rmst60-k5, harm-auc_expected-k5, random0-k5, random1-k5, drop-random-20`
  (`data/scratch/dryrun.txt`).
- Regression verifiers: rc and seconds per script in `data/scratch/regress/summary.txt`, logs alongside. All exit 0.
  The real `mlflow.db` sha256 before and after is `306af4123fda7e635892d459a2e34008ba1cb62f6c4fce152c546e0da7bdde79`.
- CUPID source: `git clone --depth 1 https://github.com/agiachris/cupid.git data/scratch/cupid_src/repo` → `2941eba9`.
- Runs in the DB (read from a copy, `data/scratch/dbq/mlflow.db`): train `0a33bc75bb` `train-smolvla-2step-persubset-v0ref`
  FINISHED; no tracked SmolVLA eval run with trajectories exists for a tracked train run (the two harness evals with
  trajectories use the external `smolvla_libero` checkpoint).
