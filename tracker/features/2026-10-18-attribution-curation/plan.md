# Plan: attribution, curation and server orchestration (vla-build, single pass)

Labels: **quoted** (from the PoC or a paper), **measured** (with the artifact), **inferred** (a design choice).

## Grounding (PoC, docs/VLA_Data_Attribution_PoC.pdf)

| § | Page | Quoted | Where |
|---|---|---|---|
| §4.2 | 9 | "Instead of one influence score per episode, the PoC computes an influence vector: one score per metric ... Results are reported as an episode × metric matrix." | `attribution.score_table` |
| §4.3 | 10 | CUPID-style influence on returns "with one return definition per metric" | `attribution.performance_influence` over `RETURN_COLUMNS` (family W) |
| §4.4 | 11 | "Agreement check. Report rank correlation (e.g. Kendall's τ) between the per-metric attribution vectors" | `attribution.agreement` → `agreement.json` |
| §9.1 Step 3 | 19 | "For each metric, CUPID-style influence estimates how much each training episode pushed that return up or down. Repeat the analysis for subsets, such as a specific task or only rollouts with a specific failure type." | `__same_task` columns; `infl_ft_<type>` |
| §9.1 Step 4 | 19 | Retrain without the selected episodes and compare against the original policy; random-k control | `drop_top_k_per_task`, `drop_random_k_per_task`, `scripts/server_pipeline.py` arms + `verify` |
| §2.1 item 4 | 3 | methods under test stay blind | `attribute_run` calls `assert_blind` first |
| §2.3 | 4 | "2–3 seeds per data configuration" | 3 seeds; `ensemble_run` |

CUPID (arXiv 2506.19121) §5.1 Prop. 1 as quoted in `vlapoc.returns`; TRAK's action influence φ'ᵀ(ΦᵀΦ/N + λI)⁻¹φ.
ATHENA (arXiv 2606.16208): projection and parameter subsets to scale influence to a VLA.

## Prior art (checked, reused)

- `vlacur.influence` / `vlacur.attribution` / `vlacur.curate` (cluster, Oct 2026): ran on lerobot/libero_10 for
  3 seeds. Ported with the estimator unchanged (verified bit-for-bit below); the I/O is rewritten around this repo.
- `vlapoc.rollouts.checkpoint_policy` (the eval stage's loader) loads checkpoints for featurizing, so a feature and
  a rollout see the same policy and processors.
- `vlapoc.rollouts` `record_trajectories`: rollouts are already a LeRobot dataset in the training schema, so the
  cluster code's camera-rename step for rollouts is not needed.
- `vlapoc.lineage.drop_top_k` / `drop_random_k` / `_check_score_keys` / `score_table_digest`: the per-task ops are
  the same ops grouped by (suite, libero_task_id).

## Design choices (inferred, recorded in run params)

1. **Centred returns.** Ψ_R(ξ) = (1/m) Σ_τ (R − R̄) M[τ, ξ] instead of Prop. 1's raw R; `--no-center` gives Prop. 1.
2. **Rollout gradients summed over steps** (CUPID's 1/H dropped): family-W returns are whole-rollout quantities.
3. **Per-task drops with matched per-task random controls**, so the task mixture does not change between arms.
4. **Seed ensemble = mean of per-seed z-scores** (TRAK ensembling over independently trained models).
5. **Features are not a lineage entity** (like the HF datasets cache): the attribute run logs their manifests,
   and refuses features from another checkpoint or another episode set.

## Acceptance checks

- A1 `influence_matrix` equals the brute-force double sum (synthetic).
- A2 `center=False` is Prop. 1; centring = raw − R̄·mean row.
- A3 `__same_task` = per-task computation; `harm_` signs; failure-type columns only with ≥ 5 rollouts.
- A4 ensemble = mean z; misaligned tables refused.
- A5 per-task ops: k per task, top-k correct with ties, seeded, matched counts, stale table and emptying k refused.
- A6 regression: the ported `influence_matrix` reproduces the cluster's `rollout_demo_influence.npy` (clean_full_s0).
- A7 GPU: `featurize` loads a cluster checkpoint via `checkpoint_policy` and reproduces the cluster's features at the
  same (episode, frame).
- A8 `scripts/server_pipeline.py --dry-run` builds the full chain.
- Pending server (needs v0 + the lineage store, which live on the owner's Mac): the chain itself.
