VERDICT: PASS
ROUTE: none
ROUND: 2

<!-- Saved verbatim by the orchestrator from the analyzer's returned text: the analyzer's own
     write was refused by the harness ("Subagents should return findings as text"). -->

## Summary (3–5 lines)
All five round-1 blocking items are fixed. I checked each one in the code and by re-running the verifiers on HEAD `520be42`:
- The Step 4 summary is now gated on the macro-KS test.
- `LOG_DIR` is created before any real `sbatch`.
- Feature files are bound to the training version's root and to the eval's recorded trajectories.
- The stale `--estimator` references are gone.
- `--functionals` runs with no `--manifest`.

All 18 verifiers exit 0 and the real `mlflow.db` sha256 is unchanged. What remains is non-blocking for the PoC:
- a one-line migration note for pre-round-2 feature files;
- the NB4 rationale text;
- the owner's cost and centring decisions, carried from round 1.

## Rebuttals received
| From | Point disputed | Their evidence | Conceded? | Why / why not |
|---|---|---|---|---|
| executor | NB4: keep the ≥2-of-5 rule instead of PoC §4.2's "every metric" | Inferred only: "the five family-W returns are partly redundant ... requiring all five would leave 'consistent' nearly empty" | Not a dispute | My round-1 NB4 offered "record as a divergence **or** require all non-neutral calls". They chose to record it (`analysis.DIVERGENCE_NOTE`, the docstring, and the analyze run's param `group_rule_divergence`), which is acceptable for labels already declared descriptive and unverified. **I do not accept the rationale** (see Non-blocking 2): redundancy argues the other way, and nothing was measured. Non-blocking, so no escalation. |

## Round-1 items
| # | Item | Status | Evidence (all re-run by me) |
|---|---|---|---|
| E1 | `step4-summary` counts directions when the macro-KS gate did not fire | **Fixed** | `step4.call(..., gate_fires)` returns `g` when the gate did not fire (step4.py l.67–78), wired from `v["gate"]["fires"]`. `--step4` → `S4 summary/selects pass: True (13/13)`, including `[PASS] 4.4h ... by seed g,g,g, supports 0, gate_fires 0/3` and `[PASS] 4.4i ... by seed C,g,C`. I reran my own round-1 probe with mixed seeds (fired/not, T/C): `g,T,C`, T=1, C=1, supports 1/3, "gate not fired in 1/3". Measured. |
| E2 | `LOG_DIR` not created before `sbatch` | **Fixed** | `Chain.submit` runs `Path(self.log_dir).mkdir(parents=True, exist_ok=True)` before every non-dry `subprocess.run` (server_pipeline.py l.115). `site.env.example` now says `LOG_DIR` must be on a filesystem the compute nodes share. `--names` → `S6 pass: True (13/13)`, with `[PASS] 6.7 ... dir existed at the sbatch call: True` (stubbed `subprocess.run`, nothing submitted). |
| E3 | Feature files not bound to their data | **Fixed** | `attribute_run` (attribution.py l.576–589) refuses three cases: train manifests whose `dataset_root` ≠ `rel(open_for_method(subset).lerobot_root)`; rollout features whose root's `sha256_json(file_digests(...))` ≠ the eval's `output.trajectories`; and any shard with another checkpoint (now `any()` over all shards, not only the first). `--integration` → `S8(a) pass: True (12/12)`, with two new negatives, each refused with nothing written: "train features were computed on ['data/lerobot/libero_spatial_10'], not on the training subset's version copy" and "rollout features were computed on [.../evals/s1/trajectories], which is not the trajectories copy eval run ... recorded". The writer side uses the same `rel()` of the same `view.lerobot_root` (stage.py `cmd_features`), so a correctly produced feature passes. The executor's re-run `--local-smoke` on real objects is logged but **not re-run by me** (about 8 min). |
| E3 consequence | Pre-round-2 feature files (absolute or mismatched `dataset_root`, e.g. October cluster) are refused | **Acceptable; add a note** | They were refused on other grounds already, or are not comparable anyway: `loss/loss` features no longer match the new `square` default; the October scores are listed out of scope ("Re-computing the October scores after the time-seeding fix"); `ensemble_run` refuses pre-S3 attribute runs; and none of them is in this repo's store. Refusing an input whose data cannot be verified is the point of the binding. Missing: nothing in the code or runbook tells a server user what to do (Non-blocking 1). |
| E4 | Stale `--estimator` references | **Fixed** | `grep -rn -e "--estimator" scripts vlapoc` returns nothing. `cupid_quality.py` header now names `verify_attribution.py` check 7a. |
| E5 | `--functionals` exits 1 without `--manifest` | **Fixed** | The bare `--functionals` run gives `[SKIP] 3.2 ... (no --manifest given; see artifacts/s3_manifest_example.json)` and `S3 pass: True (17/17)`, rc=0. |
| NB3 | Ensemble params lacked `DIVERGENCES` | **Fixed** | `ensemble_run` params now carry `aggregation` and `divergences_from_cupid` (attribution.py l.705–706). |
| NB4 | Group rule vs "every metric" | **Recorded as a named divergence** | `DIVERGENCE_NOTE`, the docstring, and param `group_rule_divergence`. The rationale is flawed (Non-blocking 2). |
| NB5 | `step4-summary` without `--chain-id` | **Fixed** | server_pipeline.py l.276 passes `--chain-id {ch.chain}`. |
| NB7 | `analyze_run` catalog read | **Fixed** (docstring) | Explanation added; boundary scan clean (`--groups` 7.5 PASS). |
| NB1, NB2, NB6 | Cost default, centred-returns default, scipy bootstrap | Left alone, as instructed | Owner decisions or optional cleanups; carried below. |

## Acceptance criteria
| Story | AC | Claimed | Reproduced? | Evidence |
|---|---|---|---|---|
| S1 | 1.1–1.6 | ✅ | ✅ | `--blind` `S1 pass: True (40s)`; `verify_method_boundary_s3` rc=0 |
| S2 | 2.1–2.3, 2.5, 2.6 | ✅ | ✅ | `verify_attribution.py` `pass: True (29/29)` |
| S2 | 2.4 | ✅ | ✅ (round 1, MPS batch check) | No change to `influence.py` this round |
| S3 | 3.1–3.6 | ✅ | ✅ | `--functionals` 17/17 (3.2 SKIP without a manifest; PASS with one, round 1) |
| S4 | 4.1–4.3, 4.6 | ✅ | ✅ | `--pipeline` 8/8 |
| S4 | 4.4, 4.5 | ✅ | ✅ | `--step4` 13/13, now gate-aware |
| S5 | 5.1–5.6 | ✅ | ✅ | `--recovery` 11/11 |
| S6 | 6.1–6.7 | ✅ | ✅ | `--names` 13/13 |
| S7 | 7.1–7.4 | ✅ | ✅ | `--groups` 9/9 |
| S8 | 8.1, 8.2 (+2 negatives), 8.4 | ✅ | ✅ | `--integration` 12/12 |
| S8 | 8.3 | ✅ | not re-run | The executor's re-run on the final code is logged; the `influence.py` path is unchanged since I checked it on MPS in round 1 |
| S8 | 8.5 | ✅ | ✅ | `verify_lineage_s3`, `verify_runs_s5 --r2`, `verify_blind_s0`, `verify_method_boundary_s3`, `verify_derive_gt_s2`, `verify_verdict_s7`, `verify_returns_s6 --synthetic`, `verify_metrics_s1`, `verify_consumers_s6`: all rc=0 |
| R7 | real store unchanged | ✅ | ✅ | `306af4123fda7e635892d459a2e34008ba1cb62f6c4fce152c546e0da7bdde79` before and after |

## PoC scope assessment
- **Covered:**
  - §3.4: declared functional per column.
  - §4.3: CUPID's estimator (ℓ_square on both sides, sum-of-sum, `net` parity), with every other difference in `DIVERGENCES`, now also on the ensemble run.
  - §4.4: the two-step verdict is now respected in the Step 4 aggregate; Kendall's τ.
  - §4.5: hypotheses committed before scoring; HRT vs F30 head-to-head; figures.
  - §9.1 Step 4: harmful, helpful and random arms, in both curation modes per owner Q1.
  - §9.2: the key contract now extends to the feature files' source data.
  - Blindness: all four entry points go through `open_for_method`.
- **Missing:** nothing in scope.
- **Over-built:** nothing new this round. The round-2 change is about 130 lines, mostly checks.
- **Silent divergences:** none. The group rule is now named.

## Overcomplication & reinvention audit
No new mechanisms this round. The round-1 audit stands: keep the custom estimator (trak's projectors are dense and CUDA-only, kronfluence uses a different Hessian; both quoted in round 1), and keep `server_pipeline.py` (submitit's examples document no dependency chaining). The new binding check reuses the existing `lineage.file_digests` and `sha256_json`, the same check `stage.cmd_features` makes.

## Feedback: planner
(none)

## Feedback: executor
(none blocking)

## Non-blocking
1. **Migration note for pre-round-2 feature files.** Add one sentence to the `attribute_run` refusal message, or to its docstring and the runbook, along the lines of: "feature files written before training-attribution round 2 (absolute or other-version `dataset_root`) must be regenerated with `stage.py features`; rewriting a manifest's `dataset_root` by hand defeats the check." Today this consequence exists only in the execution log (l.524).
2. **The NB4 rationale is backwards** (`analysis.py` `metric_groups` docstring; `DIVERGENCE_NOTE`).
   - If `rmst60`, `hrt60_lin` and `censored60` "move together", then an all-five rule is *easier* to satisfy, not emptier.
   - Redundancy is exactly what makes ≥2-of-5 weak: `harm_f30` plus `harm_censored60`, two readouts of reliability, can alone make an episode a "consistent harmer".
   - Reword the note to say the rule is a judgement call, labelled inferred, with that specific risk. Or make "consistent" require agreement across the throughput and reliability blocks.
   - Either fits a descriptive label. The owner may pick the strict PoC rule.
3. **Carried from round 1, for the owner:**
   - NB1: the default chain is 313 jobs and 66 trainings, about 264 GPU-h of training alone (inferred). `--curation per_task --metrics f30 hrt60_lin` gives 112 jobs and still covers the §4.5 headline.
   - NB2: centred returns are the default even for binary columns. That is the planner's N2 assumption, not owner-approved, and it differs from CUPID's ±1.
4. **Cost of the trajectory binding.** The rollout check hashes the whole trajectories directory once per attribute run. That is acceptable, because `cmd_features` already pays it per shard. Unmeasured at full scale (1,000 rollouts).

---

**Verification state:**
- **Real store:** `data/lineage/mlflow.db` sha256 `306af4123fda7e635892d459a2e34008ba1cb62f6c4fce152c546e0da7bdde79` before and after.
- **Tracked artifacts:** the verifiers rewrote 10 of them; I restored all 10 with `git checkout`:
  - other features': ids and timings, plus the known `s3_summary.json` drift;
  - this feature's: ids, timings and the chain sha.
- **Working tree:** shows only the owner's four uncommitted files.
- **Processes:** none left running.
- **Logs:** `data/scratch/analyzer_r2/` (`summary.txt`, per-verifier logs, `sha_before.txt`, `sha_after.txt`).
