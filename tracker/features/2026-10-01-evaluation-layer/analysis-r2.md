VERDICT: ESCALATE
ROUTE: none
ROUND: 2
CASE: dispute-r2.md

<!-- Saved verbatim by the orchestrator from the analyzer's returned text: the analyzer's own
     write was refused by the harness. -->

# Analysis, round 2: evaluation layer (PhAIL primitives on LIBERO)

Analyzer: vla-analyzer, 2026-10-06. Evidence labels: **measured** (I ran it this session; logs were under
`data/scratch/analyzer_r2/`, since deleted, and every number is quoted here), **quoted** (source line, doc or
page), **inferred** (my judgement).

Constraints kept (measured):
- Main `data/lineage/mlflow.db` sha256 was `7845d8d7…` before and after.
- The sha256-of-all-files of `data/lineage/` was `42e091a1…` before and after.
- All MLflow-writing checks ran on `data/lineage_scratch/analyzer_r2`, made with the executor's
  `scratch_lineage_copy` and since deleted.
- The executor artifacts I regenerated are byte-identical or were restored (`git status` clean).
- `~/.libero` is absent, no processes are left running, and free disk is 112 GiB.
- **One side effect.** My `--plan-guard` re-run rewrote `data/eval/verdict-W-3392bb4d-e138fec9/verdict.json`
  (gitignored; content equal except run ids). Scratch-store verdict runs default their output dir into the real
  `data/eval/` (non-blocking N1).

## Summary (3–5 lines)

**Round-1 items.** Every item is resolved and reproduced (F-P1–F-P5, E1–E5):
- returns and verdict now agree on one declared functional per metric (family W) on both SmolVLA runs, max
  \|diff\| 7.11e-15;
- the unfinished operation scores −60; there is one registry, and UPH = 3600/RMST;
- attribution grade is asserted and checked, and smoke runs are tagged smoke;
- the PhAIL export is byte-identical across `PYTHONHASHSEED`; `t0_model_sha256` catches the 7.2 mm drift; the
  batch-plan guard raises; no temp dirs leak.

**Regressions.** Round-1 ACs touched by round 2 still reproduce: S1/S2 after the registry merge, the noop
re-run against the new schema, and the six PhAIL verdicts, identical to the artifact.

**The one open point is AC7.1 GR00T–ACT (29/30).** The executor's seed-1 (i) reproduces exactly at
n_boot = 1000. At n_boot = 10,000 all 5 seeds give (ii), with AUC Holm p 0.0248–0.0324 (measured). So the
seed-1 flip is Monte-Carlo resampling risk, not a boundary property of the data. I do not concede the "≥ 4/5"
restatement, and this is escalated (case file `dispute-r2.md`).

## Rebuttals received

| From | Point disputed | Their evidence | Conceded? | Why / why not |
|---|---|---|---|---|
| executor | AC7.1 (r2): "GR00T vs ACT → (ii) … 5/5 seeds" is too strong for a Holm-boundary result; restate as "(ii) in ≥ 4/5 seeds" | `verify_verdict_s7.py --phail-seeds 0 1 2 3 4`: AUC Holm p 0.030 / **0.054** / 0.040 / 0.032 / 0.024 at n_boot 1000; round-1 5/5 came from the unsorted export | **No** | Their numbers reproduce exactly (measured). Their inference, that the AUC "sits at the α = 0.05 boundary" so that "(ii) is a Monte-Carlo outcome, not a stable property", does not hold. At n_boot 10,000, seeds 0–4 give AUC raw p 0.0068–0.0108, Holm 0.0248–0.0324, outcome (ii) 5/5 (measured). The converged Holm p ≈ 0.03 is several MC standard errors below α. The seed-1 (i) is what Gandy (2009) calls resampling risk, "the probability of a different decision than the one based on the theoretical p-value" (quoted, arXiv math/0612488). The remedy is MC precision, not a weaker expectation. See `dispute-r2.md` |
| executor | AC6.9: the round-1 leaked temp dirs were 9, not the analyzer's 11 | `artifacts/s6_ac6_9_tempdirs.txt` | **Yes** | My own round-1 `ls` listed 13 `returns_*`/`verdict_*` dirs, 4 of them created by my runs (timestamps 2026-10-05 11:39). 13 − 4 = 9. My "11" was a miscount |

## Acceptance criteria

| Story | AC | Claimed | Reproduced? | Evidence |
|---|---|---|---|---|
| S2 | AC2.9 (r2) | ✅ | ✅ | measured: exports with `PYTHONHASHSEED=0`, `=1` and unset → `cohort_mixed.parquet` `35b7181bc92c42c5…` ×3, `s2_phail_reference.json` `22549627c8b4bc17…` ×3. `phail_validate.py` → macro-KS identical for 6 pairs (OpenPI–ACT 0.110445, GR00T–ACT 0.008996); 23/26; winners GR00T/OpenPI/GR00T/ACT; AC2.5 0.628 vs 0.808; artifacts byte-identical |
| S2 | AC2.1–2.8 regression | ✅ | ✅ | as above, after the registry merge |
| S1 | AC1.1–1.7 regression | ✅ | ✅ | measured `verify_metrics_s1.py`: sha equal, KM vs scipy 3.331e-16, registry 23 entries / missing [], macro-KS equivalence identical |
| S4 | AC4.2, AC4.5 regression | ✅ | ✅ | measured: 7/7; 1 definition, 2 callers |
| S5 | AC5.2 regression | ✅ | ✅ | measured `--rerun eval-noop-smoke-a`: `30 columns compared …; differing columns: none` (the stored record predates `t0_model_sha256`; the comparator uses the common columns) |
| S5 | AC5.3 (r2) | ✅ | ✅ | measured: round-1 trace JSON-equal (round 1); the r2 smoke eval `f8f3b153` traced from a copy of the executor's scratch store → `build-v0, derive-v1, eval-…-smoke-r2, split-v0, train-full-s0`, 20 raw files, `eval.attribution_grade=false`, asserted absent, `vlapoc.smoke=true`; `s5_ac5_3_trace_r2.json` equal to the recomputed trace |
| S5 | AC5.9 (r2) | ✅ | ✅ | measured `--grade-tests` (scratch): 2 init states refused with 3 reasons; full protocol on the smoke checkpoint refused with only the smoke-tag reason; runs before/after 45/45; default eval → `eval.attribution_grade=false`, asserted absent, `vlapoc.smoke=true`, `eval.batch_plan` digest `1852ef679a90` |
| S5 | AC5.10 (r2) | ✅ | ✅ | measured `--t0-model --compare eval-act-train-full-s0-smoke-r2`: equal at batch 1/2 and on repeat; distinct across (task, init); equal to the other run's records; reused env ×3 → `e62c4829473e, 873c350ba3a4, 04f422206485` vs `t0_state` `54f7df61644e` ×3; `validate_record` without the column → `["missing columns ['t0_model_sha256']"]` |
| S6 | AC6.1 (r2) | ✅ | ✅ | measured: synthetic (incl. 16 ties) ≤ 1.07e-14; real spatial:0 0.00e+00 (26.185 s / 0.600 / 0.400); real libero_10:8 ≤ 7.11e-15 (**48.400 s / 0.300 / 0.700**, the whole-task values) |
| S6 | AC6.8 (r2) | ✅ | ✅ | measured: no `METRICS =`/`REGISTRY =` in the three modules, all import `metrics_registry`; UPH `3600 / RMST(τ_episode) (adopted …`; 11/11 entries carry family and T; `uph` vs 3600/RMST(240) on the S2 cohort 0.00e+00 |
| S6 | AC6.9 (r2) | ✅ | ✅ | measured: `data/scratch/{verdict,returns}_*` count 0 before and 0 after a `returns` run plus `verify_run`s |
| S6 | AC6.10 (r2) | ✅ | ✅ | measured `--real … --against-verdict` (copies of both SmolVLA rollouts): max \|diff\| 7.11e-15, labels carry the family |
| S6 | AC6.11 (r2) | ✅ | ✅ | measured: unfinished attempts all −60; max unfinished −60 ≤ min completed −48.4 (libero_10:8) and −3.85 (spatial:0); synthetic tails 1/58/60 s → −60 |
| S6 | AC6.2 regression | ✅ | ✅ | measured: worst 9.51% |
| S7 | AC7.1 (r2) | ❌ 29/30 | ❌ 29/30 at n_boot 1000 (reproduced); **5/5 for GR00T–ACT at n_boot 10,000** | measured `--phail-seeds 0–4`: all other pairs match in 5/5; GR00T–ACT seed 1 → (i), AUC Holm 0.0539. Probe at n_boot 10,000 (n_boot_ks 2000): (ii) in seeds 0–4, AUC Holm 0.0312 / 0.0306 / 0.0294 / 0.0324 / 0.0248, RMST Holm 0.019–0.025, gate p 0.0065–0.0100. **Disputed → `dispute-r2.md`** |
| S7 | AC7.2 regression | ✅ | ✅ | measured: OpenPI 0.050 ± 0.030, mean p 0.427 (identical to the log) |
| S7 | AC7.3 regression | ✅ | ✅ | measured: cycle A≻B≻C≻A; [family O] direction B, Δ +11.40 s |
| S7 | AC7.4 (regenerated) | ✅ | ✅ | measured from stored rollouts: [family W] D̄ 0.6000, p 0.0090 → (i) SmolVLA; 5 metrics Holm 0.0100 |
| S7 | AC7.6 (r2) | ✅ | ✅ | measured: `python -m vlapoc.verdict --phail-all` stdout identical to `s7_phail_verdicts_stdout.txt` |
| S7 | AC7.7 (r2) | ✅ | ✅ | measured `--family-consistency`: max \|verdict − S6\| 7.11e-15; libero_10:8 W f30 0.3 / auc_h 0.085 / rmst60 48.4 / hrt60 0.416095 / censored60 0.7; O `rmst60_op_cap` 38.146667 |
| S7 | AC7.8 (r2) | ✅ | ✅ | measured `--plan-guard` (scratch): a vs b raised; override labelled `'smoke (not attribution-grade), family W, unpaired plan'`; same plan → `batch_plan_match True`; pytest `4 passed` |
| S7 | AC7.9 (r2) | ✅ | ✅ | measured (scratch chain, below) and read-only on the main DB: `verify-ref-vs-drop-random-5 [eval-smoke]` (85f7d7d9) → `verify.attribution_grade=false`, label `smoke (not attribution-grade), family W`; its evals `eval.attribution_grade=false`, `vlapoc.smoke=true` |
| S9 | AC9.1 (r2) | ✅ | ✅ | measured `eval_smoke()` on the scratch store: exit 0, 126.6 s; train ×2 `vlapoc.smoke=true`; eval ×2 `attribution_grade=false`, smoke; verify `false`, family W, `batch_plan_match true`, outcome (iii); trace 20 raw files, no stand-ins, no dead ends (16 runs listed, because the bit-identical checkpoints have earlier producer runs in the copied store) |
| S9/S8 | AC9.2 (r2), AC8.5 (r2) | ✅ | ✅ | read `server-runbook.md`: AC8.5 `pretrained_ref='lerobot/smolvla_base@d9f33c94…'`, `steps=20000, batch_size=64` (matches LeRobot SmolVLA docs, quoted in round 1), `smoke=False`; AC5.8 `--batch-size 50 … --attribution-grade` only there and in AC7.5; AC7.5 `--family W` and `--family O` |

## PoC scope assessment

**Covered.** As in round 1, plus the declared families (plan r2) that make §3.4 p. 7 "Declared targets" and §4.4
p. 10 "for that metric" checkable on libero_10 (AC6.10 / AC7.7).

**Divergences, all recorded.**
- **Family W is the declared attribution family on libero_10.** §3.5 A p. 7 asks for per sub-task T. Family O
  keeps per-sub-task T with its own verdict and return. The plan's reasoning follows §9.1 Step 3 p. 19, "one
  'return' per metric for each rollout".
- `rmst60_op_cap` is pessimistic by declaration.
- The LIBERO AUC counts ties ½.
- `gate_only`.
- The human anchor is held fixed in the difference bootstrap.
- AC8.5 fine-tunes from smolvla_base.

**Missing.** Nothing at the local scope. AC5.8, AC6.7, AC7.5 and AC8.5 are pending server.

**Over-built.** Nothing new. One registry replaced three.

**Silent divergences.** None found.

## Overcomplication & reinvention audit (round-2 mechanisms)

| Mechanism (file:line) | Existing OSS alternative | Covers our need? | Evidence | Recommendation |
|---|---|---|---|---|
| `vlapoc/metrics_registry.py` (197 lines, data only) | none found for declared functionals | n/a | inferred; it replaces 3 tables (measured grep) | keep |
| `runs.scratch_lineage_copy` (`runs.py:600-622`) | `mlflow-export-import` export-all / import-all | it copies via the REST API "from one MLflow tracking server … to another" (quoted, PyPI); it needs a server and is heavy for a test sandbox of one SQLite file | measured: the helper made a correct copy (43 runs, artifact_location repointed, main DB unchanged) | keep (about 20 lines) |
| `evaluation.auc_ties_half` | `scipy.stats.mannwhitneyu(T_h, T_m).statistic / (n_h·n_m)` with censored rollouts → ∞ | yes on family W only, because every pair is comparable (max human T 24.75 s < 60 s, measured); no for general censoring | measured: scipy 0.085 == ours 0.085 on libero_10:8 | keep (encodes PhAIL's censored-pair rule); optionally add scipy as an independent cross-check |
| `batch_plan`, `batch_plan_of`, `attribution_protocol_errors` (`runs.py:559-600`) | LeRobot eval has no protocol assertion | no | quoted LeRobot docs give the pairing rule only as prose | keep |
| `rollouts.model_placement_sha256` | none needed (sha256 of `body_pos` + `site_pos`) | n/a | measured AC5.10 | keep |

Searches run this round:
- WebSearch "mlflow copy runs between tracking servers export import tool"
- WebSearch "Monte Carlo error bootstrap p-value number of bootstrap replicates near significance threshold",
  then WebFetch arXiv math/0612488
- Local: scipy `mannwhitneyu` equality

## Feedback: planner

Pending the owner's ruling on `dispute-r2.md`. Under Option A or C, restate AC7.1 (and AC7.5's verdict
commands) with the precision rule chosen. Under Option B, adopt the executor's wording.

## Feedback: executor

Pending the owner's ruling. Under Option A or C, the work is in `dispute-r2.md` "If you rule". No other blocking
items.

## Non-blocking

- **N1.** Scratch-store tests write their outputs into the real `data/eval/`:
  - `verify_run`'s `out_dir` and `eval_run`'s default `out_dir` ignore `lineage_root`;
  - examples are `data/eval/verdict-W-*`, `eval-grade-default [smoke]-…`, `eval-act-train-full-s0-smoke-r2`.
  - Default them under the scratch root when `lineage_root` is given (refactorer).
- **N2.** The round-1 records, including both real SmolVLA runs, now fail `validate_record`
  (`missing columns ['t0_model_sha256']`, measured). The returns and verdict paths don't validate, so nothing
  breaks today. A record schema-version field, or a documented legacy exemption, would make that explicit.
- **N3.** The batch-plan guard compares batch size, tasks, inits and seed. It does not compare τ_episode, render
  size or device, which also change a stochastic policy's outcomes (inferred). Consider adding them to the plan
  key.
- **N4.** `data/lineage_scratch/r2` (4.1 MB, the executor's) and `data/lineage_scratch/s5` (2.3 GB, data+lineage
  leftover) remain. Owner cleanup.
- **N5.** Carried from plan r2 as refactorer items: the data+lineage full dry run and `verify_runs_s5.py`
  re-run; the three older scripts moved onto `scratch_lineage_copy`; v1-fence enforcement in later features.
