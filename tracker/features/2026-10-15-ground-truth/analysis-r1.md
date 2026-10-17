VERDICT: ESCALATE
ROUTE: planner+executor
ROUND: 1
CASE: dispute-r1.md

<!-- Saved verbatim by the orchestrator from the analyzer's returned text: the analyzer's own
     write was refused by the harness ("Subagents should return findings as text"). -->

## Summary (3–5 lines)
All six verifiers reproduce exactly. Every AC holds except AC1.5's second clause, which the plan worded wrongly. The headline question is test validity:
- Of the five types a one-line statistic finds at AUROC 1.0, `time_shift` is detectable only through a planner-invented tail (40 steps of arm = 0) that clean data never has. That tail also adds an idle segment to a "none stated" group, confounding the metric-targeted test.
- `action_noise` (spec-literal, as in RoboDrop), `dropped_frames` (exact duplicates are realistic for a stale camera frame) and `idle_padding`'s idle-step count are legitimate detections.
- `idle_padding`'s duplicate frames and `instruction_swap`'s provenance cross-check are construction leaks. Removing them costs re-rendering or opaque keys.

Which leaks to remove before the single re-derivation is an owner trade-off, hence ESCALATE.

## Rebuttals received
| From | Point disputed | Their evidence | Conceded? | Why / why not |
|---|---|---|---|---|
| none (round 1) | n/a | n/a | n/a | No earlier analysis exists. This is the first written analysis after three stalled attempts. |

## Acceptance criteria
I re-ran everything with outputs in `data/scratch/gt_an2/` (logs in `logs/`, artifacts in `art/`). Main `mlflow.db` sha256 was `ecd5b4b420b5…` before and after. S6 overwrote `artifacts/s6_verify.json`; only `run_id` and seconds differed, and I restored it with `git checkout`.

| Story | AC | Claimed | Reproduced? | Evidence |
|---|---|---|---|---|
| S1 | 1.1–1.4, 1.6, 1.7 | ✅ | ✅ | `verify_corruptions_s1.py --version v0 --per-type 100 --seed 0`: 118 episodes per type, 0 invariant failures. Clip totals are identical (521 clipped, 0.72% of dims 0–2; 1,215 = 1.69% in (0.9375, 1]). Eligibility is 960×6 and 884, and P7 gives 907/960. |
| S1 | 1.5 clause 1 | ✅ | ✅ | Second process with `--check-hashes`: all seven types True. |
| S1 | 1.5 clause 2 | ❌ (honestly logged) | ❌ reproduced exactly | Hashes that change from seed 0 to seed 1: time_shift 0/118, gripper_release 0/118, swap 105, detour 112, the other three 118. The AC is wrong, not the code: both types have no random draw by design (plan table: "d = 20 steps", "40 steps"). The planner should amend it. |
| S2 | 2.1–2.7 | ✅ | ✅ | `verify_derive_gt_s2.py`: `check_blind = []`. Both refusals reproduce. `tasks.parquet` is equal to the parent's. PYTHONHASHSEED 0 vs 1 gives the same `dataset_version_id` `3f43df4898854eed`, with only `stats.json` key order differing. The blind-check table matches row for row. |
| S3 | 3.1–3.6 | ✅ | ✅ | `verify_method_boundary_s3.py`: the 12-row refusal table matches. The scan finds no violations and catches 6/6 planted ones. |
| S4 | 4.1–4.7 | ✅ | ✅ | `verify_injection_s4.py` on stored bytes: 91 labels, 13 per type, 9.48%. 1000/1000 hashes re-hash correctly and 91/91 labels match. The clip recount (65 / 139) matches. The trace visits 3 runs. One discrepancy: `du -sh` printed **6.8G** in my run against 7.3G in the log (non-blocking, probably a units or measurement difference). |
| S5 | 5.1–5.7 | ✅ | ✅ | `verify_recovery_s5.py`: 26/26 fixtures pass. 200 random rankers give pooled AUROC 0.4987. Both key-check rejections reproduce. The leak check gives v1 1.000 and v2 0.099 / 0.5045. The heuristic table matches the CSV (recomputed distributions below). |
| S6 | 6.1–6.5 | ✅ | ✅ | `verify_consumers_s6.py`: stats difference 2.70e-10. The SmolVLA 2-step smoke run finished in 17.5 s with a matching digest `c60d2183e24e139a`. The human-reference digest and the real store are unchanged. |

## PoC scope assessment

### Headline: test validity
I measured on v2 data from the ground-truth side (`load_answer_key('v2')` plus the stored heuristic tables).

| Type | What separates it | Clean range | Injected range | Cause | Label |
|---|---|---|---|---|---|
| time_shift | idle_steps | max 21, median 0 | exactly 40 in 13/13 | The tail alone: the shifted body contributes 0 idle steps. Clean episodes have at most 7 idle steps in their last 40, and their final arm action is exactly 0 in 9/1000. | measured |
| idle_padding | idle_steps; duplicate frames | 0 duplicate pairs across 199,340 clean pairs (P4), even though clean data has 436 exact-zero arm steps in 113 episodes | 40–43 idle steps; exactly 80 duplicate pairs | Idle count: inherent to the corruption (PoC §2.2 lists idle time as a catalog statistic). Duplicate frames: a construction artifact, because real zero-action steps in LIBERO still change the image. | measured |
| action_noise | smoothness | max 0.161 | min 0.216 | Independent per-step Gaussian noise at 0.25·σa, as in RoboDrop App. A.1. A max\|a\| statistic gives AUROC 0.88 (11/13 episodes have values beyond 0.9375), which adds nothing over smoothness. | measured |
| dropped_frames | duplicate frames | 0 | 23–92 | A sample-and-hold stale frame is an exact duplicate. Real systems behave this way (LeRobot's `read_latest`). Frozen-frame detectors such as Calibra exist. | measured / quoted |
| instruction_swap | instruction vs `task_name` | n/a | 13/13 | Plan D1 keeps true provenance. RoboDrop p.13, for its non-expert labels: "Operator provenance is used only for evaluation and is unavailable to RoboDrop and the baselines." | quoted |
| gripper_release / detour | none (best AUROC 0.505 / 0.73) | n/a | n/a | n/a | measured |

What the papers say about detectability:
- **PoC (pp. 3, 5, 20).** It only asks to "check whether your attribution method flags exactly those" and to inject 5–10%. It sets no detectability requirement.
- **RoboDrop Table 1 (p. 7).** It accepts an action corruption that is trivially separable: "DataMIL and QoQ perfectly separate the controlled action corruption" (AUROC 100.0). Length scores 53.3 on both its corruptions, which suggests (inferred) that its temporal corruption preserved length.
- **CUPID (TuckBox, around l. 1426).** Its point is that attribution matters most where "attending to heuristic properties of demonstrations" fails. In this ground truth only `gripper_release` and `detour` test that.

**Judgement.** The ground truth is sufficient for §10 "Ground truth" as a mechanism, and per-type reporting against the heuristic baselines (S5) is the right mitigation. Two exceptions:
1. **The `time_shift` tail is a defect, not a matter of taste.** It is an invented fill (plan: "inferred, since RoboDrop states neither"). It makes the type detectable only through the artifact. Because it puts 2 s of exact idle into a "none stated" group, it contaminates the §4.5 p.11 / §8.10 p.17 "assign each corrupted group to the expected metric" test.
2. **The `idle_padding` duplicate frames and the swap provenance leak are realism costs** the owner should rule on before the one re-derivation.

### Covered, missing, over-built, silent divergences
- **Covered:** §2.5 (all except viewpoint, which the owner ruled out), §8.10 / §4.5 hypotheses as labels, §8.7 clean references, the 5–10% budget, rank-based recovery, blindness and the loader.
- **Missing:** nothing the spec requires. The policy-level harm is correctly marked pending server.
- **Over-built:** nothing material. `open_for_method` runs `verify_version` on every call (about 20 s). That is acceptable because the AC asks for it.
- **Silent divergences:** none found. Every inferred parameter is labelled in `params`.

### Point 3: AC1.5 wording, clipping, trainer bypass
- **AC1.5:** the planner should amend it (see feedback).
- **Clipping:** 1.92% of translation values fall in (0.9375, 1]. That is inside [−1, 1], so owner decision 1 is literally satisfied (it is binding and not re-opened). Measured incremental leak: a max\|a\| detector reaches AUROC 0.88, below the 1.0 smoothness already gets, so the clip choice does not change detectability. Non-blocking.
- **Trainer and eval bypass:** `vlapoc/runs.py` has no `assert_blind` or fence check (grep). This is correct for the training arms, which must train on corrupted data and v2 is blind anyway. It is a gap for later attribution methods that train policies (datamodels, CUPID): an oracle-derived subset passed to `train_run` would not be refused, which falls short of owner decision 3's intent ("later screening / attribution features cannot read a leaky or fenced version by accident"). Inferred. It should be recorded as a known issue for the attribution feature. It does not block this feature.

### Point 4: the `compare_blind` change
Older versions behave as before on the real store: `check_blind(v0) = []` and `check_blind(v1) = ['episode_index']`, with v1 still refused (re-run in S2 and S3). v1 never edited instructions, so the new rule that `meta/episodes.tasks` equals `language_instruction` cannot fire there. I did not re-run `verify_blind_s0.py` myself (it rewrites another feature's artifact). I relied on the S2/S3 reproductions, plus the executor's logged regression run and timing-only diff.

## Overcomplication & reinvention audit
| Mechanism (file:line) | Existing OSS alternative | Covers our need? | Evidence | Recommendation |
|---|---|---|---|---|
| Seven corruption transforms (`corruptions.py`, 547 lines) | No library injects corruptions into LeRobot or robomimic episodes. Trajlens, Calibra and EpisodeVault detect or repair. | No | quoted (search results; Calibra HF thread, Trajlens PyPI) | **Keep** |
| Recovery metrics (`groundtruth.py:235`) | sklearn `roc_auc_score`, `roc_curve`, `average_precision_score` | Already used. Only recall@k and the tie rule are custom (a few lines). | quoted (code) | **Keep** |
| AST boundary scan (`verify_method_boundary_s3.py:46`, about 30 lines) | import-linter "forbidden" contract | Partly. It forbids imports, including indirect ones, but does not check string literals (`labels.parquet`) or attribute names (`parent_hashes`), which the AC requires. | quoted (import-linter docs) | **Keep** (non-blocking: import-linter could replace the import half) |
| `open_for_method` loader and ground-truth-chain refusal (`method_data.py`) | Nothing found: LeRobot, MLflow and DVC have no blindness or answer-key guard. | No | inferred | **Keep** |
| Stratified selection (`groundtruth.py:110`) | sklearn `StratifiedShuffleSplit` | No. It needs one type per task, 13 distinct tasks and a 6/7 suite split. | inferred | **Keep** |
| Heuristics H1–H6 (`methods/heuristics.py`, 114 lines) | Calibra (frozen-frame and jitter detectors) | Not as a pip-installable API we checked. They read catalog columns, so they are thin. | inferred | **Keep** |

Searches run: "import-linter forbidden contract documentation"; "LeRobot dataset inject corruption noise episodes tool robot demonstration corruption benchmark library".

## Feedback: planner (after the owner rules on dispute-r1)
1. **Plan "The seven corruption types" table, `time_shift` row** (implemented at `corruptions.py:271-274`). Replace the tail fill "arm 0, gripper held". Recommended: **trim**, pairing obs[0:T−40] with a[40:T] so the episode has T−40 frames. Every frame and action is then real and nothing is idle. Record in the label that the length changes by −40. Add an AC: `time_shift` idle_steps equals the parent's idle steps within a[40:T] (expect ≤ 7, measured clean maximum over any last-40 window), so H2 AUROC on `time_shift` drops to chance. Report the residual length (H1) AUROC rather than hiding it.
2. **AC1.5.** Amend to: "same (seed, episode_id) gives the same hash in two processes for all seven; a different seed changes the hash for the five stochastic types (all of `action_noise`, `dropped_frames` and `idle_padding`; at least 80% for `detour` and `instruction_swap`, which draw from finite sets); `time_shift` and `gripper_release` are deterministic by design."
3. **Budget the single re-derivation (v3, about 7 GB, about 9 min; 87 GiB free)** to include whatever the owner chooses in dispute-r1 (options B/C). Decide whether v2 is retired, following the single-record-system rule, and record it.
4. **Add a known issue for the attribution feature:** `train_run` bypasses `open_for_method`.

## Feedback: executor (after the planner revises)
1. Re-derive with the revised `time_shift` (and any owner-chosen changes). Re-run S4–S6 and the S5 heuristic table on the new version. Expect `time_shift` idle_steps AUROC ≈ 0.5.
2. In `execution-log.md` AC4.6, reconcile the 7.3G figure against my `du -sh` result of 6.8G.

## Non-blocking
- The data+lineage AC4.8 "contrast" figure (60 files / 663 MB) is stale: it is now 0 files, also on pre-S2 code (executor-measured). Add a changelog line to the data+lineage feature and fix its report.
- P7 numbers: 907/960 have a final release and 884 are eligible (I reproduced both). The plan's 944 was over all 1000 episodes. The planner should update P7's text.
- `action_noise`'s max\|a\| leak is AUROC 0.88 (measured). It is dominated by smoothness, which reaches 1.0.

Sources:
- [import-linter forbidden contract](https://import-linter.readthedocs.io/en/latest/contract_types/forbidden/)
- [Calibra HF forum thread](https://discuss.huggingface.co/t/open-source-tool-for-detecting-issues-in-robot-learning-datasets/178602)
- [Trajlens on PyPI](https://pypi.org/project/trajlens/)
- [EpisodeVault](https://discourse.openrobotics.org/t/episodevault-open-source-tool-to-debug-why-your-lerobot-model-regressed/55401)
