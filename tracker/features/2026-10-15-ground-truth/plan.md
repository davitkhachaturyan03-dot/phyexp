# Plan: Ground truth (injection experiment)

Round 1, planner, 2026-10-15. Brief: [`brief.md`](brief.md).

## Request

"start building the ground truth feature" ([brief.md](brief.md)). This is PoC §10 p. 20 "Ground truth | Injection
experiment with 5–10% corrupted episodes, using LIBERO's original demos as clean references". Corrupt a known subset
of the training copy with the §2.5 corruption types and the §8.10 metric-targeted ones, and keep the answer key on
the ground-truth side. Make the corrupted version usable by everything downstream: it must be trainable (per-subset
normalization), evaluable and scoreable. Recovery is scored from the labels against any method's score table keyed
by (`episode_id`, `content_hash`). Three owner decisions are binding:
1. Clip every corruption to [−1, 1] and record how much the clip bit.
2. All corruption types except viewpoint.
3. Enforce blindness now, through one method-facing loader that always calls `lineage.assert_blind` and respects
   the v1 fence. The answer key is read only through an explicit ground-truth-side path.

## Spec grounding

All page numbers refer to `docs/VLA_Data_Attribution_PoC.pdf`, extracted this session to `data/scratch/poc.txt`.

- **PoC §2.1 item 4 (p. 3)**, quoted: "Plant ground truth, which is the key trick for a PoC. Deliberately corrupt
  known subsets of training data: noisy or shifted actions, mismatched language instructions, altered camera
  viewpoints, or low-quality teleoperation demos. Then check whether your attribution method flags exactly those.
  This turns 'the method says X' into 'the method recovered 90% of the injected bad episodes.'"
  → The headline recovery number is **recall at k = number injected**: the fraction of injected episodes inside the
  method's top-k.
- **PoC §2.5 (p. 5)**, quoted table: "Noisy or time-shifted actions | Teleoperation lag, timestamp misalignment";
  "Swapped or wrong language instructions | Annotation errors"; "Altered camera viewpoint or dropped frames | Sensor
  drift, hardware faults"; "Low-skill or hesitant demonstrations | Inexperienced operators, idle periods,
  corrections". Also: "Inject 5–10% corrupted episodes and measure how many the method ranks near the top."
  → The budget is 5–10% of the training (candidate) set, and recovery is rank-based.
- **PoC §4.5 (p. 11)**, quoted: "Metric-targeted injections. Extend the injection test with corruptions designed to
  hit specific metrics — for example, idle or hesitant segments aimed at time-based metrics, and noisy gripper release
  near placement aimed at drops. Check whether attribution assigns each corrupted group to the expected metric. These
  mappings are hypotheses the experiment should confirm."
  → Each label carries its metric hypothesis, and only where the PoC states one. The labels are the input for a later
  metric-resolved check.
- **PoC §4.2 (p. 10)**, quoted: "The Pebblous result, where near-idle demos made a policy freeze, suggests hesitant
  data mainly moves time-based metrics." → This is the stated basis for the idle hypothesis.
- **PoC §8.10 (p. 17)**, quoted: "Corrupt episodes two ways: pad some with idle or hesitant segments (should hurt
  speed, not reliability) and add gripper-release noise to others (should hurt reliability, not speed)." Also p. 18:
  "Verify before claiming. Retrain with the disputed episodes restored and show the speed metric improves
  significantly ..." → The policy-level effect is pending server; nothing here claims it.
- **PoC §8.7 (p. 15)**, row "Benchmark demos (PoC)", quoted: "Treat LIBERO's original demos as clean; corrupt only
  the training copy | Free references plus ground truth". **PoC §8.3 (p. 14)**: references "are a small, held-out set
  of clean demonstrations for the target task, separate from the candidate data being audited."
  → Derive from v0, never edit a `reference` episode (the existing guard), and inherit the 40-reference split.
- **PoC §9.1 Step 1 (p. 18)**, quoted: "Its verdicts are stored rather than acted on, so they can later be compared
  with what actually happened." Step 4 (p. 19): "Remove the top-k most harmful episodes and retrain. As a control,
  remove k random episodes" → Score tables are stored artifacts keyed by episode version. The scorer compares them
  with labels after the fact.
- **PoC §5 (p. 11–12)**, the "Intrinsic quality | Mutual information, redundancy, smoothness | ... | DemInf, SCIZOR,
  heuristics" row → grounds the cheap heuristic rankers in S5 as legitimate baselines of the screening family.
- **PoC §2.2 (p. 4)**, quoted: "basic action statistics (smoothness, idle time, gripper toggles)" → these catalog
  columns are what the heuristic rankers read.
- **PoC §10 (p. 20)**: "Ground truth | Injection experiment with 5–10% corrupted episodes, using LIBERO's original
  demos as clean references". Later phases name what consumes this feature: "Fine-grained attribution | ... add TRAK
  or RoboDrop-style scores; compare against DemInf" and "Metric-resolved attribution | ... metric-targeted
  injections".

**Rules the build must follow:**
1. Corrupt only candidate episodes of a copy, and never a reference (§8.7 p. 15, §8.3 p. 14). Enforced by the
   existing `derive.protected_episode_ids` guard.
2. The method under test must be able to "flag exactly those" on its own (§2.1 item 4 p. 3). No bookkeeping in any
   method-facing table may differ because of an edit. Only content may differ.
3. Budget: 5–10% of the training set (§2.5 p. 5).
4. Metric hypotheses are recorded as hypotheses, never as outcomes (§4.5 p. 11).
5. Owner decision 1: every corrupted action stays in [−1, 1], and the clipped fraction is recorded per label.
6. Owner decision 3: methods read data only through the method-facing loader. Labels are read only through the
   ground-truth module.

**Papers:**
- `docs/papers/RoboDrop_Gradient_Compatibility_arXiv-2609.10021.pdf`, read from `data/scratch/papers/RoboDrop.txt`:
  - **App. A.1 (p. 13)**, quoted: "We construct the action- and temporal-corruption datasets separately by corrupting
    10% of the episodes for every task. Action corruption adds zero-mean Gaussian noise with standard deviation
    0.25σa to the demonstrated actions, where σa is the empirical action standard deviation. Temporal corruption
    shifts the action sequence by two seconds relative to the observations. For validation, we reserve two
    uncorrupted episodes per task." → Defines the noise level and the shift size. It does **not** state the shift
    direction, how the vacated steps are filled, whether the gripper is noised, or any clipping.
  - **§5.1 (p. 6–7)**, quoted: "We then assess how well these scores separate reliable from unreliable episodes using
    AUROC and best balanced accuracy." Table 7 adds an "Oracle" curation ("removes exactly the corrupted episodes",
    p. 14). → The recovery scorer reports AUROC and best balanced accuracy next to the PoC's recall@k. The oracle
    arm goes in the server runbook.
  - Table 1 (p. 7) baseline "Length ... ranks demonstrations by their duration" (AUROC 53.3 on LIBERO-10) → the
    length heuristic in S5.
- `DemInf` §"Pauses" (`data/scratch/papers/DemInf.txt` l. 1185–1192), quoted: "Because DemInf considers the average
  estimated Î across a trajectory, it is susceptible to preferring data that is predictable ... if a robot pauses for
  an extended amount of time at a particular state, the action distribution is very predictable." → Idle padding is
  a known blind spot of the MI baseline. That makes it a meaningful injection for the later "compare against DemInf"
  phase.
- `SCIZOR` RQ4 (`SCIZOR.txt` l. 430–445), quoted: "Pause denotes transitions with no movement ... Lagging captures
  motion that proceeds noticeably below the normal speed ... Move Back and Forth indicates aimless motions that don't
  contribute to task progress." → Grounds idle padding (pause) and the detour (back-and-forth) used for the §2.5
  "low-skill ... corrections" row.
- `CUPID` §6.1 (`CUPID.txt` l. 576–582): its "Oracle approximates an upper bound on performance by curating data
  with privileged access to ground-truth demonstration labels". This is the same oracle notion as RoboDrop's.

## Prior art

| Candidate | What it covers | What it misses | Evidence |
|---|---|---|---|
| This repo: `vlapoc.derive.derive_version` + `EpisodeEdit` (data+lineage S4, evaluation S0) | Writes a child version through LeRobot's own writer in parent order. Labels go in `labels.parquet`, pre-edit hashes in `parents.parquet`. Has the reference guard, length-changing edits, `compare_blind` before registration, and split inheritance | (a) Refuses task edits: `derive.py:394-395` raises "task edits must go through LeRobot modify_tasks (not supported here)". (b) `compare_blind` treats `language_instruction` (catalog) and `tasks` (`meta/episodes`) as bookkeeping (`lineage.py:698-703`), so an instruction swap would fail the blind check. (c) Only one corruption exists (v1 noise, unclipped, all 7 dims) | measured (read `vlapoc/derive.py`, `vlapoc/lineage.py` this session) |
| LeRobot 0.6.1 `dataset_tools` | `delete_episodes`, `split_dataset`, `merge_datasets`, `modify_features`, `add_features`, `remove_feature`, `modify_tasks`, `recompute_stats`, video conversion | No per-episode frame-value rewrite. `modify_tasks` works **in place** and re-indexes tasks alphabetically (`dataset_tools.py:1531` `unique_tasks = sorted(set(...))`), so it would rewrite every data file's `task_index` and change the parent's task order | quoted (docs: "`modify_tasks` modifies the dataset **in-place**"); measured (installed source, `dataset_tools.py:1436-1581`) |
| LeRobot writer `save_episode` / `meta.save_episode_tasks` | Assigns `task_index` to new tasks in first-appearance order (`dataset_metadata.py:541-565`) | A swapped instruction on an early episode would claim an early index, so `tasks.parquet` order would differ from the parent's. Pre-registering the parent's tasks in order avoids this | quoted (installed source) |
| RoboDrop (2609.10021) | Defines action noise and temporal shift, with AUROC/balanced-accuracy scoring | No released code. Separate dataset per corruption type | quoted: PoC §8.8 p. 16 "Code: no public release was found"; web search this session found no repository |
| Bedi 2026, "Auditing Demonstration Curation Metrics" (arXiv 2606.05588), `github.com/aaravbedi/scorer-fail-on-structural-defects`, CC BY 4.0 | Defect injector with AR(1) action noise, tremor, truncation, "detour (reversed mid-episode segment spliced back)" and "Early release — gripper commanded open during the carry phase, object dropped partway to goal". Scored with AUROC | Works on its own NumPy pick-and-carry simulator, not LeRobot/LIBERO. Parameters "exist only in released code", and I could not open the source files through the fetch tool | quoted (paper summary page fetched this session). Inferred: not reusable as code, but it is prior art for two of our transforms (detour, early release) |
| Calibra (`github.com/omertt27/Calibra`) | Detects "frozen camera frames", "jittery robot motion", duplicates | It is a detector, not an injector. It is a candidate heuristic baseline, not a generator | quoted (HF forum post fetched this session) |
| Pebblous, instruction–trajectory mismatch | Injects swapped instructions at 30% and detects them with multimodal experts | No code mentioned. Blog only | quoted (blog fetched this session) |
| LeRobot camera `read_latest()` | Real LeRobot recordings fill a late camera frame with the **previous** frame, up to `max_age_ms=500` | n/a. This defines the realistic shape of "dropped frames" in LeRobot data | quoted (`lerobot/cameras/opencv/camera_opencv.py:582-611`: "The frame may be stale ... (hanging camera scenario e.g.)"). Used by `robots/lekiwi`, `reachy2`, `hope_jr` (grep) |
| robomimic / RLDS / OXE | Filter keys and named subsets (robomimic). Episode formats | No corruption generators. robomimic-MH has natural quality tiers, but not for LIBERO | inferred from the format docs read in earlier features. Not re-verified this session |
| W&B / MLflow / DVC / HF Hub | Lineage of artifacts. MLflow is already the repo's run store (`vlapoc.runs`) | Not generators | measured (`vlapoc/runs.py` `STAGES`, `stage_run`) |
| scikit-learn 1.9.1 (installed in `.venv`, not yet imported by `vlapoc/`) | `roc_auc_score`, `roc_curve` (best balanced accuracy = max over thresholds of (TPR+TNR)/2), `average_precision_score` | n/a | measured (`import sklearn` → 1.9.1). New import, no new install |

**Verdict: an adapter over the in-repo `derive_version`, plus small custom transforms and sklearn for scoring.** Nothing
off the shelf generates labelled corrupted LeRobot/LIBERO episodes. The mechanism (LeRobot writer pass, labels,
blindness check, lineage, MLflow) already exists in this repo. This feature adds:
- seven pure transform functions;
- two targeted mechanism changes (instruction edits, and the instruction counted as content in `compare_blind`);
- the owner's loader;
- a ground-truth-side scorer.

Do **not** use `modify_tasks`. It works in place and re-indexes tasks alphabetically, which would change the
derived copy's task order relative to the parent (quoted source above).

### Premises verified against our data (v0 = `data/lerobot/libero_spatial_10`, 1000 episodes, 200,340 frames)

Scripts: `data/scratch/gt_plan/premises.py`, `premises2.py` (scratch, gitignored). Run with
`.venv/bin/python data/scratch/gt_plan/premises.py` from the repo root.

| # | Premise | Result | Consequence |
|---|---|---|---|
| P1 | Action range | measured: dims 0–2 span exactly [−0.9375, 0.9375]; dims 3–5 span about [−0.37, 0.38]; gripper in [−1, 1]. 1.6% / 0.08% / 3.7% of dim-0/1/2 values sit at ±0.9375 | The **clean data never exceeds ±0.9375** on dims 0–2. Clipping to [−1, 1] (owner decision 1) leaves values in (0.9375, 1] that never occur in clean data. Record that fraction per label (`frac_beyond_clean_range`). See Open questions |
| P2 | Gripper values | measured: exactly two values {−1, +1} in 200,340/200,340 frames. Every episode starts at −1 (1000/1000). Quoted robosuite `panda_gripper.py:46`: "-1 => open, 1 => closed" | Gripper noise would make non-binary values, which never occur in clean data. **Action noise touches dims 0–5 only**, and every type keeps the gripper binary |
| P3 | Clip impact of 0.25σ_a noise on all 7 dims (σ_a from v0 `stats.json`) | measured: 0.50% / 0.20% / 1.49% of dim-0/1/2 values fall outside [−1, 1]; **50.1% of gripper values** do | Confirms P2. Without the gripper the clip bites on about 0.5–1.5% of translation values |
| P4 | Duplicate frames | measured: **0** of 199,340 consecutive frame pairs have both images byte-identical | A held (stale) frame is an exact duplicate that clean data never has. Dropped frames and idle padding are therefore findable by an exact-duplicate detector. S5 measures this rather than hiding it (Calibra-style detectors exist) |
| P5 | Identical-action runs | measured: the longest run of identical consecutive action vectors is **21** steps (99th percentile 9) | Any constant pad longer than 21 steps (the time-shift tail) is outside the clean distribution. Idle pauses are sized at ≤ 20 steps each |
| P6 | Where clean idle frames sit | measured: 436 frames with arm action exactly 0. Relative position quartiles 0.0 / 0.008 / 0.021 / 0.059 / 0.996, so they sit almost all at the episode start | Mid-episode pauses never occur in clean data. Insert pauses inside [0.2T, 0.8T] |
| P7 | Release geometry | measured: 944/1000 episodes have a final +1→−1 transition, at a median 0.92 of the episode. The closed carry span before it is ≥ 40 steps in 95% of them (median 63). Median steps after release: 14 | Gripper-release eligibility = has a final release with carry span ≥ 30 steps. A 20-step premature release stays inside the carry phase |
| P8 | Lengths | measured: 75 / 172.5 / 517 steps (min/median/max). fps = 20 (`meta/info.json`) | Two seconds = 40 steps (quoted RoboDrop × measured fps). Every episode (T ≥ 75) can take a 40-step shift |
| P9 | Episode identity | measured: `episode_id` = `<suite>/<task_name>/demo_N`. The physical order is grouped by task (`episode_index` 0–49 = spatial task 0, …). Task-constant catalog columns: `suite, task_name, libero_task_id, language_instruction, scene, objects, source, operator, collection_date, source_file` | A swapped instruction is always findable by cross-checking provenance (`episode_id`, `task_name`, physical position) against the instruction. See Design decision D1 |
| P10 | Content hash includes the instruction | quoted `vlapoc/catalog.py:127-148`: `episode_content_hash(task, action, state, image, image2)` hashes `task.encode("utf-8")` | The instruction is already **content** under the repo's own identity scheme. Counting `language_instruction` / `meta/episodes.tasks` as data-derived in `compare_blind` is consistent with that |
| P11 | v0 task order | measured: `meta/tasks.parquet` lists the 20 tasks in suite and task-id order (index 0–9 spatial, 10–19 libero_10), which is first-appearance order | The derived copy must keep this mapping exactly (AC2.3) |
| P12 | PYTHONHASHSEED quirk | quoted (data+lineage `refactor.md` l. 48): caused by `dataset_tools.py:549` `save_episode_tasks(list(set(tasks)))` on the old delete/merge path. The current writer path calls `list(set(tasks))` per episode (`dataset_writer.py:284`), and each LIBERO episode has one task | inferred: the current path is unaffected. AC2.4 measures it under seeds 0 and 1 |
| P13 | Disk | measured: 91 GiB free; v0 copy 6.9 GiB, v1 6.8 GiB; `data/hf_datasets_cache` 27 GiB and `data/lineage_scratch` 7.0 GiB (evaluation N4 leftovers, owner cleanup) | One full-scale derived version only. See "Disk and version budget" |

### Design decisions (planner's, with reasons; the executor may refine mechanics, not these choices)

- **D1. Swapped instructions keep true provenance.** The swap changes the episode's instruction, which is content:
  the LeRobot task string and `task_index`, `meta/episodes.tasks`, and catalog `language_instruction`. It does
  **not** change `episode_id`, `task_name`, `libero_task_id`, `suite`, `scene`, `objects`, `source_file`, or the
  physical position. That is what a real annotation error looks like: the collection campaign is right and the
  instruction is wrong. The cost is that a provenance-vs-instruction cross-check finds every swap. S5 measures this
  (heuristic H6) and the report must say it. The alternative needs opaque episode keys, a label-independent physical
  shuffle and hidden provenance columns, so it touches the score-table key and the row-order blind rule. It is listed
  as a non-blocking owner question.
- **D2. One combined full-scale version `v2`** with all seven types, rather than one version per type as RoboDrop
  did. Reasons:
  - disk: 7 × ~7 GB versus ~7 GB;
  - server training cost: one corrupted training arm instead of seven;
  - §4.5 needs the metric-targeted groups **in the same dataset** so that one attribution run can assign each group
    to a metric.
  Cost: the policy-level harm of each type cannot be isolated from one version (out of scope, below). Per-type
  recovery is still scored, type against clean.
- **D3. Budget: 13 episodes per type × 7 = 91 corrupted of 960 candidates (9.48%).** That is inside §2.5's 5–10%.
  Each type covers 13 distinct tasks, with both suites represented (7/6 split). Each task gets 4 or 5 corrupted
  episodes of different types, close to RoboDrop's "10% ... for every task" (4.8 of 48). 14 per type would be 98/960
  = 10.2%, over the bound.

### The seven corruption types (parameters marked quoted / measured / inferred)

| `corruption_type` | Spec row | Transform | Parameters | Length | Metric hypothesis (only where the PoC states one) |
|---|---|---|---|---|---|
| `action_noise` | §2.5 "Noisy ... actions" | a[:, 0:6] += N(0, (0.25·σ_a)²), then clip to [−1, 1]. Gripper untouched | 0.25·σ_a: quoted RoboDrop A.1. σ_a = v0 `stats.json` action std: as v1. Dims 0–5: inferred from P2/P3 (diverges from v1's 7-dim literal reading, recorded) | unchanged | none stated |
| `time_shift` | §2.5 "time-shifted actions" (teleoperation lag) | a'[t] = a[t + 40] for t < T−40. For the last 40 steps: arm 0, gripper = a[T−1, 6] ("operator stopped commanding") | 40 steps = quoted 2 s × measured 20 fps. Direction (actions lead observations, the logged command takes effect 2 s later) and tail fill: inferred, since RoboDrop states neither | unchanged | none stated |
| `instruction_swap` | §2.5 "Swapped or wrong language instructions" | task string := the instruction of another task in the **same suite**, uniform, seeded | Same-suite rule: inferred (keeps the instruction a valid one for the benchmark). The string must already exist in the parent's `tasks.parquet` | unchanged | none stated |
| `dropped_frames` | §2.5 "dropped frames" | Sample-and-hold of one camera's image: within a burst, frame t shows frame (burst start − 1). State and action are untouched | Burst length 1–10 frames: quoted LeRobot `read_latest(max_age_ms=500)` = 10 frames at 20 fps. One camera per burst, chosen uniformly; bursts until ≈ 20% of frames are held: inferred | unchanged | none stated |
| `detour` | §2.5 "Low-skill ... corrections" | Pick a constant-gripper window [t, t+20] inside [0.2T, 0.8T]. Play it forward, then backward (frames t+19 … t, arm actions negated), then forward again, then continue. Every frame is a real parent frame, with no consecutive duplicates | m = 20 frames (1 s): inferred. Negated delta actions roughly reverse the motion: inferred (LIBERO OSC_POSE delta control), see R5. Prior art: Bedi 2026 "detour", SCIZOR "Move Back and Forth" | +40 | none stated |
| `idle_padding` | §8.10 "pad some with idle or hesitant segments"; §2.5 "idle periods" | Two pauses of 20 frames at distinct random points in [0.2T, 0.8T]. Each holds the observation; arm action 0, gripper held (as the existing `idle_padding_edit`) | 2 × 20: inferred. Each pause ≤ 21 steps, the clean maximum idle run (P5). +40 steps = +23% of the median length | +40 | **speed, not reliability** (§8.10 p. 17; §4.5 p. 11 "idle or hesitant segments aimed at time-based metrics") |
| `gripper_release` | §8.10 "add gripper-release noise"; §4.5 "noisy gripper release near placement aimed at drops" | Premature release: gripper := −1 on the 20 steps before the episode's final +1→−1 transition | d = 20 steps (1 s): inferred. Eligibility: carry span ≥ 30 (P7). Prior art: Bedi 2026 "Early release" | unchanged | **reliability (drops), not speed** (§8.10 p. 17; §4.5 p. 11) |

All types use the existing per-episode generator `derive._episode_rng(seed, episode_id)`, so one episode's
corruption does not depend on which others are selected. Every label's `params` records: the spec source; every
parameter and whether it is quoted or inferred; the clip counts (`n_clipped`, `frac_clipped`,
`frac_beyond_clean_range`, all 0 for types that do not change action values); and the corrupted frame ranges in
**child** frame indices. Frame ranges make segment-level scoring possible later (§1 p. 3 "Segment / step"); this
feature does not score at that level.

### Disk and version budget

- **One full-scale derived version (`v2`, ~7 GB)**, inferred from v1's 6.8 GiB. The +80 frames of 26 lengthened
  episodes add about 0.1%.
- Transient: the writer's temp dir is renamed in place (no second copy). The HF datasets cache is isolated and
  deleted (`isolated_hf_datasets_cache`).
- Small-scale tests run on a ≤ 60-episode scratch version under `data/lineage_scratch/` (≈ 0.4 GB, inferred
  60/1000 × 6.9 GB) and are deleted afterwards.
- Expected free space after the feature: ≈ 84 GiB, above the 20 GiB floor.
- No per-type versions (D2). v1 is not rebuilt (owner decision 1).

### Where things live (consumer contract)

```
method side (blind)                                  ground-truth side (answer key)
───────────────────                                  ──────────────────────────────
vlapoc/method_data.py  open_for_method(subset_ref)   vlapoc/groundtruth.py
  verify_version + assert_blind (fence, offending)     corruption transforms (or vlapoc/corruptions.py)
  refuses ground-truth-derived subsets                 select_injection(), derive_injection() -> v2
  one dataset lineage per process (no diff leak)       load_answer_key(version)  (wraps derive.load_labels)
  -> MethodView: lerobot root, episode_index list,     score_recovery(version, score_table, column, direction)
     method-facing catalog columns, reference ids        -> recall@k, precision@k, AUROC, best bal. acc.,
  never imports labels / parents / groundtruth              per type and pooled, random baseline
                                                       MLflow stage run "score" (score digest + labels sha)
method code (heuristic rankers S5, later RoboDrop,
DemInf, CUPID) -> score table keyed by
(episode_id, content_hash) of the training subset  ───────────►  score_recovery
```

## Stories

### S1: Implement the seven corruption transforms with invariant checks on real episodes

- **Why:** §2.5 p. 5 and §8.10 p. 17 corruption types. Owner decisions 1 (clip) and 2 (scope).
- **Depends on:** —
- **Deliverables (proposed):** `vlapoc/corruptions.py`, with one constructor per type returning a
  `derive.EpisodeEdit`. Also `eligible(type, episode) -> bool`, and the shared clip helper `clip_actions(a) ->
  (a_clipped, stats)`. Script `scripts/verify_corruptions_s1.py`, which runs every transform **in memory** on real v0
  candidate episodes (no dataset write) and prints an invariant table. A before/after figure, one panel per type.
- **Acceptance criteria:**
  - [ ] AC1.1 Seven types exist with the names in the table above and the parameters stated there. Each label's
    `params` carries `source` (the spec quote and page), every parameter with a `quoted`/`inferred` tag, and
    `corrupted_frames` as a list of [start, end) child frame ranges. [PoC §2.5 p.5, §8.10 p.17; vla-build]
  - [ ] AC1.2 Range: on ≥ 100 eligible v0 candidate episodes per type, every corrupted action lies in [−1, 1]. Every
    gripper value is in {−1, +1} exactly. [owner decision 1; P2 measured]
  - [ ] AC1.3 Clip accounting: for `action_noise`, `params` records `n_clipped`, `frac_clipped` and
    `frac_beyond_clean_range` (values with 0.9375 < |a| ≤ 1 on dims 0–2). The script prints their totals over the
    test episodes; expect `frac_clipped` around 0.5–1.5% on dims 0–2 (inferred from P3). For the other six types the
    counts are recorded as 0 and are asserted 0. [owner decision 1]
  - [ ] AC1.4 Lengths and content, per type:
    - `action_noise`, `time_shift`, `instruction_swap`, `dropped_frames` and `gripper_release` keep T.
    - `detour` and `idle_padding` give T + 40.
    - Every image in a `detour` / `idle_padding` / `dropped_frames` output is byte-identical to some image of the
      same parent episode.
    - `detour` produces no consecutive identical frame pair.
    - `instruction_swap` outputs a task string from the parent's `tasks.parquet`, in the same suite, ≠ the original.
    - Only the declared arrays change (e.g. `dropped_frames` leaves `action` and `state` bit-identical).
    [PoC §2.5 p.5; design table]
  - [ ] AC1.5 Determinism: the same (seed, episode_id) gives the same output content hash in two fresh processes.
    A different seed gives a different hash for every type except `instruction_swap` with a single same-suite
    alternative (not the case here: 9 alternatives). [vla-build]
  - [ ] AC1.6 Eligibility counts printed per type over the 960 v0 candidates. For `gripper_release` it matches P7's
    measurement (expect ≈ 900+, inferred from 944/1000 with carry ≥ 40 in 95%). Each type has ≥ 13 eligible episodes
    in ≥ 13 tasks. [design D3]
  - [ ] AC1.7 Grounding docstring cites PoC §2.5 p.5, §8.10 p.17, §4.5 p.11, and RoboDrop App. A.1 p.13 for the two
    types it defines. It states each inferred parameter as inferred, and the dims-0–5 divergence from v1. [vla-build]
- **Verify by:** `.venv/bin/python scripts/verify_corruptions_s1.py --version v0 --per-type 100 --seed 0`, then the
  same command in a second process with the hash comparison flag.
- **Demo value:** a seven-panel figure of the action/gripper traces and a filmstrip before and after for one episode
  per type, plus the invariant table with clip fractions.

### S2: Extend `derive_version` for instruction edits and count the instruction as content in the blind check

- **Why:** the §2.5 p. 5 "Swapped or wrong language instructions" row cannot be built today (`derive.py:394-395`
  refuses task edits). Under rule 2 (§2.1 item 4 p. 3), the instruction is content (P10), not bookkeeping.
- **Depends on:** S1
- **Deliverables (proposed):**
  - `derive.py`: allow `ep["task"]` to differ, if the new task is in the parent's `tasks.parquet` (refuse new
    strings). Pre-register the parent's tasks in parent order on the new dataset before the first `save_episode`.
    Rebuild the child table's `language_instruction` from the stored task.
  - `lineage.py`: move `language_instruction` into `DATA_DERIVED_COLUMNS`, and `tasks` out of
    `META_BOOKKEEPING_COLUMNS` into the data-derived set.
  - Labels gain one column, `hypothesis` (the §4.5 metric hypothesis or `none stated`). `corrupted_frames` stays
    inside `params`.
  - A small-scale scratch test script `scripts/verify_derive_gt_s2.py`.
- **Acceptance criteria:**
  - [ ] AC2.1 A scratch derivation (≤ 60-episode scratch version, built with the existing scratch-store helper) with
    one edit of each of the seven types registers with `check_blind == []`. Its `labels.parquet` has one row per edit
    with the columns `episode_id, parent_content_hash, content_hash, corruption_type, params, seed, hypothesis`.
    [PoC §2.1 item 4 p.3]
  - [ ] AC2.2 A swap to a string not in the parent's tasks raises `LineageError` before anything is written. So does
    an edit of a reference episode (the existing guard, re-run). [PoC §8.7 p.15; design]
  - [ ] AC2.3 The derived copy's `meta/tasks.parquet` (task → task_index) equals the parent's. The swapped episode's
    data rows carry the **target** task's index. [P11; LeRobot `save_episode_tasks` quoted]
  - [ ] AC2.4 Run the AC2.1 derivation under `PYTHONHASHSEED=0` and `=1`. The results have identical
    `tasks.parquet`, labels rows digest, content hashes and `dataset_version_id`. Report any other differing file
    by name. [brief quirk; P12]
  - [ ] AC2.5 No regression: `check_blind(v1)` still returns `['episode_index']` and v1 stays fenced, and
    `check_blind(v0) == []`. The existing `scripts/verify_derive_s4.py` and `scripts/verify_blind_s0.py` pass
    unchanged, or every changed expectation is listed with its reason. [evaluation S0 ruling A]
  - [ ] AC2.6 A deliberate leak is still caught: a test derivation that changes `task_name` (provenance) for an
    edited episode fails `compare_blind`, with `task_name` named as offending. [PoC §2.1 item 4 p.3]
  - [ ] AC2.7 The docstrings of `derive_version` and `compare_blind` state the reclassification and its reason:
    the instruction is hashed content (`catalog.py:127-148`), and provenance stays bookkeeping, as D1 says. [vla-build]
- **Verify by:** `.venv/bin/python scripts/verify_derive_gt_s2.py` (run twice, with `PYTHONHASHSEED=0` and `=1`),
  then `scripts/verify_derive_s4.py` and `scripts/verify_blind_s0.py`. Delete the scratch store afterwards and record
  `df -h`.
- **Demo value:** a blind-check table showing that an instruction swap passes while a provenance leak is caught.

### S3: Add the method-facing loader and the ground-truth boundary (owner decision 3)

- **Why:** owner decision 3, "Enforce now", and PoC §2.1 item 4 p. 3 ("flags exactly those" on its own). This
  closes the evaluation-layer known issue "The v1 fence is recorded but not enforced at method entry points".
- **Depends on:** S2
- **Deliverables (proposed):**
  - `vlapoc/method_data.py` with `open_for_method(subset_ref) -> MethodView`. The view holds `lerobot_root`,
    `episode_indices` (from `Subset.resolve`), `catalog` (the version's episode-table rows for the subset, restricted
    to `catalog.CATALOG_COLUMNS`, all of which `compare_blind` checks; exact list in the docstring), `reference_ids` (the version's inherited `reference` subset) and
    `version_id`. It also offers `iter_episodes()`.
  - `vlapoc/groundtruth.py` with `load_answer_key(version)`, the only label reader for evaluation code. It wraps
    `derive.load_labels` and `lineage.parent_hashes`.
  - A boundary test, `scripts/verify_method_boundary_s3.py`.
- **Acceptance criteria:**
  - [ ] AC3.1 `open_for_method` calls `verify_version` and `assert_blind` on the subset's version every time.
    `open_for_method("v1/candidate")` raises `LineageError` naming the fence. `v0/candidate` opens. The S2 scratch
    derived version opens. A version with one byte changed in a data file raises `VersionIntegrityError`.
    [owner decision 3; evaluation S0]
  - [ ] AC3.2 It refuses ground-truth-derived subsets: any subset whose derivation chain (`parent_subset_id`
    recursion) contains a ground-truth op (`oracle_drop`, or anything tagged `groundtruth`). The test builds one in
    a scratch store and shows the refusal. [owner decision 3]
  - [ ] AC3.3 Diff guard: in one process, after opening a subset of version X, opening a subset of any ancestor or
    descendant of X raises (e.g. `v0/candidate` after `v2/candidate`, scratch equivalent). Comparing hashes across
    lineage would name the edited episodes (`lineage.parent_hashes` docstring, quoted). [owner decision 3 "by
    accident"; PoC §2.1 item 4 p.3]
  - [ ] AC3.4 Static boundary: an AST import scan shows that neither `vlapoc/method_data.py` nor any module under
    `vlapoc/methods/` imports `vlapoc.groundtruth`, `derive.load_labels`, `lineage.parent_hashes`, or opens
    `labels.parquet` / `parents.parquet` / `fence.json` by path. The scan fails on a planted violation. [owner
    decision 3]
  - [ ] AC3.5 The `MethodView.catalog` columns are listed in the docstring, and they pass `compare_blind` for every
    derived version (they are a subset of what `compare_blind` checks). [PoC §2.1 item 4 p.3]
  - [ ] AC3.6 Grounding docstrings: PoC §2.1 item 4 p.3, owner decision 3 (brief, 2026-10-15), and the evaluation
    S0 fence. [vla-build]
- **Verify by:** `.venv/bin/python scripts/verify_method_boundary_s3.py` (uses a scratch lineage store for the
  tamper and oracle cases).
- **Demo value:** a refusal table that shows, for each version or subset, whether a method may open it and the
  reason (v1: fenced; v0: open; tampered: refused; oracle subset: refused).

### S4: Derive the full-scale injected version `v2` from v0

- **Why:** §10 p. 20 "Ground truth" deliverable, §2.5 p. 5 budget, §8.7 p. 15 clean references.
- **Depends on:** S1, S2, S3
- **Deliverables (proposed):**
  - `groundtruth.select_injection(parent, per_type=13, seed=0)`, a seeded and stratified assignment of
    (episode → type) over the parent's `candidate` split that respects eligibility.
  - `groundtruth.derive_injection(...)`, which calls `derive_version` with the seven types.
  - A CLI, `.venv/bin/python -m vlapoc.groundtruth derive --parent v0 --name v2 --per-type 13 --seed 0`, recorded as
    an MLflow `derive` stage run (via `runs.derive_run` generalized or an equivalent).
  - `artifacts/s4_*`.
- **Acceptance criteria:**
  - [ ] AC4.1 `v2` is registered with 1000 episodes and `labels.parquet` has 91 rows: 13 per type, 7 types. Every
    labelled episode is in `v0/candidate`, and none is in `v0/reference`. [PoC §2.5 p.5; §8.7 p.15; D3]
  - [ ] AC4.2 Stratification, printed as a task × type table:
    - each type spans 13 distinct tasks, with both suites present (6/7 or 7/6);
    - every task has 4 or 5 corrupted episodes, with no type repeated within a task;
    - 91/960 = 9.48% of candidates is printed.
    [PoC §2.5 p.5 "5–10%"; RoboDrop A.1 "10% ... for every task"]
  - [ ] AC4.3 `check_blind(v2) == []`. `verify_version(v2)` passes. `v2/reference` (40) and `v2/candidate` (960)
    are inherited, and the 40 reference content hashes equal v0's. [PoC §2.1 item 4 p.3; §8.3 p.14]
  - [ ] AC4.4 Re-check the invariants on the stored bytes. Read every labelled episode back with
    `read_stored_episode`. It hashes to its label's `content_hash`, and S1's AC1.2–AC1.4 invariants hold on the
    stored data, not only in memory. The 909 unedited episodes hash to their v0 hashes. [vla-build "verify against
    the artifact"]
  - [ ] AC4.5 Clip totals for the 13 `action_noise` labels are reported (sum of `n_clipped`, `frac_clipped`,
    `frac_beyond_clean_range`), and match a recount from the stored bytes. [owner decision 1]
  - [ ] AC4.6 Disk and time recorded: `du -sh` of the new copy, `df -h` before and after (free stays > 20 GiB), the
    derivation timing from the returned `timing_s`, and the MLflow run id. [brief constraint]
  - [ ] AC4.7 Lineage: `lineage trace <derive run>` reaches v0 and its raw files, and the derive run's tags include
    `labels_sha256`. [PoC §2.1 item 1 p.3]
- **Verify by:** commit first (the `derive` CLI refuses a dirty `vlapoc/`), then run the CLI. Then
  `.venv/bin/python -m vlapoc.lineage version check-blind v2` and `scripts/verify_injection_s4.py` (AC4.4–4.5).
- **Demo value:** the injected dataset itself, plus a task × type allocation table, a before/after filmstrip of one
  stored episode per type, and clip and disk numbers.

### S5: Score recovery on the ground-truth side and run the heuristic baselines through the loader

- **Why:** §2.1 item 4 p. 3 ("recovered 90% of the injected bad episodes"), §2.5 p. 5 ("how many the method ranks
  near the top"), RoboDrop §5.1 p. 7 (AUROC, best balanced accuracy), and §5 p. 11–12 (the heuristics family). This
  is the consumer contract that every later screening and attribution feature will be scored with.
- **Depends on:** S3, S4
- **Deliverables (proposed):**
  - `groundtruth.score_recovery(version, score_table, column, direction="highest", subset="candidate") -> dict`.
    It validates keys with the existing exact-key check (`lineage._check_score_keys`, made public), so a table
    computed on v0's hashes is rejected.
  - Heuristic rankers in `vlapoc/methods/heuristics.py`, which read only via `open_for_method`:
    - H1 `length_steps` (RoboDrop's "Length" baseline);
    - H2 `idle_steps`;
    - H3 `gripper_toggles`;
    - H4 `smoothness` (direction per its catalog definition);
    - H5 count of consecutive byte-identical image pairs (frozen-frame detector, Calibra-style);
    - H6 instruction ≠ the instruction of the episode's `task_name` (D1's provenance cross-check).
  - Recorded as MLflow runs: one `screen` stage run per heuristic (real, not stand-in), and one ground-truth
    scoring run (a new stage, e.g. `score`) carrying the score-table digest, the `labels_sha256` and the metrics
    JSON.
  - A CLI, `.venv/bin/python -m vlapoc.groundtruth score --version v2 --scores <parquet> --column <c>`.
- **Acceptance criteria:**
  - [ ] AC5.1 Each metric matches a hand-computed value on synthetic fixtures:
    - recall@k with k = number of positives;
    - precision@k and recall@k at k ∈ {n_pos, 2·n_pos};
    - AUROC (sklearn `roc_auc_score`);
    - best balanced accuracy (max over `roc_curve` thresholds of (TPR+TNR)/2);
    - average precision.
    The fixtures cover a perfect ranker (recall@k = 1, AUROC = 1), a reversed ranker (AUROC = 0), a constant score
    (AUROC = 0.5, and recall@k = k/N under fractional tie-counting, inferred tie rule, stated in the docstring) and
    a 3-tie case. [RoboDrop §5.1 p.7; PoC §2.1 item 4 p.3]
  - [ ] AC5.2 Results come per type (13 positives of that type vs the 869 clean candidates, RoboDrop's
    one-type-vs-clean setting) and pooled (91 vs 869). Each comes with its random-ranker expectation (recall@k =
    k/N, AUROC 0.5) and n. [RoboDrop A.1/§5.1; PoC §2.5 p.5]
  - [ ] AC5.3 Key safety: a v0-keyed score table (same episode_ids, v0 hashes) is rejected by `score_recovery`.
    So is a table missing one candidate. [PoC §9.1 Step 1 p.18; lineage `_check_score_keys`]
  - [ ] AC5.4 Run heuristics H1–H6 on `v2/candidate` through `open_for_method` and score them. Output: a 6 × 8
    table (heuristic × {7 types, pooled}) of AUROC and recall@k, saved as `artifacts/s5_recovery_heuristics.csv`
    with its command. **No expected values are set.** These are measurements, and the report states which types are
    trivially findable and which are not. [PoC §5 p.11–12; §2.2 p.4]
  - [ ] AC5.5 Positional-leak demonstration (ground-truth side, labelled "leak check, not a method"): score a ranking
    by `episode_index` against v1's labels and against v2's. v1 is expected to give recall@100 = 1.0 (its edits are
    indices 900–999, quoted fence.json). v2 is expected near the random line. The two numbers are printed side by
    side. [evaluation S0; PoC §2.1 item 4 p.3]
  - [ ] AC5.6 Every scoring call is an MLflow run whose tags include the score-table digest
    (`lineage.score_table_digest`), the version id and `labels_sha256`. `lineage trace` on it reaches v2. [PoC §2.1
    item 1 p.3]
  - [ ] AC5.7 Grounding docstrings: PoC §2.1 item 4 p.3, §2.5 p.5, RoboDrop §5.1 p.7 and Table 7 (oracle), and the
    tie rule marked inferred. [vla-build]
- **Verify by:** `.venv/bin/python scripts/verify_recovery_s5.py` (fixtures, key safety, leak check), then
  `.venv/bin/python -m vlapoc.methods.heuristics --subset v2/candidate --out data/eval/groundtruth/heuristics/`
  followed by the scoring CLI for each output.
- **Demo value:** the first real recovery table in the PoC, "what cheap heuristics already find, per corruption
  type", and the v1-vs-v2 leak check that shows why the fence and loader exist.

### S6: Prove v2 is trainable and evaluable locally, and write the server runbook

- **Why:** consumers. Per-subset normalization (evaluation owner decision 3). PoC §8.10 pp. 17–18 "Verify before
  claiming", and §2.6 p. 5 and §9.1 Step 4 p. 19 (the gate decides whether the corruption changed the policy).
  Everything that needs training or rollouts is pending server.
- **Depends on:** S4 (S5 for the runbook's scoring step)
- **Deliverables (proposed):** `scripts/verify_consumers_s6.py`; `artifacts/server-runbook.md` following the
  evaluation feature's runbook pattern (`tracker/features/2026-10-01-evaluation-layer/artifacts/server-runbook.md`).
- **Acceptance criteria:**
  - [ ] AC6.1 Per-subset stats: `python -m vlapoc.subset_stats --check v2` (the aggregate over all episodes equals
    v2's `stats.json`, to the AC8.1 tolerance). `--subset v2/candidate` gives a digest different from
    `v0/candidate`'s. The per-dim action mean/std of v2/candidate vs v0/candidate is printed. [evaluation decision
    3; S8 AC8.1]
  - [ ] AC6.2 Local training smoke: a SmolVLA fine-tune of 2 steps (the S8 configuration, MPS) on a small v2 subset
    that contains at least one episode of each of the seven types. The subset is built ground-truth-side in a
    **scratch** lineage store, so no label-derived subset lands in `data/lineage/`. The run completes and records
    `train.subset_stats.sha256`. [evaluation S8; owner decision 3]
  - [ ] AC6.3 Evaluable: v2's 20 task strings equal v0's, so the eval task list is unchanged. The human reference
    (`data/eval/human_reference/`) digest is unchanged before and after the feature, so the HRT anchor stays the
    1000 v0 originals (evaluation report). [PoC §3.5 A p.7–8; evaluation report]
  - [ ] AC6.4 The runbook lists every pending-server item with its command, its inputs (version and subset ids),
    its outputs, a disk estimate (inferred: ~7 GB HF cache per distinct 960-episode subset, measured data+lineage
    R1), and its order:
    - **GT-R1 (pending server)** SmolVLA on `v2/candidate`, with exactly the AC8.5 configuration and seeds.
    - **GT-R2 (pending server)** each checkpoint evaluated with the AC5.8 protocol (fixed batch plan,
      attribution-grade).
    - **GT-R3 (pending server)** returns (AC6.7 pattern).
    - **GT-R4 (pending server)** verdict `v2/candidate` vs `v0/candidate` (AC8.5 runs): macro-KS gate, then
      per-metric CIs, family W declared. The outcome is a finding, not an acceptance condition.
    - **GT-R5 (pending server, optional)** the oracle arm: `v2/candidate` minus the 91 labelled episodes, created
      server-side as a ground-truth-tagged subset (RoboDrop Table 7 "Oracle").
    - **GT-R6 (pending later features)** the hand-off: a screening or attribution score table goes through
      `score_recovery`. The §4.5 metric check (idle → time metrics, gripper → reliability) needs per-metric
      attribution, which is a later feature.
    [PoC §8.10 p.17–18; §2.6 p.5; §9.1 Step 4 p.19; brief]
  - [ ] AC6.5 The runbook states plainly that LIBERO has no ghost-event (drop) rule (evaluation decision 6). The
    gripper-release "drops" hypothesis can only show up as success, F(τ) or timeouts there. [PoC §4.5 p.11;
    evaluation report]
- **Verify by:** `.venv/bin/python scripts/verify_consumers_s6.py`, then read `artifacts/server-runbook.md` against
  AC6.4's list.
- **Demo value:** "the corrupted dataset trains": normalization shift numbers, the smoke run id, and a runbook a
  server operator can execute without guessing.

## Out of scope (deliberately)

- **Altered camera viewpoint.** Owner-approved out of scope (decision 2): it needs simulator re-rendering. The report
  must list it as an owner-approved divergence from §2.1 item 4 / §2.5.
- **Sim-rendered idle or hesitant frames.** Pauses repeat stored frames (exact duplicates, P4), for the same
  re-rendering reason. S5 measures how findable that makes them.
- **One version per corruption type** (RoboDrop's design). Disk (7 × ~7 GB) and 7× server training. Recorded as a
  divergence from RoboDrop A.1 (D2).
- **Policy-level harm per type.** It would need 7 training arms. Only the pooled v2-vs-v0 verdict is planned
  (GT-R4).
- **Running RoboDrop, DemInf or CUPID**, and "same dataset, opposite curations" (§4.5 p. 11). These are later
  features. This one supplies the labels, the version and the scorer they will use.
- **Segment-level recovery scoring.** Labels carry frame ranges, but scoring is episode-level only.
- **Confidence intervals on recovery metrics.** With 13 positives per type, one episode moves recall by 7.7 points,
  so the report states n instead.
- **Rebuilding v1.** Owner decision 1. It stays fenced.
- **Cross-suite instruction swaps** and novel (non-benchmark) instruction strings.
- **Injection on the PhAIL fine-tuning set** (§10 "Real-robot stage").

## Open questions

- **BLOCKING:** none. Each decision below has a reversible default that does not contradict an owner ruling.
- Non-blocking, owner FYI, **clip bound** (decision 1 applied as ruled): the clean data's translation dims never
  exceed ±0.9375 (P1, measured). Clipping to [−1, 1] therefore leaves noised values in (0.9375, 1] that are
  impossible *in this data*, though valid for the environment. AC1.3/AC4.5 record that fraction. Switching to
  ±0.9375 is a one-parameter change plus a re-derivation (~9 min) if the owner wants it.
- Non-blocking, owner question, **swap provenance (D1)**: the default keeps `episode_id`, `task_name` and physical
  position true, as in a real mislabelled dataset, so a provenance cross-check finds every swap (S5 H6 measures it).
  The fully hidden alternative needs opaque method-facing episode keys, a label-independent physical shuffle (which
  changes `compare_blind`'s row-order rule) and dropping the provenance columns from the method view. That is roughly
  one more story.
- Non-blocking, assumption: "low-skill or hesitant" (§2.5) and "idle or hesitant padding" (§8.10) are built as two
  types. The brief lists them separately: §2.5's "corrections" becomes `detour`, and "idle periods" plus §8.10
  become `idle_padding`.
- Non-blocking, assumption: action noise excludes the gripper (P2/P3). The gripper's own corruption is
  `gripper_release`.
- Non-blocking, assumption: time-shift direction = actions lead observations by 2 s (teleoperation lag), with an
  idle tail. RoboDrop does not specify either.

## Risks / premises to verify during execution

- **R1** Pre-registering tasks in LeRobot 0.6.1's writer. Checked in the source: `LeRobotDataset.create` builds
  `DatasetWriter(meta=obj.meta, ...)` (`lerobot_dataset.py:779-780`), so `ds.meta.save_episode_tasks(parent_tasks)`
  before the first `save_episode` registers the tasks on the metadata object the writer then uses. It is still
  confirmed on the bytes by AC2.3.
- **R2** Counting `language_instruction` / `tasks` as content must not loosen any other check. AC2.5 (v1 still
  offending) and AC2.6 (a provenance leak still caught) guard this.
- **R3** Disk: keep free space > 20 GiB. `data/hf_datasets_cache` (27 GiB) and `data/lineage_scratch` (7 GiB) are
  pre-existing leftovers for owner cleanup. Don't delete them without asking. Use `isolated_hf_datasets_cache` for
  every load.
- **R4** The `lineage derive` / `build` CLIs refuse a dirty `vlapoc/` or `scripts/`. Commit before the S4
  derivation, so the recorded git SHA is the code that ran.
- **R5** The `detour` reverse actions are an approximation (inferred: negated OSC_POSE deltas). Optional check if
  cheap: replay one detour's actions in LIBERO from the stored start state, and compare the end-effector at the
  bottom of the reversal with frame t. Otherwise label it inferred in the report.
- **R6** Per-type n = 13. Recovery numbers are coarse (7.7-point steps). Report n with every number.
- **R7** `_derived_episode_table` copies parent rows and refreshes only the data-derived columns. It must now also
  refresh `language_instruction` from the stored task, or `compare_blind` and the catalog will disagree with the
  data.
- **R8** The heuristics module is the first method-side code. If it needs a column the loader does not expose, the
  fix goes in the loader's documented column list (AC3.5), not around it.
- **R9** The time-shift tail (40 zero-arm steps) raises `idle_steps` to ≥ 40, beyond the clean maximum of 21. That is
  expected and visible in H2. It is not a leak, because it is content.
- **R10** A gripper-release episode whose final release is the only one, at < 0.5 of the episode (min relative
  position 0.367, P7), may not be "near placement". The executor reports how many selected episodes have their
  final release before 0.8T.
