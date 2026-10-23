# Plan: dashboard (demonstration surface), cluster and pattern discovery, server runbook

Round 1 (planner, 2026-10-22). Branch `dashboard` at `046c38d`. Evidence labels: **quoted** (source and page),
**measured** (a command run in this session; scratch outputs under `data/scratch/dash_plan/`, reproducible), **inferred**
(my reasoning, unverified).

## Changes in this revision (round 1, folded in mid-planning)
- The owner addition and follow-up (brief.md, "Addition from the owner" and "Owner follow-up") are folded in. The
  §9.2 discrepancy ("bad episodes are not always harmful") is now first-class. **New S9** covers the estimated half
  in Round A: planted labels × per-metric influence, with H1-H8 as a secondary layer. **New S10** covers the causal
  half in Round B: per-type "remove all of type X" arms and a clean-but-harmful arm vs shared clean random-13
  controls, under the two-step verdict, with a stated bound-not-proof wording.
- S5 gains a fifth page ("Bad vs harmful", §9.2). S6's demo covers it. S8's Round B covers the planted-type arms.
- The dataset question is now decided by evidence in favour of `v3/candidate` (it stays BLOCKING only because it
  replaces a recorded owner ruling). A second BLOCKING question covers the planted-type Round B budget.

## Request

The owner asked for the PoC's "new and final feature": a dashboard to present the project to stakeholders, plus the
ability to "pinpoint to exact clusters of data and identify patterns within those clusters that harm or boost a specific
policy", plus, as a separate story, a step-by-step runbook a fresh Claude Code session on the GPU server can follow end
to end, "all the way to attribution and real dashboard creation" ([brief.md](brief.md), verbatim there). Binding owner
answers (brief.md): three cluster lenses (label groups, rule-like slices, the four metric groups; **not** visual
clusters); a **live local app**; **synthetic scores, labelled** until server results exist, with real results dropping
in with no code change; two-round confirmation (clusters from scores are *estimated*; a cluster is *confirmed* only after
removal, retraining and the two-step verdict); the runbook is its own story. **Owner addition (mid-planning):** "show a
real pattern that would show that bad episodes are not always harmful for a policy". **Follow-up direction:** the
primary definition of "bad" is the planted corruption labels on v3 (read only after attribution, by scoring code). H1-H8
are an optional secondary "practical filter" layer. No new screening method in this feature.

## Spec grounding

PoC = `docs/VLA_Data_Attribution_PoC.pdf`, read this session from the PDF (`data/scratch/poc.txt`, pp. 3-11 and 18-20 in full).

- **§2.7 "Demonstration surface" (p. 5)** (quoted): "A dashboard: select a failed rollout, see its failure type, the most
  influential training episodes (with video) and their source datasets; or browse a ranked list of harmful episodes,
  aggregated by any catalog column. PhAIL's public run explorer, which shows each rollout with synchronized video,
  telemetry and event annotations, is a good design reference for the rollout side of this view." This is the
  dashboard's required content, two views.
- **§1 (p. 3)** (quoted): "Per-episode scores become most valuable when aggregated by metadata. Findings such as 'demos
  from operator 3 hurt drawer tasks' or 'episodes over 400 steps are net harmful' are usually what stakeholders act on."
  Lens 1 (label groups) and lens 2 (rule-like slices) both come from this sentence. The second example is a rule
  over a continuous descriptor, which is what a slice finder outputs.
- **§2.2 (p. 4)** (quoted): "Each episode gets a stable ID and a metadata row: source, task, language instruction, scene,
  objects, operator, collection date, episode length, and basic action statistics (smoothness, idle time, gripper
  toggles) ... Per-episode attribution scores become most valuable when aggregated by these columns."
- **§2.1 item 5 (p. 3)** (quoted): "Leave-one-group-out ablations. Drop one data source or task group, retrain, and
  measure the change. This is coarse but unambiguous, and it gives you a reference result for everything else." Round B
  (confirmation arms) is this item applied to a discovered cluster.
- **§2.6 (p. 5)** (quoted): "Each of these is a close comparison between two policies, so use the PhAIL significance gate
  (Section 3.4) to decide whether the change is real before claiming it."
- **§3.4 (p. 7)** (quoted): "Declared targets. Scalars can rank policies in opposite orders. | Every attribution result
  states which functional it explains, following PhAIL's disclosure principle." Every number the dashboard and the
  cluster stage show carries its `functionals.json` declaration.
- **§4.2 (pp. 9-10)** (quoted): "Results are reported as an episode × metric matrix. Episodes then fall into four groups:
  Consistent helpers ... Consistent harmers ... Trade-off episodes ... Metric-specific". This is lens 3. It uses the
  owner-approved ≥2-of-5 rule (`analysis.DIVERGENCE_NOTE`).
- **§4.4 (pp. 10-11)** (quoted): "Two-step verdict per intervention. First the macro-KS gate ... Then each metric's
  bootstrap CI gives the direction"; "correct for it (e.g. Bonferroni or Holm across metrics)"; "Label every claim with
  its metric and time cap τ".
- **§4.5 (p. 11)** (quoted): "Metric-targeted injections ... Check whether attribution assigns each corrupted group to the
  expected metric"; "Visualisation. An episode × metric heatmap, and a scatter of throughput influence against
  reliability influence with the four groups as quadrants."
- **§9.1 Step 4 (p. 19)** (quoted): "Influence estimates can be wrong. This step turns them from claims into verified
  results." This sentence is the reason for the estimated vs confirmed status.
- **§9.2 (p. 19)** (quoted): "A second compelling output is which kind of harm each data group causes — drops and safety
  stops versus slowness — which a single success rate cannot show." Each cluster's pattern is described per metric.
- **§9.2 (p. 19)** (quoted), the owner's addition: "The most compelling output is where screening and outcome
  attribution disagree: episodes that look bad but do not hurt the policy, and episodes that look clean but do." Here
  "look bad" is the planted label (primary, owner direction) or an H1-H8 flag (secondary). "Hurt" is per-metric
  influence (estimated) and then a removal arm (causal).
- **§9.1 Step 1 (p. 18)** (quoted): "Its verdicts are stored rather than acted on, so they can later be compared with
  what actually happened." This is the screening-vs-outcome comparison S9 implements. DemInf/RoboDrop screening itself
  stays out (owner direction).
- **§2.1 item 4 / §2.5 (pp. 3, 5)** (quoted): "Inject 5–10% corrupted episodes and measure how many the method ranks
  near the top." The v3 answer key (91/960 = 9.48%, 13 per type) is the planted "bad".
- **§10 (p. 20)** (quoted), row "Verification & demo": "Counterfactual removal tests; dashboard linking failed rollouts to
  influential episodes."

Rules the build must follow (from the sections above and earlier owner rulings):
1. Every displayed score names its functional (column, functional, τ, rollout subset, sign), from `functionals.json`
   (§3.4, §4.4).
2. Clusters derived from attribution scores are **estimated**. Only a removal arm that passes the two-step verdict
   (macro-KS gate, then per-metric Holm CIs) can make one **confirmed** (§4.4, §9.1 Step 4, brief).
3. Blindness (§2.1 item 4 p. 3; owner decision 3): cluster discovery reads training data only through
   `open_for_method`. The v3 answer key is read only by a separate scoring step, after the cluster run exists.
4. SYNTHETIC data is labelled everywhere it appears. The label can only disappear because of provenance in the lineage
   store, never because of a code switch (brief, owner answer).
5. Reading results means reading the lineage store (MLflow runs and their recorded, sha-checked artifacts), not ad-hoc
   paths (brief constraint).

Papers (read for this plan):
- **CUPID** `docs/papers/CUPID_Influence_Functions_arXiv-2506.19121.pdf` p. 11 (quoted): "one could approximate group
  effects [14] via random sampling or limit influence estimation to smaller data subsets identified using coarse-grained
  heuristics." What we take from it: summed or averaged member influence is only a first-order *estimate* of a group's
  effect. That is why a cluster's effect is labelled estimated.
- **DataMIL** `docs/papers/DataMIL_Datamodels_arXiv-2505.09603.pdf` p. 6 (quoted): "prior works clusters samples by class
  or task and then evaluates cluster-level influence instead of individual examples ... Fine-grained clusters offer
  precise selection but suffer from high noise". What we take from it: cluster-level aggregation is established
  practice and reduces per-episode noise, and lens 1 (task and scene groups) is the "by class or task" case.
- **PhAIL** `docs/papers/PhAIL_Real-Robot_VLA_Benchmark_arXiv-2605.29710.pdf` p. 2 (quoted): "per-rollout synchronized
  video, telemetry, and event annotations auditable through a public run-explorer interface"; p. 6: "Every rollout is
  published with synchronized video, telemetry, and event annotations, browsable in a run-explorer interface (Figure 4,
  Appendix B)". This is the design reference for the rollout page: video, outcome (T, event, censored, failure type)
  and the per-step trace.
- **CUPID** Fig. 4 caption, p. 8 (quoted, checked against the PDF this session): "Although quality measures (e.g.,
  DemInf, CUPID-QUALITY) help in mixed-quality settings ..., they degrade performance when higher-quality
  demonstrations induce brittle strategies at test time (TuckBox), or when quality is not the primary factor limiting
  policy success (Bookshelf)." This is published precedent for §9.2: looking good or bad and helping or hurting can
  come apart.

## What already exists (measured, this session)

| Need | Exists | Evidence |
|---|---|---|
| Per-episode, per-metric harm scores, functionals | `attribution.attribute_run` writes `attribution_scores.parquet` (`infl_`/`harm_` per family-W return and per failure type, all-rollout and `__same_task`), `functionals.json`, `agreement.json` as MLflow outputs | measured: `vlapoc/attribution.py` lines 626-662 |
| Rollout × episode influence matrix | `rollout_episode_influence.npy` is **written to disk but not recorded as an MLflow output**, and no row/column index is saved with it (rows = featurized rollouts sorted by `traj_episode_index`, columns = `subset.episodes()` order) | measured: lines 520-539, 626-662 (only three `output_file` calls). This is a gap: §2.7's first view needs it |
| Four groups, aggregate-by, heatmap, scatter | `vlapoc/analysis.py` (`metric_groups`, `aggregate_by` with bootstrap CI, `plot_heatmap`, `plot_tradeoff`, `axis_label` from declared functionals, `synthetic` flag in titles) | measured: read in full |
| Label-free behaviour descriptors, method-side | `vlapoc/methods/heuristics.py` `compute(view)`: H1-H8 incl. `length_vs_task`, `post_release_share_vs_task`, `duplicate_frame_pairs`, `instruction_mismatch`, all through `open_for_method` | measured: module docstring |
| Catalog columns that vary | v0/v3 catalog: 21 columns. Groupable and varying: `task_name` (20), `scene`, `suite` (2), `source_file`. `operator`/`collection_date` are None, one `source` | measured: `data/lineage/catalog/*/episodes.parquet` dtypes; cardinalities quoted in brief, consistent |
| Removal arms, verdicts, seed summary | `runs.select_run` ops `drop_top_k`, `drop_random_k`, `drop_group`, `*_per_task`; `stage.py verify` (two-step verdict, Holm); `vlapoc.step4` (`T`/`C`/`0`/`g` per seed, expectation from the select run's `direction`) | measured: `runs.py` 781-841, `step4.py` 1-125 |
| No op to drop an explicit episode list | None of `SELECT_OPS` drops "these N episode ids" with provenance to a cluster run | measured: `runs.py` docstring 788-802 |
| `stage` values | `runs.STAGES` has no `cluster` stage | measured: `runs.py` line 105 |
| Real rollouts with video and failure types | Two real SmolVLA (`HuggingFaceVLA/smolvla_libero`) evals with trajectories and returns: `eval-smolvla-harness-spatial0` (10 rollouts, 4 failed `timeout_no_progress`) and `eval-smolvla-harness-l10t8` (10 rollouts, 7 failed: `timeout_no_progress` ×5, `timeout_partial_1_of_2` ×2). They are not trained on our subsets, so they are only usable as demo rollouts under synthetic scores | measured: `data/eval/eval-smolvla-harness-*/returns.parquet`, `rollouts.parquet` |
| Training videos | LeRobot v3 datasets with `image` dtype (PNG bytes in parquet, 128², two cameras, 20 fps); no mp4 files | measured: `data/lerobot/libero_spatial_10/meta/info.json` |
| Browser-playable video from those frames | PyAV 15.1.0 (installed) has `libx264`. Episode 0 (98 frames) gave a 35,217-byte H.264 mp4 in 0.10 s (parquet read excluded) | measured: `data/scratch/dash_plan/scripts/enc_probe.py` |
| Real MPS smoke attribute run | `--local-smoke` ran in a scratch store and its outputs were deleted. The real store holds only one `attribute` run, a stand-in (`attribute-train-full-s0 [stand-in]`) | measured: MLflow DB query (stage counts: attribute 1, eval 14, train 12, verify 3, score 15, ...) |
| Round A job counts | `server_pipeline.py --curation per_task --metrics f30 hrt60_lin --dry-run`: **v0/candidate 112 jobs, v3/candidate 113 jobs** (v3 adds the `score` job), 24 trainings each | measured: `data/scratch/dash_plan/dry_v0.txt`, `dry_v3.txt` |
| Subsets | v3/candidate and v0/candidate: 960 episodes, 20 tasks × 48; v0/reference 40 | measured |
| Planted "bad" on v3 | `labels.parquet`: 7 types × 13 = 91 episodes (`action_noise`, `detour`, `dropped_frames`, `gripper_release`, `idle_padding`, `instruction_swap`, `time_shift`), columns `episode_id, parent_content_hash, content_hash, corruption_type, params, seed, hypothesis`. **Each type has exactly 1 episode in each of 13 distinct tasks** (max 1 per (type, task)), and 4-5 planted episodes per task | measured: query over the answer key + `Subset.load('v3/candidate')` |
| Corruption is visibly detectable | Best label-free heuristic AUROC per type (ground-truth feature, r3): action_noise 1.0 (smoothness), detour 0.96 (length_vs_task), dropped_frames 1.0, gripper_release 0.964, idle_padding 1.0, instruction_swap 1.0, time_shift 0.848 (length_vs_task lowest) | measured artifact: `tracker/features/2026-10-15-ground-truth/artifacts/s5_v3_r3_bands.csv` |
| Clean original of each planted episode | `parent_content_hash` points to the v0 original, so the corrupted and original videos can be shown side by side | measured: column present |
| Oracle arm precedent | Ground-truth runbook GT-R5: `v3/candidate` minus the 91 labelled episodes as an `oracle_drop` subset (`groundtruth: True`), trained with `train_run` directly. GT-R4: verdict `v3/candidate` vs `v0/candidate` (does the corruption hurt at all) | measured: `tracker/features/2026-10-15-ground-truth/artifacts/server-runbook.md` lines 42-43, 96-128 |
| Ground-truth subsets refused by the chain | `stage.py train` opens subsets via `method_view` → `open_for_method`, which refuses any subset whose derivation chain contains `oracle_drop`, an op containing "groundtruth", or `groundtruth: True`. A "remove type X" arm therefore cannot go through today's `stage.py train` | measured: `method_data.py` lines 70-100; `stage.py` lines 140-150 (its docstring notes oracle arms bypass via `train_run`) |
| Effect-size output of a verdict | `verdict.json` per metric: `diff`, `ci95`, `p`, `p_holm`, `significant`, `label` (functional and τ), plus `gate` | measured: `data/eval/verdict-W-3392bb4d-e138fec9/verdict.json` (e.g. f30 `diff -0.6`, `ci95 [-0.9, -0.4]`, `p_holm 0.196`) |
| Round A random arms | `random0-k5`, `random1-k5` drop 5 per task (100 episodes); `drop-random-20` drops 20 at random, not 1 per task. **None matches a 13-episode, 1-per-task type removal**, so none can serve as its control | measured: dry-run arm list and `arms()` in `server_pipeline.py` |
| Server runbook (Step 1-2 era) | `tracker/features/2026-10-01-evaluation-layer/artifacts/server-runbook.md` (env, copies, AC8.5/5.8/6.7/7.5 commands, disk); `server_pipeline.py` docstring (chain); `slurm/env.sh`, `guard.sh`, `site.env.example`, `docs/site-profile-cluster.env` | measured: read |

## Prior art

| Candidate | What it covers | What it misses | Evidence |
|---|---|---|---|
| **Streamlit 1.66** | Local Python web app. Row selection in tables (`st.dataframe(on_select="rerun", selection_mode="single-row")`, `selection.rows` are positions in the original frame even after the user sorts); click selection on Altair charts (`st.altair_chart(on_select=...)`, needs named selection params); `st.video(bytes)`; headless testing with `streamlit.testing.v1.AppTest` (no browser) | Reruns the script on every interaction, so loaders need `st.cache_data` (inferred). No built-in lineage or score semantics (that part is ours) | quoted: docs.streamlit.io st.dataframe, st.altair_chart, st.video ("Converting your video to H.264 will allow the video to be displayed"), AppTest ("via API instead of a browser UI"), all v1.66.0. measured: `pip install --dry-run streamlit` on this venv (3.14.0) resolves 5 new packages, all pure wheels (`streamlit-1.66.1-py3-none-any`, altair 6.3.0, pydeck 0.9.3, python-multipart 0.0.32, websockets 17.3), `Requires-Python >=3.11` |
| Dash 4.4 + Plotly | Explicit callback graph, plotly click events | More code per interaction. 6 new packages | measured: dry run (dash, plotly 7.1.0, janus, comm, nest-asyncio, retrying) |
| Panel 1.9 / Bokeh | Rich widgets and tables | 10 new packages. No advantage for this app | measured: dry run |
| Gradio 6.30 | Model demos (input → function → output) | Built for model demos, not multi-view analytics (quoted from a third-party comparison, anvil.works). 12 new packages | measured: dry run |
| Flask / FastAPI (already installed) | Zero new dependencies | Tables, filters, selection and video would all be hand-built HTML/JS, which is hand-rolling a framework (brief: "not a mandate to hand-roll a framework") | measured: `pip list` (Flask 3.1.3, fastapi 0.141.1, uvicorn 0.54.0) |
| **LeRobot 0.6.1 `lerobot-dataset-viz`** | Single-episode viewer (Rerun or Foxglove) for a LeRobot dataset | No scores, rankings, rollouts-to-episodes links or aggregation. Needs `rerun-sdk` (not installed). 0.6.1 ships no HTML visualizer (`visualize_dataset_html` was in older versions) | measured: installed `lerobot/scripts/` listing, `lerobot_dataset_viz.py` lines 35-67, 209-212, entry_points.txt |
| Rerun 0.40 | Time-synced multimodal episode viewer | Episode viewer only. A link-out at most | measured: dry run, one wheel `rerun_sdk-0.40.0-cp311-abi3-macosx_11_0_arm64` |
| Renumics Spotlight 1.8 | Dataframe + media explorer; has a public sliceline demo (`huggingface.co/spaces/renumics/cifar100-sliceline-demo`) | 50 new packages (measured). Docs list Python 3.8-3.11 for an older version (quoted, renumics.com getting-started), and a video dtype could not be confirmed. A single-table explorer with no rollout-to-episode linking | measured: dry run |
| FiftyOne 1.22 | Media dataset app | 66 new packages including a bundled MongoDB (`fiftyone_db`). Docs list Python 3.10-3.13 and we run 3.14 (quoted, docs.voxel51.com installation) | measured: dry run |
| **MLflow UI** (installed 3.16.1) | Run pages, params, artifacts for the lineage store we already use | Not a demo surface. Good as the "lineage link" target | measured: installed. The run URL pattern `#/experiments/<id>/runs/<run_id>` is **inferred** from a third-party implementation (kedro-dagster `get_mlflow_run_url`) and is not in MLflow's own source tree (grep found only demo `#/experiments/<id>` routes) |
| **sliceline 0.3.0** (BSD-3, DataDome; SliceLine, SIGMOD 2021) | Top-k slices (conjunctions of ≤ `max_l` predicates, support ≥ `min_sup`) where an error vector is high; `top_slices_`, `top_slices_statistics_` (size, average error) | **Needs categorical X.** It one-hot encodes every distinct value (`OneHotEncoder(handle_unknown="ignore")`, source line 775), so a continuous column gives one predicate per value and **no slices** (measured: `length_steps` continuous → `[]`); quantile-binned values work (measured: a planted `">400"` bin found with average error 1.536). **Errors should be non-negative**: the score is `alpha*((slice_err/size)/average_error - 1) - ...` (source lines 480-482) and only slices with `slice_errors > 0` start the search (lines 526-527). A signed vector still returns slices (measured), but the ratio is ill-posed when the mean is near 0 or negative (inferred) | measured: `pip install --dry-run sliceline` resolves 1 pure wheel, deps (numpy<3, scikit-learn 1.6-2, scipy) already satisfied; ran on Python 3.14 without numba (`data/scratch/dash_plan/scripts/sl_probe.py`). quoted: sliceline.readthedocs.io 0.3.1 (the docs say nothing on binning or the sign of errors, so the source settles both) |
| scikit-learn `DecisionTreeRegressor` (installed) | Axis-aligned rules on continuous features with no binning, no new dependency | It partitions the data greedily and has no top-k overlapping slices or support/effect trade-off. A fallback, not the primary choice (inferred) | measured: sklearn 1.9.1 installed |
| "Looks bad" signal for §9.2: DemInf / RoboDrop screening (PoC §9.1 Step 1) | A learned screen, the PoC's intended "screening" input to §9.2 | Not built in the repo (training-attribution plan lists screening as a separate feature; `screen` stage is a stand-in). Owner direction: no new screening method here. Not needed for the PoC claim: the planted labels are a stronger "bad" definition (known by construction), and H1-H8 already flag every type (best AUROC 0.848-1.0, measured) as the practical-filter layer | measured: `s5_v3_r3_bands.csv`; quoted: brief "Owner follow-up" |
| W&B Tables / HF Hub viewers | Hosted tables with media | Hosted, outside the MLflow lineage store; the owner asked for a local app | inferred |

**Verdict: adapter over existing tools. Nothing is "already solved, stop".**
- **Adopt Streamlit** for the live local app, with five pure-wheel dependencies. It is the smallest dependency set that
  gives table row selection, chart selection, video and headless testing out of the box. Dash would be the second
  choice. Spotlight and FiftyOne were rejected (dependency weight, Python 3.14 support, no rollout-to-episode linking).
- **Adopt sliceline** for lens 2 (one pure wheel). The adapter we write: quantile-bin the descriptors into readable
  ranges, and turn each signed harm column into two non-negative vectors (positive part = harm slices, negative part =
  boost slices).
- **Reuse** `analysis.metric_groups`/`aggregate_by`/plots, `heuristics.compute`, `select_run`/`verify`/`step4`,
  `scratch_lineage_copy`, `trace`, and PyAV for H.264 (already installed).
- **Link out** to the MLflow UI and to `python -m vlapoc.lineage trace <run>` for lineage. We do not build a lineage
  viewer.
- **For §9.2's "looks bad"**: use the planted labels (primary) and the existing H1-H8 (secondary). No DemInf or
  RoboDrop in this feature. Reuse the GT-R5 `oracle_drop` pattern for the type-removal arms.
- **Build custom** only for what no tool knows: the read-side results loader with a provenance label, the cluster
  stage, the cluster-to-arm bridge, and the dashboard pages themselves.

## Stories

Order: S1 (read side and the missing output) → S2 (clusters) → S3 (planted-type scoring) → S4 (Round B bridge) →
S9 (§9.2 discrepancy, estimated) → S10 (§9.2 causal arms) → S5 (dashboard) → S6 (synthetic demo store) → S7 (drop-in
proof) → S8 (server runbook skill). The numbers S9/S10 are kept so earlier references stay stable; their position in
the build order is after S4. S5 can be built against S6's store, so the executor may interleave S5 and S6. Both must be
finished before S7.

### S1: Record the influence matrix and build one read-side results loader with a fail-closed provenance label
- **Why:** §2.7's first view needs, for one rollout, its row of the rollout × episode matrix. Today that file is not in
  the lineage store and has no index (measured above). The brief requires reading results only from the lineage store,
  and the SYNTHETIC label must come from provenance.
- **Depends on:** —
- **Deliverables (proposed):** `attribution.attribute_run` also records `influence_matrix` (the `.npy`) and
  `influence_index` (JSON: `rollout_id` per row and `episode_id` + `content_hash` per column, in matrix order) via
  `output_file`; `vlapoc/results.py` (read-only): `find_runs(stage, lineage_root, chain=None)`,
  `score_table(run)`, `functionals(run)`, `influence(run)` (matrix + index, sha-checked), `rollouts(run)` (eval rollouts
  + returns of an attribute run), `verdicts(...)`, `provenance(run_id) -> {"label": "REAL"|"SMOKE"|"SYNTHETIC", "reasons": [...]}`.
- **Provenance rule (fail-closed, inferred design; this rule decides when SYNTHETIC disappears):** walk
  `runs.trace(run_id)`. **REAL** only if no run in the trace has `vlapoc.synthetic=true` or `vlapoc.stand_in=true`,
  every `train` run has `vlapoc.smoke=false`, and every `eval` run has `eval.attribution_grade=true`. **SYNTHETIC** if
  any run has `vlapoc.synthetic=true` or a `synthetic` param `True`. **SMOKE** otherwise (real code, not
  attribution-grade, which includes stand-ins). A missing tag counts against REAL.
- **Acceptance criteria:**
  - [ ] AC1.1 A new `attribute_run` (through the `--integration` fixture of `verify_training_attribution.py`, scratch
    store) records outputs `influence_matrix` and `influence_index`. `results.influence(run)` returns a matrix whose
    shape equals `(attribute.n_rollouts, attribute.n_episodes)`, and an index whose column order reproduces the score
    table's `episode_id` order. [PoC §2.7 p.5; §2.1 item 1 p.3]
  - [ ] AC1.2 `results.influence` refuses (ValueError) when the downloaded artifact's sha256 differs from the recorded
    one, and returns a clear "not recorded (run predates dashboard S1)" error for older runs such as the real store's
    stand-in. [brief constraint: lineage store, not ad-hoc paths]
  - [ ] AC1.3 Provenance unit check in a scratch store with fake runs: an all-real trace → `REAL`; flipping each single
    condition (one `vlapoc.synthetic=true`, one `vlapoc.smoke=true`, one `eval.attribution_grade=false`, one missing
    tag, one stand-in) → not `REAL`, with the reason named. 6/6 cases. [brief: "SYNTHETIC label disappearing only when the data is real"]
  - [ ] AC1.4 `results.py` imports nothing from `vlapoc.groundtruth` or `vlapoc.recovery` (source scan in the verifier).
    [PoC §2.1 item 4 p.3]
  - [ ] AC1.5 All existing verifiers still pass (`verify_attribution.py`, `verify_training_attribution.py` all nine
    modes, the older-feature verifiers listed in the training-attribution changelog), and `data/lineage/mlflow.db` sha256
    is unchanged by the verifier runs. [vla-build]
  - [ ] AC1.6 Grounding docstrings cite §2.7 p.5, §3.4 p.7, §2.1 items 1 and 4 p.3. [vla-build]
- **Verify by:** `.venv/bin/python scripts/verify_dashboard.py --results` (new verifier, one mode per story).
- **Demo value:** the label table (6 cases → label + reason), shown in the report as "how the dashboard decides whether
  a number is real".

### S2: Add the tracked `clusters` stage, with three lenses, pattern descriptions and estimated effects
- **Why:** the owner's "pinpoint exact clusters of data and identify patterns within those clusters that harm or boost
  a specific policy"; PoC §1 p.3, §2.2 p.4, §4.2 p.9-10, §9.2 p.19.
- **Depends on:** S1
- **Deliverables (proposed):** `vlapoc/clusters.py`; `runs.STAGES += ("cluster",)`; `stage.py clusters --scores
  <attribute|ensemble run> --name <n> [--metrics ...] [--synthetic]`; `requirements` note for `sliceline==0.3.0`.
  Outputs (MLflow `output_file`): `clusters.parquet` (one row per cluster: `cluster_id`, `lens` ∈ {`label`, `slice`,
  `metric_group`}, `definition` (text, e.g. `task_name = X`, `length_vs_task in [1.31, 2.40] AND scene = Y`, `group =
  consistent_harmer`), `direction` (harm/boost), `target_metric` (for slices), `n`, `share_of_subset`, per metric
  `<harm col>__mean/__lo/__hi/__sum`, `status = "estimated"`), `members.parquet` (`cluster_id`, `episode_id`,
  `content_hash`, member rank by harm), `patterns.json` (per cluster: descriptor contrasts inside vs outside with
  standardized difference, label composition (top `task_name`/`scene` shares), the metrics it moves and which way, and
  its top-5 member episodes by |harm|), `descriptors.parquet` (the binned descriptor table used, for audit).
- **Lens details (all per score column, each column's declaration carried on every row):**
  1. *Label groups:* `aggregate_by` over `task_name`, `scene`, `suite` (the varying groupable columns, measured).
  2. *Rule-like slices:* sliceline over quantile-binned descriptors: catalog `length_steps`, `duration_s`, `smoothness`,
     `idle_fraction`, `idle_steps`, `gripper_toggles`; heuristics `length_vs_task`, `post_release_share_vs_task`,
     `duplicate_frame_pairs`, `instruction_mismatch` (through `open_for_method`); plus `task_name`, `scene` as
     categorical. Error vectors are `max(z, 0)` (harm slices) and `max(-z, 0)` (boost slices) of each z-scored `harm_`
     column. Defaults (inferred, declared as params): 5 quantile bins, `max_l=2`, `k=5`, `alpha=0.95`, `min_sup=max(10,
     2% of N)`.
  3. *Metric groups:* `metric_groups` (≥2-of-5, owner-approved). Each non-neutral group is a cluster.
  - *Estimated effect* = member mean of each harm column with the existing percentile bootstrap CI (`aggregate_by`
    method), plus the member sum (CUPID p.11 first-order group estimate). It is labelled "estimated (first-order sum of
    individual influence; CUPID p.11)".
- **Acceptance criteria:**
  - [ ] AC2.1 On a SYNTHETIC table where a slice (`length_vs_task` top bin AND one `scene`) and one task group carry a
    planted +1.5σ harm on `harm_hrt60_lin` only, the stage returns the planted slice among the top-3 harm slices for
    `harm_hrt60_lin` and the planted task among the top-3 label groups. On `harm_f30` neither is in the top 3. [PoC §1
    p.3; §4.5 p.11 "assigns each corrupted group to the expected metric"]
  - [ ] AC2.2 A planted *boost* slice (negative harm) is found on the boost side and not on the harm side. [user
    request "harm or boost"]
  - [ ] AC2.3 Every `clusters.parquet` row and `patterns.json` entry carries the declaration (`functional`, `tau`,
    `rollout_subset`, `good_sign`) of each harm column it reports, and `status == "estimated"` for every row. [PoC §3.4
    p.7; §4.4 p.10-11; brief two-round rule]
  - [ ] AC2.4 The run is a `cluster` stage run whose input entity is the score table's sha256 and whose params hold the
    lens settings, the descriptor list, the bin edges, and `synthetic`. `trace(cluster run)` reaches the attribute
    run(s). [PoC §2.1 item 1 p.3]
  - [ ] AC2.5 Blindness: `clusters.py` imports nothing from `vlapoc.groundtruth`/`vlapoc.recovery`, reads data only via
    `open_for_method`, and is refused on a fenced version (reuse the S1 refusal fixture of training-attribution). [PoC
    §2.1 item 4 p.3]
  - [ ] AC2.6 Members of every cluster sum correctly: label-group `n` values sum to N per column; every slice member
    satisfies its definition (re-evaluated from `descriptors.parquet`). [PoC §1 p.3]
  - [ ] AC2.7 `sliceline==0.3.0` is installed into `.venv` and recorded in the report with the reason. The import works
    on Python 3.14 without numba. [brief constraint: new deps]
- **Verify by:** `.venv/bin/python scripts/verify_dashboard.py --clusters`
- **Demo value:** a SYNTHETIC slice list such as "length_vs_task ≥ 1.31 AND scene = KITCHEN_SCENE3: n = 22, mean harm on
  HRT +1.4 [1.1, 1.7], neutral on F(30)", which is the PoC §1 "episodes over 400 steps are net harmful" sentence produced
  by the code.

### S3: Score discovered clusters against the planted corruption types (v3 only, after discovery)
- **Why:** PoC §2.1 item 4 p.3 ("check whether your attribution method flags exactly those") applied to clusters;
  §4.5 p.11; brief ("used only after the method, ... blindness unchanged").
- **Depends on:** S2
- **Deliverables (proposed):** `recovery.cluster_recovery(version, clusters, members)` and `stage.py score-clusters
  --clusters <cluster run> --version v3 --subset candidate --name ...` (stage `score`). Output
  `cluster_recovery.parquet`: per (cluster, planted type) precision (share of members of that type), recall (share of the
  type inside the cluster), lift over the base rate, and a one-sided Fisher exact p (scipy, installed). A `best_match`
  per cluster, and per type the best-matching cluster.
- **Acceptance criteria:**
  - [ ] AC3.1 Refuses unless the cluster run is FINISHED and its version has an answer key. Refuses a cluster run whose
    `trace` shows any ground-truth read. [PoC §2.1 item 4 p.3]
  - [ ] AC3.2 On SYNTHETIC members built over the real v3 answer key (a cluster = all 13 `idle_padding` episodes + 13
    clean ones), precision 0.5, recall 1.0, lift ≈ 0.5 / (91/960 × 1/7 share), recomputed independently in the verifier.
    [PoC §2.1 item 4 p.3]
  - [ ] AC3.3 The cluster stage's run (S2) is created before, and independently of, the scoring run: the scoring run
    consumes the cluster run's `members` sha256 and `clusters.py` has no code path to the scorer. [brief blindness]
  - [ ] AC3.4 Output rows carry `synthetic` and the clusters' declarations. [PoC §3.4 p.7]
- **Verify by:** `.venv/bin/python scripts/verify_dashboard.py --planted`
- **Demo value:** a "discovered cluster × planted type" table (SYNTHETIC on the demo; real after Round A on v3), shown as
  "did the method's clusters land on the bad data we planted?"

### S4: Turn a chosen cluster into a confirmation arm (Round B), reusing select/train/eval/verify/step4
- **Why:** brief two-round rule; PoC §2.1 item 5 p.3 (leave-group-out), §2.6 p.5 and §4.4 p.10 (two-step verdict),
  §9.1 Step 4 p.19.
- **Depends on:** S2
- **Deliverables (proposed):**
  - `select_run` op `drop_cluster` (drops a cluster run's `members` for one `cluster_id`; records `cluster_run_id`,
    `cluster_id`, the definition, and `expected_by_metric` = the estimated sign per metric whose CI excludes 0); optional
    op `drop_random_matched` (the same per-task counts as the cluster, seeded).
  - `server_pipeline.py --round-b --clusters <cluster run name> --pick <cluster_id ...> [--matched-random]`: per picked
    cluster, select → train ×3 seeds → eval (no trajectories) → `verify vs_base` (W declared, O alongside)
    [+ `curated_vs_random` vs the matched arm] → `step4-summary` → `confirm`.
  - `step4.expected_direction` reads `expected_by_metric` for `drop_cluster` arms.
  - `stage.py confirm --clusters <run> --tag <t>` (stage `summarize`) writes `cluster_status.parquet`: per cluster and
    metric, `confirmed` / `refuted` / `not confirmed`, with the verify run ids.
- **Confirmation rule (inferred, stated before any result; owner may override):** per metric, **confirmed** if the
  expectation is met (`T`/`C` matching `expected_by_metric`) in ≥ 2 of 3 seeds and contradicted in none; **refuted** if
  the opposite significant call appears in ≥ 2 of 3 seeds; otherwise **not confirmed**. A seed counts only if its
  macro-KS gate fired (`g` otherwise, as `step4.call` does).
- **Acceptance criteria:**
  - [ ] AC4.1 `server_pipeline.py --round-b ... --dry-run` on a cluster run with 2 picked clusters prints the job list.
    The count is 2 × (3 train + 3 eval + 3 verify) + 1 select + 1 step4-summary + 1 confirm = 21 jobs, 6 trainings
    (count inferred from the chain's per-arm pattern; the verifier measures it), and 33 jobs / 12 trainings with
    `--matched-random`. [brief "~3 trainings each"; PoC §2.3 p.4 "2–3 seeds"]
  - [ ] AC4.2 The `drop_cluster` child subset has exactly N − n episodes, its derivation lists the cluster's
    members, and `trace(select run)` reaches the cluster run and the score run. A random matched child has the same
    per-task counts. [PoC §2.1 item 1 p.3; §2.6 p.5]
  - [ ] AC4.3 On SYNTHETIC verdicts in a scratch store (the training-attribution S4 pattern): a cluster with expectation
    met in 3/3 seeds → `confirmed`, 1/3 → `not confirmed`, gate not fired in 2/3 → `not confirmed`, and opposite in 2/3 →
    `refuted`. A trade-off cluster (expected harm on HRT, boost on F30) is judged per metric. [PoC §4.4 p.10-11; §9.1
    Step 4 p.19]
  - [ ] AC4.4 Refusals before any run exists: unknown `cluster_id`, a cluster run from another subset, a cluster that
    would empty a task (a whole-task label group is allowed only with an explicit `--allow-whole-task` flag recorded as a
    param, because it is the leave-one-task-out case of §2.1 item 5). [PoC §2.1 item 5 p.3]
  - [ ] AC4.5 Every arm uses the base recipe unchanged: `lineage diff` between base and arm train runs shows only
    `data.*` differences (dry-run check of the argv: same steps, batch, pretrained ref). [PoC p.1 data is the only variable; §2.3 p.4]
- **Verify by:** `.venv/bin/python scripts/verify_dashboard.py --round-b`
- **Demo value:** the Round B cost table (jobs and trainings per picked cluster) and a SYNTHETIC `cluster_status` table
  showing all three outcomes.

### S5: Build the live local dashboard (Streamlit) over the lineage store
- **Why:** PoC §2.7 p.5 (both views), §4.5 p.11 (heatmap, scatter), §3.4 p.7 (declared functional), §10 p.20
  ("dashboard linking failed rollouts to influential episodes"); owner answer "Live local app".
- **Depends on:** S1, S2 (S3, S4 outputs shown when present)
- **Deliverables (proposed):** `dashboard/app.py` (+ `dashboard/pages/`), `vlapoc/video.py` (frames from a LeRobot v3
  dataset → H.264 mp4 bytes with PyAV, cached under `data/dashboard_cache/<content_hash>.mp4`), launched with
  `.venv/bin/streamlit run dashboard/app.py -- --lineage-root <store> --scores <run name|id> [--clusters <run>]`.
  `streamlit==1.66.1` added.
  - **Header on every page:** run picker (attribute/ensemble runs found in the store); a provenance banner from
    `results.provenance` (SYNTHETIC in red, SMOKE in amber, nothing extra for REAL); the declared functionals of the
    columns in view.
  - **Rollouts** (§2.7 view 1, PhAIL run explorer as reference): rollout table (filter: failed only / failure type /
    task), and for the selected rollout its failure type, T, event, censored, the rollout video (from the eval's recorded
    trajectories), then its top-10 training episodes by influence-matrix row (both signs, labelled "pushed toward this
    rollout" / "pushed away"), each with video, `task_name`, `scene`, `source`, `source_file` (the source dataset file).
  - **Episodes** (§2.7 view 2): pick a harm column (label = `axis_label`), ranked harmful and helpful lists,
    "aggregate by" any groupable catalog column (`aggregate_by`, with CI), and video of the selected episode.
  - **Clusters:** lens picker; heatmap and throughput-vs-reliability scatter (the scatter clickable, quadrants
    labelled); slice/group list with pattern text, estimated effect and CI, and 3 example member videos; status badge
    `estimated` / `confirmed` / `refuted` / `not confirmed` from the latest `confirm` run, with the verify run ids;
    lineage links (MLflow UI run URL, and the `python -m vlapoc.lineage trace <run>` command); on v3, the planted-type
    match table from S3, under a "ground-truth check (answer key read after discovery)" heading.
  - **Bad vs harmful (§9.2, v3 only):** per metric, a quadrant scatter (x = planted or clean, with the H1-H8 filter as a
    toggle; y = harm z), the per-type table from S9 (estimated readings), and the S10 statuses and sentences when a
    `discrepancy-verdict` run exists. For any type, the corrupted example video sits next to its clean original
    (`parent_content_hash`), and the best-heuristic AUROC is shown as "visibly bad" evidence. Page heading: "ground-truth
    check (answer key read after attribution)".
- **Acceptance criteria:**
  - [ ] AC5.1 `AppTest` (headless) on the S6 demo store: every page runs with no exception, and the SYNTHETIC banner
    element is present on every page. [brief: "marked SYNTHETIC everywhere they appear"]
  - [ ] AC5.2 Rollouts page: selecting a failed rollout shows its failure type equal to `returns.parquet`'s
    `failure_type` for that `rollout_id`, a video element, and 10 episode rows whose order equals the argsort of that
    rollout's influence-matrix row (recomputed in the test). [PoC §2.7 p.5]
  - [ ] AC5.3 Episodes page: the ranked list equals the score table sorted by the chosen `harm_` column; "aggregate by"
    offers exactly the groupable catalog columns (`CATALOG_COLUMNS` minus `GROUP_COLUMNS_EXCLUDED`), and the table equals
    `aggregate_by` output. [PoC §2.7 p.5, §1 p.3]
  - [ ] AC5.4 Every displayed score column label contains its declared functional and τ (scan of all rendered
    dataframe column names, captions and chart titles; none bare). [PoC §3.4 p.7; §4.4 p.10-11]
  - [ ] AC5.5 Clusters page shows the heatmap, the scatter with four quadrant labels, the cluster list with `status`
    from S4's `confirm` run when one exists (else `estimated`), and a lineage link per cluster. [PoC §4.5 p.11; brief]
  - [ ] AC5.5b Bad-vs-harmful page: the quadrant counts equal `discrepancy_episodes.parquet`'s, statuses and sentences are
    shown verbatim from the `discrepancy-verdict` run (never re-worded in the app), the word "harmless" appears nowhere
    in the rendered page, and a selected type shows two video elements (corrupted, original). [PoC §9.2 p.19; brief caveats]
  - [ ] AC5.6 Video: `vlapoc.video` turns a real training episode and a real rollout trajectory into H.264 mp4 bytes
    whose frame count equals the episode's frame count (decoded back with PyAV); the cache returns the same bytes on the
    second call. [PoC §2.7 p.5 "with video"]
  - [ ] AC5.7 The app reads only through `results.py`/MLflow (scan: no hard-coded `data/attribution` or `data/eval`
    paths in `dashboard/`). Pointing `--lineage-root`/`--scores` at another store changes the content with no edit.
    [brief constraint]
  - [ ] AC5.8 `streamlit==1.66.1` installed into `.venv`, `streamlit run` serves locally (HTTP 200 on
    `http://localhost:<port>/_stcore/health`), stopped after the check. [owner answer: live local app]
- **Verify by:** `.venv/bin/python scripts/verify_dashboard.py --app` (AppTest + health check); then the owner runs
  `.venv/bin/streamlit run dashboard/app.py -- --lineage-root data/lineage_demo --scores attribute-demo-ens`.
- **Demo value:** the live app. For the report: rendered figures and a few mp4 clips from the same code, with the
  AppTest element dump as evidence.

### S6: Build the synthetic demo store (real videos, catalog and rollouts; synthetic scores, labelled)
- **Why:** owner answer "Synthetic scores, labelled. Real LIBERO videos and catalog, the real (not attribution-grade)
  MPS smoke run, plus synthetic influence scores"; it lets the owner present the whole flow before the server run.
- **Depends on:** S1, S2, S3, S4 (exercised on demo data)
- **Deliverables (proposed):** `scripts/dashboard_demo.py build [--with-smoke]` creating `data/lineage_demo/` as a
  `runs.scratch_lineage_copy` of the real store (so versions, subsets and catalog resolve; the real store is untouched).
  Inside it:
  - SYNTHETIC `attribute` runs ×3 seeds on **v3/candidate** (960 real episodes) against the 20 real SmolVLA
    harness rollouts (real videos, real returns, 11 real failures). The influence matrix is synthetic: noise, plus
    planted structure written **only from method-facing descriptors** (e.g. high `length_vs_task` pushes HRT, one scene
    pushes F30, a small boost slice). The generator reads through `open_for_method`, never the answer key. Scores come
    from the real `attribution.score_table` and `column_declarations`, so the columns and declarations are the real
    shape. Every run is tagged `vlapoc.synthetic=true`.
  - A real `ensemble_run` over them (if its input checks allow synthetic attribute runs; otherwise a SYNTHETIC ensemble
    with the same outputs, recorded as such), then the real `analyze`, `clusters`, `score-clusters` stages.
  - SYNTHETIC Round B verdicts for 2-3 picked clusters (`synthetic_verdict` pattern) and a real `confirm` run, so the
    dashboard shows all three statuses.
  - The real `discrepancy` stage (S9) on the synthetic ensemble, plus SYNTHETIC planted-type Round B verdicts and a real
    `discrepancy-verdict` run (S10), so the §9.2 page shows every status. The descriptor-planted harm (e.g. high
    `length_vs_task` → HRT) makes some planted types look harmful and others not, without the generator reading labels.
    The page says SYNTHETIC: this demonstrates the view, not the claim.
  - `--with-smoke`: the real 2-step SmolVLA MPS chain of `--local-smoke` (train on v0/reference, eval 2 rollouts,
    returns, features, attribute), written into the demo store and kept. It shows as **SMOKE** (about 8 min, measured
    in training-attribution S8).
- **Acceptance criteria:**
  - [ ] AC6.1 `dashboard_demo.py build --with-smoke` completes. `data/lineage/mlflow.db` sha256 is unchanged
    before vs after. [brief constraint; training-attribution AC6.4 pattern]
  - [ ] AC6.2 In the demo store every run created by the builder is tagged `vlapoc.synthetic=true` except the smoke
    chain's, and `results.provenance` gives SYNTHETIC for the demo ensemble and SMOKE for the smoke attribute run.
    [owner answer: labelled]
  - [ ] AC6.3 The synthetic generator imports nothing from `vlapoc.groundtruth` and opens v3 only through
    `open_for_method` (source scan + refusal test). The planted structure is described in the run params in plain words.
    [PoC §2.1 item 4 p.3]
  - [ ] AC6.4 The demo's cluster run recovers the planted structure (AC2.1-style check on the demo data), and its S3
    table is produced. Whatever overlap with planted types appears is reported as SYNTHETIC. It is not evidence about
    attribution. [PoC §1 p.3; honesty rule]
  - [ ] AC6.5 Disk: the demo store plus the video cache stay under 2 GiB, and free disk stays above 20 GiB (measured
    `du`/`df`). Limit inferred: 20 rollouts + 960 episodes of ~35 KB mp4 ≈ 35 MB of video. [brief constraint: disk]
- **Verify by:** `.venv/bin/python scripts/dashboard_demo.py build --with-smoke && .venv/bin/python
  scripts/verify_dashboard.py --demo`
- **Demo value:** the full stakeholder demo, end to end on this Mac, with real videos and SYNTHETIC banners.

### S7: Prove that real runs drop in with no code change
- **Why:** owner answer "Real server results must drop in later with no code change and the SYNTHETIC label
  disappearing only when the data is real".
- **Depends on:** S5, S6
- **Deliverables (proposed):** `verify_dashboard.py --dropin`.
- **Acceptance criteria:**
  - [ ] AC7.1 The `--integration` fixture chain (the real `attribute_run` → `ensemble_run` → `analyze` code path,
    synthetic features) is extended by `clusters`, in a scratch store. The dashboard AppTest loads it by run name with
    no file under `dashboard/`, `vlapoc/` or `scripts/` modified (`git status --porcelain` empty for those paths
    before/after). [brief]
  - [ ] AC7.2 The same scratch chain with its train/eval runs re-tagged to the attribution-grade values the server
    writes (`vlapoc.smoke=false`, `eval.attribution_grade=true`, no synthetic tag) renders **without** any SYNTHETIC or
    SMOKE banner. Re-adding one synthetic tag brings the banner back. [brief]
  - [ ] AC7.3 The real `--with-smoke` attribute run (S6) renders on the Rollouts page with its 2 real rollouts and the
    SMOKE banner. This is the real attribute_run output in the dashboard, with no adapter. [brief: "the real ... MPS smoke run"]
- **Verify by:** `.venv/bin/python scripts/verify_dashboard.py --dropin`
- **Demo value:** one sentence with evidence for stakeholders: "the dashboard you see on synthetic data is the
  one that will show the server's results; only the run name changes".

### S8: Write the server runbook as a repo skill (separate story)
- **Why:** owner: "an exact step-by-step runbook for a new claude session inside the actual gpu server ... all the way
  to attribution and real dashboard creation"; brief "Runbook: a separate story".
- **Depends on:** S2, S4, S5 (it documents their commands). It can be drafted earlier.
- **Deliverables (proposed):** `.claude/skills/server-run/SKILL.md` (frontmatter `name: server-run`, a description that
  triggers on "run on the server / GPU / SLURM / cluster"), and `scripts/verify_runbook.py` (local checks). Sections,
  in order:
  0. **Rules:** no code edits, no push, no deletes, no `git commit` except what the runbook names. Stop and ask on
     any failed check, failed job, odd number, low disk (< 20 GiB plus the stage's estimate), any code change needed.
     Every result is written up with quoted/measured/inferred labels.
  1. **Preflight**, with the expected output per step: `git rev-parse HEAD` equals the commit the owner names; clean
     tree; `slurm/site.env` present (from `site.env.example`, values in `docs/site-profile-cluster.env`); Python
     3.12/3.13 venv (lerobot 0.6.1 declares those, quoted in the eval-layer runbook); `pip check`; `nvidia-smi`; `df -h`
     on `SCRATCH_ROOT`; `source slurm/env.sh`; inputs copied (v0/v3 datasets byte-checked by `verify_version`, lineage
     store, human reference, LIBERO assets, `data/hf_models/smolvla_base`, `data/hf_home`); the smolvla_base revision
     check (`d9f33c94…` pinned vs the cluster cache's `5e8d12a6…`: stop and ask).
  2. **Smoke:** `scripts/lineage_dry_run.py --eval-smoke` (expected: the chain reaches 20 raw files, no stand-in), and
     `verify_training_attribution.py --local-smoke --device cuda` (expected: six stages finish, a [40, 30] table), plus
     the 30-step throughput probe and the two-process determinism check (eval-layer runbook rules 4-5).
  3. **Round A submit:** `server_pipeline.py --tag <tag> --subset <dataset> --curation per_task --metrics f30 hrt60_lin
     --dry-run` (expected count printed in the skill, measured locally), then without `--dry-run`. Record the chain id.
  4. **Monitor:** `squeue -u $USER`, `sacct -j <ids> --format=JobID,JobName,State,Elapsed`, the log folder, MLflow run
     states (a short `results.find_runs` snippet). What "healthy" looks like per phase, with inferred durations (≈4 h
     per train, quoted LeRobot docs). On a FAILED job: stop, collect the log tail, ask.
  5. **Recovery:** `--from rank|select|arms` reuse; closing a KILLED run (`MlflowClient().set_terminated`); never merge
     partial evals (eval-layer runbook).
  6. **Clusters stage** (inside the Round A chain after `ensemble`, or by hand: `stage.py clusters`), and on v3
     `score-clusters` and `discrepancy` (S9; all CPU jobs after the ensemble).
  7. **Real dashboard:** `streamlit run dashboard/app.py -- --lineage-root data/lineage --scores attribute-<tag>-ens
     --clusters clusters-<tag>` on the server, viewed through `ssh -L <port>:localhost:<port>`. Expected: no SYNTHETIC
     banner on the real ensemble (S1 provenance). If a banner appears, stop and report the listed reasons.
  8. **Round B:** the owner picks clusters (stop-and-ask point), then `server_pipeline.py --round-b --clusters
     clusters-<tag> --pick ...` with dry-run count first, then the `confirm` run, and the dashboard shows statuses.
     **Round B planted (§9.2, v3 only):** `server_pipeline.py --round-b-planted --tag <tag> --subset v3/candidate
     [--oracle]`, dry-run count first. **STOP AND ASK** with the cost table (27-33 trainings, inferred 108-132 GPU-h),
     then submit, then `discrepancy-verdict`. The skill quotes S10's wording rule, so the server session reports
     statuses with the same sentences (never "harmless").
  9. **Results back:** what to copy to the Mac (the `data/lineage/` store including `mlflow.db` and `mlartifacts/`,
     `data/attribution/<tag>*`, `data/eval/returns-*`, and rollouts parquet, not trajectories, which are 11-22 GB per
     seed, inferred in the eval-layer runbook), with `rsync` commands and a sha256 check of `mlflow.db` after copy.
  10. **Known server-only risks:** SQLite locking under concurrent jobs (MLflow on a SQLite file; symptom `database is
      locked`; what to do: stop, report, do not retry blindly); the smolvla_base revision; CUDA nondeterminism; the
      HF datasets cache size (~7 GB per distinct subset, measured locally); queue limits.
  Each step is tagged **[verified locally]** or **[unverified until a server session]**.
- **Acceptance criteria:**
  - [ ] AC8.1 `scripts/verify_runbook.py` extracts every fenced shell command from `SKILL.md`. For every `scripts/*.py`
    or `python -m vlapoc.*` invocation, the file or module exists and the named subcommand's `--help` exits 0 on this
    Mac. Reported as n/n. [vla-build: commands exist]
  - [ ] AC8.2 Every dry-run count written in the skill equals a fresh local dry run: Round A for the chosen dataset
    (measured today: v0/candidate 112, v3/candidate 113, before S2/S3/S9 add CPU jobs; the skill must carry the
    post-build numbers), Round B per picked cluster (S4), and Round B planted with and without `--oracle` (S10).
    [brief: "dry-run job counts"]
  - [ ] AC8.3 Every step carries `[verified locally]` or `[unverified until a server session]`. The verifier counts both
    and lists the unverified ones, and none of the unverified steps claims an expected output that was not produced
    somewhere. [brief: "server-specific steps are marked unverified"]
  - [ ] AC8.4 The skill has explicit **STOP AND ASK** markers at: preflight failure, revision mismatch, any FAILED job,
    disk below threshold, any needed code change, the Round B cluster pick, and a SYNTHETIC/SMOKE banner on real
    results (≥ 7 markers, counted by the verifier). [brief]
  - [ ] AC8.5 The plan's proposed `CLAUDE.md` pointer (below) is **not** applied by the executor. It is listed in the
    report for the owner. [brief: owner's uncommitted `CLAUDE.md`]
  - [ ] AC8.6 The skill's frontmatter parses (YAML `name`, `description`), and the skill appears in `.claude/skills/`.
    [vla-build]
- **Verify by:** `.venv/bin/python scripts/verify_runbook.py`
- **Demo value:** a one-page "what the server session will do" table (step → command → expected → verified locally?)
  in the report, and the count of locally verified vs unverified steps.
- **Proposed CLAUDE.md line (orchestrator asks the owner; not applied):** under "## Required workflow", add:
  `**On the GPU server, follow the \`server-run\` skill** (\`.claude/skills/server-run/SKILL.md\`): preflight → smoke → Round A → clusters → dashboard → Round B → results back, stopping to ask at every marked point.`

### S9: Show where "looks bad" and "hurts" disagree, estimated from Round A scores (PoC §9.2)
- **Why:** owner addition ("show ... that bad episodes are not always harmful for a policy"); PoC §9.2 p.19 "most
  compelling output"; §9.1 Step 1 p.18 (screening verdicts "compared with what actually happened"); §4.5 p.11
  (metric-targeted hypotheses: does each corrupted group move its expected metric?).
- **Depends on:** S1 (S2 optional: cluster ids shown when present)
- **Deliverables (proposed):** `vlapoc/discrepancy.py`. It is **ground-truth side**, next to `recovery.py`: it reads
  the answer key, so method code never imports it. `stage.py discrepancy --scores <attribute|ensemble run> --version v3
  --subset candidate --name ...` (stage `score`). Outputs (MLflow `output_file`):
  - `discrepancy_episodes.parquet`: per episode, `planted_type` (or `clean`), `parent_content_hash`, secondary flags
    `looks_bad_filter` (flagged by any H1-H8 at the top 9.48%, the injection rate; threshold inferred, declared) and
    which heuristic flagged it, per-metric z-scored harm, and per-metric `quadrant` ∈ {`bad_and_harmful`,
    `bad_but_not_harmful`, `clean_but_harmful`, `clean_and_not_harmful`} (harmful = harm z ≥ 1, the `metric_groups`
    threshold, declared).
  - `discrepancy_types.parquet`: per (planted type, metric), the member mean harm z with a percentile bootstrap 95% CI;
    the same for clean episodes; the difference and its CI; the share of members in each quadrant; the label's
    `hypothesis` text; and a per-type `estimated_reading`: "harmful on M" / "no detectable harm on M (estimated)" /
    "helpful on M", decided by whether the type-vs-clean CI excludes 0.
  - `clean_harmful.parquet`: per metric, the top-20 clean episodes by harm (candidates for "looks clean but hurts").
  - `examples.json`: per type, the most and the least harmful members, with the clean original's `content_hash`, for
    side-by-side video.
  - The same table repeated with "looks bad" = the H1-H8 filter instead of the labels (secondary layer: "what a practical
    filter would have dropped, and whether it mattered").
- **Acceptance criteria:**
  - [ ] AC9.1 On a SYNTHETIC score table over the real v3 answer key (planted: `idle_padding` members harm HRT only,
    `action_noise` members neutral, 10 clean episodes harmful on F30), the stage returns idle_padding "harmful on
    hrt60_lin" and "no detectable harm on f30 (estimated)", action_noise "no detectable harm" on all metrics, and the 10
    planted clean episodes among `clean_harmful` top-20 for f30. All four quadrants are non-empty for at least one metric.
    [PoC §9.2 p.19; §4.5 p.11]
  - [ ] AC9.2 Wording guard: the string "harmless" never appears in any output. A non-excluding CI is always written as
    "no detectable harm on <metric> (estimated; CI [lo, hi])". [orchestrator: non-significance is not proof of no harm]
  - [ ] AC9.3 "Corruption is visibly bad" evidence is attached per type: the ground-truth feature's best-heuristic
    AUROC (read from the recorded `score` runs of the ground-truth feature in the lineage store, not re-typed), and
    `examples.json` pairs each example with its clean original. [brief caveat (1): "corruption too mild"]
  - [ ] AC9.4 Blindness: the stage refuses unless the score run is FINISHED and recorded before it (it consumes the
    score table's sha256). `vlapoc/attribution.py`, `clusters.py`, `results.py` and `analysis.py` import nothing from
    `discrepancy.py`, `groundtruth` or `recovery` (source scan). [PoC §2.1 item 4 p.3]
  - [ ] AC9.5 Every row carries the declaration of its harm column (functional, τ, rollout subset) and `synthetic`.
    [PoC §3.4 p.7]
  - [ ] AC9.6 Grounding docstring cites §9.2 p.19, §9.1 Step 1 p.18, §4.5 p.11, CUPID Fig. 4 p.8. [vla-build]
- **Verify by:** `.venv/bin/python scripts/verify_dashboard.py --discrepancy`
- **Demo value:** a per-type × metric "looks bad vs hurts" table, plus the quadrant scatter, SYNTHETIC on the demo and
  real after Round A on v3. Example headline shape: "idle padding hurts speed but not success; action noise shows no
  detectable effect on any metric (estimated)".

### S10: Test the §9.2 claim causally in Round B: per-type removal arms and a clean-but-harmful arm vs clean random controls
- **Why:** the estimated half (S9) is a first-order influence reading (CUPID p.11). Only a removal arm can support
  "this bad data did not hurt this policy on this metric" or "this clean data did" (§9.1 Step 4 p.19: "This step turns
  them from claims into verified results"; §2.6 p.5; §2.1 item 5 p.3).
- **Depends on:** S4 (Round B machinery), S9
- **Deliverables (proposed):**
  - Ground-truth-side subset builders (the GT-R5 `oracle_drop` precedent, `groundtruth: True` in the derivation):
    `type-<T>` = v3/candidate minus the 13 episodes of type T; `cleanharm-<m>` = minus the 13 most harmful **clean**
    episodes on `harm_<m>__same_task`, at most 1 per task (mirrors the type structure); `cleanrandom<r>` = minus 13
    clean episodes, 1 from each of 13 seeded-random tasks (the matched control: a type is exactly 1 episode in each of
    13 tasks, measured); optional `oracle` = minus all 91 (GT-R5) with `cleanrandom91-<r>` matching its per-task
    counts (4-5 per task).
  - `stage.py train --groundtruth-arm`. It accepts a ground-truth-derived subset **only** with this flag, tags the run
    `vlapoc.groundtruth_arm=true`, and changes nothing else (same recipe). `features`, `attribute`, `select --scores`,
    `clusters` and `analyze` refuse any subset or run carrying that tag or derivation.
  - `server_pipeline.py --round-b-planted --tag <t> --subset v3/candidate [--clean-harm-metrics f30] [--random-draws 1]
    [--oracle]`: select (ground-truth side) → per arm train ×3 seeds → eval → `verify vs_base` (W, O) → `verify
    curated_vs_random` (each type/cleanharm arm vs each `cleanrandom`, family W; the oracle vs its matched control) →
    `step4-summary` → `discrepancy-verdict`.
  - `stage.py discrepancy-verdict` (stage `summarize`): per arm and metric, the status and the sentence the dashboard
    shows (rule below), with verify run ids.
- **What the evidence may say (the wording rule, stated before results; inferred design, owner may override):** per
  (arm, metric), from the `curated_vs_random` verdicts over 3 seeds (`diff`, `ci95`, `p_holm`, gate):
  - **harmful (confirmed)**: removing type X beats removing 13 clean episodes per the S4 confirmation rule (≥ 2/3 seeds,
    fired gate, Holm-significant, none contradicting). Wording: "Removing the 13 <type> episodes improved <metric> by
    <diff> [CI] relative to removing 13 clean episodes (k/3 seeds)."
  - **helpful (confirmed)**: the same, in the opposite direction ("planted but helpful").
  - **no detectable harm, bounded**: no seed significant in the harmful direction, **and** the round has a positive
    control (at least one arm in this Round B, a type, cleanharm or oracle arm, confirmed on some metric). Wording:
    "No detectable harm on <metric>: removing the 13 <type> episodes did not improve it relative to removing 13 clean
    episodes in any of 3 seeds; the 95% CI on the difference spanned [lo, hi] (union over seeds), so a benefit from
    removal larger than <hi> on this metric is not supported by this experiment. This is not proof of no harm."
  - **inconclusive**: everything else, and **all** arms when no arm in the round shows any confirmed effect ("no arm
    showed an effect; with 13 episodes (1.4% of the data) per arm the test may lack power").
  - "Clean but harmful" is claimed only for `cleanharm-<m>` arms confirmed harmful in this sense.
- **Acceptance criteria:**
  - [ ] AC10.1 `server_pipeline.py --round-b-planted --dry-run` prints the job list and counts. Inferred from the
    chain's per-arm pattern (the verifier measures and the skill carries the measured numbers): **default** (7 type + 1
    cleanharm + 1 cleanrandom) = 9 arms, 27 trainings, 108 jobs; **`--oracle`** adds 2 arms (+6 trainings, +21 jobs) =
    33 trainings, 129 jobs; **`--random-draws 2`** adds 1 arm and a second control per treated arm (+3 trainings, +33
    jobs). [brief caveat (3): cost; PoC §2.3 p.4]
  - [ ] AC10.2 Each type arm's child has 947 episodes, exactly the 13 labelled episodes of that type removed. Each
    `cleanrandom` child has 947 with 13 clean episodes in 13 distinct tasks, and no planted episode is removed. The
    oracle child has 869 (GT-R5's expected number). [PoC §2.1 item 4 p.3; §2.6 p.5]
  - [ ] AC10.3 Blindness is unchanged: `stage.py train` without `--groundtruth-arm` still refuses these subsets, and
    `features`/`attribute`/`select --scores`/`clusters` refuse them and any `vlapoc.groundtruth_arm=true` run even
    with the flag (6 refusal cases, nothing written). The existing `verify_training_attribution.py --blind` passes
    unchanged. [PoC §2.1 item 4 p.3; owner decision 3]
  - [ ] AC10.4 On SYNTHETIC verdicts (scratch store): a type confirmed harmful on HRT and unchanged on F30, with a
    positive control present → "harmful (confirmed)" on hrt and "no detectable harm, bounded" on f30, the sentence
    containing the union CI. The same verdicts with **no** arm confirmed anywhere → every status "inconclusive". The
    word "harmless" never appears. [orchestrator caveat (2): power and the all-null case]
  - [ ] AC10.5 Every arm's train argv equals the base recipe except `--subset` and `--groundtruth-arm` (dry-run diff).
    [PoC §2.3 p.4; data is the only variable]
  - [ ] AC10.6 `step4.expected_direction` for type and cleanharm arms is "treated_better" (harmful data removed) on
    every metric, unless the label's `hypothesis` names specific metrics. In that case the expectation applies to those
    metrics and the others read "-". Recorded before results. [PoC §4.5 p.11; labels `hypothesis` column]
- **Verify by:** `.venv/bin/python scripts/verify_dashboard.py --planted-arms`
- **Demo value:** the Round B §9.2 table (type × metric → status and sentence), SYNTHETIC on the demo and real after the
  server. This is the "same bad data, different harm" result the owner asked for. The cost table (arms, trainings,
  jobs, GPU-h inferred at 4 h per training) goes to the owner before submission.

## Out of scope (deliberately)
- A new screening method (DemInf mutual information, RoboDrop warm-up gradient compatibility; PoC §9.1 Step 1). Owner
  direction. The planted labels and H1-H8 serve as "looks bad" here.
- Visual-similarity or image-embedding clusters (owner answer: not built).
- Datamodels-style random-subset regression for group effects (§2.1 item 5 second bullet; DataMIL; CUPID p.11 "random
  sampling"). Round B confirms by removal instead, as the owner was told.
- Global-curation Round B arms; cross-policy comparison of patterns (the PoC's scale-up stage); real-robot data.
- Hosting, authentication, multi-user deployment. The app is local (or SSH-tunnelled on the server).
- A custom lineage viewer (MLflow UI and `trace` exist); telemetry-synchronized playback beyond video + outcome +
  failure type (PhAIL's explorer is the reference, not a target).
- Segment/step-level attribution; ghost events (owner decision).
- "Operator 3"-style findings: the LIBERO catalog records no operator or date (measured). The lens works on any column
  that varies, and the dashboard says so.
- Running anything on the server (no access from here).

## Open questions
- **BLOCKING Q1 (owner): which dataset the runbook's Round A uses. With the §9.2 addition, the evidence now points
  clearly to (a).** It stays BLOCKING only because (a) replaces the first-run command recorded in the
  training-attribution owner ruling.
  - **(a) `v3/candidate`, recommended.** 113 jobs measured today, a few more CPU jobs after S2/S3/S9 (S8 measures). The
    same 24 trainings and the same ~96 GPU-h (inferred). It is **required** for the owner's §9.2 addition: S9 and S10
    need the planted labels, and S10's type arms are removals *from the v3 base*. Their `vs_base` verdicts compare
    against Round A's v3 base runs, so with a v0 Round A, S10 would need a whole extra v3 base (3 trainings, evals,
    features, attribution). It also gives `score` (recovery per type) and `score-clusters`, the PoC's "convincing demo"
    (§2.1 item 4 p.3). Cost: results describe a dataset with 9.48% planted corruption, not clean LIBERO.
  - **(b) `v0/candidate`.** 112 jobs measured, and it is the ruled first-run command. Clusters describe clean LIBERO.
    There is no planted check, and the §9.2 demonstration needs an extra v3 Round A later.
  - Either way the runbook shows the other as a variant.
- **BLOCKING Q2 (owner): the GPU budget for the planted-type Round B (S10).** All counts are inferred from the chain
  pattern at 4 h per training and measured by the S10 dry run.
  - **(a) Recommended: default + `--oracle`.** 7 type arms, 1 clean-but-harmful arm, 1 clean random-13 control, plus
    the all-91 oracle and its matched control: 33 trainings, ~132 GPU-h, 129 jobs. The oracle is the positive control.
    If removing all 91 planted episodes shows an effect, a type-level null becomes a meaningful bound. If nothing in the
    round shows an effect, every status reads "inconclusive", by rule.
  - **(b) Minimal default:** 27 trainings, ~108 GPU-h, 108 jobs (the orchestrator's figure). It risks an all-inconclusive
    round with no positive control, other than any type that happens to show an effect.
  - **(c) (a) + a second random draw:** 36 trainings, ~144 GPU-h, 162 jobs. More robust controls.
  - Existing Round A random arms **cannot** serve as controls: they drop 100 (5 per task) or 20 (unstratified)
    episodes, not 13 at 1 per task (measured above).
- Non-blocking (the executor proceeds on these; the owner may override):
  1. **Demo store location:** a separate `data/lineage_demo/` (copy of the real store) rather than writing synthetic runs
     into `data/lineage/`. This keeps the real store's sha256 stable and keeps synthetic runs out of the real
     experiment's graph (inferred).
  2. **Round B default = removal vs base only** (3 trainings per cluster, as the owner was told). `--matched-random`
     adds a same-per-task-count random control (+3 trainings per cluster), which is needed to claim "more than random
     removal" (§2.6 p.5). Recommended for the first picked cluster if budget allows.
  3. **Confirmation rule** (S4): ≥ 2 of 3 seeds meeting the expectation with a fired gate and none contradicting.
     Inferred, stated before results, recorded as a param.
  4. **Slice-finder defaults** (S2): 5 quantile bins, `max_l=2`, `k=5`, `alpha=0.95`, `min_sup=max(10, 2%)`. Inferred,
     declared params.
  5. **The clusters job joins the Round A chain** (+1 CPU job, +1 more on v3 for `score-clusters`). The owner's
     "112 jobs" ruling was about GPU size, and these are CPU-only additions. S8 prints the new counts.
  6. **Real dashboard on the server via SSH tunnel.** Rollout videos need the base evals' trajectories (11-22 GB per
     seed, inferred), which we do not copy back. A local copy without trajectories shows "rollout video not on this
     machine" and everything else.
  7. **New dependencies:** `streamlit==1.66.1` (+ altair, pydeck, python-multipart, websockets) and `sliceline==0.3.0`,
     all pure wheels (measured). On the server's Python 3.12/3.13 venv they need installing too (`Requires-Python`
     ≥ 3.11 / ≥ 3.10, measured from wheel metadata).

## Risks / premises to verify during execution
- **R1 Selection bias in slices** (inferred): the slice finder picks the most extreme slices, so their estimated CIs are
  optimistic. That is one reason they are "estimated". The pattern text must not call them findings.
- **R2 Whole-task clusters** (inferred): removing a whole task's data will obviously hurt that task's rollouts. That is
  a leave-one-task-out (§2.1 item 5), not a pattern discovery. S4 refuses it without `--allow-whole-task`, and the
  dashboard labels it.
- **R3 `ensemble_run` on synthetic attribute runs** (unverified): its input checks may refuse synthetic attribute runs
  that have no real train run. S6 falls back to a recorded SYNTHETIC ensemble.
- **R4 `__same_task` columns on 2 tasks** (measured fact, consequence inferred): the demo's real rollouts cover only 2
  tasks (`libero_spatial/0`, `libero_10/8`), so `__same_task` scores exist for 96 of 960 episodes. The demo's synthetic
  matrix should cover all rollouts. The executor states which columns are meaningful in the demo.
- **R5 Streamlit reruns:** heavy loads (960 × N matrix, video encoding) must be cached (`st.cache_data`), or the app
  will feel slow. Measure page render time in AppTest (soft budget inferred: < 3 s, AppTest's default timeout).
- **R6 MLflow run URL** `#/experiments/<id>/runs/<run_id>` is inferred (third-party code). The executor opens one in
  `mlflow ui` to confirm, or shows only the `trace` command if it is wrong.
- **R7 LIBERO image orientation:** frames are stored rotated 180° to match the hub layout (quoted,
  `libero_convert.py` line 20). The video helper must decide whether to un-rotate for display and say so. Check by
  viewing one encoded clip.
- **R8 Answer-key timing:** S3's ordering guard (cluster run first) is a process guarantee, not a cryptographic one.
  The source scan (AC2.5, AC6.3) is the real guard.
- **R9 Server-only (unverifiable here):** SQLite locking under concurrent SLURM jobs; the smolvla_base revision
  `5e8d12a6` vs `d9f33c94`; CUDA determinism; actual durations. All are marked unverified in S8.
- **R10 Power of type arms** (inferred): 13 episodes per type is 1.35% of 960. A real but small harm may not clear the
  gate in 3 seeds. That is why the wording rule bounds rather than denies, and why an all-null round is
  "inconclusive". The oracle arm (Q2 a) is the planned positive control.
- **R11 "Corruption too mild" objection:** answered by evidence, not argument. Corrupted-vs-original videos, the
  measured best-heuristic AUROC (≥ 0.848, six types ≥ 0.96), and S9's influence readings. If a reviewer still objects,
  the GT-R4 verdict (v3/candidate vs v0/candidate) is the causal test, and it needs v0 base runs (out of scope here,
  in the ground-truth runbook).
- **R12 Ground-truth-arm path** (inferred): `--groundtruth-arm` is a new hole in the method boundary by construction.
  AC10.3's refusals are what keep it from leaking into attribution or clustering. The analyzer should audit it.
- **R13 Clean-but-harmful selection uses the labels** (to exclude planted episodes) and the scores. It is a
  ground-truth-side arm, labelled as such, and is not a method result.
- **Over-building watch:** no custom frontend, no lineage viewer, no hosted deployment, no new storage, no per-step
  telemetry sync, and no new screening method. If S5 grows past five pages, or S2 past the three lenses, cut it back.
