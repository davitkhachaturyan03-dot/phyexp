# VLA Training-Data Attribution — PoC Documentation

Reference material for the VLA data curation / attribution proof of concept.

## PoC description

- [VLA_Data_Attribution_PoC.pdf](VLA_Data_Attribution_PoC.pdf) — the source document
  for this project: PoC outline, evaluation methodology, method landscape, and a close
  reading of RoboDrop (September 2026, 21 pages).
- [dataflow.md](dataflow.md) — the end-to-end data flow the document describes, from raw
  LIBERO demonstrations through screening, evaluation, attribution and verification
  (derived from §9, pp. 18–19).

**Goal:** explain which *training data* contributed to a good or bad VLA training result,
with model / config / schedule / eval protocol held fixed so data is the only variable.

**Approach:** attribution grounded in closed-loop rollout outcomes (not offline loss),
evaluated with PhAIL's distributional primitives, validated with planted ground truth
(inject 5–10% corrupted episodes, measure recovery) plus counterfactual retraining.

**Differentiators (Section 1 of the doc):**

- Clean data can still be harmful — per-sample compatibility cannot see mixture-level harm
  (Pebblous: 93 physically correct episodes, 73% → 43%).
- No trusted reference set needed — rollout outcomes are the reference, so it works on new
  collection campaigns and heterogeneous multi-source data where "clean" is undefined.
- Diagnosis, not just filtering — it names the episodes behind a specific failure on a
  specific task.
- Objective dependence — one score per metric; curating for throughput vs. reliability
  selects different subsets of the same data.
- Timing and permanence — a warm-up-epoch verdict is permanent, yet data that conflicts
  early may matter later; outcome scores are reversible and can reweight instead of delete.
- It finds helpful data too, pointing at what to collect next; a filter only removes.
- **Headline novelty:** CUPID's estimator, new application — per-metric and
  per-failure-type returns producing an episode × metric matrix (Section 4).
- **Sharpest testable claim:** divergence on a single metric, exposing a filter's implicit
  bias toward reliability over throughput (Section 8.10).

## Papers (`papers/`)

All papers cited in the PoC document, downloaded from arXiv.

### Primary attribution / curation methods

| # | Paper | Venue / date | arXiv | Role in the PoC |
|---|-------|--------------|-------|-----------------|
| 2 | **CUPID** — Curating Data your Robot Loves with Influence Functions (Agia et al., Stanford) | CoRL 2025 | [2506.19121](https://arxiv.org/abs/2506.19121) | Closest match to the PoC; best starting point to fork. Code: [agiachris/cupid](https://github.com/agiachris/cupid) |
| 6 | **RoboDrop** — Curating VLA Post-Training Data via Local Gradient Compatibility (Xu et al.) | Sep 2026 | [2609.10021](https://arxiv.org/abs/2609.10021) | Screening component; its corruption protocol is the template for the injection test (Section 8) |
| 3 | **DataMIL** — Selecting Data for Robot Imitation Learning with Datamodels (Dass et al., UT Austin / MIT / Stanford) | ICLR 2026 | [2505.09603](https://arxiv.org/abs/2505.09603) | Reference for the retraining-based, causal end of the spectrum |
| 5 | **ATHENA** — Accelerated Multi-Task Heterogeneous Influence Functions for Robot Data Curation | 2026 | [2606.16208](https://arxiv.org/abs/2606.16208) | Reference for scaling attribution to real VLAs (~313× faster influence) |
| 4 | **Quality over Quantity (QoQ)** — Demonstration Curation via Influence Functions | 2026 | [2603.09056](https://arxiv.org/abs/2603.09056) | Simpler influence variant; useful baseline |

### Baselines to beat

| # | Paper | Venue / date | arXiv | Role in the PoC |
|---|-------|--------------|-------|-----------------|
| 7 | **DemInf** — Robot Data Curation with Mutual Information Estimators (Hejna et al., Stanford / GDM) | RSS 2025 | [2502.08623](https://arxiv.org/abs/2502.08623) | Cheap unsupervised baseline the PoC must beat |
| 8 | **Demo-SCORE** — Curating Demonstrations using Online Experience (Chen et al.) | 2025 | [2503.03707](https://arxiv.org/abs/2503.03707) | Rollout-grounded but correlational baseline |

### Mixture-level and retrieval-based curation (context)

| # | Paper | Venue / date | arXiv | Approach |
|---|-------|--------------|-------|----------|
| 9 | **Re-Mix** — Optimizing Data Mixtures for Large-Scale Imitation Learning (Hejna et al.) | 2024 | [2408.14037](https://arxiv.org/abs/2408.14037) | Learns data-source mixture weights with distributionally robust optimization |
| 10 | **SCIZOR** — Self-Supervised Data Curation for Large-Scale Imitation Learning (Zhang et al.) | 2025 | [2505.22626](https://arxiv.org/abs/2505.22626) | Removes low-quality and redundant data with self-supervised signals |
| 11 | **STRAP** — Robot Sub-Trajectory Retrieval for Augmented Policy Learning (Memmel et al.) | ICLR 2025 | [2412.15182](https://arxiv.org/abs/2412.15182) | Retrieves sub-trajectories similar to a target task |

### Evaluation and benchmarks

| # | Paper | Venue / date | arXiv | Role in the PoC |
|---|-------|--------------|-------|-----------------|
| 1 | **PhAIL** — A Real-Robot VLA Benchmark and Distributional Methodology (Arkhangelskiy, Positronic Robotics) | May 2026 | [2605.29710](https://arxiv.org/abs/2605.29710) | Evaluation layer and real-robot stage (Section 3). [phail.ai](https://phail.ai) · [Positronic-Robotics/phail-paper](https://github.com/Positronic-Robotics/phail-paper) |
| 12 | **LIBERO** — Benchmarking Knowledge Transfer for Lifelong Robot Learning (Liu et al.) | NeurIPS 2023 | [2306.03310](https://arxiv.org/abs/2306.03310) | Default simulation benchmark; task suites make failures easy to localize |

### Not downloaded

- **Ref 13** — Pebblous, *Robot Data Curation & the Closed-Loop Gap*, July 2026,
  [blog.pebblous.ai](https://blog.pebblous.ai). A blog post rather than a paper; it is
  not in the site's current RSS feed, so no direct URL was found. Cited in the PoC as
  validation of the problem framing (the GR00T mixing experiment: +93 physically correct
  synthetic episodes dropped closed-loop success 73% → 43% while offline error stayed
  near-perfect).

## Tooling referenced in the PoC (not papers)

- **Lineage:** W&B Artifacts, MLflow, or DVC
- **Simulation:** LIBERO (default), SimplerEnv (real-robot correlation)
- **Models:** SmolVLA via LeRobot (single GPU); OpenVLA-OFT or π0 via openpi (more compute)
- **Data format:** LeRobot dataset format + Parquet episode-metadata table
