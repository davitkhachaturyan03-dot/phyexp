# Data flow: raw demonstrations → screening → attribution

Derived from [VLA_Data_Attribution_PoC.pdf](VLA_Data_Attribution_PoC.pdf), primarily
**§9 Proposed combined architecture** (pp. 18–19), with the injection step from §2.5 (p. 5)
and the screening pipeline's position from §8.6 (p. 15).

Everything below is the PoC document's own design unless marked *inferred* — those are
reading notes, not spec.

---

## Top-level pipeline (§9, p. 18)

```
1. Screening  →  2. Evaluation  →  3. Attribution  →  4. Verification
```

| Stage | What the document says it does |
|---|---|
| 1. Screening | RoboDrop-style gradient scores and DemInf flag suspicious episodes |
| 2. Evaluation | PhAIL primitives: time-to-success CDFs, ghost events, HRT, failure types |
| 3. Attribution | Per-metric influence vectors (F(30 s), HRT, RMST, AUC), per object and failure mode |
| 4. Verification | Retrain without attributed episodes; macro-KS gate, per-metric CIs for direction |

---

## Full flow

```
  LIBERO demos (50/task, human teleop)
        │
        ├──────────────► clean reference set          [§8.7: originals kept clean]
        │
        ▼
  corrupt 5–10% of a copy                              [§2.5 — 4 corruption types]
        │                                              ground-truth labels created here
        ▼
  TRAINING DATASET  (full, uncurated, corrupted included)
        │
        ├──────────────────────────┐
        │                          │
        ▼                          ▼
 ┌─ STEP 1: SCREENING ─┐    ┌─ (implicit) TRAIN ─┐
 │ in: dataset +       │    │ SmolVLA on the     │
 │     trusted clean   │    │ FULL dataset       │
 │ 1. DemInf per ep.   │    └─────────┬──────────┘
 │ 2. warm-up pass,    │              │
 │    gradient vs      │              ▼
 │    matched clean    │    ┌─ STEP 2: EVALUATION ──────────────┐
 │ 3. combine →        │    │ in: the trained policy            │
 │    "suspicion" rank │    │ 1. ~50 rollouts/task, fixed       │
 │ out: every episode  │    │    initial states                 │
 │   labeled how bad   │    │ 2. success + when, via the        │
 │   it LOOKS          │    │    benchmark's goal check         │
 └──────────┬──────────┘    │ 3. classify failures from sim     │
            │               │    state: ghost events, censored  │
   verdicts STORED,         │    timeouts, failure types        │
   NOT ACTED ON             │ 4. compute T-curves, F(30 s),     │
            │               │    RMST, HRT, AUC                 │
            │               │ out: behaviour per task, per      │
            │               │      metric, per failure type     │
            │               └─────────────┬─────────────────────┘
            │                             │
            │               ┌─ STEP 3: ATTRIBUTION ─────────────┐
            │               │ in: dataset + policy + rollouts   │
            │               │ 1. one RETURN per metric per      │
            │               │    rollout                        │
            │               │ 2. CUPID influence, per metric    │
            │               │ 3. repeat on subsets (one task,   │
            │               │    one failure type)              │
            │               │ out: episode × metric matrix      │
            │               └─────────────┬─────────────────────┘
            │                             │
            └──────► COMPARE ◄────────────┘   [§9.2: the payoff]
              "looks bad" vs "actually harmful"
                          │
                          ▼
            ┌─ STEP 4: VERIFICATION ────────────────┐
            │ in: the attribution rankings          │
            │ 1. drop top-k harmful, retrain;       │
            │    control: drop k random, retrain    │
            │ 2. evaluate both as in Step 2         │
            │ 3. macro-KS: did the distribution     │
            │    change at all?                     │
            │ 4. if yes, per-metric CI → direction  │
            │ 5. repeat for top-k HELPFUL           │
            │    (removing them should hurt)        │
            │ out: confirmation or refutation       │
            └───────────────────────────────────────┘
```

---

## Step detail (§9.1, pp. 18–19)

### Step 1 — Screening

| | |
|---|---|
| **In** | The training dataset and a small set of trusted clean demos |
| **Procedure** | 1. Run DemInf over the dataset — each episode scored for how consistent its actions are with what other demos do in similar states. 2. Run a short warm-up training pass (RoboDrop-style) — compare each sample's gradient with gradients from matched clean demos, aggregate into an episode score. 3. Combine the two into a "suspicion" ranking. |
| **Out** | Every episode labeled with how bad it *looks* |
| **Why** | Cheap (no rollouts), catches obvious errors early. **Its verdicts are stored rather than acted on**, so they can later be compared with what actually happened. |

### Step 2 — Evaluation

| | |
|---|---|
| **In** | A policy trained on the dataset |
| **Procedure** | 1. Many rollouts per task (e.g. 50) with fixed initial states. 2. Per rollout, record whether and when it succeeded, using the benchmark's goal check. 3. Classify failures with rules over simulator state: ghost events (unrecoverable, e.g. object off the table), censored timeouts, failure types (wrong object, missed grasp, idle). 4. Compute the metrics: time-to-success curves, success within 30 s, RMST, HRT against human demo durations, AUC against humans. |
| **Out** | A detailed record of how the policy behaves, per task, per metric and per failure type |
| **Why** | This is the ground truth. Attribution is only as good as this measurement. |

### Step 3 — Attribution

| | |
|---|---|
| **In** | The training dataset, the trained policy, and the rollouts from Step 2 |
| **Procedure** | 1. Define one "return" per metric for each rollout — F(30 s) gives 1 or 0; for RMST the return is negative time; for AUC, whether the rollout beat a sampled human time. 2. For each metric, CUPID-style influence estimates how much each training episode pushed that return up or down. 3. Repeat for subsets, such as a specific task or only rollouts with a specific failure type. |
| **Out** | An episode × metric table of influence scores — e.g. "episode 412 hurts success within 30 s and has no effect on speed" |
| **Why** | The core of the PoC. Links behaviour back to specific data, separately for each metric. |

### Step 4 — Verification

| | |
|---|---|
| **In** | The attribution rankings |
| **Procedure** | 1. Remove the top-k most harmful episodes and retrain; as a control, remove k random episodes and retrain. 2. Evaluate both policies as in Step 2. 3. Apply the macro-KS test — did the time-to-success distribution change at all? 4. If it did, check each metric's confidence interval for the direction. 5. Repeat for the top-k most *helpful* episodes (removing them should make things worse). |
| **Out** | Confirmation, or refutation, that the attribution is causally correct |
| **Why** | Influence estimates can be wrong. This turns them from claims into verified results. |

---

## Three things about this flow that are easy to get wrong

**1. Screening does not filter anything.**
The sharpest departure from RoboDrop's own design. In §8.6 (p. 15) RoboDrop's warm-up feeds
a keep/drop rule and the main run trains on curated data only. Here, §9 Step 1 stores
verdicts rather than acting on them, and Step 2's input is a policy trained on *the dataset*
— not a curated subset. The screening pass exists to be **disagreed with**: §9.2 (p. 19)
names that disagreement as the most compelling output, and §8.10 builds its headline
experiment on it. Wiring screening into a filter destroys both.

**2. Training is never numbered.**
The document goes Step 1 → Step 2, where Step 2's input is already "a policy trained on the
dataset." The training run sits implicitly between them. In lineage terms that is the `run`
record, which everything downstream keys off — a gap to fill, not a step that doesn't exist.

**3. Returns are defined at Step 3, not Step 2.**
Step 2 stores rollout *behaviour*; Step 3 turns each rollout into one return per metric. So
the rollout store must hold primitives `(t_start, t_end, outcome, censored)`, never derived
scalars — metric definitions arrive after the rollouts exist. §4.3 (p. 10) additionally
notes HRT needs a first-order linearized approximation because it is a ratio.

---

## Left implicit by the document *(inferred)*

- **Warm-up weights.** Where they come from and whether they are discarded. §8.6 flags the
  restart-vs-continue question as open for RoboDrop; here the warm-up is purely a scoring
  pass, so restarting from the pretrained checkpoint is the only coherent choice.
- **The loop back.** Step 4 produces a curated dataset, but nothing says whether you
  re-enter at Step 1 or stop. §4.5's "same dataset, opposite curations" implies running the
  whole pipeline twice with different declared objectives rather than iterating to
  convergence.
- **Score combination.** Step 1.3 combines DemInf and RoboDrop scores into one suspicion
  ranking with no combination rule given.

---

## Related

- [VLA_Data_Attribution_PoC.pdf](VLA_Data_Attribution_PoC.pdf) — the spec
- [README.md](README.md) — paper index
- [papers/CUPID_Influence_Functions_arXiv-2506.19121.pdf](papers/CUPID_Influence_Functions_arXiv-2506.19121.pdf) — Step 3's estimator
- [papers/RoboDrop_Gradient_Compatibility_arXiv-2609.10021.pdf](papers/RoboDrop_Gradient_Compatibility_arXiv-2609.10021.pdf) — Step 1's gradient scoring
- [papers/PhAIL_Real-Robot_VLA_Benchmark_arXiv-2605.29710.pdf](papers/PhAIL_Real-Robot_VLA_Benchmark_arXiv-2605.29710.pdf) — Step 2's primitives, Step 4's gate
