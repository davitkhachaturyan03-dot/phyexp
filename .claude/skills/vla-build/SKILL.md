---
name: vla-build
description: Mandatory workflow for building anything in this VLA training-data attribution PoC. Use whenever the user asks to build, implement, add, create, write, wire up, scaffold, set up, extend, refactor or fix any code, module, script, pipeline, metric, experiment, dataset tool, dashboard or figure in this repo — including phrases like "build the lineage layer", "implement the KM CDF", "add the injection test", "let's start on attribution", "make the macro-KS gate", "set up the episode catalog". Enforces five steps — read the relevant PoC section first, check whether an off-the-shelf tool or format already solves it and verify every claim about it against the artifact rather than the prose, apply the change, ground it back into the PoC document or the papers in docs/papers/, and record the change in its feature folder under tracker/features/.
---

# Building in the VLA attribution PoC

> **Feature builds run through `feature-pipeline`.** When the user explicitly asks for a
> feature, the `feature-pipeline` skill divides these five steps between the `vla-*` subagents
> (see its ownership table). For anything smaller (a fix, a tweak, a one-off script), follow
> the five steps yourself. Step 5 tells you where the record goes.

This repo implements the proof of concept described in `docs/VLA_Data_Attribution_PoC.pdf`
(21 pages). That document is the spec. Nothing gets built here without being tied to it.

**Every build follows five steps, in order. None are optional.** Step 4 and step 5 are not
"documentation to do later". A change without grounding and a record in its feature folder
is not finished, and you must not report it as finished.

Announce the five steps briefly at the start so the user can see the workflow running, then
work through them.

---

## Step 1 — Read the relevant section of the PoC first

Open the actual pages. Do not build from memory of the document, from this skill file, or
from a summary earlier in the conversation. The spec's details are the whole point — the
censoring rules, which bootstrap, which metric, the warm-up-epoch semantics — and a
paraphrase loses exactly those.

Dump it to text and grep for your topic:

```bash
.venv/bin/python -c "
from pypdf import PdfReader
r = PdfReader('docs/VLA_Data_Attribution_PoC.pdf')
for i, pg in enumerate(r.pages):
    print(f'--- p{i+1} ---'); print(pg.extract_text())
" > /tmp/poc.txt
grep -n -i -C3 'censor\|ghost event' /tmp/poc.txt
```

`pypdf` is installed in `.venv`. If reading fails, `.venv/bin/pip install pypdf` — don't
proceed without the spec.

The document's own contents, with page numbers:

| § | Section | Pages |
|---|---------|-------|
| 1 | The core question and scope | 3 |
| 2 | PoC design | 3–5 |
| 3 | Evaluation layer: PhAIL primitives | 6–8 |
| 4 | Metric-resolved attribution | 9–11 |
| 5 | Landscape of attribution and curation approaches | 11–12 |
| 6 | Reference papers | 12–13 |
| 7 | Industry landscape | 13 |
| 8 | RoboDrop in depth | 14–18 |
| 9 | Proposed combined architecture | 18–19 |
| 10 | First milestone and next steps | 20 |

Read the whole section, not the sentence that matched your grep — the constraints that make
a build correct are usually a paragraph away from the feature that names it. Where the
section imposes a rule (how failures are encoded, what gets resampled, what a test may and
may not be used for), follow it as written rather than as you would have designed it; if
you think it is wrong, build it as specified and say so to the user.

If the request maps to no section, say so explicitly before building, and ground against a
paper instead (step 3).

---

## Step 2 — Check what already exists before writing anything

**This step applies to every request, not only builds** — a research question, a design
opinion, a "how should we do X", a one-off script. Before proposing or writing a
mechanism, establish whether a standard tool, format or library already provides it.

The failure this prevents: presenting a hand-rolled mechanism as necessary when it is a
worse version of something the ecosystem already ships, and the user only finds out by
asking. That wastes their time and misrepresents the work.

**Look in this order:**

1. **The papers in `docs/papers/`.** Most methods here have public code — CUPID's is
   MIT-licensed and the PoC names forking it as the starting point (§6). Check whether
   the thing is already implemented before reimplementing it.
2. **The robot-learning data stack.** LeRobot's dataset format, robomimic (LIBERO's HDF5
   is robomimic-format, and robomimic supports filter keys for named subsets), RLDS/TFDS,
   Open X-Embodiment conventions.
3. **The experiment-tracking and versioning stack.** W&B Artifacts, MLflow, DVC,
   HuggingFace Hub revisions — the PoC names the first three explicitly for lineage (§2.1).
4. **General-purpose equivalents.** Content-addressed stores, derivation graphs, standard
   statistics libraries. Do not hand-roll a Kaplan-Meier estimator or a KS test.

**Verify, do not assume.** Fetch the schema, read the format spec, list the repo's API.
"LeRobot probably has this" is not a finding; its `meta/episodes` table having exactly 14
columns, none of which are action statistics, is.

### Verify every claim against the artifact, not the prose

Prior art is worthless if what you believe about it is wrong. Every factual claim you rely
on — about a dataset, a library, a paper's method, or **your own data** — gets checked
against the most authoritative source available. The hierarchy:

1. **The artifact itself** — the bytes on disk, the library's source, the released dataset
2. Official documentation, or the paper
3. Derived artifacts, third-party summaries, blog posts
4. Your own inference — this is not evidence

**Search the web, and search it more than once.** For every tool, format, API, dataset or
method you rely on: run several searches with different phrasings, and read the official
documentation — not your memory of it, which is frequently a version or two stale. Use
`WebSearch` and `WebFetch`; fetch the actual doc page, changelog, issue thread or source
file rather than trusting a snippet. One search result is a lead, not a fact; corroborate
across independent sources before treating anything as settled.

Then reconcile what you found against the artifact. **Where a source disagrees with the
artifact, the artifact wins** — and you record the discrepancy rather than quietly picking
one. Searching is how you find the claims; the artifact is how you settle them. Both are
required: in the session that produced this rule I *had* fetched the paper, the library
source and the converter, and still got it wrong, because I never opened my own data.

**Before adopting anyone else's decision, verify the problem it solves exists in your data.**
An external filter, threshold, preprocessing step or workaround encodes assumptions about
*their* setup. Check each assumption against yours before inheriting it.

#### The mistake this rule exists to prevent

In this repo I adopted OpenVLA's demonstration filter — which removes 189 of 1000 LIBERO
demos — on the assumption that LIBERO contained failed demonstrations that imitation
learning should exclude. I never checked whether it did.

LIBERO's HDF5 carries per-step `rewards` and `dones`, and every one of the 1000 demos ends with
`reward == 1`. At the time I read that as the goal predicate verified true and treated it as
settling the question. It wasn't evidence: LIBERO's `scripts/create_dataset.py` (commit
`8f1084e`, lines 228–231) sets `rewards[-1] = 1` and `dones[-1] = 1` unconditionally. The
conclusion was right for a different reason, found one feature later: replaying each demo's
stored states through LIBERO's own goal check succeeds in 1000/1000 (evaluation-layer plan,
premise P8). There were no failed demonstrations.

Cost: roughly an hour, 118 lines of matching code, a 22 MB download, and a corpus 19%
smaller than it should have been — to apply a filter whose premise was false. The user
challenged that premise three times before I ran a two-field check that, it later turned out,
could not have settled it.

The second lesson is the one that took longer to learn: a field's name is not its meaning.
Before treating a stored value as evidence, read the code that produced it.

The same session produced two more prose-vs-artifact mismatches, both caught only by
checking the artifact:

- OpenVLA's paper defines its no-op rule as including a gripper condition. The released
  data does not: with the condition, 167/179 episodes match; without it, 179/179.
- The paper describes the removals as demonstrations "that fail to complete the task."
  They complete the task in the original data; they fail to *reproduce* when re-stepped
  through the simulator, which only matters for a pipeline that re-renders.

Both would have become silent, confident errors in the report.

**Then report the comparison honestly, before building:**

- what already does this, and how much of the job it does
- what it does not cover, specifically
- whether the gap justifies custom code, or whether the standard tool plus a thin adapter
  is the better trade
- if you build custom anyway, say what you are giving up (interoperability, other
  people's tooling, maintenance)

Being wrong here is cheap before the code exists and expensive afterwards. Prefer the
boring standard option; reach for custom only where you can name what the standard one
cannot do. If the honest answer is "this is already solved, use X", say that and stop —
a build that should not happen is the best possible outcome of this step.

Record the result in the feature's prior-art section (`plan.md`, then the report, step 5) so the comparison is visible in
the deliverable rather than lost in conversation.

---

## Step 3 — Apply the change

Build what was asked, scoped to the request. Match the surrounding code's style.

- Keep the spec's vocabulary in the code: `ghost_event`, `censored`, `time_to_success`,
  `rmst`, `hrt`, `macro_ks`, `episode_id`. Names that match the document make the grounding
  self-evident and reviewable.
- Python via `.venv/bin/python`. Don't add a dependency when the standard library or an
  installed package will do; when you do add one, name it in the report.
- Every module or function that implements a spec requirement carries a grounding docstring:

```python
"""Kaplan-Meier time-to-success CDF with ghost events and right-censoring.

Grounding: PoC §3.2 "The evaluation primitives" (p. 6) — ghost events enter as
T = inf so that 1 - F(inf) is exactly the hard-failure rate; timeouts are censored,
not failed. CIs use the episode-clustered bootstrap, not per-operation resampling.
"""
```

- Prefer a small runnable demo or test over a claim. Something the report can show.

---

## Step 4 — Ground the change back into the source material

The change has to point back at where it came from, and the source material's coverage of
this feature has to point forward at the change. The PDF is read-only — grounding means
recorded, bidirectional citation, not editing the PDF.

Do all three:

**a) Cite in the code.** The grounding docstring from step 2, with section number *and*
page number, plus the specific sentence or rule it implements.

**b) Ground against a paper when the PoC is thin.** The PoC is an outline; it names methods
without specifying them. When you implement something it only gestures at, open the paper
that actually defines it, in `docs/papers/`, and cite the section you used:

| Building | Read | File in `docs/papers/` |
|---|---|---|
| Influence on returns, the CUPID estimator, per-metric returns | CUPID | `CUPID_Influence_Functions_arXiv-2506.19121.pdf` |
| Gradient-compatibility screening, warm-up scoring, matched references, auto threshold | RoboDrop | `RoboDrop_Gradient_Compatibility_arXiv-2609.10021.pdf` |
| KM CDF, ghost events, censoring, HRT, clustered bootstrap, macro-KS, P-P plots, blind protocol | PhAIL | `PhAIL_Real-Robot_VLA_Benchmark_arXiv-2605.29710.pdf` |
| Datamodels regression over random subsets, temporal clustering | DataMIL | `DataMIL_Datamodels_arXiv-2505.09603.pdf` |
| Scaling influence: gradient projection, low-rank Hessian, parameter subsets | ATHENA | `ATHENA_Influence_Functions_arXiv-2606.16208.pdf` |
| Influence on validation loss, per-trajectory aggregation | QoQ | `QoQ_Quality_over_Quantity_arXiv-2603.09056.pdf` |
| MI / intrinsic-quality baseline to beat | DemInf | `DemInf_Mutual_Information_arXiv-2502.08623.pdf` |
| Rollout-classifier baseline | Demo-SCORE | `Demo-SCORE_Online_Experience_arXiv-2503.03707.pdf` |
| Task suites, goal checks, demo durations, initial states | LIBERO | `LIBERO_Benchmark_arXiv-2306.03310.pdf` |
| Source-mixture weighting | Re-Mix | `Re-Mix_Data_Mixtures_arXiv-2408.14037.pdf` |
| Dedup / redundancy filtering | SCIZOR | `SCIZOR_Self-Supervised_Curation_arXiv-2505.22626.pdf` |
| Sub-trajectory retrieval, embedding kNN | STRAP | `STRAP_Sub-Trajectory_Retrieval_arXiv-2412.15182.pdf` |

Read the paper's actual formulation rather than reconstructing it — especially for CUPID's
estimator, the HRT ratio's linearization (§4.3), and RoboDrop's matching procedure, which
the PoC flags on p. 14 as unverified and needing checking against the paper.

Record divergences honestly. Where the build departs from the spec, or where the spec is
ambiguous and you chose, the code docstring and the feature record (step 5) both say so. A
silent divergence is the one failure mode this workflow exists to prevent.

---

## Step 5: Record the change in its feature folder

**The unit of record is the feature folder**, `tracker/features/YYYY-MM-DD-<slug>/`. Everything
about a feature lives there: the request, the plan and its grounding, the execution evidence,
the analyzer's verdicts, the reading guide, the story pages, and the stakeholder report
(`report.html`). `tracker/index.html` is the single entry point. It lists every feature and
carries the **spec-coverage table** that answers "which parts of the PoC are built, and where
do they diverge?"

**Feature builds** (the `feature-pipeline`): the `vla-summarizer` writes the whole record. See
its agent file for the report contents and the report-craft rules.

**Anything smaller** (a fix, a tweak, a script, a follow-up to a built feature):

1. Find the feature folder the change belongs to (the one whose code or data it touches).
   If none fits, create `tracker/features/YYYY-MM-DD-<slug>/` with a `brief.md` stating the
   request verbatim.
2. Append an entry to that folder's `changelog.md` (create it if missing):

   ```markdown
   ## 2026-10-02: <what changed, one line>
   - **Why:** the request, verbatim or close
   - **Grounding:** PoC §X.Y p.N / paper §N, or "none needed"
   - **Prior art:** what you checked and the verdict
   - **Files:** `path` (what changed)
   - **Evidence:** `<command>` → `<real, trimmed output>`
   - **Divergence:** none / what and why
   ```
3. If the change alters something the feature's `report.html` shows (a number, a figure, a
   behaviour, a divergence), update that section of the report too, and add a dated line to its
   "Changes since first release" list. Never leave a report showing numbers the code no longer
   produces.
4. If it changes spec coverage (a new § covered, or a new divergence), update that feature's
   row in the spec-coverage table in `tracker/index.html`.

The same honesty rules apply as for reports: **every number in a changelog or report traces to
a command actually run.** Never fabricate an output.

---

## Done means

- [ ] The relevant PoC pages were read in this session, from the PDF.
- [ ] Prior art checked and reported before building: what exists, what it misses.
- [ ] Web searched (several phrasings) and official docs read for every tool, format or
      method relied on, not answered from memory.
- [ ] Every load-bearing claim verified against the artifact (data, source, released
      dataset), not from the prose, a summary, or memory. Discrepancies recorded.
- [ ] For any external decision adopted: its premise confirmed to hold in our data.
- [ ] The change is applied and runs.
- [ ] Grounding docstring in the code, with § and page.
- [ ] The defining paper consulted where the PoC only gestures.
- [ ] The change is recorded in its feature folder: the full pipeline record for a feature, or a
      `changelog.md` entry (plus report and coverage updates if affected) for anything smaller.

Report to the user: what was built, the spec section it's grounded in, any divergence, and
where it's recorded. If you skipped a step, name it. Don't call the build complete.
