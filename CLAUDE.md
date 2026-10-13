# VLA Training-Data Attribution PoC

`docs/VLA_Data_Attribution_PoC.pdf` is the spec for this repo. `docs/papers/` holds the 12
papers it cites. `tracker/features/` holds one folder per feature (plan, evidence, analyzer
verdicts, stakeholder report); `tracker/index.html` lists them with a spec-coverage table.

## Required workflow

**Any request to build, implement, add, extend or fix something here must use the
`vla-build` skill** (`.claude/skills/vla-build/SKILL.md`). Invoke it before writing code.
It enforces: read the relevant PoC section from the PDF → apply the change → ground it back
into the PoC or a paper in `docs/papers/` (code docstring) → record the change in its feature
folder under `tracker/features/` (the full pipeline record for a feature, a `changelog.md`
entry for anything smaller).

A change with no grounding and no record in its feature folder is not finished.

**When the user explicitly asks to build a feature, use the `feature-pipeline` skill**
(`.claude/skills/feature-pipeline/SKILL.md`) instead of running `vla-build` inline. It
orchestrates five subagents in `.claude/agents/`: `vla-planner` (stories and acceptance criteria
from the PoC), `vla-executor`, `vla-analyzer` (a PASS/REWORK gate that routes work back),
`vla-refactorer`, and `vla-summarizer` (per-story docs and a stakeholder HTML report). Each one
preloads `vla-build`, so the same five steps still apply, divided between the agents. Small
fixes, questions and one-off scripts don't go through the pipeline.

**Call notes go to `calls/` via the `call-notes` skill** (`.claude/skills/call-notes/SKILL.md`). That is
business record-keeping, not PoC work: no `vla-build`, no feature folder. `calls/` is gitignored.

## Ground every answer in what already exists

**For any request — a build, a research question, a design opinion, a throwaway script —
establish what off-the-shelf tooling already solves it before proposing or writing a
mechanism.** Check the released code of the papers in `docs/papers/`, the robot-learning
data stack (LeRobot, robomimic, RLDS, Open X-Embodiment), the tracking and versioning
stack (W&B Artifacts, MLflow, DVC, HF Hub), and general-purpose libraries.

**Search the web, more than once, and read the official docs** for every tool, format, API or
method relied on — several phrasings, `WebSearch`/`WebFetch`, the real doc page rather than a
snippet. One result is a lead, not a fact. Never answer from memory of a library's API.

## Label every claim by its evidence

State how you know, inline, for any factual or causal claim: **quoted** (from a source, with
the quote), **measured** (you ran it, and it is reproducible), or **inferred** (your reasoning,
unverified). The point is not the label on a good claim — it is that an *unlabelled*
mechanistic claim becomes visibly a guess at the moment of writing.

This exists because of a specific failure: I stated that OpenVLA dropped 189 LIBERO
demonstrations because their re-render diverged, as though it were established. The paper says
no such thing — it says imitation learning wants successful demonstrations. My explanation was
a guess presented as fact, and "verify your claims" did not catch it because I believed I had
verified it. Only the labelling does.

**Then verify every claim against the artifact, not the prose.** Order of authority: the bytes on
disk and the library's source, then official docs, then derived artifacts and summaries, then
your own inference (which is not evidence). Corroborate across several sources; where one
disagrees with the artifact, the artifact wins and you record the discrepancy. Before adopting
any external filter, threshold or preprocessing step, confirm the problem it solves actually
exists in our data — I once applied a filter that removed 19% of the corpus on a premise our
data disproved (every LIBERO demo reaches its goal when replayed through LIBERO's own goal
check, 1000/1000). Check how a field was produced before treating it as evidence: the HDF5
`rewards` field I first cited for this is not a goal check, because LIBERO's `create_dataset.py`
writes `rewards[-1] = 1` unconditionally.

Verify by fetching the schema or reading the spec — not from memory. Then say plainly what
exists, what it does not cover, and whether the gap justifies custom code. If it is already
solved, say so and stop. Presenting a hand-rolled mechanism as necessary when the ecosystem
already ships a better one is the failure this exists to prevent.

## Environment

Python lives in `.venv`. Use `.venv/bin/python`; `pypdf` is installed for reading the spec.
