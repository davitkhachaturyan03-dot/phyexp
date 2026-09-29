---
name: feature-pipeline
description: Multi-agent build pipeline for this VLA PoC. Use ONLY when the user explicitly asks to build a feature, e.g. "build the KM CDF feature", "implement the injection test", "let's build per-metric attribution", "new feature — …". It runs five project subagents in order (vla-planner → vla-executor → vla-analyzer, looping back on REWORK → vla-refactorer → vla-summarizer) and produces stories with acceptance criteria, verified code, and a stakeholder-ready HTML report. Do not use it for questions, research, small fixes or one-off scripts; those follow vla-build directly.
---

# Feature pipeline: orchestration protocol

You, the main session, are the **orchestrator**. The five agents in `.claude/agents/` do the
work. You dispatch them, pass state between them, route the analyzer's verdicts, and keep the
user informed. You don't write the feature code yourself.

The agents each preload `vla-build`, so its five steps are divided between them:

| vla-build step | Owner |
|---|---|
| 1. Read the PoC section | vla-planner (the executor and analyzer re-read it too) |
| 2. Prior art, verified against the artifact | vla-planner, re-audited independently by vla-analyzer (overcomplication/reinvention audit, can force REWORK) |
| 3. Apply the change | vla-executor |
| 4a/b. Grounding docstrings, defining paper | vla-executor (the refactorer preserves them) |
| 5. Record: story pages, `report.html`, index entry + spec-coverage row | vla-summarizer |

## Shared state: one folder per feature

```
tracker/features/YYYY-MM-DD-<slug>/
  brief.md            ← you: the user's request verbatim + clarifications
  plan.md             ← planner: spec grounding, prior art, stories S1..Sn with ACs
  execution-log.md    ← executor: per-round, per-story evidence
  artifacts/          ← executor (and summarizer): real outputs used by the report
  analysis-r1.md …    ← analyzer: one per round, starts with VERDICT/ROUTE/ROUND
  dispute-rN.md       ← analyzer: case file for the owner when it won't concede
  refactor.md         ← refactorer: changes, behaviour check, reading guide
  S1-<slug>.html …    ← summarizer: per-story docs
  report.html         ← summarizer: the overall stakeholder report
  changelog.md        ← later non-pipeline changes to this feature (vla-build step 5)
tracker/index.html    ← summarizer: feature entry + spec-coverage rows (the only cross-feature file)
```

Agents start cold. Every dispatch prompt must give the **feature folder path**, the stage, the
round number, and which files to read. Don't paste file contents into the prompt; point at them.

## Protocol

Announce the pipeline to the user in one line at the start, then post a one-line status at each
stage transition ("Planner done: 4 stories, prior art says adapter over LeRobot. Executing S1–S4.").

**0. Set up.** Pick a slug, create the folder, and write `brief.md` with the request verbatim,
today's date, and anything the user has said in this conversation that constrains it.

**1. Plan.** Dispatch `vla-planner`, running in the foreground since you need its result.
- If the prior-art verdict is **already solved, stop**, tell the user what exists and stop.
  That's a successful outcome.
- If `plan.md` lists **BLOCKING** questions, ask the user (AskUserQuestion), append the answers
  to `brief.md`, and send them to the planner (SendMessage to the same agent) to revise.
- Otherwise show the user the story list in a few lines and continue. Don't wait for approval
  unless the user has asked for a plan gate.

**2. Execute.** Dispatch `vla-executor` for round 1. For a plan with more than ~4 stories,
dispatch in batches that respect the dependency order (e.g. S1–S3, then S4–S6), each batch
continuing the same log.

**3. Analyze.** Dispatch `vla-analyzer` with `ROUND: N`. Read the first three lines of
`analysis-rN.md`:

| Verdict | Route | You do |
|---|---|---|
| PASS | none | go to step 4 |
| REWORK | executor | SendMessage to the executor (resume it, so it keeps context) pointing at `analysis-rN.md`, then analyze round N+1 |
| REWORK | planner | SendMessage to the planner to revise `plan.md`, then the executor implements the delta, then analyze N+1 |
| REWORK | planner+executor | the planner first, then the executor, then analyze N+1 |
| ESCALATE | — | stop the loop and run **Disputes and escalations** below |

### Disputes and escalations

The analyzer is not required to agree with anyone. When the planner or executor rebuts its
feedback and the analyzer isn't convinced, it escalates with a case file
(`dispute-rN.md`, named on the `CASE:` line). Handle it like this:

1. **Don't referee it yourself.** Don't side with either agent, don't soften the analyzer's
   position, and don't send it back for "one more try" to reach consensus. The owner decides.
2. **Present the case file in full.** Show the owner the complete `dispute-rN.md` content
   in your reply, not a summary, then ask for their ruling. Use AskUserQuestion with the case
   file's options (A/B/C, with the analyzer's recommendation first), so they can also answer
   with free-text guidance.
3. **Record the ruling.** Append it to `brief.md` under `## Owner rulings` with the date, the
   dispute file, the owner's decision and guidance verbatim.
4. **Route it** as the case file's "If you rule" section says. SendMessage to the relevant
   agent(s) with the ruling, then analyze again. The analyzer treats a recorded owner ruling as
   final and doesn't re-open it. An owner-approved divergence from the spec still goes into
   the report and the spec-coverage table in `tracker/index.html`, labelled as owner-approved.
5. A dispute round **doesn't count** toward the 3-round limit.

**Round limit: 3.** If round 3 still isn't PASS, escalate to the user with the outstanding items.
Don't loop further on your own. If an agent can't be resumed, dispatch a fresh one and tell it
which round this is and which analysis file to act on.

**4. Refactor.** Dispatch `vla-refactorer`. If its behaviour check is not clean and it couldn't
revert, send it back once. If it's still not clean, dispatch the analyzer for one more round
scoped to the refactored files.

**5. Summarize.** Dispatch `vla-summarizer`. Then check the deliverable yourself: open the
overall report and one story page in the browser pane via the `tracker` server
(`preview_start` name `tracker`, then `http://localhost:8765/tracker/features/<feature>/report.html`;
`file://` pages load as snapshots without their stylesheet), and confirm the SVGs render,
links resolve, and no `{{` placeholders remain. Send it back with specifics if not.

**6. Hand off.**
- The report is for the owner **and stakeholders**, so publish it as an Artifact. The report
  links `../../assets/tracker.css`, so make a copy in the scratchpad with that CSS inlined in a
  `<style>` block and publish the copy. Story pages stay local and linked from the repo copy.
- Reply to the user with: what was built, the spec section, the story results (ACs met/total),
  how many analyzer rounds and what they caught, any divergence from the spec, the local report
  path and the artifact link.

## Rules for the orchestrator

- Don't do an agent's job to save time. If an agent's output is weak, route it back with
  specific feedback.
- Relay honestly. If a story ended ❌ or was descoped, the final message says so.
- Run agents in the foreground (`run_in_background: false`) when the next step depends on them.
  That is every stage here, because the pipeline is sequential.
- A feature isn't finished until its folder holds `report.html` and the story pages, and
  `tracker/index.html` has its feature entry and spec-coverage rows. That is the same "done" as
  `vla-build`.
