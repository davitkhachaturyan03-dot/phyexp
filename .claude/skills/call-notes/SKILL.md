---
name: call-notes
description: Store a call digest in calls/. Use whenever the user asks to store, save, log, file or record call notes, a call transcript, a meeting summary or a conversation with a potential customer (ICP), data partner, pilot user, investor or advisor — e.g. "store these call notes", "save my call with Acme", "log this transcript", "file notes from today's pilot call". Writes one timestamped Markdown file whose name carries keywords, with a structured digest plus the raw notes verbatim, then a business-ideas file drawn from the call and the project in calls/outcome_business_ideas/.
---

# Storing call notes

Every call with someone related to the startup gets two files:

- a digest in `calls/` at the repo root;
- a business-ideas file in `calls/outcome_business_ideas/`, built from what the call revealed and what the
  project can already do.

The conventions are in `calls/README.md`. This skill is **not** PoC work: no `vla-build`, no feature folder,
no tracker entry.

## Steps

1. **Get the material.** Use what the user pasted or said: a transcript, rough notes, or a recollection.
   If they asked to store notes but gave none, ask for them. Never write a file from nothing.

2. **Get the timestamp.**
   - If the user states when the call happened, use that.
   - Otherwise use the current local time, from `date "+%Y-%m-%d %H:%M %Z"` (never from memory). Record in the
     header that it is the time of storing, not of the call.

3. **Classify the call.** `type` is one of:
   - `icp`: a potential customer;
   - `partner`: a data provider or collaborator;
   - `pilot`: someone piloting or about to pilot;
   - `investor`;
   - `advisor`;
   - `other`.

   If it's genuinely ambiguous, ask in one line.

4. **Pick keywords.** 2–4 lowercase kebab-case keywords: the organisation or person first, then the main
   topics, e.g. `acme-robotics`, `egocentric-data`, `licensing`. Use only words from the material, and no
   personal details beyond a name or company.

5. **Name the file** `calls/YYYY-MM-DD_HHMM_<type>_<keywords joined by _>.md`, keeping it under about 100
   characters. **Never overwrite:** if the name exists, append `_2`, `_3`, and so on.

6. **Write the file** with the template below. Fill each section only from the material; put
   `not stated` where it is silent.
   - Quote the other side's important statements verbatim, in quotation marks.
   - Label your own interpretation as *(inferred)*, so what they said stays distinct from what we concluded.
   - Never invent numbers, names, dates or commitments.

7. **Paste the raw material verbatim** under `## Raw notes`. Strip anything that is a credential (passwords,
   API keys, tokens); store nothing else sensitive beyond what the user gave.

8. **Write the business-ideas file** in `calls/outcome_business_ideas/`, creating the folder if it's missing.
   Name it like the call file with `_ideas` before `.md`, and apply the same never-overwrite rule. Use the
   ideas template below. Then add the `Business ideas:` link under the call file's title.
   - **Read first:**
     - the call file you just wrote;
     - the PoC spec's differentiators and §7 "Industry landscape" (`docs/VLA_Data_Attribution_PoC.pdf`,
       read with `.venv/bin/python` and `pypdf`);
     - what's built so far (`tracker/index.html`);
     - every earlier file in `calls/outcome_business_ideas/`.
   - **Start from the demand signal.** What did they say they need, or would pay for, in their words? Where
     does it differ from what the PoC builds? Every idea must follow from this call plus the project. No
     generic startup ideas.
   - **Check prior art before writing each idea**, as CLAUDE.md requires. Search several phrasings, fetch the
     real pages for each competitor or adjacent tool, and say what exists and what gap it leaves. If
     something already does it, say so, then drop or reshape the idea.
   - **Label every claim** *quoted*, *measured* or *inferred*. "Nobody sells X" is always *inferred* and
     tentative.
   - **Compare across calls.** Mark ideas that earlier calls also point to, since repeated demand is the
     strongest signal, and ideas this call weakens. Don't present an earlier idea as new; cite its file.
   - **Rank 3–5 ideas**, then give one recommendation and a concrete next step tied to this contact.
   - **If the call adds nothing new,** still write the file. Say so plainly, and list which earlier ideas it
     strengthens or weakens. Never pad with generic ideas.
   - The file is internal: don't contact anyone or publish anything because of it.

9. **Reply** with:
   - the call file's path, its 3–5 summary bullets, and the next step with its owner and date;
   - the ideas file's path and its recommendation, in one or two lines.

   Don't commit: `calls/` is gitignored on purpose, `outcome_business_ideas/` included.

**Pilot feasibility section:** take its questions from `calls/instructions/call-instructions.md`, so the two
stay in sync. Which section depends on the other side:
- someone who holds or collects data: "Questions to ask to potential pilot cooperators" (the checklist in
  the template);
- a lab that trains models: "Questions to ask labs training models". Title the section `## Lab fit`, with
  one item per section there (curation and screening, pain, evaluation, lineage, model/data/compute, buying
  data, access and the deal), marked the same way, and end with its green, yellow or red reading.

## Template

````markdown
---
date: YYYY-MM-DD
time: "HH:MM"
timezone: <e.g. PDT>
timestamp_source: call time given by user | time of storing
type: icp | partner | pilot | investor | advisor | other
organisation: <name or "not stated">
participants:
  - <name, role, side (them/us)>
keywords: [<kw1>, <kw2>, <kw3>]
source: transcript | notes | recollection
---

# <Organisation or person>: <one-line purpose of the call>

Business ideas: [<ideas file name>](outcome_business_ideas/<ideas file name>)

## Summary
- <3–5 bullets: what happened and what matters>

## Their context
- Who they are, what they do, what they're trying to achieve, their pain points.

## What they have
- Data, robots or embodiments, simulators, tools, budget, timeline: whatever was mentioned.

## What we asked and what we learned
- **Q:** … **A:** …

## Objections, risks, concerns

## Commitments and next steps
| Who | What | By when |
|---|---|---|

## Open questions
- Things still unknown that matter for the deal or pilot.

## Pilot feasibility (data, pilot or partner calls only; `## Lab fit` for labs training models; delete otherwise)
For each item, mark ✅ answered, ❓ unknown or ❌ blocker, and add one line of evidence from the call:
- **Embodiment and actions:** how it was collected (robot teleop, handheld gripper, egocentric human);
  whether actions are recorded, in what space, frame, units and rate; proprioception; calibration.
- **Evaluation access (deal-breaker):** a robot or sim to run a trained policy on; about 50 rollouts per
  task from fixed initial states; automatic success detection and timing; failure types; human reference
  runs.
- **Episode integrity:** anything filtered, trimmed or dropped, and by what rule; how success labels were
  produced; how language annotations were made; stable episode IDs.
- **Metadata:** source, task, instruction, scene, objects, operator, date, device.
- **Scale:** episodes, hours, tasks, demos per task, size, future collection campaigns.
- **Rights and privacy:** licence; whether we may modify the data (the injection test corrupts a copy);
  whether we may train on it and keep the models; faces, bystanders, consent.
- **Compute and delivery:** who trains and where; format; versioning; checksums.

## Raw notes
<verbatim transcript or notes as provided>
````

## Ideas template

````markdown
---
call: ../<call file name>
call_date: YYYY-MM-DD
written: YYYY-MM-DD
type: icp | partner | pilot | investor | advisor | other
organisation: <name or "not stated">
---

# Business ideas from <Organisation or person>, YYYY-MM-DD

## What the call revealed
- The demand signal in their words (quoted from the call file), and how it differs from what the PoC builds.

## Ideas
### 1. <short name>
- **What it is:** …
- **Who pays and why:** … (evidence from the call, quoted)
- **What we already have:** the built part of the project it reuses (feature folder or PoC section).
- **What exists already:** competitors and adjacent tools, each with a source, and the gap they leave.
- **Risks:** …

## Across calls
- Earlier ideas this call strengthens or weakens, with their file names. "First ideas file" if none.

## Recommendation and next step
- One recommendation, and the next step with its owner and the contact it involves.

## Sources
- [<title>](<url>)
````

## Notes

- **Several calls at once:** write one call file and one ideas file per call.
- **Follow-up material** for a call already stored: append it to that file under a dated
  `## Update YYYY-MM-DD` heading. Don't create a duplicate. If it changes the ideas, append a dated
  `## Update YYYY-MM-DD` section to the ideas file too.
- **"What did X say?" or "summarise my calls with pilots":** read the matching files in `calls/` (by name,
  then grep) and answer from them, citing file names.
- **"Which business ideas keep coming up?":** read `calls/outcome_business_ideas/` and answer from the
  "Across calls" sections, citing file names.
