---
name: call-notes
description: Store a call digest in calls/. Use whenever the user asks to store, save, log, file or record call notes, a call transcript, a meeting summary or a conversation with a potential customer (ICP), data partner, pilot user, investor or advisor — e.g. "store these call notes", "save my call with Acme", "log this transcript", "file notes from today's pilot call". Writes one timestamped Markdown file whose name carries keywords, with a structured digest plus the raw notes verbatim.
---

# Storing call notes

Every call with someone related to the startup gets one file in `calls/` at the repo root. Its
conventions are in `calls/README.md`. This skill is **not** PoC work: no `vla-build`, no feature folder,
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

8. **Reply** with the file path, the 3–5 summary bullets, and the next step with its owner and date. Don't
   commit: `calls/` is gitignored on purpose.

**Pilot feasibility section:** take its questions from `calls/instructions/call-instructions.md`, section
"Questions to ask to potential pilot cooperators", so the two stay in sync.

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

## Pilot feasibility (data, pilot or partner calls only; delete otherwise)
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

## Notes

- **Several calls at once:** write one file per call.
- **Follow-up material** for a call already stored: append it to that file under a dated
  `## Update YYYY-MM-DD` heading. Don't create a duplicate.
- **"What did X say?" or "summarise my calls with pilots":** read the matching files in `calls/` (by name,
  then grep) and answer from them, citing file names.
