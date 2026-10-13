# Call digests

One file per call with anyone related to the startup: potential ICPs (ideal customer profiles),
data partners, pilot users, investors, advisors. Written by the `call-notes` skill
(`.claude/skills/call-notes/SKILL.md`) whenever you ask Claude to store call notes.

## File names

```
YYYY-MM-DD_HHMM_<type>_<org-or-person>_<topic>[_<topic>].md
```

- **Timestamp:** when the call happened, in local time. If the call's time isn't given, the time it was stored.
- **`<type>`:** one of `icp`, `partner`, `pilot`, `investor`, `advisor`, `other`.
- **The rest:** 2–4 lowercase kebab-case keywords naming who it was and what it was about.

Example: `2026-10-12_1430_pilot_acme-robotics_egocentric-data_licensing.md`

Files are named this way so that `ls calls/` reads as a chronological log, and `ls calls/*_pilot_*`
or `grep -l acme calls/*` finds what you need.

## What each file contains

1. A YAML header: date, time, timezone, who, organisation, type, keywords, and source (transcript, notes or
   recollection).
2. **The digest:** summary, their context, what they have, what we asked and learned, objections, commitments
   with owners, open questions and the next step. For data and pilot calls, there's also a pilot-feasibility
   checklist.
3. **The raw notes or transcript**, verbatim, so the digest can always be checked against the source.

## Privacy

This folder holds personal and commercial information about other people. Its contents are
**gitignored**; only this README is versioned. Keep it out of any shared or published repository.
