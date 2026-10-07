VERDICT: PASS
ROUTE: none
ROUND: 3

<!-- Saved verbatim by the orchestrator from the analyzer's returned text: the analyzer's own
     write was refused by the harness. -->

# Analysis, round 3 (post-ruling re-check): evaluation layer

Analyzer: vla-analyzer, 2026-10-07. The re-check follows the owner's ruling on `dispute-r2.md`, "B. Accept 4/5
(executor)" (2026-10-06, verbatim in `brief.md` "## Owner rulings"). That ruling is final and not re-opened
here. Labels: **measured** (I ran it), **quoted**, **inferred**.

## Summary (2–3 lines)

AC7.1, as restated in plan r3, holds on the stored seed sweep, which is identical to my round-2 re-run:
- GR00T–ACT gives (ii) in 4/5 seeds; the gate fires in 5/5; RMST favours GR00T in 5/5; the AUC Holm p is
  0.024–0.054;
- the other four pairs match in 5/5.

The divergence is recorded correctly as owner-approved in plan r3 (AC7.1, AC7.5 and "Changes in this
revision (r3)"). Since round 2 only records and `plan.md` changed, so every round-2 AC result stands. The
round-2 non-blocking items are carried forward.

## Rebuttals received

| From | Point disputed | Their evidence | Conceded? | Why / why not |
|---|---|---|---|---|
| — | AC7.1 GR00T–ACT seed stability (`dispute-r2.md`) | — | n/a: **owner ruling B**, final | Accepted as an owner-approved divergence; not re-opened |

## Acceptance criteria

| Story | AC | Claimed | Reproduced? | Evidence |
|---|---|---|---|---|
| S7 | AC7.1 (r3) | ✅ under the ruling | ✅ | measured, parsing `artifacts/s7_ac7_1_seed_sweep_stdout.txt` (identical to my round-2 re-run of `verify_verdict_s7.py --phail-seeds 0 1 2 3 4`; code unchanged since): OpenPI–GR00T (iii) 5/5, gate p 0.1129–0.1174; OpenPI–ACT (iii) 5/5, gate p 0.0960–0.1079; **GR00T–ACT (ii) 4/5, gate fires 5/5 (p 0.0065–0.0100), RMST → GR00T 5/5, AUC Holm (ACT side) 0.030 / 0.054 / 0.040 / 0.032 / 0.024**; OpenPI / GR00T / ACT vs SmolVLA (i) 5/5, gate p 0.0005 |
| S7 | AC7.5 (r3 note) | pending server | n/a | read: plan r3 states that n_boot = 1000 server verdicts near α may depend on the bootstrap seed, and that the verdict records its seed |
| all | round-2 ACs (AC2.9, AC5.3, AC5.9, AC5.10, AC6.1, AC6.8–6.11, AC7.6–7.9, AC9.1, AC9.2/AC8.5, and the regressions) | ✅ | ✅ (round 2) | measured: `git diff --stat 2f4e54f..HEAD -- vlapoc scripts` is empty. The only commits after round 2 are `baf3937` (records: analysis-r2, dispute-r2, brief ruling) and `3e5cf0c` (plan r3). `mlflow.db` sha `7845d8d7…` unchanged; `git status` clean |

## The divergence record (checked)

- `plan.md` "Changes in this revision (r3)" quotes the ruling verbatim: "B. Accept 4/5 (executor)". It says
  that nothing else changed, that n_boot stays 1000 including AC7.5, and that this is an owner-approved
  divergence that the summarizer carries into the report and the §3.3 / §4.4 spec-coverage rows.
- AC7.1 carries the case file's Option B wording exactly. It cites `dispute-r2.md` and brief "Owner rulings".
- AC7.5 gains the seed caveat and records the seed.
- `brief.md` "## Owner rulings" records the question, the verbatim decision and its meaning.

## PoC scope assessment

Unchanged from round 2:
- The §10 p. 20 "Evaluation layer" row is built locally and validated on PhAIL's released data.
- AC5.8, AC6.7, AC7.5 and AC8.5 are pending server, with the runbook in place.
- All divergences are recorded. One is owner-approved: the GR00T–ACT (ii) is seed-dependent at n_boot 1000.

## Overcomplication & reinvention audit

No new mechanisms since round 2 (no code diff). See `analysis-r2.md`.

## Feedback: planner

None blocking.

## Feedback: executor

None blocking.

## Non-blocking (for the refactorer and the summarizer)

- **N1.** Scratch-store tests write outputs into the real `data/eval/` (`verify_run` / `eval_run` default
  `out_dir` ignores `lineage_root`). Default them under the scratch root.
- **N2.** The round-1 records, including both real SmolVLA runs, fail `validate_record` (missing
  `t0_model_sha256`). Add a schema-version field or a documented legacy exemption.
- **N3.** The batch-plan guard does not compare τ_episode, render size or device. Consider adding them to the
  plan key.
- **N4.** `data/lineage_scratch/r2` (4.1 MB) and `data/lineage_scratch/s5` (2.3 GB) remain. Owner cleanup.
- **N5.** Re-run the data+lineage full dry run and `verify_runs_s5.py`. Move the three older scripts onto
  `runs.scratch_lineage_copy`. Later features must enforce the v1 fence at method entry points.
- **N6 (new).** `scripts/verify_verdict_s7.py --phail-seeds` still prints `all 30 (pair, seed) outcomes match
  the r2 restatement: False`. Its check encodes the superseded 5/5 rule. Update it to the r3 criterion
  (GR00T–ACT (ii) in ≥ 4/5, gate and RMST 5/5) so the script's pass line agrees with the AC (refactorer).
- **N7 (new).** Plan AC7.1's OpenPI–ACT note "Our gate measured 0.105–0.123" is the round-1 (unsorted-export)
  range. On the sorted export it is 0.0960–0.1079 (measured above). The conclusion (iii) is unchanged; use the
  current range in the report (summarizer).
- **For the summarizer.** State the owner-approved GR00T–ACT divergence and the AC7.5 seed caveat in the report
  and in the §3.3 / §4.4 coverage rows. Name pos3 as the PhAIL fetch route. State the not-built failure types
  next to the ghost-event statement.
