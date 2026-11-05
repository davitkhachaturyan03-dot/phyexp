# Plan: curation headline results in the dashboard

Round 1 (planner), 2026-11-05. Branch `curation-results` at `8383995`. Revised r1 on 2026-11-05 after
[analysis-r1.md](analysis-r1.md) (branch at `95fb1d9`). The delta is listed in the section below. Everything else stands as accepted.

## Revision r1 (after analysis-r1)

Scope: only the analyzer's "Feedback: planner" items (B1, B3) and the self-contradiction it noted. Items already accepted
in analysis-r1 are not reopened: the across-seed rule, demo ranking by `harm_<m>`, the demo headline not being tuned,
`mark_rule` + `mark_point`, the `treated_advantage` copy, and the W-only drop-in for the headline. B2 (runbook check
8.5) and the execution-log labelling are executor items and need no plan change, because AC1.5 already requires
`verify_runbook.py` to pass.

**Changes in this revision**

1. **B1, headline rule.** New headline state **`not shown`**: the gate fired in at least ⌈2n/3⌉ seeds, but neither
   claim was confirmed or refuted. `inconclusive` is now kept only for a gate that fired in fewer than ⌈2n/3⌉ seeds.
   Both states get a bounded sentence. See S1, "The rule". **Changed ACs:**
   - **AC1.2** gains the fifth planted case (`not shown`, gate 3/3) and a boundary case (gate 2/3), and pins
     `h-inconclusive` at gate 0/3.
   - **AC1.3** adds the new headline sentences to the banned-word scan.
   - **AC1.6** requires `RULE` and the module docstring to name both states.
   - **AC4.7** is rewritten for the two states, with a card/table consistency check.
   - **S4 deliverables** add the page-docstring item 1 wording.
2. **Self-contradiction fixed.** The rule's sentence wording no longer contains the banned phrase. The plan now uses the
   executor's wording: "a change that is not detected is not evidence that removal leaves the policy unchanged".
   **Changed:** S1 "The rule", last bullet.
3. **B3, family O.** Decision: the page **shows recorded family-O verdicts in their own section**, alongside, outside
   the headline. It does not hide them behind a caption. Grounding and the measured facts are under S4 "Family O
   decision".
   - **New ACs:** AC1.7 (plain name for the O metric in step4's sentences), AC4.11 (the O section, or a visible "none
     recorded" caption) and AC5.5 (the drop-in runs the dry run's family-O verify commands and the page shows them).
   - **Changed:** the "Out of scope" family-O line, and the "Risks" section (two new premises).

No new BLOCKING question.

## Request

The owner asked: "can you add the curation headline results to the dashboard? treat this effort as a feature"
([brief.md](brief.md)). Round A of the server chain judges every curated arm with the two-step verdict, against the
original policy and against random removal. It also runs the PoC §4.5 "same dataset, opposite curations" head-to-head
and the Step 4 seed-consistency summary. All of this is recorded in the lineage store, but no dashboard page reads it
(measured by the orchestrator, brief "Context"; re-checked below). This feature adds one page, **Curation results**,
which reads those runs and shows them in plain language with provenance and lineage. It also adds a labelled synthetic
demo of the page to `data/lineage_demo`, plus the proof that real Round A runs drop in with no code change.

## Spec grounding (PDF read this session, `data/scratch/poc.txt`)

- **PoC §4.5 "Validation and the demo" (p.11)** (quoted): "'Same dataset, opposite curations.' Curate the dataset twice
  — once for HRT, once for reliability (e.g. F(30 s) or hard-failure rate) — retrain on each subset, and show that each
  curated policy wins on its own metric. This mirrors PhAIL's 'same data, opposite rankings' finding on the data side
  and would be the headline result of the PoC."
- **PoC §4.4 "Statistics and disclosure" (p.10)** (quoted): "Two-step verdict per intervention. First the macro-KS gate:
  did removing or adding this data change the time-to-success distribution at all? Then each metric's bootstrap CI
  gives the direction for that metric." "Crossing curves mean trade-offs." "Multiple comparisons. … correct for it
  (e.g. Bonferroni or Holm across metrics)". "Label every claim with its metric and time cap τ, following PhAIL's
  disclosure principle."
- **PoC §2.6 "Counterfactual verification" (p.5)** (quoted): "Remove the top-k 'harmful' episodes and retrain: the policy
  should improve more than after removing k random episodes. Remove the top-k 'helpful' episodes and retrain: the
  policy should degrade more than after random removal. Each of these is a close comparison between two policies, so
  use the PhAIL significance gate (Section 3.4) to decide whether the change is real before claiming it."
- **PoC §9.1 Step 4 "Verification" (p.19)** (quoted): items 1–5, "3 Apply the macro-KS test: did the time-to-success
  distribution change at all? 4 If it did, check each metric's confidence interval for the direction of the change.
  5 Repeat for the top-k most helpful episodes (removing them should make things worse)." "Output: Confirmation, or
  refutation, that the attribution is causally correct." "Why: Influence estimates can be wrong. This step turns them
  from claims into verified results."
- **PoC §2.3 (p.4)** (quoted): "Fixed config and steps; 2–3 seeds per data configuration | Prevents seed variance from
  masquerading as a data effect."
- **PoC §2.7 (p.5)** (quoted): "A dashboard: select a failed rollout … or browse a ranked list of harmful episodes,
  aggregated by any catalog column." This page extends the demonstration surface. §2.7 does not name it. §4.5 and §10
  do.
- **PoC §3.4 (p.7)** (quoted): "Verification gate. … Macro-KS is the significance gate; the HRT or RMST confidence interval
  gives the direction." "Declared targets. Scalars can rank policies in opposite orders. Every attribution result states
  which functional it explains, following PhAIL's disclosure principle."
- **PoC §4.3 (p.10)** (quoted): "HRT is a ratio, so it needs a first-order (linearized) approximation". This has
  consequences for the page. The HRT arm is *ranked* by the linearized return `hrt60_lin`, but it is *judged* on the
  HRT ratio `hrt60` (family W verdict metric; `metrics_registry.verdict_key_of` maps one to the other, measured in
  source). The page must state both functionals and must not present them as one.
- **PoC §10 (p.20)** (quoted): "Metric-resolved attribution | … 'same dataset, opposite curations' experiment";
  "Verification & demo | Counterfactual removal tests; dashboard linking failed rollouts to influential episodes".

**Rules the build must follow**

1. Direction is claimed only when the macro-KS gate fired. A seed whose gate did not fire contributes no direction
   (§4.4, §9.1 Step 4.3–4.4). The code already does this: `vlapoc.step4.call` returns `g` (measured in source).
2. Holm across the family-W metrics inside each verdict (§4.4). That is already in `verdict.compare_policies`, with
   owner ruling B: n_boot 1000.
3. Every number names its metric and τ (§4.4, §3.4). The page must show two declarations: the ranking functional (the
   select run's `metric_declaration`) and the verdict functional (the verdict's metric `label`, e.g.
   `F(30 s) / Success@τ [family W, τ 30 s]`).
4. The §2.6 claim is "more than random removal". So `curated_vs_random` is the comparison that tests it, and `vs_base`
   is shown alongside it.
5. Seeds: 3 per arm (§2.3). Across seeds, the existing Step 4 confirmation rule applies (`vlapoc.confirm.status_of`:
   expectation met in ≥ ⌈2n/3⌉ seeds with none contradicting → confirmed; the opposite in ≥ ⌈2n/3⌉ → refuted;
   otherwise not confirmed). This rule is **inferred**: it is recorded in `confirm.RULE` as "inferred; stated before any
   result" and was used for Round B in the dashboard feature. This plan reuses it rather than inventing a second rule.
6. Wording discipline: never "harmless", "no effect" or "no difference". A non-significant result reads "not shown at
   this sample size". The bounded-wording precedent is `vlapoc.planted_arms` and the existing `WORD` scan in
   `verify_dashboard.py`.

**Papers.** PhAIL §3.3 (outcomes i/ii/iii, gate then scalar) is already implemented and cited in `vlapoc/verdict.py`. No
new method is implemented here. This feature reads and words recorded verdicts. ATHENA §3.2/§4.2 grounds per-task top-k,
which is already cited in `server_pipeline.py`.

## What the chain records today (measured against the code)

Source: `scripts/server_pipeline.py` lines 404–436, `scripts/stage.py` `cmd_select`/`cmd_verify`/`cmd_step4_summary`,
`vlapoc/runs.py` `select_run`, `vlapoc/verdict.py` `verify_run`, and `vlapoc/step4.py`. The dry run is
`.venv/bin/python scripts/server_pipeline.py --tag v3cand --subset v3/candidate --curation per_task --metrics f30
hrt60_lin --dry-run`, which prints `would submit 116 jobs; chain v3cand-83839956; arms: ['harm-f30-k5', 'help-f30-k5',
'harm-hrt60_lin-k5', 'help-hrt60_lin-k5', 'random0-k5', 'random1-k5', 'drop-random-20']` (**measured**, output in
`data/scratch/roundA_dry.txt`).

| Output | Stage / run name | What it holds | Per seed / across seeds | Measured evidence |
|---|---|---|---|---|
| Arm selection | `select` run `select-v3cand-<arm>`, one per arm (7), in one CPU job | params `op`, `k`, `metric`, `direction`, `score_run_id`, `metric_declaration` (functional, τ), `curation` (`per_task`), `headline` (**False** for every arm: `--headline` is only passed with `--curation both`; 0 occurrences in the dry run), `parent_subset_id`, `child_n_episodes`; tag `select.dropped` (JSON, **cut at 7,900 chars**); `output.subset` = the child subset id, whose `derivation["dropped"]` is the full list | n/a | `runs.py` l.824–826, 885–888. 100 v3 episode ids serialise to 9,239 chars, so a k5-per-task arm's `select.dropped` tag is a **truncated prefix**. The full list must be read from the child subset (measured: v3/candidate has 960 episodes, 20 tasks × 48, longest id 124 chars) |
| vs base | `verify` run `verify-v3cand-<arm>-s<S>-{W,O}`, tags `step4.kind=vs_base`, `step4.treated=<arm>`, `step4.control=base`, `step4.seed` | `verdict.json`: gate D̄, p, fires; per metric A, B, Δ, 95% CI, p, `p_holm`, `diff_better`, `favours`, label with τ; outcome i/ii/iii/gate_only. Tags `verify.outcome`, `verify.direction`, `verify.gate_p`, `verify.attribution_grade` | per seed | 21 jobs × 2 families = **42 runs** (dry run: 21 `--comparison vs_base` jobs, each `W && O`) |
| curated vs random | `verify-v3cand-<arm>-vs-random<r>-k5-s<S>-W`, kind `curated_vs_random` | same verdict body, family W only | per seed, per random draw (2) | **24 runs** (4 curated arms × 2 draws × 3 seeds) |
| §4.5 head-to-head | `verify-v3cand-harm-hrt60_lin-k5-vs-harm-f30-k5-s<S>-W`, kind `head_to_head`, treated `harm-hrt60_lin-k5`, control `harm-f30-k5` (A = F30 arm, B = HRT arm) | same verdict body, family W | per seed | **3 runs**. **No single "headline verdict" is recorded.** The only across-seed record is the step4-summary row |
| Step 4 summary | `summarize` run `step4-summary-v3cand`, one CPU job depending on all 48 verify jobs | `step4_summary.csv`/`.json`: one row per (treated, kind, control, family) = **23 rows** (7 vs_base W + 7 vs_base O + 8 curated_vs_random + 1 head_to_head; inferred from the groupby over the dry-run verify set); per metric `<m>__by_seed` (T/C/0/g), `__T`, `__C`, `__expected`, `__supports`, `__consistency` (text); `gate_fires` "k/n"; `outcome_by_seed`; the treated arm's op, direction, curation, declared functional, τ. JSON adds `verify_run_ids`, chain, alpha | across seeds (counts only) | `step4.py` l.462–499, 546–568. For head_to_head, `expected_direction` sets hrt60 → treated_better and f30 → treated_worse (l.435–440). **It records counts but not a status.** No confirmed/refuted label and no sentence exist for Round A (Round B gets them from `confirm` and `discrepancy-verdict`) |
| Lineage | summary → verdict entities → verify → rollouts → eval → checkpoint → train → subset → select → score table → ensemble → attribute … | `runs.trace` walks it; `results.provenance` labels it | n/a | `runs.py` `output_file` sets `produced.<sha>` (l.250–257); `producers` finds it (l.966–974). The real `eval_run` records `input_entity("checkpoint")` (l.738) |

**Demo store and real store (measured, read-only sqlite):**
- `data/lineage_demo/mlflow.db`: 24 verify runs. `step4.kind` values are `curated_vs_random` (15) and `vs_base` (6). The
  treated arms are `cleanharm-f30`, `cluster0`, `cluster1`, `oracle` and three `type-*`. That is Round B only: **no Round A
  curation run** (no `harm-*`, `help-*`, `random*-k5`, `drop-random-20`, no `head_to_head`).
- `data/lineage/mlflow.db` (sha256 `306af412…bde79`, matching the brief): 3 verify runs (smoke) and no `step4.kind` tag.
  No Round A has been recorded here.
- The demo's existing Round B verdicts are hand-built JSON (`verify_dashboard.verdict_json`), not output from
  `verdict.compare_policies` (measured in source, l.981–991). The brief asks for this page's demo to come from the real
  verdict code where possible (S3).

**Does the page need a new server job?** No. Every per-seed number and every across-seed count is already recorded. The
one missing piece is a recorded **status and bounded sentence** per (arm, comparison, metric), plus the §4.5 headline
status. S1 adds these as extra columns of the **existing** `step4-summary` stage output. The job command, the job count
and the dependencies do not change. Consequence for the runbook: none to the counts (Round A 116/24, Round B 129/33 and
21/6 stay; `verify_runbook.py` must still pass). The server must run a commit that includes S1 for the recorded status
columns to exist. If it does not, the page computes the same columns from the recorded per-seed calls and says so
(S4 AC4.9).

## Prior art

| Candidate | What it covers | What it misses | Evidence |
|---|---|---|---|
| `vlapoc.step4.summarize`/`summary_run` (this repo) | Per-arm × comparison × metric per-seed calls, counts, expectations, declared functional; recorded and sha-checked | No status, no sentence, no §4.5 headline status | **measured**, source l.462–568 |
| `vlapoc.confirm.status_of` (this repo) | The ≥ ⌈2n/3⌉-seeds, none-contradicting rule → confirmed / refuted / not confirmed | Applied to cluster arms only | **measured**, source l.39–48 |
| `vlapoc.planted_arms.treated_advantage` and its bounded-sentence pattern (this repo) | Converts a verdict row into the treated arm's advantage with its CI in "better" units; bounded sentences, "SYNTHETIC:" prefix, the never-"harmless" rule | Planted arms only | **measured**, source l.169–178, 180–246 |
| `vlapoc.results` (`verdicts(body=True)`, `runs_with_param`, `find_runs`, `provenance`, `output_json`) | Sha-checked reads of verify and summarize runs, fail-closed provenance | No helper for "the Round A chain of this score run" and no reader for the full dropped list | **measured**, source |
| Dashboard pages `bad_vs_harmful.py`, `clusters.py` | Altair charts at ≥ 200 px, `common.banner`, `declared_caption`, lineage expander (MLflow link + trace command), `words` mapping | No curation page | **measured** |
| Altair `mark_errorbar` with pre-computed `x`/`x2` + `mark_point` | A forest plot (estimate and CI per row) with no new dependency | — | **quoted**, Altair 6.3 docs: "If the data is already pre-aggregated with low and high values of the error bars, you can directly specify `x` and `x2`". Installed: altair 6.3.0, streamlit 1.66.1 (**measured**) |
| `forestplot` (PyPI, LSYS) | Publication forest plots (matplotlib) | A new dependency; static image, so no tooltips; breaks the app's Altair pattern | **quoted**, PyPI/readthedocs: "rows correspond to a variable/study with columns including estimates, variable labels, and lower and upper confidence interval limits" |
| `st.column_config.LinkColumn` | Clickable MLflow links inside a table | — | **quoted**, Streamlit docs v1.66.0 signature includes `display_text` |
| MLflow UI run comparison | Side-by-side params, metrics and tags of selected runs | Verdict CIs live inside `verdict.json` artifacts; no across-seed rule; not plain language | **quoted** (third-party docs): the compare view shows "metrics and hyperparameters side-by-side"; artifacts are per run. **inferred**: the per-metric CIs are not comparable there without a reader |

**Verdict: adapter over what exists.** Reuse `step4` (add the status/sentence columns using `confirm.status_of`),
`planted_arms.treated_advantage`, `results`, and the existing page scaffolding with Altair `mark_errorbar`. No new
statistics, no new dependency, no new server job. The custom code is one page, a few loader functions, a status/wording
function, a demo section and checks.

## Stories

### S1: Record Round A statuses and the §4.5 headline in the step4-summary output
- **Why:** PoC §9.1 Step 4 asks for "Confirmation, or refutation". §4.5 asks for "each curated policy wins on its own
  metric". The chain records counts but no status (measured above). Recording the wording keeps the "shown as recorded,
  never re-worded in the app" discipline of the Bad vs harmful page (dashboard AC5.5b).
- **Depends on:** none.
- **Deliverables (proposed):** in `vlapoc/step4.py`, pure functions `status_row(calls, expected)` (reusing
  `confirm.status_of`) and `headline(table, records)`. `summarize` adds per metric `<m>__status` and `<m>__sentence`.
  `summary_run` writes a `headline` object into `step4_summary.json`, with per-seed readings and the across-seed status
  and sentence for every (mode, k) head-to-head pair, and records the rule text as a param. A recorded synthetic input
  (any collected verify or select run tagged `vlapoc.synthetic=true`, or a verdict carrying `"SYNTHETIC"`) prefixes every
  status and sentence with `SYNTHETIC: `. This is fail-closed and needs no flag.
- **The rule (inferred; stated before any result; recorded as a param; the owner may override):**
  - Per (arm, comparison, control, metric) with an expectation: `confirmed` / `refuted` / `not confirmed` from
    `confirm.status_of` over the seeds. A row is `inconclusive` instead of `not confirmed` when the gate fired in fewer
    than ⌈2n/3⌉ seeds (the distributions were not shown to differ). Metrics with no expectation (random arms,
    `drop-random-20`) read `no expectation (control arm)`.
  - §4.5 headline per (mode, k): speed claim = `status_of(hrt60 calls, "T")` (the HRT-curated arm better on HRT); reliability
    claim = `status_of(f30 calls, "C")` (the F30-curated arm better on F30). Both claims come from the same per-seed
    verdict, so they share one gate count `g` (seeds whose gate fired) out of `n`. **(r1)** The headline is, in this
    order:
    1. `contradicted` if either claim is refuted;
    2. `shown` if both are confirmed;
    3. `partly shown` if exactly one is confirmed and none is refuted;
    4. `inconclusive` if `g < ⌈2n/3⌉` (the two policies were not shown to differ, PoC §4.4 p.10: direction only after
       the gate fires). This matches the per-metric rule above, so both head-to-head metric rows then read
       `inconclusive` too;
    5. `not shown` otherwise: `g ≥ ⌈2n/3⌉` and neither claim is confirmed or refuted.

    Steps 1–3 cannot trigger when `g < ⌈2n/3⌉`, because confirmed and refuted each need ⌈2n/3⌉ significant calls and a
    significant call needs a fired gate. So the order is unambiguous (inferred from the definitions, and checked by the
    AC1.2 cases).
    - `not shown` sentence (bounded): "Same dataset, opposite curations: not shown at this sample size (n seeds × r
      rollouts per task). The two curated policies' time-to-success distributions differ (macro-KS gate fired in g of
      n seeds), but neither curated policy's win on its own metric replicated across seeds (speed claim on <HRT
      phrase>: <status>, s of n; reliability claim on <F30 phrase>: <status>, r of n). No winner is stated."
    - `inconclusive` sentence: "Same dataset, opposite curations: inconclusive. The macro-KS gate fired in only g of n
      seeds, so no direction is claimed for either curated policy. No winner is stated."
    - Every sentence names the gate-fired count and the number of seeds.
  - Sentences never contain "harmless", "no effect" or "no difference". A non-significant result reads "not shown at this
    sample size (n seeds × rollouts per task)" and adds **(r1)** "a change that is not detected is not evidence that
    removal leaves the policy unchanged" (the executor's wording, accepted in analysis-r1).
- **Acceptance criteria:**
  - [ ] AC1.1 On synthetic verdict records with 3 seeds (the `verify_training_attribution.py --step4` fixture, extended),
    the new columns equal a hand-computed table for these planted cases: confirmed (T,T,0), refuted (C,C,T),
    not confirmed (T,0,0 with the gate fired 3/3), inconclusive (gate fired 1/3), and no expectation (random arm).
    [PoC §9.1 Step 4 p.19; confirm.RULE]
  - [ ] AC1.2 **(r1)** The head-to-head headline gives these codes on planted synthetic cases (3 seeds, ⌈2·3/3⌉ = 2),
    with the expected codes written as literals in the fixture:
    - `shown`, `partly shown` and `contradicted` (as in round 1);
    - `h-inconclusive`: gate fired **0/3**, every call `g` → `inconclusive`;
    - `h-notshown` (the fifth case): gate fired **3/3**, hrt60 calls `C,0,0`, f30 calls `0,T,0` → `not shown`. This is
      the pattern the demo records (analysis-r1 B1, measured);
    - `h-notshown-boundary`: gate fired **2/3**, hrt60 `0,0,g`, f30 `0,0,g` → `not shown`. Gate 2/3 meets ⌈2n/3⌉, so
      this pins the threshold. The case is labelled inferred, because it tests the inferred rule.

    Every gate-not-fired seed reads "inconclusive (gate did not fire)" in its per-seed reading. Neither the
    `inconclusive` nor the `not shown` sentence contains "wins its own metric", and both contain "No winner is
    stated". [PoC §4.5 p.11; §4.4 p.10; brief "'inconclusive' where the gate did not fire"; analysis-r1 B1]
  - [ ] AC1.3 No status or sentence contains the forbidden words (case-insensitive scan for `harmless`, `no effect`,
    `no difference`). **(r1)** The scan covers the headline sentences of all six AC1.2 cases. Every sentence names the metric's
    plain name and τ. With a synthetic input, every status and sentence begins `SYNTHETIC: `. [PoC §4.4 p.10; brief
    wording discipline]
  - [ ] AC1.4 Existing columns and rows are unchanged. The `step4_summary.csv` produced by the existing `--step4`
    fixture, restricted to the old columns, is byte-identical to the one from HEAD `8383995`. `confirm` and
    `discrepancy-verdict` outputs on the demo store are unchanged (they read `collect`, not these columns).
    [vla-build regression]
  - [ ] AC1.5 No job change: the Round A dry run still prints 116 jobs, `scripts/verify_runbook.py` passes, and the
    `step4-summary` job command string is identical to the one at HEAD. [brief constraint]
  - [ ] AC1.6 The grounding docstring cites §9.1 Step 4 p.19, §4.5 p.11, §4.4 p.10 and §2.3 p.4, and labels the rule
    **inferred**. **(r1)** `step4.RULE` and the module docstring's headline bullet both name `not shown` and
    `inconclusive`, each with its gate condition (`grep -c "not shown" vlapoc/step4.py` ≥ 2, and the docstring bullet
    no longer says "inconclusive otherwise"). [vla-build; analysis-r1 B1]
  - [ ] AC1.7 **(r1, B3)** The family-O verdict metric is worded in plain language. `step4.metric_phrase("rmst60_op_cap")`
    returns a string that contains no `_` and contains `τ 60 s`. Today it returns `rmst60_op_cap (τ 60 s)` (**measured**
    in this revision with `.venv/bin/python -c "from vlapoc import step4; print(step4.metric_phrase('rmst60_op_cap'))"`).
    The plain name should come from the registry entry's `metric` field ("operation RMST, capped"), not from new
    hard-coded text. The family-W phrases are unchanged (AC1.3 still passes, and AC1.4's old columns are still
    byte-identical). [PoC §4.4 p.10 "Label every claim with its metric and time cap τ"; brief plain language]
- **Verify by:** `.venv/bin/python scripts/verify_training_attribution.py --step4` (extended) ;
  `.venv/bin/python scripts/verify_runbook.py` ; the dry-run command above | `tail -1`.
- **Demo value:** the recorded sentence the page shows verbatim, e.g. "SYNTHETIC: Removing the 100 episodes ranked most
  harmful for Success within 30 s improved Success within 30 s more than removing 100 random episodes (same per-task
  counts) in 3 of 3 seeds (gate fired 3/3): confirmed."

### S2: Read the Round A curation results from the lineage store
- **Why:** the brief says to read results from the lineage store only. The dropped list must come from the child subset,
  because the tag is truncated (measured 9,239 > 7,900 chars).
- **Depends on:** S1 (it reads the new columns, and tolerates their absence).
- **Deliverables (proposed):** in `vlapoc/results.py`:
  - `curation_runs(score_run_id)`: the chain of the `select` runs whose `score_run_id` param is this score run, the
    newest `summarize` run in that chain with `step4_summary_json` and at least one row whose `op` is a curation op
    (`drop_top_k[_per_task]`, `drop_random_k[_per_task]`), and the select runs of that chain.
  - `removed_episodes(select_run_id)`: the full dropped ids from the child subset's derivation, joined to the catalog
    (task, source) and, for ranked arms, to the ranked column's score and rank.
  - Per-seed verdict bodies through the existing `verdicts(chain=…, body=True)`.
  - All read-only and sha-checked. No ground-truth import (the existing `--results` source scan must still pass).
- **Acceptance criteria:**
  - [ ] AC2.1 On the S3 demo store, `curation_runs(<demo ensemble id>)` returns the demo's Round A summary run and its 7
    select runs, and ignores the Round B summaries in the same store (their rows have `op` `drop_cluster` or planted
    kinds). [brief: read from the store]
  - [ ] AC2.2 `removed_episodes` returns exactly `parent − child` for every arm (k5 per task → 100 ids, 5 per each of 20
    tasks; `drop-random-20` → 20). For each arm, the first ids equal the parsed prefix of the `select.dropped` tag.
    [PoC §2.6 p.5; measured truncation]
  - [ ] AC2.3 A tampered verdict or summary artifact raises (sha check), and a store with no Round A summary returns
    `None` instead of raising. [PoC §2.1 item 1 p.3 lineage]
  - [ ] AC2.4 Grounding docstrings cite §9.1 Step 4 p.19 and §2.6 p.5. `verify_dashboard.py --results` still passes.
    [vla-build]
- **Verify by:** `.venv/bin/python scripts/verify_dashboard.py --curation` (new mode; covers S2 and S4 checks) and
  `--results`.
- **Demo value:** "harm-f30-k5 removed 100 of 960 episodes (5 per task)", a list a stakeholder can open.

### S3: Add a SYNTHETIC Round A to the demo store, produced by the real verdict and step4 code
- **Why:** brief "done" item 4. Until the server runs, the page needs data. The verdicts should come from the real code,
  not from hand-built JSON.
- **Depends on:** S1, S2.
- **Deliverables (proposed):** a Round A section in `scripts/dashboard_demo.py build`, chain `demo-ra000000`:
  - **Real `select_run`s** on v3/candidate, using the demo ensemble's (synthetic) scores, for exactly the arms
    `server_pipeline.arms()` yields for `--curation per_task --metrics f30 hrt60_lin` (names and args from that
    function, not retyped).
  - **SYNTHETIC** `train` runs per arm × seed (no model; each consumes its child subset and produces a checkpoint id) and
    SYNTHETIC `eval` runs (rollouts parquet with `suite`, `libero_task_id`, `rollout_id`, `T_s`, `event`,
    `tau_episode_steps`; each consumes its checkpoint). Outcomes are drawn with planted per-arm effects: the HRT arm
    faster, the F30 arm more reliable, help arms worse, random arms ≈ base. At least one comparison is planted weak
    enough that its gate may not fire.
  - **Real `verdict.verify_run`** (family W, n_boot 1000, n_boot_ks 2000) for every vs_base, curated_vs_random and
    head_to_head comparison of the dry run, with the `step4.*` tags and `vlapoc.synthetic=true` passed through its
    `extra_tags`.
  - **Real `step4.summary_run`**.
  - Every new run is tagged `vlapoc.synthetic=true`. The manifest records the planted effects in plain words and the
    resulting headline.
  - Family O is not built in the demo (the page shows W, the declared family; O is shown if present). This is a
    documented divergence.
- **Acceptance criteria:**
  - [ ] AC3.1 After `scripts/dashboard_demo.py build --with-smoke --force`, the demo store holds 7 Round A select runs,
    3 × (7 + 1) synthetic train and eval runs, 48 W verify runs (21 vs_base + 24 curated_vs_random + 3 head_to_head, the
    dry run's W comparisons) and 1 Round A summary run. Every one is tagged synthetic, `data/lineage/mlflow.db` is
    unchanged, and `results.provenance(<summary run>)` is SYNTHETIC. [brief; owner "synthetic labelled"]
  - [ ] AC3.2 Each demo `verdict.json` was produced by `compare_policies`. Re-running `compare_policies` on the recorded
    eval rollouts of one recorded pair with the recorded seed reproduces its gate p and every metric's Δ, CI and p_holm
    exactly. [brief "real verdict/step4 code"; PoC §4.4 p.10]
  - [ ] AC3.3 The recorded summary contains at least one `confirmed`, at least one `not confirmed` or `inconclusive`, and
    a head-to-head headline. Whatever the values are, they are printed into the execution log as produced, not
    asserted to a target. [demo value; vla-build honesty]
  - [ ] AC3.4 The existing demo checks still pass after the rebuild: `verify_dashboard.py --demo --app --render --dropin`.
    The added build time is reported (**inferred** ≈ 48 × 9.3 s ≈ 7.5 min, from a measured 9.3 s per verdict on
    20 cells × 50 rollouts per arm). Demo store plus cache stay under 2 GiB (AC6.5). [dashboard S6]
  - [ ] AC3.5 The synthetic outcome generator imports nothing from `groundtruth`/`recovery`/`discrepancy` (source scan).
    The planted effects are described in the run params. [PoC §2.1 item 4 p.3]
- **Verify by:** `.venv/bin/python scripts/dashboard_demo.py build --with-smoke --force`, then
  `.venv/bin/python scripts/verify_dashboard.py --demo --curation`.
- **Demo value:** a complete Round A in the demo store, every number from the real verdict code on labelled synthetic
  outcomes.

### S4: Add the Curation results page
- **Why:** the user request, §4.5 headline, §2.6, §9.1 Step 4 output, §3.4 declared targets and §2.7 surface.
- **Depends on:** S2, S3.
- **Deliverables (proposed):** `dashboard/views/curation.py`; an entry in `common.PAGES`, `app.RENDER`,
  `verify_dashboard.PAGES` and `dashboard_screenshots.PAGES`; plain names in `dashboard/words.py` (arms such as
  "5 most harmful per task for Success within 30 s removed"; comparisons "against the original policy", "against
  random removal (draw 1)", "HRT-curated vs F30-curated"). Page layout, top to bottom:
  1. Its own provenance banner from `provenance(<Round A summary run>)` (`common.banner`), plus the score-run banner the
     header already shows.
  2. **Headline card (§4.5)**: the recorded headline status and sentence, and a forest plot (Altair
     `mark_errorbar` + `mark_point`, a rule at 0) with one row per seed for "HRT-curated advantage on HRT" and
     "F30-curated advantage on F30". Both are oriented so that "wins its own metric" is positive, using
     `planted_arms.treated_advantage` with the sign flipped for the F30 row. Rows whose gate did not fire are marked.
  3. **Arm table**: one row per (arm, comparison, control) and metric, with recorded status, calls by seed, gates fired
     k/n and the sentence. A metric picker. The arm's own metric is shown first.
  4. **Per-arm detail**: a forest plot of per-seed advantage with CI for the chosen metric, against base and against each
     random draw.
  5. **Removed episodes**: count, per-task counts, table (episode, task, ranked score and rank), one selectable
     training video (reusing `common.training_video`).
  6. **Lineage expander**: MLflow links (summary, select, verify runs) and `common.trace_command` for the summary and the
     select run.
  7. **Captions**: the ranking functional (from the select run's `metric_declaration`) and the verdict functional (from
     the verdict labels). An explicit note that the HRT arm was ranked by the linearized HRT return and judged on the HRT
     ratio (§4.3 p.10). Holm across the 5 metrics within each verdict; nothing corrected across arms or seeds, and the
     page says so.
  8. **(r1, B3) Operation-level check (family O, shown alongside).** This section sits after the arm table. It holds
     one table of the recorded family-O summary rows, with arm, comparison, status, calls by seed, gates fired k/n and
     sentence (status and sentence verbatim). Its caption gives the O functional from the registry (`rmst60_op_cap`:
     "mean over operation samples of min(T_k, 60), unfinished attempt = 60", family O, τ 60 s). The caption also says
     three things:
     - the arms were ranked by a family-W return;
     - family W is the declared verdict family;
     - family O is recorded alongside, only against the original policy, and is not part of the headline.

     O rows never enter the headline card, the family-W arm table or the forest plots: the existing
     `family == "W"` filters stay. When the chain records no O rows (the demo), the section shows the caption "No
     operation-level (family O) verdicts are recorded for this chain."
- **(r1, B1) Page docstring, item 1:** replace "'inconclusive' where the macro-KS gate did not fire" with
  "'inconclusive' where the macro-KS gate fired in fewer than ⌈2n/3⌉ seeds; 'not shown' where it fired in at least
  ⌈2n/3⌉ seeds but neither curated policy's win on its own metric replicated". Add an item for the family-O section.
- **(r1) Family O decision (B3): show in its own section.** Grounding and evidence:
  - **PoC §3.4 (p.7)** (quoted): "Declared targets. Scalars can rank policies in opposite orders. Every attribution
    result states which functional it explains, following PhAIL's disclosure principle." The same table gives the
    operation level as a PoC signal (quoted): "Operation-level (T, event) pairs give several outcome samples per
    rollout".
  - **PoC §4.4 (p.10)** (quoted): "Label every claim with its metric and time cap τ, following PhAIL's disclosure
    principle." Family O is a declared functional with a τ: `vlapoc/metrics_registry.py` l.11–14 (quoted) "**family
    O** (per-operation T_k, PhAIL §3.1 pairs) is declared separately, with the attribution functional
    ``rmst60_op_cap``", and the registry entry gives `tau: "60 s"`. So a recorded O verdict is a labelled claim. The
    disclosure principle is better served by showing it under its own label than by dropping it.
  - **PoC §9.1 Step 4 (p.19)** (quoted): "Output: Confirmation, or refutation, that the attribution is causally
    correct." The recorded O rows carry exactly such a status. `step4.expected_direction` gives harm and help arms an
    expectation for every metric, O included (source l.80–100, read), and `annotate` runs over every key of the row
    (l.314–334, read).
  - **What Round A records (measured, source):**
    - `server_pipeline.py` l.420 builds each vs_base job as `--family W && --family O`, and the module docstring l.22
      reads (quoted) "family W (declared) and O (alongside)".
    - curated_vs_random (l.427) and head_to_head (l.433) are `--family W` only.
    - `metrics_registry.VERDICT_FAMILIES["O"] == ["rmst60_op_cap"]` (l.178).
    - So Round A records 21 O verify runs. `summarize` groups by family, giving **7 O summary rows** (one per arm, vs
      base, one metric each); 23 rows in all.
  - **Why not a caption only:** a caption would hide 7 recorded, status-bearing, functional-labelled results from a
    page whose purpose is to show the chain's verdicts. Showing them costs one table: the statuses and sentences already
    exist, and the only text change is AC1.7's plain name. A caption is used only when no O row is recorded.
  - **Why outside the headline:** §4.5's claim is about the arms' *own* metrics (HRT and F(30 s)), which are family-W
    functionals, and the head-to-head is recorded in W only (l.433).
- **Acceptance criteria:**
  - [ ] AC4.1 On the demo store, every status and sentence shown equals the recorded summary's text verbatim (AppTest
    compares the rendered dataframe cells and markdown with `step4_summary.json`). [PoC §9.1 Step 4 p.19; dashboard AC5.5b]
  - [ ] AC4.2 The forest plots' data (estimate, low, high per seed) equal `treated_advantage` of the recorded
    `verdict.json` metric rows, recomputed in the test. The head-to-head rows are oriented so that positive means
    "wins its own metric". [PoC §4.5 p.11]
  - [ ] AC4.3 The removed-episodes table for each arm equals `removed_episodes` (S2), and its lineage expander links the
    select run (MLflow URL with that run id) and shows a trace command that exits 0 (the existing 5.11 runner, extended
    to this page). [brief "link to the select run; with lineage"; PoC §2.1 item 1 p.3]
  - [ ] AC4.4 Declared functional reachable: every metric name shown (dataframe columns, chart titles and axes,
    selectbox) has `<name>: <functional> [τ …]` in a caption or tooltip (the existing `unreached()` scan returns `[]`),
    and the ranking functional of the selected arm is displayed. [PoC §3.4 p.7; §4.4 p.10; dashboard AC5.4]
  - [ ] AC4.5 Plain language: `snake_labels()` returns `[]` on this page outside backticked run names and ids, the banner
    is one line with reasons in an expander, and the forbidden-word scan (`harmless`, `no effect`, `no difference`) over
    the rendered page is empty. [brief; dashboard AC5.9]
  - [ ] AC4.6 Provenance: on the demo store the page shows the SYNTHETIC banner and every recorded status begins
    `SYNTHETIC:`. The banner's label equals `results.provenance(<summary run>)["label"]`. [owner "labelled everywhere"]
  - [ ] AC4.7 **(r1)** Gate honesty, for both non-claim states. A seed whose gate did not fire shows no direction
    (the call `g`, and its forest row is marked "gate did not fire").
    - (a) On a scratch copy where the head-to-head gate fired in **0/3** seeds, the card says "inconclusive" and "No
      winner is stated", and nothing on the page states a winner (no "wins its own metric").
    - (b) On the demo store as built (head-to-head gate fired 3/3, calls `C,0,0` / `0,T,0`; analysis-r1 B1, measured),
      the card says "not shown", "No winner is stated" and does **not** contain "inconclusive". Nothing on the page
      states a winner.
    - (c) Consistency, on both stores: the card's status code equals `step4.headline` recomputed from the recorded
      head-to-head row's `<m>__by_seed`. The card reads `inconclusive` if and only if that row's hrt60 and f30
      statuses are both `inconclusive`.

    [PoC §4.4 p.10; §9.1 Step 4.3–4.4 p.19; §4.5 p.11; brief; analysis-r1 B1]
  - [ ] AC4.8 A score run with no Round A chain (the smoke attribute run in the demo store) shows an info message naming
    the command that produces one (`server_pipeline.py … --from select`) and raises no exception. [vla-build robustness]
  - [ ] AC4.9 A summary run without the S1 columns (simulated by deleting them in a scratch copy) still renders. The
    statuses are computed by the same `step4` function from the recorded `<m>__by_seed`, and the page says "computed
    at display time from the recorded per-seed calls". [drop-in for a server run on an older commit]
  - [ ] AC4.10 Screenshots of the page at 1400 px are captured and looked at, with paths in the execution log.
    [dashboard AC5.10] **(r1)** Re-captured after the B1/B3 changes, so the card shows the demo's new state.
  - [ ] AC4.11 **(r1, B3)** Family O is shown, never silently dropped.
    - (a) On the demo store, which records no O rows, AppTest finds the caption "No operation-level (family O)
      verdicts are recorded for this chain." The family-W arm table still has 16 rows (unchanged).
    - (b) On the S5 drop-in store, which records the dry run's O verifies (AC5.5), the O table has exactly the
      recorded family-O summary rows (7). Each row's status and sentence equal the recorded `rmst60_op_cap__status` /
      `__sentence` verbatim. No O row appears in the W arm table, the headline card or the forest data.
    - (c) On that store, `unreached()` and `snake_labels()` return `[]`: the O metric's functional is reachable in a
      caption, and its plain name comes from AC1.7.

    Checked by `verify_dashboard.py --curation` (a) and `--dropin` (b, c). [PoC §3.4 p.7; §4.4 p.10; §9.1 Step 4
    p.19; analysis-r1 B3]
- **Verify by:** `.venv/bin/python scripts/verify_dashboard.py --curation --app`, and
  `.venv/bin/streamlit run dashboard/app.py -- --lineage-root data/lineage_demo --scores attribute-demo-ens`
  (manual look).
- **Demo value:** the stakeholder-facing headline. "Same dataset, opposite curations: <status>" with a per-seed forest
  plot, an arm table in words, and the list of removed episodes.

### S5: Drop-in proof for real Round A runs, and browser render checks on the page
- **Why:** brief "done" item 4 asks for the drop-in proof that real Round A runs render with no code change. The brief's
  constraints say the real-browser checks must cover the new page.
- **Depends on:** S4.
- **Deliverables (proposed):**
  - Extend `verify_dashboard.py --dropin`. On a scratch store with the existing fixture, run the real
    `stage.py select` for the Round A arms, then fake train runs on the child subsets. Then run fixture eval runs that
    record the real eval's columns and the checkpoint input (a new fixture helper next to `add_eval`). Then run the
    **verify and step4-summary commands parsed from the Round A dry run** (`dry_jobs` + `argv_minus`, as `round_b_mode`
    does), and finally the AppTest Curation page.
  - Extend `--render` and `dashboard_screenshots.measure` to the page.
- **Acceptance criteria:**
  - [ ] AC5.1 Through the stage CLI, the dry run's verify and step4-summary commands (only `--lineage-root` and
    `--feats-root` added) produce a summary that the Curation page renders by score-run name. `git status --porcelain --
    dashboard vlapoc scripts` is empty before and after, and no page raises. [brief drop-in; dashboard AC7.1]
  - [ ] AC5.2 With the fixture's train and eval runs re-tagged to what the server writes (`vlapoc.smoke=false`,
    `eval.attribution_grade=true`), the page shows no SYNTHETIC or SMOKE banner and shows the `Provenance: REAL` caption.
    Re-adding one synthetic tag (on one arm eval run) brings the SYNTHETIC banner back. [PoC §2.1 item 1 p.3;
    dashboard AC7.2]
  - [ ] AC5.3 In headless Chrome at 1400 px, the Curation page's Altair plot areas are each ≥ 200 px tall (at least
    2 charts), no `st.metric` value is truncated, and no exception shows. `--render` passes for all six pages. [dashboard
    AC5.10]
  - [ ] AC5.4 Full regression: `verify_dashboard.py --results --clusters --planted --discrepancy --round-b
    --planted-arms --app --render --demo --dropin --curation`, `verify_training_attribution.py --step4` and
    `verify_runbook.py` all pass. The real store sha256 stays `306af412…bde79`. Tracked artifacts that the verifiers
    rewrite only for ids and timings are restored (dashboard analysis-r3 housekeeping). [brief constraints]
  - [ ] AC5.5 **(r1, B3)** The drop-in runs the dry run's vs_base commands **as written**, i.e. both the `--family W`
    and the `--family O` parts of each `&&` job. Today it keeps only the `--family W` parts (`verify_dashboard.py`
    l.1834, read). That gives 21 more verify runs through the stage CLI.
    - The recorded summary has 23 rows: 16 W plus 7 O, each O row with `family == "O"` and comparison `vs_base`. The
      execution log prints the count and the added drop-in time.
    - The fixture eval already records `pairs` (`add_eval_outcomes`, `verify_training_attribution.py` l.244, read).
      The family-O human anchor covers the same 20 cells as the W anchor (**measured** in this revision:
      `human_operations.parquet` has 20 `(suite, libero_task_id)` cells, a superset of `human_reference.parquet`'s 20).
      So no new fixture data is needed.
    - The existing 5.1 comparison (W rows' f30 status and sentence equal recorded) still passes.

    [brief drop-in "no code change"; PoC §9.1 Step 4 p.19; analysis-r1 B3]
- **Verify by:** `.venv/bin/python scripts/verify_dashboard.py --dropin --render`, then the full regression above.
- **Demo value:** the claim "the day Round A finishes, this page shows it unchanged, labelled REAL", backed by a passing
  check.

## Out of scope (deliberately)

- **The four other gaps the brief lists** are not requested. They go to the owner, none BLOCKING:
  1. Recovery of the planted episodes by the ranking (`score` stage, per-type AUROC/recall next to the best heuristic).
  2. A "planted type" column on the Episodes and Rollouts lists. This would also allow "of the 100 episodes the
     harm-F30 arm removed, N were planted" on this page (answer key read after attribution, under a "ground-truth check"
     heading like the other pages).
  3. Kendall's τ metric agreement (§4.4 p.11) and per-policy KM time-to-success curves.
  4. Percentile-bootstrap cluster intervals (`vlapoc/clusters.py`).

  The headline makes sense without them. Its claim is about retrained policies on their own metrics (§4.5), not about
  planted data. Gap 2 is the natural next add-on for this page.
- **Global curation arms** (`gharm-*`, `--curation global`) are not run in Round A. The page groups by the recorded
  `curation` column, so they appear in their own section if they are ever recorded. The demo does not build them.
- **Family O verdicts** in the demo (S3) are not built: the demo's synthetic rollouts carry no `pairs` column
  (`grep pairs scripts/dashboard_demo.py`, measured: no rollout writer). **(r1)** The page shows recorded O verdicts in
  their own section (AC4.11). It is exercised end to end by the drop-in (AC5.5), and on the demo it shows the visible
  "none recorded" caption. Building O into the demo would add 21 verdicts to the build for no new page behaviour.
- **Any change to server jobs, counts or the runbook.** None is needed (measured above).
- **Multiplicity correction across arms and seeds.** The across-seed rule is a replication rule, not a test. The page
  states this rather than adding a correction the PoC does not ask for (§4.4 asks for correction "across metrics").

## Open questions

- BLOCKING: none.
- Non-blocking:
  1. The **across-seed rule** (≥ ⌈2n/3⌉ seeds and none contradicting) is reused from `vlapoc.confirm`, where it is labelled
     inferred. The **headline rule** (shown / partly shown / contradicted / **not shown** / inconclusive, r1) is new
     and inferred. Both are recorded as params, and the owner may override either.
  2. **"inconclusive" vs "not confirmed" / "not shown"**: a metric row is inconclusive when the gate fired in fewer than
     ⌈2n/3⌉ seeds, and not confirmed when the gate fired but the direction did not replicate. **(r1)** The headline
     follows the same split: `inconclusive` when the gate fired in fewer than ⌈2n/3⌉ seeds, `not shown` otherwise
     when neither claim is confirmed or refuted. This is my reading of the brief's "'inconclusive' where the gate did
     not fire", now applied consistently to the card and the table.
  5. **(r1) Family O on the page** is shown alongside in its own section (S4 item 8), not in the headline. If the owner
     prefers the page to show W only, AC4.11 becomes "caption only" with no other change.
  3. **`headline` param**: with `--curation per_task` it is False for every arm (measured). The page treats `per_task`
     arms as the headline (owner Q1: "per-task headline") rather than reading that param.
  4. **The Round A summary for a chain sharing its SHA with a later Round B**: `ch.chain = <tag>-<sha8>`, so a Round B
     submitted at the same commit shares the chain id, and its step4-summary would include Round A verifies (**inferred**
     from `server_pipeline.py` l.356, 264). The page picks the newest summary that has curation-op rows and filters rows
     by op, so either summary renders the same Round A rows.

## Risks / premises to verify during execution

- **Provenance cost.** `results.provenance(<summary>)` walks every verify, eval, train, select and upstream run (69
  verify runs on the server). Measure the page's first-load time on the demo store and report it. If it is over about
  10 s, cache by run id (it is already cached with `st.cache_data`). **inferred** risk.
- **The fixture eval for S5 must match the real eval's columns** (`T_s`, `event`, `tau_episode_steps`, `suite`,
  `libero_task_id`, `rollout_id`, `pairs` for family O). Read them from a real eval record (`returns.load_eval_rollouts`
  on `eval-smolvla-harness-spatial0`) before writing the helper. Do not use memory.
- **`verify_run` uses `stage_run` without `allow_dirty`.** The demo build already requires a clean tree. S5's fixture
  goes through the stage CLI, which needs a clean tree too (as in AC7.1).
- **Human anchor cells.** The verdict only compares cells present in `data/eval/human_reference/human_reference.parquet`.
  **Measured:** the 20 v3/candidate task cells are all among the 20 human-anchor cells. Synthetic rollouts must use those
  cells.
- **Demo regression.** Rebuilding the demo changes run ids. Existing checks that pin ids or artifact files must still
  pass (AC3.4), and rewritten tracked artifacts are restored if only ids and timings changed.
- **(r1) Drop-in readers assume W columns.**
  - Code that loops over all `summary_json["rows"]` and reads `f30__status`, such as the 5.1 `want` set in
    `verify_dashboard.py`, will raise `KeyError` once O rows exist. Filter by `family` there.
  - The page's existing `family == "W"` filters are what keep O out of the headline and W table. Keep them, and add the
    O section separately (inferred from the source read; confirm when running AC5.5).
- **(r1) Family-O verdict on the fixture.** `compare_policies(..., family="O")` has not been run on the drop-in fixture
  before. The fixture writes one pair per rollout (`[T or τ, event]`). Confirm before relying on it that the O verify
  completes and its `rmst60_op_cap` row has `p_holm` and `diff_better`. If it fails, report it; do not drop O from the
  drop-in.
- **Scratch files** go under `data/scratch/` only. `data.zip` is never touched. The real store sha must stay
  `306af4123fda7e635892d459a2e34008ba1cb62f6c4fce152c546e0da7bdde79`.
