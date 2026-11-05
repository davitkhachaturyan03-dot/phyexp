"""Step 4 summary: one table over the verify runs of a chain, with a seed-consistency count per comparison.

Where this sits (docs/dataflow.md, Step 4 "Verification"): last stage of the chain. It reads the ``verify`` runs that
``scripts/stage.py verify`` wrote for each retrained arm and summarises them in one ``summarize`` run; it trains and evaluates nothing.
Reading order: ``collect`` (find the verdicts), ``call`` (T/C/0/g per metric and seed), ``summarize`` (one row per arm and comparison),
``summary_run`` (record it).

Grounding
---------
- PoC §9.1 Step 4 (p. 19), quoted: "1 Remove the top-k most harmful episodes and retrain. As a control, remove k random
  episodes and retrain. 2 Evaluate both policies as in Step 2. 3 Apply the macro-KS test ... 4 If it did, check each
  metric's confidence interval for the direction of the change. 5 Repeat for the top-k most helpful episodes (removing
  them should make things worse)." §2.6 (p. 5): "Remove the top-k 'harmful' episodes and retrain: the policy should
  improve more than after removing k random episodes. Remove the top-k 'helpful' episodes and retrain: the policy
  should degrade more than after random removal."
- PoC §4.5 (p. 11), "Same dataset, opposite curations": curate twice, once for HRT and once for reliability (F(30 s)),
  "retrain on each subset, and show that each curated policy wins on its own metric". The ``head_to_head`` comparison
  is that test.
- PoC §2.3 (p. 4): "2–3 seeds per data configuration". Each (arm, comparison) is verified once per training seed;
  this module counts, per metric, in how many seeds the Holm-corrected verdict went each way.
- PoC §3.4 (p. 7) and §4.4 (p. 10-11): every result carries the functional it explains. Each row carries the declared
  functional (column, metric, τ, rollout subset) of the select run that built the treated arm.

What it reads
-------------
``stage.py verify`` tags every Step 4 verify run ``step4.kind`` (``vs_base`` | ``curated_vs_random`` |
``head_to_head``), ``step4.treated`` (arm B, bare arm name such as ``harm-f30-k5``), ``step4.control`` (arm A; ``base`` for the original
policy) and ``step4.seed``, plus the chain tag ``vlapoc.chain``. The run's ``verdict.json`` (``vlapoc.verdict.compare_policies``)
gives, per family-W metric, ``p_holm`` and ``diff_better`` (> 0: arm A, the control, is better). The treated arm's select
run (``select-<tag>-<arm>``) gives its operation, direction, curation mode and declared functional.

Per metric and seed the call is ``T`` (gate fired, Holm p < α and the treated arm is better), ``C`` (gate fired, Holm p < α
and the control is better), ``0`` (gate fired, not significant) or ``g`` (the macro-KS gate did not fire: no directional
support, PoC §4.4 p. 10, §9.1 Step 4.4 p. 19; training-attribution round 2). The two-step verdict (macro-KS gate, then per-metric CIs) is PhAIL §3.3 as
``vlapoc.verdict`` implements it; this module aggregates, it does not re-test. It refuses verdicts from several chains
under one tag unless a chain id is given, and a verify run whose chain differs from the requested one.

Status, sentence and the §4.5 headline (curation-results S1; the rule is **inferred**, stated before any result, recorded as a run param
and open to the owner's override)
-------------------------------------------------------------------------------------------------------------------------------------
``annotate`` adds, per metric ``<m>``, ``<m>__status_code`` / ``<m>__status`` / ``<m>__sentence`` to a summary row and ``headline`` reads the
PoC §4.5 (p. 11) "same dataset, opposite curations" claim from the ``head_to_head`` rows. Both are pure functions of the recorded per-seed
calls (``<m>__by_seed``), so the dashboard can compute the same text from an older summary run that lacks the columns.

- **Per metric with an expectation** (PoC §9.1 Step 4.3-4.5 p. 19, §2.6 p. 5; across seeds the replication rule of ``vlapoc.confirm``, ceil(2n/3) seeds,
  none contradicting, PoC §2.3 p. 4 "2-3 seeds per data configuration"): ``confirmed`` / ``refuted`` / ``not confirmed`` from ``confirm.status_of``;
  ``inconclusive`` instead of ``not confirmed`` when the macro-KS gate fired in fewer than ceil(2n/3) seeds (the two policies were not shown to
  differ, PoC §4.4 (p. 10): direction is claimed only after the gate fires). A metric with no expectation (a random-removal control arm) reads
  ``no expectation (control arm)``.
- **Headline per head-to-head pair** (PoC §4.5 p. 11: "show that each curated policy wins on its own metric"): speed claim = the HRT-curated arm is better on
  ``hrt60`` (``status_of(..., "T")``), reliability claim = the F(30)-curated arm is better on ``f30`` (``status_of(..., "C")``). Both claims come from the same
  verdicts and share one gate count g (seeds whose macro-KS gate fired). In this order: ``contradicted`` if either claim is refuted; ``shown`` if both are
  confirmed; ``partly shown`` if exactly one is confirmed (none refuted); ``inconclusive`` if g < ceil(2n/3) (the two policies were not shown to differ, PoC §4.4
  (p. 10): direction only after the gate fires); ``not shown`` otherwise (g >= ceil(2n/3), the policies differ, but neither claim replicated). The first three
  cannot happen when g < ceil(2n/3), so the order is unambiguous (the gate 2/3 boundary is a fixture case).
- **Wording** (PoC §4.4 (p. 10) "Label every claim with its metric and time cap τ"): every sentence names the metric's plain name and its τ; a result that is
  not significant reads "not shown at this sample size" and says that is not evidence for anything else. The words "harmless", "no effect" and
  "no difference" are never used. With a synthetic input (a verify or select run tagged ``vlapoc.synthetic=true``, or a verdict carrying
  ``"SYNTHETIC"``) every status and sentence begins ``SYNTHETIC: `` (fail-closed, no flag). ``<m>__status_code`` keeps the canonical word.
- The HRT arm is *ranked* by the linearized return ``hrt60_lin`` and *judged* on the HRT ratio ``hrt60`` (PoC §4.3 p. 10: "HRT is a ratio, so it needs
  a first-order (linearized) approximation"); the sentence says so for that metric.

Expected direction (stated before any result exists): removing the most harmful episodes should leave the treated policy
better than its control (``treated_better``); removing the most helpful should leave it worse (``treated_worse``);
for the HRT-vs-reliability head-to-head the HRT-curated arm should win on ``hrt60`` and the F(30)-curated arm on ``f30``.
Where no expectation applies the cell reads ``-``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

KINDS = ("vs_base", "curated_vs_random", "head_to_head")
ALPHA = 0.05  # the verdict's own alpha (compare_policies default); recorded per verdict as v["alpha"]


def expected_direction(kind: str, arm_direction: str | None, metric: str, treated: str, control: str,
                       expected_by_metric: dict | None = None) -> str | None:
    """The declared expectation for one metric (module docstring): ``treated_better``, ``treated_worse`` or ``None``.

    ``expected_by_metric`` (dashboard S4/S10): when the treated arm's select run recorded one (a ``drop_cluster`` arm: the estimated sign per
    metric whose interval excludes zero; a planted-type arm whose label names metrics), the expectation is exactly that mapping and every
    metric it does not name reads ``None``, for ``vs_base`` and ``curated_vs_random`` alike."""
    if expected_by_metric is not None and kind != "head_to_head":
        return expected_by_metric.get(metric)
    if kind == "head_to_head":
        if metric == "hrt60" and "hrt60" in treated:
            return "treated_better"
        if metric == "f30" and "f30" in control:
            return "treated_worse"  # the control (F(30)-curated) arm should win its own metric
        return None
    if arm_direction == "highest":
        return "treated_better"  # harmful removed
    if arm_direction == "lowest":
        return "treated_worse"  # helpful removed
    return None


def call(metric_row: dict, alpha: float = ALPHA, gate_fires: bool = True) -> str:
    """``T`` / ``C`` / ``0`` / ``g`` for one verdict metric row (``diff_better`` > 0 means arm A, the control, is better).

    The two-step verdict is gated: PoC §4.4 (p. 10) and §9.1 Step 4.3-4.4 (p. 19), "Apply the macro-KS test ... If it did
    [fire], check each metric's confidence interval for the direction of the change". A seed whose macro-KS gate did not fire
    (``gate_fires`` False) gets ``g`` whatever its per-metric Holm p-values say: it contributes no directional support in
    either direction (consistent with ``vlapoc.verdict.outcome``, which reports outcome "iii" and no direction then)."""
    if not gate_fires:
        return "g"
    if metric_row["p_holm"] >= alpha or metric_row["diff_better"] == 0:
        return "0"
    return "C" if metric_row["diff_better"] > 0 else "T"


# ---------------------------------------------------------------------------------------------
# status, sentence and the §4.5 headline (curation-results S1)
# ---------------------------------------------------------------------------------------------

RULE = ("per metric with an expectation: confirmed = expectation met (significant call in the expected direction, macro-KS gate fired) in >= ceil(2n/3) of n seeds and "
        "contradicted in none; refuted = the opposite significant call in >= ceil(2n/3) seeds; inconclusive = neither, and the gate fired in fewer than ceil(2n/3) seeds; "
        "otherwise not confirmed. Same-dataset-opposite-curations headline: shown = speed claim (HRT-curated arm better on the HRT ratio) and reliability claim "
        "(F(30)-curated arm better on F(30)) both confirmed; contradicted = either refuted; partly shown = exactly one confirmed and none refuted; inconclusive = none of those and "
        "the gate fired in fewer than ceil(2n/3) seeds; not shown = none of those and the gate fired in at least ceil(2n/3) seeds but neither claim replicated "
        "(inferred; stated before any result; the across-seed rule is vlapoc.confirm's)")
STATUS_NONE = "no expectation (control arm)"
HEADLINE_PAIR = ("hrt60", "f30")  # PoC §4.5 p. 11: the speed metric (HRT-curated arm should win it) and the reliability metric (F(30)-curated arm should win it)
_CALL_WORDS = {"T": "the treated policy was better", "C": "the control policy was better", "0": "not shown at this sample size", "g": "inconclusive (gate did not fire)"}


def _name_of(key: str) -> str:
    """Plain name of a verdict metric key (``f30`` -> ``Success within 30 s``), from ``metrics_registry.SHORT_NAMES``."""
    from vlapoc.metrics_registry import SHORT_NAMES, entry

    base = {"hrt60": "hrt60_lin", "auc_h": "auc_expected"}.get(key, key)
    return SHORT_NAMES.get(base) or entry(key)["metric"]  # family O keys (``rmst60_op_cap``) take the registry's plain ``metric`` name


def metric_phrase(key: str) -> str:
    """``Success within 30 s (τ 30 s)``: the plain name and the time cap of a verdict metric, so no sentence carries a bare number (PoC §4.4 p. 10)."""
    from vlapoc.metrics_registry import entry

    tau = str(entry(key)["tau"])
    return f"{_name_of(key)} ({'no τ cap' if tau.startswith('none') else 'τ ' + tau.split(' =')[0]})"


def status_of_calls(calls: list[str], expected: str | None) -> tuple[str, int, int]:
    """``(status code, seeds as expected, seeds contradicting)`` of one metric's per-seed calls against its expectation (module docstring rule)."""
    from vlapoc.confirm import status_of

    want = {"treated_better": "T", "treated_worse": "C"}.get(expected or "")
    if want is None:
        return STATUS_NONE, 0, 0
    status, met, contra = status_of(calls, want)
    need = math.ceil(2 * len(calls) / 3) if calls else 1
    if status == "not confirmed" and len(calls) - calls.count("g") < need:
        status = "inconclusive"
    return status, met, contra


def _removed(row: dict) -> str:
    """What the treated arm removed, in words (``the 5 episodes per task ranked most harmful for Speed vs human (HRT)``)."""
    import re

    k = row.get("k")
    per_task = str(row.get("op", "")).endswith("_per_task")
    n = f"{int(float(k))} episodes{' per task' if per_task else ''}" if k not in (None, "", "nan") and str(k) != "nan" else "the top-ranked episodes"
    if str(row.get("op", "")).startswith("drop_random"):
        return f"{n} chosen at random"
    base = re.sub(r"^harm_|__same_task$", "", str(row.get("metric_column", "")))
    what = _name_of(base) if base else "the ranked metric"
    return f"the {n} ranked " + ("most harmful" if row.get("direction") == "highest" else "most helpful") + f" for {what}"


def _control_words(row: dict) -> str:
    """The control of a comparison in words."""
    c, kind = str(row.get("control", "")), row.get("comparison")
    if kind == "vs_base":
        return "the original policy"
    if kind == "head_to_head":
        return f"the policy curated the other way ({c})"
    return f"removing the same number of random episodes per task ({c})"


def sentence_of(row: dict, m: str, status: str, met: int, contra: int, calls: list[str]) -> str:
    """The bounded sentence of one metric of one summary row (module docstring: names the metric and τ; "not shown at this sample size" for a result that is not significant)."""
    n = len(calls)
    gates = n - calls.count("g")
    mp, rem, ctl = metric_phrase(m), _removed(row), _control_words(row)
    rpt = row.get("rollouts_per_task")
    size = f"{n} seeds" + (f" x {int(float(rpt))} rollouts per task" if rpt not in (None, "") and str(rpt) != "nan" else "")
    gate = f"macro-KS gate fired in {gates} of {n}"
    exp = row.get(f"{m}__expected")
    tail = (" The arm was ranked by the linearized HRT return and judged on the HRT ratio (PoC §4.3)." if m == "hrt60" and "hrt60_lin" in str(row.get("metric_column", "")) else "")
    if status == STATUS_NONE:
        return f"No expectation for this arm; reported for reference. {mp}: calls by seed {','.join(calls)} ({gate}).{tail}"
    better = exp == "treated_better"
    if row.get("comparison") == "head_to_head":
        return f"{mp}: {met} of {n} seeds went as the PoC expects, {contra} the other way ({gate}): {status}.{tail}"
    verb = "improved" if better else "made worse"
    if status == "confirmed":
        return f"Removing {rem} {verb} {mp} compared with {ctl} in {met} of {n} seeds ({gate}): confirmed.{tail}"
    if status == "refuted":
        return f"Removing {rem} made {mp} {'worse instead of better' if better else 'better instead of worse'} compared with {ctl} in {contra} of {n} seeds; {met} as expected ({gate}): refuted.{tail}"
    if status == "inconclusive":
        return f"Inconclusive: the macro-KS gate (did the time-to-success distribution change at all?) fired in only {gates} of {n} seeds, so no direction is claimed for {mp}.{tail}"
    return (f"Not shown at this sample size ({size}): removing {rem} moved {mp} as expected compared with {ctl} in {met} of {n} seeds ({gate}). "
            f"A change that is not detected is not evidence that removal leaves the policy unchanged: not confirmed.{tail}")


def annotate(row: dict, keys: list[str], synthetic: bool = False) -> dict:
    """The new columns of one summary row: per metric ``<m>__status_code`` (canonical word), ``<m>__status`` (display: ``SYNTHETIC: `` first when
    ``synthetic``) and ``<m>__sentence``. Needs only ``<m>__by_seed``, ``<m>__expected`` and the row's identifiers, so the dashboard uses the same
    function on a summary run that predates these columns."""
    pre = "SYNTHETIC: " if synthetic else ""
    out = {}
    for m in keys:
        calls = [c for c in str(row[f"{m}__by_seed"]).split(",") if c]
        exp = row.get(f"{m}__expected")
        code, met, contra = status_of_calls(calls, None if exp in (None, "-") else exp)
        out[f"{m}__status_code"] = code
        out[f"{m}__status"] = pre + code
        out[f"{m}__sentence"] = pre + sentence_of(row, m, code, met, contra, calls)
    return out


def headline(rows: list[dict], synthetic: bool = False) -> list[dict]:
    """PoC §4.5 (p. 11) "same dataset, opposite curations", one record per ``head_to_head`` family-W row (treated = HRT-curated arm, control = F(30)-curated arm):
    the speed and reliability claims (each with its status, seeds as expected, seeds contradicting, gates fired), the per-seed readings, the headline status
    (module docstring) and its sentence. Needs the rows' ``<m>__by_seed`` only."""
    speed_m, rel_m = HEADLINE_PAIR
    pre = "SYNTHETIC: " if synthetic else ""
    out = []
    for row in rows:
        if row.get("comparison") != "head_to_head" or row.get("family", "W") != "W" or f"{speed_m}__by_seed" not in row or f"{rel_m}__by_seed" not in row:
            continue
        sc = [c for c in str(row[f"{speed_m}__by_seed"]).split(",") if c]
        rc = [c for c in str(row[f"{rel_m}__by_seed"]).split(",") if c]
        n = len(sc)
        s_status, s_met, s_con = status_of_calls(sc, "treated_better")
        r_status, r_met, r_con = status_of_calls(rc, "treated_worse")
        gates = n - sc.count("g")
        if "refuted" in (s_status, r_status):
            code = "contradicted"
        elif s_status == "confirmed" and r_status == "confirmed":
            code = "shown"
        elif "confirmed" in (s_status, r_status):
            code = "partly shown"
        elif gates < math.ceil(2 * n / 3):
            code = "inconclusive"
        else:
            code = "not shown"
        seeds = [s for s in str(row.get("seeds", "")).split(",") if s != ""]
        per_seed = []
        for i, (a, b) in enumerate(zip(sc, rc)):
            per_seed.append({"seed": seeds[i] if i < len(seeds) else str(i), "speed_call": a, "reliability_call": b,
                             "speed": {"T": "HRT-curated policy faster", "C": "F(30)-curated policy faster", "0": "not shown at this sample size",
                                       "g": "inconclusive (gate did not fire)"}[a],
                             "reliability": {"C": "F(30)-curated policy more reliable", "T": "HRT-curated policy more reliable", "0": "not shown at this sample size",
                                             "g": "inconclusive (gate did not fire)"}[b]})
        sp, rp = metric_phrase(speed_m), metric_phrase(rel_m)
        if code == "shown":
            text = (f"Same dataset, opposite curations: shown. The policy trained without the episodes ranked most harmful for speed was faster on {sp} in {s_met} of {n} seeds, "
                    f"and the policy trained without those ranked most harmful for reliability was better on {rp} in {r_met} of {n} seeds (macro-KS gate fired in {gates} of {n}): "
                    "each curated policy wins on its own metric.")
        elif code == "contradicted":
            text = (f"Same dataset, opposite curations: contradicted. At least one curated policy lost on its own metric in at least {math.ceil(2 * n / 3)} of {n} seeds "
                    f"(speed claim on {sp}: {s_status}; reliability claim on {rp}: {r_status}).")
        elif code == "partly shown":
            text = (f"Same dataset, opposite curations: partly shown. Speed claim on {sp}: {s_status} ({s_met} of {n} seeds); reliability claim on {rp}: {r_status} ({r_met} of {n} seeds). "
                    "Only one of the two curated policies is shown to win its own metric.")
        elif code == "not shown":
            rr = row.get("rollouts_per_task")
            size = f"{n} seeds" + (f" x {int(float(rr))} rollouts per task" if rr not in (None, "") and str(rr) != "nan" else "")
            text = (f"Same dataset, opposite curations: not shown at this sample size ({size}). The two curated policies' time-to-success distributions differ (macro-KS gate fired in {gates} of {n} seeds), "
                    f"but neither curated policy's win on its own metric replicated across seeds (speed claim on {sp}: {s_status}, {s_met} of {n}; reliability claim on {rp}: {r_status}, {r_met} of {n}). "
                    "A change that is not detected is not evidence that the curations leave the policies unchanged. No winner is stated.")
        else:
            text = (f"Same dataset, opposite curations: inconclusive. The macro-KS gate fired in only {gates} of {n} seeds, so no direction is claimed for either curated policy. No winner is stated.")
        out.append({"treated": row["arm"], "control": row["control"], "family": row.get("family", "W"), "n_seeds": n, "seeds": seeds, "gate_fires": row.get("gate_fires"),
                    "status_code": code, "status": pre + code, "sentence": pre + text,
                    "speed": {"metric": speed_m, "status": s_status, "met": s_met, "contradicted": s_con, "calls": sc},
                    "reliability": {"metric": rel_m, "status": r_status, "met": r_met, "contradicted": r_con, "calls": rc},
                    "per_seed": per_seed, "rule": RULE})
    return out


def treated_advantage(row: dict) -> tuple[float, float, float]:
    """``(advantage, lo, hi)`` of the treated arm B over the control A in "better" units, from one verdict metric row (what the Curation results page plots).
    ``diff`` and ``ci95`` are A minus B in raw units and ``diff_better`` is sign-corrected (> 0: A, the control, is better), so B's advantage is ``-diff_better`` and its
    interval ``-sgn * ci95`` with ``sgn = +1`` for a higher-is-better metric. This is the only copy: ``vlapoc.planted_arms`` (ground-truth side) imports it from here, so the method-side
    dashboard page never has to import the ground-truth module."""
    sgn = 1.0 if row.get("higher_is_better", True) else -1.0
    if "higher_is_better" not in row and row.get("diff"):
        sgn = math.copysign(1.0, row["diff_better"] / row["diff"]) if row["diff_better"] else 1.0
    lo, hi = sorted(-sgn * c for c in row["ci95"])
    return -float(row["diff_better"]), float(lo), float(hi)


def summarize(records: list[dict], arms: dict[str, dict]) -> pd.DataFrame:
    """The Step 4 table: one row per (treated arm, comparison kind, control, family).

    ``records``: one dict per verify run: ``kind``, ``treated``, ``control``, ``seed``, ``verdict`` (a ``verdict.json``).
    ``arms``: ``{arm name: {"op", "direction", "metric", "declaration"}}`` from the select runs (missing arms get blanks).
    Columns: identifiers; ``n_seeds``, ``seeds``, ``gate_fires`` (seeds whose macro-KS gate fired), ``outcome_by_seed``;
    per metric ``<m>__by_seed`` (T/C/0 in seed order), ``<m>__T``, ``<m>__C``, ``<m>__expected``, ``<m>__supports``
    (seeds whose significant call matches the expectation), ``<m>__consistency`` (text); and the declared functional.
    """
    df = pd.DataFrame([{"kind": r["kind"], "treated": r["treated"], "control": r["control"], "seed": int(r["seed"]),
                        "family": r["verdict"]["family"], "verdict": r["verdict"], "synthetic": bool(r.get("synthetic"))} for r in records])
    rows = []
    for (treated, kind, control, family), g in df.groupby(["treated", "kind", "control", "family"], sort=True):
        g = g.sort_values("seed")
        arm = arms.get(treated, {})
        decl = arm.get("declaration") or {}
        row = {"arm": treated, "comparison": kind, "control": control, "family": family,
               "op": arm.get("op", ""), "direction": arm.get("direction", ""), "metric_column": arm.get("metric", ""),
               "curation": arm.get("curation", ""), "n_seeds": len(g), "seeds": ",".join(str(s) for s in g["seed"]),
               "gate_fires": f"{sum(v['gate']['fires'] for v in g['verdict'])}/{len(g)}",
               "outcome_by_seed": ",".join(v["outcome"] for v in g["verdict"]),
               "functional": decl.get("functional", ""), "tau": decl.get("tau", ""),
               "rollout_subset": decl.get("rollout_subset", ""), "good_sign": decl.get("good_sign", "")}
        keys = list(g["verdict"].iloc[0]["metrics"])
        for m in keys:
            calls = [call(v["metrics"][m], v.get("alpha", ALPHA), bool(v["gate"]["fires"])) for v in g["verdict"]]
            exp = expected_direction(kind, arm.get("direction"), m, treated, control, arm.get("expected_by_metric"))
            want = {"treated_better": "T", "treated_worse": "C"}.get(exp)
            n_t, n_c = calls.count("T"), calls.count("C")
            supports = sum(c == want for c in calls) if want else None
            row[f"{m}__by_seed"] = ",".join(calls)
            row[f"{m}__T"], row[f"{m}__C"] = n_t, n_c
            row[f"{m}__expected"] = exp or "-"
            row[f"{m}__supports"] = supports if supports is not None else "-"
            row[f"{m}__consistency"] = (f"treated better in {n_t}/{len(g)} seeds, control better in {n_c}/{len(g)}, gate not fired in {calls.count('g')}/{len(g)}"
                                        + (f"; expectation ({exp}) met in {supports}/{len(g)}" if want else ""))
        # curation-results S1: new columns come last, so the columns above are unchanged
        synthetic = bool(arm.get("synthetic")) or bool(g["synthetic"].any()) or any(bool(v.get("SYNTHETIC")) for v in g["verdict"])
        v0 = g["verdict"].iloc[0]
        sizes = v0.get("n_episodes") or {}
        row["k"] = arm.get("k", "")
        row["rollouts_per_task"] = round(min(sizes.values()) / len(v0["cells"])) if sizes and v0.get("cells") else ""
        row["synthetic"] = synthetic
        row.update(annotate(row, keys, synthetic))
        rows.append(row)
    return pd.DataFrame(rows)


def collect(tag: str, chain: str | None = None, lineage_root: Path | None = None) -> tuple[list[dict], dict[str, dict], str]:
    """Read the Step 4 verify runs of one chain: ``(records, arms, chain id)``.

    Refuses (``ValueError``) when the tag matches several chains and no ``chain`` is given, when nothing matches, and
    when a run's chain is not the requested one.
    """
    from vlapoc.attribution import _artifact_json
    from vlapoc.runs import client, experiment_id

    c = client(lineage_root)
    runs = [r for r in c.search_runs([experiment_id(lineage_root)], "tags.stage = 'verify' and attributes.status = 'FINISHED'",
                                     max_results=10000)
            if "step4.kind" in r.data.tags and r.data.tags.get("vlapoc.chain", "").startswith(f"{tag}-")]
    chains = sorted({r.data.tags["vlapoc.chain"] for r in runs})
    if chain is None:
        if len(chains) > 1:
            raise ValueError(f"tag {tag!r} matches verify runs of several chains {chains}; pass the chain id to summarize one")
        chain = chains[0] if chains else None
    if chain is None:
        raise ValueError(f"no FINISHED step4 verify runs under tag {tag!r}")
    runs = [r for r in runs if r.data.tags["vlapoc.chain"] == chain]  # only the requested chain's verdicts are summarized
    if not runs:
        raise ValueError(f"chain {chain!r} has no FINISHED step4 verify runs under tag {tag!r}")
    records = [{"kind": r.data.tags["step4.kind"], "treated": r.data.tags["step4.treated"], "control": r.data.tags["step4.control"],
                "seed": r.data.tags["step4.seed"], "run_id": r.info.run_id, "name": r.info.run_name,
                "verdict_sha256": r.data.tags["output.verdict.sha256"], "synthetic": r.data.tags.get("vlapoc.synthetic") == "true",
                "verdict": _artifact_json(r, "verdict", lineage_root)} for r in runs]
    arms = {}
    for treated in sorted({x["treated"] for x in records}):
        sel = c.search_runs([experiment_id(lineage_root)], f"tags.mlflow.runName = 'select-{tag}-{treated}' "
                            "and tags.stage = 'select' and attributes.status = 'FINISHED'", max_results=5)
        sel = [s for s in sel if s.data.tags.get("vlapoc.chain") == chain]
        if sel:
            p = sel[0].data.params
            decl = p.get("metric_declaration")
            ebm = p.get("expected_by_metric")
            arms[treated] = {"expected_by_metric": json.loads(ebm) if ebm and ebm.startswith("{") else None, "groundtruth_arm": p.get("groundtruth_arm") == "True", "cluster_run_id": p.get("cluster_run_id"), "cluster_id": p.get("cluster_id"),
                             "definition": p.get("definition"), "arm_kind": p.get("arm_kind"), "corruption_type": p.get("corruption_type"),
                             "n_dropped": p.get("n_dropped"), "child_n_episodes": p.get("child_n_episodes"), "k": p.get("k", ""), "synthetic": sel[0].data.tags.get("vlapoc.synthetic") == "true", "op": p.get("op", ""), "direction": p.get("direction", ""), "metric": p.get("metric", ""),
                             "curation": p.get("curation", ""),
                             "declaration": json.loads(decl) if decl and decl.startswith("{") else None}
    return records, arms, chain


def summary_run(tag: str, name: str, chain: str | None = None, out_dir: Path | None = None, lineage_root: Path | None = None,
                allow_dirty: bool = False, run_chain: str | None = None) -> dict:
    """Stage ``summarize``: write ``step4_summary.csv`` / ``.json`` and record them as one MLflow run (PoC §9.1 Step 4
    p. 19). Returns ``{"run_id", "table", "rows", "n_verify_runs", "chain"}``."""
    from vlapoc.lineage import rel
    from vlapoc.runs import stage_run

    records, arms, chain_id = collect(tag, chain, lineage_root)
    table = summarize(records, arms)
    synthetic = bool(table["synthetic"].any())
    head = headline(table.to_dict("records"), synthetic)
    out_dir = Path(out_dir) if out_dir else Path("data") / "attribution" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv, js = out_dir / "step4_summary.csv", out_dir / "step4_summary.json"
    table.to_csv(csv, index=False)
    js.write_text(json.dumps({"tag": tag, "chain": chain_id, "alpha": ALPHA, "n_verify_runs": len(records), "synthetic": synthetic, "rule": RULE, "headline": head,
                              "verify_run_ids": sorted(r["run_id"] for r in records), "rows": table.to_dict("records")}, indent=1, default=str))
    with stage_run("summarize", name, lineage_root, chain=run_chain or chain_id, allow_dirty=allow_dirty,
                   extra_tags={"vlapoc.synthetic": "true"} if synthetic else None) as r:
        for rec in records:
            r.input_entity("verdict", rec["verdict_sha256"], f"verify.{rec['name']}"[:200])
        r.params({"tag": tag, "source_chain": chain_id, "n_verify_runs": len(records), "alpha": ALPHA, "rows": len(table),
                  "kinds": sorted({x["kind"] for x in records}), "rule": RULE, "synthetic": synthetic,
                  "headline": {h["treated"] + " vs " + h["control"]: h["status"] for h in head}})
        r.output_file("step4_summary_csv", csv, kind="table")
        r.output_file("step4_summary_json", js, kind="table")
        return {"run_id": r.run_id, "table": rel(csv), "rows": len(table), "n_verify_runs": len(records), "chain": chain_id, "headline": {h["treated"] + " vs " + h["control"]: h["status"] for h in head}}


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m vlapoc.step4 --tag <t> --name <run name> [--chain <id>]``."""
    p = argparse.ArgumentParser(prog="python -m vlapoc.step4")
    p.add_argument("--tag", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--chain", default=None, help="full chain id (<tag>-<sha8>) when the tag matches several")
    p.add_argument("--lineage-root", type=Path, default=None)
    a = p.parse_args(argv)
    print(json.dumps(summary_run(a.tag, a.name, a.chain, lineage_root=a.lineage_root), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
