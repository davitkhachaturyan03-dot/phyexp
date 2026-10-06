"""The verification gate: PhAIL's two-step verdict (macro-KS significance, then per-metric CIs for direction).

Grounding
---------
- PoC §4.4 (p. 10): "Two-step verdict per intervention. First the macro-KS gate: did removing or adding this
  data change the time-to-success distribution at all? Then each metric's bootstrap CI gives the direction
  for that metric." "Crossing curves mean trade-offs. When the gate fires but metrics disagree in sign, the
  CDFs cross". "Multiple comparisons. Four metrics multiply the number of claims; correct for it (e.g.
  Bonferroni or Holm across metrics) before labeling an episode as a trade-off." "Label every claim with its
  metric and time cap τ".
- PoC §3.2 (p. 6): scoring and significance are separate jobs; "KS cannot rank on its own". §3.4 (p. 7):
  "Macro-KS is the significance gate; the HRT or RMST confidence interval gives the direction"; "Declared targets.
  Every attribution result states which functional it explains". §2.6 (p. 5) and §9.1 Step 4 (p. 19): the
  counterfactual check this gate decides.
- PhAIL §3.3 (p. 6), the outcomes: "(i) A is uniformly better", "(ii) A and B are different but neither is
  uniformly better (CDFs cross – reasonable scalars give opposite signs ...)", "(iii) A and B are
  indistinguishable at the available N"; the gate is "regimes (i)+(ii) vs (iii)"; "Direction (outcome (i)
  vs (ii)) comes from a scalar"; stratified pooled logrank "as a sanity-check". App. I (p. 18): pairwise
  KS sup-signs can cycle, so the KS sign is never used here. App. J (p. 18-19): null calibration.

Families (plan r2 "Declared functionals"; ``vlapoc.metrics_registry``): a verdict is computed in one declared family
and every label says which. ``"W"`` (LIBERO default): whole-task (T, E) per rollout, a whole-task macro-KS gate per
task, the metrics f30, auc_h (ties ½), rmst60, hrt60, censored60, the whole-task human anchor. ``"O"``: operation
pairs, an operation-level gate, ``rmst60_op_cap``, the operation human anchor. ``"PhAIL"``: PhAIL's own semantics
(operation pairs with ghosts, τ 240 s, AUC ties dropped) for reproducing PhAIL (S2/S7). Family W is the declared
attribution verdict: its per-arm, per-cell point estimates equal the cell means of the S6 returns (AC6.10/AC7.7).

What is reused (S1, ``vlapoc.evaluation``): ``macro_ks`` (PhAIL's pooled-resample episode-clustered bootstrap,
equal to PhAIL's ``per_obj_macro_bootstrap_ks`` on PhAIL's data, S2 AC2.1), ``logrank_stratified``, the scalar
metrics and the LIBERO arm builders. New here: the joint episode-clustered bootstrap of the metric *differences*
(whole episodes within each cell, arms resampled independently, the human anchor held fixed because it is the
common yardstick of both arms; S2's single-policy HRT CI resamples it instead, to match PhAIL Table 2), the
percentile-bootstrap two-sided p, Holm's step-down adjustment (``holm``; statsmodels is not installed and scipy has
no Holm), the outcome rule, and the batch-plan guard (the eval batch plan must match between arms: LeRobot's
paired-comparison guidance, and the measured batch effect on SmolVLA, plan r2 AC7.8).

Outcome rule (``outcome``), at level α (default 0.05):
- gate does not fire (macro-KS p >= α) -> ``"iii"`` (indistinguishable at this N);
- gate fires, ≥ 1 metric significant after Holm, all significant metrics favour the same arm -> ``"i"``
  with that arm as ``direction``;
- gate fires, significant metrics favour different arms -> ``"ii"`` (CDFs cross: a trade-off);
- gate fires, no metric significant after Holm -> ``"gate_only"``: the distributions differ but no declared
  scalar resolves a direction at this N (accepted divergence, plan r2: PoC §4.4 "each metric's bootstrap CI gives
  the direction", and no CI excludes 0).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vlapoc import evaluation as ev
from vlapoc import metrics_registry as mr
from vlapoc.libero_convert import REPO_ROOT

Arm = dict


def holm(pvals: list[float]) -> list[float]:
    """Holm's step-down adjusted p-values (Holm 1979): sort ascending, p_(i) × (m - i) for i = 0..m-1, running
    maximum, capped at 1, returned in the input order. Rejecting ``adj < α`` controls the family-wise error."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj_sorted = np.minimum(1.0, np.maximum.accumulate(p[order] * (m - np.arange(m))))
    out = np.empty(m)
    out[order] = adj_sorted
    return out.tolist()


def metric_value(entry: dict, arm: Arm, human: Arm | None) -> float:
    """A registry verdict metric (``entry["verdict"]``) on an arm: ``pooled`` metrics pool the arm's cells, ``macro``
    metrics average cells with equal weights. Applied to a one-cell arm this is the cell's value."""
    v = entry["verdict"]
    a, kind = v["args"], v["compute"]
    if kind == "F_at":
        return ev.F_at(ev.pooled(arm), a["t"])
    if kind == "one_minus_F":
        return 1.0 - ev.F_at(ev.pooled(arm), a["t"])
    if kind == "rmst":
        return ev.rmst(ev.pooled(arm), a["tau"])
    if kind == "rmst_capped":
        return ev.rmst_capped(ev.pooled(arm), a["tau"])
    if kind == "hrt":
        return ev.hrt(arm, human, a["tau"])
    if kind == "auc_ties_half":
        return ev.macro_auc_ties_half(arm, human)
    if kind == "auc_ties_dropped":
        return ev.macro_auc(arm, human)
    raise ValueError(f"unknown verdict compute {kind!r} in registry entry {entry['key']}")


def _resample(arm: Arm, rng: np.random.Generator) -> Arm:
    return {c: [arm[c][i] for i in rng.integers(0, len(arm[c]), size=len(arm[c]))] if arm[c] else [] for c in sorted(arm)}


def outcome(gate_p: float, metrics: dict, alpha: float = 0.05) -> tuple[str, str | None]:
    """The PhAIL §3.3 outcome from the gate p and the per-metric results (``{name: {"diff_better", "p_holm"}}``,
    ``diff_better > 0`` = arm A better). Direction comes only from scalars."""
    if gate_p >= alpha:
        return "iii", None
    sig = {k: v for k, v in metrics.items() if v["p_holm"] < alpha}
    if not sig:
        return "gate_only", None
    signs = {np.sign(v["diff_better"]) for v in sig.values()}
    if len(signs) == 1:
        return "i", "A" if signs.pop() > 0 else "B"
    return "ii", None


def _plan_key(plan: dict | None):
    return None if plan is None else json.dumps({k: v for k, v in plan.items() if k != "provenance"}, sort_keys=True)


def compare_policies(arm_a: Arm, arm_b: Arm, human: Arm | None, family: str = "W", n_boot: int = 1000,
                     n_boot_ks: int = 2000, seed: int = 0, alpha: float = 0.05, names: tuple[str, str] = ("A", "B"),
                     batch_plans: tuple[dict | None, dict | None] | None = None, allow_plan_mismatch: bool = False) -> dict:
    """Two-step verdict between two arms (``{cell: [Episode]}``) over their common cells, in a declared family.

    Step 1, significance: ``evaluation.macro_ks`` on the family's pairs (D̄ and pooled-resample p, ``n_boot_ks``,
    rng ``seed``). Step 2, direction: for each verdict metric of the family (``metrics_registry.VERDICT_FAMILIES``),
    Δ = m(A) − m(B) with a joint episode-clustered bootstrap (``n_boot``, rng ``seed + 1``; both arms resampled
    within cells, ``human`` fixed), its 95% percentile CI, the two-sided percentile-bootstrap
    p = min(1, 2 · min(#(Δ* ≤ 0) + 1, #(Δ* ≥ 0) + 1) / (n_boot + 1)), and Holm's adjustment across the family's
    metrics. ``diff_better`` > 0 means A is better. Each metric also reports its per-cell point estimates for both
    arms (``cells``), the values the S6 return identities are checked against.

    ``batch_plans``: the two arms' eval batch plans (``vlapoc.runs.batch_plan_of``). If both are known and differ,
    raise unless ``allow_plan_mismatch``, which labels the verdict "unpaired plan" (plan r2 AC7.8).
    """
    if family not in mr.VERDICT_FAMILIES:
        raise ValueError(f"family must be one of {list(mr.VERDICT_FAMILIES)}")
    plan_match = None
    if batch_plans is not None and batch_plans[0] is not None and batch_plans[1] is not None:
        plan_match = _plan_key(batch_plans[0]) == _plan_key(batch_plans[1])
        if not plan_match and not allow_plan_mismatch:
            raise ValueError(f"eval batch plans differ between {names[0]} and {names[1]} (a stochastic policy's outcome depends "
                             f"on the batch plan); pass allow_plan_mismatch=True to compute an 'unpaired plan' verdict")
    cells = [c for c in sorted(set(arm_a) & set(arm_b)) if arm_a[c] and arm_b[c] and (human is None or human.get(c))]
    a = {c: arm_a[c] for c in cells}
    b = {c: arm_b[c] for c in cells}
    h = {c: human[c] for c in cells} if human is not None else None
    gate = ev.macro_ks(a, b, n_boot_ks, np.random.default_rng(seed))
    entries = mr.verdict_metrics(family)
    point_a = [metric_value(e, a, h) for e in entries]
    point_b = [metric_value(e, b, h) for e in entries]
    rng = np.random.default_rng(seed + 1)
    boot = np.empty((n_boot, len(entries)))
    for i in range(n_boot):
        ra, rb = _resample(a, rng), _resample(b, rng)
        boot[i] = [metric_value(e, ra, h) - metric_value(e, rb, h) for e in entries]
    res, raw_p = {}, []
    for j, e in enumerate(entries):
        d = point_a[j] - point_b[j]
        bj = boot[:, j]
        p = min(1.0, 2.0 * min((np.sum(bj <= 0) + 1) / (n_boot + 1), (np.sum(bj >= 0) + 1) / (n_boot + 1)))
        raw_p.append(p)
        sgn = 1.0 if e["verdict"]["higher_is_better"] else -1.0
        per_cell = {c: {"A": metric_value(e, {c: a[c]}, {c: h[c]} if h else None),
                        "B": metric_value(e, {c: b[c]}, {c: h[c]} if h else None)} for c in cells}
        res[e["key"]] = {"label": f"{e['metric']} [family {family}, τ {e['tau']}]", "family": family, "tau": e["tau"],
                         "aggregate": e["verdict"]["aggregate"], "A": point_a[j], "B": point_b[j], "diff": d,
                         "ci95": [float(np.quantile(bj, 0.025)), float(np.quantile(bj, 0.975))], "p": p,
                         "higher_is_better": e["verdict"]["higher_is_better"], "diff_better": sgn * d,
                         "favours": names[0] if sgn * d > 0 else (names[1] if sgn * d < 0 else None), "cells": per_cell}
    for k, pa in zip(res, holm(raw_p)):
        res[k]["p_holm"] = pa
        res[k]["significant"] = pa < alpha
    out, direction = outcome(gate["p"], res, alpha)
    lr = ev.logrank_stratified(a, b)
    label = f"family {family}" + ("" if plan_match is not False else ", unpaired plan")
    return {"arms": list(names), "family": family, "label": label, "cells": cells,
            "n_episodes": {names[0]: sum(len(v) for v in a.values()), names[1]: sum(len(v) for v in b.values())},
            "tau_s": mr.FAMILY_TAU_S[family], "alpha": alpha,
            "gate": {"test": f"macro-KS (pooled-resample episode-clustered bootstrap), family {family} pairs",
                     "D": gate["D"], "p": gate["p"], "D_cells": gate["D_cells"], "n_boot": n_boot_ks, "fires": gate["p"] < alpha},
            "metrics": res, "holm_family": list(res), "outcome": out,
            "direction": {"A": names[0], "B": names[1]}.get(direction) if direction else None,
            "batch_plan_match": plan_match, "logrank_sanity": {"chi2": lr["chi2"], "p": lr["p"]}, "n_boot": n_boot, "seed": seed}


def describe(v: dict) -> str:
    a, b = v["arms"]
    head = (f"[{v['label']}] {a} vs {b} ({len(v['cells'])} cells, n {v['n_episodes'][a]}/{v['n_episodes'][b]} episodes): "
            f"macro-KS D̄ {v['gate']['D']:.4f} p {v['gate']['p']:.4f} ({'fires' if v['gate']['fires'] else 'does not fire'}) -> "
            f"outcome ({v['outcome']})" + (f", direction {v['direction']}" if v["direction"] else ""))
    rows = [f"    {k:13s} {m['label']:58s} {a} {m['A']:.4f} | {b} {m['B']:.4f} | Δ {m['diff']:+.4f} [{m['ci95'][0]:+.4f}, "
            f"{m['ci95'][1]:+.4f}] p {m['p']:.4f} Holm {m['p_holm']:.4f} -> favours {m['favours']}{' *' if m['significant'] else ''}"
            for k, m in v["metrics"].items()]
    return "\n".join([head, *rows, f"    sanity: stratified logrank chi2 {v['logrank_sanity']['chi2']:.2f} p {v['logrank_sanity']['p']:.3g}"])


def null_calibration(arm: Arm, trials: int = 200, n_boot: int = 500, seed: int = 0, alpha: float = 0.05) -> dict:
    """PhAIL App. J setup (i), same-model split: per trial, each cell's episodes are split uniformly at random into
    two halves and the macro-KS gate is run on them; returns the empirical rejection rate at ``alpha``."""
    rng = np.random.default_rng(seed)
    ps_ = []
    for _ in range(trials):
        a, b = {}, {}
        for c in sorted(arm):
            idx = rng.permutation(len(arm[c]))
            half = len(idx) // 2
            a[c] = [arm[c][i] for i in idx[:half]]
            b[c] = [arm[c][i] for i in idx[half:2 * half]]
        ps_.append(ev.macro_ks(a, b, n_boot, np.random.default_rng(int(rng.integers(2**63))))["p"])
    p = np.asarray(ps_)
    return {"trials": trials, "n_boot": n_boot, "mean_p": float(p.mean()), "reject_rate": float((p < alpha).mean()),
            "reject_rate_se": float(np.sqrt((p < alpha).mean() * (1 - (p < alpha).mean()) / trials)), "alpha": alpha,
            "p_values": p.tolist()}


# ---------------------------------------------------------------------------------------------
# eval runs -> arms, and the recorded verify stage
# ---------------------------------------------------------------------------------------------


def eval_rollouts(run_id: str, lineage_root: Path | None = None):
    """The rollouts parquet an eval run logged, checked against its recorded sha256 (``returns.load_eval_rollouts``:
    a ``TemporaryDirectory`` download, so nothing is left under ``data/scratch``)."""
    from vlapoc.returns import load_eval_rollouts

    return load_eval_rollouts(run_id, lineage_root, make_out_dir=False)[0]


def verify_run(eval_a: str, eval_b: str, name: str, family: str = "W", n_boot: int = 1000, n_boot_ks: int = 2000,
               seed: int = 0, tau: float | None = None, out_dir: Path | None = None, chain: str | None = None,
               lineage_root: Path | None = None, allow_plan_mismatch: bool = False) -> dict:
    """Stage ``verify`` (PoC §9.1 Step 4 p. 19): the two-step verdict between two eval runs' rollouts in a declared
    family, as one MLflow run whose inputs are both eval runs' ``rollouts`` entities (so ``trace`` reaches each
    policy's checkpoint and, for trained policies, the dataset version and raw files). Writes ``verdict.json``.

    ``verify.attribution_grade`` is true only when both eval runs carry an *asserted* attribution grade
    (``eval.attribution_grade.asserted=true``, set by ``vlapoc.runs.eval_run`` only after its protocol check, plan r2
    AC5.9/AC7.9); every other verdict is labelled "smoke (not attribution-grade)". Round-1 runs tagged
    ``eval.attribution_grade=true`` without the asserted tag meant only "has a train run" and do not count.
    """
    from vlapoc.runs import batch_plan_of, client, stage_run

    del tau  # the family fixes τ (metrics_registry.FAMILY_TAU_S)
    c = client(lineage_root)
    ra, rb = c.get_run(eval_a), c.get_run(eval_b)
    da, db = eval_rollouts(eval_a, lineage_root), eval_rollouts(eval_b, lineage_root)
    arm_a, arm_b = ev.libero_arm(da, family), ev.libero_arm(db, family)
    human = ev.libero_human_arm(family, sorted(set(arm_a) | set(arm_b)))
    names = (ra.info.run_name, rb.info.run_name)
    plans = (batch_plan_of(ra), batch_plan_of(rb))
    v = compare_policies(arm_a, arm_b, human, family, n_boot=n_boot, n_boot_ks=n_boot_ks, seed=seed, names=names,
                         batch_plans=plans, allow_plan_mismatch=allow_plan_mismatch)
    grade = all(r.data.tags.get("eval.attribution_grade.asserted") == "true" for r in (ra, rb))
    v["label"] = ("attribution-grade counterfactual verdict" if grade else "smoke (not attribution-grade)") + ", " + v["label"]
    v.update({"eval_runs": [eval_a, eval_b], "attribution_grade": grade, "batch_plans": list(plans)})
    out_dir = Path(out_dir) if out_dir else REPO_ROOT / "data" / "eval" / f"verdict-{family}-{eval_a[:8]}-{eval_b[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "verdict.json"
    path.write_text(json.dumps(v, indent=1, default=str))
    with stage_run("verify", name, lineage_root, chain=chain) as r:
        for role, run in (("rollouts_a", ra), ("rollouts_b", rb)):
            r.input_entity("rollouts", run.data.tags["output.rollouts.sha256"], role)
        r.params({"eval_a": eval_a, "eval_b": eval_b, "family": family, "tau_s": mr.FAMILY_TAU_S[family], "n_boot": n_boot,
                  "n_boot_ks": n_boot_ks, "seed": seed, "alpha": v["alpha"], "metrics": list(v["metrics"]), "multiplicity": "holm",
                  "allow_plan_mismatch": allow_plan_mismatch,
                  "human_anchor": f"data/eval/human_reference/{'human_reference' if family == 'W' else 'human_operations'}.parquet "
                                  "(all 1000 v0 originals, fixed)"})
        r.output_file("verdict", path, kind="verdict")
        r.tags({"verify.outcome": v["outcome"], "verify.direction": v["direction"] or "none", "verify.family": family,
                "verify.gate_p": f"{v['gate']['p']:.6g}", "verify.attribution_grade": str(grade).lower(),
                "verify.batch_plan_match": str(v["batch_plan_match"]).lower(), "verify.label": v["label"]})
        v["run_id"] = r.run_id
    return v


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m vlapoc.verdict")
    p.add_argument("--phail", nargs=2, metavar=("A", "B"), help="two PhAIL models (OpenPI, GR00T, ACT, SmolVLA)")
    p.add_argument("--phail-all", action="store_true", help="all six PhAIL model pairs")
    p.add_argument("--null-calibration", metavar="MODEL", help="PhAIL App. J (i) same-model split for one model")
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--eval-runs", nargs=2, metavar=("A", "B"), help="two eval run ids: recorded as a verify stage run")
    p.add_argument("--family", default="W", choices=["W", "O"], help="LIBERO verdict family (registry); W is declared")
    p.add_argument("--allow-plan-mismatch", action="store_true")
    p.add_argument("--lineage-root", type=Path, default=None, help="a scratch lineage store (tests)")
    p.add_argument("--name", default="verify")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--n-boot-ks", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, help="write the verdict(s) JSON here")
    a = p.parse_args(argv)
    results = []
    if a.phail or a.phail_all:
        from itertools import combinations

        arms = ev.phail_cohort_arms()
        pairs = list(combinations(("OpenPI", "GR00T", "ACT", "SmolVLA"), 2)) if a.phail_all else [tuple(a.phail)]
        for x, y in pairs:
            v = compare_policies(arms[x], arms[y], arms["Teleop"], "PhAIL", n_boot=a.n_boot, n_boot_ks=a.n_boot_ks,
                                 seed=a.seed, names=(x, y))
            print(describe(v), flush=True)
            results.append(v)
    if a.null_calibration:
        arms = ev.phail_cohort_arms()
        r = null_calibration(arms[a.null_calibration], a.trials, 500, a.seed)
        print(f"null calibration (PhAIL App. J setup (i), same-model split) {a.null_calibration}: {r['trials']} trials x "
              f"{r['n_boot']} inner resamples: mean p {r['mean_p']:.3f}, rejection rate at α=0.05 {r['reject_rate']:.3f} "
              f"± {1.96 * r['reject_rate_se']:.3f} (1.96 SE)")
        results.append({"null_calibration": a.null_calibration, **r})
    if a.eval_runs:
        v = verify_run(*a.eval_runs, a.name, a.family, a.n_boot, a.n_boot_ks, a.seed, lineage_root=a.lineage_root,
                       allow_plan_mismatch=a.allow_plan_mismatch)
        print(describe(v))
        print(f"verify run {v['run_id']} ({v['label']})")
        results.append(v)
    if a.out and results:
        Path(a.out).write_text(json.dumps(results if len(results) > 1 else results[0], indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
