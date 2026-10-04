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
  "Macro-KS is the significance gate; the HRT or RMST confidence interval gives the direction". §2.6 (p. 5)
  and §9.1 Step 4 (p. 19): the counterfactual check this gate decides.
- PhAIL §3.3 (p. 6), the outcomes: "(i) A is uniformly better", "(ii) A and B are different but neither is
  uniformly better (CDFs cross – reasonable scalars give opposite signs ...)", "(iii) A and B are
  indistinguishable at the available N"; the gate is "regimes (i)+(ii) vs (iii)"; "Direction (outcome (i)
  vs (ii)) comes from a scalar"; stratified pooled logrank "as a sanity-check". App. I (p. 18): pairwise
  KS sup-signs can cycle, so the KS sign is never used here. App. J (p. 18-19): null calibration.

What is reused (S1, ``vlapoc.evaluation``): ``macro_ks`` (PhAIL's pooled-resample episode-clustered bootstrap,
equal to PhAIL's ``per_obj_macro_bootstrap_ks`` on PhAIL's data, S2 AC2.1), ``logrank_stratified``, and the
scalar metrics ``F_at``, ``rmst``, ``hrt``, ``macro_auc`` (each reproduced on PhAIL's data in S2). New here:
the joint episode-clustered bootstrap of the metric *differences* (same resampling scheme as
``evaluation.cluster_bootstrap``: whole episodes within each cell, arms resampled independently, the human
anchor held fixed, because it is the common yardstick of both arms; plan: the HRT anchor is fixed across arms),
the percentile-bootstrap two-sided p, Holm's step-down adjustment (``holm``; statsmodels is not installed and
scipy has no Holm, so a small tested function), and the outcome rule.

Outcome rule (``outcome``), at level α (default 0.05):
- gate does not fire (macro-KS p >= α) -> ``"iii"`` (indistinguishable at this N);
- gate fires, ≥ 1 metric significant after Holm, all significant metrics favour the same arm -> ``"i"``
  with that arm as ``direction``;
- gate fires, significant metrics favour different arms -> ``"ii"`` (CDFs cross: a trade-off);
- gate fires, no metric significant after Holm -> ``"gate_only"``: the distributions differ but no declared
  scalar resolves a direction at this N. PhAIL's three-way taxonomy does not name this case; it is reported
  rather than forced into (i) or (ii) (a recorded divergence).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from vlapoc import evaluation as ev
from vlapoc.libero_convert import REPO_ROOT

Arm = dict


@dataclass(frozen=True)
class Metric:
    """A declared scoring scalar: ``fn(arm, human, tau) -> float``; ``higher_is_better`` fixes the sign."""

    name: str
    label: str
    fn: Callable[[Arm, Arm | None, float], float]
    higher_is_better: bool


def _f(t_s: float) -> Callable:
    return lambda arm, human, tau: ev.F_at(ev.pooled(arm), t_s)


# The PoC §4.1 four yardsticks (p. 9): F(30 s), HRT (per-cell macro), global RMST, AUC vs human (macro)
METRICS = {
    "F30": Metric("F30", "F(30 s), pooled", _f(30.0), True),
    "HRT": Metric("HRT", "HRT(τ), per-cell macro vs the human anchor", lambda a, h, tau: ev.hrt(a, h, tau), True),
    "RMST": Metric("RMST", "RMST(τ) s, pooled (global)", lambda a, h, tau: ev.rmst(ev.pooled(a), tau), False),
    "AUC": Metric("AUC", "AUC vs human, per-cell macro", lambda a, h, tau: ev.macro_auc(a, h), True),
}


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


def compare_policies(arm_a: Arm, arm_b: Arm, human: Arm | None, tau: float, metrics: tuple[str, ...] = ("F30", "HRT", "RMST", "AUC"),
                     n_boot: int = 1000, n_boot_ks: int = 2000, seed: int = 0, alpha: float = 0.05,
                     names: tuple[str, str] = ("A", "B")) -> dict:
    """Two-step verdict between two arms (``{cell: [Episode]}``) over their common cells.

    Step 1, significance: ``evaluation.macro_ks`` (D̄ and pooled-resample p, ``n_boot_ks``, rng ``seed``).
    Step 2, direction: for each metric, Δ = m(A) − m(B) with a joint episode-clustered bootstrap (``n_boot``,
    rng ``seed + 1``; both arms resampled within cells, ``human`` fixed), its 95% percentile CI, the two-sided
    percentile-bootstrap p = min(1, 2 · min(#(Δ* ≤ 0) + 1, #(Δ* ≥ 0) + 1) / (n_boot + 1)), and Holm's
    adjustment across the metrics. ``diff_better`` is Δ signed so that > 0 means A is better.
    Sanity line: ``evaluation.logrank_stratified``. The KS statistic is reported unsigned: it is never a ranker
    (PhAIL App. I). Returns a JSON-able dict.
    """
    cells = [c for c in sorted(set(arm_a) & set(arm_b)) if arm_a[c] and arm_b[c] and (human is None or human.get(c))]
    a = {c: arm_a[c] for c in cells}
    b = {c: arm_b[c] for c in cells}
    h = {c: human[c] for c in cells} if human is not None else None
    gate = ev.macro_ks(a, b, n_boot_ks, np.random.default_rng(seed))
    ms = [METRICS[m] for m in metrics]
    point_a = [m.fn(a, h, tau) for m in ms]
    point_b = [m.fn(b, h, tau) for m in ms]
    rng = np.random.default_rng(seed + 1)
    boot = np.empty((n_boot, len(ms)))
    for i in range(n_boot):
        ra, rb = _resample(a, rng), _resample(b, rng)
        boot[i] = [m.fn(ra, h, tau) - m.fn(rb, h, tau) for m in ms]
    res = {}
    raw_p = []
    for j, m in enumerate(ms):
        d = point_a[j] - point_b[j]
        bj = boot[:, j]
        p = min(1.0, 2.0 * min((np.sum(bj <= 0) + 1) / (n_boot + 1), (np.sum(bj >= 0) + 1) / (n_boot + 1)))
        raw_p.append(p)
        sgn = 1.0 if m.higher_is_better else -1.0
        res[m.name] = {"label": m.label, "tau_s": tau if m.name in ("HRT", "RMST") else (30.0 if m.name == "F30" else None),
                       "A": point_a[j], "B": point_b[j], "diff": d, "ci95": [float(np.quantile(bj, 0.025)), float(np.quantile(bj, 0.975))],
                       "p": p, "higher_is_better": m.higher_is_better, "diff_better": sgn * d,
                       "favours": names[0] if sgn * d > 0 else (names[1] if sgn * d < 0 else None)}
    for m, pa in zip(res, holm(raw_p)):
        res[m]["p_holm"] = pa
        res[m]["significant"] = pa < alpha
    out, direction = outcome(gate["p"], res, alpha)
    lr = ev.logrank_stratified(a, b)
    return {"arms": list(names), "cells": cells, "n_episodes": {names[0]: sum(len(v) for v in a.values()),
                                                                  names[1]: sum(len(v) for v in b.values())},
            "tau_s": tau, "alpha": alpha, "gate": {"test": "macro-KS (pooled-resample episode-clustered bootstrap)",
                                                   "D": gate["D"], "p": gate["p"], "D_cells": gate["D_cells"],
                                                   "n_boot": n_boot_ks, "fires": gate["p"] < alpha},
            "metrics": res, "holm_family": list(res), "outcome": out,
            "direction": {"A": names[0], "B": names[1]}.get(direction) if direction else None,
            "logrank_sanity": {"chi2": lr["chi2"], "p": lr["p"]}, "n_boot": n_boot, "seed": seed}


def describe(v: dict) -> str:
    a, b = v["arms"]
    head = (f"{a} vs {b} ({len(v['cells'])} cells, n {v['n_episodes'][a]}/{v['n_episodes'][b]} episodes, τ {v['tau_s']:g} s): "
            f"macro-KS D̄ {v['gate']['D']:.4f} p {v['gate']['p']:.4f} ({'fires' if v['gate']['fires'] else 'does not fire'}) -> "
            f"outcome ({v['outcome']})" + (f", direction {v['direction']}" if v["direction"] else ""))
    rows = [f"    {k:4s} {m['label']:44s} {a} {m['A']:.4f} | {b} {m['B']:.4f} | Δ {m['diff']:+.4f} [{m['ci95'][0]:+.4f}, "
            f"{m['ci95'][1]:+.4f}] p {m['p']:.4f} Holm {m['p_holm']:.4f} -> favours {m['favours']}{' *' if m['significant'] else ''}"
            for k, m in v["metrics"].items()]
    return "\n".join([head, *rows, f"    sanity: stratified logrank chi2 {v['logrank_sanity']['chi2']:.2f} p {v['logrank_sanity']['p']:.3g}"])


# ---------------------------------------------------------------------------------------------
# LIBERO arms from rollout records (S5) and the human reference (S3/S4)
# ---------------------------------------------------------------------------------------------


def libero_arm(rollouts) -> Arm:
    """``{suite/task: [Episode]}`` from a rollout record: each rollout's operation-level pairs (S4 rule, stored in
    ``pairs``), so libero_10 contributes per-sub-task T as in S1/S4."""
    arm: dict = {}
    for r in rollouts.itertuples():
        arm.setdefault(f"{r.suite}/{int(r.libero_task_id)}", []).append(
            ev.Episode(r.rollout_id, tuple((float(t), int(e)) for t, e in json.loads(r.pairs))))
    return arm


def libero_human_arm(cells: list[str] | None = None) -> Arm:
    """The human anchor per task: all 1000 v0 originals' operation-level pairs (S4 ``human_operations.parquet``)."""
    import pandas as pd

    h = pd.read_parquet(REPO_ROOT / "data" / "eval" / "human_reference" / "human_operations.parquet")
    arm: dict = {}
    for (s, t, e), g in h.sort_values(["episode_id", "op_k"]).groupby(["suite", "libero_task_id", "episode_id"]):
        arm.setdefault(f"{s}/{int(t)}", []).append(ev.Episode(e, tuple((float(x), int(y)) for x, y in zip(g["T_op_s"], g["event"]))))
    return {c: v for c, v in arm.items() if cells is None or c in cells}


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
# CLI
# ---------------------------------------------------------------------------------------------


def _eval_rollouts(run_id: str):
    """The rollouts parquet an eval run logged (its MLflow artifact), checked against the run's recorded sha256."""
    import tempfile

    import pandas as pd

    from vlapoc.catalog import sha256_file
    from vlapoc.runs import client

    c = client()
    run = c.get_run(run_id)
    tmp = Path(tempfile.mkdtemp(prefix="verdict_", dir=REPO_ROOT / "data" / "scratch"))
    path = Path(c.download_artifacts(run_id, run.data.tags.get("output.rollouts.artifact", "rollouts.parquet"), str(tmp)))
    if sha256_file(path) != run.data.tags["output.rollouts.sha256"]:
        raise ValueError(f"eval run {run_id}: rollouts artifact differs from its recorded sha256")
    return pd.read_parquet(path)


def verify_run(eval_a: str, eval_b: str, name: str, n_boot: int = 1000, n_boot_ks: int = 2000, seed: int = 0,
               tau: float = 60.0, out_dir: Path | None = None, chain: str | None = None) -> dict:
    """Stage ``verify`` (PoC §9.1 Step 4 p. 19): the two-step verdict between two eval runs' rollouts, as one MLflow
    run whose inputs are both eval runs' ``rollouts`` entities (so ``trace`` reaches each policy's checkpoint and,
    for trained policies, the dataset version and raw files). Writes ``verdict.json``."""
    from vlapoc.runs import client, stage_run

    c = client()
    ra, rb = c.get_run(eval_a), c.get_run(eval_b)
    da, db = _eval_rollouts(eval_a), _eval_rollouts(eval_b)
    arm_a, arm_b = libero_arm(da), libero_arm(db)
    human = libero_human_arm(sorted(set(arm_a) | set(arm_b)))
    names = (ra.info.run_name, rb.info.run_name)
    v = compare_policies(arm_a, arm_b, human, tau, n_boot=n_boot, n_boot_ks=n_boot_ks, seed=seed, names=names)
    grade = all(r.data.tags.get("eval.attribution_grade") == "true" for r in (ra, rb))
    v.update({"eval_runs": [eval_a, eval_b], "attribution_grade": grade,
              "label": "attribution-grade counterfactual verdict" if grade else
              "smoke: at least one arm is not attribution-grade (external checkpoint or untrained policy); exercises the code only"})
    out_dir = Path(out_dir) if out_dir else REPO_ROOT / "data" / "eval" / f"verdict-{eval_a[:8]}-{eval_b[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "verdict.json"
    path.write_text(json.dumps(v, indent=1, default=str))
    with stage_run("verify", name, chain=chain) as r:
        for role, run in (("rollouts_a", ra), ("rollouts_b", rb)):
            r.input_entity("rollouts", run.data.tags["output.rollouts.sha256"], role)
        r.params({"eval_a": eval_a, "eval_b": eval_b, "tau_s": tau, "n_boot": n_boot, "n_boot_ks": n_boot_ks, "seed": seed,
                  "alpha": v["alpha"], "metrics": list(v["metrics"]), "multiplicity": "holm",
                  "human_anchor": "data/eval/human_reference/human_operations.parquet (all 1000 v0 originals, fixed)"})
        r.output_file("verdict", path, kind="verdict")
        r.tags({"verify.outcome": v["outcome"], "verify.direction": v["direction"] or "none",
                "verify.gate_p": f"{v['gate']['p']:.6g}", "verify.attribution_grade": str(grade).lower()})
        v["run_id"] = r.run_id
    return v


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m vlapoc.verdict")
    p.add_argument("--phail", nargs=2, metavar=("A", "B"), help="two PhAIL models (OpenPI, GR00T, ACT, SmolVLA)")
    p.add_argument("--phail-all", action="store_true", help="all six PhAIL model pairs")
    p.add_argument("--null-calibration", metavar="MODEL", help="PhAIL App. J (i) same-model split for one model")
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--eval-runs", nargs=2, metavar=("A", "B"), help="two eval run ids: recorded as a verify stage run")
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
            v = compare_policies(arms[x], arms[y], arms["Teleop"], 240.0, n_boot=a.n_boot, n_boot_ks=a.n_boot_ks,
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
        v = verify_run(*a.eval_runs, a.name, a.n_boot, a.n_boot_ks, a.seed)
        print(describe(v))
        print(f"verify run {v['run_id']} ({v['label']})")
        results.append(v)
    if a.out and results:
        Path(a.out).write_text(json.dumps(results if len(results) > 1 else results[0], indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
