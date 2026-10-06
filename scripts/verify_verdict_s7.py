"""Verify evaluation-layer S7 (the two-step verdict): Holm, the outcome rule, KS never ranks (AC7.3, PhAIL App. I),
and summaries of the PhAIL runs (AC7.1, AC7.2). Runnable as a script or with pytest (``test_*``).

Grounding: PoC §4.4 p. 10 (two-step verdict, Holm/Bonferroni across metrics), §3.2 p. 6 ("KS cannot rank on its
own"); PhAIL §3.3 p. 6 (outcomes (i)/(ii)/(iii)), App. I p. 18 (three CDFs whose sup-signs cycle A≻B≻C≻A).

    .venv/bin/python scripts/verify_verdict_s7.py          # unit checks (LABELLED SYNTHETIC data)
    .venv/bin/python -m pytest -q scripts/verify_verdict_s7.py
    .venv/bin/python scripts/verify_verdict_s7.py --phail-seeds 0 1 2 3 4   # AC7.1 (r2): restated outcomes in every seed
    .venv/bin/python scripts/verify_verdict_s7.py --family-consistency data/eval/<run> [...]   # AC7.7
    .venv/bin/python scripts/verify_verdict_s7.py --plan-guard <eval a> <eval b mismatched> <eval c same plan as b> --lineage-root <scratch>  # AC7.8
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vlapoc import evaluation as ev  # noqa: E402
from vlapoc import verdict as vd  # noqa: E402

ART = REPO_ROOT / "tracker" / "features" / "2026-10-01-evaluation-layer" / "artifacts"


def test_holm_hand_example():
    # sorted p 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1 = 0.02, 0.03, 0.06, 0.04 -> running max 0.02, 0.03, 0.06, 0.06
    assert np.allclose(vd.holm([0.01, 0.04, 0.03, 0.005]), [0.03, 0.06, 0.06, 0.02])
    assert vd.holm([0.2]) == [0.2]
    assert vd.holm([0.6, 0.7]) == [1.0, 1.0]  # capped at 1


def test_outcome_rule():
    m = lambda d, p: {"diff_better": d, "p_holm": p}  # noqa: E731
    assert vd.outcome(0.20, {"RMST": m(+1, 0.001)}) == ("iii", None)  # gate first: no direction without it
    assert vd.outcome(0.01, {"RMST": m(+1, 0.01), "AUC": m(+1, 0.02), "F30": m(-1, 0.5)}) == ("i", "A")
    assert vd.outcome(0.01, {"RMST": m(-1, 0.01), "AUC": m(-1, 0.30)}) == ("i", "B")
    assert vd.outcome(0.01, {"RMST": m(+1, 0.01), "AUC": m(-1, 0.02)}) == ("ii", None)
    assert vd.outcome(0.01, {"RMST": m(+1, 0.20), "AUC": m(-1, 0.30)}) == ("gate_only", None)


def intransitive_arms(rep: int = 20) -> dict:
    """SYNTHETIC step CDFs in the shape of PhAIL App. I Fig. 5 (each with a T = inf asymptote or a late step):
    F_A = 0.5 from t = 1 (half ghosts); F_B = 0.1 on [1, 10), 0.8 from 10 (20% ghosts); F_C = 0.3 on [1, 50), 1 from 50.
    ``rep`` copies of a 10-episode pattern per arm, one cell."""
    inf = float("inf")
    pat = {"A": [1.0] * 5 + [inf] * 5, "B": [1.0] + [10.0] * 7 + [inf] * 2, "C": [1.0] * 3 + [50.0] * 7}
    return {k: {"x": [ev.Episode(f"{k}{r}_{i}", ((t, 1),)) for r in range(rep) for i, t in enumerate(v)]} for k, v in pat.items()}


def sup_sign(arm_a: dict, arm_b: dict) -> tuple[float, str]:
    grid = np.arange(0.0, 100.0, 0.25)
    d = ev.F_curve(arm_a["x"], grid) - ev.F_curve(arm_b["x"], grid)
    i = int(np.argmax(np.abs(d)))
    return float(abs(d[i])), ("first" if d[i] > 0 else "second")


def test_ks_sign_cycles_but_verdict_direction_comes_from_rmst():
    arms = intransitive_arms()
    signs = {pair: sup_sign(arms[pair[0]], arms[pair[1]]) for pair in (("A", "B"), ("B", "C"), ("C", "A"))}
    winners = {pair: pair[0] if s == "first" else pair[1] for pair, (_, s) in signs.items()}
    assert winners == {("A", "B"): "A", ("B", "C"): "B", ("C", "A"): "C"}  # A ≻ B ≻ C ≻ A: a cycle
    rmst = {k: ev.rmst(arms[k]["x"], 60.0) for k in arms}
    assert rmst["B"] < rmst["A"] and rmst["B"] < rmst["C"]  # the scalar ranks B first (transitive by construction)
    # family O's only verdict metric is the capped RMST(60), which on all-event data equals RMST(60) (registry rmst60_op_cap)
    v = vd.compare_policies(arms["A"], arms["B"], None, "O", n_boot=400, n_boot_ks=400, seed=0, names=("A", "B"))
    assert set(v["gate"]) == {"test", "D", "p", "D_cells", "n_boot", "fires"} and v["gate"]["D"] >= 0  # unsigned KS
    assert v["gate"]["fires"] and v["direction"] == "B"  # KS sup-sign says A; the verdict's direction is RMST's
    return signs, rmst, v


EXPECTED_AC7_1 = {("OpenPI", "GR00T"): "iii", ("OpenPI", "ACT"): "iii", ("GR00T", "ACT"): "ii",
                  ("OpenPI", "SmolVLA"): "i", ("GR00T", "SmolVLA"): "i", ("ACT", "SmolVLA"): "i"}  # plan r2 AC7.1


def phail_seeds(seeds: list[int]) -> None:
    """AC7.1 (r2): the six PhAIL pairs on the sorted export, family PhAIL, in every seed; outcomes vs the restatement."""
    arms = ev.phail_cohort_arms()
    out, ok = {}, True
    for (x, y), want in EXPECTED_AC7_1.items():
        for sd in seeds:
            v = vd.compare_policies(arms[x], arms[y], arms["Teleop"], "PhAIL", n_boot=1000, n_boot_ks=2000, seed=sd, names=(x, y))
            got = v["outcome"]
            ok &= got == want
            out[f"{x}-{y}-seed{sd}"] = {"gate_p": v["gate"]["p"], "D": v["gate"]["D"], "outcome": got, "expected": want,
                                        "direction": v["direction"], "p_holm": {k: m["p_holm"] for k, m in v["metrics"].items()},
                                        "favours": {k: m["favours"] for k, m in v["metrics"].items()}}
            print(f"AC7.1 {x} vs {y} seed {sd}: gate D̄ {v['gate']['D']:.4f} p {v['gate']['p']:.4f} -> ({got}) expected ({want}) "
                  f"{'OK' if got == want else 'MISMATCH'}; Holm p " + ", ".join(f"{k} {m['p_holm']:.3f}->{m['favours']}" for k, m in v["metrics"].items()),
                  flush=True)
    print(f"AC7.1 all {len(out)} (pair, seed) outcomes match the r2 restatement: {ok}")
    json.dump({"seeds": seeds, "results": out, "pass": ok}, open(ART / "s7_ac7_1_seed_sweep.json", "w"), indent=1)


def family_consistency(paths: list[Path]) -> None:
    """AC7.7 (r2): from the verdict side, family-W per-arm metrics == S6 cell means; family-O rmst60_op_cap == the
    operation-sample mean of ret_rmst60_op_cap; every verdict label contains its family."""
    import pandas as pd

    from vlapoc import returns as rt

    worst, labels_ok = 0.0, True
    for path in paths:
        ro = pd.read_parquet(path / "rollouts.parquet")
        ret, _ = rt.rollout_returns(ro)
        ops = rt.operation_returns(ro)
        for fam in ("W", "O"):
            arm = ev.libero_arm(ro, fam)
            v = vd.compare_policies(arm, arm, ev.libero_human_arm(fam, sorted(arm)), fam, n_boot=20, n_boot_ks=20, names=("A", "A2"))
            labels_ok &= f"family {fam}" in v["label"] and all(f"family {fam}" in m["label"] for m in v["metrics"].values())
            for key, m in v["metrics"].items():
                if fam == "O":
                    s6 = -float(ops["ret_rmst60_op_cap"].mean())
                elif key == "hrt60":
                    s6 = -float(ret.groupby("cell")["ret_hrt60_lin"].mean().sum())
                else:
                    col, sign = {"f30": ("ret_f30", 1), "rmst60": ("ret_rmst60", -1), "auc_h": ("ret_auc_expected", 1),
                                 "censored60": ("ret_censored60", 1)}[key]
                    s6 = sign * float(ret[col].mean())
                d = abs(s6 - m["A"])
                worst = max(worst, d)
                print(f"AC7.7 {path.name} [{v['label']}] {key}: verdict A {m['A']:.6f} vs S6 {s6:.6f} (|diff| {d:.1e})")
    ok = worst <= 1e-9 and labels_ok
    print(f"AC7.7 max |verdict - S6| = {worst:.2e}; labels carry their family: {labels_ok}; pass: {ok}")
    json.dump({"max_abs_diff": worst, "labels_ok": labels_ok, "pass": ok}, open(ART / "s7_ac7_7_family_consistency.json", "w"), indent=1)


def test_plan_guard_synthetic():
    arms = intransitive_arms(rep=3)
    p1 = {"batch_size": 5, "tasks": [["libero_spatial", 0]], "n_init": 10}
    p2 = {**p1, "batch_size": 1}
    try:
        vd.compare_policies(arms["A"], arms["B"], None, "O", n_boot=10, n_boot_ks=10, batch_plans=(p1, p2))
        raised = False
    except ValueError:
        raised = True
    assert raised
    v = vd.compare_policies(arms["A"], arms["B"], None, "O", n_boot=10, n_boot_ks=10, batch_plans=(p1, p2), allow_plan_mismatch=True)
    assert "unpaired plan" in v["label"] and v["batch_plan_match"] is False
    v2 = vd.compare_policies(arms["A"], arms["B"], None, "O", n_boot=10, n_boot_ks=10,
                             batch_plans=(p1, {**p1, "provenance": "reconstructed from round-1 params"}))
    assert v2["batch_plan_match"] is True and "unpaired" not in v2["label"]


def plan_guard_runs(a: str, b_mismatch: str, c_same: str, lineage_root: Path) -> None:
    """AC7.8 on recorded eval runs (scratch lineage store): a vs b (different plans) raises, then the override labels
    'unpaired plan'; c vs b_same... (same plan) passes."""
    from vlapoc.runs import batch_plan_of, client

    cl = client(lineage_root)
    pa, pb, pc = (batch_plan_of(cl.get_run(r)) for r in (a, b_mismatch, c_same))
    try:
        vd.verify_run(a, b_mismatch, "verify-plan-guard-refused [smoke]", lineage_root=lineage_root, n_boot=50, n_boot_ks=50)
        refused = None
    except ValueError as e:
        refused = str(e)[:140]
    v = vd.verify_run(a, b_mismatch, "verify-plan-guard-override [smoke]", lineage_root=lineage_root, n_boot=50, n_boot_ks=50,
                      allow_plan_mismatch=True)
    v2 = vd.verify_run(c_same, b_mismatch, "verify-plan-guard-same-plan [smoke]", lineage_root=lineage_root, n_boot=50, n_boot_ks=50)
    print(f"AC7.8 plans: a {json.dumps({k: pa[k] for k in ('batch_size', 'tasks', 'n_init')})}; b {json.dumps({k: pb[k] for k in ('batch_size', 'tasks', 'n_init')})}; "
          f"c {json.dumps({k: pc[k] for k in ('batch_size', 'tasks', 'n_init')})}")
    print(f"AC7.8 a vs b without override raised: {refused is not None} ({refused}); with override: run {v['run_id'][:8]} label {v['label']!r}; "
          f"c vs b (same plan): run {v2['run_id'][:8]} batch_plan_match {v2['batch_plan_match']}, label {v2['label']!r}")
    json.dump({"refused": refused, "override_label": v["label"], "same_plan_match": v2["batch_plan_match"], "same_plan_label": v2["label"],
               "pass": refused is not None and "unpaired plan" in v["label"] and v2["batch_plan_match"] is True},
              open(ART / "s7_ac7_8_plan_guard.json", "w"), indent=1)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--phail-seeds", type=int, nargs="+")
    ap.add_argument("--family-consistency", type=Path, nargs="+")
    ap.add_argument("--plan-guard", nargs=3)
    ap.add_argument("--lineage-root", type=Path)
    a = ap.parse_args()
    if a.phail_seeds:
        phail_seeds(a.phail_seeds)
        return 0
    if a.family_consistency:
        family_consistency([(REPO_ROOT / x).resolve() for x in a.family_consistency])
        return 0
    if a.plan_guard:
        plan_guard_runs(*a.plan_guard, (REPO_ROOT / a.lineage_root).resolve())
        return 0
    test_plan_guard_synthetic()
    print("AC7.8 SYNTHETIC plan guard: different plans raise; override labels 'unpaired plan'; a reconstructed plan equal to an explicit one passes: pass")
    test_holm_hand_example()
    print("AC7.x holm: hand example [0.01, 0.04, 0.03, 0.005] -> " + str(vd.holm([0.01, 0.04, 0.03, 0.005])) + "; single p unchanged; capped at 1: pass")
    test_outcome_rule()
    print("AC7.x outcome rule: (iii) without the gate, (i) A, (i) B, (ii), gate_only on labelled-synthetic metric sets: pass")
    signs, rmst, v = test_ks_sign_cycles_but_verdict_direction_comes_from_rmst()
    print("AC7.3 SYNTHETIC App. I CDFs: KS sup-sign winners " + "; ".join(f"{a} vs {b}: sup {d:.2f} -> {a if s == 'first' else b}"
                                                                     for (a, b), (d, s) in signs.items())
          + " (a cycle A≻B≻C≻A)")
    m = v["metrics"]["rmst60_op_cap"]
    print(f"AC7.3 RMST(60 s): A {rmst['A']:.2f}, B {rmst['B']:.2f}, C {rmst['C']:.2f} (a scalar: B ranks first, no cycle)")
    print(f"AC7.3 verdict A vs B [{v['label']}]: gate D̄ {v['gate']['D']:.2f} (unsigned) p {v['gate']['p']:.4f} -> outcome ({v['outcome']}), "
          f"direction {v['direction']} from capped RMST(60) Δ {m['diff']:+.2f} s [{m['ci95'][0]:+.2f}, {m['ci95'][1]:+.2f}], "
          f"while the KS sup-sign favours A")
    json.dump({"sup_signs": {f"{a}-{b}": s for (a, b), s in signs.items()}, "rmst60": rmst, "verdict_A_B": v},
              open(ART / "s7_ac7_3_intransitive.json", "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
