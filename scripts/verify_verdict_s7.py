"""Verify evaluation-layer S7 (the two-step verdict): Holm, the outcome rule, KS never ranks (AC7.3, PhAIL App. I),
and summaries of the PhAIL runs (AC7.1, AC7.2). Runnable as a script or with pytest (``test_*``).

Grounding: PoC §4.4 p. 10 (two-step verdict, Holm/Bonferroni across metrics), §3.2 p. 6 ("KS cannot rank on its
own"); PhAIL §3.3 p. 6 (outcomes (i)/(ii)/(iii)), App. I p. 18 (three CDFs whose sup-signs cycle A≻B≻C≻A).

    .venv/bin/python scripts/verify_verdict_s7.py          # unit checks (LABELLED SYNTHETIC data)
    .venv/bin/python -m pytest -q scripts/verify_verdict_s7.py
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
    rmst = {k: ev.rmst(arms[k]["x"], 70.0) for k in arms}
    assert rmst["B"] < rmst["C"] < rmst["A"]  # the scalar is transitive
    v = vd.compare_policies(arms["A"], arms["B"], None, 70.0, metrics=("RMST",), n_boot=400, n_boot_ks=400, seed=0, names=("A", "B"))
    assert set(v["gate"]) == {"test", "D", "p", "D_cells", "n_boot", "fires"} and v["gate"]["D"] >= 0  # unsigned KS
    assert v["gate"]["fires"] and v["direction"] == "B"  # KS sup-sign says A; the verdict's direction is RMST's
    return signs, rmst, v


def main() -> int:
    test_holm_hand_example()
    print("AC7.x holm: hand example [0.01, 0.04, 0.03, 0.005] -> " + str(vd.holm([0.01, 0.04, 0.03, 0.005])) + "; single p unchanged; capped at 1: pass")
    test_outcome_rule()
    print("AC7.x outcome rule: (iii) without the gate, (i) A, (i) B, (ii), gate_only on labelled-synthetic metric sets: pass")
    signs, rmst, v = test_ks_sign_cycles_but_verdict_direction_comes_from_rmst()
    print("AC7.3 SYNTHETIC App. I CDFs: KS sup-sign winners " + "; ".join(f"{a} vs {b}: sup {d:.2f} -> {a if s == 'first' else b}"
                                                                     for (a, b), (d, s) in signs.items())
          + " (a cycle A≻B≻C≻A)")
    print(f"AC7.3 RMST(70 s): A {rmst['A']:.2f}, B {rmst['B']:.2f}, C {rmst['C']:.2f} (transitive: B < C < A)")
    print(f"AC7.3 verdict A vs B: gate D̄ {v['gate']['D']:.2f} (unsigned) p {v['gate']['p']:.4f} -> outcome ({v['outcome']}), direction "
          f"{v['direction']} from RMST Δ {v['metrics']['RMST']['diff']:+.2f} s [{v['metrics']['RMST']['ci95'][0]:+.2f}, "
          f"{v['metrics']['RMST']['ci95'][1]:+.2f}], while the KS sup-sign favours A")
    json.dump({"sup_signs": {f"{a}-{b}": s for (a, b), s in signs.items()}, "rmst70": rmst, "verdict_A_B": v},
              open(ART / "s7_ac7_3_intransitive.json", "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
