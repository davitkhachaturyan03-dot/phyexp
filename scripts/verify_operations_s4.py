"""Verify evaluation-layer S4 operation rule on LABELLED-SYNTHETIC traces (AC4.2) and the single-implementation
rule (AC4.5). No simulator needed: the traces are hand-built boolean arrays.

Grounding: PoC §3.5 A p. 7 ("for long-horizon LIBERO-10 tasks, per sub-task"), §3.4 p. 7 ("Operation-level
(T, event) pairs"); PhAIL §3.1 p. 4 (T pairs, censored tail).

    .venv/bin/python scripts/verify_operations_s4.py     # writes artifacts/s4_verify_operations.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vlapoc.libero_goals import operations_from_trace  # noqa: E402

ART = REPO_ROOT / "tracker" / "features" / "2026-10-01-evaluation-layer" / "artifacts"
FPS = 20


def trace(T: int, on: dict[int, list[tuple[int, int]]], C: int) -> np.ndarray:
    """(T, C) bool with conjunct c true on each half-open [a, b) interval in ``on[c]``."""
    x = np.zeros((T, C), dtype=bool)
    for c, spans in on.items():
        for a, b in spans:
            x[a:b, c] = True
    return x


def case(name: str, truth: np.ndarray, tau_end: int | None, expect_pairs: list[tuple[float, int]], expect_n: int) -> dict:
    succ = truth.all(axis=1)
    r = operations_from_trace(truth, succ, tau_end, labels=[f"g{c}" for c in range(truth.shape[1])], key=name)
    got = [(round(t, 9), e) for t, e in r["episode"].pairs]
    ok = got == [(round(t, 9), e) for t, e in expect_pairs] and r["n_ops"] == expect_n
    print(f"AC4.2 SYNTHETIC {name}: n_ops {r['n_ops']} (maintained {r['maintained']}), t_ops {r['t_ops']} steps, "
          f"pairs {got} -> {'PASS' if ok else 'FAIL (expected ' + str(expect_pairs) + ')'}")
    return {"name": name, "n_ops": r["n_ops"], "t_ops": r["t_ops"], "pairs": got, "expected": expect_pairs, "pass": ok,
            "op_conjuncts": r["op_conjuncts"], "censored": r["censored"]}


def main() -> int:
    out = {"data": "LABELLED-SYNTHETIC boolean traces; 20 steps = 1 s"}
    res = [
        # two conjuncts, g0 first then g1 (stays true): ops at 40 and 100 -> T = 2.0 s, 3.0 s
        case("two conjuncts, g0 then g1", trace(121, {0: [(40, 121)], 1: [(100, 121)]}, 2), None, [(2.0, 1), (3.0, 1)], 2),
        # the same in the other order: the count rule gives identical times
        case("two conjuncts, g1 then g0", trace(121, {1: [(40, 121)], 0: [(100, 121)]}, 2), None, [(2.0, 1), (3.0, 1)], 2),
        # g0 true 40-60, off, back at 80; g1 at 100: op1 at 40 (first time count >= 1), op2 at 100
        case("conjunct toggles off and on", trace(121, {0: [(40, 60), (80, 121)], 1: [(100, 121)]}, 2), None,
             [(2.0, 1), (3.0, 1)], 2),
        # maintained g2 (true at t0) broken 50-110; ops g0 at 40, g1 at 60 -> count 2 at 60 but success only at 110
        case("maintained conjunct broken mid-rollout", trace(121, {0: [(40, 121)], 1: [(60, 121)], 2: [(0, 50), (110, 121)]}, 3),
             None, [(2.0, 1), (3.5, 1)], 2),
        # both operations become true at step 80: T1 = 4.0 s, T2 = 0
        case("simultaneous completion", trace(121, {0: [(80, 121)], 1: [(80, 121)]}, 2), None, [(4.0, 1), (0.0, 1)], 2),
        # timeout at τ_end = 1200 steps (60 s) after 1 of 2 ops (g0 at 300): (15 s, event) + censored (45 s)
        case("timeout after 1 of 2 ops", trace(1201, {0: [(300, 1201)]}, 2), 1200, [(15.0, 1), (45.0, 0)], 2),
        # nothing done by τ_end: one censored pair of length τ_end = 60 s
        case("zero ops done", trace(1201, {}, 2), 1200, [(60.0, 0)], 2),
    ]
    out["AC4.2"] = {"cases": res, "pass": all(r["pass"] for r in res)}
    print(f"AC4.2 {sum(r['pass'] for r in res)}/{len(res)} synthetic cases pass")
    # AC4.5: one implementation, imported by the human-reference script and the rollout harness
    defs = {str(p.relative_to(REPO_ROOT)): len(re.findall(r"^\s*def operations_from_trace\(", p.read_text(), re.M))
            for p in [*sorted((REPO_ROOT / "vlapoc").glob("*.py")), *sorted((REPO_ROOT / "scripts").glob("*.py"))]}
    n_defs = sum(defs.values())
    users = {f: bool(re.search(r"operations_from_trace", (REPO_ROOT / f).read_text())) for f in
             ("scripts/human_reference.py", "vlapoc/rollouts.py") if (REPO_ROOT / f).is_file()}
    print(f"AC4.5 definitions of operations_from_trace in vlapoc/ + scripts/: {n_defs} "
          f"({[f for f, n in defs.items() if n]}); callers: {users}")
    out["AC4.5"] = {"n_definitions": n_defs, "defined_in": [f for f, n in defs.items() if n], "callers": users,
                    "pass": n_defs == 1 and all(users.values()) and len(users) == 2}
    (ART / "s4_verify_operations.json").write_text(json.dumps(out, indent=1))
    return 0 if out["AC4.2"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
