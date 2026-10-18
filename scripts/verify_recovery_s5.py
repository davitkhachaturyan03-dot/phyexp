"""Verify ground-truth S5 scoring (AC5.1, AC5.2 structure, AC5.3, AC5.5) on fixtures and the real v1/v3.

Grounding: PoC §2.1 item 4 p.3 ("recovered 90% of the injected bad episodes"), §2.5 p.5; RoboDrop §5.1 p.7
(AUROC, best balanced accuracy); evaluation-layer S0 (the v1 positional leak, ``fence.json``).

    .venv/bin/python scripts/verify_recovery_s5.py

Reads the real store (v0, v1, v3 episode tables and labels); writes only ``artifacts/s5_verify.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vlapoc.groundtruth import leak_check, load_answer_key, recovery_metrics, score_recovery  # noqa: E402
from vlapoc.lineage import LineageError, Subset  # noqa: E402

FEATURE_DIR = REPO_ROOT / "tracker" / "features" / "2026-10-15-ground-truth"


def fixtures() -> tuple[list[dict], bool]:
    """AC5.1: each metric against a hand-computed value (derivations in the comments)."""
    y10 = [1, 1] + [0] * 8
    cases = [
        # perfect: positives hold the two top scores. recall@2 = 1, recall@4 = 1, AUROC 1, AP 1, bal acc 1
        ("perfect", list(range(10, 0, -1)), y10, "highest",
         dict(recall_at_npos=1.0, precision_at_npos=1.0, recall_at_2npos=1.0, auroc=1.0, average_precision=1.0, best_balanced_accuracy=1.0)),
        # same ranking written as 'lowest'
        ("perfect, direction lowest", list(range(1, 11)), y10, "lowest",
         dict(recall_at_npos=1.0, auroc=1.0, average_precision=1.0)),
        # reversed: positives at ranks 9 and 10. recall@2 = recall@4 = 0, AUROC 0, AP = (1/9 + 2/10) / 2, bal acc 0.5
        ("reversed", list(range(1, 11)), y10, "highest",
         dict(recall_at_npos=0.0, recall_at_2npos=0.0, auroc=0.0, average_precision=(1 / 9 + 2 / 10) / 2, best_balanced_accuracy=0.5)),
        # constant: fractional ties give recall@k = k/n = 0.2 (k=2) and 0.4 (k=4); AUROC 0.5; AP = n_pos/n = 0.2
        ("constant", [1.0] * 10, y10, "highest",
         dict(recall_at_npos=0.2, precision_at_npos=0.2, recall_at_2npos=0.4, auroc=0.5, average_precision=0.2, best_balanced_accuracy=0.5)),
        # 3-tie: scores [3,2,2,2,1,1], positives are items 0 and 1. top-2 = {item 0} + 1 of the 3 tied (1 positive among them):
        # expected positives 1 + 1/3, so recall@2 = precision@2 = 2/3; k=4 covers the whole tie group: recall 1.
        # AUROC = (4 + (0.5 + 0.5 + 1 + 1)) / 8 = 7/8; roc thresholds give (TPR+TNR)/2 = 0.75, 0.75, 0.5; AP = 0.5*1 + 0.5*0.5
        ("3-tie", [3, 2, 2, 2, 1, 1], [1, 1, 0, 0, 0, 0], "highest",
         dict(recall_at_npos=2 / 3, precision_at_npos=2 / 3, recall_at_2npos=1.0, auroc=7 / 8, best_balanced_accuracy=0.75, average_precision=0.75)),
    ]
    rows, ok = [], True
    for name, s, y, d, exp in cases:
        got = recovery_metrics(s, y, d)
        for k, v in exp.items():
            good = abs(got[k] - v) < 1e-12
            ok &= good
            rows.append({"case": name, "metric": k, "expected": round(v, 6), "got": round(got[k], 6), "ok": good})
    return rows, bool(ok)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v3")
    ap.add_argument("--prefix", default="s5_v3")
    a = ap.parse_args()
    V = a.version
    out: dict = {}
    ok = True
    rows, ok1 = fixtures()
    print("AC5.1 fixtures (hand-computed vs recovery_metrics)")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"AC5.1 all {len(rows)} comparisons equal: {ok1}")
    out["AC5.1"] = {"rows": rows, "pass": ok1}
    ok &= ok1

    # ---- AC5.2: per type and pooled, with random baselines (random scores, seeded, 200 draws) ----------------
    cand = Subset.load(f"{V}/candidate")
    keys = cand.keys()
    rng = np.random.default_rng(0)
    pooled_auc, pooled_rec, type_auc = [], [], {}
    for _ in range(200):
        t = keys.copy()
        t["r"] = rng.random(len(t))
        res = score_recovery(V, t, "r")
        pooled_auc.append(res["pooled"]["auroc"])
        pooled_rec.append(res["pooled"]["recall_at_npos"])
        for ty, m in res["per_type"].items():
            type_auc.setdefault(ty, []).append(m["auroc"])
    ex = res
    ok2 = abs(np.mean(pooled_auc) - 0.5) < 0.01 and abs(np.mean(pooled_rec) - 91 / 960) < 0.01 \
        and res["pooled"]["n"] == 960 and res["pooled"]["n_pos"] == 91 \
        and all(m["n"] == 882 and m["n_pos"] == 13 for m in res["per_type"].values())
    print(f"\nAC5.2 structure: pooled n={ex['pooled']['n']} n_pos={ex['pooled']['n_pos']}; per-type n={sorted({m['n'] for m in ex['per_type'].values()})} "
          f"n_pos={sorted({m['n_pos'] for m in ex['per_type'].values()})}; random_recall_at_npos pooled {ex['pooled']['random_recall_at_npos']:.4f}, "
          f"per type {ex['per_type']['detour']['random_recall_at_npos']:.4f}; random_auroc {ex['pooled']['random_auroc']}")
    print(f"AC5.2 200 random rankers: mean pooled AUROC {np.mean(pooled_auc):.4f} (std {np.std(pooled_auc):.4f}), mean recall@k {np.mean(pooled_rec):.4f} "
          f"(expected {91 / 960:.4f}); mean per-type AUROC {{{', '.join(f'{k}: {np.mean(v):.3f}' for k, v in type_auc.items())}}}")
    out["AC5.2"] = {"pooled_n": ex["pooled"]["n"], "per_type_n": sorted({m["n"] for m in ex["per_type"].values()}),
                    "random_mean_auroc": float(np.mean(pooled_auc)), "random_mean_recall": float(np.mean(pooled_rec)), "pass": bool(ok2)}
    ok &= bool(ok2)

    # ---- AC5.3: key safety ---------------------------------------------------------------------------------
    v0k = Subset.load("v0/candidate").keys().copy()
    v0k["r"] = 0.5
    lab = load_answer_key(V)
    try:
        score_recovery(V, v0k, "r")
        m1 = None
    except LineageError as e:
        m1 = str(e)
    miss = keys.iloc[1:].copy()
    miss["r"] = 0.5
    try:
        score_recovery(V, miss, "r")
        m2 = None
    except LineageError as e:
        m2 = str(e)
    nchg = int((v0k.set_index("episode_id")["content_hash"] != keys.set_index("episode_id")["content_hash"].loc[v0k["episode_id"]]).sum())
    print(f"\nAC5.3 v0-keyed table ({nchg} of 960 hashes differ from {V}'s, the labelled ones): rejected: {m1 is not None}: {m1}")
    print(f"AC5.3 table missing one candidate: rejected: {m2 is not None}: {m2}")
    ok3 = m1 is not None and m2 is not None and nchg == len(lab)
    out["AC5.3"] = {"v0_keyed_rejected": m1, "missing_rejected": m2, "hashes_differing": nchg, "pass": ok3}
    ok &= ok3

    # ---- AC5.5: positional leak, ground-truth side -----------------------------------------------------------
    l1, l2 = leak_check("v1"), leak_check(V)
    r1, r2 = l1["pooled"], l2["pooled"]
    print("\nAC5.5 leak check (ranking by episode_index; ground-truth side, not a method)")
    print(pd.DataFrame([{"version": "v1", "n_injected": r1["n_pos"], "recall_at_npos": r1["recall_at_npos"], "auroc": r1["auroc"],
                         "random_recall": r1["random_recall_at_npos"]},
                        {"version": V, "n_injected": r2["n_pos"], "recall_at_npos": r2["recall_at_npos"], "auroc": r2["auroc"],
                         "random_recall": r2["random_recall_at_npos"]}]).to_string(index=False))
    ok5 = r1["recall_at_npos"] == 1.0 and r1["n_pos"] == 100 and abs(r2["recall_at_npos"] - r2["random_recall_at_npos"]) < 0.12
    out["AC5.5"] = {"v1": r1, V: r2, "pass": ok5}
    ok &= bool(ok5)
    out["pass"] = bool(ok)
    (FEATURE_DIR / "artifacts" / f"{a.prefix}_verify.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nall S5 verifier checks pass: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
