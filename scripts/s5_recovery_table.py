"""Assemble the S5 heuristic x corruption-type recovery table from the per-heuristic ``recovery_*.json`` files
written by ``python -m vlapoc.groundtruth score`` (ground-truth feature S5, AC5.4).

    .venv/bin/python scripts/s5_recovery_table.py --dir data/eval/groundtruth/heuristics --out <artifacts>/s5_recovery_heuristics.csv

Rows: one per (heuristic, metric in {auroc, recall_at_npos}); columns: the seven corruption types (one type vs the
clean candidates) and ``pooled`` (all 91 vs the 869 clean). Random-ranker expectations are appended as rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HEURISTICS = ("length_steps", "idle_steps", "gripper_toggles", "smoothness", "duplicate_frame_pairs", "instruction_mismatch")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prior-dir", type=Path, help="recovery JSONs of the superseded version (v2), for the side-by-side time_shift rows")
    ap.add_argument("--bands-out", type=Path, help="write the AC5.8 per-type band table here (CSV)")
    a = ap.parse_args()
    rows = []
    for h in HEURISTICS:
        res = json.loads((a.dir / f"recovery_{h}.json").read_text())
        for metric in ("auroc", "recall_at_npos"):
            r = {"heuristic": h, "metric": metric, **{t: m[metric] for t, m in sorted(res["per_type"].items())}, "pooled": res["pooled"][metric]}
            rows.append(r)
    res = json.loads((a.dir / f"recovery_{HEURISTICS[0]}.json").read_text())
    rows.append({"heuristic": "random ranker (expectation)", "metric": "auroc", **{t: 0.5 for t in res["per_type"]}, "pooled": 0.5})
    rows.append({"heuristic": "random ranker (expectation)", "metric": "recall_at_npos",
                 **{t: m["random_recall_at_npos"] for t, m in res["per_type"].items()}, "pooled": res["pooled"]["random_recall_at_npos"]})
    df = pd.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(a.out, index=False)
    print(df.round(3).to_string(index=False))
    print(f"n per type: {sorted({m['n_pos'] for m in res['per_type'].values()})} injected vs {res['pooled']['n'] - res['pooled']['n_pos']} clean; "
          f"pooled {res['pooled']['n_pos']} vs {res['pooled']['n'] - res['pooled']['n_pos']}")
    # ---- AC5.8: per-type bands (inferred reporting bands, not pass thresholds) ----------------------------------
    types = sorted(res["per_type"])
    brows = []
    for t in types:
        per_h = {h: json.loads((a.dir / f"recovery_{h}.json").read_text())["per_type"][t]["auroc"] for h in HEURISTICS}
        best_h = max(per_h, key=per_h.get)
        flip_h = max(per_h, key=lambda h: max(per_h[h], 1 - per_h[h]))
        best = per_h[best_h]
        band = "(a) found, AUROC >= 0.95" if best >= 0.95 else "(b) partly found, 0.6-0.95" if best >= 0.6 else "(c) not found, < 0.6"
        brows.append({"type": t, "best_heuristic": best_h, "best_auroc_as_scored": round(best, 3), "band": band,
                      "best_direction_free": flip_h, "best_auroc_direction_free": round(max(per_h[flip_h], 1 - per_h[flip_h]), 3)})
    bdf = pd.DataFrame(brows)
    print("\nAC5.8 per-type bands (best of the six heuristics, a-priori direction; direction-free value alongside)")
    print(bdf.to_string(index=False))
    if a.bands_out:
        bdf.to_csv(a.bands_out, index=False)
    if a.prior_dir:
        rows2 = []
        for h in HEURISTICS:
            new_ = json.loads((a.dir / f"recovery_{h}.json").read_text())["per_type"]["time_shift"]
            old_ = json.loads((a.prior_dir / f"recovery_{h}.json").read_text())["per_type"]["time_shift"]
            rows2.append({"heuristic": h, "auroc_v2_idle_tail": round(old_["auroc"], 3), "auroc_v3_trim": round(new_["auroc"], 3),
                          "recall_at_13_v2": round(old_["recall_at_npos"], 3), "recall_at_13_v3": round(new_["recall_at_npos"], 3)})
        print("\ntime_shift, r1 (v2, idle tail) vs r2 (v3, trim), one type vs 869 clean")
        print(pd.DataFrame(rows2).to_string(index=False))
        pd.DataFrame(rows2).to_csv(a.out.with_name(a.out.stem + "_time_shift_v2_vs_v3.csv"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
