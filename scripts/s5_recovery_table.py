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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
