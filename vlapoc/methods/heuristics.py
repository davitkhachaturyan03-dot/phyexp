"""Cheap heuristic rankers (screening baselines), run through the method-facing loader (ground-truth feature S5).

Grounding
---------
- PoC §5 (p. 11-12), row "Intrinsic quality | Mutual information, redundancy, smoothness | ... | DemInf, SCIZOR,
  heuristics": heuristics are a legitimate baseline family for a screening method to beat.
- PoC §2.2 (p. 4): "basic action statistics (smoothness, idle time, gripper toggles)": H2-H4 read exactly those
  catalog columns.
- RoboDrop Table 1 (p. 7), baseline "Length ... ranks demonstrations by their duration": H1.
- Calibra (a detector of "frozen camera frames", prior art in the plan) inspires H5, SCIZOR RQ4 "Pause" grounds
  the reading of H2/H5.
- PoC §2.1 item 4 (p. 3): a method must "flag exactly those" on its own, so every ranker here reads data only via
  ``vlapoc.method_data.open_for_method`` (owner decision 3) and never imports ``vlapoc.groundtruth``.

Each heuristic gives one score per candidate episode (``episode_id``, ``content_hash``, ``score``); a HIGH score means
"more suspicious" (direction fixed a priori, before any result was seen):

- H1 ``length_steps``: ``catalog.length_steps`` (long = suspicious).
- H2 ``idle_steps``: ``catalog.idle_steps`` (many all-zero arm actions).
- H3 ``gripper_toggles``: ``catalog.gripper_toggles``.
- H4 ``smoothness``: ``catalog.smoothness`` is the mean L2 step change of the arm action ("lower is smoother", PoC §2.2),
  so a high value (rough) is suspicious.
- H5 ``duplicate_frame_pairs``: consecutive frame pairs whose image is byte-identical, summed over both cameras
  (clean v0 has none: measured, 0 of 199,340 pairs). A frozen-frame detector.
- H6 ``instruction_mismatch``: 1 if the episode's ``language_instruction`` differs from the most common instruction
  of its ``task_name`` in the same subset, else 0 (the provenance cross-check of plan design D1; it uses only
  method-facing columns).

CLI: ``.venv/bin/python -m vlapoc.methods.heuristics --subset v2/candidate --out data/eval/groundtruth/heuristics/``
writes ``<name>.parquet`` per heuristic and records one ``screen`` stage run each (``--no-track`` skips MLflow).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vlapoc.method_data import MethodView, open_for_method

HEURISTICS = ("length_steps", "idle_steps", "gripper_toggles", "smoothness", "duplicate_frame_pairs", "instruction_mismatch")


def duplicate_frame_pairs(view: MethodView) -> pd.Series:
    """H5: per episode, consecutive byte-identical image pairs over both cameras (index: episode_id)."""
    out = {}
    for eid, ep in view.iter_episodes():
        out[eid] = sum(int((ep[k][1:] == ep[k][:-1]).all(axis=(1, 2, 3)).sum()) for k in ("image", "image2"))
    return pd.Series(out)


def instruction_mismatch(view: MethodView) -> pd.Series:
    """H6: 1 where the instruction is not the modal instruction of the episode's ``task_name`` in this subset."""
    c = view.catalog
    mode = c.groupby("task_name")["language_instruction"].agg(lambda s: s.value_counts().index[0])
    return pd.Series((c["language_instruction"].to_numpy() != mode.loc[c["task_name"]].to_numpy()).astype(int), index=c["episode_id"])


def compute(view: MethodView, only: tuple[str, ...] = HEURISTICS) -> dict[str, pd.DataFrame]:
    """Score tables ``{name: DataFrame(episode_id, content_hash, <name>)}`` for the requested heuristics."""
    c = view.catalog
    tables = {}
    for h in only:
        if h in ("length_steps", "idle_steps", "gripper_toggles", "smoothness"):
            vals = c[h].astype(float).to_numpy()
        elif h == "duplicate_frame_pairs":
            vals = duplicate_frame_pairs(view).loc[c["episode_id"]].astype(float).to_numpy()
        elif h == "instruction_mismatch":
            vals = instruction_mismatch(view).loc[c["episode_id"]].astype(float).to_numpy()
        else:
            raise ValueError(h)
        tables[h] = pd.DataFrame({"episode_id": c["episode_id"].to_numpy(), "content_hash": c["content_hash"].to_numpy(), h: vals})
    return tables


def main(argv: list[str] | None = None) -> int:
    """Open the subset through the loader, compute H1-H6, write score tables, record ``screen`` runs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-track", action="store_true")
    ap.add_argument("--only", nargs="*", default=list(HEURISTICS))
    a = ap.parse_args(argv)
    view = open_for_method(a.subset)
    tables = compute(view, tuple(a.only))
    a.out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for h, t in tables.items():
        path = a.out / f"{h}.parquet"
        t.to_parquet(path, index=False)
        if not a.no_track:
            from vlapoc.lineage import Subset, score_table_digest
            from vlapoc.runs import stage_run

            cand = Subset.load(a.subset)
            ref_name = a.subset.split("/")[0] + "/reference"
            with stage_run("screen", f"screen-heuristic-{h}") as r:
                r.input_subset(cand, "candidate", a.subset)
                r.input_subset(Subset.load(ref_name), "reference", ref_name)
                r.params({"method": f"heuristic {h}", "direction": "highest = more suspicious", "n_rows": len(t),
                          "reads": "vlapoc.method_data.open_for_method"})
                r.output_file("screening_scores", path, kind="score_table", entity_id=score_table_digest(t, h))
                runs[h] = r.run_id
        print(f"{h}: {len(t)} rows -> {path}")
    if runs:
        (a.out / "screen_runs.json").write_text(json.dumps(runs, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
