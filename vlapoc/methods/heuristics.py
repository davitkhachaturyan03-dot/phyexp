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

Round 3 (plan r3, analysis-r2; both label-free, task-normalised, computed from method-facing data only):

- H7 ``length_vs_task``: ``length_steps`` divided by the median ``length_steps`` of the episode's ``task_name`` in the same
  subset. Scored in BOTH directions (file ``length_vs_task.parquet`` with direction ``highest``, and the same scores as
  ``length_vs_task_lowest.parquet`` with direction ``lowest``): a trim makes episodes shorter than their task's norm, a
  detour or padding longer. Raw length (H1) mixes tasks of very different length; this removes that.
- H8 ``post_release_share_vs_task``: the share of the episode after the final gripper release,
  (T - i) / T for the last step i with gripper > 0 followed by gripper < 0 (0 if none), minus the median of that share
  over the episode's ``task_name`` (high = suspicious). A premature release moves the final release earlier, so
  the share after it grows. It reads only the action's gripper dimension through ``iter_episodes``.
These two were added after round 2 showed that task-free statistics left ``time_shift``, ``detour`` and ``gripper_release``
looking undetected; they are baselines to beat, not claims about any method (plan AC5.9).

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

HEURISTICS = ("length_steps", "idle_steps", "gripper_toggles", "smoothness", "duplicate_frame_pairs", "instruction_mismatch",
              "length_vs_task", "length_vs_task_lowest", "post_release_share_vs_task")
#: ranking direction fixed before scoring ("highest" = high score is more suspicious)
DIRECTIONS = {h: "highest" for h in HEURISTICS} | {"length_vs_task_lowest": "lowest"}


def scan_episodes(view: MethodView) -> pd.DataFrame:
    """One pass over the stored episodes (index: episode_id): ``duplicate_frame_pairs`` (H5, consecutive byte-identical
    image pairs over both cameras) and ``post_release_share`` ((T - i) / T for the last step i with gripper > 0 followed
    by gripper < 0, 0 if none; the raw input of H8)."""
    rows = {}
    for eid, ep in view.iter_episodes():
        g = ep["action"][:, 6]
        rel = np.where((g[:-1] > 0) & (g[1:] < 0))[0]
        last = rel[-1] if len(rel) else len(g)
        rows[eid] = {"duplicate_frame_pairs": sum(int((ep[k][1:] == ep[k][:-1]).all(axis=(1, 2, 3)).sum()) for k in ("image", "image2")),
                     "post_release_share": (len(g) - last) / len(g)}
    return pd.DataFrame(rows).T


def minus_task_median(values: pd.Series, task: pd.Series) -> pd.Series:
    """``values`` minus the median of ``values`` within each ``task`` (label-free normalisation)."""
    return values - values.groupby(task.to_numpy()).transform("median")


def instruction_mismatch(view: MethodView) -> pd.Series:
    """H6: 1 where the instruction is not the modal instruction of the episode's ``task_name`` in this subset."""
    c = view.catalog
    mode = c.groupby("task_name")["language_instruction"].agg(lambda s: s.value_counts().index[0])
    return pd.Series((c["language_instruction"].to_numpy() != mode.loc[c["task_name"]].to_numpy()).astype(int), index=c["episode_id"])


def compute(view: MethodView, only: tuple[str, ...] = HEURISTICS) -> dict[str, pd.DataFrame]:
    """Score tables ``{name: DataFrame(episode_id, content_hash, <column>)}`` for the requested heuristics. The column
    of ``length_vs_task_lowest`` is named ``length_vs_task`` (same scores as H7; only the direction differs)."""
    c = view.catalog
    ids = c["episode_id"]
    scan = scan_episodes(view) if {"duplicate_frame_pairs", "post_release_share_vs_task"} & set(only) else None
    tables = {}
    for h in only:
        col = "length_vs_task" if h == "length_vs_task_lowest" else h
        if h in ("length_steps", "idle_steps", "gripper_toggles", "smoothness"):
            vals = c[h].astype(float).to_numpy()
        elif h == "duplicate_frame_pairs":
            vals = scan.loc[ids, h].astype(float).to_numpy()
        elif h == "instruction_mismatch":
            vals = instruction_mismatch(view).loc[ids].astype(float).to_numpy()
        elif h in ("length_vs_task", "length_vs_task_lowest"):
            ln = c["length_steps"].astype(float)
            vals = (ln / ln.groupby(c["task_name"].to_numpy()).transform("median")).to_numpy()
        elif h == "post_release_share_vs_task":
            share = pd.Series(scan.loc[ids, "post_release_share"].astype(float).to_numpy(), index=c.index)
            vals = minus_task_median(share, c["task_name"]).to_numpy()
        else:
            raise ValueError(h)
        tables[h] = pd.DataFrame({"episode_id": ids.to_numpy(), "content_hash": c["content_hash"].to_numpy(), col: vals})
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
        col = [x for x in t.columns if x not in ("episode_id", "content_hash")][0]
        if not a.no_track:
            from vlapoc.lineage import Subset, score_table_digest
            from vlapoc.runs import stage_run

            cand = Subset.load(a.subset)
            ref_name = a.subset.split("/")[0] + "/reference"
            with stage_run("screen", f"screen-heuristic-{h}") as r:
                r.input_subset(cand, "candidate", a.subset)
                r.input_subset(Subset.load(ref_name), "reference", ref_name)
                r.params({"method": f"heuristic {h}", "direction": f"{DIRECTIONS[h]} = more suspicious", "n_rows": len(t),
                          "reads": "vlapoc.method_data.open_for_method"})
                r.output_file("screening_scores", path, kind="score_table", entity_id=score_table_digest(t, col))
                runs[h] = r.run_id
        print(f"{h}: {len(t)} rows -> {path}")
    (a.out / "directions.json").write_text(json.dumps({h: {"column": [x for x in t.columns if x not in ("episode_id", "content_hash")][0],
                                                            "direction": DIRECTIONS[h]} for h, t in tables.items()}, indent=1))
    if runs:
        (a.out / "screen_runs.json").write_text(json.dumps(runs, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
