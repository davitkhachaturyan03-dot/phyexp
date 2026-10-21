"""Recovery of planted corruptions by an attribution table: type x harm-column AUROC / recall next to the best heuristic.

Ground-truth side (imports ``vlapoc.groundtruth``): this module reads the answer key and must never be imported by
``vlapoc.influence`` / ``vlapoc.attribution`` or any method code (owner decision 3; boundary scan in
``scripts/verify_training_attribution.py --recovery``).

Grounding
---------
- PoC §2.1 item 4 (p. 3): "Deliberately corrupt known subsets of training data ... Then check whether your attribution
  method flags exactly those. This turns 'the method says X' into 'the method recovered 90% of the injected bad episodes.'"
- PoC §4.5 (p. 11), quoted: "Metric-targeted injections. Extend the injection test with corruptions designed to hit
  specific metrics — for example, idle or hesitant segments aimed at time-based metrics, and noisy gripper release near
  placement aimed at drops. Check whether attribution assigns each corrupted group to the expected metric. These mappings
  are hypotheses the experiment should confirm."
- Ground-truth feature, limitation D (owner): a method must beat each corruption type's best simple heuristic
  (``tracker/features/2026-10-15-ground-truth/artifacts/s5_v3_r3_bands.csv``).

Hypotheses declared BEFORE any attribution score exists (training-attribution S5, committed ahead of the scoring code)
-----------------------------------------------------------------------------------------------------------------------
``EXPECTED_METRICS`` maps a corruption type to the family-W metrics whose ``harm_`` column should rank that type's
episodes highest. Only the two mappings the PoC names are declared: ``idle_padding`` -> time-based metrics (``rmst60``,
``hrt60_lin``); ``gripper_release`` -> drops (``f30``, ``censored60``: a drop ends in no success within the cap). Every
other type has no PoC-stated mapping, so none is invented here: it is ``None`` and reported descriptively. The mapping is
a hypothesis; the output reports whether each type's best column matches it, never that it was "confirmed" by a
synthetic table.
"""

from __future__ import annotations

EXPECTED_METRICS: dict[str, tuple[str, ...] | None] = {
    "idle_padding": ("rmst60", "hrt60_lin"),  # PoC §4.5: "idle or hesitant segments aimed at time-based metrics"
    "gripper_release": ("f30", "censored60"),  # PoC §4.5: "noisy gripper release near placement aimed at drops"
    "action_noise": None,
    "detour": None,
    "dropped_frames": None,
    "instruction_swap": None,
    "time_shift": None,
}

# ---------------------------------------------------------------------------------------------
# scoring (added after the hypotheses above were committed)
# ---------------------------------------------------------------------------------------------

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

from vlapoc.libero_convert import REPO_ROOT  # noqa: E402

GT_ARTIFACTS = REPO_ROOT / "tracker" / "features" / "2026-10-15-ground-truth" / "artifacts"
BANDS_CSV = GT_ARTIFACTS / "s5_v3_r3_bands.csv"
HEURISTICS_CSV = GT_ARTIFACTS / "s5_v3_r3_recovery_heuristics.csv"


def recovery_matrix(version: str, score_table: pd.DataFrame, subset: str = "candidate", lineage_root: Path | None = None,
                    bands: Path = BANDS_CSV, heuristics: Path = HEURISTICS_CSV) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Score every ``harm_`` column (direction ``highest``: a large harm score = more likely corrupted) against the answer key.

    Returns ``(matrix, summary, skipped)``. ``matrix`` has one row per (corruption type or ``pooled``, harm column):
    ``auroc`` and ``recall_at_npos`` (k = number injected) from ``groundtruth.score_recovery``, the type's best simple
    heuristic with its AUROC and recall from the ground-truth feature's artifacts, and ``auroc_minus_heuristic``.
    ``summary`` has one row per type: its best harm column (by AUROC), the hypothesis from ``EXPECTED_METRICS``,
    ``matches_hypothesis`` (``None`` where the PoC states no mapping) and ``beats_heuristic``. Columns with a NaN are
    skipped and named in ``skipped``. ``LineageError`` if the table does not key exactly the subset (``score_recovery``).
    """
    from vlapoc.groundtruth import score_recovery

    b = pd.read_csv(bands).set_index("type")
    h = pd.read_csv(heuristics)
    cols = [c for c in score_table.columns if c.startswith("harm_")]
    skipped = [c for c in cols if score_table[c].isna().any()]
    rows = []
    for c in [c for c in cols if c not in skipped]:
        res = score_recovery(version, score_table, c, "highest", subset, lineage_root)
        for ty, m in {**res["per_type"], "pooled": res["pooled"]}.items():
            row = {"type": ty, "column": c, "auroc": m["auroc"], "recall_at_npos": m["recall_at_npos"], "n_pos": m["n_pos"], "n": m["n"]}
            if ty in b.index:
                hn = b.loc[ty, "best_heuristic"]
                hr = h[h["heuristic"] == hn].set_index("metric")
                row |= {"best_heuristic": hn, "best_heuristic_auroc": float(b.loc[ty, "best_auroc_as_scored"]),
                        "best_heuristic_recall_at_npos": float(hr.loc["recall_at_npos", ty]),
                        "auroc_minus_heuristic": m["auroc"] - float(b.loc[ty, "best_auroc_as_scored"])}
            rows.append(row)
    matrix = pd.DataFrame(rows)
    out = []
    for ty, g in matrix[matrix["type"] != "pooled"].groupby("type"):
        best = g.sort_values("auroc", ascending=False).iloc[0]
        base = best["column"].removeprefix("harm_").split("__")[0]
        exp = EXPECTED_METRICS.get(ty)
        out.append({"type": ty, "best_column": best["column"], "best_auroc": best["auroc"], "best_recall_at_npos": best["recall_at_npos"],
                    "best_heuristic": best["best_heuristic"], "best_heuristic_auroc": best["best_heuristic_auroc"],
                    "beats_heuristic": bool(best["auroc"] > best["best_heuristic_auroc"]),
                    "expected_metrics": ",".join(exp) if exp else "", "matches_hypothesis": (base in exp) if exp else None})
    return matrix, pd.DataFrame(out), skipped


def recovery_run(score_run_id: str, version: str, name: str, subset: str = "candidate", out_dir: Path | None = None,
                 lineage_root: Path | None = None, chain: str | None = None, allow_dirty: bool = False,
                 synthetic: bool = False) -> dict:
    """Stage ``score`` for a whole attribution table: ``recovery_matrix`` of an attribute run's scores against ``version``'s
    answer key, written as ``recovery_matrix.csv`` / ``recovery_summary.csv`` and recorded as one run that consumes the version
    and the score table. Ground-truth side; run it only after the attribution exists. ``synthetic`` labels the outputs.
    PoC §2.1 item 4 p. 3, §4.5 p. 11."""
    from vlapoc.attribution import _artifact_table
    from vlapoc.lineage import rel, score_table_digest
    from vlapoc.runs import client, resolve_version, stage_run

    run = client(lineage_root).get_run(score_run_id)
    table = _artifact_table(run, "attribution_scores", lineage_root)
    matrix, summary, skipped = recovery_matrix(version, table, subset, lineage_root)  # refuses a table keyed to another version
    out_dir = Path(out_dir) if out_dir else Path("data") / "attribution" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    mpath, spath = out_dir / "recovery_matrix.csv", out_dir / "recovery_summary.csv"
    matrix.to_csv(mpath, index=False)
    summary.to_csv(spath, index=False)
    v = resolve_version(version, lineage_root)
    with stage_run("score", name, lineage_root, chain=chain, allow_dirty=allow_dirty) as r:
        r.input_version(v["dataset_version_id"], "scored_version")
        r.input_entity("score_table", run.data.tags["output.attribution_scores.sha256"], "scores")
        r.params({"score_run_id": score_run_id, "version": version, "subset": subset, "direction": "highest",
                  "n_columns": int(matrix["column"].nunique()), "skipped_nan_columns": skipped, "synthetic": synthetic,
                  "expected_metrics": {k: list(x) for k, x in EXPECTED_METRICS.items() if x}})
        r.output_file("recovery_matrix", mpath, kind="file")
        r.output_file("recovery_summary", spath, kind="file")
        return {"run_id": r.run_id, "matrix": rel(mpath), "summary": rel(spath), "n_rows": len(matrix), "skipped": skipped}
