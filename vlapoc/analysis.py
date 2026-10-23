"""Metric-resolved views of an attribution table: the four groups, aggregation by catalog column, the heatmap and the
throughput-vs-reliability scatter, and the ``analyze`` stage that records them.

Where this sits (docs/dataflow.md, Step 3 "Attribution"): after ``vlapoc.attribution`` has written a score table (one attribute or
ensemble run), ``scripts/stage.py analyze`` calls ``analyze_run`` here to turn it into the PoC's episode x metric views. Nothing
downstream consumes these labels; they are for reading. Reading order: ``metric_groups``, ``aggregate_by``, the two figure
functions, then ``analyze_run``.

Grounding
---------
- PoC §4.2 (pp. 9-10), quoted: "Results are reported as an episode × metric matrix." The four groups of episodes:
  "Consistent helpers / Consistent harmers / Trade-off episodes / Metric-specific".
- PoC §4.5 (p. 11), "Visualisation": "An episode × metric heatmap, and a scatter of throughput influence against
  reliability influence with the four groups as quadrants."
- PoC §1 (p. 3): "Per-episode scores become most valuable when aggregated by metadata"; §2.2 (p. 4) and §9.2 (p. 19):
  aggregation by the catalog's columns (source, operator, task, scene, ...), ``catalog.CATALOG_COLUMNS``.
- PoC §4.4 (pp. 10-11): "Label every claim with its metric and time cap τ" (axis labels here come from the declared
  functionals, ``vlapoc.attribution.column_declarations``) and the multiple-comparison caution: "correct for it (e.g.
  Bonferroni or Holm across metrics) before labeling an episode as a trade-off".

Status of the group labels: DESCRIPTIVE, UNVERIFIED (divergence from §4.4, recorded)
------------------------------------------------------------------------------------
``metric_groups`` thresholds the ensemble's z-scored ``harm_`` columns at |z| >= ``z`` (default 1; an inferred, declared
parameter, not from the PoC). §4.4 asks for a multiplicity correction before an episode is called a trade-off, but a
per-episode significance test is not available from 2-3 seeds. So the labels describe where an episode sits in the table;
the verified claim lives at the intervention level (``vlapoc.step4``: curated vs random and the head-to-head, with Holm).
The ``analyze`` stage records this status in its params.

Method side: this module imports nothing from ``vlapoc.groundtruth``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DIVERGENCE_NOTE = ("owner-approved 2026-10-22 (\"Keep 2-of-5, documented\"): consistent helper/harmer = at least 2 non-zero calls, all of one sign (not 'every metric' as PoC §4.2 p. 10 reads); "
                   "a judgement call (inferred, not measured). Risk: two near-redundant reliability readouts (harm_f30 + harm_censored60) "
                   "can alone make an episode 'consistent'")
GROUPS = ("consistent_helper", "consistent_harmer", "trade_off", "metric_specific", "neutral")
DEFAULT_METRICS = ("f30", "rmst60", "auc_expected", "hrt60_lin", "censored60")  # family-W returns, one harm column each
STATUS = "descriptive, unverified (no per-episode multiplicity-corrected test from 2-3 seeds; PoC §4.4)"
GROUP_COLORS = {"consistent_helper": "#2a9d8f", "consistent_harmer": "#d1495b", "trade_off": "#edae49",
                "metric_specific": "#6c8ebf", "neutral": "#c8c8c8"}


def metric_groups(table: pd.DataFrame, columns: list[str], z: float = 1.0) -> pd.Series:
    """Label each episode (row) from its ``harm_`` columns (positive = harmful), z-scored across episodes first.

    Per metric the call is +1 (harmful, z >= ``z``), -1 (helpful, z <= -``z``) or 0. Then: no non-zero call ->
    ``neutral``; exactly one -> ``metric_specific``; all non-zero calls +1 -> ``consistent_harmer``; all -1 ->
    ``consistent_helper``; both signs -> ``trade_off``.

    Named divergence from PoC §4.2 (p. 10): the PoC's "consistent helpers/harmers" read as helping/harming *every* metric;
    here a consistent group needs at least two non-zero calls all of one sign (>= 2 of the 5 metrics), not all five. The
    >= 2 rule is a judgement call (inferred; nothing was measured to choose it) and the owner may prefer the strict all-metrics
    rule. Its specific risk: the five family-W returns are not independent, and two near-redundant reliability readouts
    (``harm_f30`` and ``harm_censored60``) can alone make an episode a "consistent harmer" while the throughput metrics are flat.
    It is recorded in ``DIVERGENCE_NOTE`` and in the analyze run's params. Descriptive and unverified (module docstring). ``z`` is a declared
    parameter (inferred default 1).
    """
    zs = pd.DataFrame({c: (table[c] - table[c].mean()) / table[c].std(ddof=0) for c in columns})
    calls = np.where(zs >= z, 1, np.where(zs <= -z, -1, 0))
    n_pos, n_neg = (calls == 1).sum(1), (calls == -1).sum(1)
    out = np.full(len(table), "neutral", dtype=object)
    out[(n_pos + n_neg) == 1] = "metric_specific"
    out[(n_pos >= 2) & (n_neg == 0)] = "consistent_harmer"
    out[(n_neg >= 2) & (n_pos == 0)] = "consistent_helper"
    out[(n_pos >= 1) & (n_neg >= 1)] = "trade_off"
    return pd.Series(out, index=table.index, name="group")


def aggregate_by(table: pd.DataFrame, catalog_column: str, columns: list[str] | None = None, episodes: pd.DataFrame | None = None,
                 n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Mean score per group of a catalog column: ``n``, and for each score column ``<c>__mean`` with a percentile-bootstrap 95%
    CI over the group's episodes (``__lo``, ``__hi``). ``catalog_column`` must be in ``catalog.CATALOG_COLUMNS`` and not a
    per-episode identifier (``lineage.GROUP_COLUMNS_EXCLUDED``); if the table lacks it, pass ``episodes`` (catalog rows,
    e.g. ``Subset.episodes()``) to join on ``episode_id``. The groups' ``n`` sum to the table's rows (PoC §1 p. 3, §2.2 p. 4).
    """
    from vlapoc.catalog import CATALOG_COLUMNS
    from vlapoc.lineage import GROUP_COLUMNS_EXCLUDED

    if catalog_column not in CATALOG_COLUMNS:
        raise ValueError(f"{catalog_column!r} is not a catalog column; choose one of {CATALOG_COLUMNS}")
    if catalog_column in GROUP_COLUMNS_EXCLUDED:
        raise ValueError(f"{catalog_column!r} identifies single episodes; grouping by it is not an aggregation")
    t = table
    if catalog_column not in t.columns:
        if episodes is None:
            raise ValueError(f"the table has no {catalog_column!r}; pass episodes=<catalog rows> to join it")
        t = t.merge(episodes[["episode_id", catalog_column]], on="episode_id", how="left", validate="one_to_one")
    cols = columns or [c for c in t.columns if c.startswith("harm_") and "__" not in c]
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in t.groupby(catalog_column, dropna=False, sort=True):
        row = {catalog_column: key, "n": len(g)}
        for c in cols:
            v = g[c].to_numpy(float)
            boot = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1) if len(v) > 1 else np.array([v.mean()])
            row |= {f"{c}__mean": float(v.mean()), f"{c}__lo": float(np.quantile(boot, 0.025)), f"{c}__hi": float(np.quantile(boot, 0.975))}
        rows.append(row)
    return pd.DataFrame(rows)


def axis_label(column: str, decls: dict[str, dict]) -> str:
    """``<column>: <functional> [τ <tau>, <subset>]`` from the declared functionals (PoC §4.4)."""
    d = decls.get(column)
    return column if d is None else f"{column}: {d['functional']} [τ {d['tau']}, {d['rollout_subset']}, good_sign {d['good_sign']:+g}]"


def _title(text: str, synthetic: bool) -> str:
    return ("SYNTHETIC DATA — not a result about any policy\n" if synthetic else "") + text


def heatmap_text(columns: list[str], decls: dict[str, dict], synthetic: bool) -> dict:
    """The title and x tick labels ``plot_heatmap`` draws (also written to ``figure_labels.json`` so they can be audited)."""
    return {"title": _title("Episode x metric attribution heatmap (PoC §4.2, §4.5)\ngroup labels: " + STATUS.split(" (")[0], synthetic),
            "xticklabels": [axis_label(c, decls) for c in columns]}


def tradeoff_text(x: str, y: str, decls: dict[str, dict], synthetic: bool) -> dict:
    """The title and axis labels ``plot_tradeoff`` draws."""
    return {"title": _title("Throughput vs reliability influence, positive = harmful (PoC §4.5)\ngroup labels: " + STATUS.split(" (")[0], synthetic),
            "xlabel": axis_label(x, decls) + "\n(throughput)", "ylabel": axis_label(y, decls) + "\n(reliability)"}


def plot_heatmap(table: pd.DataFrame, columns: list[str], groups: pd.Series, decls: dict[str, dict], path: Path,
                 synthetic: bool = False, tick_labels: dict[str, str] | None = None) -> Path:
    """Episode × metric heatmap of z-scored harm (red = harmful, blue = helpful), episodes sorted by group then by mean harm
    (PoC §4.5). Each column's label names its functional and τ from ``functionals.json``; ``tick_labels`` (column -> short text) replaces the long
    tick labels where the full declaration is shown beside the figure (the dashboard)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = pd.DataFrame({c: (table[c] - table[c].mean()) / table[c].std(ddof=0) for c in columns})
    order = sorted(range(len(table)), key=lambda i: (GROUPS.index(groups.iloc[i]), -float(z.iloc[i].mean())))
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(z.iloc[order].to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest")
    ax.set_xticks(range(len(columns)))
    txt = heatmap_text(columns, decls, synthetic)
    ax.set_xticklabels([tick_labels[c] for c in columns] if tick_labels else txt["xticklabels"], rotation=35 if not tick_labels else 20, ha="right", fontsize=6 if not tick_labels else 9)
    ax.set_ylabel("training episodes (sorted by group, then mean harm)")
    bounds = np.cumsum([sum(groups.iloc[i] == g for i in range(len(table))) for g in GROUPS])
    for b in bounds[:-1]:
        ax.axhline(b - 0.5, color="black", lw=0.6)
    spans = [(g, lo, hi) for g, lo, hi in zip(GROUPS, np.r_[0, bounds[:-1]], bounds) if hi > lo]
    ax.set_yticks([(lo + hi - 1) / 2 for _, lo, hi in spans])
    ax.set_yticklabels([f"{g.replace('_', ' ')} (n={hi - lo})" for g, lo, hi in spans], fontsize=7)
    fig.colorbar(im, ax=ax, label="z-scored harm score (positive = harmful)")
    ax.set_title(txt["title"], fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return Path(path)


def plot_tradeoff(table: pd.DataFrame, groups: pd.Series, decls: dict[str, dict], path: Path, x: str = "harm_rmst60",
                  y: str = "harm_f30", synthetic: bool = False) -> Path:
    """Scatter of throughput influence (``x``, default ``harm_rmst60``; or ``harm_hrt60_lin``) against reliability influence
    (``y``, default ``harm_f30``), coloured by group, quadrants labelled (PoC §4.5): upper right consistent harmers, lower left
    consistent helpers, the other two trade-offs; metric-specific episodes lie near an axis. Positive = harmful."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 7))
    for g in GROUPS:
        m = (groups == g).to_numpy()
        ax.scatter(table.loc[m, x], table.loc[m, y], s=14, c=GROUP_COLORS[g], label=f"{g.replace('_', ' ')} (n={int(m.sum())})", alpha=0.85)
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(0, color="black", lw=0.6)
    xl, yl = ax.get_xlim(), ax.get_ylim()
    kw = dict(fontsize=8, alpha=0.7, ha="center", va="center")
    ax.text(xl[1] * 0.6, yl[1] * 0.92, "consistent harmers", **kw)
    ax.text(xl[0] * 0.6, yl[0] * 0.92, "consistent helpers", **kw)
    ax.text(xl[0] * 0.6, yl[1] * 0.92, "trade-off", **kw)
    ax.text(xl[1] * 0.6, yl[0] * 0.92, "trade-off", **kw)
    txt = tradeoff_text(x, y, decls, synthetic)
    ax.set_xlabel(txt["xlabel"], fontsize=7)
    ax.set_ylabel(txt["ylabel"], fontsize=7)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    ax.set_title(txt["title"], fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


def analyze_run(score_run_id: str, name: str, z: float = 1.0, columns: list[str] | None = None, by: tuple[str, ...] = ("task_name",),
                out_dir: Path | None = None, lineage_root: Path | None = None, chain: str | None = None, allow_dirty: bool = False,
                synthetic: bool = False) -> dict:
    """Stage ``analyze`` (recorded as stage ``summarize``): groups, aggregation by catalog column(s), heatmap and scatter of an
    attribute (or ensemble) run's score table, linked to that run (input entity = the score table's sha256, param
    ``score_run_id``) with the group threshold and the "descriptive, unverified" status in its params (PoC §4.2, §4.4, §4.5).
    ``synthetic`` puts SYNTHETIC in every figure title and in the params.

    Reading catalog rows through ``Subset.load(...).episodes()`` (for ``aggregate_by``) is not a method-boundary breach: it is
    the catalog's bookkeeping columns of the training subset this attribute run already consumed through ``open_for_method``
    (``attribute_run``), the same columns ``MethodView.catalog`` exposes, and no label or parent hash is read (boundary scan)."""
    from vlapoc.attribution import _artifact_json, _artifact_table
    from vlapoc.lineage import Subset, rel
    from vlapoc.runs import client, stage_run

    from vlapoc.method_data import refuse_groundtruth_arm

    run = client(lineage_root).get_run(score_run_id)
    refuse_groundtruth_arm(run, "analyze")
    table = _artifact_table(run, "attribution_scores", lineage_root)
    decls = {r["column"]: r for r in _artifact_json(run, "functionals", lineage_root)}
    cols = [f"harm_{m}" for m in (columns or DEFAULT_METRICS)]
    missing = [c for c in cols if c not in table.columns]
    if missing:
        raise ValueError(f"score table lacks {missing}")
    out_dir = Path(out_dir) if out_dir else Path("data") / "attribution" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = metric_groups(table, cols, z)
    gpath = out_dir / "groups.csv"
    table[["episode_id", "content_hash"]].assign(group=groups.to_numpy()).to_csv(gpath, index=False)
    episodes = Subset.load(run.data.tags["input.training"].split(":", 1)[1], lineage_root).episodes(lineage_root)
    outputs = {"groups": gpath}
    for c in by:
        p = out_dir / f"aggregate_{c}.csv"
        aggregate_by(table, c, cols, episodes).to_csv(p, index=False)
        outputs[f"aggregate_{c}"] = p
    x = "harm_rmst60" if "harm_rmst60" in cols else cols[0]
    y = "harm_f30" if "harm_f30" in cols else cols[-1]
    outputs["heatmap"] = plot_heatmap(table, cols, groups, decls, out_dir / "heatmap.png", synthetic)
    outputs["tradeoff"] = plot_tradeoff(table, groups, decls, out_dir / "tradeoff.png", x, y, synthetic)
    lpath = out_dir / "figure_labels.json"
    lpath.write_text(json.dumps({"heatmap": heatmap_text(cols, decls, synthetic), "tradeoff": tradeoff_text(x, y, decls, synthetic)},
                                indent=1, ensure_ascii=False))
    outputs["figure_labels"] = lpath
    counts = groups.value_counts().to_dict()
    with stage_run("summarize", name, lineage_root, chain=chain, allow_dirty=allow_dirty) as r:
        r.input_entity("score_table", run.data.tags["output.attribution_scores.sha256"], "scores")
        r.params({"score_run_id": score_run_id, "group_z_threshold": z, "group_z_threshold_status": "inferred default 1; declared parameter",
                  "group_status": STATUS, "group_rule_divergence": DIVERGENCE_NOTE, "columns": cols, "aggregate_by": list(by), "synthetic": synthetic,
                  "group_counts": {g: int(counts.get(g, 0)) for g in GROUPS}, "x_throughput": x, "y_reliability": y})
        for k, p in outputs.items():
            r.output_file(k, p, kind="figure" if str(p).endswith(".png") else "table")
        return {"run_id": r.run_id, "out_dir": rel(out_dir), "group_counts": {g: int(counts.get(g, 0)) for g in GROUPS}}


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m vlapoc.analysis --scores <attribute run id> --name <name> [--z 1.0]``."""
    import argparse

    p = argparse.ArgumentParser(prog="python -m vlapoc.analysis")
    p.add_argument("--scores", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--z", type=float, default=1.0)
    p.add_argument("--lineage-root", type=Path, default=None)
    a = p.parse_args(argv)
    print(json.dumps(analyze_run(a.scores, a.name, a.z, lineage_root=a.lineage_root), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
