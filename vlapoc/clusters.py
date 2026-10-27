"""The ``clusters`` stage: pinpoint exact clusters of training episodes and describe the pattern inside each one that harms or
boosts a metric, from an attribution table (dashboard S2). Method side: reads no answer key.

Where this sits: after ``vlapoc.attribution`` has written a score table (one attribute or ensemble run), ``scripts/stage.py clusters``
calls ``clusters_run`` here. It records one ``cluster`` stage run whose outputs the dashboard reads through ``vlapoc.results``. The
clusters it finds are **estimated** (first-order sums of individual influence); only a removal arm that passes the two-step verdict
can make one **confirmed** (Round B, dashboard S4). Reading order: ``descriptor_table`` -> ``bin_descriptors`` -> the three lenses
(``label_clusters``, ``slice_clusters``, ``group_clusters``) -> ``pattern_of`` -> ``clusters_run``.

Grounding
---------
- PoC §1 (p. 3), quoted: "Per-episode scores become most valuable when aggregated by metadata. Findings such as 'demos from operator
  3 hurt drawer tasks' or 'episodes over 400 steps are net harmful' are usually what stakeholders act on." Lens 1 (label groups)
  is the first example, lens 2 (rule-like slices over behaviour descriptors) the second: a rule over a continuous descriptor.
- PoC §2.2 (p. 4): the catalog's "episode length, and basic action statistics (smoothness, idle time, gripper toggles)" are the
  slice finder's descriptors. The label-free heuristics H7/H8 (task-relative length, post-release share), H5 (duplicate frames)
  and H6 (instruction mismatch) of ``vlapoc.methods.heuristics`` are added, read through ``open_for_method``.
- PoC §4.2 (pp. 9-10): the four groups (``analysis.metric_groups``, the owner-approved >= 2-of-5 rule) are lens 3.
  "Metric-specific: move one metric, neutral on the rest ... explain specific behaviours, e.g. slowness or drops": each such
  group is split by the metric it moves and the sign, so one cluster means one behaviour (a divergence from "one cluster per
  group", recorded in the run params).
- PoC §3.4 (p. 7) and §4.4 (pp. 10-11): "Every attribution result states which functional it explains" / "Label every claim with
  its metric and time cap tau": every row and every pattern carries the declaration (functional, tau, rollout subset, good sign)
  of each harm column it reports, copied from the score run's ``functionals.json``. A run with no declarations is refused.
- PoC §9.2 (p. 19): "which kind of harm each data group causes - drops and safety stops versus slowness": each pattern says which
  metrics the cluster moves and in which direction.
- PoC §2.1 item 4 (p. 3): "check whether your attribution method flags exactly those". Cluster discovery is method code: it opens
  the training subset only through ``vlapoc.method_data.open_for_method`` (a fenced, leaky or ground-truth-derived subset is
  refused), imports nothing from ``vlapoc.groundtruth`` / ``vlapoc.recovery`` / ``vlapoc.discrepancy`` and never sees a label.
  Scoring clusters against the planted types is a separate stage (``vlapoc.recovery.cluster_recovery``, run afterwards).
- CUPID (docs/papers/CUPID_Influence_Functions_arXiv-2506.19121.pdf) p. 11, quoted: "one could approximate group effects [14] via
  random sampling or limit influence estimation to smaller data subsets identified using coarse-grained heuristics". A cluster's
  mean/sum of member influence is only a first-order estimate of removing the cluster, hence ``status = "estimated"``.
- DataMIL (docs/papers/DataMIL_Datamodels_arXiv-2505.09603.pdf) p. 6: cluster-level aggregation by class or task is established
  practice and reduces per-example noise.
- SliceLine (Sagadeeva and Boehm, SIGMOD 2021; ``sliceline==0.3.0``, BSD-3, pure wheel, installed into ``.venv`` on Python 3.14.0;
  numba is optional): top-k conjunctions of at most ``max_l`` predicates, support >= ``min_sup``, where an error vector is high.
  Measured on this venv (dashboard S2 premises): it one-hot encodes every distinct value, so continuous descriptors must be
  discretised first (this module quantile-bins them), and the error vector must be non-negative (score = alpha*((slice error /
  size) / mean error - 1) - (1 - alpha)*(N / size - 1)), so each z-scored harm column becomes two non-negative vectors,
  ``max(z, 0)`` (harm slices) and ``max(-z, 0)`` (boost slices).

Declared parameters (inferred defaults, recorded in the run params; the owner may override): 5 quantile bins (a descriptor whose
minimum value holds more than 1/5 of the episodes, such as ``idle_steps``, gets that value as its own bin; a bin smaller than
``min_sup`` is merged into its neighbour), ``max_l = 2``, ``k = 5``,
``alpha = 0.95``, ``min_sup = max(10, 2% of N)``, group threshold z = 1, 1000 bootstrap resamples. Measured sensitivity (960
episodes, +1.5 sigma planted): a 40-member slice is found at rank <= 2 with alpha 0.95; a 17-19-member slice is not (alpha 0.99
finds it, but then always returns k slices, mostly noise). So small slices can be missed at the defaults.

Clusters with identical member sets are reported once (``dedupe``: the kept row lists the equivalent definitions).

Caveat carried into every pattern (R1): a slice finder picks the most extreme slices, so the percentile-bootstrap interval of a
discovered slice's mean is optimistic. These are estimates to confirm, not findings.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from vlapoc.analysis import DEFAULT_METRICS, DIVERGENCE_NOTE, aggregate_by, metric_groups
from vlapoc.libero_convert import REPO_ROOT

CATALOG_DESCRIPTORS = ("length_steps", "duration_s", "smoothness", "idle_fraction", "idle_steps", "gripper_toggles")
HEURISTIC_DESCRIPTORS = ("length_vs_task", "post_release_share_vs_task", "duplicate_frame_pairs", "instruction_mismatch")
CATEGORICAL_DESCRIPTORS = ("task_name", "scene")
LABEL_COLUMNS = ("task_name", "scene", "suite")  # the groupable catalog columns that vary (measured in the dashboard plan)
DEFAULTS = {"n_bins": 5, "max_l": 2, "k": 5, "alpha": 0.95, "min_sup_fraction": 0.02, "min_sup_floor": 10, "z": 1.0, "n_boot": 1000, "seed": 0}
STATUS = "estimated"
STATUS_NOTE = ("estimated: first-order sum of individual influence (CUPID p. 11); a cluster is confirmed only after removing it, "
               "retraining and passing the two-step verdict (PoC §4.4, §9.1 Step 4)")
SELECTION_CAVEAT = ("slices are selected for extreme harm, so their intervals are optimistic; treat them as estimates to confirm. "
                    "Readings compare a cluster's mean with the subset mean, per metric, uncorrected for the number of metrics and clusters")
DEFAULT_CACHE = REPO_ROOT / "data" / "dashboard_cache" / "descriptors"


# ---------------------------------------------------------------------------------------------
# descriptors: what an episode looks like, from method-facing data only
# ---------------------------------------------------------------------------------------------


def descriptor_table(view, heuristics: bool = True, cache_dir: Path | None = DEFAULT_CACHE) -> pd.DataFrame:
    """One row per episode of the ``MethodView`` (keys ``episode_id``, ``content_hash``): the catalog descriptors
    (``CATALOG_DESCRIPTORS``), the label columns (``LABEL_COLUMNS``) and, if ``heuristics``, the label-free heuristics of
    ``vlapoc.methods.heuristics`` (``HEURISTIC_DESCRIPTORS``; H5 and H8 scan every frame of every episode, about 85 s for
    960 episodes, measured). Everything comes from ``view`` (``open_for_method``), so no answer key is read.

    The heuristic part is cached under ``cache_dir`` (``None`` disables), keyed by the subset id and checked against the episode
    hashes, so a changed episode recomputes."""
    from vlapoc.methods import heuristics as H

    c = view.catalog
    df = c[["episode_id", "content_hash", *LABEL_COLUMNS, *CATALOG_DESCRIPTORS]].reset_index(drop=True).copy()
    if heuristics:
        path = None if cache_dir is None else Path(cache_dir) / f"descriptors-{view.subset_id[:16]}.parquet"
        cached = pd.read_parquet(path) if path is not None and path.is_file() else None
        if cached is not None and cached[["episode_id", "content_hash"]].equals(df[["episode_id", "content_hash"]]):
            heur = cached
        else:
            tabs = H.compute(view, HEURISTIC_DESCRIPTORS)
            heur = df[["episode_id", "content_hash"]].copy()
            for name in HEURISTIC_DESCRIPTORS:
                t = tabs[name]
                heur[H.score_column(t)] = t[H.score_column(t)].to_numpy()
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                heur.to_parquet(path, index=False)
        for name in HEURISTIC_DESCRIPTORS:
            df[name] = heur[name].to_numpy()
    return df


def _fmt(x: float) -> str:
    return f"{float(x):.4g}"


def bin_spec(x: pd.Series, n_bins: int, min_bin: int = 1) -> dict | None:
    """The binning of one numeric descriptor, or ``None`` if it is constant.

    ``kind="value"`` when it has at most ``n_bins`` distinct values (each its own bin). Otherwise ``kind="cuts"``: bin ``i`` is
    ``(cuts[i-1], cuts[i]]`` (first bin ``<= cuts[0]``, last ``> cuts[-1]``), cuts are data values from the quantiles
    (``method="lower"``); when the minimum value holds more than ``1/n_bins`` of the episodes (zero-inflated descriptors such as
    ``idle_steps``) it is its own bin and the rest is cut into ``n_bins - 1`` quantile bins. A bin with fewer than ``min_bin`` episodes
    is merged into its smaller neighbour (down to two bins), because the slice finder cannot return a slice below its minimum support
    (measured on v3: ``duplicate_frame_pairs > 0`` has 26 episodes and split into bins of 7, 18 and 1, none usable; merged it is one bin
    of 26). ``labels`` read like ``"(1.08, 1.31]"``."""
    v = x.to_numpy(float)
    if np.isnan(v).any():
        raise ValueError(f"descriptor {x.name!r} has NaN values")
    uniq = np.unique(v)
    if len(uniq) <= 1:
        return None
    if len(uniq) <= n_bins:
        return {"kind": "value", "values": [float(u) for u in uniq], "labels": [f"= {_fmt(u)}" for u in uniq]}
    if (v == uniq[0]).mean() > 1 / n_bins:
        cuts = [uniq[0], *np.quantile(v[v > uniq[0]], np.linspace(0, 1, n_bins)[1:-1], method="lower")]
    else:
        cuts = list(np.quantile(v, np.linspace(0, 1, n_bins + 1)[1:-1], method="lower"))
    cuts = sorted({float(c) for c in cuts if c < uniq[-1]})
    while len(cuts) > 1:
        sizes = np.bincount(np.searchsorted(np.array(cuts), v, side="left"), minlength=len(cuts) + 1)
        i = int(np.argmin(sizes))
        if sizes[i] >= min_bin:
            break
        left = sizes[i - 1] if i > 0 else np.inf
        right = sizes[i + 1] if i < len(sizes) - 1 else np.inf
        del cuts[i - 1 if left <= right else i]  # merge bin i into its smaller neighbour: drop the cut between them
    labels = [f"= {_fmt(cuts[0])}" if cuts[0] == uniq[0] else f"<= {_fmt(cuts[0])}"]
    labels += [f"({_fmt(a)}, {_fmt(b)}]" for a, b in zip(cuts[:-1], cuts[1:])]
    labels.append(f"> {_fmt(cuts[-1])}")
    return {"kind": "cuts", "cuts": cuts, "labels": labels}


def encode_numeric(x, spec: dict) -> np.ndarray:
    """Integer bin codes of raw values under a ``bin_spec`` (the same function the verifier re-evaluates with)."""
    edges = np.array(spec["values"] if spec["kind"] == "value" else spec["cuts"])
    return np.searchsorted(edges, np.asarray(x, float), side="left")  # bin i = (edges[i-1], edges[i]]; a "value" spec matches exactly


def bin_descriptors(desc: pd.DataFrame, n_bins: int = DEFAULTS["n_bins"], min_bin: int = 1) -> tuple[pd.DataFrame, dict, dict]:
    """``(codes, specs, dropped)``: an integer code per descriptor and episode (``codes`` columns are the kept descriptors), the
    ``bin_spec`` of each (categorical descriptors: ``kind="categorical"`` with the sorted ``categories``), and ``{descriptor: reason}``
    for the descriptors dropped (constant). The slice finder searches over these codes."""
    codes, specs, dropped = {}, {}, {}
    for c in (*CATALOG_DESCRIPTORS, *HEURISTIC_DESCRIPTORS):
        if c not in desc:
            continue
        s = bin_spec(desc[c], n_bins, min_bin)
        if s is None:
            dropped[c] = "constant"
            continue
        specs[c] = s
        codes[c] = encode_numeric(desc[c], s)
    for c in CATEGORICAL_DESCRIPTORS:
        cats = sorted(desc[c].astype(str).unique())
        specs[c] = {"kind": "categorical", "categories": cats, "labels": [f"= {x}" for x in cats]}
        codes[c] = pd.Categorical(desc[c].astype(str), categories=cats).codes.astype(int)
    return pd.DataFrame(codes, index=desc.index), specs, dropped


def predicate_text(feature: str, code: int, specs: dict) -> str:
    """``length_vs_task > 1.31`` / ``scene = KITCHEN_SCENE3`` for a predicate."""
    return f"{feature} {specs[feature]['labels'][int(code)]}"


def predicate_mask(codes: pd.DataFrame, predicates: list[dict]) -> np.ndarray:
    """Boolean mask of the episodes satisfying every ``{"feature", "code"}`` predicate (a conjunction)."""
    m = np.ones(len(codes), bool)
    for p in predicates:
        m &= codes[p["feature"]].to_numpy() == int(p["code"])
    return m


# ---------------------------------------------------------------------------------------------
# estimated effect of a set of episodes
# ---------------------------------------------------------------------------------------------


def member_stats(table: pd.DataFrame, mask: np.ndarray, cols: list[str], n_boot: int, rng: np.random.Generator) -> dict:
    """Per harm column: the members' mean with a percentile-bootstrap 95% CI over the members (the method of
    ``analysis.aggregate_by``), the sum (CUPID p. 11's first-order group estimate), and ``__z``: the mean in units of the
    column's s.d. over the whole subset, measured from the subset mean."""
    out = {}
    for c in cols:
        v = table.loc[mask, c].to_numpy(float)
        boot = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1) if len(v) > 1 else np.array([v.mean()])
        col = table[c].to_numpy(float)
        out |= {f"{c}__mean": float(v.mean()), f"{c}__lo": float(np.quantile(boot, 0.025)), f"{c}__hi": float(np.quantile(boot, 0.975)),
                f"{c}__sum": float(v.sum()), f"{c}__z": float((v.mean() - col.mean()) / col.std(ddof=0))}
    return out


def reading(lo: float, hi: float, subset_mean: float) -> str:
    """``harmful`` / ``helpful`` / ``no detectable shift`` from whether a mean's interval lies above / below the subset mean."""
    return "harmful" if lo > subset_mean else ("helpful" if hi < subset_mean else "no detectable shift")


def _row(cluster_id: str, lens: str, definition: str, direction: str, target: str, mask: np.ndarray, table: pd.DataFrame, cols: list[str],
         n_boot: int, rng: np.random.Generator, extra: dict | None = None) -> dict:
    stats = member_stats(table, mask, cols, n_boot, rng)
    row = {"cluster_id": cluster_id, "lens": lens, "definition": definition, "direction": direction, "target_metric": target,
           "n": int(mask.sum()), "share_of_subset": float(mask.mean()), "status": STATUS, **stats, **(extra or {})}
    for c in cols:
        row[f"{c}__reading"] = reading(row[f"{c}__lo"], row[f"{c}__hi"], float(table[c].mean()))
    return row


# ---------------------------------------------------------------------------------------------
# the three lenses
# ---------------------------------------------------------------------------------------------


def label_clusters(table: pd.DataFrame, catalog: pd.DataFrame, cols: list[str], n_boot: int, seed: int) -> tuple[list[dict], list[np.ndarray]]:
    """Lens 1: one cluster per value of each label column (``task_name``, ``scene``, ``suite``): ``analysis.aggregate_by`` (so the
    numbers are exactly that function's), plus the member sum, the reading per metric and the rank of the group on each metric among
    the groups of the same column (``harm_rank__<col>``: 1 = most harmful, ``boost_rank__<col>``: 1 = most helpful).

    ``direction``: ``harm`` / ``boost`` if some metric reads that way and none the other, ``mixed`` if both, ``none`` if no metric
    shifts. The ``n`` of the groups of a column sum to the table's rows (PoC §1 p. 3)."""
    rows, masks = [], []
    key = table[["episode_id"]].merge(catalog[["episode_id", *[c for c in LABEL_COLUMNS if c not in table.columns]]], on="episode_id", how="left", validate="one_to_one")
    t = table.copy()
    for c in LABEL_COLUMNS:
        t[c] = table[c] if c in table.columns else key[c].to_numpy()
    for lc in LABEL_COLUMNS:
        agg = aggregate_by(t, lc, cols, None, n_boot, seed).set_index(lc)
        ranks = {c: agg[f"{c}__mean"].rank(ascending=False, method="first") for c in cols}
        for value, a in agg.iterrows():
            mask = (t[lc] == value).to_numpy()
            row = {"cluster_id": f"label:{lc}={value}", "lens": "label", "definition": f"{lc} = {value}", "target_metric": "", "n": int(a["n"]),
                   "share_of_subset": float(mask.mean()), "status": STATUS, "label_column": lc}
            reads = set()
            for c in cols:
                v = t.loc[mask, c].to_numpy(float)
                row |= {f"{c}__mean": float(a[f"{c}__mean"]), f"{c}__lo": float(a[f"{c}__lo"]), f"{c}__hi": float(a[f"{c}__hi"]), f"{c}__sum": float(v.sum()),
                        f"{c}__z": float((v.mean() - t[c].mean()) / t[c].std(ddof=0))}
                row[f"{c}__reading"] = reading(row[f"{c}__lo"], row[f"{c}__hi"], float(t[c].mean()))
                row[f"harm_rank__{c}"] = float(ranks[c].loc[value])
                row[f"boost_rank__{c}"] = float(len(agg) + 1 - ranks[c].loc[value])
                reads.add(row[f"{c}__reading"])
            row["direction"] = ("mixed" if {"harmful", "helpful"} <= reads else "harm" if "harmful" in reads else "boost" if "helpful" in reads else "none")
            rows.append(row)
            masks.append(mask)
    return rows, masks


def slice_clusters(codes: pd.DataFrame, specs: dict, table: pd.DataFrame, cols: list[str], p: dict, rng: np.random.Generator) -> tuple[list[dict], list[np.ndarray]]:
    """Lens 2: SliceLine over the binned descriptors, per harm column and per side. Harm side: errors ``max(z, 0)``; boost side:
    ``max(-z, 0)`` (z = the z-scored harm column). One cluster per returned slice (``slice:<metric>:<harm|boost>:<rank>``), with its
    predicates (JSON, re-evaluable from ``descriptors.parquet``), SliceLine's score and the slice's average error."""
    from sliceline.slicefinder import Slicefinder

    n = len(table)
    min_sup = max(int(p["min_sup_floor"]), int(round(p["min_sup_fraction"] * n)))
    feats = list(codes.columns)
    X = codes.to_numpy()
    rows, masks = [], []
    for c in cols:
        z = ((table[c] - table[c].mean()) / table[c].std(ddof=0)).to_numpy(float)
        for side, err in (("harm", np.maximum(z, 0.0)), ("boost", np.maximum(-z, 0.0))):
            if not (err > 0).any():
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sf = Slicefinder(alpha=p["alpha"], k=p["k"], max_l=p["max_l"], min_sup=min_sup, verbose=False).fit(X, err)
            for rank, (s, st) in enumerate(zip(sf.top_slices_, sf.top_slices_statistics_), 1):
                preds = [{"feature": feats[i], "code": int(v)} for i, v in enumerate(s) if not pd.isna(v)]
                mask = predicate_mask(codes, preds)
                metric = c.removeprefix("harm_")
                rows.append(_row(f"slice:{metric}:{side}:{rank}", "slice", " AND ".join(predicate_text(q["feature"], q["code"], specs) for q in preds),
                                 side, c, mask, table, cols, p["n_boot"], rng,
                                 {"predicates": json.dumps(preds), "rank": rank, "slice_score": st["slice_score"],
                                  "slice_average_error": st["slice_average_error"]}))
                masks.append(mask)
    return rows, masks


def group_clusters(table: pd.DataFrame, cols: list[str], p: dict, rng: np.random.Generator) -> tuple[list[dict], list[np.ndarray]]:
    """Lens 3: the four groups of PoC §4.2 (``analysis.metric_groups``, >= 2-of-5 rule, owner-approved): one cluster each for
    ``consistent_helper``, ``consistent_harmer`` and ``trade_off``, and one per (metric, sign) for ``metric_specific``. Empty
    groups give no cluster."""
    z = p["z"]
    groups = metric_groups(table, cols, z)
    zs = pd.DataFrame({c: (table[c] - table[c].mean()) / table[c].std(ddof=0) for c in cols})
    rows, masks = [], []

    def add(cid: str, definition: str, direction: str, target: str, mask: np.ndarray, group: str) -> None:
        if mask.sum() == 0:
            return
        rows.append(_row(cid, "metric_group", definition, direction, target, mask, table, cols, p["n_boot"], rng, {"group": group}))
        masks.append(mask)

    g = groups.to_numpy()
    add("group:consistent_harmer", f"group = consistent_harmer (>= 2 metrics harmful at z >= {z:g}, none helpful)", "harm", "", g == "consistent_harmer", "consistent_harmer")
    add("group:consistent_helper", f"group = consistent_helper (>= 2 metrics helpful at z <= -{z:g}, none harmful)", "boost", "", g == "consistent_helper", "consistent_helper")
    add("group:trade_off", f"group = trade_off (harmful on some metrics, helpful on others, |z| >= {z:g})", "mixed", "", g == "trade_off", "trade_off")
    for c in cols:
        metric = c.removeprefix("harm_")
        spec = g == "metric_specific"
        add(f"group:metric_specific:{metric}:harm", f"group = metric_specific, harmful on {metric} only (z >= {z:g})", "harm", c,
            spec & (zs[c].to_numpy() >= z), "metric_specific")
        add(f"group:metric_specific:{metric}:boost", f"group = metric_specific, helpful on {metric} only (z <= -{z:g})", "boost", c,
            spec & (zs[c].to_numpy() <= -z), "metric_specific")
    return rows, masks


# ---------------------------------------------------------------------------------------------
# patterns: what is different about the members
# ---------------------------------------------------------------------------------------------


def declaration_of(decls: dict[str, dict], col: str) -> dict:
    """The fields of a column's declaration a reader needs next to its number (PoC §3.4 p. 7; §4.4 p. 10: metric and tau)."""
    d = decls[col]
    return {"column": col, "functional": d["functional"], "tau": d["tau"], "rollout_subset": d["rollout_subset"], "good_sign": d["good_sign"],
            "return_definition": d.get("return_definition"), "weighting": d.get("weighting")}


def pattern_of(row: dict, mask: np.ndarray, table: pd.DataFrame, desc: pd.DataFrame, cols: list[str], decls: dict[str, dict], n_top: int = 5) -> dict:
    """The pattern of one cluster: how its descriptors differ from the rest (standardised difference, in s.d. of the whole subset),
    its label composition (top ``task_name`` / ``scene`` shares against their share of the subset), which metrics it moves and which
    way (relative to the subset mean), its ``n_top`` most influential members on the primary metric, and one sentence. Everything is
    labelled ``estimated`` (``STATUS_NOTE``)."""
    contrasts = []
    for d in (*CATALOG_DESCRIPTORS, *HEURISTIC_DESCRIPTORS):
        if d not in desc:
            continue
        x = desc[d].to_numpy(float)
        sd = x.std(ddof=0)
        if sd == 0 or mask.all():
            continue
        contrasts.append({"descriptor": d, "mean_in": float(x[mask].mean()), "mean_out": float(x[~mask].mean()),
                          "std_diff": float((x[mask].mean() - x[~mask].mean()) / sd)})
    contrasts.sort(key=lambda r: -abs(r["std_diff"]))
    composition = {}
    for lc in ("task_name", "scene"):
        inside = desc.loc[mask, lc].value_counts(normalize=True).head(3)
        allv = desc[lc].value_counts(normalize=True)
        composition[lc] = [{"value": str(k), "share_in_cluster": float(v), "share_in_subset": float(allv[k]), "lift": float(v / allv[k])} for k, v in inside.items()]
    moves = []
    for c in cols:
        moves.append({"metric": c.removeprefix("harm_"), "column": c, "reading": row[f"{c}__reading"], "mean": row[f"{c}__mean"],
                      "ci95": [row[f"{c}__lo"], row[f"{c}__hi"]], "z": row[f"{c}__z"], "subset_mean": float(table[c].mean())})
    primary = row["target_metric"] or max(cols, key=lambda c: abs(row[f"{c}__z"]))
    order = table.loc[mask, primary].to_numpy(float)
    ids = np.flatnonzero(mask)
    top = ids[np.argsort(-order if row["direction"] != "boost" else order)][:n_top]
    top_members = [{"episode_id": table["episode_id"].iloc[i], "content_hash": table["content_hash"].iloc[i], "harm": float(table[primary].iloc[i]),
                    "harm_z": float((table[primary].iloc[i] - table[primary].mean()) / table[primary].std(ddof=0))} for i in top]
    from vlapoc.metrics_registry import SHORT_NAMES

    nm = lambda m: SHORT_NAMES.get(m["metric"], m["metric"])  # noqa: E731  plain names in the recorded text
    harmful = [nm(m) for m in moves if m["reading"] == "harmful"]
    helpful = [nm(m) for m in moves if m["reading"] == "helpful"]
    flat = [nm(m) for m in moves if m["reading"] == "no detectable shift"]
    parts = [f"{row['definition']}: n = {row['n']} ({100 * row['share_of_subset']:.1f}% of the subset)."]
    if harmful:
        parts.append("Estimated harmful on " + ", ".join(harmful) + ".")
    if helpful:
        parts.append("Estimated helpful on " + ", ".join(helpful) + ".")
    if flat:
        parts.append("No detectable shift on " + ", ".join(flat) + ".")
    if contrasts and abs(contrasts[0]["std_diff"]) >= 0.5:
        k = contrasts[0]
        parts.append(f"Most different descriptor: {k['descriptor']} ({k['mean_in']:.3g} inside vs {k['mean_out']:.3g} outside, {k['std_diff']:+.2f} s.d.).")
    parts.append("Status: " + STATUS_NOTE + ("; " + SELECTION_CAVEAT if row["lens"] == "slice" else "") + ".")
    return {"cluster_id": row["cluster_id"], "lens": row["lens"], "definition": row["definition"], "direction": row["direction"],
            "status": STATUS, "n": row["n"], "share_of_subset": row["share_of_subset"], "primary_metric": primary,
            "metrics": moves, "descriptor_contrasts": contrasts, "label_composition": composition, "top_members": top_members,
            "declarations": {c: declaration_of(decls, c) for c in cols}, "text": " ".join(parts)}


# ---------------------------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------------------------


def harm_columns(table: pd.DataFrame, metrics: list[str] | None) -> list[str]:
    """``harm_<metric>`` columns for ``metrics`` (default ``analysis.DEFAULT_METRICS``); ``ValueError`` if the table lacks one.
    A name already starting ``harm_`` is taken as a full column name (e.g. ``harm_f30__same_task``)."""
    cols = [m if m.startswith("harm_") else f"harm_{m}" for m in (metrics or DEFAULT_METRICS)]
    missing = [c for c in cols if c not in table.columns]
    if missing:
        raise ValueError(f"score table lacks {missing}")
    return cols


def dedupe(rows: list[dict], masks: list[np.ndarray]) -> tuple[list[dict], list[np.ndarray], int]:
    """Report clusters with identical member sets once (dashboard round 2, AC2.9). The first row in lens order (label, slice, metric group) is kept; the others
    become its ``equivalent_definitions`` (JSON list of ``{cluster_id, lens, definition, direction, target_metric}``), so e.g. ``scene = KITCHEN_SCENE3`` and
    ``task_name = KITCHEN_SCENE3_turn_on_the_stove...`` (the same 48 episodes in the demo) are one cluster that names both. Returns ``(rows, masks, n_removed)``."""
    seen: dict[bytes, int] = {}
    keep_rows, keep_masks, n = [], [], 0
    for r, m in zip(rows, masks):
        key = np.asarray(m, bool).tobytes()
        if key in seen:
            keep_rows[seen[key]]["_eq"].append({k: r[k] for k in ("cluster_id", "lens", "definition", "direction", "target_metric")})
            n += 1
        else:
            seen[key] = len(keep_rows)
            r["_eq"] = []
            keep_rows.append(r)
            keep_masks.append(m)
    for r in keep_rows:
        r["equivalent_definitions"] = json.dumps(r.pop("_eq"))
        r["n_equivalent"] = len(json.loads(r["equivalent_definitions"]))
    return keep_rows, keep_masks, n


def discover(table: pd.DataFrame, view, decls: dict[str, dict], cols: list[str], params: dict | None = None,
             cache_dir: Path | None = DEFAULT_CACHE, heuristics: bool = True) -> dict:
    """The untracked core of ``clusters_run``: ``{"clusters", "members", "patterns", "descriptors", "bins", "dropped"}`` from a
    score table, the training subset's ``MethodView`` and the column declarations. ``ValueError`` if the table does not key
    exactly the view's episode versions or a column is undeclared (PoC §3.4 p. 7)."""
    p = {**DEFAULTS, **(params or {})}
    undeclared = [c for c in cols if c not in decls]
    if undeclared:
        raise ValueError(f"columns {undeclared} have no declared functional in the score run's functionals.json (PoC §3.4 p. 7)")
    keys = view.catalog[["episode_id", "content_hash"]]
    if len(table) != len(keys) or set(map(tuple, table[["episode_id", "content_hash"]].to_numpy())) != set(map(tuple, keys.to_numpy())):
        raise ValueError("the score table does not key exactly the training subset's episode versions")
    table = table.reset_index(drop=True)
    desc = descriptor_table(view, heuristics, cache_dir).set_index("episode_id").loc[table["episode_id"]].reset_index()
    desc.index = table.index
    codes, specs, dropped = bin_descriptors(desc, p["n_bins"], max(int(p["min_sup_floor"]), int(round(p["min_sup_fraction"] * len(table)))))
    rng = np.random.default_rng(p["seed"])
    rows, masks = [], []
    for fn, args in ((label_clusters, (table, view.catalog, cols, p["n_boot"], p["seed"])),
                     (slice_clusters, (codes, specs, table, cols, p, rng)),
                     (group_clusters, (table, cols, p, rng))):
        r, m = fn(*args)
        rows += r
        masks += m
    rows, masks, n_dup = dedupe(rows, masks)
    decl_json = json.dumps({c: declaration_of(decls, c) for c in cols}, sort_keys=True)
    for r in rows:
        r["declarations"] = decl_json
        r["synthetic"] = False
    clusters = pd.DataFrame(rows)
    members, patterns = [], []
    for r, m in zip(rows, masks):
        pat = pattern_of(r, m, table, desc, cols, decls)
        patterns.append(pat)
        primary = pat["primary_metric"]
        order = table.loc[m, primary].to_numpy(float)
        ids = np.flatnonzero(m)
        ranked = ids[np.argsort(-order if r["direction"] != "boost" else order, kind="stable")]
        members.append(pd.DataFrame({"cluster_id": r["cluster_id"], "episode_id": table["episode_id"].iloc[ranked].to_numpy(),
                                     "content_hash": table["content_hash"].iloc[ranked].to_numpy(), "rank": np.arange(1, len(ranked) + 1),
                                     "primary_metric": primary, "harm": table[primary].iloc[ranked].to_numpy()}))
    descriptors = desc.copy()
    for c in codes.columns:
        descriptors[f"{c}__bin"] = codes[c].to_numpy()
    return {"clusters": clusters, "members": pd.concat(members, ignore_index=True), "patterns": patterns, "descriptors": descriptors,
            "bins": specs, "dropped": dropped, "params": p, "columns": cols, "n_deduped": n_dup}


def clusters_run(score_run_id: str, name: str, metrics: list[str] | None = None, out_dir: Path | None = None, lineage_root: Path | None = None,
                 chain: str | None = None, allow_dirty: bool = False, synthetic: bool = False, params: dict | None = None,
                 cache_dir: Path | None = DEFAULT_CACHE, heuristics: bool = True) -> dict:
    """Stage ``cluster`` (dashboard S2; PoC §1 p. 3, §4.2 pp. 9-10, §9.2 p. 19): the three lenses over an attribute or ensemble run's
    score table, recorded as one MLflow run.

    Refuses before any run exists: a score run without ``functionals.json`` (``NotRecordedError``), a training subset that
    ``open_for_method`` refuses (fenced, leaky, ground-truth-derived or tampered version: ``LineageError`` /
    ``VersionIntegrityError``; PoC §2.1 item 4 p. 3), a table that does not key the subset. The run consumes the score table's
    sha256 (so ``trace`` reaches the attribute run(s)) and the training subset; its params hold the lens settings, descriptor list,
    bin edges and ``synthetic``. ``synthetic`` tags the run ``vlapoc.synthetic=true`` (``results.provenance`` then says SYNTHETIC).
    Outputs: ``clusters`` (parquet: one row per cluster, ``status = "estimated"``), ``members``, ``patterns`` (JSON), ``descriptors``
    (the binned table, for audit) and ``bins`` (JSON). Returns ``{"run_id", "out_dir", "n_clusters", ...}``."""
    from vlapoc import results
    from vlapoc.lineage import Subset, rel
    from vlapoc.method_data import open_for_method, refuse_groundtruth_arm
    from vlapoc.runs import client, stage_run

    run = client(lineage_root).get_run(score_run_id)
    refuse_groundtruth_arm(run, "clusters")
    table = results.score_table(score_run_id, lineage_root)
    decls = results.functionals(score_run_id, lineage_root)  # NotRecordedError for a run that declares none
    subset = Subset.load(run.data.tags["input.training"].split(":", 1)[1], lineage_root)
    view = open_for_method(subset.subset_id, lineage_root)  # refuses fenced / leaky / ground-truth-derived / tampered input
    cols = harm_columns(table, metrics)
    out = discover(table, view, decls, cols, params, cache_dir, heuristics)
    out["clusters"]["synthetic"] = bool(synthetic)
    p = out["params"]
    out_dir = Path(out_dir) if out_dir is not None else Path("data") / "attribution" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"clusters": out_dir / "clusters.parquet", "members": out_dir / "members.parquet", "patterns": out_dir / "patterns.json",
             "descriptors": out_dir / "descriptors.parquet", "bins": out_dir / "bins.json"}
    out["clusters"].to_parquet(files["clusters"], index=False)
    out["members"].to_parquet(files["members"], index=False)
    files["patterns"].write_text(json.dumps(out["patterns"], indent=1, ensure_ascii=False))
    out["descriptors"].to_parquet(files["descriptors"], index=False)
    files["bins"].write_text(json.dumps({"bins": out["bins"], "dropped": out["dropped"]}, indent=1, ensure_ascii=False))
    edges = {c: s.get("cuts") or s.get("values") or f"categorical({len(s['categories'])})" for c, s in out["bins"].items()}
    n_by_lens = out["clusters"]["lens"].value_counts().to_dict()
    with stage_run("cluster", name, lineage_root, chain=chain, allow_dirty=allow_dirty, extra_tags={"vlapoc.synthetic": "true"} if synthetic else None) as r:
        r.input_entity("score_table", run.data.tags["output.attribution_scores.sha256"], "scores")
        r.input_subset(subset, "training", None)
        r.params({"score_run_id": score_run_id, "columns": cols, "lenses": ["label", "slice", "metric_group"], "label_columns": list(LABEL_COLUMNS),
                  "descriptors": list(out["bins"]), "descriptors_dropped": out["dropped"], "bin_edges": edges, "n_bins": p["n_bins"],
                  "slice_finder": {"package": "sliceline==0.3.0", "alpha": p["alpha"], "k": p["k"], "max_l": p["max_l"],
                                   "min_sup": max(int(p["min_sup_floor"]), int(round(p["min_sup_fraction"] * len(out["descriptors"])))),
                                   "errors": "max(z, 0) harm side, max(-z, 0) boost side, z = z-scored harm column"},
                  "group_z_threshold": p["z"], "group_rule": DIVERGENCE_NOTE, "n_boot": p["n_boot"], "seed": p["seed"],
                  "status": STATUS, "status_note": STATUS_NOTE, "selection_caveat": SELECTION_CAVEAT,
                  "metric_specific_split": "one cluster per (metric, sign), not one per group (PoC §4.2 p. 10 'explain specific behaviours')",
                  "heuristic_descriptors": bool(heuristics), "n_episodes": len(out["descriptors"]), "n_clusters": len(out["clusters"]), "n_duplicate_member_sets_merged": out["n_deduped"],
                  "synthetic": synthetic})
        for k, path in files.items():
            r.output_file(k, path, kind="table")
        r.tags({"cluster.n_clusters": len(out["clusters"])})
        return {"run_id": r.run_id, "out_dir": rel(out_dir), "n_clusters": len(out["clusters"]), "n_by_lens": {k: int(v) for k, v in n_by_lens.items()},
                "columns": cols}
