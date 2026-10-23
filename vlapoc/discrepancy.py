"""Where "looks bad" and "hurts" disagree (PoC §9.2), estimated from Round A scores: planted labels (and, as a secondary layer, a
label-free heuristic filter) against per-metric influence. Ground-truth side: reads the answer key; method code never imports this.

Where this sits (docs/dataflow.md, after the injection test): after an attribute or ensemble run on a version that carries an answer key
(v3) exists, ``scripts/stage.py discrepancy`` calls ``discrepancy_run``. It is the estimated half of the claim "bad episodes are not
always harmful for a policy"; the causal half (remove all of type X and compare with removing the same number of clean episodes) is
Round B (dashboard S10). Reading order: ``quadrants`` -> ``type_table`` -> ``filter_flags`` -> ``heuristic_evidence`` -> ``discrepancy_run``.

Grounding
---------
- PoC §9.2 (p. 19), quoted: "The most compelling output is where screening and outcome attribution disagree: episodes that look bad but
  do not hurt the policy, and episodes that look clean but do." "Looks bad" is, first, the planted corruption label of the answer key
  (owner direction 2026-10-22: planted labels, not a new screening method), and second, a flag of the label-free heuristics H1-H8
  (``vlapoc.methods.heuristics``; "what a practical filter would have dropped"). "Hurts" is the episode's per-metric influence.
  Four quadrants per metric: ``bad_and_harmful``, ``bad_but_not_harmful``, ``clean_but_harmful``, ``clean_and_not_harmful``.
- PoC §9.1 Step 1 (p. 18), quoted: "Its verdicts are stored rather than acted on, so they can later be compared with what actually
  happened." This stage is that comparison for the labels and the heuristic filter. No DemInf / RoboDrop screening is built here.
- PoC §4.5 (p. 11): "Check whether attribution assigns each corrupted group to the expected metric." Each type's per-metric reading is that check
  in estimated form (``vlapoc.recovery.EXPECTED_METRICS`` holds the hypotheses declared before any score existed).
- PoC §2.1 item 4 (p. 3) and owner decision 3: the answer key is read here, after the attribution exists: the stage consumes the score
  table's sha256 and refuses an unfinished or non-attribute run, and ``vlapoc/attribution.py``, ``clusters.py``, ``results.py`` and
  ``analysis.py`` never import this module (``scripts/verify_dashboard.py --discrepancy`` scans their source).
- PoC §3.4 (p. 7) and §4.4 (pp. 10-11): every row carries the declaration (functional, tau, rollout subset, good sign) of its harm column.
- CUPID (docs/papers/CUPID_Influence_Functions_arXiv-2506.19121.pdf) Fig. 4 caption, p. 8, quoted: quality measures "degrade performance when
  higher-quality demonstrations induce brittle strategies at test time (TuckBox), or when quality is not the primary factor limiting
  policy success (Bookshelf)". Published precedent for looking good or bad and helping or hurting coming apart.
- CUPID p. 11, quoted: "one could approximate group effects [14] via random sampling". A type's mean influence is a first-order
  estimate of removing the type, so every reading here is labelled estimated.

How the wording is bounded (the owner's caution: non-significance is not proof)
------------------------------------------------------------------------------
A type is compared with the clean episodes, per metric, on the subset-z-scored harm column (z = (x - mean) / s.d., ddof 0). The
difference of the two means gets a percentile bootstrap 95% CI (1000 resamples of each group). If the interval excludes 0 the reading is
"harmful on <metric>" (above) or "helpful on <metric>" (below). If it does not, the reading is always written
"no detectable harm on <metric> (estimated; CI [lo, hi])": an interval that spans zero cannot show there is no harm, and with 13 episodes
per type (1.4% of the data) it can be wide. The word that would claim the opposite never appears in any output (``verify_dashboard.py
--discrepancy`` scans every output file). Only a removal arm that passes the two-step verdict can support that claim (Round B).

Declared parameters (inferred, recorded in the run params): "harmful" means harm z >= 1 (the ``analysis.metric_groups`` threshold);
the filter flags, per heuristic, the top ``q`` of episodes with ``q`` = the injection rate (labelled / N = 91/960 = 9.48% on v3), including
ties only if the flagged set stays within ``q * N``; "looks bad by the filter" = flagged by any of H1-H8 (H7 in both directions);
1000 bootstrap resamples, seed 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vlapoc.analysis import DEFAULT_METRICS
from vlapoc.libero_convert import REPO_ROOT

QUADRANTS = ("bad_and_harmful", "bad_but_not_harmful", "clean_but_harmful", "clean_and_not_harmful")
HARMFUL_Z = 1.0  # the analysis.metric_groups threshold (declared)
N_BOOT = 1000
CLEAN = "clean"
# H1-H8 of vlapoc.methods.heuristics: (name, descriptor column in clusters.descriptor_table, sign: +1 high is suspicious, -1 low is suspicious)
FILTER_HEURISTICS = (("H1_length_steps", "length_steps", 1), ("H2_idle_steps", "idle_steps", 1), ("H3_gripper_toggles", "gripper_toggles", 1),
                     ("H4_smoothness", "smoothness", 1), ("H5_duplicate_frame_pairs", "duplicate_frame_pairs", 1),
                     ("H6_instruction_mismatch", "instruction_mismatch", 1), ("H7_length_vs_task_highest", "length_vs_task", 1),
                     ("H7_length_vs_task_lowest", "length_vs_task", -1), ("H8_post_release_share_vs_task", "post_release_share_vs_task", 1))


def zscore(x) -> np.ndarray:
    """(x - mean) / s.d. (ddof 0), the scale of the ``metric_groups`` threshold."""
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std(ddof=0)


def quadrant_of(bad: np.ndarray, z: np.ndarray, harmful_z: float = HARMFUL_Z) -> np.ndarray:
    """The four §9.2 quadrants of one metric: ``bad`` is the looks-bad flag, harmful means ``z >= harmful_z``."""
    harmful = np.asarray(z) >= harmful_z
    bad = np.asarray(bad, bool)
    return np.where(bad & harmful, QUADRANTS[0], np.where(bad & ~harmful, QUADRANTS[1], np.where(~bad & harmful, QUADRANTS[2], QUADRANTS[3])))


def boot_mean_ci(v: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of a mean (the method of ``analysis.aggregate_by``)."""
    v = np.asarray(v, float)
    if len(v) < 2:
        return float(v.mean()), float(v.mean())
    b = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def boot_diff_ci(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of mean(a) - mean(b), resampling each group independently."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    da = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(1) if len(a) > 1 else np.full(n_boot, a.mean())
    db = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(1) if len(b) > 1 else np.full(n_boot, b.mean())
    d = da - db
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def welch_diff_ci(a: np.ndarray, b: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Welch t interval for mean(a) - mean(b) (scipy ``t.ppf``, Welch-Satterthwaite degrees of freedom): a cross-check of ``boot_diff_ci``.
    Measured on this repo's setting (13 vs 869 episodes, normal null, 1500 trials; ``artifacts/s9_null_calibration_SYNTHETIC.json``): the
    percentile bootstrap interval excludes 0 in 8.5% of trials at nominal 5%, the Welch interval in 4.9%. The reported ``reading`` follows
    the plan (percentile bootstrap); ``reading_welch`` is stored beside it so the difference is visible."""
    from scipy import stats

    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = float(np.sqrt(va + vb))
    df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
    tc = float(stats.t.ppf(0.5 + level / 2, df))
    d = float(a.mean() - b.mean())
    return d - tc * se, d + tc * se


def reading_of(lo: float, hi: float) -> str:
    """``harmful`` / ``helpful`` / ``no_detectable_harm`` from whether an interval of (group - reference) excludes 0."""
    return "harmful" if lo > 0 else ("helpful" if hi < 0 else "no_detectable_harm")


def estimated_reading(metric: str, lo: float, hi: float) -> tuple[str, str]:
    """``(reading, sentence)`` from the CI of the type-minus-clean difference (module docstring: the wording is bounded)."""
    ci = f"CI [{lo:.2f}, {hi:.2f}]"
    if lo > 0:
        return "harmful", f"harmful on {metric} (estimated; {ci})"
    if hi < 0:
        return "helpful", f"helpful on {metric} (estimated; {ci})"
    return "no_detectable_harm", f"no detectable harm on {metric} (estimated; {ci})"


def filter_flags(desc: pd.DataFrame, q: float) -> pd.DataFrame:
    """The practical-filter layer: per heuristic of ``FILTER_HEURISTICS`` a boolean column (``desc`` = ``clusters.descriptor_table`` rows):
    True for the top ``q`` of episodes by that heuristic (direction fixed a priori), ties included only if the flagged set stays within
    ``ceil(q * N)``. Returns the flag columns, indexed like ``desc``."""
    n = len(desc)
    k = int(np.ceil(q * n))
    out = {}
    for name, col, sign in FILTER_HEURISTICS:
        v = sign * desc[col].to_numpy(float)
        thr = np.sort(v)[::-1][k - 1]
        out[name] = (v >= thr) if (v >= thr).sum() <= k else (v > thr)
    return pd.DataFrame(out, index=desc.index)


def heuristic_evidence(version: str = "v3", subset: str = "candidate", lineage_root: Path | None = None) -> pd.DataFrame:
    """Per planted type, the AUROC of each label-free heuristic, read from the recorded ``score`` runs of the ground-truth feature in the
    lineage store (their ``recovery`` outputs, sha-checked; never re-typed): columns ``planted_type``, ``heuristic`` (the score column),
    ``direction``, ``auroc``, ``run_id``. The newest run per (column, direction) on ``<version>/<subset>`` is used. Empty if none is recorded.
    This is the "the corruption is visibly bad" evidence (PoC §2.1 item 4 p. 3; owner caveat "corruption too mild")."""
    from vlapoc import results
    from vlapoc.runs import client

    c = client(lineage_root)
    rows, seen = [], set()
    for r in results.find_runs("score", lineage_root):  # newest first
        p = c.get_run(r["run_id"]).data.params
        if p.get("subset") != f"{version}/{subset}" or "recovery" not in r["outputs"] or (p.get("column"), p.get("direction")) in seen:
            continue
        seen.add((p.get("column"), p.get("direction")))
        rec = results.output_json(r["run_id"], "recovery", lineage_root)
        for t, m in rec["per_type"].items():
            rows.append({"planted_type": t, "heuristic": p["column"], "direction": p["direction"], "auroc": float(m["auroc"]), "run_id": r["run_id"]})
    return pd.DataFrame(rows, columns=["planted_type", "heuristic", "direction", "auroc", "run_id"])


def best_heuristics(evidence: pd.DataFrame) -> pd.DataFrame:
    """The best heuristic (highest AUROC) per planted type of a ``heuristic_evidence`` table."""
    if evidence.empty:
        return pd.DataFrame(columns=["planted_type", "heuristic", "direction", "auroc", "run_id"])
    return evidence.sort_values(["planted_type", "auroc"], ascending=[True, False]).drop_duplicates("planted_type").reset_index(drop=True)


def _decl(decls: dict, col: str) -> dict:
    d = decls[col]
    return {"harm_column": col, "functional": d["functional"], "tau": d["tau"], "rollout_subset": d["rollout_subset"], "good_sign": d["good_sign"]}


def episode_table(table: pd.DataFrame, labels: pd.DataFrame, cols: list[str], flags: pd.DataFrame, decls: dict, synthetic: bool) -> pd.DataFrame:
    """One row per episode: planted type (or ``clean``), the parent hash, the filter flags, and per metric the harm, z and both quadrants
    (``quadrant__<m>`` with "bad" = planted, ``quadrant_filter__<m>`` with "bad" = flagged by any of H1-H8)."""
    lab = labels.set_index("episode_id")
    typ = table["episode_id"].map(lab["corruption_type"]).fillna(CLEAN).to_numpy()
    out = table[["episode_id", "content_hash", "task_name"]].copy()
    out["planted_type"] = typ
    out["parent_content_hash"] = table["episode_id"].map(lab["parent_content_hash"]).to_numpy()
    out["looks_bad_filter"] = flags.any(axis=1).to_numpy()
    out["flagged_by"] = [",".join(flags.columns[row]) for row in flags.to_numpy()]
    bad = typ != CLEAN
    for c in cols:
        m = c.removeprefix("harm_")
        z = zscore(table[c])
        out[f"harm__{m}"] = table[c].to_numpy(float)
        out[f"z__{m}"] = z
        out[f"quadrant__{m}"] = quadrant_of(bad, z)
        out[f"quadrant_filter__{m}"] = quadrant_of(out["looks_bad_filter"].to_numpy(), z)
    out["declarations"] = json.dumps({c: _decl(decls, c) for c in cols}, sort_keys=True)
    out["synthetic"] = bool(synthetic)
    return out


def type_table(ep: pd.DataFrame, labels: pd.DataFrame, cols: list[str], decls: dict, best: pd.DataFrame, synthetic: bool, n_boot: int = N_BOOT,
               seed: int = 0) -> pd.DataFrame:
    """Per (planted type, metric): the type's mean z with a bootstrap CI, the clean episodes' mean z, the difference with its CI, the reading,
    the share of members in each quadrant, the label's hypothesis and the best heuristic's AUROC (the visibly-bad evidence)."""
    rng = np.random.default_rng(seed)
    hyp = labels.groupby("corruption_type")["hypothesis"].first().to_dict() if "hypothesis" in labels else {}
    bh = best.set_index("planted_type") if len(best) else pd.DataFrame()
    clean = ep["planted_type"] == CLEAN
    rows = []
    for t in sorted(set(ep["planted_type"]) - {CLEAN}):
        sel = ep["planted_type"] == t
        for c in cols:
            m = c.removeprefix("harm_")
            zt, zc = ep.loc[sel, f"z__{m}"].to_numpy(), ep.loc[clean, f"z__{m}"].to_numpy()
            lo, hi = boot_mean_ci(zt, rng, n_boot)
            clo, chi = boot_mean_ci(zc, rng, n_boot)
            dlo, dhi = boot_diff_ci(zt, zc, rng, n_boot)
            reading, sentence = estimated_reading(m, dlo, dhi)
            wlo, whi = welch_diff_ci(zt, zc)
            q = ep.loc[sel, f"quadrant__{m}"].value_counts(normalize=True)
            rows.append({"planted_type": t, "metric": m, "n": int(sel.sum()), "mean_z": float(zt.mean()), "mean_z_lo": lo, "mean_z_hi": hi,
                         "clean_n": int(clean.sum()), "clean_mean_z": float(zc.mean()), "clean_mean_z_lo": clo, "clean_mean_z_hi": chi,
                         "diff": float(zt.mean() - zc.mean()), "diff_lo": dlo, "diff_hi": dhi, "reading": reading, "estimated_reading": sentence,
                         "diff_lo_welch": wlo, "diff_hi_welch": whi, "reading_welch": reading_of(wlo, whi),
                         "share_bad_and_harmful": float(q.get(QUADRANTS[0], 0.0)), "share_bad_but_not_harmful": float(q.get(QUADRANTS[1], 0.0)),
                         "hypothesis": hyp.get(t), "best_heuristic": bh.loc[t, "heuristic"] if t in getattr(bh, "index", []) else None,
                         "best_heuristic_direction": bh.loc[t, "direction"] if t in getattr(bh, "index", []) else None,
                         "best_heuristic_auroc": float(bh.loc[t, "auroc"]) if t in getattr(bh, "index", []) else None,
                         **_decl(decls, c), "synthetic": bool(synthetic)})
    return pd.DataFrame(rows)


def filter_table(ep: pd.DataFrame, flags: pd.DataFrame, cols: list[str], decls: dict, synthetic: bool, n_boot: int = N_BOOT, seed: int = 0) -> pd.DataFrame:
    """The secondary layer, "what a practical filter would have dropped, and whether it mattered": per filter (``any_H1_H8`` and each
    heuristic) and metric: how many episodes it flags and how many of those are planted, the mean z of flagged vs unflagged episodes with the
    difference's CI and the same bounded reading, and the filter-based quadrant counts."""
    rng = np.random.default_rng(seed)
    sets = {"any_H1_H8": ep["looks_bad_filter"].to_numpy()} | {h: flags[h].to_numpy() for h in flags.columns}
    planted = (ep["planted_type"] != CLEAN).to_numpy()
    rows = []
    for name, f in sets.items():
        if f.sum() == 0:
            continue
        for c in cols:
            m = c.removeprefix("harm_")
            z = ep[f"z__{m}"].to_numpy()
            dlo, dhi = boot_diff_ci(z[f], z[~f], rng, n_boot) if (~f).sum() else (np.nan, np.nan)
            reading, sentence = estimated_reading(m, dlo, dhi) if (~f).sum() else ("n/a", "n/a")
            quad = quadrant_of(f, z)
            rows.append({"filter": name, "metric": m, "n_flagged": int(f.sum()), "share_flagged": float(f.mean()), "n_flagged_planted": int((f & planted).sum()),
                         "precision_vs_planted": float((f & planted).sum() / f.sum()), "recall_of_planted": float((f & planted).sum() / max(planted.sum(), 1)),
                         "mean_z_flagged": float(z[f].mean()), "mean_z_unflagged": float(z[~f].mean()) if (~f).sum() else np.nan, "diff": float(z[f].mean() - z[~f].mean()) if (~f).sum() else np.nan,
                         "diff_lo": dlo, "diff_hi": dhi, "reading": reading, "estimated_reading": sentence,
                         **{f"n_{qn}": int((quad == qn).sum()) for qn in QUADRANTS}, **_decl(decls, c), "synthetic": bool(synthetic)})
    return pd.DataFrame(rows)


def clean_harmful(ep: pd.DataFrame, cols: list[str], decls: dict, synthetic: bool, top: int = 20) -> pd.DataFrame:
    """Per metric, the ``top`` clean episodes with the highest harm z: the candidates for "looks clean but hurts"."""
    rows = []
    cl = ep[ep["planted_type"] == CLEAN]
    for c in cols:
        m = c.removeprefix("harm_")
        d = cl.sort_values(f"z__{m}", ascending=False).head(top)
        for rank, (_, r) in enumerate(d.iterrows(), 1):
            rows.append({"metric": m, "rank": rank, "episode_id": r["episode_id"], "content_hash": r["content_hash"], "task_name": r["task_name"],
                         "harm": float(r[f"harm__{m}"]), "z": float(r[f"z__{m}"]), "looks_bad_filter": bool(r["looks_bad_filter"]), "flagged_by": r["flagged_by"],
                         **_decl(decls, c), "synthetic": bool(synthetic)})
    return pd.DataFrame(rows)


def examples(ep: pd.DataFrame, cols: list[str], decls: dict, best: pd.DataFrame, labels: pd.DataFrame, synthetic: bool) -> dict:
    """Per planted type and metric, the most and the least harmful member, each with the clean original's hash (the same ``episode_id`` in the
    parent version, ``parent_content_hash``) for side-by-side video, plus the type's visibly-bad evidence (the best heuristic's AUROC)."""
    hyp = labels.groupby("corruption_type")["hypothesis"].first().to_dict() if "hypothesis" in labels else {}
    bh = best.set_index("planted_type") if len(best) else pd.DataFrame()
    out = {"synthetic": bool(synthetic), "types": {}}
    for t in sorted(set(ep["planted_type"]) - {CLEAN}):
        sub = ep[ep["planted_type"] == t]
        entry = {"hypothesis": hyp.get(t), "n": int(len(sub)),
                 "visibly_bad_evidence": ({"best_heuristic": bh.loc[t, "heuristic"], "direction": bh.loc[t, "direction"], "auroc": float(bh.loc[t, "auroc"]),
                                           "source_run_id": bh.loc[t, "run_id"]} if t in getattr(bh, "index", []) else None), "by_metric": {}}
        for c in cols:
            m = c.removeprefix("harm_")
            s = sub.sort_values(f"z__{m}", ascending=False)
            pick = lambda r, which: {  # noqa: E731
                "which": which, "episode_id": r["episode_id"], "content_hash": r["content_hash"], "original_episode_id": r["episode_id"],
                "original_content_hash": r["parent_content_hash"], "z": float(r[f"z__{m}"]), "harm": float(r[f"harm__{m}"]), "task_name": r["task_name"],
                "quadrant": r[f"quadrant__{m}"]}
            entry["by_metric"][m] = {"declaration": _decl(decls, c), "most_harmful": pick(s.iloc[0], "most harmful member"), "least_harmful": pick(s.iloc[-1], "least harmful member")}
        out["types"][t] = entry
    return out


def quadrant_counts(ep: pd.DataFrame, cols: list[str]) -> dict:
    """``{metric: {quadrant: n}}`` for the label-based and the filter-based definition of "looks bad"."""
    res = {"labels": {}, "filter": {}}
    for c in cols:
        m = c.removeprefix("harm_")
        res["labels"][m] = {q: int((ep[f"quadrant__{m}"] == q).sum()) for q in QUADRANTS}
        res["filter"][m] = {q: int((ep[f"quadrant_filter__{m}"] == q).sum()) for q in QUADRANTS}
    return res


def discrepancy_run(score_run_id: str, version: str, name: str, subset: str = "candidate", metrics: list[str] | None = None,
                    out_dir: Path | None = None, lineage_root: Path | None = None, chain: str | None = None, allow_dirty: bool = False,
                    synthetic: bool = False, n_boot: int = N_BOOT, cache_dir: Path | None = None, heuristics: bool = True) -> dict:
    """Stage ``score`` for §9.2 (dashboard S9): the quadrant table, the per-type readings, the filter layer, the clean-but-harmful candidates
    and the example pairs, recorded as one MLflow run that consumes the version and the score table's sha256.

    Refuses (nothing written) unless: the score run is a FINISHED ``attribute`` run (a seed or an ensemble) with ``functionals.json``, the
    version has an answer key, and the score table keys exactly ``<version>/<subset>`` (``lineage.check_score_keys``). ``synthetic`` (or a
    synthetic score run) labels every output and tags the run ``vlapoc.synthetic=true``. ``heuristics=False`` skips the H1-H8 filter layer
    (the frame scan costs ~85 s per 960 episodes the first time; cached after). Returns ``{"run_id", "out_dir", "quadrants", "readings"}``."""
    from vlapoc import results
    from vlapoc.clusters import DEFAULT_CACHE, descriptor_table, harm_columns
    from vlapoc.groundtruth import load_answer_key
    from vlapoc.lineage import Subset, check_score_keys, rel
    from vlapoc.method_data import open_for_method
    from vlapoc.runs import client, resolve_version, stage_run

    run = client(lineage_root).get_run(score_run_id)
    if run.info.status != "FINISHED" or run.data.tags.get("stage") != "attribute" or "output.attribution_scores.sha256" not in run.data.tags:
        raise ValueError(f"run {score_run_id[:12]} is not a FINISHED attribute run with a recorded score table (status {run.info.status}, "
                         f"stage {run.data.tags.get('stage')!r}); the answer key is read only after the attribution is recorded (PoC §2.1 item 4 p. 3)")
    table = results.score_table(score_run_id, lineage_root)
    decls = results.functionals(score_run_id, lineage_root)
    v = resolve_version(version, lineage_root)
    sub = Subset.load(f"{v['name']}/{subset}", lineage_root)
    labels = load_answer_key(version, lineage_root)  # LineageError if the version has no answer key
    cols = harm_columns(table, metrics)
    check_score_keys(sub, table, cols[0], lineage_root)  # the table must key exactly the subset's episode versions
    undeclared = [c for c in cols if c not in decls]
    if undeclared:
        raise ValueError(f"columns {undeclared} have no declared functional (PoC §3.4 p. 7)")
    synthetic = bool(synthetic) or run.data.params.get("synthetic") == "True" or run.data.tags.get("vlapoc.synthetic") == "true"
    q = len(labels) / len(sub.episode_ids)
    if heuristics:
        view = open_for_method(sub.subset_id, lineage_root)  # the filter layer's descriptors are method-facing data only
        desc = descriptor_table(view, True, cache_dir if cache_dir is not None else DEFAULT_CACHE).set_index("episode_id").loc[table["episode_id"]].reset_index()
        flags = filter_flags(desc, q)
    else:
        flags = pd.DataFrame({"none": np.zeros(len(table), bool)})
    table = table.reset_index(drop=True)
    flags.index = table.index
    evidence = heuristic_evidence(version, subset, lineage_root)
    best = best_heuristics(evidence)
    ep = episode_table(table, labels, cols, flags, decls, synthetic)
    types = type_table(ep, labels, cols, decls, best, synthetic, n_boot)
    filt = filter_table(ep, flags, cols, decls, synthetic, n_boot) if heuristics else pd.DataFrame()
    ch = clean_harmful(ep, cols, decls, synthetic)
    ex = examples(ep, cols, decls, best, labels, synthetic)
    qc = quadrant_counts(ep, cols)
    summary = {"synthetic": synthetic, "bad_definition": {"primary": "planted corruption label of the answer key (read after attribution)",
                                                         "secondary": "flagged by any of H1-H8 at the top q of each heuristic, q = injection rate"},
               "harmful_z": HARMFUL_Z, "filter_q": q, "n_flagged_filter": int(ep["looks_bad_filter"].sum()), "n": len(ep), "n_planted": int((ep["planted_type"] != CLEAN).sum()),
               "quadrant_counts": qc, "readings": {f"{r.planted_type}|{r.metric}": r.estimated_reading for r in types.itertuples()},
               "claim_limit": "estimated from first-order influence (CUPID p. 11); an interval spanning zero is not proof of no harm; Round B tests removal"}
    out_dir = Path(out_dir) if out_dir else Path("data") / "attribution" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"discrepancy_episodes": out_dir / "discrepancy_episodes.parquet", "discrepancy_types": out_dir / "discrepancy_types.parquet",
             "clean_harmful": out_dir / "clean_harmful.parquet", "examples": out_dir / "examples.json", "discrepancy_summary": out_dir / "discrepancy_summary.json"}
    ep.to_parquet(files["discrepancy_episodes"], index=False)
    types.to_parquet(files["discrepancy_types"], index=False)
    ch.to_parquet(files["clean_harmful"], index=False)
    files["examples"].write_text(json.dumps(ex, indent=1, ensure_ascii=False))
    files["discrepancy_summary"].write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    if heuristics:
        files["discrepancy_filter"] = out_dir / "discrepancy_filter.parquet"
        filt.to_parquet(files["discrepancy_filter"], index=False)
    with stage_run("score", name, lineage_root, chain=chain, allow_dirty=allow_dirty, extra_tags={"vlapoc.synthetic": "true"} if synthetic else None) as r:
        r.input_version(v["dataset_version_id"], "scored_version")
        r.input_entity("score_table", run.data.tags["output.attribution_scores.sha256"], "scores")
        r.params({"kind": "discrepancy (PoC §9.2: where looks-bad and hurts disagree), estimated", "score_run_id": score_run_id, "version": version, "subset": subset,
                  "columns": cols, "harmful_z": HARMFUL_Z, "filter_q": q, "filter_heuristics": [h[0] for h in FILTER_HEURISTICS] if heuristics else [],
                  "n_boot": n_boot, "synthetic": synthetic, "heuristic_evidence_runs": sorted(set(evidence["run_id"])) if len(evidence) else [],
                  "bad_definition": summary["bad_definition"], "claim_limit": summary["claim_limit"]})
        for k, p in files.items():
            r.output_file(k, p, kind="file")
        return {"run_id": r.run_id, "out_dir": rel(out_dir), "quadrants": qc["labels"], "readings": summary["readings"]}
