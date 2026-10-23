"""Recovery of planted corruptions by an attribution table: type x harm-column AUROC / recall next to the best heuristic.

Ground-truth side (imports ``vlapoc.groundtruth``): this module reads the answer key and must never be imported by
``vlapoc.influence`` / ``vlapoc.attribution`` or any method code (owner decision 3; boundary scan in
``scripts/verify_training_attribution.py --recovery``).

Where this sits (docs/dataflow.md, the injection test, PoC §2.1 item 4): after attribution, only on a version that carries an answer
key (v3). ``scripts/stage.py score`` calls ``recovery_run``; ``recovery_matrix`` does the scoring. ``EXPECTED_METRICS`` below was
committed before any scoring code existed (see ``b4790ee``), so the hypotheses cannot have been fitted to results. ``cluster_recovery`` / ``cluster_recovery_run`` (dashboard S3, below) score the clusters of
``vlapoc.clusters`` against the same answer key, after they were discovered.

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

import json  # noqa: E402
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
    from vlapoc.lineage import rel
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


# ---------------------------------------------------------------------------------------------
# discovered clusters vs the planted types (dashboard S3)
# ---------------------------------------------------------------------------------------------

ANY_PLANTED = "any_planted"  # pseudo type: every labelled episode (the pooled injection)


def ground_truth_reads(trace: dict, lineage_root: Path | None = None) -> list[str]:
    """Names of the runs in a ``runs.trace`` that read the answer key or scored against it: any ``score`` stage run (recovery /
    cluster recovery), any run with an input ``scored_version`` or a tag ``vlapoc.groundtruth*``. A cluster run whose upstream
    shows one is refused by ``cluster_recovery_run`` (its discovery was not blind; PoC §2.1 item 4 p. 3)."""
    from vlapoc.runs import client

    c = client(lineage_root)
    hits = []
    for rid in trace["runs"]:
        t = c.get_run(rid).data.tags
        if t.get("stage") == "score" or "input.scored_version" in t or any(k.startswith("vlapoc.groundtruth") for k in t):
            hits.append(f"{trace['runs'][rid]['name']} ({t.get('stage')})")
    return hits


def cluster_recovery(version: str, clusters: pd.DataFrame, members: pd.DataFrame, subset: str = "candidate",
                     lineage_root: Path | None = None) -> pd.DataFrame:
    """Score discovered clusters against the planted corruption types of ``version`` (ground-truth side; run it only after the clusters
    exist). One row per (cluster, planted type) plus the pseudo type ``any_planted``:

    - ``n_cluster``, ``n_type`` (the type's episodes in the subset), ``n_overlap``, ``n_subset``;
    - ``precision`` = overlap / cluster size (the share of the cluster that is of that type), ``recall`` = overlap / type size,
      ``base_rate`` = type size / subset size and ``lift`` = precision / base_rate;
    - ``p_fisher``: one-sided Fisher exact p (scipy ``fisher_exact``, alternative "greater": the cluster holds more of the type than
      chance), on the 2x2 table (in cluster, of type);
    - ``is_best_type_for_cluster`` (the real type with the smallest p, ties by higher lift) and ``is_best_cluster_for_type`` (per type
      the cluster with the smallest p, ties by higher lift);
    - the cluster's ``lens``, ``definition``, ``direction``, ``target_metric``, ``declarations`` and ``synthetic`` columns, copied.

    ``LineageError`` unless ``version`` has an answer key and every member (``episode_id``, ``content_hash``) is an episode version of
    ``<version>/<subset>`` (so a cluster found on another version, whose hashes differ, is refused)."""
    from scipy.stats import fisher_exact

    from vlapoc.groundtruth import load_answer_key
    from vlapoc.lineage import LineageError, Subset, resolve_version

    v = resolve_version(version, lineage_root)
    labels = load_answer_key(version, lineage_root)  # LineageError if the version has no answer key
    sub = Subset.load(f"{v['name']}/{subset}", lineage_root)
    keys = set(zip(sub.keys(lineage_root)["episode_id"], sub.keys(lineage_root)["content_hash"]))
    mk = set(zip(members["episode_id"], members["content_hash"]))
    if not mk <= keys:
        raise LineageError(f"{len(mk - keys)} cluster members are not episode versions of {v['name']}/{subset} "
                           "(the clusters were found on another version or subset)")
    typ = dict(zip(labels["episode_id"], labels["corruption_type"]))
    ids = list(sub.episode_ids)
    n = len(ids)
    type_of = pd.Series([typ.get(i, "") for i in ids], index=ids)
    types = sorted(set(labels["corruption_type"]))
    type_n = {t: int((type_of == t).sum()) for t in types} | {ANY_PLANTED: int((type_of != "").sum())}
    rows = []
    for _, c in clusters.iterrows():
        mem = members.loc[members["cluster_id"] == c["cluster_id"], "episode_id"]
        for t in [*types, ANY_PLANTED]:
            in_type = (type_of != "") if t == ANY_PLANTED else (type_of == t)
            k = int(in_type.loc[list(mem)].sum())
            nc, kt = len(mem), type_n[t]
            base = kt / n
            prec = k / nc if nc else 0.0
            p = fisher_exact([[k, nc - k], [kt - k, n - nc - (kt - k)]], alternative="greater")[1]
            rows.append({"cluster_id": c["cluster_id"], "planted_type": t, "n_cluster": nc, "n_type": kt, "n_overlap": k, "n_subset": n,
                         "precision": prec, "recall": k / kt if kt else 0.0, "base_rate": base, "lift": prec / base if base else 0.0,
                         "p_fisher": float(p), **{f: c[f] for f in ("lens", "definition", "direction", "target_metric", "declarations", "synthetic") if f in c}})
    out = pd.DataFrame(rows)
    real = out[out["planted_type"] != ANY_PLANTED]
    best_t = real.sort_values(["cluster_id", "p_fisher", "lift"], ascending=[True, True, False]).drop_duplicates("cluster_id").set_index("cluster_id")["planted_type"].to_dict()
    best_c = real.sort_values(["planted_type", "p_fisher", "lift"], ascending=[True, True, False]).drop_duplicates("planted_type").set_index("planted_type")["cluster_id"].to_dict()
    out["is_best_type_for_cluster"] = [best_t.get(r.cluster_id) == r.planted_type for r in out.itertuples()]
    out["is_best_cluster_for_type"] = [best_c.get(r.planted_type) == r.cluster_id for r in out.itertuples()]
    return out


def cluster_recovery_run(cluster_run_id: str, version: str, name: str, subset: str = "candidate", out_dir: Path | None = None,
                         lineage_root: Path | None = None, chain: str | None = None, allow_dirty: bool = False, synthetic: bool = False) -> dict:
    """Stage ``score`` for a cluster run (dashboard S3; PoC §2.1 item 4 p. 3 "check whether your attribution method flags exactly those",
    §4.5 p. 11; ground-truth side): ``cluster_recovery`` of a ``cluster`` run's clusters and members against ``version``'s answer key,
    recorded as one run that consumes the version, the cluster run's ``members`` and ``clusters`` sha256 and the score table behind it.

    Refuses (``ValueError``/``LineageError``, nothing written) unless the cluster run is FINISHED, is a ``cluster`` stage run with recorded
    members, its version has an answer key, and its ``runs.trace`` shows no ground-truth read (``ground_truth_reads``). The scoring run is
    created after the cluster run and the cluster stage (``vlapoc.clusters``) has no code path to this function (a source scan in
    ``scripts/verify_dashboard.py --planted``). ``synthetic`` (or a synthetic cluster run) tags the run ``vlapoc.synthetic=true``."""
    from vlapoc import results
    from vlapoc.lineage import rel
    from vlapoc.runs import client, resolve_version, stage_run, trace

    run = client(lineage_root).get_run(cluster_run_id)
    if run.info.status != "FINISHED" or run.data.tags.get("stage") != "cluster" or "output.members.sha256" not in run.data.tags:
        raise ValueError(f"run {cluster_run_id[:12]} is not a FINISHED cluster run with recorded members (status {run.info.status}, stage {run.data.tags.get('stage')!r})")
    hits = ground_truth_reads(trace(cluster_run_id, lineage_root), lineage_root)
    if hits:
        raise ValueError(f"cluster run {cluster_run_id[:12]} has a ground-truth read upstream ({hits}); its clusters are not blind discoveries (PoC §2.1 item 4 p. 3)")
    co = results.clusters(cluster_run_id, lineage_root)
    table = cluster_recovery(version, co["clusters"], co["members"], subset, lineage_root)
    synthetic = bool(synthetic) or run.data.params.get("synthetic") == "True" or run.data.tags.get("vlapoc.synthetic") == "true"
    table["synthetic"] = synthetic
    out_dir = Path(out_dir) if out_dir else Path("data") / "attribution" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "cluster_recovery.parquet"
    table.to_parquet(path, index=False)
    real = table[table["planted_type"] != ANY_PLANTED]
    summary = {"synthetic": synthetic, "note": "overlap of discovered clusters with the planted types (answer key read after discovery)",
               "per_cluster_best_type": {r.cluster_id: {"planted_type": r.planted_type, "precision": r.precision, "recall": r.recall, "lift": r.lift, "p_fisher": r.p_fisher}
                                         for r in real[real["is_best_type_for_cluster"]].itertuples()},
               "per_type_best_cluster": {r.planted_type: {"cluster_id": r.cluster_id, "definition": r.definition, "precision": r.precision, "recall": r.recall,
                                                          "lift": r.lift, "p_fisher": r.p_fisher}
                                         for r in real[real["is_best_cluster_for_type"]].itertuples()}}
    spath = out_dir / "cluster_recovery_summary.json"
    spath.write_text(json.dumps(summary, indent=1, default=str))
    v = resolve_version(version, lineage_root)
    with stage_run("score", name, lineage_root, chain=chain, allow_dirty=allow_dirty, extra_tags={"vlapoc.synthetic": "true"} if synthetic else None) as r:
        r.input_version(v["dataset_version_id"], "scored_version")
        r.input_entity("members", run.data.tags["output.members.sha256"], "members")
        r.input_entity("clusters", run.data.tags["output.clusters.sha256"], "clusters")
        r.params({"cluster_run_id": cluster_run_id, "version": version, "subset": subset, "n_clusters": int(table["cluster_id"].nunique()),
                  "n_types": int(real["planted_type"].nunique()), "synthetic": synthetic,
                  "kind": "cluster_recovery (discovered clusters vs planted types; answer key read after discovery)"})
        r.output_file("cluster_recovery", path, kind="file")
        r.output_file("cluster_recovery_summary", spath, kind="file")
        return {"run_id": r.run_id, "table": rel(path), "n_rows": len(table), "n_clusters": int(table["cluster_id"].nunique())}
