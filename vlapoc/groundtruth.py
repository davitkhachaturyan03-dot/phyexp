"""Ground-truth side of the injection experiment: the answer key (ground-truth feature S3).

Grounding
---------
- PoC §2.1 item 4 (p. 3): "This turns 'the method says X' into 'the method recovered 90% of the injected bad
  episodes.'" The labels written by ``derive.derive_version`` are the answer key; recovery is scored against
  them (S5).
- Owner decision 3 (brief.md, 2026-10-15): the key is readable only through this module; method code goes
  through ``vlapoc.method_data.open_for_method`` and never imports this file (boundary test
  ``scripts/verify_method_boundary_s3.py``).

``load_answer_key(version)`` is the only label reader for evaluation code. It wraps ``derive.load_labels`` and
cross-checks the labels against the lineage-only ``parents.parquet`` (``lineage.parent_hashes``).
S4 adds the seeded stratified selection (``select_injection``), the injected-version derivation
(``derive_injection`` / ``derive_injection_run``) and a CLI::

    .venv/bin/python -m vlapoc.groundtruth derive --parent v0 --name v2 --per-type 13 --seed 0

Grounding for the selection (S4): PoC §2.5 (p. 5) "Inject 5-10% corrupted episodes" and §10 (p. 20) "Ground
truth | Injection experiment with 5-10% corrupted episodes, using LIBERO's original demos as clean references";
§8.7 (p. 15) "Treat LIBERO's original demos as clean; corrupt only the training copy" (only the parent's
``candidate`` split is drawn from; ``derive_version`` also refuses reference episodes); RoboDrop App. A.1 (p. 13)
"corrupting 10% of the episodes for every task" (each task gets 4 or 5 corrupted episodes here). Design D2/D3 of
the plan: one combined version with all seven types, 13 episodes per type, 91 of 960 candidates = 9.48%.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from vlapoc import corruptions as C
from vlapoc.catalog import read_episodes_meta
from vlapoc.derive import derive_version, load_labels
from vlapoc.libero_convert import REPO_ROOT
from vlapoc.lineage import LineageError, Subset, parent_hashes, resolve_version, version_episodes


def load_answer_key(version: str, lineage_root: Path | None = None) -> pd.DataFrame:
    """The answer key of a derived version: its ``labels.parquet`` rows (``episode_id``, ``parent_content_hash``,
    ``content_hash``, ``corruption_type``, ``params`` JSON, ``seed``, and ``hypothesis`` where present).

    Cross-checks before returning: every label's ``parent_content_hash`` equals the version's
    ``parents.parquet`` entry, and its ``content_hash`` equals the version's episode table entry for that
    ``episode_id``; ``LineageError`` otherwise. Ground-truth side only.
    """
    labels = load_labels(version, lineage_root)
    ph = parent_hashes(version, lineage_root).set_index("episode_id")["parent_content_hash"]
    ep = version_episodes(version, lineage_root).set_index("episode_id")["content_hash"]
    ids = labels["episode_id"]
    if not (labels["parent_content_hash"].to_numpy() == ph.loc[ids].to_numpy()).all():
        raise LineageError(f"{version}: labels disagree with parents.parquet")
    if not (labels["content_hash"].to_numpy() == ep.loc[ids].to_numpy()).all():
        raise LineageError(f"{version}: labels' content_hash differs from the version's episode table")
    return labels


def candidate_actions(root: Path, episode_indices) -> dict[int, np.ndarray]:
    """``{episode_index: action (T, 7)}`` for the requested episodes, reading only the ``action`` column."""
    want = {int(i) for i in episode_indices}
    meta = read_episodes_meta(root)
    out = {}
    for (ci, fi), g in meta[meta["episode_index"].isin(want)].groupby(["data/chunk_index", "data/file_index"]):
        path = root / "data" / f"chunk-{int(ci):03d}" / f"file-{int(fi):03d}.parquet"
        t = pq.read_table(path, columns=["episode_index", "frame_index", "action"]).to_pandas()
        for idx, grp in t[t["episode_index"].isin(want)].groupby("episode_index"):
            out[int(idx)] = np.stack(grp.sort_values("frame_index")["action"].to_numpy()).astype(np.float32)
    return out


def _task_order(eps: pd.DataFrame, rng: np.random.Generator) -> list[tuple[str, int]]:
    """Tasks (suite, libero_task_id) interleaved across the two suites (spatial, long, spatial, ...) with a seeded
    shuffle inside each suite, so any 13 consecutive entries hold 7 and 6 of the two suites."""
    by = {s: sorted(set(g["libero_task_id"])) for s, g in eps.groupby("suite")}
    suites = sorted(by)
    if len(suites) != 2 or len(by[suites[0]]) != len(by[suites[1]]):
        raise LineageError(f"stratified selection expects two suites with equal task counts, got { {s: len(v) for s, v in by.items()} }")
    perm = {s: [by[s][i] for i in rng.permutation(len(by[s]))] for s in suites}
    return [(s, perm[s][k]) for k in range(len(by[suites[0]])) for s in suites]


def _window_offsets(n_tasks: int, per_type: int, n_types: int) -> list[int]:
    """Start offsets (one per type, cyclic windows of ``per_type`` tasks) whose coverage of every task is
    ``floor`` or ``ceil`` of ``n_types * per_type / n_tasks``; found by scanning evenly spaced starts."""
    total = n_types * per_type
    lo, hi = total // n_tasks, -(-total // n_tasks)
    for step in range(1, n_tasks):
        offs = [(k * step) % n_tasks for k in range(n_types)]
        cover = np.zeros(n_tasks, int)
        for o in offs:
            cover[[(o + j) % n_tasks for j in range(per_type)]] += 1
        if cover.min() >= lo and cover.max() <= hi:
            return offs
    raise LineageError("no balanced window layout")


def select_injection(parent: str = "v0", per_type: int = 13, seed: int = 0, lineage_root: Path | None = None) -> dict:
    """Seeded, stratified assignment (episode -> corruption type) over the parent's ``candidate`` split.

    Rules (S4 ACs): ``per_type`` distinct episodes per type; each type covers ``per_type`` distinct tasks with both
    suites present (cyclic windows over a suite-interleaved task order); every task receives 4 or 5 corrupted
    episodes with no type repeated inside a task; an episode gets at most one edit; every chosen episode is
    eligible for its type (``corruptions.eligible``, on the stored actions). Tasks of a type are the cyclic window
    ``[offset_k, offset_k + per_type)`` of the interleaved order; the episode inside a (type, task) cell is a seeded
    draw among the eligible, still unused candidates of that task.

    Returns ``{"assignment": [{episode_id, episode_index, task_name, suite, libero_task_id, corruption_type}],
    "rule": str, "n_candidates": int, "fraction": float}``. The arrangement of windows, the suite shuffles and the
    draws all come from ``numpy.random.default_rng(seed)``.
    """
    pv = resolve_version(parent, lineage_root)
    cand = Subset.load(f"{pv['name']}/candidate", lineage_root).episodes(lineage_root)
    rng = np.random.default_rng(seed)
    order = _task_order(cand, rng)
    offsets = _window_offsets(len(order), per_type, len(C.TYPES))
    acts = candidate_actions(REPO_ROOT / pv["lerobot_root"], cand["episode_index"])
    used: set[str] = set()
    rows = []
    # scarcest type first, so it cannot be starved by the others
    scarcity = {t: sum(C.eligible(t, {"action": acts[int(i)]}) for i in cand["episode_index"]) for t in C.TYPES}
    for t in sorted(C.TYPES, key=lambda x: scarcity[x]):
        k = C.TYPES.index(t)
        for j in range(per_type):
            suite, tid = order[(offsets[k] + j) % len(order)]
            pool = cand[(cand["suite"] == suite) & (cand["libero_task_id"] == tid) & ~cand["episode_id"].isin(used)]
            pool = [r for r in pool.sort_values("episode_index").itertuples() if C.eligible(t, {"action": acts[int(r.episode_index)]})]
            if not pool:
                raise LineageError(f"no eligible unused candidate for {t} in {suite}/{tid}")
            r = pool[int(rng.integers(len(pool)))]
            used.add(r.episode_id)
            rows.append({"episode_id": r.episode_id, "episode_index": int(r.episode_index), "task_name": r.task_name,
                         "suite": suite, "libero_task_id": int(tid), "corruption_type": t})
    rows.sort(key=lambda x: x["episode_index"])
    rule = (f"{per_type} episodes per type over {len(C.TYPES)} types from {pv['name']}/candidate ({len(cand)}): cyclic windows of "
            f"{per_type} tasks over a suite-interleaved, seeded task order (window offsets {offsets}); within a cell a "
            f"seeded draw among eligible unused candidates; numpy.random.default_rng({seed})")
    return {"assignment": rows, "rule": rule, "n_candidates": len(cand), "fraction": len(rows) / len(cand),
            "eligible_counts": scarcity, "window_offsets": offsets}


def build_edits(parent: str, assignment: list[dict], seed: int, lineage_root: Path | None = None):
    """One ``EpisodeEdit`` per assigned episode (``corruptions.make_edit``), with the selection recorded in ``params``."""
    pv = resolve_version(parent, lineage_root)
    root = REPO_ROOT / pv["lerobot_root"]
    sigma = np.array(json.loads((root / "meta" / "stats.json").read_text())["action"]["std"], dtype=np.float64)
    pe = version_episodes(pv["dataset_version_id"], lineage_root)
    suite_tasks = {s: sorted(g["language_instruction"].unique()) for s, g in pe.groupby("suite")}
    instr = dict(zip(pe["episode_id"], pe["language_instruction"]))
    edits = []
    for a in assignment:
        e = C.make_edit(a["corruption_type"], a["episode_id"], seed, sigma_a=sigma, original_task=instr[a["episode_id"]],
                        same_suite_tasks=suite_tasks[a["suite"]])
        e.params["selection"] = {"stratum": f"{a['suite']}/{a['libero_task_id']}/{a['corruption_type']}",
                                 "sigma_a_source": f"{pv['name']} meta/stats.json action.std"}
        edits.append(e)
    return edits


def derive_injection(parent: str = "v0", name: str = "v2", per_type: int = 13, seed: int = 0,
                     lineage_root: Path | None = None) -> dict:
    """Derive the injected version: all seven types on ``per_type`` candidates each (``select_injection``), through
    ``derive.derive_version`` (writer pass in parent order, ``compare_blind`` before registration, labels with
    ``hypothesis``). Full scale writes a ~7 GB copy next to the parent (plan: ~9 min)."""
    pv = resolve_version(parent, lineage_root)
    sel = select_injection(parent, per_type, seed, lineage_root)
    edits = build_edits(parent, sel["assignment"], seed, lineage_root)
    cand = Subset.load(f"{pv['name']}/candidate", lineage_root)
    out = derive_version(
        pv["dataset_version_id"], edits, name,
        note=(f"ground truth: {len(edits)} corrupted candidates ({per_type} per type x {len(C.TYPES)} types, clipped to [-1, 1]), "
              f"seed {seed}"),
        derivation={"selection_rule": sel["rule"], "selected_from_subset": cand.subset_id, "per_type": per_type, "seed": seed,
                    "types": list(C.TYPES), "clip": [C.CLIP_LOW, C.CLIP_HIGH], "fraction_of_candidates": sel["fraction"],
                    "viewpoint": "out of scope (owner decision 2, 2026-10-15): needs simulator re-rendering"},
        lineage_root=lineage_root)
    return {**out, "selection_rule": sel["rule"], "fraction": sel["fraction"]}


def derive_injection_run(parent: str = "v0", name: str = "v2", per_type: int = 13, seed: int = 0,
                         lineage_root: Path | None = None, chain: str | None = None) -> tuple[str, dict]:
    """Stage ``derive`` (MLflow): ``derive_injection`` recorded like ``runs.derive_run`` (consumes the parent version
    and its ``candidate`` subset; produces the child version, its labels and its three inherited subsets), plus the
    ``labels_sha256`` tag. Refuses a dirty ``vlapoc/`` or ``scripts/`` (``runs.stage_run``)."""
    import mlflow

    from vlapoc.lineage import _root
    from vlapoc.runs import stage_run

    pv = resolve_version(parent, lineage_root)
    cand = Subset.load(f"{pv['name']}/candidate", lineage_root)
    with stage_run("derive", f"derive-{name}", lineage_root, chain=chain) as r:
        r.input_version(pv["dataset_version_id"], "parent")
        r.input_subset(cand, "selection_pool", f"{pv['name']}/candidate")
        out = derive_injection(pv["dataset_version_id"], name, per_type, seed, lineage_root)
        vid = out["dataset_version_id"]
        vdir = _root(lineage_root) / "versions" / vid
        r.params({"parent_version_id": pv["dataset_version_id"], "corruption_types": ",".join(C.TYPES), "per_type": per_type,
                  "seed": seed, "n_edits": out["n_edits"], "clip": "[-1, 1]", "mechanism": out["mechanism"]})
        r.tags({f"derive.timing.{k}": v for k, v in out["timing_s"].items()})
        r.tags({"labels_sha256": out["labels_sha256"], "ground_truth.fraction": f"{out['fraction']:.4f}"})
        r.produced("version", vid, f"version.{name}")
        r.output_file("labels", vdir / "labels.parquet", kind="labels")
        mlflow.log_artifact(str(vdir / "derivation.json"))
        mlflow.log_artifact(str(vdir / "version.json"))
        names = json.loads((_root(lineage_root) / "names.json").read_text())["subsets"]
        for role in ("all", "reference", "candidate"):
            r.produced("subset", names[f"{name}/{role}"], f"subset.{name}/{role}")
        return r.run_id, out


def main(argv: list[str] | None = None) -> int:
    """CLI: ``derive`` (S4)."""
    ap = argparse.ArgumentParser(prog="python -m vlapoc.groundtruth")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("derive", help="derive the injected version (all seven types) from a parent")
    d.add_argument("--parent", default="v0")
    d.add_argument("--name", default="v2")
    d.add_argument("--per-type", type=int, default=13)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--no-track", action="store_true", help="skip the MLflow run")
    a = ap.parse_args(argv)
    if a.cmd == "derive":
        if a.no_track:
            out = derive_injection(a.parent, a.name, a.per_type, a.seed)
            print(json.dumps(out, indent=1, default=str))
        else:
            run_id, out = derive_injection_run(a.parent, a.name, a.per_type, a.seed)
            print(json.dumps({"derive_run_id": run_id, **out}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["load_answer_key", "select_injection", "build_edits", "derive_injection", "derive_injection_run",
           "candidate_actions"]
