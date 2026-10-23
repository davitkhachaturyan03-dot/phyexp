"""Read-side results loader for the dashboard and the cluster stages: everything is read from the lineage store, and every
result carries a fail-closed provenance label (dashboard S1).

Where this sits: after ``vlapoc.attribution`` (and ``vlapoc.analysis`` / ``vlapoc.clusters`` ...) have written their
outputs as MLflow runs, the dashboard (``dashboard/``) and the ``clusters`` stage read them back here and nowhere else. Nothing
here writes to the store. Reading order: ``find_runs``, ``score_table`` / ``functionals`` / ``influence`` / ``rollouts`` /
``verdicts``, then ``provenance``.

Grounding
---------
- PoC §2.7 (p. 5), quoted: "A dashboard: select a failed rollout, see its failure type, the most influential training episodes
  (with video) and their source datasets; or browse a ranked list of harmful episodes, aggregated by any catalog column."
  ``influence`` returns the rollout × episode matrix of one attribute run with its row/column labels (the first view needs it);
  ``rollouts`` joins the eval record and the returns table (failure type, T, censored) to the matrix rows.
- PoC §3.4 (p. 7), quoted: "Every attribution result states which functional it explains" - ``functionals`` returns the
  ``functionals.json`` declaration of every score column, so the dashboard can name the functional next to every number.
- PoC §2.1 item 1 (p. 3), quoted: "Every training run should be tied to exact dataset versions (content hashes) ... Every eval
  rollout should be tied back to that checkpoint." Every artifact read here is checked against the sha256 the producing run
  recorded (``ValueError`` on a mismatch), and ``provenance`` walks ``runs.trace`` so the label comes from the recorded chain.
- PoC §2.1 item 4 (p. 3), "check whether your attribution method flags exactly those": this module is on the method side. It
  imports nothing from ``vlapoc.groundtruth``, ``vlapoc.recovery`` or ``vlapoc.discrepancy`` (a source scan in
  ``scripts/verify_dashboard.py --results`` enforces it).

Provenance rule (fail-closed; owner answer "the SYNTHETIC label disappearing only when the data is real")
-----------------------------------------------------------------------------------------------------
``provenance(run_id)`` walks ``runs.trace(run_id)`` and inspects every run in it:

- **SYNTHETIC** if any run carries tag ``vlapoc.synthetic=true`` or a ``synthetic`` param that is true. Beats everything else.
- **REAL** only if no run is a stand-in (tag ``vlapoc.stand_in=true``), the trace contains at least one ``train`` and one ``eval``
  run, every ``train`` run is tagged ``vlapoc.smoke=false`` and every ``eval`` run ``eval.attribution_grade=true``. A *missing* tag
  counts against REAL.
- **SMOKE** otherwise (real code, not attribution-grade: a smoke run, a stand-in, a run whose chain is not fully recorded). The
  ``reasons`` list names each run and condition that kept the label from being REAL.

The label depends only on what the lineage store recorded, never on a code switch or a command-line flag.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from vlapoc.libero_convert import REPO_ROOT

os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")  # artifact reads here are small and frequent; no progress bars
LABELS = ("REAL", "SMOKE", "SYNTHETIC")


class NotRecordedError(ValueError):
    """A run has no such recorded output (it predates the story that records it, or it is the wrong kind of run)."""


def _client(lineage_root: Path | None):
    from vlapoc.runs import client

    return client(lineage_root)


def _tmpdir() -> tempfile.TemporaryDirectory:
    (REPO_ROOT / "data" / "scratch").mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="results_", dir=REPO_ROOT / "data" / "scratch")


def _fetch(run, tag: str, lineage_root: Path | None, reader):
    """Download output ``tag`` of ``run`` (an MLflow artifact), check its sha256 against the recorded one, and return
    ``reader(path)``. ``NotRecordedError`` if the run recorded no such output; ``ValueError`` if the bytes differ."""
    from vlapoc.catalog import sha256_file

    t = run.data.tags
    if f"output.{tag}.artifact" not in t:
        raise NotRecordedError(f"run {run.info.run_id[:12]} ({run.info.run_name}) has no recorded output {tag!r}")
    with _tmpdir() as tmp:
        path = Path(_client(lineage_root).download_artifacts(run.info.run_id, t[f"output.{tag}.artifact"], tmp))
        if sha256_file(path) != t[f"output.{tag}.sha256"]:
            raise ValueError(f"run {run.info.run_id}: artifact {t[f'output.{tag}.artifact']} differs from its recorded sha256 "
                             f"(output.{tag}.sha256); refusing to read it")
        return reader(path)


def _get_run(run_id: str, lineage_root: Path | None):
    return _client(lineage_root).get_run(run_id)


# ---------------------------------------------------------------------------------------------
# finding runs
# ---------------------------------------------------------------------------------------------


def find_runs(stage: str, lineage_root: Path | None = None, chain: str | None = None, name: str | None = None) -> list[dict]:
    """FINISHED runs of ``stage``, newest first: ``{run_id, name, stage, chain, start_time, synthetic_tag, outputs}``.

    ``chain`` keeps only runs whose ``vlapoc.chain`` tag starts with it (``stage.py`` convention); ``name`` an exact run name.
    ``outputs`` lists the recorded output names (``output.<name>.artifact`` tags), so a caller can see what a run offers."""
    from vlapoc.runs import STAGES, experiment_id

    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    q = f"tags.stage = '{stage}' and attributes.status = 'FINISHED'"
    if name:
        q += f" and tags.mlflow.runName = '{name}'"
    runs = _client(lineage_root).search_runs([experiment_id(lineage_root)], q, order_by=["attributes.start_time DESC"], max_results=5000)
    out = []
    for r in runs:
        t = r.data.tags
        if chain and not t.get("vlapoc.chain", "").startswith(chain):
            continue
        out.append({"run_id": r.info.run_id, "name": r.info.run_name, "stage": stage, "chain": t.get("vlapoc.chain"),
                    "start_time": r.info.start_time, "synthetic_tag": t.get("vlapoc.synthetic") == "true",
                    "outputs": sorted(k[len("output."):-len(".artifact")] for k in t if k.startswith("output.") and k.endswith(".artifact"))})
    return out


def runs_with_param(stage: str, key: str, value: str, lineage_root: Path | None = None) -> list[dict]:
    """FINISHED runs of ``stage`` whose MLflow param ``key`` equals ``value``, newest first (same fields as ``find_runs``): how the dashboard finds the runs
    that were made from another run (a cluster run's ``score_run_id``, a confirm run's ``cluster_run_id``)."""
    from vlapoc.runs import STAGES, experiment_id

    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    q = f"tags.stage = '{stage}' and attributes.status = 'FINISHED' and params.{key} = '{value}'"
    runs = _client(lineage_root).search_runs([experiment_id(lineage_root)], q, order_by=["attributes.start_time DESC"], max_results=5000)
    return [{"run_id": r.info.run_id, "name": r.info.run_name, "stage": stage, "chain": r.data.tags.get("vlapoc.chain"), "start_time": r.info.start_time,
             "synthetic_tag": r.data.tags.get("vlapoc.synthetic") == "true", "params": dict(r.data.params),
             "outputs": sorted(k[len("output."):-len(".artifact")] for k in r.data.tags if k.startswith("output.") and k.endswith(".artifact"))} for r in runs]


def run_summary(run_id: str, lineage_root: Path | None = None) -> dict:
    """``{"name", "stage", "params", "tags"}`` of one run (for display: lineage panel, banner reasons)."""
    r = _get_run(run_id, lineage_root)
    return {"name": r.info.run_name, "stage": r.data.tags.get("stage"), "params": dict(r.data.params), "tags": dict(r.data.tags), "experiment_id": r.info.experiment_id}


def resolve_run(ref: str, stage: str, lineage_root: Path | None = None, chain: str | None = None) -> str:
    """``ref`` itself if it is a run id, else the newest FINISHED ``stage`` run named ``ref`` (``ValueError`` if none)."""
    import re

    if re.fullmatch(r"[0-9a-f]{32}", ref):
        return ref
    runs = find_runs(stage, lineage_root, chain, ref)
    if not runs:
        raise ValueError(f"no FINISHED {stage} run named {ref!r}")
    return runs[0]["run_id"]


# ---------------------------------------------------------------------------------------------
# reading outputs
# ---------------------------------------------------------------------------------------------


def score_table(run_id: str, lineage_root: Path | None = None) -> pd.DataFrame:
    """The episode × metric table (``attribution_scores.parquet``) of an attribute or ensemble run, sha-checked."""
    return _fetch(_get_run(run_id, lineage_root), "attribution_scores", lineage_root, pd.read_parquet)


def functionals(run_id: str, lineage_root: Path | None = None) -> dict[str, dict]:
    """``{column: declaration}`` from the run's ``functionals.json`` (PoC §3.4 p. 7), sha-checked. ``NotRecordedError`` for a
    run that declares none (a stand-in or a pre-training-attribution-S3 run)."""
    rows = _fetch(_get_run(run_id, lineage_root), "functionals", lineage_root, lambda p: json.loads(p.read_text()))
    return {r["column"]: r for r in rows}


def output_table(run_id: str, name: str, lineage_root: Path | None = None) -> pd.DataFrame:
    """Any recorded parquet output ``name`` of a run (e.g. ``clusters``, ``members``), sha-checked."""
    return _fetch(_get_run(run_id, lineage_root), name, lineage_root, pd.read_parquet)


def output_json(run_id: str, name: str, lineage_root: Path | None = None):
    """Any recorded JSON output ``name`` of a run (e.g. ``patterns``), sha-checked."""
    return _fetch(_get_run(run_id, lineage_root), name, lineage_root, lambda p: json.loads(p.read_text()))


def clusters(run_id: str, lineage_root: Path | None = None) -> dict:
    """The outputs of a ``cluster`` stage run (``vlapoc.clusters``): ``{"clusters": DataFrame, "members": DataFrame, "patterns": list,
    "descriptors": DataFrame, "bins": dict}``, each sha-checked (dashboard S2)."""
    run = _get_run(run_id, lineage_root)
    if run.data.tags.get("stage") != "cluster":
        raise NotRecordedError(f"run {run_id[:12]} is a {run.data.tags.get('stage')!r} run, not a cluster run")
    return {"clusters": output_table(run_id, "clusters", lineage_root), "members": output_table(run_id, "members", lineage_root),
            "patterns": output_json(run_id, "patterns", lineage_root), "descriptors": output_table(run_id, "descriptors", lineage_root),
            "bins": output_json(run_id, "bins", lineage_root)["bins"]}


def seed_runs(run_id: str, lineage_root: Path | None = None) -> list[str]:
    """The per-seed attribute run ids behind an ensemble run (param ``attribute_run_ids``), or ``[run_id]`` for a seed run."""
    p = _get_run(run_id, lineage_root).data.params
    return json.loads(p["attribute_run_ids"]) if "attribute_run_ids" in p else [run_id]


def influence(run_id: str, lineage_root: Path | None = None) -> tuple[np.ndarray, dict]:
    """``(matrix, index)`` of an attribute run: the rollout × episode influence matrix M[τ, ξ] (PoC §2.7 p. 5, first view) and
    its labels ``{"rollout_id": [...rows], "episode_id": [...cols], "content_hash": [...cols]}``.

    Both files are checked against the sha256 the run recorded (``ValueError`` on a mismatch). ``NotRecordedError`` (a
    ``ValueError``) for a run that predates dashboard S1 ("not recorded (run predates dashboard S1)") and for an ensemble run
    (which has no matrix of its own: use ``seed_runs``). The matrix shape is checked against the run's
    ``attribute.n_rollouts`` / ``attribute.n_episodes`` tags and the index lengths."""
    run = _get_run(run_id, lineage_root)
    if "attribute_run_ids" in run.data.params:
        raise NotRecordedError(f"run {run_id[:12]} is an ensemble; it has no rollout x episode matrix of its own "
                               f"(per-seed runs: {json.loads(run.data.params['attribute_run_ids'])})")
    if "output.influence_matrix.artifact" not in run.data.tags:
        raise NotRecordedError(f"influence matrix not recorded (run {run_id[:12]} {run.info.run_name!r} predates dashboard S1; "
                               "re-run the attribute stage to record it)")
    m = _fetch(run, "influence_matrix", lineage_root, np.load)
    index = _fetch(run, "influence_index", lineage_root, lambda p: json.loads(p.read_text()))
    shape = (len(index["rollout_id"]), len(index["episode_id"]))
    t = run.data.tags
    if m.shape != shape or (int(t.get("attribute.n_rollouts", shape[0])), int(t.get("attribute.n_episodes", shape[1]))) != shape:
        raise ValueError(f"run {run_id[:12]}: matrix {m.shape}, index {shape}, tags "
                         f"({t.get('attribute.n_rollouts')}, {t.get('attribute.n_episodes')}) disagree")
    return m, index


def rollouts(run_id: str, lineage_root: Path | None = None) -> pd.DataFrame:
    """The rollouts behind an attribute run: its eval run's record joined to the returns table (``failure_type``, ``ret_*``),
    one row per rollout, with ``matrix_row`` (the row of ``influence``'s matrix, NaN for a rollout without a recorded trajectory).
    Everything is read from the eval and returns runs the attribute run's params name, sha-checked."""
    from vlapoc.returns import load_eval_rollouts

    run = _get_run(run_id, lineage_root)
    p = run.data.params
    if "eval_run_ids" in p:  # a run recorded over several eval runs (the dashboard demo's synthetic runs): params hold matching id lists
        pairs = list(zip(json.loads(p["eval_run_ids"]), json.loads(p["returns_run_ids"])))
    elif "eval_run_id" in p:
        pairs = [(p["eval_run_id"], p["returns_run_id"])]
    else:
        raise NotRecordedError(f"run {run_id[:12]} names no eval run (an ensemble has none: use a seed run, see seed_runs)")
    (REPO_ROOT / "data" / "scratch").mkdir(parents=True, exist_ok=True)
    parts = []
    for ev_id, rt_id in pairs:
        ro, _ = load_eval_rollouts(ev_id, lineage_root, make_out_dir=False)
        ret = _fetch(_get_run(rt_id, lineage_root), "returns", lineage_root, pd.read_parquet)
        keep = [c for c in ret.columns if c not in ro.columns or c == "rollout_id"]
        parts.append(ro.merge(ret[keep], on="rollout_id", how="left", validate="one_to_one").assign(eval_run_id=ev_id))
    df = pd.concat(parts, ignore_index=True)
    row = {}
    if "output.influence_index.artifact" in run.data.tags:
        row = {r: i for i, r in enumerate(_fetch(run, "influence_index", lineage_root, lambda q: json.loads(q.read_text()))["rollout_id"])}
    df["matrix_row"] = df["rollout_id"].map(row).astype(float)
    return df


def verdicts(lineage_root: Path | None = None, chain: str | None = None, body: bool = False) -> list[dict]:
    """The ``verify`` runs (two-step verdicts, PoC §4.4 p. 10) of a store: ``{run_id, name, chain, kind, treated, control, seed,
    outcome, direction, family, attribution_grade, label}`` from their tags (the ``step4.*`` tags only for a Step 4
    comparison). ``body=True`` adds ``verdict`` (``verdict.json``, sha-checked; ``diff``, ``ci95``, ``p_holm``, gate per metric)."""
    from vlapoc.runs import experiment_id

    runs = _client(lineage_root).search_runs([experiment_id(lineage_root)], "tags.stage = 'verify' and attributes.status = 'FINISHED'",
                                             order_by=["attributes.start_time DESC"], max_results=5000)
    out = []
    for r in runs:
        t = r.data.tags
        if chain and not t.get("vlapoc.chain", "").startswith(chain):
            continue
        rec = {"run_id": r.info.run_id, "name": r.info.run_name, "chain": t.get("vlapoc.chain"), "kind": t.get("step4.kind"),
               "treated": t.get("step4.treated"), "control": t.get("step4.control"), "seed": t.get("step4.seed"),
               "outcome": t.get("verify.outcome"), "direction": t.get("verify.direction"), "family": t.get("verify.family"),
               "attribution_grade": t.get("verify.attribution_grade") == "true", "label": t.get("verify.label")}
        if body:
            rec["verdict"] = _fetch(r, "verdict", lineage_root, lambda p: json.loads(p.read_text()))
        out.append(rec)
    return out


# ---------------------------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------------------------


def provenance(run_id: str, lineage_root: Path | None = None) -> dict:
    """``{"label": "REAL" | "SMOKE" | "SYNTHETIC", "reasons": [...], "n_runs": n}`` for a run, from its recorded trace (module
    docstring: the fail-closed rule). ``reasons`` names every run and condition that kept the label from being REAL (empty only
    for REAL), each as ``"<kind>: <run name> (<stage>): <condition>"`` with kind ``synthetic`` or ``smoke``."""
    from vlapoc.runs import trace

    tr = trace(run_id, lineage_root)
    c = _client(lineage_root)
    synthetic, smoke = [], []
    stages = {"train": 0, "eval": 0}
    for rid in tr["runs"]:
        r = c.get_run(rid)
        t, p, stage = r.data.tags, r.data.params, r.data.tags.get("stage")
        who = f"{r.info.run_name} ({stage}, {rid[:8]})"
        if t.get("vlapoc.synthetic") == "true":
            synthetic.append(f"synthetic: {who}: tag vlapoc.synthetic=true")
        if str(p.get("synthetic", "")).lower() == "true":
            synthetic.append(f"synthetic: {who}: param synthetic=True")
        if t.get("vlapoc.stand_in") == "true":
            smoke.append(f"smoke: {who}: tag vlapoc.stand_in=true (a stand-in, not the real stage)")
        if stage == "train":
            stages["train"] += 1
            v = t.get("vlapoc.smoke")
            if v != "false":
                smoke.append(f"smoke: {who}: tag vlapoc.smoke is {v!r}, not 'false'")
        if stage == "eval":
            stages["eval"] += 1
            v = t.get("eval.attribution_grade")
            if v != "true":
                smoke.append(f"smoke: {who}: tag eval.attribution_grade is {v!r}, not 'true'")
    for k, n in stages.items():
        if n == 0:
            smoke.append(f"smoke: trace of {run_id[:8]}: no {k} run reached (the chain is not fully recorded)")
    label = "SYNTHETIC" if synthetic else ("SMOKE" if smoke else "REAL")
    return {"label": label, "reasons": synthetic + smoke, "n_runs": len(tr["runs"])}


# ---------------------------------------------------------------------------------------------
# catalog rows, dataset roots and trajectories (dashboard S5)
# ---------------------------------------------------------------------------------------------


def training_episodes(run_id: str, lineage_root: Path | None = None) -> tuple[str, pd.DataFrame]:
    """``(subset_id, catalog rows)`` of the training subset an attribute, ensemble or cluster run consumed (tag ``input.training``): the catalog
    columns of PoC §2.2 p. 4 (``task_name``, ``scene``, ``source``, ``source_file``, ``length_steps`` ...), sorted by ``episode_index``. The catalog is
    bookkeeping (the same columns ``MethodView.catalog`` exposes), not a label."""
    from vlapoc.lineage import Subset

    sid = _get_run(run_id, lineage_root).data.tags["input.training"].split(":", 1)[1]
    return sid, Subset.load(sid, lineage_root).episodes(lineage_root).sort_values("episode_index").reset_index(drop=True)


def dataset_root(subset_id: str, lineage_root: Path | None = None) -> Path:
    """The LeRobot root (absolute) of the dataset version a subset belongs to."""
    from vlapoc.lineage import Subset, resolve_version

    return REPO_ROOT / resolve_version(Subset.load(subset_id, lineage_root).dataset_version_id, lineage_root)["lerobot_root"]


def parent_version(subset_id: str, lineage_root: Path | None = None) -> tuple[Path, pd.DataFrame] | None:
    """``(LeRobot root, episode table)`` of the version a derived version was made from (``parent_version_id`` of ``version.json``), or ``None`` for
    a base version. The episode table has ``episode_id``, ``episode_index``, ``content_hash``: it lets the dashboard show the clean original beside a
    corrupted episode (the answer key's ``parent_content_hash`` points to the same original)."""
    from vlapoc.lineage import Subset, resolve_version, version_episodes

    vid = Subset.load(subset_id, lineage_root).dataset_version_id
    pid = resolve_version(vid, lineage_root).get("parent_version_id")
    if not pid:
        return None
    return REPO_ROOT / resolve_version(pid, lineage_root)["lerobot_root"], version_episodes(pid, lineage_root)


def trajectory_root(eval_run_id: str, lineage_root: Path | None = None, hints: dict | None = None) -> Path | None:
    """Where the recorded trajectories of an eval run live on this machine, or ``None`` (then the dashboard says "rollout video not on this machine":
    the server's trajectories are 11-22 GB per seed and are not copied back). Candidates: ``hints[eval_run_id]`` (a run may record it), then
    ``data/eval/<eval run name>/trajectories`` (the ``stage.py eval`` convention), then a single ``data/eval/<name>-*/trajectories``."""
    name = _get_run(eval_run_id, lineage_root).info.run_name
    cands = [Path(hints[eval_run_id])] if hints and eval_run_id in hints else []
    cands += [REPO_ROOT / "data" / "eval" / name / "trajectories"]
    cands += sorted((REPO_ROOT / "data" / "eval").glob(f"{name}-*/trajectories"))
    for c in cands:
        c = c if c.is_absolute() else REPO_ROOT / c
        if (c / "meta" / "info.json").is_file():
            return c
    return None
