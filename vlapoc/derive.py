"""Derived dataset versions with ground-truth labels: the injection hook (S4).

Grounding
---------
- PoC §2.1 item 4 (p. 3): "Deliberately corrupt known subsets of training data ... Then check whether
  your attribution method flags exactly those." A derived version records exactly which episode
  versions changed and how (``labels.parquet``), so a later method's ranking can be scored against it.
- PoC §2.5 (p. 5): corruption types, "Inject 5–10% corrupted episodes". PoC §10 (p. 20): "Ground truth |
  Injection experiment with 5–10% corrupted episodes, using LIBERO's original demos as clean references".
- PoC §8.7 (p. 15), row "Benchmark demos (PoC)": "Treat LIBERO's original demos as clean; corrupt only
  the training copy". ``derive_version`` refuses any edit to an episode in a ``reference`` split of the
  parent version or any of its ancestors (PoC §8.3 p. 14: references are "separate from the candidate
  data being audited").
- PoC §8.10 (p. 17): "pad some with idle or hesitant segments", so edits may change episode length;
  LeRobot recomputes ``meta/episodes`` and stats for the rewritten episodes.
- RoboDrop App. A.1 (p. 13): "corrupting 10% of the episodes for every task. Action corruption adds
  zero-mean Gaussian noise with standard deviation 0.25σa to the demonstrated actions, where σa is the
  empirical action standard deviation." ``robodrop_action_noise`` implements this literally, as a
  lineage demo only. The corruption suite (time shift, instruction swap, viewpoint, hesitant demos,
  gripper-release noise) is a separate, owner-approved later feature (brief.md, 2026-09-29).

Mechanism (LeRobot tools, no hand-written parquet surgery)
-----------------------------------------------------------
1. ``lerobot.datasets.dataset_tools.delete_episodes``: copy of the parent without the edited episodes
   (parquet rows copied, PNG bytes untouched).
2. ``LeRobotDataset.create`` / ``add_frame`` / ``save_episode``: the edited episodes, from the parent's
   decoded frames passed through each edit's transform.
3. ``lerobot.datasets.dataset_tools.merge_datasets``: kept + edited, in that order.
Measured on lerobot 0.6.1 (smoke dataset): kept hashes unchanged and in order, edited episodes appended,
length-changing edit consistent in ``meta/episodes``; no crash (cf. upstream issue #2283).

Identity is re-established by content, not assumed: every kept episode must hash to its parent hash and
every edited episode to the hash of its transformed arrays, or the derivation fails.

Unedited episodes keep their ``content_hash``. LeRobot's ``episode_index`` is positional: edited
episodes move to the end of the derived dataset, and the version's episodes table maps them back.

Where it sits
-------------
The "corrupt 5-10% of a copy, ground-truth labels created here" box of docs/dataflow.md. It takes a
registered version (``vlapoc.lineage``) and writes a child version next to it:
``data/lerobot/<parent>__<name>/`` plus ``data/lineage/versions/<id>/`` with ``labels.parquet`` (the
ground truth) and ``derivation.json``. The parent's reference/candidate split is carried over, so
the child's ``<name>/candidate`` is the training set later stages use. ``vlapoc.runs.derive_run``
records a derivation as an MLflow run.

Usage (from the repo root; the v1 demo derivation, ~5 min, writes a ~6.8 GB copy)::

    .venv/bin/python -m vlapoc.lineage derive --parent v0 --name v1 --demo-action-noise --per-task 5 --seed 0

Other edits go through ``derive_version(parent, [EpisodeEdit, ...], name)`` from Python.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from vlapoc.catalog import (
    DEFAULT_CATALOG_ROOT,
    IMAGE_KEYS,
    decode_png,
    episode_content_hash,
    hash_stored_dataset,
    read_episodes_meta,
)
from vlapoc.libero_convert import REPO_ROOT, SUITES
from vlapoc.lineage import (
    LineageError,
    Subset,
    _bind_name,
    _demo_number,
    _root,
    _write_json,
    isolated_hf_datasets_cache,
    register_version,
    rel,
    resolve_version,
    sha256_json,
    verify_version,
    version_episodes,
)


@dataclass(frozen=True)
class EpisodeEdit:
    """One ground-truth edit: which episode, what was done (type, params, seed), and the transform.

    ``transform`` receives the episode as ``read_stored_episode`` returns it and must return the same
    keys with the same per-frame shapes and dtypes; it may change the number of frames but not the
    task string. ``corruption_type``, ``params`` and ``seed`` are written to ``labels.parquet``.
    """

    episode_id: str
    corruption_type: str
    params: dict
    seed: int | None
    transform: Callable[[dict], dict]  # canonical episode dict -> canonical episode dict


# ---------------------------------------------------------------------------------------------
# reading stored episodes (canonical form: task, action, state, image, image2)
# ---------------------------------------------------------------------------------------------


def read_stored_episode(root: Path, episode_index: int) -> dict:
    """Decode one episode of a stored LeRobot dataset into the canonical arrays used for hashing.

    Returns ``task`` (str), ``action`` (T, 7) and ``state`` (T, 8) float32, ``image`` and ``image2``
    (T, 128, 128, 3) uint8: the canonical keys ``vlapoc.libero_convert.load_raw_episode`` also returns,
    and the dict an ``EpisodeEdit.transform`` receives. Reads only the one data file holding the episode.
    """
    root = Path(root)
    meta = read_episodes_meta(root)
    r = meta[meta["episode_index"] == episode_index].iloc[0]
    path = root / "data" / f"chunk-{int(r['data/chunk_index']):03d}" / f"file-{int(r['data/file_index']):03d}.parquet"
    t = pq.read_table(path, filters=[("episode_index", "=", int(episode_index))]).to_pandas()
    g = t.sort_values("frame_index")
    return {
        "task": r["tasks"][0],
        "action": np.stack(g["action"].to_numpy()).astype(np.float32),
        "state": np.stack(g["observation.state"].to_numpy()).astype(np.float32),
        "image": np.stack([decode_png(c) for c in g[IMAGE_KEYS[0]]]),
        "image2": np.stack([decode_png(c) for c in g[IMAGE_KEYS[1]]]),
    }


def _check_canonical(ep: dict, like: dict) -> None:
    """Raise unless an edited episode keeps the canonical layout: equal frame counts across arrays,
    and per-frame shapes and dtypes equal to the source episode's (the length itself may change)."""
    n = len(ep["action"])
    for k in ("action", "state", "image", "image2"):
        if len(ep[k]) != n:
            raise LineageError(f"edited episode: {k} has {len(ep[k])} frames, action has {n}")
        if ep[k].shape[1:] != like[k].shape[1:] or ep[k].dtype != like[k].dtype:
            raise LineageError(f"edited episode: {k} shape/dtype {ep[k].shape[1:]}/{ep[k].dtype} "
                               f"!= {like[k].shape[1:]}/{like[k].dtype}")


# ---------------------------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------------------------


def ancestry(version_ref: str, lineage_root: Path | None = None) -> list[str]:
    """Version ids from ``version_ref`` back to its root (``parent_version_id`` chain)."""
    out = []
    v = resolve_version(version_ref, lineage_root)
    while v is not None:
        out.append(v["dataset_version_id"])
        v = resolve_version(v["parent_version_id"], lineage_root) if v.get("parent_version_id") else None
    return out


def protected_episode_ids(version_ref: str, lineage_root: Path | None = None) -> set[str]:
    """Episode IDs in any ``reference`` split of this version or an ancestor: never to be edited."""
    chain = set(ancestry(version_ref, lineage_root))
    out: set[str] = set()
    sdir = _root(lineage_root) / "splits"
    for p in sorted(sdir.glob("*.json")) if sdir.is_dir() else []:
        rec = json.loads(p.read_text())
        if rec["dataset_version_id"] in chain:
            out |= set(Subset.load(rec["subsets"]["reference"], lineage_root).episode_ids)
    return out


# ---------------------------------------------------------------------------------------------
# derive
# ---------------------------------------------------------------------------------------------


def derive_version(
    parent: str,
    edits: list[EpisodeEdit],
    name: str,
    out_root: Path | None = None,
    note: str | None = None,
    derivation: dict | None = None,
    lineage_root: Path | None = None,
    image_writer_threads: int = 4,
    isolate_hf_cache: bool = True,
) -> dict:
    """Write a new dataset version = parent with ``edits`` applied; register it with labels.

    Returns a summary with the new ``dataset_version_id``, counts and paths. Raises ``LineageError``
    before writing anything if an edit targets a protected (reference) or unknown episode.

    The derived version's episode table holds the parent's catalog columns plus ``parent_content_hash``
    and nothing about the edits: ground truth lives only in ``labels.parquet`` (PoC §2.1 item 4 p.3, the
    method under test must not read the answer; AC4.7).

    ``isolate_hf_cache=True`` (default) routes the HF ``datasets`` caches written by the derivation's
    ``LeRobotDataset`` loads (parent, kept, edit, merged) to a temp dir that is deleted afterwards, so a
    derivation leaves the shared cache unchanged (AC4.8). ``False`` reproduces the round-1 behaviour
    (orphaned caches) and exists only to measure it.

    Args:
        parent: the parent version (name or id).
        edits: one ``EpisodeEdit`` per changed episode; at most one per episode.
        name: name for the child version; its dataset goes to ``<parent root>__<name>`` unless
            ``out_root`` is given (which must not exist yet).
        note: free text stored in ``version.json``.
        derivation: extra keys merged into ``derivation.json`` (e.g. the selection rule).

    Steps: guards -> ``_write_derived_dataset`` (LeRobot tools) -> ``_derived_episode_table`` (identity
    re-established from the stored bytes) -> catalog, registration, ``labels.parquet``,
    ``derivation.json`` and the inherited splits.
    """
    t0 = time.time()
    pv = resolve_version(parent, lineage_root)
    pvid = pv["dataset_version_id"]
    # --- guards: all of these raise before anything is written -------------------------------
    ids = [e.episode_id for e in edits]
    if not ids:
        raise LineageError("no edits")
    if len(set(ids)) != len(ids):
        raise LineageError("an episode is edited twice in one derivation")
    eps = version_episodes(pvid, lineage_root).sort_values("episode_index").reset_index(drop=True)
    unknown = sorted(set(ids) - set(eps["episode_id"]))
    if unknown:
        raise LineageError(f"{len(unknown)} edited episode_id(s) are not in {pv['name']}: {unknown[:3]}")
    protected = sorted(set(ids) & protected_episode_ids(pvid, lineage_root))
    if protected:
        raise LineageError(
            f"refusing to edit {len(protected)} reference-split episode(s) (PoC §8.7 p.15: corrupt only the "
            f"training copy): {protected[:3]}"
        )
    parent_root = REPO_ROOT / pv["lerobot_root"]
    verify_version(pvid, lineage_root=lineage_root)  # never derive from a tampered copy

    out_root = Path(out_root) if out_root is not None else parent_root.parent / f"{parent_root.name}__{name}"
    if out_root.exists():
        raise LineageError(f"{rel(out_root)} already exists")

    # --- write the derived dataset ---------------------------------------------------------------
    info = json.loads((parent_root / "meta" / "info.json").read_text())
    idx_of = dict(zip(eps["episode_id"], eps["episode_index"]))
    edits = sorted(edits, key=lambda e: idx_of[e.episode_id])  # deterministic write order
    expected, timing = _write_derived_dataset(parent_root, out_root, edits, idx_of, info, image_writer_threads,
                                              isolate_hf_cache, t0)

    # --- re-establish identity by content -----------------------------------------------------
    child = _derived_episode_table(eps, edits, expected, out_root, info["fps"])

    # --- catalog + registration + labels -------------------------------------------------------
    cat_dir = DEFAULT_CATALOG_ROOT / out_root.name if lineage_root is None else _root(lineage_root) / "catalog" / out_root.name
    cat_dir.mkdir(parents=True, exist_ok=True)
    child.to_parquet(cat_dir / "episodes.parquet", index=False)
    praw = _root(lineage_root) / "versions" / pvid / "raw_files.parquet"
    if praw.is_file():
        shutil.copy(praw, cat_dir / "raw_files.parquet")
    (cat_dir / "catalog.json").write_text(json.dumps(
        {"lerobot_root": rel(out_root), "n_episodes": len(child), "hash_scheme": "vlapoc.episode.v1",
         "derived_from": pvid}, indent=2))
    record = register_version(out_root, name, cat_dir, parent_version_id=pvid, note=note, lineage_root=lineage_root)
    vid = record["dataset_version_id"]
    vdir = _root(lineage_root) / "versions" / vid
    labels = pd.DataFrame([{
        "episode_id": e.episode_id,
        "parent_content_hash": child.loc[child["episode_id"] == e.episode_id, "parent_content_hash"].item(),
        "content_hash": expected[e.episode_id],
        "corruption_type": e.corruption_type,
        "params": json.dumps(e.params, sort_keys=True),
        "seed": e.seed,
    } for e in edits])
    labels.to_parquet(vdir / "labels.parquet", index=False)
    deriv = {
        "parent_version_id": pvid, "parent_name": pv["name"], "dataset_version_id": vid, "name": name,
        "n_edits": len(edits), "mechanism": "lerobot delete_episodes + LeRobotDataset writer + merge_datasets",
        "hf_datasets_cache": ("isolated temp dir under data/scratch, deleted after the derivation" if isolate_hf_cache
                              else "shared HF_DATASETS_CACHE (derivation-time caches left behind)"),
        "labels_sha256": hashlib.sha256((vdir / "labels.parquet").read_bytes()).hexdigest(),
        "labels_rows_digest": sha256_json(labels.sort_values("episode_id").to_dict("records")),
        **(derivation or {}),
    }
    _write_json(vdir / "derivation.json", deriv)
    inherited = inherit_splits(pvid, vid, name, lineage_root)
    return {
        "dataset_version_id": vid, "name": name, "lerobot_root": rel(out_root), "parent_version_id": pvid,
        "n_edits": len(edits), "n_episodes": len(child), "n_frames": int(child["length_steps"].sum()),
        "labels": rel(vdir / "labels.parquet"), "labels_sha256": deriv["labels_sha256"],
        "inherited_splits": inherited,
        "timing_s": {"delete": round(timing["delete"], 1), "write_edits": round(timing["write_edits"], 1),
                     "merge": round(timing["merge"], 1), "total": round(time.time() - t0, 1)},
    }


def _write_derived_dataset(parent_root: Path, out_root: Path, edits: list[EpisodeEdit], idx_of: dict, info: dict,
                           image_writer_threads: int, isolate_hf_cache: bool,
                           t0: float) -> tuple[dict[str, str], dict[str, float]]:
    """Write the derived dataset to ``out_root`` with LeRobot's own tools (module docstring, "Mechanism").

    ``delete_episodes`` copies the parent without the edited episodes into a temp dir; LeRobot's writer
    stores each edit's transformed episode into a second temp dir; ``merge_datasets`` joins them, kept
    episodes first, then the edited ones in ``edits`` order. Both temp dirs are deleted.

    Returns ``(expected, timing)``: ``expected[episode_id]`` is the content hash each edited episode must
    have once stored (computed from its transformed arrays), and ``timing`` the phase durations in
    seconds. ``timing["delete"]`` counts from ``t0``, the start of ``derive_version``, so it includes
    the guards, the parent's verification and, in a fresh process, the one-time LeRobot import below
    (1.3-1.9 s, measured; the full-scale delete phase takes 166-181 s).
    """
    from lerobot.datasets.dataset_tools import delete_episodes, merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    tmp_kept = out_root.parent / f".{out_root.name}.tmp_kept"
    tmp_edit = out_root.parent / f".{out_root.name}.tmp_edit"
    for t in (tmp_kept, tmp_edit):
        if t.exists():
            shutil.rmtree(t)
    # LeRobotDataset.create adds these five bookkeeping features itself (DEFAULT_FEATURES).
    user_features = {k: v for k, v in info["features"].items()
                     if k not in ("timestamp", "frame_index", "episode_index", "index", "task_index")}
    repo_id = f"local/{out_root.name}"
    # Every LeRobotDataset load below writes an HF datasets Arrow cache (plan R1); route them all to a
    # temp dir that is deleted with the temporary datasets (AC4.8, analysis-r1 E3).
    cache_ctx = isolated_hf_datasets_cache() if isolate_hf_cache else contextlib.nullcontext()
    with cache_ctx:
        parent_ds = LeRobotDataset(f"local/{parent_root.name}", root=parent_root)
        delete_episodes(parent_ds, [int(idx_of[e.episode_id]) for e in edits], output_dir=tmp_kept,
                        repo_id=f"{repo_id}_kept")
        t_delete = time.time() - t0

        ed = LeRobotDataset.create(repo_id=f"{repo_id}_edit", fps=info["fps"], features=user_features, root=tmp_edit,
                                   robot_type=info.get("robot_type"), use_videos=False,
                                   image_writer_threads=image_writer_threads)
        expected: dict[str, str] = {}
        for e in edits:
            src = read_stored_episode(parent_root, int(idx_of[e.episode_id]))
            new = e.transform(dict(src))
            _check_canonical(new, src)
            if new["task"] != src["task"]:
                raise LineageError("task edits must go through LeRobot modify_tasks (not supported here)")
            expected[e.episode_id] = episode_content_hash(new["task"], new["action"], new["state"], new["image"], new["image2"])
            for i in range(len(new["action"])):
                ed.add_frame({IMAGE_KEYS[0]: new["image"][i], IMAGE_KEYS[1]: new["image2"][i],
                              "observation.state": new["state"][i], "action": new["action"][i], "task": new["task"]})
            ed.save_episode()
        ed.finalize()
        staging = tmp_edit / "images"
        if staging.is_dir() and not any(p.is_file() for p in staging.rglob("*")):
            shutil.rmtree(staging)
        t_edit = time.time() - t0 - t_delete

        merge_datasets([LeRobotDataset(f"{repo_id}_kept", root=tmp_kept), LeRobotDataset(f"{repo_id}_edit", root=tmp_edit)],
                       output_repo_id=repo_id, output_dir=out_root)
        shutil.rmtree(tmp_kept)
        shutil.rmtree(tmp_edit)
    t_merge = time.time() - t0 - t_delete - t_edit
    return expected, {"delete": t_delete, "write_edits": t_edit, "merge": t_merge}


def _derived_episode_table(eps: pd.DataFrame, edits: list[EpisodeEdit], expected: dict[str, str], out_root: Path,
                           fps: int) -> pd.DataFrame:
    """The derived version's episode table, rebuilt from the stored bytes and checked against the parent.

    ``eps`` is the parent's table sorted by ``episode_index``. The merged dataset stores kept episodes
    first (parent order) and edited ones after them (``edits`` order), so stored row i is ``order[i]``.
    Every kept episode must hash to its parent hash, and every edited one to ``expected``, or
    ``LineageError``. The result is the parent's rows in that order with ``parent_content_hash`` added
    and this copy's ``episode_index``, ``content_hash``, length, action statistics and ``duration_s``.
    Ground truth stays out of it (AC4.7): it lives only in ``labels.parquet``.
    """
    edited_ids = [e.episode_id for e in edits]
    order = [i for i in eps["episode_id"] if i not in set(edited_ids)] + edited_ids
    stored = hash_stored_dataset(out_root)
    if len(stored) != len(order):
        raise LineageError(f"derived dataset has {len(stored)} episodes, expected {len(order)}")
    child = eps.set_index("episode_id").loc[order].reset_index()
    child["parent_content_hash"] = child["content_hash"]
    new_hash = stored["content_hash"].to_numpy()
    kept_mask = ~child["episode_id"].isin(edited_ids).to_numpy()
    if not (new_hash[kept_mask] == child.loc[kept_mask, "parent_content_hash"].to_numpy()).all():
        raise LineageError("a kept episode changed content during delete/merge")
    if [expected[i] for i in edited_ids] != list(new_hash[~kept_mask]):
        raise LineageError("stored edited episodes do not hash to their transformed content")
    child["episode_index"] = stored["episode_index"].to_numpy()
    child["content_hash"] = new_hash
    for c in ("length_steps", "smoothness", "idle_fraction", "idle_steps", "gripper_toggles"):
        child[c] = stored[c].to_numpy()
    child["duration_s"] = child["length_steps"] / fps
    return child


def inherit_splits(parent_vid: str, child_vid: str, child_name: str, lineage_root: Path | None = None) -> list[dict]:
    """Carry every split of the parent over to the child: same episode_ids, child's episode versions.

    Binds ``<child_name>/reference`` and ``<child_name>/candidate`` and writes an inherited split record.
    Returns ``[{"split_id", "reference": n, "candidate": n}]``, one entry per parent split.
    """
    out = []
    sdir = _root(lineage_root) / "splits"
    for p in sorted(sdir.glob("*.json")):
        rec = json.loads(p.read_text())
        if rec["dataset_version_id"] != parent_vid:
            continue
        subs = {}
        for role in ("reference", "candidate"):
            src = Subset.load(rec["subsets"][role], lineage_root)
            s = Subset(child_vid, src.episode_ids, {"op": "inherit_split", "role": role,
                                                    "from_subset_id": src.subset_id, "from_split_id": rec["split_id"]})
            s.save(lineage_root)
            subs[role] = s
        split_id = sha256_json({"dataset_version_id": child_vid, "inherited_from": rec["split_id"]})
        crec = {**{k: rec[k] for k in ("rule", "seed", "reference_per_task", "numpy_version", "grounding")},
                "split_id": split_id, "dataset_version_id": child_vid, "version_name": child_name,
                "inherited_from_split_id": rec["split_id"],
                "subsets": {r: s.subset_id for r, s in subs.items()},
                "sizes": {r: len(s.episode_ids) for r, s in subs.items()}}
        _write_json(sdir / f"{split_id}.json", crec)
        for r, s in subs.items():
            _bind_name("subsets", f"{child_name}/{r}", s.subset_id, lineage_root)
        out.append({"split_id": split_id, **crec["sizes"]})
    return out


# ---------------------------------------------------------------------------------------------
# the one demo corruption and the length-changing test edit
# ---------------------------------------------------------------------------------------------


def _episode_rng(seed: int, episode_id: str) -> np.random.Generator:
    """Generator seeded by (seed, episode_id): an episode's noise does not depend on which other
    episodes are edited or in what order."""
    return np.random.default_rng(np.random.SeedSequence([seed, int(hashlib.sha256(episode_id.encode()).hexdigest()[:16], 16)]))


def robodrop_action_noise(episode_id: str, sigma_a: np.ndarray, seed: int, scale: float = 0.25) -> EpisodeEdit:
    """RoboDrop App. A.1 p.13 action corruption: a + N(0, (scale*σa)^2), per action dim, all 7 dims.

    ``sigma_a`` is the per-dim action std (7 values, action units), e.g. ``meta/stats.json`` ``action.std``.
    No clipping (``params["clip"] = None``): noised actions can leave LIBERO's [-1, 1] action range and
    the gripper dim stops being exactly +-1 (analysis-r1; the injection feature decides on clipping).
    """
    sigma_a = np.asarray(sigma_a, dtype=np.float64)

    def f(ep: dict) -> dict:
        rng = _episode_rng(seed, episode_id)
        noise = rng.normal(0.0, scale * sigma_a, size=ep["action"].shape)
        return {**ep, "action": (ep["action"].astype(np.float64) + noise).astype(np.float32)}

    params = {
        "source": "RoboDrop App. A.1 p.13: zero-mean Gaussian noise, std 0.25*sigma_a, sigma_a = empirical action std",
        "std_scale": scale, "sigma_a": [float(x) for x in sigma_a],
        "dims": "all 7 action dims, including the gripper (literal reading; RoboDrop does not say whether the gripper is noised)",
        "clip": None, "dtype": "noise drawn in float64, result cast to float32",
        "rng": "numpy default_rng(SeedSequence([seed, int(sha256(episode_id)[:16], 16)])).normal",
        "demo": "lineage demo derivation; the injection feature owns the corruption suite",
    }
    return EpisodeEdit(episode_id, "action_gaussian_noise", params, seed, f)


def idle_padding_edit(episode_id: str, n_frames: int = 20, at: str = "middle") -> EpisodeEdit:
    """Test edit (AC4.4, PoC §8.10 p.17 "pad ... with idle segments"): insert ``n_frames`` idle frames.

    Idle frame = the observation at the insertion point repeated, non-gripper action exactly 0, gripper
    action held at the previous command. Used to prove length-changing edits; not a corruption suite.
    The frames always go in at the middle (frame T // 2); ``at`` is only recorded in ``params``.
    """
    def f(ep: dict) -> dict:
        t = len(ep["action"]) // 2
        rep = lambda a: np.concatenate([a[:t], np.repeat(a[t:t + 1], n_frames, axis=0), a[t:]], axis=0)  # noqa: E731
        pad = np.repeat(ep["action"][t - 1:t], n_frames, axis=0).copy()
        pad[:, :6] = 0.0
        return {**ep, "action": np.concatenate([ep["action"][:t], pad, ep["action"][t:]], axis=0),
                "state": rep(ep["state"]), "image": rep(ep["image"]), "image2": rep(ep["image2"])}

    return EpisodeEdit(episode_id, "idle_padding_test", {"n_frames": n_frames, "at": at,
                       "idle": "obs repeated, a[:6]=0, gripper held"}, None, f)


def select_per_task(subset: Subset, per_task: int, seed: int, lineage_root: Path | None = None) -> tuple[list[str], str]:
    """``per_task`` episodes per task from ``subset``, seeded; returns (episode_ids, rule).

    Same scheme as ``make_splits``: tasks in (suite, libero_task_id) order, episodes sorted by demo
    number, one ``default_rng(seed)`` consumed across tasks. ``rule`` is the human-readable record.
    """
    eps = subset.episodes(lineage_root)
    rank = {s: i for i, s in enumerate(SUITES)}
    rng = np.random.default_rng(seed)
    out = []
    for (suite, tid), g in sorted(eps.groupby(["suite", "libero_task_id"]), key=lambda kv: (rank.get(kv[0][0], 9), kv[0])):
        ids = sorted(g["episode_id"], key=_demo_number)
        out += [ids[i] for i in sorted(rng.choice(len(ids), size=per_task, replace=False))]
    rule = (f"per (suite, libero_task_id) in suite order, episodes of subset {subset.subset_id[:12]} sorted by demo "
            f"number, numpy.random.default_rng({seed}).choice(n, {per_task}, replace=False)")
    return out, rule


def derive_demo_action_noise(parent: str = "v0", name: str = "v1", per_task: int = 5, seed: int = 0,
                             candidate: str | None = None, lineage_root: Path | None = None) -> dict:
    """The S4 demo derivation: RoboDrop action noise on ``per_task`` candidate episodes per task.

    Episodes come from the parent's ``candidate`` split (never the references); σ_a is the parent's
    ``meta/stats.json`` action std. v1 = ``derive_demo_action_noise("v0", "v1", 5, 0)``: 100 edits (10%).
    """
    pv = resolve_version(parent, lineage_root)
    cand = Subset.load(candidate or f"{pv['name']}/candidate", lineage_root)
    ids, rule = select_per_task(cand, per_task, seed, lineage_root)
    stats = json.loads((REPO_ROOT / pv["lerobot_root"] / "meta" / "stats.json").read_text())
    sigma = np.array(stats["action"]["std"], dtype=np.float64)
    edits = [robodrop_action_noise(i, sigma, seed) for i in ids]
    for e in edits:
        e.params["sigma_a_source"] = f"{pv['name']} meta/stats.json action.std ({pv['n_episodes']} episodes)"
        e.params["selection"] = rule
    return derive_version(
        pv["dataset_version_id"], edits, name,
        note=f"lineage demo: RoboDrop action noise on {len(ids)} candidate episodes ({per_task}/task), seed {seed}",
        derivation={"selection_rule": rule, "selected_from_subset": cand.subset_id, "per_task": per_task,
                    "seed": seed, "corruption_type": "action_gaussian_noise"},
        lineage_root=lineage_root,
    )


def load_labels(version: str, lineage_root: Path | None = None) -> pd.DataFrame:
    """A derived version's ground truth (``labels.parquet``): one row per edited episode with
    ``episode_id``, ``parent_content_hash``, ``content_hash``, ``corruption_type``, ``params`` (JSON)
    and ``seed``. Evaluation code only; the method under test must not read it (plan Rule 5)."""
    vid = resolve_version(version, lineage_root)["dataset_version_id"]
    p = _root(lineage_root) / "versions" / vid / "labels.parquet"
    if not p.is_file():
        raise LineageError(f"version {version} has no labels (not a derived version)")
    return pd.read_parquet(p)


__all__ = ["EpisodeEdit", "derive_version", "derive_demo_action_noise", "robodrop_action_noise",
           "idle_padding_edit", "select_per_task", "protected_episode_ids", "ancestry", "load_labels",
           "read_stored_episode", "inherit_splits"]
