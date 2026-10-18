"""Dataset versions, named splits and episode subsets that resolve to LeRobot ``episodes=``.

Grounding
---------
- PoC §2.1 item 1 (p. 3): "Every training run should be tied to exact dataset versions (content
  hashes), episode IDs, mixture weights, ...". A *version* is identified by its content
  (``dataset_version_id``); a *subset* names exactly which episode versions a stage consumes.
- PoC §8.3 (p. 14): references "are a small, held-out set of clean demonstrations for the target task,
  separate from the candidate data being audited"; PoC §8.7 (p. 15), row "Benchmark demos (PoC)":
  "Treat LIBERO's original demos as clean; corrupt only the training copy". ``make_splits`` creates the
  held-out ``reference`` split and the ``candidate`` split, disjoint and persisted.
- RoboDrop App. A.1 (p. 13, docs/papers/RoboDrop_Gradient_Compatibility_arXiv-2609.10021.pdf): "For
  validation, we reserve two uncorrupted episodes per task, giving 20 clean validation episodes for
  each suite." Hence the default ``reference_per_task = 2`` (40 for our two suites).
- PoC §2.1 item 5 (p. 3) and §10 (p. 20) group leave-one-out: ``drop_group`` (causal baseline).
- PoC §2.6 (p. 5): "Remove the top-k “harmful” episodes and retrain: the policy should improve more than
  after removing k random episodes." PoC §9.1 Step 4 (p. 19): "Remove the top-k most harmful episodes
  and retrain. As a control, remove k random episodes and retrain." ``drop_top_k`` / ``drop_random_k``
  derive those child subsets and record how (parent, score-table digest, metric, k, seed).
- LeRobot 0.6.1 ``LeRobotDataset(repo_id, root, episodes=[...])`` loads only the listed
  ``episode_index`` values (``io_utils.load_nested_dataset``: PyArrow predicate pushdown
  ``pa_ds.field("episode_index").isin(episodes)``); ``Subset.resolve()`` produces that list, which is
  also what ``lerobot-train --dataset.episodes=[...]`` takes (``DatasetConfig.episodes``).

Identifiers (all sha256 hex over canonical JSON: sorted keys, no whitespace)
---------------------------------------------------------------------------
- ``dataset_version_id`` = H({"scheme": "vlapoc.version.v1", "episode_hash_scheme": ...,
  "schema": {fps, features: {name: {dtype, shape, names}}}, "episodes": sorted [episode_id, content_hash]}).
  It depends only on content, so re-converting the same raw data reproduces it even though the parquet
  bytes differ (each LeRobot data file embeds a random HF ``datasets`` fingerprint; measured).
- ``subset_id`` = H({"scheme": "vlapoc.subset.v1", "dataset_version_id", "episode_ids" (sorted),
  "derivation"}).
- Per-file sha256 of the physical copy (``files.json``) is recorded separately: ``verify_version``
  uses it to detect any byte change before a stage consumes the copy.

Layout under ``data/lineage/`` (all paths stored relative to the repo root; never absolute)::

    names.json                              {"versions": {"v0": id}, "subsets": {"v0/reference": sid, ...}}
    versions/<dataset_version_id>/version.json, episodes.parquet, files.json, raw_files.parquet
                                            (+ labels.parquet, derivation.json for a derived version)
    subsets/<subset_id>.json
    splits/<split_id>.json

A *reference* to a version or subset (every ``ref`` argument below) is a name from ``names.json``
(``v0``, ``v0/candidate``), a full 64-hex id, or a unique id prefix of at least 8 hex characters.

Where it sits
-------------
Third step (docs/dataflow.md). It takes the catalog (``vlapoc.catalog``) and turns the dataset into
named, content-addressed objects that every later stage consumes and produces: the clean reference
set and the training (candidate) set of the "Full flow" diagram, and the Step 4 "drop top-k / drop
k random" selections. ``vlapoc.derive`` adds derived (corrupted) versions on top, and ``vlapoc.runs``
records each stage as an MLflow run that names these ids.

Usage (from the repo root; every command prints JSON)::

    # dataset versions
    .venv/bin/python -m vlapoc.lineage version register --root data/lerobot/libero_spatial_10 --name v0
    .venv/bin/python -m vlapoc.lineage version show v0
    .venv/bin/python -m vlapoc.lineage version verify v0        # sha256 of every file vs files.json (~3 s)
    .venv/bin/python -m vlapoc.lineage version recompute v0     # re-hash every episode, recompute the id (~19 s)
    .venv/bin/python -m vlapoc.lineage version diff v0 v1 [--ids]
    .venv/bin/python -m vlapoc.lineage version check-blind v1   # edits visible in method-facing tables?
    # splits and subsets
    .venv/bin/python -m vlapoc.lineage split make --version v0 --reference-per-task 2 --seed 0 [--track]
    .venv/bin/python -m vlapoc.lineage subset show v0/reference [--ids]
    .venv/bin/python -m vlapoc.lineage names
    # stages recorded as MLflow runs (vlapoc.runs, vlapoc.derive)
    .venv/bin/python -m vlapoc.lineage build verify v0
    .venv/bin/python -m vlapoc.lineage derive --parent v0 --name v1 --demo-action-noise --per-task 5 --seed 0
    # lineage queries over the recorded runs (vlapoc.runs)
    .venv/bin/python -m vlapoc.lineage trace <run_id>
    .venv/bin/python -m vlapoc.lineage consumers <episode_id>@<content_hash>
    .venv/bin/python -m vlapoc.lineage diff <run_a> <run_b> [--ids]
    .venv/bin/python -m vlapoc.lineage check-scores <scores.parquet> <train_run_id>

``register``, ``split make``, ``build`` and ``derive`` write to ``data/lineage/``. ``--track``,
``build`` and ``derive`` also add an MLflow run, and refuse to start while ``vlapoc/`` or ``scripts/``
has uncommitted changes, so the recorded git SHA is the code that ran.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from vlapoc.catalog import CATALOG_COLUMNS, DEFAULT_CATALOG_ROOT, HASH_DOMAIN, hash_stored_dataset, sha256_file
from vlapoc.libero_convert import REPO_ROOT, SUITES

LINEAGE_ROOT = REPO_ROOT / "data" / "lineage"
VERSION_SCHEME = "vlapoc.version.v1"
SUBSET_SCHEME = "vlapoc.subset.v1"
EPISODE_HASH_SCHEME = HASH_DOMAIN.decode().rstrip("\0")


class LineageError(RuntimeError):
    """A lineage invariant is violated (unknown name, stale key, disjointness, ...)."""


class VersionIntegrityError(LineageError):
    """The files of a dataset copy no longer match the digests recorded for its version."""


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def canonical_json(obj) -> bytes:
    """UTF-8 JSON with sorted keys and no whitespace: the byte form every id in this module hashes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_json(obj) -> str:
    """sha256 (hex) of ``canonical_json(obj)``; the same object always gives the same id."""
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def rel(path: Path) -> str:
    """Repo-relative POSIX path; refuses paths outside the repo so no manifest holds an absolute path."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError as e:
        raise LineageError(f"{p} is outside the repository; lineage records only repo-relative paths") from e


def _root(lineage_root: Path | None) -> Path:
    """The lineage store to use: ``lineage_root`` if given (tests use scratch copies), else ``data/lineage``."""
    return Path(lineage_root) if lineage_root is not None else LINEAGE_ROOT


def _names_path(lineage_root: Path | None) -> Path:
    return _root(lineage_root) / "names.json"


def load_names(lineage_root: Path | None = None) -> dict:
    """``names.json``: ``{"versions": {name: dataset_version_id}, "subsets": {name: subset_id}}``.

    Names are convenience aliases; every record and MLflow run stores the ids themselves.
    """
    p = _names_path(lineage_root)
    if p.is_file():
        return json.loads(p.read_text())
    return {"versions": {}, "subsets": {}}


def _bind_name(kind: str, name: str, value: str, lineage_root: Path | None, force: bool = False) -> None:
    """Bind ``names.json[kind][name] = value``; rebinding a name to a different id needs ``force``."""
    names = load_names(lineage_root)
    old = names[kind].get(name)
    if old is not None and old != value and not force:
        raise LineageError(f"name {name!r} is already bound to {old[:12]}; refusing to rebind to {value[:12]}")
    names[kind][name] = value
    p = _names_path(lineage_root)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(names, indent=2, sort_keys=True))


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def _demo_number(episode_id: str) -> int:
    """``k`` of ``.../demo_<k>``, for numeric (not string) ordering of a task's episodes."""
    return int(episode_id.rsplit("demo_", 1)[1])


def _resolve_ref(ref: str, bound_names: dict[str, str], known_ids: list[str], kind: str) -> str:
    """Full id for a reference: a name bound in ``names.json``, a full id, or a unique id prefix.

    ``known_ids`` are the ids present on disk. An exact match wins; otherwise a prefix of at least 8
    characters may match, and it must match exactly one id. Raises ``LineageError`` on no match or an
    ambiguous one.
    """
    wanted = bound_names.get(ref, ref)
    matches = [i for i in known_ids if i == wanted]
    if not matches and len(wanted) >= 8:
        matches = [i for i in known_ids if i.startswith(wanted)]
    if len(matches) != 1:
        raise LineageError(f"unknown or ambiguous {kind} {ref!r}")
    return matches[0]


# ---------------------------------------------------------------------------------------------
# versions
# ---------------------------------------------------------------------------------------------


def schema_of(info: dict) -> dict:
    """The content-defining part of LeRobot ``meta/info.json``: fps and each feature's dtype/shape/names."""
    return {
        "fps": info["fps"],
        "features": {
            k: {"dtype": v["dtype"], "shape": list(v["shape"]), "names": v.get("names")}
            for k, v in sorted(info["features"].items())
        },
    }


def version_id(schema: dict, pairs) -> str:
    """``dataset_version_id`` from the feature schema and the (episode_id, content_hash) pairs.

    The pairs are sorted first, so episode order (LeRobot's positional ``episode_index``) does not
    affect the id. Raises ``LineageError`` on a duplicate ``episode_id``.
    """
    eps = sorted([str(e), str(h)] for e, h in pairs)
    if len({e for e, _ in eps}) != len(eps):
        raise LineageError("duplicate episode_id in version")
    return sha256_json(
        {"scheme": VERSION_SCHEME, "episode_hash_scheme": EPISODE_HASH_SCHEME, "schema": schema, "episodes": eps}
    )


def _file_digest(args: tuple[str, str]) -> tuple[str, str, int]:
    root, relpath = args
    p = Path(root) / relpath
    return relpath, sha256_file(p), p.stat().st_size


def file_digests(root: Path, workers: int = 8) -> list[dict]:
    """``[{"path", "sha256", "size_bytes"}]`` for every file under a LeRobot dataset root, sorted by path.

    These identify one physical copy, not the content: two conversions of the same data differ here
    (random HF ``datasets`` fingerprint in each parquet file) while their ``dataset_version_id`` agrees.
    """
    root = Path(root)
    files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_file_digest, [(str(root), f) for f in files]))
    return [{"path": f, "sha256": h, "size_bytes": s} for f, h, s in out]


def register_version(
    root: Path,
    name: str,
    catalog_dir: Path | None = None,
    parent_version_id: str | None = None,
    note: str | None = None,
    lineage_root: Path | None = None,
    force_name: bool = False,
) -> dict:
    """Register the LeRobot dataset at ``root`` (with its S2 catalog) as a content-addressed version.

    Args:
        root: dataset root inside the repo.
        name: name to bind in ``names.json`` (also binds ``<name>/all`` to the full-episode subset).
        catalog_dir: catalog written by ``vlapoc.catalog``; default ``data/lineage/catalog/<root name>``.
            Its ``catalog.json`` must name this ``root``.
        parent_version_id: set by ``vlapoc.derive`` for a derived version.
        force_name: allow rebinding an existing name to a different id.

    Returns:
        The ``version.json`` record written to ``versions/<dataset_version_id>/``.

    Re-registering content that is already registered rewrites that version's ``version.json`` (name,
    note, timestamp) while ``names.json`` keeps the old name bound too, so ``resolve_version`` of the
    old name then returns a record carrying the new name (measured in a scratch store).
    """
    root = Path(root).resolve()
    catalog_dir = Path(catalog_dir) if catalog_dir is not None else DEFAULT_CATALOG_ROOT / root.name
    cat_meta = json.loads((catalog_dir / "catalog.json").read_text())
    if cat_meta["lerobot_root"] != rel(root):
        raise LineageError(f"catalog {catalog_dir} describes {cat_meta['lerobot_root']}, not {rel(root)}")
    episodes = pd.read_parquet(catalog_dir / "episodes.parquet")
    info = json.loads((root / "meta" / "info.json").read_text())
    if info["total_episodes"] != len(episodes):
        raise LineageError(f"catalog has {len(episodes)} rows, dataset has {info['total_episodes']} episodes")
    schema = schema_of(info)
    vid = version_id(schema, zip(episodes["episode_id"], episodes["content_hash"]))
    vdir = _root(lineage_root) / "versions" / vid
    files = file_digests(root)
    record = {
        "dataset_version_id": vid,
        "name": name,
        "lerobot_root": rel(root),
        "schema": schema,
        "scheme": VERSION_SCHEME,
        "episode_hash_scheme": EPISODE_HASH_SCHEME,
        "n_episodes": int(len(episodes)),
        "n_frames": int(episodes["length_steps"].sum()),
        "parent_version_id": parent_version_id,
        "note": note,
        "registered_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files_total_bytes": int(sum(f["size_bytes"] for f in files)),
    }
    vdir.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(vdir / "episodes.parquet", index=False)
    _write_json(vdir / "files.json", files)
    if (catalog_dir / "raw_files.parquet").is_file():
        pd.read_parquet(catalog_dir / "raw_files.parquet").to_parquet(vdir / "raw_files.parquet", index=False)
    _write_json(vdir / "version.json", record)
    _bind_name("versions", name, vid, lineage_root, force=force_name)
    # every version also exposes its full episode set as a subset named "<name>/all"
    full = Subset(vid, tuple(episodes["episode_id"]), {"op": "all"})
    full.save(lineage_root)
    _bind_name("subsets", f"{name}/all", full.subset_id, lineage_root, force=force_name)
    return record


def resolve_version(ref: str, lineage_root: Path | None = None) -> dict:
    """``version.json`` for a version name (``v0``), full id, or unique id prefix (>= 8 hex chars)."""
    vroot = _root(lineage_root) / "versions"
    known = [p.name for p in vroot.iterdir()] if vroot.is_dir() else []
    vid = _resolve_ref(ref, load_names(lineage_root)["versions"], known, "version")
    return json.loads((vroot / vid / "version.json").read_text())


def version_episodes(ref: str, lineage_root: Path | None = None) -> pd.DataFrame:
    """The version's episode table: one row per episode, exactly ``CATALOG_COLUMNS``. It maps
    ``episode_id`` to this copy's positional ``episode_index``, and it is what screening and attribution
    read (method-facing).

    For a derived version it must not reveal which episodes were edited (PoC §2.1 item 4 p. 3). That
    holds only for versions that pass ``check_blind``: v1 (derived before evaluation-layer S0) still has
    its edited episodes at ``episode_index`` 900-999 and is fenced (``fence_record``); use
    ``assert_blind`` before handing a version to a method under test. Pre-edit hashes are lineage-only
    (``parent_hashes``)."""
    vid = resolve_version(ref, lineage_root)["dataset_version_id"]
    return pd.read_parquet(_root(lineage_root) / "versions" / vid / "episodes.parquet")


def recompute_version_id(ref: str, root: Path | None = None, lineage_root: Path | None = None) -> str:
    """Re-hash every episode from the stored data and recompute ``dataset_version_id``.

    Episode identity (``episode_index`` -> ``episode_id``) comes from the version's episodes table,
    because LeRobot's positional index is not an identity; the content hashes come from the bytes.
    """
    v = resolve_version(ref, lineage_root)
    root = Path(root) if root is not None else REPO_ROOT / v["lerobot_root"]
    eps = version_episodes(v["dataset_version_id"], lineage_root)
    stored = hash_stored_dataset(root)
    idx_to_id = dict(zip(eps["episode_index"], eps["episode_id"]))
    info = json.loads((root / "meta" / "info.json").read_text())
    return version_id(schema_of(info), [(idx_to_id[i], h) for i, h in zip(stored["episode_index"], stored["content_hash"])])


def verify_version(ref: str, root: Path | None = None, lineage_root: Path | None = None) -> dict:
    """Recompute every file digest of a dataset copy; raise ``VersionIntegrityError`` on any mismatch.

    ``root`` overrides the registered location (e.g. to check a copy on another machine). Returns
    ``{"version", "files_checked", "ok": True}``. ``train_run`` and ``derive_version`` call this before
    reading a copy, so a single changed byte stops the stage before it starts.
    """
    v = resolve_version(ref, lineage_root)
    root = Path(root) if root is not None else REPO_ROOT / v["lerobot_root"]
    expected = {f["path"]: f for f in json.loads(
        (_root(lineage_root) / "versions" / v["dataset_version_id"] / "files.json").read_text())}
    actual = {f["path"]: f for f in file_digests(root)}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(p for p in set(expected) & set(actual) if expected[p]["sha256"] != actual[p]["sha256"])
    if missing or extra or changed:
        raise VersionIntegrityError(
            f"dataset copy at {root} does not match version {v['name']} ({v['dataset_version_id'][:12]}): "
            f"changed={changed} missing={missing} extra={extra}"
        )
    return {"version": v["dataset_version_id"], "files_checked": len(expected), "ok": True}


# ---------------------------------------------------------------------------------------------
# subsets
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Subset:
    """A selection of episode versions from one dataset version.

    The record is a selection: no dataset file is rewritten or reindexed; ``resolve()`` maps its
    ``episode_id``s to the version's ``episode_index`` values. Loading it is not free on disk, though:
    LeRobot 0.6.1 loads ``episodes=`` through ``datasets.Dataset.from_parquet(..., filters=...)``
    (``lerobot/datasets/io_utils.py:79-82``), and HF ``datasets`` materialises an Arrow cache of the
    filtered rows under ``HF_DATASETS_CACHE`` for every distinct episode list (measured: 6.4-6.6 GB for
    a 940-960 episode subset of this dataset; plan risk R1). Wrap throwaway loads in
    ``isolated_hf_datasets_cache()``.
    """

    dataset_version_id: str
    episode_ids: tuple[str, ...]
    derivation: dict = field(default_factory=dict)

    def __post_init__(self):
        ids = tuple(sorted(set(self.episode_ids)))
        if len(ids) != len(self.episode_ids):
            raise LineageError("duplicate episode_id in subset")
        object.__setattr__(self, "episode_ids", ids)

    @property
    def subset_id(self) -> str:
        """sha256 over (scheme, ``dataset_version_id``, sorted ``episode_ids``, ``derivation``): the same
        episodes selected by a different rule get a different id, so the record keeps how it was made."""
        return sha256_json(
            {"scheme": SUBSET_SCHEME, "dataset_version_id": self.dataset_version_id,
             "episode_ids": list(self.episode_ids), "derivation": self.derivation}
        )

    def episodes(self, lineage_root: Path | None = None) -> pd.DataFrame:
        """The version's catalog rows for this subset (includes ``episode_index`` and ``content_hash``)."""
        eps = version_episodes(self.dataset_version_id, lineage_root)
        sel = eps[eps["episode_id"].isin(self.episode_ids)]
        if len(sel) != len(self.episode_ids):
            unknown = sorted(set(self.episode_ids) - set(eps["episode_id"]))
            raise LineageError(f"subset names {len(unknown)} episode_id(s) not in version: {unknown[:3]}")
        return sel.sort_values("episode_id").reset_index(drop=True)

    def keys(self, lineage_root: Path | None = None) -> pd.DataFrame:
        """(episode_id, content_hash) for every episode version in the subset."""
        return self.episodes(lineage_root)[["episode_id", "content_hash"]]

    def resolve(self, lineage_root: Path | None = None) -> list[int]:
        """LeRobot ``episode_index`` values for ``LeRobotDataset(..., episodes=)`` / ``--dataset.episodes``."""
        return sorted(int(i) for i in self.episodes(lineage_root)["episode_index"])

    def to_record(self) -> dict:
        """The JSON record stored as ``subsets/<subset_id>.json``."""
        return {"subset_id": self.subset_id, "scheme": SUBSET_SCHEME, "dataset_version_id": self.dataset_version_id,
                "n_episodes": len(self.episode_ids), "episode_ids": list(self.episode_ids),
                "derivation": self.derivation}

    def save(self, lineage_root: Path | None = None) -> Path:
        """Write ``subsets/<subset_id>.json`` (idempotent) after checking every id is in the version."""
        self.episodes(lineage_root)  # validates membership
        p = _root(lineage_root) / "subsets" / f"{self.subset_id}.json"
        _write_json(p, self.to_record())
        return p

    @classmethod
    def load(cls, ref: str, lineage_root: Path | None = None) -> "Subset":
        """Load by subset name (``v0/candidate``), full id, or unique prefix (>= 8 hex chars).

        Raises ``LineageError`` if the file's content no longer hashes to its id (edited by hand).
        """
        sdir = _root(lineage_root) / "subsets"
        sid = _resolve_ref(ref, load_names(lineage_root)["subsets"], [p.stem for p in sdir.glob("*.json")], "subset")
        rec = json.loads((sdir / f"{sid}.json").read_text())
        s = cls(rec["dataset_version_id"], tuple(rec["episode_ids"]), rec["derivation"])
        if s.subset_id != rec["subset_id"]:
            raise LineageError(f"subset file {sid} does not hash to its id (edited by hand?)")
        return s


def make_splits(
    version: str, reference_per_task: int = 2, seed: int = 0, lineage_root: Path | None = None
) -> dict:
    """Held-out clean ``reference`` split (k per task, stratified, seeded) and the ``candidate`` rest.

    Rule: tasks are visited in (suite order, libero_task_id) order; within a task, episodes are sorted
    by demo number and ``numpy.random.default_rng(seed).choice(n, k, replace=False)`` (one generator,
    consumed in that task order) picks the reference episodes.

    Writes both subsets, ``splits/<split_id>.json`` and the names ``<version>/reference`` and
    ``<version>/candidate``. Deterministic, so re-running rewrites identical records. Returns
    ``{"record": <split record>, "reference": Subset, "candidate": Subset}``.
    """
    v = resolve_version(version, lineage_root)
    vid = v["dataset_version_id"]
    eps = version_episodes(vid, lineage_root)
    rng = np.random.default_rng(seed)
    reference: list[str] = []
    suite_rank = {s: i for i, s in enumerate(SUITES)}
    tasks = sorted(eps[["suite", "libero_task_id"]].drop_duplicates().itertuples(index=False),
                   key=lambda t: (suite_rank.get(t.suite, len(SUITES)), t.suite, t.libero_task_id))
    for t in tasks:
        ids = sorted(eps[(eps["suite"] == t.suite) & (eps["libero_task_id"] == t.libero_task_id)]["episode_id"],
                     key=_demo_number)
        if len(ids) <= reference_per_task:
            raise LineageError(f"task {t} has only {len(ids)} episodes")
        pick = rng.choice(len(ids), size=reference_per_task, replace=False)
        reference += [ids[i] for i in sorted(pick)]
    rule = ("stratified by (suite, libero_task_id); per task, episodes sorted by demo number; "
            "numpy.random.default_rng(seed).choice(n, reference_per_task, replace=False), one generator "
            "consumed in (suite, libero_task_id) order")
    parent = Subset(vid, tuple(eps["episode_id"]), {"op": "all"})
    common = {"op": "split", "parent_subset_id": parent.subset_id, "rule": rule,
              "reference_per_task": reference_per_task, "seed": seed, "numpy_version": np.__version__}
    ref_s = Subset(vid, tuple(reference), {**common, "role": "reference"})
    cand_s = Subset(vid, tuple(sorted(set(eps["episode_id"]) - set(reference))), {**common, "role": "candidate"})
    if set(ref_s.episode_ids) & set(cand_s.episode_ids):
        raise LineageError("reference and candidate overlap")
    for s in (ref_s, cand_s):
        s.save(lineage_root)
    split_id = sha256_json({"dataset_version_id": vid, "rule": rule, "seed": seed,
                            "reference_per_task": reference_per_task})
    rec = {
        "split_id": split_id, "dataset_version_id": vid, "version_name": v["name"], "rule": rule, "seed": seed,
        "reference_per_task": reference_per_task, "numpy_version": np.__version__,
        "subsets": {"reference": ref_s.subset_id, "candidate": cand_s.subset_id},
        "sizes": {"reference": len(ref_s.episode_ids), "candidate": len(cand_s.episode_ids)},
        "grounding": "PoC §8.3 p.14, §8.7 p.15; RoboDrop App. A.1 p.13 (two uncorrupted episodes per task)",
    }
    _write_json(_root(lineage_root) / "splits" / f"{split_id}.json", rec)
    _bind_name("subsets", f"{v['name']}/reference", ref_s.subset_id, lineage_root)
    _bind_name("subsets", f"{v['name']}/candidate", cand_s.subset_id, lineage_root)
    return {"record": rec, "reference": ref_s, "candidate": cand_s}


def score_table_digest(score_table: pd.DataFrame, metric: str) -> str:
    """Digest of the (episode_id, content_hash, score) rows a selection used; floats hashed exactly."""
    rows = sorted([str(e), str(h), float(s).hex()] for e, h, s in
                  zip(score_table["episode_id"], score_table["content_hash"], score_table[metric]))
    return sha256_json({"metric": metric, "rows": rows})


def check_score_keys(parent: Subset, score_table: pd.DataFrame, metric: str, lineage_root: Path | None) -> None:
    """Raise unless the score table keys exactly the parent's (episode_id, content_hash) pairs, once each,
    with no NaN in ``metric``: no stale hashes, no unscored parent episodes, no extras."""
    for c in ("episode_id", "content_hash", metric):
        if c not in score_table.columns:
            raise LineageError(f"score table lacks column {c!r}")
    if score_table["episode_id"].duplicated().any():
        raise LineageError("score table has duplicate episode_id rows")
    if score_table[metric].isna().any():
        raise LineageError(f"score table has NaN in {metric!r}")
    keys = parent.keys(lineage_root)
    want = set(zip(keys["episode_id"], keys["content_hash"]))
    have = set(zip(score_table["episode_id"], score_table["content_hash"]))
    if have != want:
        stale = sorted(have - want)[:3]
        missing = sorted(want - have)[:3]
        raise LineageError(
            f"score table keys != parent subset's episode versions: {len(have - want)} unknown/stale "
            f"(e.g. {stale}), {len(want - have)} unscored (e.g. {missing})"
        )


_check_score_keys = check_score_keys  # the original private name, kept for existing callers


def drop_top_k(
    parent: Subset, score_table: pd.DataFrame, metric: str, k: int, direction: str = "highest",
    score_source: str | None = None, lineage_root: Path | None = None,
) -> Subset:
    """Child subset without the k top-ranked episodes by ``metric`` (PoC §2.6 p.5, §9.1 Step 4 p.19).

    ``direction="highest"`` drops the k largest scores, ``"lowest"`` the k smallest; ties are broken by
    ``episode_id`` ascending. The score table must key exactly the parent's episode versions
    (episode_id AND content_hash), so a table computed on another version is rejected.
    """
    if direction not in ("highest", "lowest"):
        raise ValueError("direction must be 'highest' or 'lowest'")
    if not 0 < k < len(parent.episode_ids):
        raise ValueError(f"k={k} out of range for a parent of {len(parent.episode_ids)} episodes")
    _check_score_keys(parent, score_table, metric, lineage_root)
    t = score_table[["episode_id", metric]].copy()
    t["_s"] = -t[metric] if direction == "highest" else t[metric]
    dropped = sorted(t.sort_values(["_s", "episode_id"], kind="mergesort")["episode_id"].head(k))
    deriv = {"op": "drop_top_k", "parent_subset_id": parent.subset_id, "metric": metric, "k": k,
             "direction": direction, "tie_break": "episode_id ascending",
             "score_table_digest": score_table_digest(score_table, metric), "score_source": score_source,
             "dropped": dropped}
    child = Subset(parent.dataset_version_id, tuple(sorted(set(parent.episode_ids) - set(dropped))), deriv)
    child.save(lineage_root)
    return child


def drop_random_k(parent: Subset, k: int, seed: int, lineage_root: Path | None = None) -> Subset:
    """Random-k control (PoC §9.1 Step 4 p.19): ``default_rng(seed).choice(n, k, replace=False)`` over
    the parent's sorted episode_ids. The dropped IDs are recorded too, so the child is reproducible
    even if a future numpy changes ``Generator.choice``'s stream.

    Random subsets for datamodels-style regression (PoC §2.1 item 5 p.4 "Train many runs on random data
    subsets, then fit the effect of each group on success"; DataMIL p.4 "we first sample m random
    subsets of the prior dataset") use this same op: a random subset of size ``n`` from a parent of
    ``N`` episodes is ``drop_random_k(parent, k=N - n, seed=s)``.
    """
    if not 0 < k < len(parent.episode_ids):
        raise ValueError(f"k={k} out of range for a parent of {len(parent.episode_ids)} episodes")
    idx = np.random.default_rng(seed).choice(len(parent.episode_ids), size=k, replace=False)
    dropped = sorted(parent.episode_ids[i] for i in idx)
    deriv = {"op": "drop_random_k", "parent_subset_id": parent.subset_id, "k": k, "seed": seed,
             "rng": "numpy.random.default_rng(seed).choice(n, k, replace=False) over sorted parent episode_ids",
             "numpy_version": np.__version__, "dropped": dropped}
    child = Subset(parent.dataset_version_id, tuple(sorted(set(parent.episode_ids) - set(dropped))), deriv)
    child.save(lineage_root)
    return child


# Per-episode identity/hash columns: unique (or nearly) per episode, so "dropping a group" by them is
# not a group ablation. drop_group refuses them.
GROUP_COLUMNS_EXCLUDED = {"episode_id", "episode_index", "content_hash", "init_state_sha256",
                          "source_file", "source_demo"}


def drop_group(parent: Subset, column: str, value, lineage_root: Path | None = None) -> Subset:
    """Group leave-one-out: child subset without every parent episode whose catalog ``column == value``.

    Grounding: PoC §2.1 item 5 (p. 3) "Leave-one-group-out ablations. Drop one data source or task group,
    retrain, and measure the change"; PoC §10 (p. 20) "Causal baseline | Group-level leave-one-out
    ablations, judged with the macro-KS gate".

    ``column`` must be a scalar §2.2 catalog column (``vlapoc.catalog.CATALOG_COLUMNS``, e.g. ``suite``,
    ``task_name``, ``scene``); per-episode identity/hash columns and list columns (``objects``) are
    rejected, and so is anything outside the catalog, so ground truth (which lives only in a derived
    version's ``labels.parquet``) cannot be grouped on. Raises when ``value`` matches no parent episode
    or every one.

    Use ``task_name`` for leave-one-task-out. ``libero_task_id`` is the task's index *within its suite*
    (0-9 in each suite), so ``drop_group(parent, "libero_task_id", 2)`` drops one task from each suite
    (measured by the analyzer: 96 episodes from two tasks, one per suite, on a 960-episode candidate split).

    The derivation records ``op``, ``column``, ``value`` (numpy scalars stored as plain Python values),
    the parent subset id and the sorted dropped ids.
    """
    if column not in CATALOG_COLUMNS or column in GROUP_COLUMNS_EXCLUDED:
        allowed = sorted(set(CATALOG_COLUMNS) - GROUP_COLUMNS_EXCLUDED - {"objects"})
        raise LineageError(f"drop_group: {column!r} is not a groupable catalog column; allowed: {allowed}")
    eps = parent.episodes(lineage_root)
    if eps[column].map(lambda x: isinstance(x, (list, tuple, np.ndarray))).any():
        raise LineageError(f"drop_group: {column!r} is list-valued; group on a scalar column")
    mask = eps[column] == value
    n = int(mask.sum())
    if n == 0 or n == len(eps):
        raise LineageError(f"drop_group: {column} == {value!r} matches {n} of {len(eps)} parent episodes; "
                           "it must match some but not all")
    dropped = sorted(eps.loc[mask, "episode_id"])
    val = value.item() if hasattr(value, "item") else value
    deriv = {"op": "drop_group", "parent_subset_id": parent.subset_id, "column": column, "value": val,
             "dropped": dropped}
    child = Subset(parent.dataset_version_id, tuple(sorted(set(parent.episode_ids) - set(dropped))), deriv)
    child.save(lineage_root)
    return child


@contextmanager
def isolated_hf_datasets_cache(parent_dir: Path | None = None):
    """Route every HF ``datasets`` Arrow cache written inside the block to a fresh temp dir, deleted on exit.

    Why: ``LeRobotDataset(...)`` loads through ``datasets.Dataset.from_parquet`` and leaves a cache per
    distinct episode list (plan risk R1; analysis-r1 E3: 13 entries from one 20-episode derivation).
    How: ``datasets/builder.py:365`` resolves the cache root when each builder is constructed
    (``self._cache_dir_root = str(cache_dir or config.HF_DATASETS_CACHE)``) and no module imports the
    value by name (grep, datasets 4.8.5), so swapping ``datasets.config.HF_DATASETS_CACHE`` for the
    duration of the block routes every builder created in it. The temp dir sits under the gitignored
    ``data/scratch/`` (repo-relative), never under a session-bound path.
    """
    import datasets

    base = Path(parent_dir) if parent_dir is not None else REPO_ROOT / "data" / "scratch"
    base.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="hf_datasets_cache_", dir=base))
    old = datasets.config.HF_DATASETS_CACHE
    datasets.config.HF_DATASETS_CACHE = tmp
    try:
        yield tmp
    finally:
        datasets.config.HF_DATASETS_CACHE = old
        shutil.rmtree(tmp, ignore_errors=True)


def diff_versions(a: str, b: str, lineage_root: Path | None = None) -> dict:
    """Episode-version difference between two dataset versions (by episode_id and content_hash).

    Returns both full ids, the counts ``changed`` and ``identical`` (episodes present in both), the
    sorted ``added``/``removed`` episode ids, and ``changed_ids``. Reads only the episode tables, not
    the data files.
    """
    ea = version_episodes(a, lineage_root).set_index("episode_id")["content_hash"]
    eb = version_episodes(b, lineage_root).set_index("episode_id")["content_hash"]
    common = ea.index.intersection(eb.index)
    changed = sorted(common[(ea[common] != eb[common]).to_numpy()])
    return {
        "a": resolve_version(a, lineage_root)["dataset_version_id"],
        "b": resolve_version(b, lineage_root)["dataset_version_id"],
        "changed": len(changed), "identical": int(len(common) - len(changed)),
        "added": sorted(set(eb.index) - set(ea.index)), "removed": sorted(set(ea.index) - set(eb.index)),
        "changed_ids": changed,
    }


# ---------------------------------------------------------------------------------------------
# blindness of derived versions (evaluation-layer S0)
# ---------------------------------------------------------------------------------------------
# Grounding: PoC §2.1 item 4 (p. 3): "Deliberately corrupt known subsets of training data ... Then check
# whether your attribution method flags exactly those." Screening and attribution read a derived
# version's method-facing tables (episode table, catalog copy, LeRobot ``meta/episodes``). If any
# bookkeeping in them differs between edited and unedited episodes, a method can "flag exactly those"
# without having learned anything. v1 had two such channels: a ``parent_content_hash`` column that
# differed for exactly the 100 labelled rows, and row order (edited episodes at ``episode_index``
# 900-999, in their own data files). Data-derived columns may differ (a noised episode's smoothness
# does change); that is the signal a method is supposed to find.
#
# Ground-truth feature S2 (2026-10-15) reclassified the instruction: ``language_instruction`` (catalog) and
# ``tasks`` (``meta/episodes``) are CONTENT, not bookkeeping. Reason: ``catalog.episode_content_hash`` hashes
# the task string (``task.encode("utf-8")``, ``vlapoc/catalog.py:127-148``), so the instruction is already part
# of an episode version's identity, and a swapped instruction (PoC §2.5 p. 5 "Swapped or wrong language
# instructions") is a content edit that cannot be built if the blind check forbids it. Provenance columns
# (``episode_id``, ``task_name``, ``libero_task_id``, ``suite``, ``scene``, ``objects``, ``source_*``) stay
# bookkeeping: a swap leaves the collection record true (plan design D1). The matching consistency rule is that
# a version's ``meta/episodes.tasks`` must equal its own table's ``language_instruction`` per episode.

#: Columns whose value is computed from the episode's own content; an edit may change them. Includes the
#: instruction (hashed content, see the note above).
DATA_DERIVED_COLUMNS = ("content_hash", "length_steps", "duration_s", "smoothness", "idle_fraction", "idle_steps",
                        "gripper_toggles", "language_instruction")
#: Columns describing where an episode came from or sits; a derivation must copy them unchanged.
#: Any other column found in a table (not data-derived, not ``episode_id``) is treated as bookkeeping too.
BOOKKEEPING_COLUMNS = ("episode_index", "source_file", "source_demo", "init_state_sha256", "suite", "task_name",
                       "libero_task_id", "scene", "objects", "source", "operator", "collection_date")
#: LeRobot ``meta/episodes`` columns compared per episode; every other column there (length, data-file
#: and frame-range layout, per-episode stats, ``tasks``) follows the stored bytes and is data-derived.
META_BOOKKEEPING_COLUMNS = ("episode_index",)
ROW_ORDER = "(row order)"
META_MISMATCH = "(meta/episodes does not match the episode table)"


def _canon(v) -> str:
    """A comparable string for one table cell (numpy scalars and arrays become JSON values)."""
    return json.dumps(v, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def _differing(a: pd.Series, b: pd.Series) -> int:
    """Number of positions where two equally long series hold different values (cell by cell)."""
    return sum(_canon(x) != _canon(y) for x, y in zip(a, b))


def _compare_episode_tables(child: pd.DataFrame, parent: pd.DataFrame) -> dict:
    """Rule 1 (same columns), same episodes, same row order, Rule 2 (bookkeeping equal per episode_id)."""
    added = sorted(set(child.columns) - set(parent.columns))
    removed = sorted(set(parent.columns) - set(child.columns))
    out = {"added_columns": added, "removed_columns": removed, "differing_rows": {}}
    off = added + removed
    if len(child) != len(parent) or set(child["episode_id"]) != set(parent["episode_id"]) \
            or child["episode_id"].duplicated().any():
        out["offending"] = sorted(set(off + ["episode_id"]))
        return out
    if list(child["episode_id"]) != list(parent["episode_id"]):
        off.append(ROW_ORDER)
    c = child.set_index("episode_id")
    p = parent.set_index("episode_id").loc[c.index]
    for col in c.columns.intersection(p.columns):
        n = _differing(c[col], p[col])
        if n:
            out["differing_rows"][col] = n
            if col not in DATA_DERIVED_COLUMNS:
                off.append(col)
    out["offending"] = sorted(set(off))
    return out


def _meta_columns(root: Path) -> set[str]:
    """Every column name in a LeRobot dataset's ``meta/episodes`` parquet files (stats columns included)."""
    import pyarrow.parquet as pq

    return {n for f in (Path(root) / "meta" / "episodes").rglob("*.parquet") for n in pq.read_schema(f).names}


def _compare_meta(child_root: Path, child: pd.DataFrame, parent_root: Path, parent: pd.DataFrame) -> dict:
    """LeRobot ``meta/episodes`` of the child vs the parent, joined to ``episode_id`` through each
    version's own episode table: same column set, the table's lengths describe this copy, and
    ``episode_index`` (physical order) and ``tasks`` equal per episode."""
    from vlapoc.catalog import read_episodes_meta

    cc, pc = _meta_columns(child_root), _meta_columns(parent_root)
    out = {"added_columns": sorted(cc - pc), "removed_columns": sorted(pc - cc), "differing_rows": {}}
    off = out["added_columns"] + out["removed_columns"]
    metas = []
    for root, table in ((child_root, child), (parent_root, parent)):
        m = read_episodes_meta(root)
        id_of = dict(zip(table["episode_index"], table["episode_id"]))
        if sorted(id_of) != sorted(m["episode_index"]) or \
                (m["length"].to_numpy() != m["episode_index"].map(dict(zip(table["episode_index"], table["length_steps"]))).to_numpy()).any():
            out["offending"] = sorted(set(off + [META_MISMATCH]))
            return out
        metas.append(m.assign(episode_id=m["episode_index"].map(id_of)).set_index("episode_id"))
    cm, pm = metas[0], metas[1].loc[metas[0].index]
    if "language_instruction" in child.columns:  # the instruction is content: the two copies must agree
        li = child.set_index("episode_id")["language_instruction"].loc[cm.index]
        if (cm["tasks"].map(lambda t: list(t)[0]).to_numpy() != li.to_numpy()).any():
            out["offending"] = sorted(set(off + [META_MISMATCH]))
            return out
    for col in META_BOOKKEEPING_COLUMNS:
        n = _differing(cm[col], pm[col])
        if n:
            out["differing_rows"][col] = n
            off.append(col)
    out["offending"] = sorted(set(off))
    return out


def compare_blind(child: pd.DataFrame, parent: pd.DataFrame, child_root: Path, parent_root: Path,
                  catalog: pd.DataFrame | None = None) -> dict:
    """Do a derived version's method-facing tables reveal which episodes were edited?

    Compares, against the parent: the child's episode table (``child``), its catalog copy (``catalog``,
    if given) and its LeRobot ``meta/episodes`` (under ``child_root``). Rules (evaluation-layer plan S0):

    1. Column sets are identical (a derivation adds no column, e.g. no ``parent_content_hash``).
    2. Every bookkeeping column (``BOOKKEEPING_COLUMNS``, and any column that is neither data-derived nor
       ``episode_id``) equals the parent's value per ``episode_id``; so does the row order. In
       ``meta/episodes``, ``episode_index`` (physical order) does.

    Data-derived columns (``DATA_DERIVED_COLUMNS``; in ``meta/episodes`` length, file layout, stats and
    ``tasks``) are exempt. Reclassification (ground-truth feature S2, 2026-10-15): the instruction
    (``language_instruction`` in the tables, ``tasks`` in ``meta/episodes``) is hashed content
    (``catalog.episode_content_hash`` hashes the task string, ``vlapoc/catalog.py:127-148``), so an instruction
    swap may change it. Provenance (``task_name``, ``libero_task_id``, ``suite``, ``scene``, ``objects``,
    ``source_file``, ``source_demo``, ``episode_id``) stays bookkeeping: a swap leaves the collection record
    true (plan design D1), and changing any of it still fails. To keep the instruction from becoming a free
    channel, ``meta/episodes.tasks`` must equal the child table's ``language_instruction`` per episode
    (otherwise ``META_MISMATCH``). Returns ``{"offending": sorted column names (union over tables), "tables": per-table
    detail}``; an empty ``offending`` means blind. ``derive_version`` calls this before registering.
    """
    tables = {"version_episodes": _compare_episode_tables(child, parent)}
    if catalog is not None:
        tables["catalog"] = _compare_episode_tables(catalog, parent)
    tables["meta/episodes"] = _compare_meta(child_root, child, parent_root, parent)
    return {"offending": sorted({c for t in tables.values() for c in t["offending"]}), "tables": tables}


def _catalog_copy(v: dict, lineage_root: Path | None) -> Path:
    """Where ``derive_version`` writes a version's catalog copy (``<lineage root>/catalog/<dataset dir>``)."""
    return _root(lineage_root) / "catalog" / Path(v["lerobot_root"]).name / "episodes.parquet"


def blind_report(ref: str, lineage_root: Path | None = None) -> dict:
    """``compare_blind`` for a registered version against its parent, plus its fence record.

    A root version (no parent) has nothing derived and reports no offending column.
    """
    v = resolve_version(ref, lineage_root)
    out = {"version": v["dataset_version_id"], "name": v["name"], "parent": v.get("parent_version_id"),
           "fence": fence_record(ref, lineage_root)}
    if not v.get("parent_version_id"):
        return {**out, "offending": [], "tables": {}}
    pv = resolve_version(v["parent_version_id"], lineage_root)
    cat = _catalog_copy(v, lineage_root)
    res = compare_blind(version_episodes(ref, lineage_root), version_episodes(pv["dataset_version_id"], lineage_root),
                        REPO_ROOT / v["lerobot_root"], REPO_ROOT / pv["lerobot_root"],
                        pd.read_parquet(cat) if cat.is_file() else None)
    return {**out, **res}


def check_blind(ref: str, lineage_root: Path | None = None) -> list[str]:
    """The method-facing columns of a derived version that differ from its parent where they must not
    (``compare_blind``). ``[]`` means a method under test cannot tell edited episodes from the tables."""
    return blind_report(ref, lineage_root)["offending"]


def parent_hashes(ref: str, lineage_root: Path | None = None) -> pd.DataFrame:
    """LINEAGE-ONLY: ``(episode_id, parent_content_hash)`` for every episode of a derived version.

    The pre-edit fingerprint, kept in ``versions/<id>/parents.parquet`` and never in a method-facing
    table (PoC §2.1 item 4 p. 3): comparing it with ``content_hash`` names the edited episodes, so
    evaluation and lineage code may read it and a method under test may not. The same holds for the
    parent version's own table and ``diff_versions(parent, child)``, which reveal the same thing.
    """
    v = resolve_version(ref, lineage_root)
    p = _root(lineage_root) / "versions" / v["dataset_version_id"] / "parents.parquet"
    if not p.is_file():
        raise LineageError(f"version {ref} has no parents.parquet (not a derived version)")
    return pd.read_parquet(p)


def fence_record(ref: str, lineage_root: Path | None = None) -> dict | None:
    """The version's ``fence.json`` (why it must not be given to a method under test), or ``None``."""
    v = resolve_version(ref, lineage_root)
    p = _root(lineage_root) / "versions" / v["dataset_version_id"] / "fence.json"
    return json.loads(p.read_text()) if p.is_file() else None


def write_fence(ref: str, reason: str, details: dict, lineage_root: Path | None = None) -> Path:
    """Fence a version from method use: write ``versions/<id>/fence.json`` (lineage-only storage)."""
    v = resolve_version(ref, lineage_root)
    p = _root(lineage_root) / "versions" / v["dataset_version_id"] / "fence.json"
    _write_json(p, {"dataset_version_id": v["dataset_version_id"], "name": v["name"], "method_use": "fenced",
                    "reason": reason, "fenced_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    **details})
    return p


def assert_blind(ref: str, lineage_root: Path | None = None) -> None:
    """Raise ``LineageError`` unless the version may be handed to a method under test: not fenced and
    ``check_blind`` empty. Screening and attribution entry points call this on their input version.

    Two refusal messages: a version whose fence carries ``superseded_by`` and that leaks no column (an injected
    version replaced by a newer one, see ``vlapoc.groundtruth.retire_version``) is told which version to use
    instead; any other fence or leaking column (v1) gets the "not blind" message with the offending columns.
    """
    rep = blind_report(ref, lineage_root)
    fence = rep["fence"]
    if fence is not None and fence.get("superseded_by") and not rep["offending"]:
        raise LineageError(f"version {rep['name']} ({rep['version'][:12]}) is fenced: {fence.get('reason', 'superseded')} "
                           f"({str(fence['superseded_by'])[:12]}); use the superseding version (ground-truth feature r3)")
    if rep["fence"] is not None or rep["offending"]:
        raise LineageError(f"version {rep['name']} ({rep['version'][:12]}) is not blind to its edits: fenced="
                           f"{rep['fence'] is not None}, offending={rep['offending']} (PoC §2.1 item 4 p.3)")


# ---------------------------------------------------------------------------------------------
# CLI (usage in the module docstring)
# ---------------------------------------------------------------------------------------------
# One handler per leaf command, attached with argparse's ``set_defaults(func=...)``. Handlers for the
# MLflow-backed commands import ``vlapoc.runs`` / ``vlapoc.derive`` inside the function: both modules
# import this one, so a top-level import here would be circular. (Under ``python -m vlapoc.lineage``
# this file runs as ``__main__`` and ``vlapoc.runs`` loads a second copy as ``vlapoc.lineage``, which
# is why an error raised from runs prints as ``vlapoc.lineage.LineageError``.)


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def _head(items: list, n: int) -> list:
    """The first ``n`` items, plus a ``"... <k> more"`` marker when there are more."""
    return items[:n] + ([f"... {len(items) - n} more"] if len(items) > n else [])


def _cmd_version_register(a) -> None:
    _print(register_version(a.root, a.name, a.catalog, note=a.note))


def _cmd_version_show(a) -> None:
    v = resolve_version(a.ref)
    v["episodes_by_suite"] = version_episodes(a.ref).groupby("suite", sort=False).size().to_dict()
    _print(v)


def _cmd_version_verify(a) -> None:
    _print(verify_version(a.ref, a.root))


def _cmd_version_recompute(a) -> None:
    registered = resolve_version(a.ref)["dataset_version_id"]
    recomputed = recompute_version_id(a.ref, a.root)
    _print({"registered": registered, "recomputed": recomputed, "match": recomputed == registered})


def _cmd_version_diff(a) -> None:
    out = diff_versions(a.a, a.b)
    if not a.ids:
        out["changed_ids"] = _head(out["changed_ids"], 3)
    _print(out)
    print(f"changed {out['changed']}, identical {out['identical']}, added {len(out['added'])}, removed {len(out['removed'])}")


def _cmd_version_check_blind(a) -> None:
    rep = blind_report(a.ref)
    _print(rep)
    print(f"check_blind({rep['name']}) = {rep['offending']}; fenced: {rep['fence'] is not None}")


def _cmd_split_make(a) -> None:
    if a.track:  # split_run calls make_splits inside its MLflow run and returns the same record
        from vlapoc.runs import split_run

        run_id, record = split_run(a.version, a.reference_per_task, a.seed)
        print(json.dumps({"split_run_id": run_id}))
    else:
        record = make_splits(a.version, a.reference_per_task, a.seed)["record"]
    _print(record)


def _cmd_subset_show(a) -> None:
    s = Subset.load(a.ref)
    rec = s.to_record()
    if not a.ids:
        rec["episode_ids"] = f"<{len(s.episode_ids)} ids; --ids to print>"
    rec["resolve_head"] = s.resolve()[:10]
    _print(rec)


def _cmd_names(a) -> None:
    _print(load_names())


def _cmd_build_verify(a) -> None:
    from vlapoc.runs import build_run

    print(json.dumps({"build_run_id": build_run(a.version)}))


def _cmd_derive(a) -> None:
    if a.no_track:
        from vlapoc.derive import derive_demo_action_noise

        _print(derive_demo_action_noise(a.parent, a.name, a.per_task, a.seed))
    else:
        from vlapoc.runs import derive_run

        run_id, out = derive_run(a.parent, a.name, a.per_task, a.seed)
        _print({"derive_run_id": run_id, **out})


def _cmd_trace(a) -> None:
    from vlapoc.runs import trace

    _print(trace(a.run_id))


def _cmd_consumers(a) -> None:
    from vlapoc.runs import consumers

    episode_id, _, content_hash = a.episode_version.rpartition("@")
    _print(consumers(episode_id, content_hash))


def _cmd_run_diff(a) -> None:
    from vlapoc.runs import diff

    out = diff(a.run_a, a.run_b)
    if not a.ids:
        for k in ("added", "removed", "changed_version"):
            out["data"][k] = _head(out["data"][k], 5)
    _print(out)


def _cmd_check_scores(a) -> None:
    from vlapoc.runs import check_scores

    _print(check_scores(pd.read_parquet(a.table), a.train_run_id))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m vlapoc.lineage", description="Dataset versions, splits, subsets.")
    sub = p.add_subparsers(dest="cmd", required=True)

    version = sub.add_parser("version").add_subparsers(dest="vcmd", required=True)
    c = version.add_parser("register")
    c.add_argument("--root", type=Path, required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--catalog", type=Path, default=None)
    c.add_argument("--note", default=None)
    c.set_defaults(func=_cmd_version_register)
    c = version.add_parser("show")
    c.add_argument("ref")
    c.set_defaults(func=_cmd_version_show)
    c = version.add_parser("verify")
    c.add_argument("ref")
    c.add_argument("--root", type=Path, default=None)
    c.set_defaults(func=_cmd_version_verify)
    c = version.add_parser("recompute")
    c.add_argument("ref")
    c.add_argument("--root", type=Path, default=None)
    c.set_defaults(func=_cmd_version_recompute)
    c = version.add_parser("diff")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--ids", action="store_true")
    c.set_defaults(func=_cmd_version_diff)
    c = version.add_parser("check-blind", help="does a derived version's method-facing table reveal its edits?")
    c.add_argument("ref")
    c.set_defaults(func=_cmd_version_check_blind)

    c = sub.add_parser("split").add_subparsers(dest="scmd", required=True).add_parser("make")
    c.add_argument("--version", required=True)
    c.add_argument("--reference-per-task", type=int, default=2)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--track", action="store_true", help="record as an MLflow split run")
    c.set_defaults(func=_cmd_split_make)

    c = sub.add_parser("subset").add_subparsers(dest="sscmd", required=True).add_parser("show")
    c.add_argument("ref")
    c.add_argument("--ids", action="store_true", help="print the episode IDs too")
    c.set_defaults(func=_cmd_subset_show)

    sub.add_parser("names").set_defaults(func=_cmd_names)

    c = sub.add_parser("build").add_subparsers(dest="bcmd", required=True).add_parser(
        "verify", help="MLflow build run for an existing version (re-verify vs raw)")
    c.add_argument("version")
    c.set_defaults(func=_cmd_build_verify)

    c = sub.add_parser("derive", help="S4 demo derivation (RoboDrop action noise), recorded as an MLflow run")
    c.add_argument("--parent", default="v0")
    c.add_argument("--name", default="v1")
    c.add_argument("--demo-action-noise", action="store_true", required=True)
    c.add_argument("--per-task", type=int, default=5)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--no-track", action="store_true", help="derive without an MLflow run")
    c.set_defaults(func=_cmd_derive)

    c = sub.add_parser("trace")
    c.add_argument("run_id")
    c.set_defaults(func=_cmd_trace)
    c = sub.add_parser("consumers")
    c.add_argument("episode_version", help="<episode_id>@<content_hash>")
    c.set_defaults(func=_cmd_consumers)
    c = sub.add_parser("diff")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.add_argument("--ids", action="store_true")
    c.set_defaults(func=_cmd_run_diff)
    c = sub.add_parser("check-scores")
    c.add_argument("table", type=Path)
    c.add_argument("train_run_id")
    c.set_defaults(func=_cmd_check_scores)
    return p


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` (default: ``sys.argv[1:]``), run the chosen command, return exit code 0.

    Errors propagate as exceptions (non-zero exit with a traceback); argparse errors exit with 2.
    """
    args = _build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
