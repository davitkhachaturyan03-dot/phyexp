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
    subsets/<subset_id>.json
    splits/<split_id>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from vlapoc.catalog import DEFAULT_CATALOG_ROOT, HASH_DOMAIN, hash_stored_dataset, sha256_file
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
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_json(obj) -> str:
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def rel(path: Path) -> str:
    """Repo-relative POSIX path; refuses paths outside the repo so no manifest holds an absolute path."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError as e:
        raise LineageError(f"{p} is outside the repository; lineage records only repo-relative paths") from e


def _root(lineage_root: Path | None) -> Path:
    return Path(lineage_root) if lineage_root is not None else LINEAGE_ROOT


def _names_path(lineage_root: Path | None) -> Path:
    return _root(lineage_root) / "names.json"


def load_names(lineage_root: Path | None = None) -> dict:
    p = _names_path(lineage_root)
    if p.is_file():
        return json.loads(p.read_text())
    return {"versions": {}, "subsets": {}}


def _bind_name(kind: str, name: str, value: str, lineage_root: Path | None, force: bool = False) -> None:
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
    return int(episode_id.rsplit("demo_", 1)[1])


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
    """``dataset_version_id`` from the feature schema and the (episode_id, content_hash) pairs."""
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
    """sha256 + size of every file in a LeRobot dataset root (data parquet files and meta/)."""
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
    """Register the LeRobot dataset at ``root`` (with its S2 catalog) as a content-addressed version."""
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
    names = load_names(lineage_root)
    vid = names["versions"].get(ref, ref)
    vroot = _root(lineage_root) / "versions"
    cand = [p.name for p in vroot.iterdir()] if vroot.is_dir() else []
    matches = [c for c in cand if c == vid] or ([c for c in cand if c.startswith(vid)] if len(vid) >= 8 else [])
    if len(matches) != 1:
        raise LineageError(f"unknown or ambiguous version {ref!r}")
    return json.loads((vroot / matches[0] / "version.json").read_text())


def version_episodes(ref: str, lineage_root: Path | None = None) -> pd.DataFrame:
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

    ``root`` overrides the registered location (e.g. to check a copy on another machine).
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
    """A selection of episode versions from one dataset version. Never a data copy."""

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
        return {"subset_id": self.subset_id, "scheme": SUBSET_SCHEME, "dataset_version_id": self.dataset_version_id,
                "n_episodes": len(self.episode_ids), "episode_ids": list(self.episode_ids),
                "derivation": self.derivation}

    def save(self, lineage_root: Path | None = None) -> Path:
        self.episodes(lineage_root)  # validates membership
        p = _root(lineage_root) / "subsets" / f"{self.subset_id}.json"
        _write_json(p, self.to_record())
        return p

    @classmethod
    def load(cls, ref: str, lineage_root: Path | None = None) -> "Subset":
        """Load by subset name (``v0/candidate``), full id, or unique prefix (>= 8 hex chars)."""
        sid = load_names(lineage_root)["subsets"].get(ref, ref)
        sdir = _root(lineage_root) / "subsets"
        cand = [p.stem for p in sdir.glob("*.json")]
        matches = [c for c in cand if c == sid] or ([c for c in cand if c.startswith(sid)] if len(sid) >= 8 else [])
        if len(matches) != 1:
            raise LineageError(f"unknown or ambiguous subset {ref!r}")
        rec = json.loads((sdir / f"{matches[0]}.json").read_text())
        s = cls(rec["dataset_version_id"], tuple(rec["episode_ids"]), rec["derivation"])
        if s.subset_id != rec["subset_id"]:
            raise LineageError(f"subset file {matches[0]} does not hash to its id (edited by hand?)")
        return s


def make_splits(
    version: str, reference_per_task: int = 2, seed: int = 0, lineage_root: Path | None = None
) -> dict:
    """Held-out clean ``reference`` split (k per task, stratified, seeded) and the ``candidate`` rest.

    Rule: tasks are visited in (suite order, libero_task_id) order; within a task, episodes are sorted
    by demo number and ``numpy.random.default_rng(seed).choice(n, k, replace=False)`` (one generator,
    consumed in that task order) picks the reference episodes.
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


def _check_score_keys(parent: Subset, score_table: pd.DataFrame, metric: str, lineage_root: Path | None) -> None:
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
    even if a future numpy changes ``Generator.choice``'s stream."""
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


def diff_versions(a: str, b: str, lineage_root: Path | None = None) -> dict:
    """Episode-version difference between two dataset versions (by episode_id and content_hash)."""
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
# CLI
# ---------------------------------------------------------------------------------------------


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m vlapoc.lineage", description="Dataset versions, splits, subsets.")
    sub = p.add_subparsers(dest="cmd", required=True)
    pv = sub.add_parser("version").add_subparsers(dest="vcmd", required=True)
    r = pv.add_parser("register")
    r.add_argument("--root", type=Path, required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--catalog", type=Path, default=None)
    r.add_argument("--note", default=None)
    s = pv.add_parser("show")
    s.add_argument("ref")
    vv = pv.add_parser("verify")
    vv.add_argument("ref")
    vv.add_argument("--root", type=Path, default=None)
    rc = pv.add_parser("recompute")
    rc.add_argument("ref")
    rc.add_argument("--root", type=Path, default=None)
    ps = sub.add_parser("split").add_subparsers(dest="scmd", required=True)
    m = ps.add_parser("make")
    m.add_argument("--version", required=True)
    m.add_argument("--reference-per-task", type=int, default=2)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--track", action="store_true", help="record as an MLflow split run")
    pss = sub.add_parser("subset").add_subparsers(dest="sscmd", required=True)
    ss = pss.add_parser("show")
    ss.add_argument("ref")
    ss.add_argument("--ids", action="store_true", help="print the episode IDs too")
    sub.add_parser("names")
    vd = pv.add_parser("diff")
    vd.add_argument("a")
    vd.add_argument("b")
    vd.add_argument("--ids", action="store_true")
    pb = sub.add_parser("build").add_subparsers(dest="bcmd", required=True)
    bv = pb.add_parser("verify", help="MLflow build run for an existing version (re-verify vs raw)")
    bv.add_argument("version")
    d = sub.add_parser("derive", help="S4 demo derivation (RoboDrop action noise), recorded as an MLflow run")
    d.add_argument("--parent", default="v0")
    d.add_argument("--name", default="v1")
    d.add_argument("--demo-action-noise", action="store_true", required=True)
    d.add_argument("--per-task", type=int, default=5)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--no-track", action="store_true", help="derive without an MLflow run")
    t = sub.add_parser("trace")
    t.add_argument("run_id")
    cs = sub.add_parser("consumers")
    cs.add_argument("episode_version", help="<episode_id>@<content_hash>")
    df_ = sub.add_parser("diff")
    df_.add_argument("run_a")
    df_.add_argument("run_b")
    df_.add_argument("--ids", action="store_true")
    ck = sub.add_parser("check-scores")
    ck.add_argument("table", type=Path)
    ck.add_argument("train_run_id")
    a = p.parse_args(argv)

    if a.cmd == "version" and a.vcmd == "register":
        _print(register_version(a.root, a.name, a.catalog, note=a.note))
    elif a.cmd == "version" and a.vcmd == "show":
        v = resolve_version(a.ref)
        eps = version_episodes(a.ref)
        v["episodes_by_suite"] = eps.groupby("suite", sort=False).size().to_dict()
        _print(v)
    elif a.cmd == "version" and a.vcmd == "verify":
        _print(verify_version(a.ref, a.root))
    elif a.cmd == "version" and a.vcmd == "recompute":
        v = resolve_version(a.ref)
        new = recompute_version_id(a.ref, a.root)
        _print({"registered": v["dataset_version_id"], "recomputed": new, "match": new == v["dataset_version_id"]})
    elif a.cmd == "split":
        if a.track:
            from vlapoc.runs import split_run

            print(json.dumps({"split_run_id": split_run(a.version, a.reference_per_task, a.seed)}))
        out = make_splits(a.version, a.reference_per_task, a.seed)
        _print(out["record"])
    elif a.cmd == "subset":
        s = Subset.load(a.ref)
        rec = s.to_record()
        if not a.ids:
            rec["episode_ids"] = f"<{len(s.episode_ids)} ids; --ids to print>"
        rec["resolve_head"] = s.resolve()[:10]
        _print(rec)
    elif a.cmd == "names":
        _print(load_names())
    elif a.cmd == "version" and a.vcmd == "diff":
        out = diff_versions(a.a, a.b)
        if not a.ids:
            out["changed_ids"] = out["changed_ids"][:3] + ([f"... {len(out['changed_ids']) - 3} more"] if len(out["changed_ids"]) > 3 else [])
        _print(out)
        print(f"changed {out['changed']}, identical {out['identical']}, added {len(out['added'])}, removed {len(out['removed'])}")
    elif a.cmd == "build":
        from vlapoc.runs import build_run

        print(json.dumps({"build_run_id": build_run(a.version)}))
    elif a.cmd == "derive":
        if a.no_track:
            from vlapoc.derive import derive_demo_action_noise

            _print(derive_demo_action_noise(a.parent, a.name, a.per_task, a.seed))
        else:
            from vlapoc.runs import derive_run

            rid, out = derive_run(a.parent, a.name, a.per_task, a.seed)
            _print({"derive_run_id": rid, **out})
    elif a.cmd == "trace":
        from vlapoc.runs import trace

        _print(trace(a.run_id))
    elif a.cmd == "consumers":
        from vlapoc.runs import consumers

        eid, _, h = a.episode_version.rpartition("@")
        _print(consumers(eid, h))
    elif a.cmd == "diff":
        from vlapoc.runs import diff

        out = diff(a.run_a, a.run_b)
        if not a.ids:
            for k in ("added", "removed", "changed_version"):
                out["data"][k] = out["data"][k][:5] + ([f"... {len(out['data'][k]) - 5} more"] if len(out["data"][k]) > 5 else [])
        _print(out)
    elif a.cmd == "check-scores":
        from vlapoc.runs import check_scores

        _print(check_scores(pd.read_parquet(a.table), a.train_run_id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
