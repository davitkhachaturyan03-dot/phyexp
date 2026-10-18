"""The method-facing loader: the only way method code reads a dataset version (ground-truth feature S3).

Where this sits: between the stored dataset versions and any screening or attribution method (for now
``vlapoc.methods.heuristics``). ``vlapoc.groundtruth`` is the other side of the boundary: it reads the labels, this
module never does. Reading order: ``open_for_method`` (the four checks), then ``MethodView.iter_episodes``.

Grounding
---------
- PoC §2.1 item 4 (p. 3): "Deliberately corrupt known subsets of training data ... Then check whether your
  attribution method flags exactly those." A method that can read the answer off the data has not recovered
  anything, so what a method may open is enforced here, not left to convention.
- Owner decision 3 (brief.md, 2026-10-15), "Enforce now": one method-facing loader that always calls
  ``lineage.assert_blind`` and respects the v1 fence, so later screening and attribution features cannot read a
  leaky or fenced version by accident. The answer key is read only through ``vlapoc.groundtruth``.
- Evaluation-layer S0 (2026-10-02, owner ruling A): the fence (``versions/<id>/fence.json``) and
  ``check_blind``/``assert_blind``; this closes that feature's known issue "The v1 fence is recorded but not
  enforced at method entry points".

What ``open_for_method(subset_ref)`` does, every call
-----------------------------------------------------
1. ``Subset.load(subset_ref)``; ``verify_version`` on its version (a changed byte raises
   ``VersionIntegrityError``).
2. ``assert_blind`` on the version (fenced, or any bookkeeping column differing from the parent: ``LineageError``
   naming the fence / offending columns).
3. Refuses a ground-truth-derived subset: any subset whose derivation chain (``parent_subset_id`` /
   ``from_subset_id`` recursion) contains ``op == "oracle_drop"``, an op containing ``"groundtruth"``, or a
   ``groundtruth`` tag. Such a subset is built from the labels, so training or scoring on it is not blind.
4. Diff guard: one dataset lineage per process. Opening a version that is an ancestor or a descendant of one
   already opened in this process raises, because comparing the two versions' content hashes names the edited
   episodes (``lineage.parent_hashes`` docstring). ``_reset_session()`` clears the registry (tests only).

``MethodView`` (frozen) carries ``version_id``, ``subset_id``, ``lerobot_root``, ``episode_indices`` (from
``Subset.resolve``), ``catalog`` and ``reference_ids``, plus ``iter_episodes()``.

``catalog`` columns (exactly ``catalog.CATALOG_COLUMNS``; every one is checked by ``compare_blind``, which
compares each bookkeeping column per ``episode_id`` and tolerates only the data-derived ones, including the
instruction since S2): ``episode_id``, ``episode_index``, ``content_hash``, ``suite``, ``task_name``,
``libero_task_id``, ``language_instruction``, ``scene``, ``objects``, ``source``, ``operator``,
``collection_date``, ``length_steps``, ``duration_s``, ``init_state_sha256``, ``smoothness``,
``idle_fraction``, ``idle_steps``, ``gripper_toggles``, ``source_file``, ``source_demo``. Note (plan design D1):
``task_name`` and ``episode_id`` stay true after an instruction swap, so a provenance-vs-instruction
cross-check can find swapped episodes; that is a measured property, not a leak in the loader.

This module must not import ``vlapoc.groundtruth`` or read labels, parent hashes or fence files; the boundary
test scans its source for that.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from vlapoc.catalog import CATALOG_COLUMNS, read_episodes_meta, read_tasks
from vlapoc.derive import _canonical_episode, ancestry
from vlapoc.libero_convert import REPO_ROOT
from vlapoc.lineage import (
    LineageError,
    Subset,
    assert_blind,
    load_names,
    resolve_version,
    verify_version,
    version_episodes,
)

_OPENED: dict[tuple[str, str], str] = {}  # (lineage root, version id) -> subset ref first opened
_GT_OPS = ("oracle_drop",)


def _reset_session() -> None:
    """Forget which versions this process opened (tests only; a real run never calls this)."""
    _OPENED.clear()


def _is_groundtruth(deriv: dict) -> bool:
    """Is this subset derivation built from ground truth?"""
    op = str(deriv.get("op", ""))
    return op in _GT_OPS or "groundtruth" in op or bool(deriv.get("groundtruth"))


def _groundtruth_in_chain(subset: Subset, lineage_root: Path | None, depth: int = 0) -> str | None:
    """The first ground-truth op found walking the derivation chain from ``subset``, or ``None``."""
    d = subset.derivation
    if _is_groundtruth(d):
        return str(d.get("op", "groundtruth"))
    for key in ("parent_subset_id", "from_subset_id"):
        pid = d.get(key)
        if pid and depth < 50:
            try:
                parent = Subset.load(pid, lineage_root)
            except LineageError:
                continue
            hit = _groundtruth_in_chain(parent, lineage_root, depth + 1)
            if hit:
                return hit
    return None


def _check_lineage_isolation(vid: str, lineage_root: Path | None) -> None:
    """Raise if ``vid`` is an ancestor or descendant of a version already opened in this process."""
    key_root = str(lineage_root)
    mine = set(ancestry(vid, lineage_root))
    for (root, other), ref in _OPENED.items():
        if root != key_root or other == vid:
            continue
        if other in mine or vid in set(ancestry(other, lineage_root)):
            raise LineageError(
                f"refusing to open a second version of one lineage in this process: {ref!r} ({other[:12]}) was "
                f"opened and {vid[:12]} is its ancestor or descendant; comparing their content hashes would "
                "name the edited episodes (PoC §2.1 item 4 p.3; owner decision 3)")


@dataclass(frozen=True)
class MethodView:
    """What a method under test may see of a subset (module docstring lists the catalog columns)."""

    version_id: str
    subset_id: str
    lerobot_root: Path
    episode_indices: tuple[int, ...]
    catalog: pd.DataFrame
    reference_ids: tuple[str, ...]

    def iter_episodes(self):
        """Yield ``(episode_id, episode)`` in ``episode_index`` order for the subset's episodes.

        ``episode`` is the canonical dict (``task``, ``action`` (T, 7), ``state`` (T, 8), ``image``, ``image2``);
        each data file is read once.
        """
        meta = read_episodes_meta(self.lerobot_root)
        tasks = read_tasks(self.lerobot_root)
        id_of = dict(zip(self.catalog["episode_index"], self.catalog["episode_id"]))
        sel = meta[meta["episode_index"].isin(self.episode_indices)]
        for (ci, fi), g in sorted(sel.groupby(["data/chunk_index", "data/file_index"])):
            path = self.lerobot_root / "data" / f"chunk-{int(ci):03d}" / f"file-{int(fi):03d}.parquet"
            table = pq.read_table(path, filters=[("episode_index", "in", [int(i) for i in g["episode_index"]])]).to_pandas()
            for idx, frames in table.groupby("episode_index", sort=True):
                yield id_of[int(idx)], _canonical_episode(frames.sort_values("frame_index"), tasks[int(frames["task_index"].iloc[0])])


def open_for_method(subset_ref: str, lineage_root: Path | None = None) -> MethodView:
    """Open a subset for a method under test (steps 1-4 in the module docstring).

    ``subset_ref`` is a subset name (``v0/candidate``), id or unique prefix. Raises ``LineageError`` (fenced,
    not blind, ground-truth-derived, or a second version of one lineage), ``VersionIntegrityError`` (a file of
    the version changed), and registers the version as opened for the diff guard only on success.
    """
    subset = Subset.load(subset_ref, lineage_root)
    vid = subset.dataset_version_id
    verify_version(vid, lineage_root=lineage_root)
    assert_blind(vid, lineage_root)
    hit = _groundtruth_in_chain(subset, lineage_root)
    if hit:
        raise LineageError(f"subset {subset_ref!r} is ground-truth-derived (op {hit!r}); a method under test must not "
                           "be given a subset built from the labels (owner decision 3)")
    _check_lineage_isolation(vid, lineage_root)
    v = resolve_version(vid, lineage_root)
    cat = subset.episodes(lineage_root)[CATALOG_COLUMNS].sort_values("episode_index").reset_index(drop=True)
    refs: tuple[str, ...] = ()
    ref_name = f"{v['name']}/reference"
    if ref_name in load_names(lineage_root)["subsets"]:
        refs = tuple(Subset.load(ref_name, lineage_root).episode_ids)
    _OPENED[(str(lineage_root), vid)] = subset_ref
    return MethodView(vid, subset.subset_id, REPO_ROOT / v["lerobot_root"],
                      tuple(int(i) for i in subset.resolve(lineage_root)), cat, refs)


__all__ = ["open_for_method", "MethodView"]
