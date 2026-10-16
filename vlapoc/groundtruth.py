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
Later stories add selection, the injected-version derivation and recovery scoring here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vlapoc.derive import load_labels
from vlapoc.lineage import LineageError, parent_hashes, version_episodes


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


__all__ = ["load_answer_key"]
