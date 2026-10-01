"""Episode identity (stable ``episode_id`` + ``content_hash``) and the §2.2 episode catalog.

Grounding
---------
- PoC §2.1 item 1 (p. 3): "Every training run should be tied to exact dataset versions (content
  hashes), episode IDs, ...". This module defines both at episode resolution.
- PoC §2.2 "Episode catalog" (p. 4): "Each episode gets a stable ID and a metadata row: source, task,
  language instruction, scene, objects, operator, collection date, episode length, and basic action
  statistics (smoothness, idle time, gripper toggles). LeRobot's dataset format plus a Parquet
  metadata table works well." ``build_catalog`` writes that Parquet table.
- PoC §3.5 A (pp. 7-8): "Human reference = durations of LIBERO's original human-teleoperated
  demonstrations on the same tasks." ``duration_s`` is that duration (see below).
- PoC §1 (p. 3): "Per-episode scores become most valuable when aggregated by metadata."

Identity
--------
``episode_id = "<suite>/<task_name>/demo_<k>"`` comes from the raw source (suite directory, HDF5 stem
minus ``_demo``, robomimic demo key), not from LeRobot's positional ``episode_index``, which LeRobot's
own edit tools renumber (``dataset_tools._copy_and_reindex_data``: "Mapping from old episode indices to
new indices"). The ``demo_<k>`` key is also how CUPID's released code addresses demos
(``robomimic_replay_image_dataset.py``), so a CUPID fork can map back via ``source_file``/``source_demo``.

``content_hash`` (sha256, hex) over, in order::

    b"vlapoc.episode.v1\\0"
    uint64-LE byte length of the task string, then the task string (UTF-8)
    uint64-LE number of frames
    for each frame, in frame order:
        action              float32 little-endian, 7 values
        observation.state   float32 little-endian, 8 values
        observation.images.image   uint8 HWC 128x128x3 (decoded pixels, not PNG bytes)
        observation.images.image2  uint8 HWC 128x128x3

Bookkeeping columns (``index``, ``episode_index``, ``frame_index``, ``timestamp``, ``task_index``) are
excluded because they change when a dataset is reindexed. Pixels are hashed after decoding, so the
hash does not depend on PNG encoder settings or parquet file layout (measured: two conversions of the
same data write different parquet bytes, because each file embeds a random HF ``datasets``
fingerprint, while the decoded content is identical).

Catalog columns that LIBERO cannot supply
-----------------------------------------
- ``operator``: null. LIBERO p. 6: demos were "collected by human experts through teleoperation with
  3Dconnexion Spacemouse"; no per-demo operator identity is published or stored in the HDF5.
- ``collection_date``: null. Not stored in the HDF5 attributes and not published.
- ``scene``: for libero_10, the ``<ROOM>_SCENE<n>`` prefix of the task/BDDL name (e.g.
  ``KITCHEN_SCENE3``); libero_spatial task names carry no scene prefix, so ``scene`` is the raw
  ``env_args.problem_name`` (``libero_tabletop_manipulation``, one shared tabletop scene whose object
  layout varies by task).
- ``objects``: the ``*_main`` bodies of the per-demo MuJoCo model XML (``model_file`` attribute): every
  object instance in the scene, including distractors and fixtures (e.g. LIVING_ROOM_SCENE2 lists
  ketchup and milk that its tasks never touch; libero_spatial lists the cabinet and stove). Robot,
  mount and table are not ``*_main`` bodies and so are excluded. Constant within each task (measured).

Action statistics (no invented thresholds)
------------------------------------------
- ``smoothness``: mean over steps of ||a[t, :6] - a[t-1, :6]||_2 (lower is smoother).
- ``idle_fraction``: share of steps whose six non-gripper action dims are all exactly 0.0.
- ``gripper_toggles``: number of t with a[t, 6] != a[t-1, 6]; the gripper action is exactly +-1 in all
  1000 raw demos (measured), so this counts open/close switches.

``duration_s`` = ``length_steps / 20`` (20 Hz control, raw ``env_args.control_freq``). Every raw demo's
only ``reward == 1`` is at its last step (re-verified by ``--verify-raw``), so the last step is the
first success and ``duration_s`` is the human time-to-success T that the evaluation feature uses as the
HRT reference (PoC §3.5 pp. 7-8). The PoC notes this is approximate because demos and rollouts do not
share initial states (p. 8).

Where it sits
-------------
Second step (docs/dataflow.md, between "LIBERO demos" and the training dataset). It reads the
dataset ``vlapoc.libero_convert`` wrote, and writes ``data/lineage/catalog/<dataset>/`` with
``episodes.parquet`` (one row per episode, ``CATALOG_COLUMNS``), ``raw_files.parquet`` (sha256 of
each raw HDF5 file) and ``catalog.json``. ``vlapoc.lineage.register_version`` copies these into the
version directory. ``hash_stored_dataset`` and ``episode_content_hash`` are reused everywhere a
hash is recomputed: version recompute (S3), derivation re-checks (S4) and the build run (S5).

Usage (from the repo root)::

    .venv/bin/python -m vlapoc.catalog --version v0 --verify-raw              # rebuild v0's catalog (~30 s)
    .venv/bin/python -m vlapoc.catalog --version v0 --verify-raw --out data/scratch/catalog_check

Without ``--out`` this overwrites ``data/lineage/catalog/libero_spatial_10/``. A rebuild from
unchanged data is identical to the stored catalog (measured).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from vlapoc.libero_convert import (
    DEFAULT_OUT,
    DEFAULT_RAW_ROOT,
    FPS,
    REPO_ROOT,
    RawEpisodeRef,
    iter_raw_episodes,
    load_raw_episode,
)

HASH_DOMAIN = b"vlapoc.episode.v1\0"
IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
DEFAULT_CATALOG_ROOT = REPO_ROOT / "data" / "lineage" / "catalog"
# LIBERO paper p. 6 (docs/papers/LIBERO_Benchmark_arXiv-2306.03310.pdf).
SOURCE = "LIBERO human teleoperation (3Dconnexion SpaceMouse), 50 demos per task"
_SCENE_PREFIX = re.compile(r"^((?:[A-Z]+_)+SCENE\d+)_")


def episode_id_of(ref: RawEpisodeRef) -> str:
    """The stable ``episode_id``, ``"<suite>/<task_name>/demo_<k>"``, of a raw demo."""
    return f"{ref.suite}/{ref.task_name}/{ref.source_demo}"


def episode_content_hash(
    task: str, action: np.ndarray, state: np.ndarray, image: np.ndarray, image2: np.ndarray
) -> str:
    """sha256 (hex) of the canonical episode content (byte layout in the module docstring).

    Inputs, all with T frames: ``action`` (T, 7) and ``state`` (T, 8), cast to little-endian float32;
    ``image`` and ``image2`` (T, 128, 128, 3) uint8 decoded pixels; ``task`` the instruction string.
    Every hash in the lineage layer goes through this one function, whether the arrays came from
    raw HDF5, from the stored parquet, or from an edit's transform.
    """
    n = len(action)
    if not (len(state) == len(image) == len(image2) == n):
        raise ValueError("per-frame arrays must have equal length")
    action = np.ascontiguousarray(action, dtype="<f4")
    state = np.ascontiguousarray(state, dtype="<f4")
    image = np.ascontiguousarray(image, dtype=np.uint8)
    image2 = np.ascontiguousarray(image2, dtype=np.uint8)
    t = task.encode("utf-8")
    h = hashlib.sha256()
    h.update(HASH_DOMAIN)
    h.update(struct.pack("<Q", len(t)))
    h.update(t)
    h.update(struct.pack("<Q", n))
    for i in range(n):
        h.update(action[i].tobytes())
        h.update(state[i].tobytes())
        h.update(image[i].tobytes())
        h.update(image2[i].tobytes())
    return h.hexdigest()


def sha256_file(path: Path, bufsize: int = 1 << 22) -> str:
    """sha256 (hex) of a file's bytes, read in 4 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------
# Reading a stored LeRobot v3 dataset
# ---------------------------------------------------------------------------------------------


def read_episodes_meta(root: Path) -> pd.DataFrame:
    """LeRobot ``meta/episodes`` (all files) without the per-episode stats columns."""
    cols = ["episode_index", "tasks", "length", "data/chunk_index", "data/file_index",
            "dataset_from_index", "dataset_to_index"]
    files = sorted((Path(root) / "meta" / "episodes").rglob("*.parquet"))
    df = pd.concat(pq.read_table(f, columns=cols).to_pandas() for f in files)
    return df.sort_values("episode_index").reset_index(drop=True)


def read_tasks(root: Path) -> dict[int, str]:
    """``task_index -> task string`` from ``meta/tasks.parquet``.

    Handles both pandas views of the file: a ``task`` column, or the task string as the index with
    only ``task_index`` as a column, which is what ``to_pandas()`` gives for LeRobot 0.6.1's file
    (it stores pandas metadata that restores ``task`` as the index; measured on v0).
    """
    t = pq.read_table(Path(root) / "meta" / "tasks.parquet").to_pandas()
    if "task" in t.columns:
        return dict(zip(t["task_index"], t["task"]))
    return {int(v): k for k, v in t["task_index"].items()}


def data_files(root: Path) -> list[Path]:
    """Every ``data/chunk-*/file-*.parquet`` of a LeRobot v3 dataset, sorted."""
    return sorted((Path(root) / "data").rglob("*.parquet"))


def decode_png(cell: dict) -> np.ndarray:
    """One stored image cell (``{"bytes": <PNG>, ...}``) -> (H, W, 3) uint8 RGB pixels.

    The single stored-side decoder: ``vlapoc.derive.read_stored_episode`` uses it too, so a derived
    episode's expected hash and its stored hash decode pixels the same way.
    """
    return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))


def _hash_data_file(args: tuple[str, dict[int, str]]) -> list[dict]:
    """Hash every episode stored in one data parquet file, plus its action statistics."""
    path, tasks = args
    table = pq.read_table(path).to_pandas()
    out = []
    for ep_idx, g in table.groupby("episode_index", sort=True):
        g = g.sort_values("frame_index")
        if not np.array_equal(g["frame_index"].to_numpy(), np.arange(len(g))):
            raise ValueError(f"episode {ep_idx} in {path} has non-contiguous frame_index")
        task_idx = g["task_index"].unique()
        if len(task_idx) != 1:
            raise ValueError(f"episode {ep_idx} has {len(task_idx)} tasks")
        action = np.stack(g["action"].to_numpy()).astype(np.float32)
        state = np.stack(g["observation.state"].to_numpy()).astype(np.float32)
        img = np.stack([decode_png(c) for c in g[IMAGE_KEYS[0]]])
        img2 = np.stack([decode_png(c) for c in g[IMAGE_KEYS[1]]])
        task = tasks[int(task_idx[0])]
        out.append(
            {
                "episode_index": int(ep_idx),
                "content_hash": episode_content_hash(task, action, state, img, img2),
                "task": task,
                "length_steps": len(g),
                **action_stats(action),
            }
        )
    return out


def hash_stored_dataset(root: Path, workers: int = 8) -> pd.DataFrame:
    """Per-episode content hash computed from the stored LeRobot dataset (decoded PNG pixels).

    Returns one row per episode, sorted by ``episode_index``, with columns ``episode_index``,
    ``content_hash``, ``task``, ``length_steps`` and the ``action_stats`` columns. Data files are
    hashed in parallel, one process per file (``version recompute v0``, which is mostly this call,
    takes 19 s on 8 workers; measured). Raises if an episode spans two data files, which the per-file
    hashing assumes never happens.
    """
    root = Path(root)
    tasks = read_tasks(root)
    meta = read_episodes_meta(root)
    files = data_files(root)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for part in ex.map(_hash_data_file, [(str(f), tasks) for f in files]) for r in part]
    df = pd.DataFrame(rows).sort_values("episode_index").reset_index(drop=True)
    if len(df) != len(meta) or not (df["episode_index"].to_numpy() == meta["episode_index"].to_numpy()).all():
        raise ValueError("an episode is split across data files or missing; hashing assumes one file per episode")
    if not (df["length_steps"].to_numpy() == meta["length"].to_numpy()).all():
        raise ValueError("stored frame counts disagree with meta/episodes lengths")
    return df


# ---------------------------------------------------------------------------------------------
# Raw side
# ---------------------------------------------------------------------------------------------


def action_stats(action: np.ndarray) -> dict:
    """§2.2 "basic action statistics (smoothness, idle time, gripper toggles)"; see module docstring.

    ``action`` is (T, 7): six arm dims then the gripper. Returns ``smoothness`` (mean L2 step change of
    the arm dims, action units per step), ``idle_steps``, ``idle_fraction`` and ``gripper_toggles``.
    """
    a = action.astype(np.float64)
    d = np.diff(a[:, :6], axis=0)
    return {
        "smoothness": float(np.linalg.norm(d, axis=1).mean()) if len(d) else 0.0,
        "idle_steps": int((a[:, :6] == 0.0).all(axis=1).sum()),
        "idle_fraction": float((a[:, :6] == 0.0).all(axis=1).mean()),
        "gripper_toggles": int((a[1:, 6] != a[:-1, 6]).sum()),
    }


def _raw_file_worker(args: tuple[str, list[tuple[int, dict]]]) -> list[dict]:
    """Per raw file: re-hash each demo from raw (rotate + cast) and read its catalog metadata."""
    source_file, items = args
    path = REPO_ROOT / source_file
    out = []
    with h5py.File(path, "r") as h5:
        env_args = json.loads(h5["data"].attrs["env_args"])
        problem_name = env_args["problem_name"]
        control_freq = int(env_args["env_kwargs"]["control_freq"])
        for order, ref in items:
            g = h5["data"][ref["source_demo"]]
            xml = g.attrs["model_file"]
            objects = [b[: -len("_main")] for b in re.findall(r'<body[^>]*name="([^"]+_main)"', xml)]
            init_state = np.ascontiguousarray(g.attrs["init_state"], dtype="<f8")
            rewards = g["rewards"][:]
            dones = g["dones"][:]
            out.append(
                {
                    "order": order,
                    "objects": objects,
                    "problem_name": problem_name,
                    "control_freq": control_freq,
                    "init_state_sha256": hashlib.sha256(init_state.tobytes()).hexdigest(),
                    "init_state_eq_states0": bool(np.array_equal(g.attrs["init_state"], g["states"][0])),
                    "raw_num_samples": int(g.attrs["num_samples"]),
                    "reward_sum": int(rewards.sum()),
                    "first_success_step": int(np.argmax(rewards == 1)) if rewards.any() else -1,
                    "dones_last_only": bool(dones.sum() == 1 and dones[-1] == 1),
                }
            )
    # content hash via the same raw->canonical mapping the converter used
    for rec, (order, ref) in zip(out, items):
        ep = load_raw_episode(path, ref["source_demo"])
        rec["raw_content_hash"] = episode_content_hash(ep["task"], ep["action"], ep["state"], ep["image"], ep["image2"])
        rec["language_instruction"] = ep["task"]
    return out


def scan_raw(raw_root: Path = DEFAULT_RAW_ROOT, workers: int = 8) -> pd.DataFrame:
    """One row per raw demo, in converter output order, with raw re-hash and catalog metadata.

    Columns: the identity columns (``episode_id``, ``suite``, ``task_name``, ``libero_task_id``,
    ``source_file``, ``source_demo``), ``raw_content_hash`` and ``language_instruction``, the
    catalog metadata (``objects``, ``init_state_sha256``, ``problem_name``) and the raw checks
    (``reward_sum``, ``first_success_step``, ``dones_last_only``, ``raw_num_samples``, ...).
    One worker process per raw file.
    """
    refs = list(iter_raw_episodes(raw_root))
    by_file: dict[str, list] = {}
    for i, r in enumerate(refs):
        by_file.setdefault(r.source_file, []).append((i, r.__dict__))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [x for part in ex.map(_raw_file_worker, list(by_file.items())) for x in part]
    raw = pd.DataFrame(rows).sort_values("order").reset_index(drop=True)
    ids = pd.DataFrame(
        [{"episode_id": episode_id_of(r), "suite": r.suite, "task_name": r.task_name,
          "libero_task_id": r.libero_task_id, "source_file": r.source_file, "source_demo": r.source_demo}
         for r in refs]
    )
    return pd.concat([ids, raw.drop(columns=["order"])], axis=1)


def raw_file_hashes(raw_root: Path = DEFAULT_RAW_ROOT, workers: int = 8) -> pd.DataFrame:
    """sha256 + size of every raw HDF5 file (the dataset's provenance root, PoC §2.1 p.3)."""
    files = sorted({r.source_file for r in iter_raw_episodes(raw_root)})
    with ProcessPoolExecutor(max_workers=workers) as ex:
        digests = list(ex.map(sha256_file, [REPO_ROOT / f for f in files]))
    return pd.DataFrame(
        {"source_file": files, "sha256": digests, "size_bytes": [(REPO_ROOT / f).stat().st_size for f in files]}
    )


def scene_of(task_name: str, problem_name: str) -> str:
    """``scene`` column: the ``<ROOM>_SCENE<n>`` task-name prefix, else the raw ``problem_name``."""
    m = _SCENE_PREFIX.match(task_name)
    return m.group(1) if m else problem_name


# ---------------------------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------------------------

# Columns of episodes.parquet, in order. A derived version's table has these plus parent_content_hash.
CATALOG_COLUMNS = [
    "episode_id", "episode_index", "content_hash", "suite", "task_name", "libero_task_id",
    "language_instruction", "scene", "objects", "source", "operator", "collection_date",
    "length_steps", "duration_s", "init_state_sha256", "smoothness", "idle_fraction",
    "idle_steps", "gripper_toggles", "source_file", "source_demo",
]


def build_catalog(root: Path = DEFAULT_OUT, raw_root: Path = DEFAULT_RAW_ROOT, workers: int = 8) -> tuple[pd.DataFrame, dict]:
    """Build the §2.2 catalog for the dataset at ``root`` (converted from ``raw_root`` by S1).

    Returns (catalog, checks). ``checks`` holds the raw cross-checks: the stored-vs-raw hash
    comparison and the reward/length re-verification.
    """
    stored = hash_stored_dataset(root, workers)
    raw = scan_raw(raw_root, workers)
    if len(stored) != len(raw):
        raise ValueError(f"stored dataset has {len(stored)} episodes, raw has {len(raw)}")
    # Converter writes raw demos in iter_raw_episodes order, so row i <-> episode_index i.
    cat = raw.copy()
    cat["episode_index"] = stored["episode_index"].to_numpy()
    cat["content_hash"] = stored["content_hash"].to_numpy()
    cat["length_steps"] = stored["length_steps"].to_numpy()
    for c in ("smoothness", "idle_fraction", "idle_steps", "gripper_toggles"):
        cat[c] = stored[c].to_numpy()
    cat["scene"] = [scene_of(t, p) for t, p in zip(cat["task_name"], cat["problem_name"])]
    cat["source"] = SOURCE
    cat["operator"] = pd.Series([None] * len(cat), dtype="object")
    cat["collection_date"] = pd.Series([None] * len(cat), dtype="object")
    cat["duration_s"] = cat["length_steps"] / FPS
    checks = {
        "n_episodes": int(len(cat)),
        "unique_episode_ids": int(cat["episode_id"].nunique()),
        "hash_match_raw": int((cat["content_hash"] == cat["raw_content_hash"]).sum()),
        "task_string_match": int((stored["task"].to_numpy() == cat["language_instruction"].to_numpy()).sum()),
        "length_eq_raw_num_samples": int((cat["length_steps"] == cat["raw_num_samples"]).sum()),
        "reward_sum_eq_1": int((cat["reward_sum"] == 1).sum()),
        "first_success_is_last_step": int((cat["first_success_step"] == cat["length_steps"] - 1).sum()),
        "dones_last_only": int(cat["dones_last_only"].sum()),
        "init_state_eq_states0": int(cat["init_state_eq_states0"].sum()),
        "control_freq_values": sorted(set(int(x) for x in cat["control_freq"])),
    }
    return cat[CATALOG_COLUMNS].copy(), checks


def write_catalog(cat: pd.DataFrame, raw_files: pd.DataFrame, out_dir: Path, root: Path) -> None:
    """Write ``episodes.parquet``, ``raw_files.parquet`` and ``catalog.json`` (with the repo-relative
    dataset root, which ``register_version`` checks) into ``out_dir``. ``root`` must be inside the repo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cat.to_parquet(out_dir / "episodes.parquet", index=False)
    raw_files.to_parquet(out_dir / "raw_files.parquet", index=False)
    rel_root = Path(root).resolve().relative_to(REPO_ROOT).as_posix()
    (out_dir / "catalog.json").write_text(
        json.dumps({"lerobot_root": rel_root, "n_episodes": len(cat), "hash_scheme": HASH_DOMAIN.decode().rstrip("\0")}, indent=2)
    )


def summarize(cat: pd.DataFrame) -> dict:
    """Per-suite summary printed by the CLI: counts, duration_s [min, median, max], idle steps,
    gripper toggle range, smoothness [min, median, max] and scenes."""
    s = {}
    for suite, g in cat.groupby("suite", sort=False):
        s[suite] = {
            "episodes": int(len(g)),
            "tasks": int(g["task_name"].nunique()),
            "duration_s": [float(g["duration_s"].min()), float(g["duration_s"].median()), float(g["duration_s"].max())],
            "idle_steps_total": int(g["idle_steps"].sum()),
            "episodes_with_idle": int((g["idle_steps"] > 0).sum()),
            "gripper_toggles": [int(g["gripper_toggles"].min()), int(g["gripper_toggles"].max())],
            "smoothness": [round(float(g["smoothness"].min()), 4), round(float(g["smoothness"].median()), 4),
                           round(float(g["smoothness"].max()), 4)],
            "scenes": sorted(g["scene"].unique().tolist()),
        }
    return s


def main(argv: list[str] | None = None) -> int:
    """CLI entry (usage in the module docstring). Exit 1 under ``--verify-raw`` if any hash mismatches."""
    p = argparse.ArgumentParser(description="Build the §2.2 episode catalog and check it against raw.")
    p.add_argument("--root", type=Path, default=DEFAULT_OUT, help="LeRobot dataset root")
    p.add_argument("--version", default=None, help="a registered version name (e.g. v0); overrides --root")
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW_ROOT)
    p.add_argument("--out", type=Path, default=None, help="default: data/lineage/catalog/<root name>/")
    p.add_argument("--verify-raw", action="store_true", help="fail unless every stored hash equals its raw re-hash")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args(argv)
    root = Path(a.root)
    if a.version:
        from vlapoc.lineage import resolve_version

        root = REPO_ROOT / resolve_version(a.version)["lerobot_root"]
    t0 = time.time()
    cat, checks = build_catalog(root, a.raw, a.workers)
    raw_files = raw_file_hashes(a.raw, a.workers)
    out = a.out or DEFAULT_CATALOG_ROOT / root.name
    write_catalog(cat, raw_files, out, root)
    print(f"catalog: {len(cat)} rows x {len(cat.columns)} columns -> {out.resolve().relative_to(REPO_ROOT)} ({time.time() - t0:.0f}s)")
    print(f"unique episode_id: {checks['unique_episode_ids']}/{checks['n_episodes']}")
    print(f"{checks['hash_match_raw']}/{checks['n_episodes']} hashes match raw")
    print("checks:", json.dumps(checks))
    print("raw files:", len(raw_files), "sha256 recorded;", f"{raw_files['size_bytes'].sum() / 1e9:.2f} GB")
    print("per-suite summary:", json.dumps(summarize(cat), indent=1))
    if a.verify_raw and checks["hash_match_raw"] != checks["n_episodes"]:
        print("FAIL: stored dataset is not bit-exact to raw", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
