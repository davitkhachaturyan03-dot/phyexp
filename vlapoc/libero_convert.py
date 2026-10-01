"""Convert raw LIBERO demonstrations (libero_spatial + libero_10) into one LeRobot v3 dataset.

Grounding
---------
- PoC §10 "First milestone" (p. 20): Foundation = "Lineage tracking; episode catalog; LIBERO (one or
  two suites) + SmolVLA". We convert the two suites on disk.
- PoC §2.2 "Episode catalog" (p. 4): "LeRobot's dataset format plus a Parquet metadata table works
  well." This module produces the LeRobot dataset; ``vlapoc.catalog`` produces the metadata table.
- PoC §2.1 item 2 (p. 3): "SmolVLA via LeRobot". The schema below is LeRobot's LIBERO schema so that
  ``lerobot-train`` / ``lerobot-eval`` and SmolVLA need no key remapping:
    * keys and names copied from ``lerobot/libero`` meta/info.json @ a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4
      and ``HuggingFaceVLA/libero`` @ 86958911c0f959db2bbbdb107eb3e17c5f9c798e (the PNG variant);
    * camera mapping from ``lerobot/envs/libero.py`` (lerobot 0.6.1):
      ``{"agentview_image": "image", "robot0_eye_in_hand_image": "image2"}``;
    * state from ``lerobot/processor/env_processor.py``:
      ``state = torch.cat((eef_pos, eef_axisangle, gripper_qpos), dim=-1)`` (8 dims). Raw LIBERO stores
      exactly these as ``obs/ee_pos``, ``obs/ee_ori`` (axis-angle) and ``obs/gripper_states``
      (measured: ``ee_pos == robot_states[:, 2:5]``, ``gripper_states == robot_states[:, 0:2]``,
      ``ee_ori == quat2axisangle(robot_states[:, 5:9])``);
    * images: the env processor "Rotates images by 180 degrees by flipping both height and width
      dimensions", so raw frames are stored flipped in H and W, matching the hub layout.
- ``fps = 20``: raw ``env_args.control_freq = 20`` (measured) and LeRobot's own LIBERO env config says
  ``fps: int = 20  # Must match robosuite's default control_freq (20 Hz)`` (lerobot/envs/configs.py).
- Task order (``libero_task_id``): LIBERO ``libero/libero/benchmark/libero_suite_task_map.py``, last
  changed in commit 4f0dd188960c25b608babc7de34e1e20a9d1306f ("init"); identical at master
  8f1084e3132a39270c3a13ebe37270a43ece2a01 (fetched 2026-09-29). This is not alphabetical order.

Deliberate divergences from the hub LIBERO datasets (``lerobot/libero``, ``HuggingFaceVLA/libero``)
------------------------------------------------------------------------------------------------
1. **All 1000 demos, unfiltered, full length.** The hub corpus is OpenVLA's "no-noops" set: 811 of our
   1000 demos, 68 of them shortened. Every raw demo ends with its only ``reward == 1`` at the last step
   (re-verified by ``scripts/verify_libero_convert.py``), so there are no failed demos to drop
   (CLAUDE.md / vla-build rule against the 189-demo filter).
2. **Native 128x128** frames, no upsampling or simulator re-render (hub: 256x256 re-render).
3. **fps 20** (hub declares 10 although the data is 20 Hz control).
4. **Lossless PNG-in-parquet** (``use_videos=False``, dtype ``image``) like ``HuggingFaceVLA/libero``,
   not AV1 video like ``lerobot/libero``. This keeps every stored frame bit-exact to raw, which the
   per-episode content hash (``vlapoc.catalog``) relies on.

Episode order in the output: suites in ``SUITES`` order, tasks in LIBERO benchmark order, demos in
numeric ``demo_<k>`` order. LeRobot's ``episode_index`` is positional and is NOT the episode's identity;
``vlapoc.catalog`` assigns the stable ``episode_id``.

Where it sits
-------------
First step of the pipeline (docs/dataflow.md, the "LIBERO demos (50/task, human teleop)" box):
raw HDF5 under ``data/raw/libero/`` in, the v0 dataset under ``data/lerobot/libero_spatial_10/`` out.
``vlapoc.catalog`` then hashes every stored episode, and ``vlapoc.lineage`` registers the result as
version ``v0``. ``load_raw_episode`` is also reused by the catalog to re-hash raw data.

Usage (from the repo root)::

    .venv/bin/python -m vlapoc.libero_convert --out data/lerobot/libero_spatial_10
    .venv/bin/python -m vlapoc.libero_convert --out data/scratch/smoke20 --limit-per-task 1   # smoke test

The full conversion took 7.9 min and wrote 7.36 GB (file bytes) on the M4 Pro (execution log, S1).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "raw" / "libero"
DEFAULT_OUT = REPO_ROOT / "data" / "lerobot" / "libero_spatial_10"
DEFAULT_REPO_ID = "local/libero_spatial_10"

SUITES = ("libero_spatial", "libero_10")
FPS = 20
IMAGE_SHAPE = (128, 128, 3)
ROBOT_TYPE = "panda"

# LIBERO benchmark task order, libero_suite_task_map.py @ 4f0dd188 (see module docstring).
LIBERO_TASK_ORDER: dict[str, tuple[str, ...]] = {
    "libero_spatial": (
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    ),
    "libero_10": (
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    ),
}

# LeRobot LIBERO schema (keys, dtypes, names from the hub info.json files named above; image shape
# is our native 128x128; dtype "image" because storage is lossless PNG, see divergence 4).
FEATURES: dict[str, dict] = {
    "observation.images.image": {
        "dtype": "image",
        "shape": list(IMAGE_SHAPE),
        "names": ["height", "width", "channel"],
    },
    "observation.images.image2": {
        "dtype": "image",
        "shape": list(IMAGE_SHAPE),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {"dtype": "float32", "shape": [8], "names": ["state"]},
    "action": {"dtype": "float32", "shape": [7], "names": ["actions"]},
}


@dataclass(frozen=True)
class RawEpisodeRef:
    """Where one raw demonstration lives, plus its task coordinates.

    ``source_file`` is a repo-relative POSIX path, so it is machine-independent; it falls back to the
    absolute path only if the raw root lies outside the repository (see ``_rel``).
    ``libero_task_id`` is the task's position in the LIBERO benchmark order *within its suite*
    (0-9 in each suite), so it is not unique across suites; ``task_name`` is.
    """

    suite: str
    libero_task_id: int
    task_name: str
    source_file: str  # e.g. "data/raw/libero/libero_spatial/<task>_demo.hdf5"
    source_demo: str  # e.g. "demo_7"


def raw_task_path(raw_root: Path, suite: str, task_name: str) -> Path:
    """Raw LIBERO file for a task: the HDF5 stem is the task name plus ``_demo`` (measured)."""
    return raw_root / suite / f"{task_name}_demo.hdf5"


def _demo_keys(h5: h5py.File) -> list[str]:
    """``demo_<k>`` keys of a raw file in numeric order (string order would put demo_10 before demo_2)."""
    return sorted(h5["data"].keys(), key=lambda s: int(s.split("_")[1]))


def _rel(path: Path) -> str:
    """Repo-relative POSIX path, or the absolute path for a file outside the repo.

    Unlike ``vlapoc.lineage.rel``, this does not refuse outside paths: the converter may be pointed at
    raw data anywhere, while the lineage store must never record an absolute path.
    """
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def iter_raw_episodes(raw_root: Path = DEFAULT_RAW_ROOT, limit_per_task: int | None = None):
    """Yield a ``RawEpisodeRef`` for every raw demo in output order: suite, LIBERO task order, numeric demo order.

    This order *is* the converted dataset's ``episode_index`` order (the converter writes episodes in
    it), which ``vlapoc.catalog.build_catalog`` relies on to line raw rows up with stored episodes.
    ``limit_per_task`` keeps only the first k demos of each task (smoke tests). Raises
    ``FileNotFoundError`` if any of the 20 task files is missing.
    """
    for suite in SUITES:
        for task_id, task_name in enumerate(LIBERO_TASK_ORDER[suite]):
            path = raw_task_path(raw_root, suite, task_name)
            if not path.is_file():
                raise FileNotFoundError(f"missing raw LIBERO file: {path}")
            with h5py.File(path, "r") as h5:
                keys = _demo_keys(h5)
            if limit_per_task is not None:
                keys = keys[:limit_per_task]
            for key in keys:
                yield RawEpisodeRef(suite, task_id, task_name, _rel(path), key)


def language_instruction(h5: h5py.File) -> str:
    """The task string: ``data.attrs['problem_info']['language_instruction']`` of the raw file."""
    return json.loads(h5["data"].attrs["problem_info"])["language_instruction"]


def load_raw_episode(path: Path, demo_key: str) -> dict:
    """Read one demo and map it to the LeRobot LIBERO schema (rotate 180 deg, cast to float32).

    Returns ``task`` (str) and arrays ``image``/``image2`` (T,128,128,3) uint8, ``state`` (T,8)
    float32, ``action`` (T,7) float32, plus raw bookkeeping ``rewards``, ``dones``, ``num_samples``.
    This is the single definition of "raw -> canonical frames"; the converter writes it and
    ``vlapoc.catalog`` re-hashes raw data through it to prove the stored dataset is bit-exact.
    """
    with h5py.File(path, "r") as h5:
        task = language_instruction(h5)
        g = h5["data"][demo_key]
        obs = g["obs"]
        agent = obs["agentview_rgb"][:]
        wrist = obs["eye_in_hand_rgb"][:]
        state = np.concatenate(
            [obs["ee_pos"][:], obs["ee_ori"][:], obs["gripper_states"][:]], axis=1
        ).astype(np.float32)
        action = g["actions"][:].astype(np.float32)
        rewards = g["rewards"][:]
        dones = g["dones"][:]
        num_samples = int(g.attrs["num_samples"])
    return {
        "task": task,
        # 180-degree rotation == flip H and W (lerobot env_processor.py); contiguous for PIL/hashing.
        "image": np.ascontiguousarray(agent[:, ::-1, ::-1, :]),
        "image2": np.ascontiguousarray(wrist[:, ::-1, ::-1, :]),
        "state": state,
        "action": action,
        "rewards": rewards,
        "dones": dones,
        "num_samples": num_samples,
    }


def convert(
    out_root: Path = DEFAULT_OUT,
    raw_root: Path = DEFAULT_RAW_ROOT,
    repo_id: str = DEFAULT_REPO_ID,
    limit_per_task: int | None = None,
    image_writer_threads: int = 4,
    overwrite: bool = False,
    log_every: int = 25,
) -> dict:
    """Write the LeRobot v3 dataset with LeRobot's own writer (create/add_frame/save_episode/finalize).

    No filtering: every demo in every raw file is written at full length.

    Args:
        out_root: dataset root to create. LeRobot refuses an existing root, so with ``overwrite=True``
            an existing *LeRobot* root (one with ``meta/info.json``) is deleted first; any other
            non-empty directory is refused.
        raw_root: directory holding ``<suite>/<task>_demo.hdf5``.
        repo_id: LeRobot repo id recorded in the dataset (local only; nothing is pushed).
        limit_per_task: first k demos per task (smoke tests); ``None`` = all 50.
        image_writer_threads: LeRobot's async PNG writer threads (4 measured 449 vs 296 frames/s at 0).
        log_every: print progress every n episodes.

    Returns:
        ``{"episodes", "frames", "wall_time_s", "root"}`` with ``root`` repo-relative.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out_root = Path(out_root)
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(f"{out_root} exists; pass --overwrite to replace it")
        if any(out_root.iterdir()) and not (out_root / "meta" / "info.json").is_file():
            raise RuntimeError(f"refusing to delete {out_root}: not a LeRobot dataset root")
        shutil.rmtree(out_root)
    out_root.parent.mkdir(parents=True, exist_ok=True)

    refs = list(iter_raw_episodes(raw_root, limit_per_task))
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        features=FEATURES,
        root=out_root,
        robot_type=ROBOT_TYPE,
        use_videos=False,
        image_writer_threads=image_writer_threads,
    )
    t0 = time.time()
    n_frames = 0
    for i, ref in enumerate(refs):
        ep = load_raw_episode(REPO_ROOT / ref.source_file, ref.source_demo)
        for t in range(ep["num_samples"]):
            ds.add_frame(
                {
                    "observation.images.image": ep["image"][t],
                    "observation.images.image2": ep["image2"][t],
                    "observation.state": ep["state"][t],
                    "action": ep["action"][t],
                    "task": ep["task"],
                }
            )
        ds.save_episode()
        n_frames += ep["num_samples"]
        if (i + 1) % log_every == 0 or i + 1 == len(refs):
            dt = time.time() - t0
            print(
                f"[convert] {i + 1}/{len(refs)} episodes, {n_frames} frames, {dt:.0f}s elapsed, "
                f"{n_frames / max(dt, 1e-9):.0f} frames/s",
                flush=True,
            )
    ds.finalize()
    # LeRobot's writer leaves its (now empty) PNG staging tree behind; remove it only if file-free.
    staging = out_root / "images"
    if staging.is_dir() and not any(p.is_file() for p in staging.rglob("*")):
        shutil.rmtree(staging)
    wall = time.time() - t0
    return {"episodes": len(refs), "frames": n_frames, "wall_time_s": round(wall, 1), "root": _rel(out_root)}


def main(argv: list[str] | None = None) -> int:
    """CLI entry (usage in the module docstring); prints the ``convert`` summary as JSON."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW_ROOT)
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--limit-per-task", type=int, default=None, help="smoke tests only")
    p.add_argument("--threads", type=int, default=4, help="LeRobot async PNG writer threads")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    summary = convert(a.out, a.raw, a.repo_id, a.limit_per_task, a.threads, a.overwrite)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
