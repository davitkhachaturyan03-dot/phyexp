"""Verify the S1 conversion (raw LIBERO -> LeRobot v3) against its acceptance criteria.

Grounding: PoC §10 p.20 (LIBERO one or two suites + SmolVLA), §2.2 p.4 (LeRobot dataset format),
and the vla-build rule that no demo is filtered on an unverified premise (every demo is checked to
end with its only reward == 1). Prints evidence for AC1.1-AC1.6 and writes a JSON summary.

    .venv/bin/python scripts/verify_libero_convert.py [--root data/lerobot/libero_spatial_10] [--summary <abs path>]

The summary defaults to the feature's ``artifacts/s1_summary.json``. A ``--summary`` given as a
relative path is written, but the final print then fails (``relative_to`` on a relative path), so pass
an absolute one. The full-dataset load writes a ~7 GB HF ``datasets`` cache under
``HF_DATASETS_CACHE`` the first time (plan risk R1); later runs reuse it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vlapoc.libero_convert import (  # noqa: E402
    DEFAULT_OUT,
    DEFAULT_RAW_ROOT,
    DEFAULT_REPO_ID,
    FEATURES,
    iter_raw_episodes,
)

FEATURE_DIR = REPO_ROOT / "tracker" / "features" / "2026-09-28-data-lineage-layer"
REF_INFOS = {
    "lerobot/libero@a1aaacb7": FEATURE_DIR / "artifacts" / "reference" / "lerobot_libero_a1aaacb7_info.json",
    "HuggingFaceVLA/libero@86958911": FEATURE_DIR
    / "artifacts"
    / "reference"
    / "HuggingFaceVLA_libero_86958911_info.json",
}
SCHEMA_KEYS = ("observation.images.image", "observation.images.image2", "observation.state", "action")


def feature_diff(ours: dict, ref: dict) -> list[str]:
    """Differences in keys, dtype, shape and names of every feature (fps reported separately)."""
    out = []
    for k in sorted(set(ours) | set(ref)):
        if k not in ours or k not in ref:
            out.append(f"{k}: only in {'ours' if k in ours else 'reference'}")
            continue
        for field in ("dtype", "shape", "names"):
            if ours[k].get(field) != ref[k].get(field):
                out.append(f"{k}.{field}: ours={ours[k].get(field)} ref={ref[k].get(field)}")
    return out


def main(argv: list[str] | None = None) -> int:
    """Print AC1.1-AC1.6 evidence for the dataset at ``--root`` and write the JSON summary."""
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--summary", type=Path, default=FEATURE_DIR / "artifacts" / "s1_summary.json")
    a = p.parse_args(argv)
    root: Path = a.root
    summary: dict = {}

    # ---- AC1.1 info.json + load -------------------------------------------------------------
    info = json.loads((root / "meta" / "info.json").read_text())
    ac11 = {k: info[k] for k in ("codebase_version", "fps", "total_episodes", "total_frames", "total_tasks", "robot_type")}
    print("AC1.1 info.json:", ac11)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds_full = LeRobotDataset(DEFAULT_REPO_ID, root=root)
    print(f"AC1.1 LeRobotDataset loads: num_episodes={ds_full.num_episodes} num_frames={ds_full.num_frames}")
    ok11 = (
        info["codebase_version"] == "v3.0"
        and info["fps"] == 20
        and info["total_episodes"] == 1000
        and info["total_frames"] == 200340
        and info["total_tasks"] == 20
        and ds_full.num_frames == 200340
    )
    summary["AC1.1"] = {**ac11, "loads": True, "pass": ok11}

    # ---- AC1.2 feature diff vs the hub LIBERO schemas ---------------------------------------
    ours_feats = info["features"]
    summary["AC1.2"] = {}
    for name, path in REF_INFOS.items():
        ref = json.loads(path.read_text())
        diff = feature_diff(ours_feats, ref["features"])
        print(f"AC1.2 feature diff vs {name} (fps ours={info['fps']} ref={ref['fps']}):")
        for line in diff:
            print("   ", line)
        summary["AC1.2"][name] = {"diff": diff, "fps_ours": info["fps"], "fps_ref": ref["fps"]}
    sub = LeRobotDataset(DEFAULT_REPO_ID, root=root, episodes=[0, 999])
    item = sub[0]
    shapes = {k: list(v.shape) for k, v in item.items() if hasattr(v, "shape")}
    print(f"AC1.2 LeRobotDataset(episodes=[0, 999]): {len(sub)} frames; item[0] shapes {shapes}")
    summary["AC1.2"]["episodes_0_999_frames"] = len(sub)
    summary["AC1.2"]["item_shapes"] = shapes

    # ---- AC1.5 no filtering: counts, lengths vs raw, reward re-verification ----------------
    ep_meta = ds_full.meta.episodes
    lengths = list(ep_meta["length"])
    tasks = [t[0] for t in ep_meta["tasks"]]
    per_task = Counter(tasks)
    refs = list(iter_raw_episodes(DEFAULT_RAW_ROOT))
    assert len(refs) == len(lengths) == 1000, (len(refs), len(lengths))
    len_match = 0
    reward_ok = 0
    dones_ok = 0
    contiguous_demo_keys = True
    task_match = 0
    by_file: dict[str, list] = {}
    for i, r in enumerate(refs):
        by_file.setdefault(r.source_file, []).append((i, r))
    for src, items in by_file.items():
        with h5py.File(REPO_ROOT / src, "r") as h5:
            keys = sorted(h5["data"].keys(), key=lambda s: int(s.split("_")[1]))
            contiguous_demo_keys &= keys == [f"demo_{k}" for k in range(len(keys))]
            instr = json.loads(h5["data"].attrs["problem_info"])["language_instruction"]
            for i, r in items:
                g = h5["data"][r.source_demo]
                n = int(g.attrs["num_samples"])
                rw = g["rewards"][:]
                dn = g["dones"][:]
                len_match += int(lengths[i] == n == len(rw))
                reward_ok += int(rw.sum() == 1 and rw[-1] == 1)
                dones_ok += int(dn.sum() == 1 and dn[-1] == 1)
                task_match += int(tasks[i] == instr)
    print(
        f"AC1.5 per-task episode counts: {len(per_task)} tasks, counts={sorted(set(per_task.values()))}; "
        f"demo keys contiguous demo_0..demo_49 in every file: {contiguous_demo_keys}"
    )
    print(f"AC1.5 episode length == raw num_samples: {len_match}/1000; task string == raw instruction: {task_match}/1000")
    print(f"AC1.5 raw rewards: exactly one reward==1, at the last step: {reward_ok}/1000; dones likewise: {dones_ok}/1000")
    summary["AC1.5"] = {
        "n_tasks": len(per_task),
        "per_task_counts": sorted(set(per_task.values())),
        "length_match": len_match,
        "task_match": task_match,
        "reward_last_step_only": reward_ok,
        "dones_last_step_only": dones_ok,
        "pass": len(per_task) == 20 and set(per_task.values()) == {50} and len_match == reward_ok == 1000,
    }

    # ---- AC1.3 / AC1.4 every frame of 3 episodes per suite, through the LeRobot loader ------
    spot = [0, 250, 499, 500, 750, 999]
    res13 = {}
    for e in spot:
        r = refs[e]
        ds = LeRobotDataset(DEFAULT_REPO_ID, root=root, episodes=[e])
        n = len(ds)
        st = np.stack([ds[j]["observation.state"].numpy() for j in range(n)])
        ac = np.stack([ds[j]["action"].numpy() for j in range(n)])
        # images come back float32 CHW in [0,1]; x/255*255 rounds back exactly for uint8 values
        im = np.stack(
            [np.rint(ds[j]["observation.images.image"].numpy() * 255).astype(np.uint8).transpose(1, 2, 0) for j in range(n)]
        )
        im2 = np.stack(
            [np.rint(ds[j]["observation.images.image2"].numpy() * 255).astype(np.uint8).transpose(1, 2, 0) for j in range(n)]
        )
        with h5py.File(REPO_ROOT / r.source_file, "r") as h5:
            obs = h5["data"][r.source_demo]["obs"]
            raw_state = np.concatenate([obs["ee_pos"][:], obs["ee_ori"][:], obs["gripper_states"][:]], 1).astype(np.float32)
            raw_agent = obs["agentview_rgb"][:]
            raw_wrist = obs["eye_in_hand_rgb"][:]
            raw_act = h5["data"][r.source_demo]["actions"][:].astype(np.float32)
        res13[e] = {
            "episode": f"{r.suite}/{r.task_name}/{r.source_demo}",
            "frames": n,
            "state_equal": bool(np.array_equal(st, raw_state)),
            "action_equal": bool(np.array_equal(ac, raw_act)),
            "image_eq_raw_rot180": bool(np.array_equal(im, np.flip(raw_agent, axis=(1, 2)))),
            "image_eq_raw_unrotated": bool(np.array_equal(im, raw_agent)),
            "image2_eq_raw_rot180": bool(np.array_equal(im2, np.flip(raw_wrist, axis=(1, 2)))),
        }
        print(f"AC1.3/1.4 episode_index {e}: {res13[e]}")
    summary["AC1.3_AC1.4"] = res13

    # ---- AC1.6 storage ---------------------------------------------------------------------
    du = subprocess.run(["du", "-sk", str(root)], capture_output=True, text=True).stdout.split()[0]
    size_bytes = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    n_data = len(list((root / "data").rglob("*.parquet")))
    has_videos = (root / "videos").exists()
    print(
        f"AC1.6 on-disk: du={int(du) / 1024 / 1024:.2f} GiB, sum(file sizes)={size_bytes / 1e9:.3f} GB "
        f"({size_bytes / 200340:.0f} B/frame), data parquet files={n_data}, videos dir exists={has_videos}, "
        f"image dtypes={[ours_feats[k]['dtype'] for k in SCHEMA_KEYS[:2]]}"
    )
    summary["AC1.6"] = {
        "du_kib": int(du),
        "size_bytes": size_bytes,
        "bytes_per_frame": round(size_bytes / 200340),
        "data_parquet_files": n_data,
        "videos_dir": has_videos,
    }
    assert FEATURES["observation.images.image"]["dtype"] == "image"

    a.summary.parent.mkdir(parents=True, exist_ok=True)
    a.summary.write_text(json.dumps(summary, indent=2, default=str))
    print(f"summary -> {a.summary.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
