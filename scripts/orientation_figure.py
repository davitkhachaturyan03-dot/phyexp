"""Orientation figure for AC1.4: raw LIBERO frame vs our converted frame vs a hub reference frame.

Grounding: lerobot/processor/env_processor.py (lerobot 0.6.1) "Rotates images by 180 degrees by
flipping both height and width dimensions ... This accounts for the HuggingFaceVLA/libero camera
orientation convention." Our converter stores raw frames flipped in H and W so that training data and
the LeRobot LIBERO eval env agree. Reference frames: HuggingFaceVLA/libero @ 86958911 (datasets-server
JPEG previews of frame 0 of hub episodes 1272 and 6), saved under artifacts/reference/.

The hub frame is a different demo re-rendered at 256x256, so pixels do not match exactly; the figure
shows orientation. As supporting numbers, it prints the mean absolute difference between the hub frame
and our frame (upsampled 2x, nearest) and between the hub frame and the unrotated raw frame.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import h5py  # noqa: E402

from vlapoc.libero_convert import DEFAULT_OUT, DEFAULT_RAW_ROOT, raw_task_path  # noqa: E402

FEATURE_DIR = REPO_ROOT / "tracker" / "features" / "2026-09-28-data-lineage-layer"
REF = FEATURE_DIR / "artifacts" / "reference"
CASES = [
    # (label, suite, task_name, our episode_index, hub reference prefix)
    ("libero_spatial task 0", "libero_spatial",
     "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate", 0, "spatial_t0"),
    ("libero_10 task 2 (KITCHEN_SCENE3)", "libero_10",
     "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it", 500 + 2 * 50, "l10_scene3"),
]
CAMS = [("image", "agentview_rgb"), ("image2", "eye_in_hand_rgb")]


def stored_frame(root: Path, episode_index: int, cam: str, frame: int = 0) -> np.ndarray:
    """Decode one stored PNG frame straight from the LeRobot parquet (no loader transforms)."""
    import pandas as pd

    ep = pd.concat(
        pq.read_table(f, columns=["episode_index", "data/chunk_index", "data/file_index"]).to_pandas()
        for f in sorted((root / "meta" / "episodes").rglob("*.parquet"))
    )
    row = ep[ep["episode_index"] == episode_index].iloc[0]
    path = root / "data" / f"chunk-{int(row['data/chunk_index']):03d}" / f"file-{int(row['data/file_index']):03d}.parquet"
    t = pq.read_table(path, columns=["episode_index", "frame_index", f"observation.images.{cam}"]).to_pandas()
    sel = t[(t["episode_index"] == episode_index) & (t["frame_index"] == frame)].iloc[0]
    import io

    return np.asarray(Image.open(io.BytesIO(sel[f"observation.images.{cam}"]["bytes"])).convert("RGB"))


def up2(a: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(a).resize((256, 256), Image.NEAREST))


def main() -> int:
    root = DEFAULT_OUT
    tiles, stats = [], []
    for label, suite, task, ep_idx, ref in CASES:
        with h5py.File(raw_task_path(DEFAULT_RAW_ROOT, suite, task), "r") as h5:
            raw = {c: h5["data"]["demo_0"]["obs"][k][0] for c, k in CAMS}
        for cam, _ in CAMS:
            ours = stored_frame(root, ep_idx, cam)
            hub = np.asarray(Image.open(REF / f"HuggingFaceVLA_libero_86958911_{ref}_{cam}.jpg").convert("RGB"))
            mad_ours = float(np.abs(up2(ours).astype(int) - hub.astype(int)).mean())
            mad_raw = float(np.abs(up2(raw[cam]).astype(int) - hub.astype(int)).mean())
            stats.append({"case": label, "camera": cam, "mad_hub_vs_ours_rot180": round(mad_ours, 2),
                          "mad_hub_vs_raw_unrotated": round(mad_raw, 2),
                          "ours_equals_raw_rot180": bool(np.array_equal(ours, raw[cam][::-1, ::-1]))})
            tiles.append((f"{label} / {cam}", [("raw HDF5 (as stored)", up2(raw[cam])),
                                              ("ours (LeRobot, rot 180)", up2(ours)),
                                              ("HuggingFaceVLA/libero", hub)]))
    pad, head, lab = 8, 22, 18
    W = 3 * 256 + 4 * pad
    H = len(tiles) * (head + lab + 256 + pad) + pad
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    y = pad
    for title, row in tiles:
        d.text((pad, y + 4), title, fill="black")
        y += head
        for i, (cap, img) in enumerate(row):
            x = pad + i * (256 + pad)
            d.text((x, y + 2), cap, fill=(60, 60, 60))
            canvas.paste(Image.fromarray(img), (x, y + lab))
        y += lab + 256 + pad
    out = FEATURE_DIR / "artifacts" / "s1_orientation.png"
    canvas.save(out)
    (FEATURE_DIR / "artifacts" / "s1_orientation_stats.json").write_text(json.dumps(stats, indent=2))
    for s in stats:
        print(s)
    print(f"figure -> {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
