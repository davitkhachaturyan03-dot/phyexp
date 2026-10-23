"""Browser-playable video from LeRobot v3 episodes (dashboard S5): frames of one training episode or one recorded rollout trajectory become
H.264 mp4 bytes, cached on disk.

Where this sits: ``dashboard/`` shows "the most influential training episodes (with video)" and the rollout itself (PoC §2.7 p. 5; PhAIL's run
explorer shows "each rollout with synchronized video", PhAIL p. 6). Training episodes and eval trajectories are both LeRobot v3 datasets with ``image``
columns (PNG bytes inside the data parquet, 20 fps, no mp4 files; measured in the dashboard plan), so one function serves both.

Grounding
---------
- PoC §2.7 (p. 5), quoted: "the most influential training episodes (with video) and their source datasets"; "PhAIL's public run explorer, which shows each
  rollout with synchronized video, telemetry and event annotations, is a good design reference for the rollout side of this view."
- PyAV 15.1.0 (installed) with ``libx264`` writes the mp4; Streamlit's ``st.video`` plays H.264 ("Converting your video to H.264 will allow the video to
  be displayed", docs.streamlit.io st.video, v1.66).

Orientation (plan risk R7, checked by looking at a frame): ``libero_convert`` stores the agent-view frames rotated 180 degrees to match the hub layout, and the
stored frame already shows the scene upright (robot at the top, table in front), so frames are encoded as stored, not un-rotated.

Reads only what ``vlapoc.lineage`` and ``vlapoc.catalog`` expose for a dataset root; it never reads an answer key.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from vlapoc.libero_convert import REPO_ROOT

DEFAULT_CACHE = REPO_ROOT / "data" / "dashboard_cache" / "video"
CAMERAS = {"image": "observation.images.image", "image2": "observation.images.image2"}


def frames_of(root: Path, episode_index: int, camera: str = "image") -> np.ndarray:
    """``(T, H, W, 3)`` uint8 frames of one episode of the LeRobot v3 dataset at ``root`` (frame order), reading only that episode's rows
    (parquet predicate on ``episode_index``)."""
    import pyarrow.parquet as pq
    from PIL import Image

    from vlapoc.catalog import read_episodes_meta

    root = Path(root)
    meta = read_episodes_meta(root)
    row = meta[meta["episode_index"] == int(episode_index)]
    if row.empty:
        raise ValueError(f"{root}: no episode {episode_index}")
    row = row.iloc[0]
    path = root / "data" / f"chunk-{int(row['data/chunk_index']):03d}" / f"file-{int(row['data/file_index']):03d}.parquet"
    col = CAMERAS[camera]
    t = pq.read_table(path, columns=["frame_index", col], filters=[("episode_index", "==", int(episode_index))]).to_pandas().sort_values("frame_index")
    if len(t) != int(row["length"]):
        raise ValueError(f"{root} episode {episode_index}: {len(t)} rows, meta says {int(row['length'])}")
    return np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in t[col]])


def fps_of(root: Path) -> int:
    return int(json.loads((Path(root) / "meta" / "info.json").read_text())["fps"])


def encode_mp4(frames: np.ndarray, fps: int = 20) -> bytes:
    """H.264 mp4 bytes of ``frames`` (``libx264``, yuv420p, crf 28, faststart; width and height cropped to even)."""
    import av

    h, w = frames.shape[1] // 2 * 2, frames.shape[2] // 2 * 2
    buf = io.BytesIO()
    c = av.open(buf, "w", format="mp4")
    st = c.add_stream("libx264", rate=fps)
    st.width, st.height, st.pix_fmt = w, h, "yuv420p"
    st.options = {"crf": "28", "preset": "veryfast", "movflags": "faststart"}
    for f in frames:
        for p in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f[:h, :w]), format="rgb24")):
            c.mux(p)
    for p in st.encode():
        c.mux(p)
    c.close()
    return buf.getvalue()


def decode_frame_count(mp4: bytes) -> int:
    """Number of video frames in mp4 bytes (decoded back with PyAV)."""
    import av

    with av.open(io.BytesIO(mp4)) as c:
        return sum(1 for _ in c.decode(video=0))


def episode_video(root: Path, episode_index: int, key: str, camera: str = "image", cache_dir: Path | None = DEFAULT_CACHE) -> bytes:
    """mp4 bytes of one episode, cached as ``<cache_dir>/<key>-<camera>.mp4`` (``key``: the episode's ``content_hash`` for training data, a
    rollout/trajectory identity for eval trajectories; the cache returns the same bytes on the second call). ``cache_dir=None`` disables caching."""
    path = None if cache_dir is None else Path(cache_dir) / f"{key}-{camera}.mp4"
    if path is not None and path.is_file():
        return path.read_bytes()
    data = encode_mp4(frames_of(root, episode_index, camera), fps_of(root))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data
