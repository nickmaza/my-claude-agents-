"""LeRobot dataset v3.0 export — the layout read by current ``lerobot`` releases.

Episodes are concatenated into shared files; per-episode offsets live in the
episodes table::

    meta/info.json  meta/stats.json  meta/tasks.parquet
    meta/episodes/chunk-000/file-000.parquet     one row per episode (+ its stats)
    data/chunk-000/file-000.parquet              all frames of all episodes
    videos/observation.images.<camera>/chunk-000/file-000.mp4   all episodes back to back

The structure mirrors what ``LeRobotDataset.create(...).save_episode()`` writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..dataset import QUANTILES, RobotDataset, feature_stats
from .lerobot_v21 import encode_video

DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"


def _vector(arr: np.ndarray) -> pa.Array:
    return pa.FixedSizeListArray.from_arrays(pa.array(arr.reshape(-1), type=pa.float32()), arr.shape[1])


def _image_stats(pixels: np.ndarray) -> dict:
    """Per-channel stats of sampled pixels (N, 3) in [0, 1], shaped (3, 1, 1) like lerobot's."""
    def shaped(v):
        return np.asarray(v, np.float64).reshape(3, 1, 1).tolist()
    out = {"min": shaped(pixels.min(0)), "max": shaped(pixels.max(0)), "mean": shaped(pixels.mean(0)),
           "std": shaped(pixels.std(0)), "count": [int(len(pixels))]}
    out.update({k: shaped(np.quantile(pixels, q, axis=0)) for k, q in QUANTILES.items()})
    return out


def export_lerobot(ds: RobotDataset, out_dir: Path) -> Path:
    import pandas as pd

    cam_key = f"observation.images.{ds.camera}"
    fps = int(round(ds.fps)) if abs(ds.fps - round(ds.fps)) < 1e-3 else ds.fps
    tasks = sorted({e.task for e in ds.episodes})
    task_index = {t: i for i, t in enumerate(tasks)}
    vectors = {"observation.state": lambda e: e.state, "action": lambda e: e.action}
    if ds.has_env_state:
        vectors["observation.environment_state"] = lambda e: e.env_state

    # --- frames table -------------------------------------------------------
    cols: dict[str, list] = {k: [] for k in [*vectors, "timestamp", "frame_index", "episode_index", "index", "task_index"]}
    start = 0
    spans = []
    for i, ep in enumerate(ds.episodes):
        n = len(ep)
        frame_index = np.arange(n, dtype=np.int64)
        for k, get in vectors.items():
            cols[k].append(get(ep).astype(np.float32))
        cols["timestamp"].append((frame_index / ds.fps).astype(np.float32))
        cols["frame_index"].append(frame_index)
        cols["episode_index"].append(np.full(n, i, np.int64))
        cols["index"].append(start + frame_index)
        cols["task_index"].append(np.full(n, task_index[ep.task], np.int64))
        spans.append((start, start + n))
        start += n
    data = {k: np.concatenate(v) for k, v in cols.items()}

    hf_features = {k: {"feature": {"dtype": "float32", "_type": "Value"}, "length": int(data[k].shape[1]), "_type": "List"}
                   for k in vectors}
    hf_features.update({k: {"dtype": "float32" if k == "timestamp" else "int64", "_type": "Value"}
                        for k in ("timestamp", "frame_index", "episode_index", "index", "task_index")})
    table = pa.table({k: _vector(data[k]) if k in vectors else pa.array(data[k]) for k in data})
    table = table.replace_schema_metadata({"huggingface": json.dumps({"info": {"features": hf_features}})})
    data_path = out_dir / DATA_PATH.format(chunk_index=0, file_index=0)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, data_path)

    # --- video: every episode back to back in one file -----------------------
    video = None
    if ds.has_images:
        def frames():
            for ep in ds.episodes:
                yield from ep.iter_images()
        video = encode_video(frames(), out_dir / VIDEO_PATH.format(video_key=cam_key, chunk_index=0, file_index=0),
                             ds.fps, keep_samples=True)
        if video["frames"] != ds.total_frames:
            raise RuntimeError(f"Encoded {video['frames']} video frames for {ds.total_frames} dataset frames")

    # --- episodes table -----------------------------------------------------
    rows = []
    for i, (ep, (a, b)) in enumerate(zip(ds.episodes, spans)):
        row = {"episode_index": i, "tasks": [ep.task], "length": b - a,
               "data/chunk_index": 0, "data/file_index": 0, "dataset_from_index": a, "dataset_to_index": b}
        if video:
            row.update({f"videos/{cam_key}/chunk_index": 0, f"videos/{cam_key}/file_index": 0,
                        f"videos/{cam_key}/from_timestamp": a / ds.fps, f"videos/{cam_key}/to_timestamp": b / ds.fps})
        stats = {k: feature_stats(data[k][a:b], quantiles=True) for k in data}
        if video:
            stats[cam_key] = _image_stats(np.concatenate(video["samples"][a:b]))
        for key, st in stats.items():
            for name, value in st.items():
                row[f"stats/{key}/{name}"] = value
        row.update({"meta/episodes/chunk_index": 0, "meta/episodes/file_index": 0})
        rows.append(row)
    ep_path = out_dir / EPISODES_PATH.format(chunk_index=0, file_index=0)
    ep_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), ep_path)

    # --- tasks, stats, info --------------------------------------------------
    meta = out_dir / "meta"
    pd.DataFrame({"task_index": list(range(len(tasks)))}, index=pd.Index(tasks, name="task")).to_parquet(meta / "tasks.parquet")
    stats = {k: feature_stats(data[k], quantiles=True) for k in data}
    if video:
        stats[cam_key] = _image_stats(np.concatenate(video["samples"]))
    (meta / "stats.json").write_text(json.dumps(stats, indent=2))

    def scalar(dtype):
        return {"dtype": dtype, "shape": [1], "names": None}

    features = {
        "observation.state": {"dtype": "float32", "shape": [ds.state_dim], "names": ds.state_names},
        "action": {"dtype": "float32", "shape": [ds.state_dim], "names": ds.action_names},
    }
    if ds.has_env_state:
        features["observation.environment_state"] = {
            "dtype": "float32", "shape": [len(ds.env_state_names)], "names": ds.env_state_names}
    if video:
        features[cam_key] = {
            "dtype": "video", "shape": [video["height"], video["width"], 3],
            "names": ["height", "width", "channels"],
            "info": {"video.height": video["height"], "video.width": video["width"], "video.codec": video["codec"],
                     "video.pix_fmt": "yuv420p", "video.fps": fps, "video.channels": 3, "has_audio": False,
                     "is_depth_map": False},
        }
    features.update({"timestamp": scalar("float32"), "frame_index": scalar("int64"),
                     "episode_index": scalar("int64"), "index": scalar("int64"), "task_index": scalar("int64")})
    info = {
        "codebase_version": "v3.0", "fps": fps, "features": features,
        "total_episodes": len(ds.episodes), "total_frames": ds.total_frames, "total_tasks": len(tasks),
        "chunks_size": 1000, "data_files_size_in_mb": 100, "video_files_size_in_mb": 200,
        "data_path": DATA_PATH, "video_path": VIDEO_PATH if video else None,
        "robot_type": f"human_video_{ds.embodiment}", "splits": {"train": f"0:{len(ds.episodes)}"},
        "footage2robot": {"action_mode": ds.action_mode, **ds.info},
    }
    (meta / "info.json").write_text(json.dumps(info, indent=4))
    return out_dir
