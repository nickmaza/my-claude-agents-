"""LeRobot dataset v2.1 export — the layout used by lerobot releases before v3.0.

Layout::

    meta/info.json  meta/tasks.jsonl  meta/episodes.jsonl  meta/episodes_stats.jsonl
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.<camera>/episode_000000.mp4
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..dataset import RobotDataset, feature_stats

CHUNK_SIZE = 1000
MAX_VIDEO_SIDE = 640


def _video_size(w: int, h: int) -> tuple[int, int]:
    s = min(1.0, MAX_VIDEO_SIDE / max(w, h))
    return max(2, int(w * s) // 2 * 2), max(2, int(h * s) // 2 * 2)


def encode_video(frames: Iterable[np.ndarray], path: Path, fps: float, keep_samples: bool = False) -> dict:
    """Encode RGB frames to H.264/yuv420p MP4. Returns video info + image stats.

    With ``keep_samples`` the result also carries ``samples``: a list with one
    (N, 3) array of sampled pixels in [0, 1] per source frame, for statistics.
    """
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    proc = writer = None
    size = None
    codec = "h264"
    sample: list[np.ndarray] = []
    n = 0
    for frame in frames:
        if size is None:
            size = _video_size(frame.shape[1], frame.shape[0])
            if shutil.which("ffmpeg"):
                proc = subprocess.Popen(
                    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                     "-s", f"{size[0]}x{size[1]}", "-r", f"{fps:.6f}", "-i", "-", "-an",
                     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "2", "-crf", "23", str(path)],
                    stdin=subprocess.PIPE)
            else:   # portable fallback; players/loaders may prefer H.264
                codec = "mp4v"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        if proc:
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if keep_samples:      # every frame, sparsely, so per-episode stats stay cheap in memory
            sample.append(frame[::16, ::16].reshape(-1, 3).astype(np.float32) / 255.0)
        elif n % 10 == 0:
            sample.append(frame[::4, ::4].reshape(-1, 3).astype(np.float32) / 255.0)
        n += 1
    if proc:
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed encoding {path}")
    elif writer:
        writer.release()
    px = np.concatenate(sample)
    stats = {k: np.asarray(v).reshape(3, 1, 1).tolist() for k, v in
             {"min": px.min(0), "max": px.max(0), "mean": px.mean(0), "std": px.std(0)}.items()}
    stats["count"] = [len(sample)]
    out = {"width": size[0], "height": size[1], "codec": codec, "frames": n, "stats": stats}
    if keep_samples:
        out["samples"] = sample
    return out


def _fixed_list(arr: np.ndarray) -> pa.Array:
    return pa.FixedSizeListArray.from_arrays(pa.array(arr.reshape(-1), type=pa.float32()), arr.shape[1])


def export_lerobot_v21(ds: RobotDataset, out_dir: Path) -> Path:
    meta = out_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    cam_key = f"observation.images.{ds.camera}"
    fps = int(round(ds.fps)) if abs(ds.fps - round(ds.fps)) < 1e-3 else ds.fps
    tasks = sorted({e.task for e in ds.episodes})
    task_index = {t: i for i, t in enumerate(tasks)}

    episodes_meta, stats_meta = [], []
    video_info = None
    global_index = 0
    for i, ep in enumerate(ds.episodes):
        n = len(ep)
        chunk = i // CHUNK_SIZE
        frame_index = np.arange(n, dtype=np.int64)
        timestamp = (frame_index / ds.fps).astype(np.float32)
        index = global_index + frame_index
        table = pa.table({
            "observation.state": _fixed_list(ep.state),
            **({"observation.environment_state": _fixed_list(ep.env_state)} if ds.has_env_state else {}),
            "action": _fixed_list(ep.action),
            "timestamp": pa.array(timestamp, type=pa.float32()),
            "frame_index": pa.array(frame_index),
            "episode_index": pa.array(np.full(n, i, np.int64)),
            "index": pa.array(index),
            "task_index": pa.array(np.full(n, task_index[ep.task], np.int64)),
        })
        data_path = out_dir / f"data/chunk-{chunk:03d}/episode_{i:06d}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, data_path)

        stats = {
            "observation.state": feature_stats(ep.state), "action": feature_stats(ep.action),
            "timestamp": feature_stats(timestamp), "frame_index": feature_stats(frame_index),
            "episode_index": feature_stats(np.full(n, i)), "index": feature_stats(index),
            "task_index": feature_stats(np.full(n, task_index[ep.task])),
        }
        if ds.has_env_state:
            stats["observation.environment_state"] = feature_stats(ep.env_state)
        if ds.has_images:
            video_info = encode_video(
                ep.iter_images(), out_dir / f"videos/chunk-{chunk:03d}/{cam_key}/episode_{i:06d}.mp4", ds.fps)
            stats[cam_key] = video_info["stats"]
        episodes_meta.append({"episode_index": i, "tasks": [ep.task], "length": n})
        stats_meta.append({"episode_index": i, "stats": stats})
        global_index += n

    def scalar(dtype):
        return {"dtype": dtype, "shape": [1], "names": None}

    features = {
        "observation.state": {"dtype": "float32", "shape": [ds.state_dim], "names": ds.state_names},
        "action": {"dtype": "float32", "shape": [ds.state_dim], "names": ds.action_names},
        "timestamp": scalar("float32"), "frame_index": scalar("int64"),
        "episode_index": scalar("int64"), "index": scalar("int64"), "task_index": scalar("int64"),
    }
    if ds.has_env_state:
        features["observation.environment_state"] = {
            "dtype": "float32", "shape": [len(ds.env_state_names)], "names": ds.env_state_names}
    if video_info:
        features[cam_key] = {
            "dtype": "video", "shape": [video_info["height"], video_info["width"], 3],
            "names": ["height", "width", "channels"],
            "info": {"video.fps": fps, "video.height": video_info["height"],
                     "video.width": video_info["width"], "video.channels": 3,
                     "video.codec": video_info["codec"], "video.pix_fmt": "yuv420p",
                     "video.is_depth_map": False, "has_audio": False},
        }
    n_ep = len(ds.episodes)
    info = {
        "codebase_version": "v2.1",
        "robot_type": f"human_video_{ds.embodiment}",
        "total_episodes": n_ep, "total_frames": ds.total_frames, "total_tasks": len(tasks),
        "total_videos": n_ep if video_info else 0,
        "total_chunks": (n_ep - 1) // CHUNK_SIZE + 1, "chunks_size": CHUNK_SIZE,
        "fps": fps, "splits": {"train": f"0:{n_ep}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        if video_info else None,
        "features": features,
        "footage2robot": {"action_mode": ds.action_mode, **ds.info},
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2))

    def jsonl(name, rows):
        (meta / name).write_text("".join(json.dumps(r) + "\n" for r in rows))

    jsonl("tasks.jsonl", [{"task_index": i, "task": t} for t, i in task_index.items()])
    jsonl("episodes.jsonl", episodes_meta)
    jsonl("episodes_stats.jsonl", stats_meta)
    return out_dir
