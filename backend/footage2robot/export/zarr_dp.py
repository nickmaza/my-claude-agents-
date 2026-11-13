"""Diffusion Policy / UMI style Zarr replay buffer.

Layout::

    data/state   (N, D)            data/action  (N, D)
    data/img     (N, H, W, 3) u8   (if video available)
    meta/episode_ends (E,) int64   cumulative end index of each episode
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from ..dataset import RobotDataset


def _open_group(path: Path):
    try:   # zarr >= 3: write the v2 on-disk format that Diffusion Policy/UMI read
        return zarr.open_group(str(path), mode="w", zarr_format=2)
    except TypeError:
        return zarr.open_group(str(path), mode="w")


def _put(group, name: str, arr: np.ndarray, chunks: tuple) -> None:
    if hasattr(group, "create_array"):
        a = group.create_array(name, shape=arr.shape, dtype=arr.dtype, chunks=chunks)
        a[...] = arr
    else:
        group.create_dataset(name, data=arr, chunks=chunks)


def export_zarr(ds: RobotDataset, out_dir: Path) -> Path:
    path = out_dir / "replay_buffer.zarr"
    root = _open_group(path)
    data, meta = root.create_group("data"), root.create_group("meta")
    state = np.concatenate([e.state for e in ds.episodes])
    action = np.concatenate([e.action for e in ds.episodes])
    ends = np.cumsum([len(e) for e in ds.episodes]).astype(np.int64)
    _put(data, "state", state, (min(len(state), 1024), state.shape[1]))
    _put(data, "action", action, (min(len(action), 1024), action.shape[1]))
    if ds.has_env_state:
        env = np.concatenate([e.env_state for e in ds.episodes])
        _put(data, "object", env, (min(len(env), 1024), env.shape[1]))
    if ds.has_images:
        img = np.concatenate([e.images(ds.image_size) for e in ds.episodes])
        _put(data, "img", img, (min(len(img), 32),) + img.shape[1:])
    _put(meta, "episode_ends", ends, (len(ends),))
    root.attrs.update({"fps": ds.fps, "embodiment": ds.embodiment, "state_names": ds.state_names,
                       "action_names": ds.action_names, "action_mode": ds.action_mode,
                       "object_names": ds.env_state_names if ds.has_env_state else [],
                       "tasks": [e.task for e in ds.episodes]})
    (out_dir / "dataset.json").write_text(json.dumps(ds.describe(), indent=2))
    return path
