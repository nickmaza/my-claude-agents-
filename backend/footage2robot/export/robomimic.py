"""robomimic-style HDF5 export.

Layout::

    data/                      attrs: total, env_args (JSON)
      demo_0/                  attrs: num_samples, task
        obs/state              (T, D) float32
        obs/object             (T, 5) float32       (if an object was tracked)
        obs/<camera>_image     (T, H, W, 3) uint8   (if video available)
        next_obs/state
        actions                (T, D) float32
        rewards, dones         (T,)  sparse success signal on the final step
    mask/train, mask/valid     demo keys for a 90/10 split
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from ..dataset import RobotDataset


def export_robomimic(ds: RobotDataset, out_dir: Path) -> Path:
    path = out_dir / "demos.hdf5"
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        total = 0
        for i, ep in enumerate(ds.episodes):
            g = data.create_group(f"demo_{i}")
            n = len(ep)
            g.attrs["num_samples"] = n
            g.attrs["task"] = ep.task
            g.create_dataset("obs/state", data=ep.state)
            g.create_dataset("next_obs/state", data=np.concatenate([ep.state[1:], ep.state[-1:]]))
            if ds.has_env_state:
                g.create_dataset("obs/object", data=ep.env_state)
                g.create_dataset("next_obs/object", data=np.concatenate([ep.env_state[1:], ep.env_state[-1:]]))
            if ds.has_images:
                g.create_dataset(f"obs/{ds.camera}_image", data=ep.images(ds.image_size),
                                 compression="gzip", compression_opts=4, chunks=True)
            g.create_dataset("actions", data=ep.action)
            g.create_dataset("states", data=ep.state)
            done = np.zeros(n, np.int64)
            done[-1] = 1
            g.create_dataset("rewards", data=done.astype(np.float32))
            g.create_dataset("dones", data=done)
            total += n
        data.attrs["total"] = total
        data.attrs["env_args"] = json.dumps({
            "env_name": f"footage2robot/{ds.embodiment}", "type": "human_video", "env_kwargs": {
                "fps": ds.fps, "state_names": ds.state_names, "action_names": ds.action_names,
                "action_mode": ds.action_mode, **ds.info}})
        keys = [f"demo_{i}" for i in range(len(ds.episodes))]
        n_valid = max(1, len(keys) // 10) if len(keys) > 1 else 0
        f.create_dataset("mask/train", data=np.array(keys[:len(keys) - n_valid], dtype="S"))
        f.create_dataset("mask/valid", data=np.array(keys[len(keys) - n_valid:], dtype="S"))
    return path
