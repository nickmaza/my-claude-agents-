"""Framework-agnostic exports: NPZ per episode, and plain JSON."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..dataset import RobotDataset


def export_npz(ds: RobotDataset, out_dir: Path) -> Path:
    for i, ep in enumerate(ds.episodes):
        arrays = {
            "state": ep.state, "action": ep.action,
            "timestamp": (np.arange(len(ep)) / ds.fps).astype(np.float32),
            "tracked": ep.tracked if ep.tracked is not None else np.ones(len(ep), bool),
        }
        if ds.has_images:
            arrays["images"] = ep.images(ds.image_size)
        np.savez_compressed(out_dir / f"episode_{i:06d}.npz", **arrays)
    meta = ds.describe()
    meta["episodes"] = [{"file": f"episode_{i:06d}.npz", "length": len(ep), "task": ep.task, **ep.meta}
                        for i, ep in enumerate(ds.episodes)]
    (out_dir / "dataset.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def export_json(ds: RobotDataset, out_dir: Path) -> Path:
    doc = ds.describe()
    doc["episodes"] = [{
        "task": ep.task, "length": len(ep), "meta": ep.meta,
        "timestamp": np.round(np.arange(len(ep)) / ds.fps, 5).tolist(),
        "state": np.round(ep.state.astype(np.float64), 6).tolist(),
        "action": np.round(ep.action.astype(np.float64), 6).tolist(),
    } for ep in ds.episodes]
    (out_dir / "dataset.json").write_text(json.dumps(doc))
    return out_dir
