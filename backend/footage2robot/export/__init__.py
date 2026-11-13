"""Dataset exporters. Each takes a RobotDataset and writes one on-disk format."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..dataset import RobotDataset

FORMATS = {
    "lerobot": "LeRobot v3.0 (Parquet + MP4) — current Hugging Face LeRobot: ACT, Diffusion Policy, pi0, SmolVLA",
    "lerobot_v21": "LeRobot v2.1 (Parquet + MP4) — older LeRobot releases and tools",
    "rlds": "RLDS / TFDS (TFRecord) — Open X-Embodiment, Octo, OpenVLA, RT-X",
    "robomimic": "robomimic HDF5 — robomimic, RoboSuite, MimicGen, LIBERO-style loaders",
    "zarr": "Diffusion Policy Zarr replay buffer — Diffusion Policy, UMI",
    "npz": "NumPy .npz per episode + JSON metadata — framework-agnostic",
    "json": "Plain JSON (states/actions, no images) — inspection and web tooling",
}


def get_exporter(fmt: str) -> Callable[[RobotDataset, Path], Path]:
    if fmt == "lerobot":
        from .lerobot_v3 import export_lerobot as fn
    elif fmt == "lerobot_v21":
        from .lerobot_v21 import export_lerobot_v21 as fn
    elif fmt == "rlds":
        from .rlds import export_rlds as fn
    elif fmt == "robomimic":
        from .robomimic import export_robomimic as fn
    elif fmt == "zarr":
        from .zarr_dp import export_zarr as fn
    elif fmt == "npz":
        from .simple import export_npz as fn
    elif fmt == "json":
        from .simple import export_json as fn
    else:
        raise KeyError(f"Unknown format '{fmt}'. Options: {sorted(FORMATS)}")
    return fn


def export(dataset: RobotDataset, fmt: str, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return get_exporter(fmt)(dataset, out_dir)


__all__ = ["FORMATS", "export", "get_exporter"]
