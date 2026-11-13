"""Download and cache MediaPipe model bundles."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

_BASE = "https://storage.googleapis.com/mediapipe-models"
MODEL_URLS = {
    "pose_lite": f"{_BASE}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_full": f"{_BASE}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_heavy": f"{_BASE}/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "hand": f"{_BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}


def cache_dir() -> Path:
    d = Path(os.environ.get("FOOTAGE2ROBOT_MODEL_DIR", Path.home() / ".cache" / "footage2robot" / "models"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_model(name: str) -> Path:
    if name not in MODEL_URLS:
        raise KeyError(f"Unknown model '{name}'. Options: {sorted(MODEL_URLS)}")
    path = cache_dir() / f"{name}.task"
    if not path.exists():
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URLS[name], tmp)
        tmp.replace(path)
    return path
