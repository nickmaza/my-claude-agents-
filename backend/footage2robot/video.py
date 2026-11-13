"""Video reading helpers built on OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    num_frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps if self.fps else 0.0


def probe(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        return VideoInfo(
            path=str(path), fps=float(fps),
            num_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def frame_stride(source_fps: float, target_fps: float | None) -> int:
    """Keep every ``stride``-th frame so the output rate is <= target_fps."""
    if not target_fps or target_fps >= source_fps:
        return 1
    return max(1, int(round(source_fps / target_fps)))


def iter_frames(path: str | Path, stride: int = 1, start_s: float = 0.0,
                end_s: float | None = None) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(source_frame_index, timestamp_s, rgb_frame)`` tuples."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        idx = 0
        if start_s > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
            idx = int(round(start_s * fps))
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t = idx / fps
            if end_s is not None and t > end_s:
                break
            if (idx - int(round(start_s * fps))) % stride == 0:
                yield idx, t, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            idx += 1
    finally:
        cap.release()


def resize_frame(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Letterbox-free resize to (width, height)."""
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
