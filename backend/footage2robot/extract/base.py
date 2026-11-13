"""Extractor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from ..schema import HumanMotion

ProgressFn = Callable[[float, str], None]


class Extractor(ABC):
    name: str = "base"

    @abstractmethod
    def extract(self, video_path: str, *, target_fps: float | None = None,
                start_s: float = 0.0, end_s: float | None = None,
                progress: ProgressFn | None = None) -> HumanMotion:
        """Recover per-frame human motion from a video file."""
