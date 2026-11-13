"""Core data structures shared across the pipeline.

``HumanMotion`` holds everything the extractor recovers from a clip: 3D body
landmarks, 3D hand landmarks (per side), their 2D image projections (for
overlays), and per-frame confidences. All arrays are time-major.

Conventions
-----------
* Body landmarks follow the 33-point MediaPipe/BlazePose topology.
* Hand landmarks follow the 21-point MediaPipe hand topology.
* ``side`` index 0 is the person's LEFT, 1 is the person's RIGHT (anatomical,
  not image-left/right).
* 3D coordinates are in metres in the extractor's camera-aligned frame:
  +x image-right, +y image-down, +z away from the camera. Retargeting
  converts these into a body-centric robot frame.
* Missing observations are NaN; confidences are 0 where nothing was seen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BODY_LANDMARKS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner",
    "right_eye", "right_eye_outer", "left_ear", "right_ear", "mouth_left",
    "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]
B = {name: i for i, name in enumerate(BODY_LANDMARKS)}

HAND_LANDMARKS = [
    "wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]
H = {name: i for i, name in enumerate(HAND_LANDMARKS)}

BODY_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
    (23, 24), (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), (24, 26),
    (26, 28), (28, 30), (30, 32), (28, 32), (15, 17), (15, 19), (15, 21),
    (17, 19), (16, 18), (16, 20), (16, 22), (18, 20), (0, 11), (0, 12),
]
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

SIDES = ("left", "right")


@dataclass
class HumanMotion:
    """Per-frame human motion recovered from one video clip."""

    fps: float
    width: int
    height: int
    body_3d: np.ndarray        # (T, 33, 3) metres, NaN when missing
    body_2d: np.ndarray        # (T, 33, 2) normalised image coords [0, 1]
    body_conf: np.ndarray      # (T, 33) visibility/confidence in [0, 1]
    hands_3d: np.ndarray       # (T, 2, 21, 3) metres, hand-centred, NaN when missing
    hands_2d: np.ndarray       # (T, 2, 21, 2) normalised image coords
    hands_conf: np.ndarray     # (T, 2) per-hand presence confidence
    source: str = ""
    meta: dict = field(default_factory=dict)
    # Optional metric world frame (set by marker calibration): rotation (3, 3) with
    # the world axes as columns, and the world origin per frame (T, 3), both in the
    # same hip-centred camera-aligned coordinates as body_3d.
    world_R: np.ndarray | None = None
    world_origin: np.ndarray | None = None

    @property
    def num_frames(self) -> int:
        return int(self.body_3d.shape[0])

    @property
    def timestamps(self) -> np.ndarray:
        return np.arange(self.num_frames, dtype=np.float64) / self.fps

    @classmethod
    def empty(cls, num_frames: int, fps: float, width: int, height: int, source: str = "") -> "HumanMotion":
        t = num_frames
        nan = np.nan
        return cls(
            fps=fps, width=width, height=height,
            body_3d=np.full((t, 33, 3), nan, np.float32),
            body_2d=np.full((t, 33, 2), nan, np.float32),
            body_conf=np.zeros((t, 33), np.float32),
            hands_3d=np.full((t, 2, 21, 3), nan, np.float32),
            hands_2d=np.full((t, 2, 21, 2), nan, np.float32),
            hands_conf=np.zeros((t, 2), np.float32),
            source=source,
        )

    def coverage(self) -> dict:
        """Fraction of frames where body / each hand was detected."""
        body = np.isfinite(self.body_3d[:, B["left_shoulder"], 0])
        return {
            "body": float(body.mean()) if self.num_frames else 0.0,
            "left_hand": float(np.isfinite(self.hands_3d[:, 0, 0, 0]).mean()) if self.num_frames else 0.0,
            "right_hand": float(np.isfinite(self.hands_3d[:, 1, 0, 0]).mean()) if self.num_frames else 0.0,
        }

    # -- persistence ---------------------------------------------------
    _ARRAYS = ("body_3d", "body_2d", "body_conf", "hands_3d", "hands_2d", "hands_conf")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        header = {"fps": self.fps, "width": self.width, "height": self.height,
                  "source": self.source, "meta": self.meta}
        arrays = {k: getattr(self, k) for k in self._ARRAYS}
        if self.world_R is not None and self.world_origin is not None:
            arrays.update(world_R=self.world_R, world_origin=self.world_origin)
        np.savez_compressed(path, header=np.array(json.dumps(header)), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "HumanMotion":
        with np.load(path, allow_pickle=False) as z:
            header = json.loads(str(z["header"]))
            arrays = {k: z[k] for k in cls._ARRAYS}
            if "world_R" in z:
                arrays.update(world_R=z["world_R"], world_origin=z["world_origin"])
        return cls(fps=header["fps"], width=header["width"], height=header["height"],
                   source=header.get("source", ""), meta=header.get("meta", {}), **arrays)

    def to_overlay_json(self) -> dict:
        """Compact 2D keypoints for drawing skeleton overlays in the UI."""

        def clean(a: np.ndarray) -> list:
            r = np.round(a.astype(np.float64), 4)
            return np.where(np.isfinite(r), r, None).tolist()

        return {
            "fps": self.fps, "width": self.width, "height": self.height,
            "body_edges": BODY_EDGES, "hand_edges": HAND_EDGES,
            "body": clean(self.body_2d),
            "body_conf": clean(self.body_conf),
            "hands": clean(self.hands_2d),
        }
