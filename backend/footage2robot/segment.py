"""Split a long clip into demonstration episodes.

A demonstration is a stretch of purposeful hand motion. Episodes are separated
where both hands are still (or out of view) for a while *and* nothing is being
held — a pause while gripping an object is part of the task, not a boundary.
"""

from __future__ import annotations

import numpy as np

from .process.filters import smooth
from .retarget.geometry import body_frames, to_frame
from .retarget.hands import ee_position_cam, gripper_closedness
from .schema import HumanMotion


def hand_speed(motion: HumanMotion) -> np.ndarray:
    """Fastest hand's speed in the body frame, m/s, (T,). Untracked frames count as 0."""
    fr = body_frames(motion.body_3d.astype(np.float64))
    speeds = []
    for side in (0, 1):
        p = to_frame(ee_position_cam(motion, side).astype(np.float64), fr["torso_R"], fr["torso_origin"])
        v = np.linalg.norm(np.gradient(p, axis=0), axis=-1) * motion.fps
        speeds.append(np.where(np.isfinite(v), v, 0.0))
    speed = np.max(speeds, axis=0)
    return smooth(speed[:, None], 0.1 * motion.fps)[:, 0]


def auto_split(motion: HumanMotion, *, speed_threshold: float = 0.06, min_pause_s: float = 0.7,
               min_episode_s: float = 0.8, pad_s: float = 0.25) -> list[list[int]]:
    """Episode boundaries as ``[start, end)`` frame pairs. Falls back to the whole clip."""
    n = motion.num_frames
    whole = [[0, n]]
    if n < 2:
        return whole
    fps = motion.fps
    speed = hand_speed(motion)
    holding = np.zeros(n, bool)
    for side in (0, 1):
        g = gripper_closedness(motion, side)
        holding |= np.where(np.isfinite(g), g, 0.0) > 0.6
    idle = (speed < speed_threshold) & ~holding

    # idle runs long enough to count as a pause
    pause = np.zeros(n, bool)
    start = None
    for t in range(n + 1):
        if t < n and idle[t]:
            start = t if start is None else start
        elif start is not None:
            if t - start >= min_pause_s * fps:
                pause[start:t] = True
            start = None

    pad = int(round(pad_s * fps))
    episodes: list[list[int]] = []
    start = None
    for t in range(n + 1):
        if t < n and not pause[t]:
            start = t if start is None else start
        elif start is not None:
            a, b = max(0, start - pad), min(n, t + pad)
            if b - a >= min_episode_s * fps:
                if episodes and a <= episodes[-1][1]:
                    episodes[-1][1] = b
                else:
                    episodes.append([a, b])
            start = None
    return episodes or whole


def validate_segments(segments: list, num_frames: int, min_length: int = 5) -> list[list[int]]:
    """Clamp, sort and sanity-check user-edited segments."""
    out = []
    for seg in segments:
        if not isinstance(seg, (list, tuple)) or len(seg) != 2:
            raise ValueError("Each segment must be [start, end]")
        a, b = int(seg[0]), int(seg[1])
        a, b = max(0, a), min(num_frames, b)
        if b - a < min_length:
            raise ValueError(f"Segment [{seg[0]}, {seg[1]}] is shorter than {min_length} frames")
        out.append([a, b])
    out.sort()
    for prev, cur in zip(out, out[1:]):
        if cur[0] < prev[1]:
            raise ValueError("Segments must not overlap")
    return out
