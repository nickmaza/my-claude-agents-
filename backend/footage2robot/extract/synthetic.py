"""Procedural "person doing a pick-and-place" motion.

Used for tests, demos and for exercising the downstream pipeline without a
GPU or real footage. Produces the same arrays a real extractor would, in the
same camera-aligned frame (+x image-right, +y down, +z away from camera), for a
person facing the camera.
"""

from __future__ import annotations

import os

import numpy as np

from ..schema import B, HumanMotion
from .base import Extractor, ProgressFn


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def synth_hand(wrist: np.ndarray, forward: np.ndarray, normal: np.ndarray, grip: float,
               mirror: float) -> np.ndarray:
    """21 hand landmarks. ``grip`` 0=open .. 1=closed; ``mirror`` +1/-1 per side."""
    d = _unit(forward)
    n = _unit(normal - np.dot(normal, d) * d)
    lat = np.cross(n, d) * mirror
    pts = np.zeros((21, 3))
    pts[0] = wrist
    theta = 0.15 + 1.25 * grip
    lengths = [0.045, 0.028, 0.022]
    # index, middle, ring, pinky
    for f, offset in enumerate([0.025, 0.005, -0.014, -0.032]):
        base = wrist + 0.09 * d + offset * lat
        pts[5 + 4 * f] = base
        p = base
        for k in range(3):
            ang = theta * (k + 1)
            p = p + lengths[k] * (np.cos(ang) * d + np.sin(ang) * n)
            pts[6 + 4 * f + k] = p
    # thumb: sweeps from lateral towards index fingertip as grip closes
    base = wrist + 0.025 * d + 0.025 * lat
    pts[1] = base
    tdir = _unit((0.6 - 0.3 * grip) * d + (0.8 - 0.6 * grip) * lat + 0.5 * grip * n)
    p = base
    for k, length in enumerate([0.035, 0.03, 0.025]):
        p = p + length * tdir
        pts[2 + k] = p
    return pts


class SyntheticExtractor(Extractor):
    name = "synthetic"

    def __init__(self, duration_s: float = 4.0, fps: float = 30.0, seed: int = 0):
        self.duration_s = duration_s
        self.fps = fps
        self.seed = seed

    def extract(self, video_path: str = "", *, target_fps: float | None = None,
                start_s: float = 0.0, end_s: float | None = None,
                progress: ProgressFn | None = None) -> HumanMotion:
        fps = float(target_fps or self.fps)
        duration = self.duration_s
        width, height = 640, 480
        if video_path and os.path.exists(video_path):
            try:
                from ..video import probe
                info = probe(video_path)
                duration, width, height = info.duration or duration, info.width, info.height
                if not target_fps:
                    fps = info.fps
            except ValueError:
                pass
        if end_s is not None:
            duration = min(duration, end_s)
        duration = max(duration - start_s, 1.0 / fps)
        n = max(2, int(round(duration * fps)))
        rng = np.random.default_rng(self.seed)
        m = HumanMotion.empty(n, fps, width, height, source=video_path or "synthetic")
        t = np.arange(n) / fps
        phase = 2 * np.pi * t / max(duration, 1e-3)

        fwd = np.array([0.0, 0.0, -1.0])      # person's forward (toward camera)
        up = np.array([0.0, -1.0, 0.0])
        left = np.array([1.0, 0.0, 0.0])       # person's left = image right

        for i in range(n):
            body = np.zeros((33, 3))
            torso_lean = 0.08 * np.sin(phase[i])
            shoulder_c = 0.5 * up + torso_lean * fwd
            body[B["left_shoulder"]] = shoulder_c + 0.18 * left
            body[B["right_shoulder"]] = shoulder_c - 0.18 * left
            body[B["left_hip"]] = 0.1 * left
            body[B["right_hip"]] = -0.1 * left
            head = shoulder_c + 0.2 * up + 0.05 * fwd
            for k in range(11):
                offset = 0.04 * left if k in (1, 2, 3, 7, 9) else -0.04 * left if k in (4, 5, 6, 8, 10) else 0.0
                body[k] = head + offset + (0.09 * fwd if k == 0 else 0.0)

            # right arm: reach forward & down to "pick", then lift
            reach = 0.5 - 0.5 * np.cos(phase[i])
            r_sh = body[B["right_shoulder"]]
            r_wr = r_sh + (0.25 + 0.3 * reach) * fwd - (0.45 - 0.25 * reach) * up - 0.05 * left
            r_el = 0.5 * (r_sh + r_wr) - 0.12 * up * (1 - reach) - 0.06 * left
            # left arm: gentle sway, mostly at rest
            l_sh = body[B["left_shoulder"]]
            l_wr = l_sh - 0.5 * up + 0.12 * fwd + 0.05 * np.sin(2 * phase[i]) * fwd + 0.05 * left
            l_el = l_sh - 0.27 * up + 0.04 * left
            for side, (sh, el, wr) in {"left": (l_sh, l_el, l_wr), "right": (r_sh, r_el, r_wr)}.items():
                body[B[f"{side}_elbow"]] = el
                body[B[f"{side}_wrist"]] = wr
                fd = _unit(wr - el)
                body[B[f"{side}_index"]] = wr + 0.12 * fd
                body[B[f"{side}_pinky"]] = wr + 0.1 * fd + (-0.02 if side == "right" else 0.02) * left
                body[B[f"{side}_thumb"]] = wr + 0.06 * fd + 0.03 * fwd
            # legs: slight knee bend oscillation
            bend = 0.05 * (1 - np.cos(phase[i]))
            for side, sgn in (("left", 1), ("right", -1)):
                hip = body[B[f"{side}_hip"]]
                knee = hip - 0.42 * up + bend * fwd
                ankle = knee - 0.42 * up
                body[B[f"{side}_knee"]] = knee
                body[B[f"{side}_ankle"]] = ankle
                body[B[f"{side}_heel"]] = ankle - 0.06 * up - 0.04 * fwd
                body[B[f"{side}_foot_index"]] = ankle - 0.08 * up + 0.14 * fwd + 0.02 * sgn * left
            body += 0.002 * rng.standard_normal(body.shape)   # tracking noise
            m.body_3d[i] = body
            m.body_conf[i] = 0.95

            # gripper closes while the arm is extended (the "pick"), opens to release
            grip_r = float(np.clip(1.6 * np.sin(phase[i] / 2) ** 4 - 0.1, 0, 1))
            for side, wr, el, grip, mirror in ((0, l_wr, l_el, 0.2, -1.0), (1, r_wr, r_el, grip_r, 1.0)):
                pts = synth_hand(wr, wr - el, fwd + 0.3 * up, grip, mirror)
                m.hands_3d[i, side] = pts - pts.mean(axis=0)   # hand-centred like MediaPipe
                m.hands_2d[i, side] = self._project(pts)
                m.hands_conf[i, side] = 0.97
            m.body_2d[i] = self._project(body)
            if progress and i % 20 == 0:
                progress((i + 1) / n, f"Synthesising frame {i + 1}/{n}")

        m.meta.update({"extractor": self.name, "source_fps": fps, "stride": 1, "start_s": start_s})
        if progress:
            progress(1.0, "Synthetic motion ready")
        return m

    @staticmethod
    def _project(p: np.ndarray) -> np.ndarray:
        z = p[:, 2] + 2.5
        f = 1.1
        return np.stack([0.5 + f * p[:, 0] / z, 0.55 + f * p[:, 1] / z], axis=1).astype(np.float32)
