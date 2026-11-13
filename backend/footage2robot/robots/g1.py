"""Unitree G1: direct joint-angle mapping from human limb angles.

The G1's shoulders and hips are pitch(y) -> roll(x) -> yaw(z) chains followed by
a y-axis hinge, in a frame with x forward, y left, z up. That is exactly the
decomposition used by ``retarget.humanoid``, so arm and leg angles carry over
with only offsets (the G1 elbow is bent 90 degrees at zero) and sign flips.
Wrists come from the hand's orientation relative to the forearm.
"""

from __future__ import annotations

import numpy as np

from ..retarget.geometry import body_frames, frame_from_axes, unit, unwrap
from ..retarget.hands import ee_orientation_cam, gripper_closedness
from ..retarget.humanoid import HUMANOID_JOINTS, humanoid_joint_angles
from ..schema import HumanMotion
from .urdf import parse_urdf

G1_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


def _rx(a):
    c, s, o, z = np.cos(a), np.sin(a), np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([o, z, z], -1), np.stack([z, c, -s], -1), np.stack([z, s, c], -1)], -2)


def _ry(a):
    c, s, o, z = np.cos(a), np.sin(a), np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, z, s], -1), np.stack([z, o, z], -1), np.stack([-s, z, c], -1)], -2)


def _rz(a):
    c, s, o, z = np.cos(a), np.sin(a), np.ones_like(a), np.zeros_like(a)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1), np.stack([z, z, o], -1)], -2)


def g1_joint_angles(motion: HumanMotion, profile) -> tuple[np.ndarray, list[str], dict]:
    generic = dict(zip(HUMANOID_JOINTS, humanoid_joint_angles(motion).astype(np.float64).T))
    fr = body_frames(motion.body_3d.astype(np.float64))
    out: dict[str, np.ndarray] = {}

    # waist: G1 order is yaw(z) -> roll(x) -> pitch(y)
    rel = np.swapaxes(fr["pelvis_R"], -1, -2) @ fr["torso_R"]
    out["waist_roll_joint"] = np.arcsin(np.clip(rel[:, 2, 1], -1, 1))
    out["waist_yaw_joint"] = np.arctan2(-rel[:, 0, 1], rel[:, 1, 1])
    out["waist_pitch_joint"] = np.arctan2(-rel[:, 2, 0], rel[:, 2, 2])

    for i, s in enumerate(("left", "right")):
        for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee"):
            out[f"{s}_{j}_joint"] = generic[f"{s}_{j}"]
        out[f"{s}_ankle_pitch_joint"] = -generic[f"{s}_ankle_pitch"]     # G1: positive = toes down
        out[f"{s}_ankle_roll_joint"] = np.where(np.isfinite(generic[f"{s}_knee"]), 0.0, np.nan)
        sp, sr, sy = (generic[f"{s}_shoulder_{k}"] for k in ("pitch", "roll", "yaw"))
        elbow = np.pi / 2 - generic[f"{s}_elbow"]                        # G1 zero = forearm forward
        out[f"{s}_shoulder_pitch_joint"], out[f"{s}_shoulder_roll_joint"] = sp, sr
        out[f"{s}_shoulder_yaw_joint"], out[f"{s}_elbow_joint"] = sy, elbow

        # wrist: hand orientation in the forearm frame, roll(x) -> pitch(y) -> yaw(z).
        # G1 hand frame at zero: x along the fingers, z towards the thumb side.
        R_fore = _ry(sp) @ _rx(sr) @ _rz(sy) @ _ry(elbow)
        Rh = ee_orientation_cam(motion, i).astype(np.float64)           # columns: normal, knuckles, approach
        hand_cam = frame_from_axes(np.cross(Rh[:, :, 1], Rh[:, :, 2]), Rh[:, :, 1])   # y = z × x, z = knuckles
        hand_cam = np.stack([unit(Rh[:, :, 2]), hand_cam[:, :, 1], hand_cam[:, :, 2]], -1)
        hand = np.swapaxes(fr["torso_R"], -1, -2) @ hand_cam
        W = np.swapaxes(R_fore, -1, -2) @ hand
        out[f"{s}_wrist_pitch_joint"] = np.arcsin(np.clip(W[:, 0, 2], -1, 1))
        out[f"{s}_wrist_roll_joint"] = np.arctan2(-W[:, 1, 2], W[:, 2, 2])
        out[f"{s}_wrist_yaw_joint"] = np.arctan2(-W[:, 0, 1], W[:, 0, 0])

    limits = {j.name: (j.lower, j.upper) for j in parse_urdf(profile.urdf).values() if j.movable}
    q = np.stack([out[n] for n in G1_JOINTS], axis=-1)
    lo = np.array([limits[n][0] for n in G1_JOINTS])
    hi = np.array([limits[n][1] for n in G1_JOINTS])
    # Angles are continuous in time (unwrapped). Shift each joint by whole turns so its typical
    # value is in the principal range, rather than wrapping every sample (which would tear it).
    q = unwrap(q)
    med = np.array([np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0 for col in q.T])
    q = q - 2 * np.pi * np.round(med / (2 * np.pi))
    clipped = (q < lo) | (q > hi)
    good = np.isfinite(q)
    info = {"robot": profile.key, "units": "radians (hand_closed in [0, 1])",
            "limit_clipped_fraction": round(float(clipped[good].mean()) if good.any() else 0.0, 3),
            "notes": profile.notes}
    q = np.clip(q, lo, hi)
    hands = np.stack([gripper_closedness(motion, 0), gripper_closedness(motion, 1)], -1)
    state = np.concatenate([q, hands], axis=-1).astype(np.float32)
    return state, G1_JOINTS + ["left_hand_closed", "right_hand_closed"], info
