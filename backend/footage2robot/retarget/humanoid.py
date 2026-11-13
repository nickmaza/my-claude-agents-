"""Whole-body joint angles for humanoid retargeting.

Angles follow a generic humanoid kinematic tree (Unitree/GR-1-like ordering):
each ball joint is decomposed as R = Ry(pitch) · Rx(roll) · Rz(yaw) with the
zero pose being a straight limb hanging down along -z, followed by a 1-DoF
hinge (elbow / knee). Map names to your robot's joints (and clip to its limits)
with a ``joint_map`` in the export config or downstream.
"""

from __future__ import annotations

import numpy as np

from ..schema import B, HumanMotion
from .geometry import angle_between, body_frames, matrix_to_euler, to_frame, unit, unwrap
from .hands import gripper_closedness

HUMANOID_JOINTS = [
    "waist_yaw", "waist_roll", "waist_pitch", "head_yaw", "head_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch",
    "left_hand_closed", "right_hand_closed",
]


def _ball_joint(u: np.ndarray, ref: np.ndarray):
    """Pitch/roll of limb direction ``u`` and yaw from a reference direction.

    ``u``/``ref`` are unit vectors (T, 3) in the parent frame.
    """
    roll = np.arcsin(np.clip(u[:, 1], -1, 1))
    pitch = np.arctan2(-u[:, 0], -u[:, 2])
    cp, sp, cr, sr = np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    # columns of Ry(pitch)·Rx(roll); local = (Ry·Rx)^T · ref
    x_axis = np.stack([cp, np.zeros_like(cp), -sp], -1)
    y_axis = np.stack([sp * sr, cr, cp * sr], -1)
    lx = np.sum(x_axis * ref, -1)
    ly = np.sum(y_axis * ref, -1)
    yaw = np.arctan2(ly, lx)
    yaw = np.where(np.hypot(lx, ly) < 0.17, np.nan, yaw)   # ill-defined when ref ∥ limb
    return pitch, roll, yaw


def _hold_nan(a: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Forward/backward fill NaNs in ``a`` on frames where ``valid`` is True."""
    a = a.copy()
    idx = np.where(np.isfinite(a))[0]
    if len(idx) == 0:
        a[valid] = 0.0
        return a
    fill = np.where(valid & ~np.isfinite(a))[0]
    nearest = idx[np.clip(np.searchsorted(idx, fill), 1, len(idx)) - 1]
    a[fill] = a[nearest]
    return a


def humanoid_joint_angles(motion: HumanMotion) -> np.ndarray:
    """(T, 25) joint angles in radians (+ 2 hand closedness values in [0, 1])."""
    body = motion.body_3d.astype(np.float64)
    fr = body_frames(body)
    T = body.shape[0]
    cols: dict[str, np.ndarray] = {}

    # waist: torso orientation relative to pelvis
    rel = np.swapaxes(fr["pelvis_R"], -1, -2) @ fr["torso_R"]
    e = matrix_to_euler(rel)
    cols["waist_roll"], cols["waist_pitch"], cols["waist_yaw"] = e[:, 0], e[:, 1], e[:, 2]

    # head: ear midpoint -> nose direction in torso frame
    ears = 0.5 * (body[:, B["left_ear"]] + body[:, B["right_ear"]])
    zero = np.zeros((T, 3))
    d = unit(to_frame(body[:, B["nose"]] - ears, fr["torso_R"], zero))
    cols["head_yaw"] = np.arctan2(d[:, 1], d[:, 0])
    cols["head_pitch"] = -np.arcsin(np.clip(d[:, 2], -1, 1))

    for s in ("left", "right"):
        sh, el, wr = (body[:, B[f"{s}_{k}"]] for k in ("shoulder", "elbow", "wrist"))
        u = unit(to_frame(el - sh, fr["torso_R"], zero))
        f = unit(to_frame(wr - el, fr["torso_R"], zero))
        pitch, roll, yaw = _ball_joint(u, f)
        valid = np.isfinite(pitch)
        cols[f"{s}_shoulder_pitch"], cols[f"{s}_shoulder_roll"] = pitch, roll
        cols[f"{s}_shoulder_yaw"] = _hold_nan(yaw, valid)
        cols[f"{s}_elbow"] = angle_between(u, f)

        hip, knee, ank = (body[:, B[f"{s}_{k}"]] for k in ("hip", "knee", "ankle"))
        heel, toe = body[:, B[f"{s}_heel"]], body[:, B[f"{s}_foot_index"]]
        th = unit(to_frame(knee - hip, fr["pelvis_R"], zero))
        shin = unit(to_frame(ank - knee, fr["pelvis_R"], zero))
        foot = unit(to_frame(toe - heel, fr["pelvis_R"], zero))
        pitch, roll, yaw = _ball_joint(th, foot)
        valid = np.isfinite(pitch)
        cols[f"{s}_hip_pitch"], cols[f"{s}_hip_roll"] = pitch, roll
        cols[f"{s}_hip_yaw"] = _hold_nan(yaw, valid)
        cols[f"{s}_knee"] = angle_between(th, shin)
        cols[f"{s}_ankle_pitch"] = angle_between(shin, foot) - np.pi / 2

    cols["left_hand_closed"] = gripper_closedness(motion, 0)
    cols["right_hand_closed"] = gripper_closedness(motion, 1)
    out = np.stack([cols[n] for n in HUMANOID_JOINTS], axis=-1)
    yaw_cols = [i for i, n in enumerate(HUMANOID_JOINTS) if n.endswith("_yaw")]
    out[:, yaw_cols] = unwrap(out[:, yaw_cols])
    return out.astype(np.float32)
