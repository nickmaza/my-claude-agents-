"""Retargeting: HumanMotion -> robot state trajectories for several embodiments."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..schema import BODY_LANDMARKS, HAND_LANDMARKS, SIDES, HumanMotion
from .geometry import ROTATION_NAMES, body_frames, encode_rotation, rotate_into, to_frame
from .hands import (FINGER_JOINT_NAMES, ee_orientation_cam, ee_position_cam, finger_joint_angles,
                    gripper_closedness, hand_activity, hand_points_cam)
from .humanoid import HUMANOID_JOINTS, humanoid_joint_angles
from ..robots import ROBOTS, retarget_robot


@dataclass
class RetargetConfig:
    rotation: str = "euler"        # euler | rotvec | quat | rot6d
    scale: float = 1.0             # human metres -> robot metres (workspace scaling)
    base_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)  # added to EE xyz after scaling
    side: str = "auto"             # hand for single-hand embodiments: auto | left | right
    frame: str = "auto"            # auto (marker if calibrated, else torso) | torso | marker


@dataclass
class Trajectory:
    """A robot state trajectory for one embodiment."""

    embodiment: str
    state: np.ndarray              # (T, D) float32, NaN where the human was not tracked
    names: list[str]
    info: dict = field(default_factory=dict)
    # indices of state dims that are absolute targets even in delta-action mode
    absolute_dims: list[int] = field(default_factory=list)
    # dims that may be untracked without invalidating the frame (e.g. legs in
    # upper-body footage); they are hold-filled, or zero if never seen
    optional_dims: list[int] = field(default_factory=list)
    # extra per-frame validity (T,), e.g. "torso tracked" when every dim is optional
    valid: np.ndarray | None = None


EMBODIMENTS = {
    "single_arm": "Single arm + parallel gripper (end-effector pose of the most active hand)",
    "bimanual": "Two arms + grippers (left & right end-effector poses)",
    "dex_hand": "Dexterous hand (wrist pose + 19 finger joint angles of the most active hand)",
    "humanoid": "Humanoid whole-body joint angles (waist, head, arms, legs, hands)",
    "keypoints": "Raw human 3D keypoints in the body frame (body + both hands)",
    **{key: f"{p.label} (joint positions via {'limb-angle mapping' if p.kind == 'humanoid' else 'inverse kinematics'})"
       for key, p in ROBOTS.items()},
}
GENERIC_EMBODIMENTS = ["single_arm", "bimanual", "dex_hand", "humanoid", "keypoints"]


def register_robot(profile) -> None:
    """Make a custom robot profile available as an embodiment."""
    ROBOTS[profile.key] = profile
    EMBODIMENTS[profile.key] = f"{profile.label} (joint positions via inverse kinematics)"


def reference_frame(motion: HumanMotion, cfg: RetargetConfig) -> tuple[np.ndarray, np.ndarray, str]:
    """Frame that end-effector poses are expressed in: ``(R (T,3,3), origin (T,3), label)``."""
    has_marker = motion.world_R is not None and motion.world_origin is not None
    if cfg.frame == "marker" and not has_marker:
        raise ValueError("frame='marker' needs a clip calibrated with a workspace marker")
    if has_marker and cfg.frame in ("auto", "marker"):
        R = np.broadcast_to(motion.world_R.astype(np.float64), (motion.num_frames, 3, 3))
        return R, motion.world_origin.astype(np.float64), "marker"
    fr = body_frames(motion.body_3d.astype(np.float64))
    return fr["torso_R"], fr["torso_origin"], "torso"


FRAME_LABELS = {
    "torso": "torso (x fwd, y left, z up), origin mid-shoulders",
    "marker": "workspace marker (x right, y up on the print, z out of it), origin marker centre",
}


def _torso_tracked(motion: HumanMotion) -> np.ndarray:
    return np.isfinite(body_frames(motion.body_3d.astype(np.float64))["torso_R"]).all(axis=(1, 2))


def pick_side(motion: HumanMotion, cfg: RetargetConfig) -> int:
    if cfg.side in SIDES:
        return SIDES.index(cfg.side)
    return int(np.argmax([hand_activity(motion, 0), hand_activity(motion, 1)]))


def _ee_pose(motion: HumanMotion, side: int, cfg: RetargetConfig) -> tuple[np.ndarray, list[str]]:
    ref_R, ref_origin, _ = reference_frame(motion, cfg)
    pos = to_frame(ee_position_cam(motion, side).astype(np.float64), ref_R, ref_origin)
    pos = pos * cfg.scale + np.asarray(cfg.base_offset)
    R = rotate_into(ee_orientation_cam(motion, side).astype(np.float64), ref_R)
    rot = encode_rotation(R, cfg.rotation)
    return np.concatenate([pos, rot], axis=-1), ["x", "y", "z"] + ROTATION_NAMES[cfg.rotation]


def _arm(motion: HumanMotion, side: int, cfg: RetargetConfig, prefix: str = ""):
    pose, names = _ee_pose(motion, side, cfg)
    grip = gripper_closedness(motion, side)[:, None]
    state = np.concatenate([pose, grip], axis=-1)
    return state, [f"{prefix}{n}" for n in names + ["gripper"]]


def retarget(motion: HumanMotion, embodiment: str, cfg: RetargetConfig | None = None) -> Trajectory:
    cfg = cfg or RetargetConfig()
    optional: list[int] = []
    valid = None
    if embodiment == "single_arm":
        side = pick_side(motion, cfg)
        state, names = _arm(motion, side, cfg)
        info = {"hand": SIDES[side], "frame": FRAME_LABELS[reference_frame(motion, cfg)[2]]}
        absolute = [len(names) - 1]
    elif embodiment == "bimanual":
        parts = [_arm(motion, s, cfg, prefix=f"{SIDES[s]}_") for s in (0, 1)]
        state = np.concatenate([p[0] for p in parts], axis=-1)
        names = parts[0][1] + parts[1][1]
        info = {"frame": FRAME_LABELS[reference_frame(motion, cfg)[2]]}
        d = len(parts[0][1])
        absolute = [d - 1, 2 * d - 1]
    elif embodiment == "dex_hand":
        side = pick_side(motion, cfg)
        pose, names = _ee_pose(motion, side, cfg)
        state = np.concatenate([pose, finger_joint_angles(motion, side)], axis=-1)
        names = [f"wrist_{n}" for n in names] + FINGER_JOINT_NAMES
        info = {"hand": SIDES[side], "finger_units": "radians",
                "frame": FRAME_LABELS[reference_frame(motion, cfg)[2]]}
        absolute = list(range(pose.shape[1], state.shape[1]))
    elif embodiment == "humanoid":
        state, names = humanoid_joint_angles(motion), list(HUMANOID_JOINTS)
        info = {"units": "radians (hand_closed in [0,1])"}
        absolute = list(range(len(names)))
        optional = list(range(len(names)))
        valid = _torso_tracked(motion)
    elif embodiment == "keypoints":
        ref_R, ref_origin, frame = reference_frame(motion, cfg)
        body = to_frame(motion.body_3d.astype(np.float64), ref_R, ref_origin)
        hands = [to_frame(hand_points_cam(motion, s).astype(np.float64), ref_R, ref_origin) for s in (0, 1)]
        T = motion.num_frames
        flags = np.stack([np.isfinite(h[:, 0, 0]) for h in hands], axis=-1).astype(np.float64)
        state = np.concatenate([body.reshape(T, -1)] + [h.reshape(T, -1) for h in hands] + [flags], axis=-1)
        names = [f"body_{n}_{a}" for n in BODY_LANDMARKS for a in "xyz"] + \
            [f"{s}_hand_{n}_{a}" for s in SIDES for n in HAND_LANDMARKS for a in "xyz"] + \
            [f"{s}_hand_tracked" for s in SIDES]
        core = {f"body_{n}_{a}" for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
                for a in "xyz"}
        optional = [i for i, n in enumerate(names) if n not in core]
        info = {"frame": FRAME_LABELS[frame], "units": "metres"}
        absolute = list(range(len(names)))
    elif embodiment in ROBOTS:
        state, names, info = retarget_robot(motion, ROBOTS[embodiment], cfg)
        absolute = list(range(len(names)))
        if ROBOTS[embodiment].kind == "humanoid":    # any limb may be out of frame
            optional = list(range(len(names)))
            valid = _torso_tracked(motion)
    else:
        raise KeyError(f"Unknown embodiment '{embodiment}'. Options: {sorted(EMBODIMENTS)}")
    info.update({"rotation": cfg.rotation, "scale": cfg.scale})
    return Trajectory(embodiment, state.astype(np.float32), names, info, absolute, optional, valid)


__all__ = ["EMBODIMENTS", "GENERIC_EMBODIMENTS", "RetargetConfig", "Trajectory", "retarget", "pick_side",
           "reference_frame", "register_robot"]
