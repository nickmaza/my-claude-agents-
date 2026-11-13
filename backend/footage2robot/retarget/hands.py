"""Hand-level features: end-effector pose, gripper closedness, finger joints."""

from __future__ import annotations

import numpy as np

from ..schema import B, H, HumanMotion
from .geometry import angle_between, frame_from_z_y, unit

FINGERS = {
    "thumb": [H["wrist"], H["thumb_cmc"], H["thumb_mcp"], H["thumb_ip"], H["thumb_tip"]],
    "index": [H["wrist"], H["index_mcp"], H["index_pip"], H["index_dip"], H["index_tip"]],
    "middle": [H["wrist"], H["middle_mcp"], H["middle_pip"], H["middle_dip"], H["middle_tip"]],
    "ring": [H["wrist"], H["ring_mcp"], H["ring_pip"], H["ring_dip"], H["ring_tip"]],
    "pinky": [H["wrist"], H["pinky_mcp"], H["pinky_pip"], H["pinky_dip"], H["pinky_tip"]],
}
FLEX_JOINTS = {
    "thumb": ["cmc", "mcp", "ip"],
    "index": ["mcp", "pip", "dip"],
    "middle": ["mcp", "pip", "dip"],
    "ring": ["mcp", "pip", "dip"],
    "pinky": ["mcp", "pip", "dip"],
}
ABDUCTION_PAIRS = [("thumb", "index"), ("index", "middle"), ("middle", "ring"), ("ring", "pinky")]

FINGER_JOINT_NAMES = [f"{f}_{j}_flex" for f in FINGERS for j in FLEX_JOINTS[f]] + \
    [f"{a}_{b}_spread" for a, b in ABDUCTION_PAIRS]

# Thumb-index tip distance / palm length, at which we call the hand fully closed / open.
GRIP_CLOSED_RATIO = 0.25
GRIP_OPEN_RATIO = 0.95


def hand_points_cam(motion: HumanMotion, side: int) -> np.ndarray:
    """Hand landmarks placed in the body's camera frame (T, 21, 3).

    MediaPipe hand world landmarks are hand-centred, so we translate them so the
    hand's wrist sits on the body model's wrist.
    """
    hand = motion.hands_3d[:, side]
    body_wrist = motion.body_3d[:, B["left_wrist"] if side == 0 else B["right_wrist"]]
    return hand - hand[:, :1] + body_wrist[:, None]


def ee_orientation_cam(motion: HumanMotion, side: int) -> np.ndarray:
    """Hand frame in camera coordinates (T, 3, 3); columns x, y, z.

    z = approach direction (wrist -> middle knuckle), y = across the knuckles
    (pinky -> index), x = palm normal (y × z). Falls back to the body model's
    coarse hand points (wrist/index/pinky) when the hand tracker lost the hand.
    """
    h = motion.hands_3d[:, side]
    z = h[:, H["middle_mcp"]] - h[:, H["wrist"]]
    y = h[:, H["index_mcp"]] - h[:, H["pinky_mcp"]]
    R = frame_from_z_y(z, y)

    s = "left" if side == 0 else "right"
    bw, bi, bp = (motion.body_3d[:, B[f"{s}_{k}"]] for k in ("wrist", "index", "pinky"))
    R_body = frame_from_z_y(0.5 * (bi + bp) - bw, bi - bp)
    missing = ~np.isfinite(R[:, 0, 0])
    R[missing] = R_body[missing]
    return R


def ee_position_cam(motion: HumanMotion, side: int) -> np.ndarray:
    """Grasp point: centre of the palm (wrist, index & pinky knuckles), camera frame."""
    pts = hand_points_cam(motion, side)
    palm = pts[:, [H["wrist"], H["index_mcp"], H["pinky_mcp"]]].mean(axis=1)
    s = "left" if side == 0 else "right"
    coarse = motion.body_3d[:, [B[f"{s}_wrist"], B[f"{s}_index"], B[f"{s}_pinky"]]].mean(axis=1)
    missing = ~np.isfinite(palm[:, 0])
    palm[missing] = coarse[missing]
    return palm


def gripper_closedness(motion: HumanMotion, side: int) -> np.ndarray:
    """0 = fully open, 1 = fully closed (pinch/grasp). (T,)"""
    h = motion.hands_3d[:, side]
    palm_len = np.linalg.norm(h[:, H["middle_mcp"]] - h[:, H["wrist"]], axis=-1)
    pinch = np.linalg.norm(h[:, H["thumb_tip"]] - h[:, H["index_tip"]], axis=-1)
    # whole-hand curl: fingertips close to the wrist also indicates a power grasp
    tips = h[:, [H["index_tip"], H["middle_tip"], H["ring_tip"], H["pinky_tip"]]]
    curl = np.linalg.norm(tips - h[:, None, H["wrist"]], axis=-1).mean(axis=-1)
    ratio = np.minimum(pinch / (palm_len + 1e-9), (curl / (palm_len + 1e-9) - 1.0) * 1.2)
    closed = 1.0 - (ratio - GRIP_CLOSED_RATIO) / (GRIP_OPEN_RATIO - GRIP_CLOSED_RATIO)
    closed = np.clip(closed, 0.0, 1.0)

    # coarse fallback from the body model's thumb/index points
    s = "left" if side == 0 else "right"
    bt, bi, bw = (motion.body_3d[:, B[f"{s}_{k}"]] for k in ("thumb", "index", "wrist"))
    coarse = 1 - np.clip((np.linalg.norm(bt - bi, axis=-1) / (np.linalg.norm(bi - bw, axis=-1) + 1e-9) - 0.2) / 0.5, 0, 1)
    missing = ~np.isfinite(closed)
    closed[missing] = coarse[missing]
    return closed.astype(np.float32)


def finger_joint_angles(motion: HumanMotion, side: int) -> np.ndarray:
    """(T, 19) radians: 15 flexion angles (3 per finger) + 4 spread angles."""
    h = motion.hands_3d[:, side]
    flex = []
    for chain in FINGERS.values():
        for k in range(1, 4):
            flex.append(angle_between(h[:, chain[k]] - h[:, chain[k - 1]],
                                      h[:, chain[k + 1]] - h[:, chain[k]]))
    # spread: angle between proximal phalanges, projected into the palm plane
    normal = unit(np.cross(h[:, H["index_mcp"]] - h[:, H["wrist"]], h[:, H["pinky_mcp"]] - h[:, H["wrist"]]))

    def proj(v):
        return v - np.sum(v * normal, axis=-1, keepdims=True) * normal

    spread = []
    for a, b in ABDUCTION_PAIRS:
        ca, cb = FINGERS[a], FINGERS[b]
        va = h[:, ca[2]] - h[:, ca[1]] if a != "thumb" else h[:, ca[3]] - h[:, ca[2]]
        vb = h[:, cb[2]] - h[:, cb[1]]
        spread.append(angle_between(proj(va), proj(vb)))
    return np.stack(flex + spread, axis=-1).astype(np.float32)


def hand_activity(motion: HumanMotion, side: int) -> float:
    """Score how much a hand is 'doing' the task: path length x tracking coverage."""
    pos = ee_position_cam(motion, side)
    good = np.isfinite(pos[:, 0])
    if good.sum() < 2:
        return 0.0
    p = pos[good]
    path = float(np.linalg.norm(np.diff(p, axis=0), axis=-1).sum())
    grip = gripper_closedness(motion, side)
    grip_var = float(np.nanstd(grip)) if np.isfinite(grip).any() else 0.0
    coverage = float(np.isfinite(motion.hands_3d[:, side, 0, 0]).mean())
    return (path + grip_var) * (0.5 + coverage)
