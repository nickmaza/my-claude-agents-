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


def _project_rotations(R: np.ndarray) -> np.ndarray:
    """Nearest proper rotation to each (possibly blended) 3x3 matrix; NaN rows stay NaN."""
    out = np.full_like(R, np.nan)
    ok = np.isfinite(R).all(axis=(1, 2))
    if ok.any():
        u, _, vt = np.linalg.svd(R[ok])
        d = np.sign(np.linalg.det(u @ vt))
        u[:, :, 2] *= d[:, None]
        out[ok] = u @ vt
    return out


def ee_orientation_cam(motion: HumanMotion, side: int) -> np.ndarray:
    """Hand frame in camera coordinates (T, 3, 3); columns x, y, z.

    z = approach direction (wrist -> middle knuckle), y = across the knuckles
    (pinky -> index), x = palm normal (y × z).

    The hand tracker gives a precise orientation but drops out now and then; the
    body model's coarse hand points (wrist/index/pinky) are always there but
    noisy. Switching between the two mid-clip makes the orientation jump, so
    when the hand tracker covers a reasonable share of the clip its orientation
    is carried through its own gaps (interpolated, held at the ends) and the
    coarse estimate is used only for hands the tracker never really saw.
    """
    h = motion.hands_3d[:, side]
    z = h[:, H["middle_mcp"]] - h[:, H["wrist"]]
    y = h[:, H["index_mcp"]] - h[:, H["pinky_mcp"]]
    R = frame_from_z_y(z, y)

    s = "left" if side == 0 else "right"
    bw, bi, bp = (motion.body_3d[:, B[f"{s}_{k}"]] for k in ("wrist", "index", "pinky"))
    R_body = frame_from_z_y(0.5 * (bi + bp) - bw, bi - bp)
    tracked = np.isfinite(R[:, 0, 0])
    if tracked.mean() >= 0.3:
        idx = np.where(tracked)[0]
        t = np.arange(len(R))
        flat = R.reshape(len(R), 9).copy()
        for c in range(9):
            flat[:, c] = np.interp(t, idx, flat[idx, c])      # linear inside gaps, held at the ends
        from ..process.filters import smooth
        flat = smooth(flat, 0.08 * motion.fps)                # orientation is the noisiest signal we have
        R = _project_rotations(flat.reshape(-1, 3, 3).astype(np.float64))
        R[~np.isfinite(bw[:, 0])] = np.nan                    # no arm in view: nothing to orient
    else:
        R[~tracked] = R_body[~tracked]
    return R


def _carry_through_gaps(x: np.ndarray, min_coverage: float = 0.3) -> np.ndarray | None:
    """Interpolate a (T, D) signal across its NaN gaps (held at the ends).

    Returns None when too little of it was observed to be worth carrying.
    """
    ok = np.isfinite(x).all(axis=1)
    if ok.mean() < min_coverage:
        return None
    idx, t = np.where(ok)[0], np.arange(len(x))
    return np.stack([np.interp(t, idx, x[idx, c]) for c in range(x.shape[1])], axis=1)


def ee_position_cam(motion: HumanMotion, side: int) -> np.ndarray:
    """Grasp point: centre of the palm (wrist, index & pinky knuckles), camera frame.

    The palm sits a few centimetres from the body model's wrist; that offset
    comes from the hand tracker and is carried through the tracker's dropouts
    (see ``ee_orientation_cam``), so the grasp point does not hop when the hand
    tracker flickers.
    """
    hand = motion.hands_3d[:, side]
    s = "left" if side == 0 else "right"
    wrist = motion.body_3d[:, B[f"{s}_wrist"]].astype(np.float64)
    offset = hand[:, [H["wrist"], H["index_mcp"], H["pinky_mcp"]]].mean(axis=1) - hand[:, H["wrist"]]
    carried = _carry_through_gaps(offset.astype(np.float64))
    if carried is not None:
        return wrist + carried
    coarse = motion.body_3d[:, [B[f"{s}_wrist"], B[f"{s}_index"], B[f"{s}_pinky"]]].mean(axis=1).astype(np.float64)
    palm = wrist + offset
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
    return _median_filter(closed, motion.fps).astype(np.float32)


def _median_filter(x: np.ndarray, fps: float, window_s: float = 0.17) -> np.ndarray:
    """Running median that ignores NaNs: removes single-frame open/close flicker, keeps real grasps."""
    half = max(1, int(round(window_s * fps / 2)))
    out = x.copy()
    for t in np.where(np.isfinite(x))[0]:
        near = x[max(0, t - half):t + half + 1]
        out[t] = np.median(near[np.isfinite(near)])
    return out


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
