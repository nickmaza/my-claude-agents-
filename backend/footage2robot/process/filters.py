"""Temporal clean-up of tracked motion: gap filling and zero-phase smoothing."""

from __future__ import annotations

import copy

import numpy as np

from ..schema import HumanMotion


def fill_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate NaN runs of length <= ``max_gap`` along axis 0.

    Leading/trailing gaps are left as NaN (we never extrapolate).
    """
    x = np.array(x, dtype=np.float32, copy=True)
    flat = x.reshape(x.shape[0], -1)
    t = np.arange(flat.shape[0])
    for c in range(flat.shape[1]):
        col = flat[:, c]
        good = np.isfinite(col)
        if good.all() or not good.any():
            continue
        bad_idx = np.where(~good)[0]
        # split bad indices into contiguous runs
        runs = np.split(bad_idx, np.where(np.diff(bad_idx) != 1)[0] + 1)
        for run in runs:
            lo, hi = run[0] - 1, run[-1] + 1
            if lo < 0 or hi >= len(col) or len(run) > max_gap:
                continue
            col[run] = np.interp(t[run], [lo, hi], [col[lo], col[hi]])
        flat[:, c] = col
    return flat.reshape(x.shape)


def _gauss_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    return k / k.sum()


def smooth(x: np.ndarray, sigma_frames: float) -> np.ndarray:
    """Zero-phase Gaussian smoothing along axis 0 that respects NaN gaps."""
    if sigma_frames <= 0 or x.shape[0] < 3:
        return x.astype(np.float32, copy=True)
    k = _gauss_kernel(sigma_frames)
    flat = x.reshape(x.shape[0], -1).astype(np.float64)
    finite = np.isfinite(flat)
    vals = np.where(finite, flat, 0.0)
    num = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, vals)
    den = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, finite.astype(np.float64))
    out = np.where(finite, num / np.maximum(den, 1e-9), np.nan)
    return out.reshape(x.shape).astype(np.float32)


def rotation_outliers(R: np.ndarray, fps: float, window_s: float = 0.5, max_deviation_rad: float = 0.6,
                      erode: int = 2) -> np.ndarray:
    """Frames whose orientation (T, 3, 3) cannot be trusted, (T,) bool.

    A torso or a hand does not turn tens of degrees and back within a frame or
    two, but trackers emit exactly that when they flip between interpretations
    or are about to lose (or have just re-found) their target. A frame is
    rejected when its orientation deviates strongly from the local median
    orientation, or when it borders a tracking dropout.
    """
    T = R.shape[0]
    valid = np.isfinite(R).all(axis=(1, 2))
    bad = np.zeros(T, bool)
    half = max(1, int(round(window_s * fps)))
    for t in np.where(valid)[0]:
        lo, hi = max(0, t - half), min(T, t + half + 1)
        near = R[lo:hi][valid[lo:hi]]
        if len(near) < 3:
            continue
        u, _, vt = np.linalg.svd(np.median(near, axis=0))       # nearest rotation to the element-wise median
        ref = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
        angle = np.arccos(np.clip((np.trace(ref.T @ R[t]) - 1) / 2, -1, 1))
        bad[t] = angle > max_deviation_rad
    # frames next to a dropout are the tracker failing or recovering
    missing = ~valid
    if erode and 0 < missing.sum() < T:
        near_gap = missing.copy()
        for k in range(1, erode + 1):
            near_gap[k:] |= missing[:-k]
            near_gap[:-k] |= missing[k:]
        bad |= near_gap & valid
    return bad


def implausible_frames(body_3d: np.ndarray, fps: float) -> np.ndarray:
    """Frames whose body estimate cannot be trusted (see ``rotation_outliers``)."""
    from ..retarget.geometry import body_frames
    return rotation_outliers(body_frames(body_3d.astype(np.float64))["torso_R"], fps)


def implausible_hand_frames(hand_3d: np.ndarray, fps: float) -> np.ndarray:
    """Frames where one hand's (T, 21, 3) orientation flips away from its neighbours."""
    from ..retarget.geometry import frame_from_z_y
    from ..schema import H
    h = hand_3d.astype(np.float64)
    R = frame_from_z_y(h[:, H["middle_mcp"]] - h[:, H["wrist"]], h[:, H["index_mcp"]] - h[:, H["pinky_mcp"]])
    return rotation_outliers(R, fps, window_s=0.35, max_deviation_rad=0.7, erode=1)


def clean_motion(motion: HumanMotion, max_gap_s: float = 0.25, smooth_s: float = 0.05,
                 min_confidence: float = 0.5) -> HumanMotion:
    """Return a cleaned copy: drop untrustworthy points and frames, fill short gaps, smooth."""
    m = copy.deepcopy(motion)
    max_gap = max(1, int(round(max_gap_s * m.fps)))
    sigma = smooth_s * m.fps

    low = m.body_conf < min_confidence
    m.body_3d[low] = np.nan
    m.body_2d[low] = np.nan
    hand_low = m.hands_conf < min_confidence
    m.hands_3d[hand_low] = np.nan
    m.hands_2d[hand_low] = np.nan

    rejected = implausible_frames(m.body_3d, m.fps)
    for name in ("body_3d", "body_2d", "hands_3d", "hands_2d"):
        getattr(m, name)[rejected] = np.nan
    m.meta["rejected_frames"] = int(rejected.sum())
    for side in (0, 1):
        flipped = implausible_hand_frames(m.hands_3d[:, side], m.fps)
        m.hands_3d[flipped, side] = np.nan
        m.hands_2d[flipped, side] = np.nan

    for name in ("body_3d", "body_2d", "hands_3d", "hands_2d"):
        setattr(m, name, smooth(fill_gaps(getattr(m, name), max_gap), sigma))
    # confidences: carry forward through filled gaps so downstream masks agree
    m.body_conf = np.where(np.isfinite(m.body_3d[..., 0]), np.maximum(m.body_conf, min_confidence), 0).astype(np.float32)
    m.hands_conf = np.where(np.isfinite(m.hands_3d[..., 0, 0]), np.maximum(m.hands_conf, min_confidence), 0).astype(np.float32)
    m.meta["cleaned"] = {"max_gap_s": max_gap_s, "smooth_s": smooth_s, "min_confidence": min_confidence}
    return m
