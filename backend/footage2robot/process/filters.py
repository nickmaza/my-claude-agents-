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


def clean_motion(motion: HumanMotion, max_gap_s: float = 0.25, smooth_s: float = 0.05,
                 min_confidence: float = 0.3) -> HumanMotion:
    """Return a cleaned copy: drop low-confidence points, fill short gaps, smooth."""
    m = copy.deepcopy(motion)
    max_gap = max(1, int(round(max_gap_s * m.fps)))
    sigma = smooth_s * m.fps

    low = m.body_conf < min_confidence
    m.body_3d[low] = np.nan
    m.body_2d[low] = np.nan
    hand_low = m.hands_conf < min_confidence
    m.hands_3d[hand_low] = np.nan
    m.hands_2d[hand_low] = np.nan

    for name in ("body_3d", "body_2d", "hands_3d", "hands_2d"):
        setattr(m, name, smooth(fill_gaps(getattr(m, name), max_gap), sigma))
    # confidences: carry forward through filled gaps so downstream masks agree
    m.body_conf = np.where(np.isfinite(m.body_3d[..., 0]), np.maximum(m.body_conf, min_confidence), 0).astype(np.float32)
    m.hands_conf = np.where(np.isfinite(m.hands_3d[..., 0, 0]), np.maximum(m.hands_conf, min_confidence), 0).astype(np.float32)
    m.meta["cleaned"] = {"max_gap_s": max_gap_s, "smooth_s": smooth_s, "min_confidence": min_confidence}
    return m
