"""Vectorised 3D geometry helpers (time-major arrays, NaN-propagating)."""

from __future__ import annotations

import numpy as np

from ..schema import B


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cos = np.sum(unit(a) * unit(b), axis=-1)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def frame_from_axes(y_hint: np.ndarray, z_hint: np.ndarray) -> np.ndarray:
    """Right-handed rotation (…, 3, 3) whose columns are x, y, z.

    ``y_hint`` is kept exactly (normalised); ``z_hint`` is orthogonalised to it.
    """
    y = unit(y_hint)
    z = unit(z_hint - np.sum(z_hint * y, axis=-1, keepdims=True) * y)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=-1)


def frame_from_z_y(z_hint: np.ndarray, y_hint: np.ndarray) -> np.ndarray:
    """Rotation with columns x, y, z where ``z_hint`` is kept exactly."""
    z = unit(z_hint)
    y = unit(y_hint - np.sum(y_hint * z, axis=-1, keepdims=True) * z)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=-1)


def body_frames(body_3d: np.ndarray) -> dict[str, np.ndarray]:
    """Torso and pelvis frames (camera->body) plus their origins.

    Body frame convention (REP-103-like): +x forward (out of the chest),
    +y the person's left, +z up.
    """
    ls, rs = body_3d[:, B["left_shoulder"]], body_3d[:, B["right_shoulder"]]
    lh, rh = body_3d[:, B["left_hip"]], body_3d[:, B["right_hip"]]
    sh_mid, hip_mid = 0.5 * (ls + rs), 0.5 * (lh + rh)
    up = sh_mid - hip_mid
    torso = frame_from_axes(ls - rs, up)
    pelvis = frame_from_axes(lh - rh, up)
    return {"torso_R": torso, "torso_origin": sh_mid, "pelvis_R": pelvis, "pelvis_origin": hip_mid}


def to_frame(points: np.ndarray, R: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Express camera-frame points (T, ..., 3) in frame (R: T,3,3 columns=axes)."""
    extra = points.ndim - 2
    Rt = np.swapaxes(R, -1, -2).reshape(R.shape[0], *([1] * extra), 3, 3)
    o = origin.reshape(origin.shape[0], *([1] * extra), 3)
    return np.squeeze(Rt @ (points - o)[..., None], -1)


def rotate_into(R_cam: np.ndarray, frame_R: np.ndarray) -> np.ndarray:
    """Rotation of a camera-frame orientation, expressed in ``frame_R``."""
    return np.swapaxes(frame_R, -1, -2) @ R_cam


# --- rotation representations -------------------------------------------------

def matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """(…,3,3) -> (…,3) [roll, pitch, yaw] with R = Rz(yaw) Ry(pitch) Rx(roll)."""
    roll = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    pitch = np.arcsin(np.clip(-R[..., 2, 0], -1, 1))
    yaw = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    return np.stack([roll, pitch, yaw], axis=-1)


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """(…,3,3) -> (…,4) quaternion [x, y, z, w], sign-continuous over time axis 0."""
    m = R
    tr = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = np.stack([
        1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
        1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
        1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
        1 + tr,
    ], axis=-1)
    q = np.sqrt(np.clip(q, 0, None)) / 2
    q[..., 0] = np.copysign(q[..., 0], m[..., 2, 1] - m[..., 1, 2])
    q[..., 1] = np.copysign(q[..., 1], m[..., 0, 2] - m[..., 2, 0])
    q[..., 2] = np.copysign(q[..., 2], m[..., 1, 0] - m[..., 0, 1])
    q = unit(q)
    return make_quat_continuous(q)


def make_quat_continuous(q: np.ndarray) -> np.ndarray:
    q = q.copy()
    prev = None
    for t in range(q.shape[0]):
        if not np.all(np.isfinite(q[t])):
            continue
        if prev is not None and np.dot(prev, q[t]) < 0:
            q[t] = -q[t]
        prev = q[t]
    return q


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    xyz, w = q[..., :3], q[..., 3:]
    n = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2 * np.arctan2(n, w)
    return xyz / np.maximum(n, 1e-9) * angle


def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """First two columns of R, flattened (Zhou et al. 2019 continuous representation)."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


ROTATION_DIMS = {"euler": 3, "rotvec": 3, "quat": 4, "rot6d": 6}
ROTATION_NAMES = {
    "euler": ["roll", "pitch", "yaw"],
    "rotvec": ["rx", "ry", "rz"],
    "quat": ["qx", "qy", "qz", "qw"],
    "rot6d": [f"r6d_{i}" for i in range(6)],
}


def encode_rotation(R: np.ndarray, kind: str) -> np.ndarray:
    if kind == "euler":
        return unwrap(matrix_to_euler(R))
    if kind == "quat":
        return matrix_to_quat(R)
    if kind == "rotvec":
        return quat_to_rotvec(matrix_to_quat(R))
    if kind == "rot6d":
        return matrix_to_rot6d(R)
    raise KeyError(f"Unknown rotation encoding '{kind}'. Options: {sorted(ROTATION_DIMS)}")


def unwrap(angles: np.ndarray) -> np.ndarray:
    """np.unwrap along time that skips NaNs."""
    out = np.array(angles, dtype=np.float64, copy=True)
    flat = out.reshape(out.shape[0], -1)
    for c in range(flat.shape[1]):
        good = np.isfinite(flat[:, c])
        if good.sum() > 1:
            flat[good, c] = np.unwrap(flat[good, c])
    return flat.reshape(out.shape)
