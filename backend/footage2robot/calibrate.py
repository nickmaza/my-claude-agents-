"""Metric calibration: scale from a body measurement, and a workspace frame from a marker.

Monocular trackers return a plausible-but-arbitrary scale. Two cheap fixes:

* **Body measurement** — the user gives their height or (more direct) their
  shoulder-to-wrist arm length; the tracked skeleton is scaled to match.
* **Printed marker** — an ArUco marker of known size lying in the workspace
  defines a fixed, metric world frame (x/y along the marker edges, z out of
  the marker), so trajectories no longer depend on where the person stands.

Camera intrinsics are not known for arbitrary footage, so a pinhole camera with
a configurable field of view is assumed. That bounds the accuracy of the marker
frame; the body-measurement scale does not depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import B, HumanMotion

DEFAULT_FOV_DEG = 65.0      # field of view along the image's long side (typical phone main camera)
MARKER_DICT = "DICT_4X4_50"
MARKER_ID = 0

# Fractions of stature (Drillis & Contini), applied to tracked joint-centre chains.
ARM_FRACTION = 0.332        # shoulder -> elbow -> wrist
TORSO_FRACTION = 0.288      # hip centre -> shoulder centre
SHOULDER_HEIGHT_FRACTION = 0.818
ANKLE_HEIGHT_M = 0.07


def _seg(body: np.ndarray, a: str, b: str) -> float:
    d = np.linalg.norm(body[:, B[a]] - body[:, B[b]], axis=-1)
    return float(np.nanmedian(d)) if np.isfinite(d).any() else float("nan")


def arm_length(motion: HumanMotion) -> float:
    """Median tracked shoulder->elbow->wrist length (both arms averaged), in tracker units."""
    vals = [_seg(motion.body_3d, f"{s}_shoulder", f"{s}_elbow") + _seg(motion.body_3d, f"{s}_elbow", f"{s}_wrist")
            for s in ("left", "right")]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def estimate_height(motion: HumanMotion) -> float:
    """Stature implied by the tracked skeleton, in tracker units (NaN if nothing usable)."""
    b = motion.body_3d
    torso = np.linalg.norm(0.5 * (b[:, 11] + b[:, 12]) - 0.5 * (b[:, 23] + b[:, 24]), axis=-1)
    torso = float(np.nanmedian(torso)) if np.isfinite(torso).any() else float("nan")
    estimates = [arm_length(motion) / ARM_FRACTION, torso / TORSO_FRACTION]
    legs = [_seg(b, f"{s}_hip", f"{s}_knee") + _seg(b, f"{s}_knee", f"{s}_ankle") for s in ("left", "right")]
    legs = [v for v in legs if np.isfinite(v)]
    if legs and np.isfinite(torso):
        estimates.append((ANKLE_HEIGHT_M + float(np.mean(legs)) + torso) / SHOULDER_HEIGHT_FRACTION)
    estimates = [e for e in estimates if np.isfinite(e)]
    return float(np.median(estimates)) if estimates else float("nan")


def body_scale(motion: HumanMotion, person_height_m: float | None = None,
               arm_length_m: float | None = None) -> tuple[float, dict]:
    """Scale factor that makes the tracked skeleton match the given measurement."""
    height, arm = estimate_height(motion), arm_length(motion)
    info: dict = {"tracked_height_m": round(height, 3), "tracked_arm_length_m": round(arm, 3)}
    scale = 1.0
    if arm_length_m and np.isfinite(arm) and arm > 0:
        scale, info["method"] = arm_length_m / arm, "arm_length"
    elif person_height_m and np.isfinite(height) and height > 0:
        scale, info["method"] = person_height_m / height, "height"
    scale = float(np.clip(scale, 0.5, 2.0))     # a wilder factor means bad tracking, not a tiny person
    info["scale"] = round(scale, 4)
    return scale, info


def apply_scale(motion: HumanMotion, scale: float) -> None:
    """Scale all 3D landmarks in place."""
    if scale != 1.0:
        motion.body_3d = motion.body_3d * scale
        motion.hands_3d = motion.hands_3d * scale


# --- camera model -----------------------------------------------------------

@dataclass
class Camera:
    width: int
    height: int
    fov_deg: float = DEFAULT_FOV_DEG

    @property
    def f(self) -> float:
        return 0.5 * max(self.width, self.height) / np.tan(np.deg2rad(self.fov_deg) / 2)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.f, 0, self.width / 2], [0, self.f, self.height / 2], [0, 0, 1]], np.float64)

    def project(self, pts: np.ndarray) -> np.ndarray:
        """Camera-frame points (..., 3) -> normalised image coords (..., 2)."""
        z = pts[..., 2]
        return np.stack([(self.f * pts[..., 0] / z + self.width / 2) / self.width,
                         (self.f * pts[..., 1] / z + self.height / 2) / self.height], axis=-1)


def root_translation(motion: HumanMotion, cam: Camera, min_conf: float = 0.5) -> np.ndarray:
    """Camera-frame position of the body's local origin for every frame, (T, 3).

    The tracker's 3D landmarks are camera-aligned but centred on the hips. With
    the 2D landmarks and a camera model, the missing translation follows from a
    linear least-squares fit per frame (rotation is already known).
    """
    T = motion.num_frames
    out = np.full((T, 3), np.nan)
    f, cx, cy = cam.f, cam.width / 2, cam.height / 2
    for t in range(T):
        P = motion.body_3d[t].astype(np.float64)
        uv = motion.body_2d[t].astype(np.float64) * [cam.width, cam.height]
        ok = np.isfinite(P).all(1) & np.isfinite(uv).all(1) & (motion.body_conf[t] >= min_conf)
        if ok.sum() < 6:
            continue
        P, uv = P[ok], uv[ok]
        du, dv = uv[:, 0] - cx, uv[:, 1] - cy
        n = len(P)
        A = np.zeros((2 * n, 3))
        A[:n, 0], A[:n, 2] = f, -du
        A[n:, 1], A[n:, 2] = f, -dv
        rhs = np.concatenate([du * P[:, 2] - f * P[:, 0], dv * P[:, 2] - f * P[:, 1]])
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        if sol[2] > 0.2:
            out[t] = sol
    return out


# --- marker -----------------------------------------------------------------

def marker_image(size_px: int = 800, border_px: int = 80) -> np.ndarray:
    """Printable marker (white border included), uint8 grayscale."""
    import cv2

    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, MARKER_DICT))
    img = cv2.aruco.generateImageMarker(d, MARKER_ID, size_px)
    return cv2.copyMakeBorder(img, border_px, border_px, border_px, border_px, cv2.BORDER_CONSTANT, value=255)


def find_marker(video_path: str, marker_size_m: float, cam: Camera | None = None,
                max_samples: int = 40) -> dict | None:
    """Median pose of the workspace marker in the camera frame.

    Returns ``{"R": 3x3 (marker axes as columns), "t": 3, "detections": n}`` or
    None if the marker is not seen. Assumes a static camera.
    """
    import cv2

    from .video import iter_frames, probe

    info = probe(video_path)
    cam = cam or Camera(info.width, info.height)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX     # sub-pixel corners: far better tilt
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, MARKER_DICT)), params)
    h = marker_size_m / 2
    obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)
    stride = max(1, info.num_frames // max_samples)
    Rs, ts = [], []
    for _, _, rgb in iter_frames(video_path, stride):
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        if ids is None:
            continue
        for c, i in zip(corners, ids.flatten()):
            if i != MARKER_ID:
                continue
            # a planar square has two candidate poses; keep the one that reprojects best
            n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, c.reshape(4, 2).astype(np.float64), cam.K, None,
                                                        flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if n:
                best = int(np.argmin(np.asarray(errs).reshape(-1)))
                Rs.append(cv2.Rodrigues(rvecs[best])[0])
                ts.append(np.asarray(tvecs[best]).reshape(3))
    if not Rs:
        return None
    t = np.median(np.array(ts), axis=0)
    # average rotation: project the mean matrix back onto SO(3)
    u, _, vt = np.linalg.svd(np.mean(np.array(Rs), axis=0))
    R = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    return {"R": R.tolist(), "t": t.tolist(), "detections": len(Rs)}
