"""Video reading helpers built on OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    num_frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps if self.fps else 0.0


def probe(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        return VideoInfo(
            path=str(path), fps=float(fps),
            num_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def frame_stride(source_fps: float, target_fps: float | None) -> int:
    """Keep every ``stride``-th frame so the output rate is <= target_fps."""
    if not target_fps or target_fps >= source_fps:
        return 1
    return max(1, int(round(source_fps / target_fps)))


def iter_frames(path: str | Path, stride: int = 1, start_s: float = 0.0,
                end_s: float | None = None) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield ``(source_frame_index, timestamp_s, rgb_frame)`` tuples."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        idx = 0
        if start_s > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
            idx = int(round(start_s * fps))
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t = idx / fps
            if end_s is not None and t > end_s:
                break
            if (idx - int(round(start_s * fps))) % stride == 0:
                yield idx, t, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            idx += 1
    finally:
        cap.release()


def resize_frame(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Letterbox-free resize to (width, height)."""
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


BROWSER_SAFE_EXTS = {".mp4"}


def needs_normalizing(path: str | Path) -> bool:
    """True for containers/codecs that are unreliable to index or to play back in a browser.

    Browser recordings (WebM from MediaRecorder) have no duration and a variable
    frame rate, and phone .mov files are often HEVC, which most browsers cannot
    play. Both are re-encoded to constant-frame-rate H.264 before processing.
    """
    import shutil
    import subprocess

    path = Path(path)
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        return False
    if path.suffix.lower() not in BROWSER_SAFE_EXTS:
        return True
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name,r_frame_rate,avg_frame_rate,nb_frames", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60, check=True).stdout
        import json
        stream = json.loads(out)["streams"][0]
    except Exception:
        return False

    def rate(v: str) -> float:
        a, _, b = v.partition("/")
        return float(a) / float(b or 1) if float(b or 1) else 0.0

    r, avg = rate(stream.get("r_frame_rate", "0")), rate(stream.get("avg_frame_rate", "0"))
    variable = avg <= 0 or abs(r - avg) / max(avg, 1e-6) > 0.02
    return stream.get("codec_name") != "h264" or variable or not str(stream.get("nb_frames", "")).isdigit()


def normalize(src: str | Path, dst: str | Path, max_fps: float = 60.0) -> Path:
    """Re-encode to constant-frame-rate H.264/yuv420p MP4 (no audio)."""
    import subprocess

    src, dst = Path(src), Path(dst)
    fps = 30.0
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=avg_frame_rate", "-of", "csv=p=0", str(src)],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        a, _, b = out.partition("/")
        value = float(a) / float(b or 1)
        if 1.0 <= value <= 240.0:
            fps = min(value, max_fps)
    except (ValueError, ZeroDivisionError, subprocess.SubprocessError):
        pass
    tmp = dst.with_name(dst.stem + ".part.mp4")
    proc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-an", "-vf",
         f"fps={fps:.4f},scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)],
        capture_output=True, text=True)
    if proc.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise ValueError(f"Could not read video: {proc.stderr.strip()[-300:] or 'ffmpeg failed'}")
    tmp.replace(dst)
    return dst
