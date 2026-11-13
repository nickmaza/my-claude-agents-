"""FastAPI server: upload footage, queue conversion jobs, edit, preview and download datasets."""

from __future__ import annotations

import hmac
import json
import os
import re
import shutil
import threading
import zipfile
from pathlib import Path

import numpy as np
from fastapi import Body, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..dataset import ACTION_MODES
from ..export import FORMATS
from ..pipeline import PipelineConfig, apply_edits
from ..retarget import EMBODIMENTS, GENERIC_EMBODIMENTS
from ..retarget.geometry import ROTATION_DIMS
from ..robots import ROBOTS
from ..schema import HumanMotion
from . import auth
from .auth import User, current_user, optional_user
from .store import JobStore
from .worker import run_worker

DATA_DIR = Path(os.environ.get("FOOTAGE2ROBOT_DATA_DIR", "f2r_data")).resolve()
MAX_UPLOAD_BYTES = int(float(os.environ.get("FOOTAGE2ROBOT_MAX_UPLOAD_MB", "2048")) * 1024 * 1024)
MAX_CLIPS = int(os.environ.get("FOOTAGE2ROBOT_MAX_CLIPS", "50"))
MAX_ACTIVE_JOBS = int(os.environ.get("FOOTAGE2ROBOT_MAX_ACTIVE_JOBS", "3"))
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
FRONTEND_DIST = Path(os.environ.get(
    "FOOTAGE2ROBOT_FRONTEND", Path(__file__).resolve().parents[3] / "frontend" / "dist"))
_ID = re.compile(r"^[0-9a-f]{12}$")
_CONFIG_KEYS = {
    "task", "target_fps", "start_s", "end_s", "embodiments", "formats", "action_mode", "rotation", "side",
    "scale", "smooth_s", "include_images", "image_size", "pose_model", "person_height_m", "arm_length_m",
    "marker_size_m", "fov_deg", "base_offset", "split", "track_objects", "object_backend",
}

app = FastAPI(title="footage2robot", version=__version__)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
store = JobStore(DATA_DIR)

# `footage2robot serve` runs separate worker processes and sets this to 0. Running the app
# directly (uvicorn, tests) gets one embedded worker thread so jobs still run.
if os.environ.get("FOOTAGE2ROBOT_EMBEDDED_WORKER", "1") != "0":
    threading.Thread(target=run_worker, args=(store,), daemon=True, name="f2r-worker").start()


def _owned(job_id: str, user: User) -> dict:
    job = store.get(job_id) if _ID.match(job_id) else None
    if job is None or job["owner"] != user.uid:      # same answer for "missing" and "not yours"
        raise HTTPException(404, "Job not found")
    return job


def _readable(job_id: str, user: User | None, key: str | None) -> dict:
    """File endpoints: the owner's token, or the job's access key in the URL (for <video>/<a>)."""
    job = store.get(job_id) if _ID.match(job_id) else None
    if job is not None:
        if user is not None and job["owner"] == user.uid:
            return job
        if key and hmac.compare_digest(key.encode(), job["access_key"].encode()):
            return job
    raise HTTPException(404, "Job not found")


def _config(raw: str) -> PipelineConfig:
    try:
        data = json.loads(raw or "{}")
        if not isinstance(data, dict):
            raise ValueError("config must be a JSON object")
        unknown = set(data) - _CONFIG_KEYS
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg = PipelineConfig(**data)
        cfg.image_size = tuple(int(v) for v in cfg.image_size)
        cfg.base_offset = tuple(float(v) for v in cfg.base_offset)
        cfg.task = str(cfg.task).strip()[:500] or "do the demonstrated task"
        if cfg.rotation not in ROTATION_DIMS or cfg.side not in ("auto", "left", "right") \
                or cfg.pose_model not in ("lite", "full", "heavy"):
            raise ValueError("Invalid rotation, side or pose_model")
        if len(cfg.image_size) != 2 or not all(16 <= v <= 1024 for v in cfg.image_size):
            raise ValueError("image_size must be [width, height] within 16..1024")
        if len(cfg.base_offset) != 3 or not all(abs(v) <= 5 for v in cfg.base_offset):
            raise ValueError("base_offset must be [x, y, z] within ±5 m")
        if cfg.target_fps is not None and not 1 <= float(cfg.target_fps) <= 240:
            raise ValueError("target_fps must be within 1..240")
        if not 0.01 <= float(cfg.scale) <= 100 or not 20 <= float(cfg.fov_deg) <= 140:
            raise ValueError("scale must be within 0.01..100 and fov_deg within 20..140")
        cfg.validate()
        return cfg
    except (ValueError, TypeError) as e:
        raise HTTPException(422, f"Invalid config: {e}") from e


def _check_quota(user: User) -> None:
    if store.active_count(user.uid) >= MAX_ACTIVE_JOBS:
        raise HTTPException(429, f"You already have {MAX_ACTIVE_JOBS} conversions in progress")


@app.get("/api/config")
def client_config() -> dict:
    """Public: tells the web app whether sign-in is required and how to reach Firebase."""
    return {"version": __version__, "auth_required": auth.project_id() is not None, "firebase": auth.web_config()}


@app.get("/api/options")
def options() -> dict:
    return {"version": __version__, "embodiments": EMBODIMENTS, "formats": FORMATS,
            "generic_embodiments": GENERIC_EMBODIMENTS, "robots": list(ROBOTS),
            "action_modes": ACTION_MODES, "rotations": list(ROTATION_DIMS),
            "video_extensions": sorted(VIDEO_EXTS), "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
            "max_clips": MAX_CLIPS}


@app.get("/api/marker.png")
def marker_png() -> Response:
    """Printable workspace marker."""
    import cv2

    from ..calibrate import marker_image
    ok, png = cv2.imencode(".png", marker_image())
    if not ok:
        raise HTTPException(500, "Could not render the marker")
    return Response(png.tobytes(), media_type="image/png",
                    headers={"Content-Disposition": 'inline; filename="footage2robot_marker.png"'})


@app.get("/api/me")
def me(user: User = Depends(current_user)) -> dict:
    return {"uid": user.uid, "email": user.email}


@app.get("/api/jobs")
def list_jobs(user: User = Depends(current_user)) -> list[dict]:
    return store.list(user.uid)


@app.post("/api/jobs", status_code=201)
async def create_job(files: list[UploadFile] = File(...), config: str = Form("{}"),
                     user: User = Depends(current_user)) -> dict:
    cfg = _config(config)
    _check_quota(user)
    if not files or len(files) > MAX_CLIPS:
        raise HTTPException(422, f"Upload between 1 and {MAX_CLIPS} videos")
    job_id = store.new_id()
    upload_dir = store.dir(job_id) / "uploads"
    paths, names, total = [], [], 0
    try:
        for i, up in enumerate(files):
            name = Path(up.filename or f"clip_{i}").name
            ext = Path(name).suffix.lower()
            if ext not in VIDEO_EXTS:
                raise HTTPException(415, f"'{name}': unsupported type. Allowed: {', '.join(sorted(VIDEO_EXTS))}")
            dest = upload_dir / f"clip_{i:03d}{ext}"     # never trust client filenames on disk
            with dest.open("wb") as out:
                while chunk := await up.read(1 << 20):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
                    out.write(chunk)
            paths.append(dest)
            names.append(name[:200])
    except BaseException:
        shutil.rmtree(store.dir(job_id), ignore_errors=True)
        raise
    return store.public(store.create(job_id, user.uid, paths, names, cfg))


@app.post("/api/jobs/demo", status_code=201)
def create_demo_job(config: str = Form("{}"), user: User = Depends(current_user)) -> dict:
    """Run the pipeline on procedurally generated motion (no footage needed)."""
    cfg = _config(config)
    _check_quota(user)
    cfg.extractor = "synthetic"
    cfg.task = "pick up the object (synthetic demo)"
    cfg.track_objects = False
    cfg.marker_size_m = None
    job_id = store.new_id()
    video = store.dir(job_id) / "uploads" / "synthetic"
    return store.public(store.create(job_id, user.uid, [video], ["synthetic demo"], cfg))


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, user: User = Depends(current_user)) -> dict:
    return store.public(_owned(job_id, user))


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str, user: User = Depends(current_user)) -> dict:
    _owned(job_id, user)
    if not store.delete(job_id):
        raise HTTPException(409, "Job is still running")
    return {"deleted": job_id}


@app.post("/api/jobs/{job_id}/edits")
def edit_job(job_id: str, body: dict = Body(...), user: User = Depends(current_user)) -> dict:
    """Change episode boundaries / the tracked object, then re-export without re-tracking.

    Body: ``{"clips": {"0": {"segments": [[start, end], ...], "object_prompt": [frame, x, y] | null}},
    "task": "..."}``. Frames are clip frames at the job's frame rate.
    """
    job = _owned(job_id, user)
    out = store.out_dir(job_id)
    if job["status"] in ("queued", "running"):
        raise HTTPException(409, "Job is still running")
    if not (out / "tracking.json").exists():
        raise HTTPException(409, "This job has no tracking to edit")
    clips = body.get("clips", {})
    if not isinstance(clips, dict):
        raise HTTPException(422, "clips must be an object keyed by clip index")
    try:
        apply_edits(out, clips)
    except (ValueError, TypeError, KeyError) as e:
        raise HTTPException(422, f"Invalid edit: {e}") from e
    config = dict(job["config"])
    if any(isinstance(c, dict) and c.get("object_prompt") for c in clips.values()):
        config["track_objects"] = True
    if isinstance(body.get("task"), str) and body["task"].strip():
        config["task"] = body["task"].strip()[:500]
    if config != job["config"]:
        payload = {k: job[k] for k in ("clips", "videos", "access_key")}
        payload.update(task=config["task"], config=config)
        store.update(job_id, payload=json.dumps(payload))
    if not store.request_rebuild(job_id):
        raise HTTPException(409, "Job is still running")
    return store.public(store.get(job_id))


@app.get("/api/jobs/{job_id}/clips/{clip}/video")
def clip_video(job_id: str, clip: int, key: str | None = None, user: User | None = Depends(optional_user)):
    job = _readable(job_id, user, key)
    if not 0 <= clip < len(job["videos"]) or not Path(job["videos"][clip]).is_file():
        raise HTTPException(404, "Clip has no video")
    return FileResponse(job["videos"][clip])


@app.get("/api/jobs/{job_id}/clips/{clip}/overlay")
def clip_overlay(job_id: str, clip: int, user: User = Depends(current_user)):
    job = _owned(job_id, user)
    out = store.out_dir(job_id)
    path = out / "motion" / f"episode_{clip:03d}.npz"
    if not 0 <= clip < len(job["videos"]) or not path.is_file():
        raise HTTPException(404, "Tracking not available yet")
    motion = HumanMotion.load(path)
    data = motion.to_overlay_json()
    data["start_s"] = motion.meta.get("start_s", 0.0)
    tracking = json.loads((out / "tracking.json").read_text())["clips"][clip]
    data["segments"] = tracking["segments"]
    data["object_prompt"] = tracking.get("object_prompt")
    obj = out / "motion" / f"object_{clip:03d}.npz"
    data["object"] = None
    if obj.is_file() and tracking.get("object_prompt"):
        with np.load(obj) as z:
            data["object"] = np.round(z["boxes"].astype(np.float64), 4).tolist()
    return JSONResponse(data)


@app.get("/api/jobs/{job_id}/datasets/{embodiment}/preview")
def dataset_preview(job_id: str, embodiment: str, user: User = Depends(current_user)):
    _owned(job_id, user)
    if embodiment not in EMBODIMENTS:
        raise HTTPException(404, "Unknown embodiment")
    path = store.out_dir(job_id) / "datasets" / embodiment / "preview.json"
    if not path.is_file():
        raise HTTPException(404, "No preview for this embodiment")
    return FileResponse(path, media_type="application/json")


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str, embodiment: str | None = None, format: str | None = None,
             key: str | None = None, user: User | None = Depends(optional_user)):
    """Zip of one export (embodiment + format), one embodiment, or everything."""
    job = _readable(job_id, user, key)
    if job["status"] != "done":
        raise HTTPException(409, "Job has not finished")
    if (embodiment and embodiment not in EMBODIMENTS) or (format and format not in FORMATS):
        raise HTTPException(404, "Unknown embodiment or format")
    if format and not embodiment:
        raise HTTPException(422, "format requires embodiment")
    base = store.out_dir(job_id)
    src = base / "datasets"
    label = "all"
    if embodiment:
        src, label = src / embodiment, embodiment
        if format:
            src, label = src / format, f"{embodiment}_{format}"
    if not src.is_dir():
        raise HTTPException(404, "Export not found")
    zip_dir = store.dir(job_id) / "zips"
    zip_dir.mkdir(exist_ok=True)
    zip_path = zip_dir / f"{label}.zip"
    if not zip_path.exists():
        tmp = zip_path.with_suffix(".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:   # payloads are already compressed
            for f in sorted(src.rglob("*")):
                if f.is_file() and f.name != "preview.json":
                    z.write(f, f.relative_to(src.parent if embodiment else base))
            if not embodiment:
                summary = dict(job["summary"] or {})
                z.writestr("summary.json", json.dumps(summary, indent=2))
        tmp.replace(zip_path)
    return FileResponse(zip_path, media_type="application/zip", filename=f"footage2robot_{job_id}_{label}.zip")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "version": __version__}


if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
