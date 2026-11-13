"""FastAPI server: upload footage, run conversion jobs, preview and download datasets."""

from __future__ import annotations

import json
import os
import re
import shutil
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..dataset import ACTION_MODES
from ..export import FORMATS
from ..pipeline import PipelineConfig
from ..retarget import EMBODIMENTS
from ..retarget.geometry import ROTATION_DIMS
from ..schema import HumanMotion
from .jobs import JobStore

DATA_DIR = Path(os.environ.get("FOOTAGE2ROBOT_DATA_DIR", "f2r_data")).resolve()
MAX_UPLOAD_BYTES = int(float(os.environ.get("FOOTAGE2ROBOT_MAX_UPLOAD_MB", "2048")) * 1024 * 1024)
MAX_CLIPS = int(os.environ.get("FOOTAGE2ROBOT_MAX_CLIPS", "50"))
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
FRONTEND_DIST = Path(os.environ.get(
    "FOOTAGE2ROBOT_FRONTEND", Path(__file__).resolve().parents[3] / "frontend" / "dist"))
_ID = re.compile(r"^[0-9a-f]{12}$")

app = FastAPI(title="footage2robot", version=__version__)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
store = JobStore(DATA_DIR)


def _job(job_id: str) -> dict:
    job = store.get(job_id) if _ID.match(job_id) else None
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


def _config(raw: str) -> PipelineConfig:
    try:
        data = json.loads(raw or "{}")
        if not isinstance(data, dict):
            raise ValueError("config must be a JSON object")
        allowed = {"task", "target_fps", "start_s", "end_s", "embodiments", "formats", "action_mode",
                   "rotation", "side", "scale", "smooth_s", "include_images", "image_size", "pose_model"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg = PipelineConfig(**data)
        cfg.image_size = tuple(int(v) for v in cfg.image_size)
        cfg.task = str(cfg.task).strip()[:500] or "do the demonstrated task"
        if cfg.rotation not in ROTATION_DIMS or cfg.side not in ("auto", "left", "right") \
                or cfg.pose_model not in ("lite", "full", "heavy"):
            raise ValueError("Invalid rotation, side or pose_model")
        if len(cfg.image_size) != 2 or not all(16 <= v <= 1024 for v in cfg.image_size):
            raise ValueError("image_size must be [width, height] within 16..1024")
        if cfg.target_fps is not None and not 1 <= float(cfg.target_fps) <= 240:
            raise ValueError("target_fps must be within 1..240")
        if not 0.01 <= float(cfg.scale) <= 100:
            raise ValueError("scale must be within 0.01..100")
        cfg.validate()
        return cfg
    except (ValueError, TypeError) as e:
        raise HTTPException(422, f"Invalid config: {e}") from e


@app.get("/api/options")
def options() -> dict:
    return {"version": __version__, "embodiments": EMBODIMENTS, "formats": FORMATS,
            "action_modes": ACTION_MODES, "rotations": list(ROTATION_DIMS),
            "video_extensions": sorted(VIDEO_EXTS), "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
            "defaults": {"target_fps": 30, "action_mode": "absolute", "rotation": "euler",
                         "side": "auto", "image_size": [224, 224]}}


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return store.list()


@app.post("/api/jobs", status_code=201)
async def create_job(files: list[UploadFile] = File(...), config: str = Form("{}")) -> dict:
    cfg = _config(config)
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
    return store.create(job_id, paths, names, cfg)


@app.post("/api/jobs/demo", status_code=201)
def create_demo_job(config: str = Form("{}")) -> dict:
    """Run the pipeline on procedurally generated motion (no footage needed)."""
    cfg = _config(config)
    cfg.extractor = "synthetic"
    cfg.task = "pick up the object (synthetic demo)"
    job_id = store.new_id()
    return store.create(job_id, [store.dir(job_id) / "uploads" / "synthetic"], ["synthetic demo"], cfg)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return store.public(_job(job_id))


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    _job(job_id)
    if not store.delete(job_id):
        raise HTTPException(409, "Job is still running")
    return {"deleted": job_id}


@app.get("/api/jobs/{job_id}/clips/{clip}/video")
def clip_video(job_id: str, clip: int):
    job = _job(job_id)
    if not 0 <= clip < len(job["videos"]) or not Path(job["videos"][clip]).is_file():
        raise HTTPException(404, "Clip has no video")
    return FileResponse(job["videos"][clip])


@app.get("/api/jobs/{job_id}/clips/{clip}/overlay")
def clip_overlay(job_id: str, clip: int):
    job = _job(job_id)
    path = store.out_dir(job_id) / "motion" / f"episode_{clip:03d}.npz"
    if not 0 <= clip < len(job["videos"]) or not path.is_file():
        raise HTTPException(404, "Tracking not available yet")
    motion = HumanMotion.load(path)
    data = motion.to_overlay_json()
    data["start_s"] = motion.meta.get("start_s", 0.0)
    return JSONResponse(data)


@app.get("/api/jobs/{job_id}/datasets/{embodiment}/preview")
def dataset_preview(job_id: str, embodiment: str):
    _job(job_id)
    if embodiment not in EMBODIMENTS:
        raise HTTPException(404, "Unknown embodiment")
    path = store.out_dir(job_id) / "datasets" / embodiment / "preview.json"
    if not path.is_file():
        raise HTTPException(404, "No preview for this embodiment")
    return FileResponse(path, media_type="application/json")


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str, embodiment: str | None = None, format: str | None = None):
    """Zip of one export (embodiment + format), one embodiment, or everything."""
    job = _job(job_id)
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
                z.write(base / "summary.json", "summary.json")
        tmp.replace(zip_path)
    return FileResponse(zip_path, media_type="application/zip", filename=f"footage2robot_{job_id}_{label}.zip")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "version": __version__}


if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
