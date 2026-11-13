"""Filesystem-backed job store and a background worker for pipeline runs."""

from __future__ import annotations

import json
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..pipeline import PipelineConfig, run_pipeline


class JobStore:
    def __init__(self, root: Path, workers: int = 1):
        self.root = Path(root)
        (self.root / "jobs").mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="f2r-job")
        for f in sorted((self.root / "jobs").glob("*/job.json")):
            try:
                job = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if job["status"] in ("queued", "running"):   # server restarted mid-run
                job.update(status="failed", error="Interrupted by a server restart")
                f.write_text(json.dumps(job, indent=2))
            self._jobs[job["id"]] = job

    # -- paths -----------------------------------------------------------
    def dir(self, job_id: str) -> Path:
        return self.root / "jobs" / job_id

    def out_dir(self, job_id: str) -> Path:
        return self.dir(job_id) / "out"

    # -- CRUD ------------------------------------------------------------
    def new_id(self) -> str:
        job_id = uuid.uuid4().hex[:12]
        (self.dir(job_id) / "uploads").mkdir(parents=True)
        return job_id

    def create(self, job_id: str, videos: list[Path], names: list[str], cfg: PipelineConfig) -> dict:
        job = {
            "id": job_id, "status": "queued", "progress": 0.0, "message": "Queued",
            "created_at": time.time(), "finished_at": None, "error": None,
            "task": cfg.task, "clips": names, "videos": [str(v) for v in videos],
            "embodiments": cfg.embodiments, "formats": cfg.formats, "summary": None,
        }
        with self._lock:
            self._jobs[job_id] = job
        self._save(job)
        self._pool.submit(self._run, job_id, cfg)
        return self.public(job)

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def list(self) -> list[dict]:
        with self._lock:
            jobs = [dict(j) for j in self._jobs.values()]
        return [self.public(j, brief=True) for j in sorted(jobs, key=lambda j: -j["created_at"])]

    def delete(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job["status"] in ("queued", "running"):
                return False
            del self._jobs[job_id]
        shutil.rmtree(self.dir(job_id), ignore_errors=True)
        return True

    @staticmethod
    def public(job: dict, brief: bool = False) -> dict:
        out = {k: v for k, v in job.items() if k != "videos"}
        if brief:
            out.pop("summary", None)
        return out

    # -- worker ----------------------------------------------------------
    def _update(self, job_id: str, persist: bool = True, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.update(fields)
            snapshot = dict(job)
        if persist:
            self._save(snapshot)

    def _save(self, job: dict) -> None:
        path = self.dir(job["id"]) / "job.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, indent=2))
        tmp.replace(path)

    def _run(self, job_id: str, cfg: PipelineConfig) -> None:
        job = self.get(job_id)
        if job is None:
            return
        self._update(job_id, status="running", message="Starting")
        try:
            summary = run_pipeline(
                job["videos"], self.out_dir(job_id), cfg,
                progress=lambda p, msg: self._update(job_id, persist=False, progress=round(p, 4), message=msg))
            for clip, name in zip(summary["clips"], job["clips"]):
                clip["name"] = name
                clip.pop("video", None)
            summary["config"].pop("name", None)
            ok = any(d["exports"] for d in summary["datasets"])
            self._update(job_id, status="done" if ok else "failed", progress=1.0,
                         message="Done" if ok else "No dataset could be produced",
                         error=None if ok else "; ".join(summary["warnings"]) or "No usable tracking",
                         summary=summary, finished_at=time.time())
        except Exception as e:
            (self.dir(job_id) / "error.txt").write_text(traceback.format_exc())
            self._update(job_id, status="failed", error=f"{type(e).__name__}: {e}",
                         message="Failed", finished_at=time.time())
