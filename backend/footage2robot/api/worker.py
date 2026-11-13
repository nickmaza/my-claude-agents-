"""Worker loop: claims queued jobs from the store and runs the pipeline."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import traceback

from ..pipeline import PipelineConfig, build, run_pipeline
from .store import JobStore


def _config(data: dict) -> PipelineConfig:
    cfg = PipelineConfig(**data)
    cfg.image_size = tuple(cfg.image_size)
    cfg.base_offset = tuple(cfg.base_offset)
    return cfg


def _prepare_videos(store: JobStore, job: dict, progress) -> list[str]:
    """Re-encode clips that are unreliable to index or play (browser WebM, HEVC .mov, ...)."""
    from pathlib import Path

    from ..video import needs_normalizing, normalize
    videos = list(job["videos"])
    changed = False
    for i, video in enumerate(videos):
        src = Path(video)
        if src.is_file() and ".norm" not in src.name and needs_normalizing(src):
            progress(0.0, f"Preparing clip {i + 1}/{len(videos)}")
            videos[i] = str(normalize(src, src.with_name(f"{src.stem}.norm.mp4")))
            src.unlink(missing_ok=True)
            changed = True
    if changed:
        payload = {k: job[k] for k in ("task", "clips", "config", "access_key")}
        payload["videos"] = videos
        store.update(job["id"], payload=json.dumps(payload))
    return videos


def process(store: JobStore, job: dict) -> None:
    job_id = job["id"]
    last = [0.0]

    def progress(p: float, msg: str) -> None:
        now = time.time()
        if now - last[0] >= 0.3 or p >= 1.0:      # keep database writes modest
            last[0] = now
            store.update(job_id, progress=round(p, 4), message=msg, heartbeat=now)

    stop = threading.Event()

    def heartbeat() -> None:                      # long silent steps must not look like a dead worker
        while not stop.wait(15):
            store.update(job_id, heartbeat=time.time())

    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        cfg = _config(job["config"])
        out = store.out_dir(job_id)
        if job["stage"] == "rebuild" and (out / "tracking.json").exists():
            summary = build(out, cfg, progress)
        else:
            videos = _prepare_videos(store, job, progress)
            summary = run_pipeline(videos, out, cfg, progress)
        for clip, name in zip(summary["clips"], job["clips"]):
            clip["name"] = name
            clip.pop("video", None)
        summary["config"].pop("name", None)
        ok = any(d["exports"] for d in summary["datasets"])
        store.update(job_id, status="done" if ok else "failed", progress=1.0,
                     message="Done" if ok else "No dataset could be produced",
                     error=None if ok else ("; ".join(summary["warnings"]) or "No usable tracking"),
                     summary=summary, finished_at=time.time())
    except Exception as e:
        (store.dir(job_id) / "error.txt").write_text(traceback.format_exc())
        store.update(job_id, status="failed", error=f"{type(e).__name__}: {e}", message="Failed",
                     finished_at=time.time())
    finally:
        stop.set()


def run_worker(store: JobStore, stop: threading.Event | None = None, poll_s: float = 0.5) -> None:
    worker = f"{socket.gethostname()}:{os.getpid()}:{threading.get_ident()}"
    stop = stop or threading.Event()
    while not stop.is_set():
        job = store.claim(worker)
        if job is None:
            stop.wait(poll_s)
            continue
        process(store, job)
