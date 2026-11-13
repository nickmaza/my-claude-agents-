"""SQLite-backed job table and queue, shared by the API server and worker processes."""

from __future__ import annotations

import json
import secrets
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from ..pipeline import PipelineConfig

STALE_AFTER_S = 120.0       # a running job whose worker stopped reporting is re-queued
ACTIVE = ("queued", "running")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    status TEXT NOT NULL,           -- queued | running | done | failed
    stage TEXT NOT NULL,            -- convert | rebuild
    progress REAL NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    queued_at REAL NOT NULL,
    finished_at REAL,
    heartbeat REAL,
    worker TEXT,
    error TEXT,
    payload TEXT NOT NULL,          -- JSON: task, clips, videos, config, access_key
    summary TEXT                    -- JSON pipeline summary
);
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs (status, queued_at);
CREATE INDEX IF NOT EXISTS jobs_owner ON jobs (owner, created_at);
"""


class JobStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "jobs").mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "jobs.db"
        with self._db() as db:
            db.executescript(_SCHEMA)

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA journal_mode=WAL")
            yield db
        finally:
            db.close()

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

    def create(self, job_id: str, owner: str, videos: list[Path], names: list[str], cfg: PipelineConfig) -> dict:
        now = time.time()
        payload = {"task": cfg.task, "clips": names, "videos": [str(v) for v in videos],
                   "config": asdict(cfg), "access_key": secrets.token_urlsafe(24)}
        with self._db() as db:
            db.execute("INSERT INTO jobs (id, owner, status, stage, message, created_at, queued_at, payload) "
                       "VALUES (?, ?, 'queued', 'convert', 'Queued', ?, ?, ?)",
                       (job_id, owner, now, now, json.dumps(payload)))
        return self.get(job_id)

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        job = dict(row)
        job.update(json.loads(job.pop("payload")))
        job["summary"] = json.loads(job["summary"]) if job["summary"] else None
        job["embodiments"] = job["config"]["embodiments"]
        job["formats"] = job["config"]["formats"]
        return job

    def get(self, job_id: str) -> dict | None:
        with self._db() as db:
            return self._row(db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())

    def list(self, owner: str) -> list[dict]:
        with self._db() as db:
            rows = db.execute("SELECT * FROM jobs WHERE owner = ? ORDER BY created_at DESC LIMIT 200", (owner,)).fetchall()
        return [self.public(self._row(r), brief=True) for r in rows]

    def active_count(self, owner: str) -> int:
        with self._db() as db:
            return db.execute("SELECT COUNT(*) FROM jobs WHERE owner = ? AND status IN ('queued', 'running')",
                              (owner,)).fetchone()[0]

    def delete(self, job_id: str) -> bool:
        with self._db() as db:
            cur = db.execute("DELETE FROM jobs WHERE id = ? AND status NOT IN ('queued', 'running')", (job_id,))
        if cur.rowcount:
            shutil.rmtree(self.dir(job_id), ignore_errors=True)
        return bool(cur.rowcount)

    @staticmethod
    def public(job: dict, brief: bool = False) -> dict:
        hidden = {"videos", "worker", "heartbeat", "queued_at", "owner", "config"}
        out = {k: v for k, v in job.items() if k not in hidden}
        if brief:
            out.pop("summary", None)
            out.pop("access_key", None)
        return out

    def update(self, job_id: str, **fields) -> None:
        if "summary" in fields and fields["summary"] is not None:
            fields["summary"] = json.dumps(fields["summary"])
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._db() as db:
            db.execute(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))

    # -- queue -----------------------------------------------------------
    def request_rebuild(self, job_id: str) -> bool:
        """Queue a finished job for re-export (after edits). False if it is busy."""
        with self._db() as db:
            cur = db.execute(
                "UPDATE jobs SET status = 'queued', stage = 'rebuild', progress = 0, message = 'Queued', "
                "error = NULL, queued_at = ? WHERE id = ? AND status NOT IN ('queued', 'running')",
                (time.time(), job_id))
        if cur.rowcount:
            shutil.rmtree(self.dir(job_id) / "zips", ignore_errors=True)
        return bool(cur.rowcount)

    def claim(self, worker: str) -> dict | None:
        """Atomically take the oldest queued job (re-queuing any whose worker died)."""
        now = time.time()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute("UPDATE jobs SET status = 'queued', message = 'Re-queued after a worker stopped' "
                           "WHERE status = 'running' AND heartbeat < ?", (now - STALE_AFTER_S,))
                row = db.execute("SELECT id FROM jobs WHERE status = 'queued' ORDER BY queued_at LIMIT 1").fetchone()
                if row is not None:
                    db.execute("UPDATE jobs SET status = 'running', worker = ?, heartbeat = ?, message = 'Starting' "
                               "WHERE id = ?", (worker, now, row["id"]))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return self.get(row["id"]) if row is not None else None
