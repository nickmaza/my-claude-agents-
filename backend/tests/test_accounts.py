"""Sign-in, per-user isolation, the job queue and edits through the API."""
import io
import json
import threading
import time
import zipfile

import pytest

from footage2robot.api import auth
from footage2robot.api.store import STALE_AFTER_S, JobStore
from footage2robot.pipeline import PipelineConfig

CFG = {"embodiments": ["single_arm"], "formats": ["json"], "include_images": False}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    os.environ["FOOTAGE2ROBOT_DATA_DIR"] = str(tmp_path_factory.mktemp("data"))
    from fastapi.testclient import TestClient

    from footage2robot.api.app import app
    return TestClient(app)


@pytest.fixture
def signed_in(monkeypatch):
    """Require sign-in; tokens 'tok-<name>' are accepted as user <name>."""
    monkeypatch.setenv("FOOTAGE2ROBOT_FIREBASE_PROJECT", "demo-project")

    def fake_verify(token):
        if not token.startswith("tok-"):
            raise ValueError("bad token")
        return auth.User(uid=token[4:], email=f"{token[4:]}@example.com")
    monkeypatch.setattr(auth, "verify_token", fake_verify)


def hdr(name):
    return {"Authorization": f"Bearer tok-{name}"}


def wait(client, job_id, headers=None, timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}", headers=headers).json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.1)
    raise TimeoutError(job)


def test_config_endpoint(client, monkeypatch):
    assert client.get("/api/config").json()["auth_required"] is False
    monkeypatch.setenv("FOOTAGE2ROBOT_FIREBASE_PROJECT", "demo-project")
    monkeypatch.setenv("FOOTAGE2ROBOT_FIREBASE_WEB_CONFIG", json.dumps({"apiKey": "k", "projectId": "demo-project"}))
    cfg = client.get("/api/config").json()
    assert cfg["auth_required"] is True and cfg["firebase"]["projectId"] == "demo-project"


def test_sign_in_required(client, signed_in):
    assert client.get("/api/jobs").status_code == 401
    assert client.get("/api/jobs", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.post("/api/jobs/demo", data={"config": json.dumps(CFG)}).status_code == 401
    assert client.get("/api/me", headers=hdr("alice")).json() == {"uid": "alice", "email": "alice@example.com"}
    assert client.get("/api/options").status_code == 200           # public
    assert client.get("/api/marker.png").headers["content-type"] == "image/png"


def test_users_cannot_see_each_others_jobs(client, signed_in):
    r = client.post("/api/jobs/demo", data={"config": json.dumps(CFG)}, headers=hdr("alice"))
    assert r.status_code == 201
    jid = r.json()["id"]
    job = wait(client, jid, hdr("alice"))
    assert job["status"] == "done" and "owner" not in job

    assert [j["id"] for j in client.get("/api/jobs", headers=hdr("alice")).json()] == [jid]
    assert client.get("/api/jobs", headers=hdr("bob")).json() == []
    for path in (f"/api/jobs/{jid}", f"/api/jobs/{jid}/clips/0/overlay", f"/api/jobs/{jid}/datasets/single_arm/preview",
                 f"/api/jobs/{jid}/download"):
        assert client.get(path, headers=hdr("bob")).status_code == 404, path
        assert client.get(path).status_code in (401, 404), path
    assert client.delete(f"/api/jobs/{jid}", headers=hdr("bob")).status_code == 404
    assert client.post(f"/api/jobs/{jid}/edits", json={"clips": {}}, headers=hdr("bob")).status_code == 404

    # file links work without a header only with the job's access key
    key = job["access_key"]
    assert "access_key" not in client.get("/api/jobs", headers=hdr("alice")).json()[0]
    assert client.get(f"/api/jobs/{jid}/download?key=wrong").status_code == 404
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/jobs/{jid}/download?key={key}").content))
    assert "summary.json" in z.namelist()
    assert "/uploads/" not in z.read("summary.json").decode()       # no server paths in downloads
    assert client.delete(f"/api/jobs/{jid}", headers=hdr("alice")).status_code == 200


def test_edit_and_rebuild(client):
    r = client.post("/api/jobs/demo", data={"config": json.dumps(CFG)})
    jid = r.json()["id"]
    job = wait(client, jid)
    frames = job["summary"]["clips"][0]["frames"]
    assert job["summary"]["clips"][0]["segments"] == [[0, frames]]
    assert job["summary"]["datasets"][0]["num_episodes"] == 1

    bad = client.post(f"/api/jobs/{jid}/edits", json={"clips": {"0": {"segments": [[0, 50], [40, 90]]}}})
    assert bad.status_code == 422
    r = client.post(f"/api/jobs/{jid}/edits",
                    json={"clips": {"0": {"segments": [[0, 50], [60, frames]]}}, "task": "stack the blocks"})
    assert r.status_code == 200 and r.json()["status"] == "queued"
    job = wait(client, jid)
    assert job["status"] == "done" and job["task"] == "stack the blocks"
    d = job["summary"]["datasets"][0]
    assert d["num_episodes"] == 2 and [e["length"] for e in d["episodes"]] == [50, frames - 60]
    assert d["tasks"] == ["stack the blocks"]
    assert client.get(f"/api/jobs/{jid}/clips/0/overlay").json()["segments"] == [[0, 50], [60, frames]]


def test_quota(client, monkeypatch):
    import footage2robot.api.app as app_module
    monkeypatch.setattr(app_module, "MAX_ACTIVE_JOBS", 0)
    assert client.post("/api/jobs/demo", data={"config": json.dumps(CFG)}).status_code == 429


def test_queue_claims_each_job_once_and_recovers_stale(tmp_path):
    store = JobStore(tmp_path)
    ids = []
    for _ in range(6):
        jid = store.new_id()
        store.create(jid, "u", [tmp_path / "x.mp4"], ["x.mp4"], PipelineConfig())
        ids.append(jid)
    claimed, lock = [], threading.Lock()

    def worker(name):
        while (job := store.claim(name)) is not None:
            with lock:
                claimed.append(job["id"])
    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(claimed) == sorted(ids)                           # every job taken exactly once
    assert claimed[:1] and store.claim("late") is None
    assert store.active_count("u") == 6 and not store.delete(ids[0])   # running jobs cannot be deleted

    store.update(ids[0], heartbeat=time.time() - STALE_AFTER_S - 5)  # its worker died
    again = store.claim("rescuer")
    assert again["id"] == ids[0] and again["worker"] == "rescuer"
    store.update(ids[1], status="done")
    assert store.request_rebuild(ids[1]) and store.get(ids[1])["stage"] == "rebuild"
    assert not store.request_rebuild(ids[2])                          # still running
