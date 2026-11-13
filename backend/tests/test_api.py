import io
import json
import time
import zipfile

import pytest


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os
    os.environ["FOOTAGE2ROBOT_DATA_DIR"] = str(tmp_path_factory.mktemp("data"))
    from fastapi.testclient import TestClient

    from footage2robot.api.app import app
    return TestClient(app)


def wait(client, job_id, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.1)
    raise TimeoutError(job)


CFG = {"embodiments": ["single_arm", "humanoid"], "formats": ["lerobot", "json"]}


def test_options(client):
    opts = client.get("/api/options").json()
    assert "lerobot" in opts["formats"] and "humanoid" in opts["embodiments"]


def test_demo_job_end_to_end(client):
    r = client.post("/api/jobs/demo", data={"config": json.dumps(CFG)})
    assert r.status_code == 201
    job = wait(client, r.json()["id"])
    assert job["status"] == "done", job
    assert "videos" not in job and "video" not in job["summary"]["clips"][0]   # no server paths leak
    jid = job["id"]
    assert any(j["id"] == jid for j in client.get("/api/jobs").json())
    overlay = client.get(f"/api/jobs/{jid}/clips/0/overlay").json()
    assert len(overlay["body"]) == job["summary"]["clips"][0]["frames"]
    preview = client.get(f"/api/jobs/{jid}/datasets/single_arm/preview").json()
    assert preview["state_names"][-1] == "gripper"
    assert client.get(f"/api/jobs/{jid}/clips/0/video").status_code == 404       # synthetic: no footage

    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/jobs/{jid}/download?embodiment=single_arm&format=lerobot").content))
    assert "lerobot/meta/info.json" in z.namelist()
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/jobs/{jid}/download").content))
    names = z.namelist()
    assert "summary.json" in names and "datasets/humanoid/json/dataset.json" in names
    assert not any(n.endswith("preview.json") for n in names)
    assert client.get(f"/api/jobs/{jid}/download?embodiment=dex_hand").status_code == 404

    assert client.delete(f"/api/jobs/{jid}").json() == {"deleted": jid}
    assert client.get(f"/api/jobs/{jid}").status_code == 404


def test_upload_job(client, video):
    with open(video, "rb") as f:
        r = client.post("/api/jobs", files=[("files", ("../../evil name.mp4", f, "video/mp4"))],
                        data={"config": json.dumps({**CFG, "formats": ["json"]})})
    assert r.status_code == 201, r.text
    # the test clip has no person in it, so the real tracker must fail cleanly
    job = wait(client, r.json()["id"], timeout=180)
    assert job["clips"] == ["evil name.mp4"]
    assert job["status"] == "failed" and job["error"]
    assert client.get(f"/api/jobs/{job['id']}/clips/0/video").status_code == 200
    assert client.get(f"/api/jobs/{job['id']}/download").status_code == 409


def test_rejects_bad_input(client):
    bad = client.post("/api/jobs", files=[("files", ("notes.txt", b"hi", "text/plain"))])
    assert bad.status_code == 415
    for cfg in ({"formats": ["nope"]}, {"extractor": "synthetic"}, {"image_size": [4, 4]}, {"target_fps": 0}):
        r = client.post("/api/jobs/demo", data={"config": json.dumps(cfg)})
        assert r.status_code == 422, cfg
    assert client.post("/api/jobs/demo", data={"config": "not json"}).status_code == 422
    assert client.get("/api/jobs/../../etc").status_code == 404
    assert client.get("/api/jobs/zzzzzzzzzzzz").status_code == 404
