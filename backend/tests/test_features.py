"""Calibration, episode splitting, object tracking and edits."""
import copy
import json

import cv2
import numpy as np
import pytest

from footage2robot import calibrate
from footage2robot.extract import get_extractor
from footage2robot.extract.synthetic import CAMERA_OFFSET
from footage2robot.pipeline import PipelineConfig, apply_edits, build, run_pipeline
from footage2robot.process import clean_motion
from footage2robot.retarget import retarget
from footage2robot.schema import HumanMotion
from footage2robot.segment import auto_split, validate_segments


def test_body_scale(motion):
    arm = calibrate.arm_length(motion)
    scale, info = calibrate.body_scale(motion, arm_length_m=arm * 1.2)
    assert scale == pytest.approx(1.2, abs=1e-3) and info["method"] == "arm_length"
    scale, info = calibrate.body_scale(motion, person_height_m=calibrate.estimate_height(motion) * 0.9)
    assert scale == pytest.approx(0.9, abs=1e-3) and info["method"] == "height"
    assert 1.5 < info["tracked_height_m"] < 1.95                     # synthetic person is adult-sized
    assert calibrate.body_scale(motion)[0] == 1.0
    assert calibrate.body_scale(motion, person_height_m=20.0)[0] == 2.0    # clamped: bad input, not a giant
    m = copy.deepcopy(motion)
    calibrate.apply_scale(m, 1.2)
    assert calibrate.arm_length(m) == pytest.approx(arm * 1.2, rel=1e-4)
    # generic end-effector positions scale with the body
    assert np.allclose(retarget(m, "single_arm").state[:, :3], 1.2 * retarget(motion, "single_arm").state[:, :3], atol=1e-4)


def test_root_translation_recovers_camera_distance():
    raw = get_extractor("synthetic").extract("")
    root = calibrate.root_translation(raw, calibrate.Camera(raw.width, raw.height))
    assert np.allclose(root, CAMERA_OFFSET, atol=0.02)


@pytest.fixture(scope="module")
def marker_video(tmp_path_factory):
    """640x480 clip with the workspace marker lying fronto-parallel 1 m away, off-centre."""
    cam = calibrate.Camera(640, 480)
    px, centre = 100, np.array([0.2, 0.1, 1.0])
    size_m = px * centre[2] / cam.f               # edge length that projects to exactly 100 px
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, px)
    frame = np.full((480, 640), 255, np.uint8)
    u = int(round(cam.f * centre[0] / centre[2] + 320 - px / 2))
    v = int(round(cam.f * centre[1] / centre[2] + 240 - px / 2))
    frame[v:v + px, u:u + px] = marker
    path = tmp_path_factory.mktemp("marker") / "marker.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
    for _ in range(45):
        w.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    w.release()
    return str(path), size_m, centre


def test_marker_pose(marker_video):
    path, size, centre = marker_video
    found = calibrate.find_marker(path, size)
    assert found is not None and found["detections"] > 5
    assert np.allclose(found["t"], centre, atol=0.03)
    R = np.array(found["R"])
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6) and np.linalg.det(R) == pytest.approx(1.0)
    assert abs(R[2, 2]) > 0.98                                        # marker faces the camera
    assert calibrate.marker_image().shape == (960, 960)


def test_marker_frame_is_metric_and_fixed(marker_video, tmp_path):
    """With a marker, positions are relative to it — independent of where the person stands."""
    from footage2robot.retarget.hands import ee_position_cam
    path, size, centre = marker_video
    cfg = PipelineConfig(extractor="synthetic", embodiments=["single_arm"], formats=["json"],
                         marker_size_m=size, include_images=False)
    summary = run_pipeline([path], tmp_path, cfg)
    assert summary["clips"][0]["calibration"]["marker"] and not summary["warnings"]
    d = summary["datasets"][0]
    assert d["episodes"][0]["frame"].startswith("workspace marker")
    motion = HumanMotion.load(tmp_path / "motion/episode_000.npz")
    hand_cam = ee_position_cam(motion, 1).astype(np.float64) + CAMERA_OFFSET     # true camera-frame position
    R = np.array(calibrate.find_marker(path, size)["R"])
    want = (hand_cam - centre) @ R
    got = np.array(json.loads((tmp_path / "datasets/single_arm/json/dataset.json").read_text())["episodes"][0]["state"])
    assert np.allclose(got[:, :3], want, atol=0.04)


def _repeat(motion, reps=3, pause_frames=45):
    still = {k: np.repeat(getattr(motion, k)[:1], pause_frames, axis=0) for k in HumanMotion._ARRAYS}
    out = copy.deepcopy(motion)
    for k in HumanMotion._ARRAYS:
        parts = []
        for r in range(reps):
            parts += [getattr(motion, k)] + ([still[k]] if r < reps - 1 else [])
        setattr(out, k, np.concatenate(parts))
    return out


def test_auto_split(motion):
    assert auto_split(motion) == [[0, motion.num_frames]]            # one continuous demonstration
    long = _repeat(motion)
    episodes = auto_split(long)
    assert len(episodes) == 3
    n = motion.num_frames
    for k, (a, b) in enumerate(episodes):
        start = k * (n + 45)
        assert abs(a - start) < 20 and abs(b - (start + n)) < 20
    assert auto_split(_repeat(motion, reps=1)) == [[0, n]]


def test_validate_segments():
    assert validate_segments([[50, 90], [0, 40]], 100) == [[0, 40], [50, 90]]
    assert validate_segments([[-5, 500]], 100) == [[0, 100]]
    for bad in ([[0, 3]], [[0, 50], [40, 90]], [[1, 2, 3]], ["x"]):
        with pytest.raises(ValueError):
            validate_segments(bad, 100)


@pytest.fixture(scope="module")
def ball_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("ball") / "ball.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for i in range(45):
        frame = np.full((240, 320, 3), 40, np.uint8)
        cv2.circle(frame, (40 + 5 * i, 120), 20, (0, 200, 255), -1)
        w.write(frame)
    w.release()
    return str(path)


def test_object_tracking_follows_the_object(ball_video):
    from footage2robot.objects import track_object
    boxes = track_object(ball_video, np.arange(45), (0, 40 / 320, 0.5))
    assert boxes.shape == (45, 5) and boxes[:, 4].all()
    assert np.allclose(boxes[:, 0], (40 + 5 * np.arange(45)) / 320, atol=0.02)
    assert np.allclose(boxes[:, 1], 0.5, atol=0.02) and np.allclose(boxes[:, 2], 40 / 320, atol=0.03)
    # prompt on empty background: nothing to track
    assert not track_object(ball_video, np.arange(45), (0, 0.9, 0.1))[:, 4].any()


def test_edits_rebuild_with_segments_and_object(ball_video, tmp_path):
    import h5py
    import pyarrow.parquet as pq
    cfg = PipelineConfig(extractor="synthetic", embodiments=["single_arm"], image_size=(32, 32),
                         formats=["lerobot", "robomimic", "zarr", "npz", "json"])
    summary = run_pipeline([ball_video], tmp_path, cfg)
    assert summary["datasets"][0]["num_episodes"] == 1 and summary["datasets"][0]["env_state_names"] == []

    apply_edits(tmp_path, {"0": {"segments": [[0, 20], [25, 45]], "object_prompt": [0, 40 / 320, 0.5]}})
    cfg.track_objects = True
    summary = build(tmp_path, cfg)
    d = summary["datasets"][0]
    assert not summary["warnings"], summary["warnings"]
    assert d["num_episodes"] == 2 and [e["length"] for e in d["episodes"]] == [20, 20]
    assert d["env_state_names"][0] == "object_cx" and summary["clips"][0]["object_tracked"]
    root = tmp_path / "datasets/single_arm"
    second = np.load(root / "npz/episode_000001.npz")
    assert np.allclose(second["env_state"][:, 0], (40 + 5 * np.arange(25, 45)) / 320, atol=0.02)
    # the second episode's images start at clip frame 25
    cap = cv2.VideoCapture(ball_video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 25)
    want = cv2.resize(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB), (32, 32), interpolation=cv2.INTER_AREA)
    assert np.abs(second["images"][0].astype(int) - want).mean() < 1.0
    table = pq.read_table(root / "lerobot/data/chunk-000/file-000.parquet")
    assert "observation.environment_state" in table.schema.names
    info = json.loads((root / "lerobot/meta/info.json").read_text())
    assert info["features"]["observation.environment_state"]["shape"] == [5]
    with h5py.File(root / "robomimic/demos.hdf5") as f:
        assert f["data/demo_1/obs/object"].shape == (20, 5)
    with pytest.raises(ValueError):
        apply_edits(tmp_path, {"0": {"segments": [[0, 30], [20, 45]]}})
    with pytest.raises(ValueError):
        apply_edits(tmp_path, {"3": {"segments": [[0, 30]]}})


def test_auto_split_in_pipeline(motion, tmp_path, monkeypatch):
    long = _repeat(motion)
    import footage2robot.pipeline as pipeline

    class Fake:
        def extract(self, *a, **k):
            return copy.deepcopy(long)
    monkeypatch.setattr(pipeline, "get_extractor", lambda *a, **k: Fake())
    cfg = PipelineConfig(extractor="synthetic", embodiments=["single_arm", "so101"], formats=["json"],
                         split="auto", include_images=False, smooth_s=0.0)
    summary = run_pipeline(["long.mp4"], tmp_path, cfg)
    assert len(summary["clips"][0]["segments"]) == 3
    assert [d["num_episodes"] for d in summary["datasets"]] == [3, 3]
    preview = json.loads((tmp_path / "datasets/so101/preview.json").read_text())
    assert [e["offset"] for e in preview["episodes"]] == [s[0] for s in summary["clips"][0]["segments"]]
