import json

import numpy as np
import pytest

from footage2robot.pipeline import PipelineConfig, run_pipeline


@pytest.fixture(scope="module")
def result(video, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    cfg = PipelineConfig(extractor="synthetic", task="pick up the cup", embodiments=["single_arm", "humanoid"],
                         image_size=(64, 48), formats=["lerobot", "lerobot_v21", "robomimic", "zarr", "npz", "json"])
    return out, run_pipeline([video, video], out, cfg)


def test_summary(result):
    out, summary = result
    assert not summary["warnings"]
    assert [d["embodiment"] for d in summary["datasets"]] == ["single_arm", "humanoid"]
    d = summary["datasets"][0]
    assert d["num_episodes"] == 2 and d["total_frames"] == 90 and d["has_images"]
    assert set(d["exports"]) == {"lerobot", "lerobot_v21", "robomimic", "zarr", "npz", "json"} and not d["errors"]
    assert json.loads((out / "summary.json").read_text())["clips"][0]["frames"] == 45
    preview = json.loads((out / "datasets/single_arm/preview.json").read_text())
    assert [e["clip"] for e in preview["episodes"]] == [0, 1] and len(preview["episodes"][0]["state"]) == 45


def test_lerobot_v3(result):
    import cv2
    import pyarrow.parquet as pq
    root = result[0] / "datasets/single_arm/lerobot"
    info = json.loads((root / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0" and info["total_episodes"] == 2 and info["total_frames"] == 90
    data = pq.read_table(root / info["data_path"].format(chunk_index=0, file_index=0)).to_pydict()
    assert data["index"] == list(range(90)) and data["episode_index"] == [0] * 45 + [1] * 45
    assert data["frame_index"] == list(range(45)) * 2
    assert np.allclose(data["action"][0], data["observation.state"][1])
    episodes = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    assert [(e["dataset_from_index"], e["dataset_to_index"]) for e in episodes] == [(0, 45), (45, 90)]
    key = "videos/observation.images.cam"
    assert episodes[1][f"{key}/from_timestamp"] == pytest.approx(1.5) and episodes[1]["tasks"] == ["pick up the cup"]
    assert np.shape(episodes[0]["stats/observation.images.cam/mean"]) == (3, 1, 1)
    assert len(episodes[0]["stats/observation.state/q50"]) == 7
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pandas()
    assert tasks.index.name == "task" and tasks.loc["pick up the cup", "task_index"] == 0
    stats = json.loads((root / "meta/stats.json").read_text())
    assert stats["action"]["count"] == [90] and "q99" in stats["observation.state"]
    video = root / info["video_path"].format(video_key="observation.images.cam", chunk_index=0, file_index=0)
    assert int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT)) == 90


def test_lerobot_loads_in_the_lerobot_library(result):
    """Load the export with the real library. Needs FOOTAGE2ROBOT_LEROBOT_PYTHON (an interpreter with
    `lerobot[dataset]` installed) because lerobot pins dependencies that clash with ours."""
    import os
    import subprocess
    python = os.environ.get("FOOTAGE2ROBOT_LEROBOT_PYTHON")
    if not python:
        pytest.skip("set FOOTAGE2ROBOT_LEROBOT_PYTHON to run")
    script = (
        "import sys, json, numpy as np\n"
        "from lerobot.datasets import LeRobotDataset\n"
        "ds = LeRobotDataset('local/f2r', root=sys.argv[1])\n"
        "a, b = ds[0], ds[len(ds) - 1]\n"
        "print(json.dumps({'n': len(ds), 'episodes': ds.num_episodes, 'state': a['observation.state'].tolist(),\n"
        "  'image': list(a['observation.images.cam'].shape), 'last_ep': int(b['episode_index']), 'task': b['task'],\n"
        "  'last_image_mean': float(b['observation.images.cam'].mean())}))\n")
    out = subprocess.run([python, "-c", script, str(result[0] / "datasets/single_arm/lerobot")],
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    npz = np.load(result[0] / "datasets/single_arm/npz/episode_000000.npz")
    assert got["n"] == 90 and got["episodes"] == 2 and got["last_ep"] == 1 and got["task"] == "pick up the cup"
    assert np.allclose(got["state"], npz["state"][0], atol=1e-5)
    assert got["image"] == [3, 120, 160] and 0.05 < got["last_image_mean"] < 0.6


def test_lerobot_v21(result):
    import cv2
    import pyarrow.parquet as pq
    root = result[0] / "datasets/single_arm/lerobot_v21"
    info = json.loads((root / "meta/info.json").read_text())
    assert info["codebase_version"] == "v2.1" and info["total_episodes"] == 2 and info["total_frames"] == 90
    assert info["features"]["observation.state"]["shape"] == [7]
    assert info["features"]["observation.images.cam"]["dtype"] == "video"
    t0 = pq.read_table(root / info["data_path"].format(episode_chunk=0, episode_index=0)).to_pydict()
    t1 = pq.read_table(root / info["data_path"].format(episode_chunk=0, episode_index=1)).to_pydict()
    assert t0["frame_index"] == list(range(45)) and t1["index"] == list(range(45, 90))
    assert np.allclose(t0["timestamp"], np.arange(45) / 30, atol=1e-6)
    assert np.allclose(t0["action"][0], t0["observation.state"][1])        # absolute action = next state
    video = root / info["video_path"].format(episode_chunk=0, episode_index=1, video_key="observation.images.cam")
    cap = cv2.VideoCapture(str(video))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 45
    for name, n in (("tasks", 1), ("episodes", 2), ("episodes_stats", 2)):
        rows = [json.loads(line) for line in (root / f"meta/{name}.jsonl").read_text().splitlines()]
        assert len(rows) == n
    assert rows[0]["stats"]["observation.state"]["count"] == [45]
    assert np.shape(rows[0]["stats"]["observation.images.cam"]["mean"]) == (3, 1, 1)


def test_robomimic(result):
    import h5py
    with h5py.File(result[0] / "datasets/humanoid/robomimic/demos.hdf5") as f:
        assert f["data"].attrs["total"] == 90 and set(f["data"]) == {"demo_0", "demo_1"}
        demo = f["data/demo_0"]
        assert demo.attrs["num_samples"] == 45 and demo["actions"].shape == (45, 25)
        assert demo["obs/cam_image"].shape == (45, 48, 64, 3) and demo["obs/cam_image"].dtype == np.uint8
        assert demo["dones"][-1] == 1 and demo["dones"][:-1].sum() == 0
        assert "state_names" in json.loads(f["data"].attrs["env_args"])["env_kwargs"]
        assert len(f["mask/train"]) + len(f["mask/valid"]) == 2


def test_zarr(result):
    import zarr
    root = result[0] / "datasets/single_arm/zarr/replay_buffer.zarr"
    assert (root / ".zgroup").exists()                    # v2 on-disk format
    z = zarr.open_group(str(root), mode="r")
    assert z["data/state"].shape == (90, 7) and z["data/img"].shape == (90, 48, 64, 3)
    assert list(z["meta/episode_ends"][:]) == [45, 90]


def test_npz_and_json(result):
    root = result[0] / "datasets/single_arm"
    ep = np.load(root / "npz/episode_000001.npz")
    assert ep["state"].shape == (45, 7) and ep["images"].shape == (45, 48, 64, 3)
    doc = json.loads((root / "json/dataset.json").read_text())
    assert doc["state_names"][-1] == "gripper" and len(doc["episodes"][0]["state"]) == 45
    assert np.allclose(doc["episodes"][0]["state"], ep["state"], atol=1e-5)


def test_images_follow_source_frames(result, video):
    """Exported frames must be the video frames the states were computed from."""
    import cv2
    ep = np.load(result[0] / "datasets/single_arm/npz/episode_000000.npz")
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 20)
    _, bgr = cap.read()
    want = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (64, 48), interpolation=cv2.INTER_AREA)
    assert np.abs(ep["images"][20].astype(int) - want).mean() < 1.0


def test_rlds(video, tmp_path):
    pytest.importorskip("tensorflow")
    from footage2robot.export.rlds import RLDSUnavailable, _import_tfds
    try:
        tfds = _import_tfds()
    except RLDSUnavailable:
        pytest.skip("tensorflow-datasets not installed")
    cfg = PipelineConfig(extractor="synthetic", task="pick up the cup", embodiments=["single_arm"],
                         formats=["rlds"], image_size=(32, 32))
    summary = run_pipeline([video], tmp_path, cfg)
    assert not summary["warnings"], summary["warnings"]
    builder = tfds.builder_from_directory(str(tmp_path / "datasets/single_arm/rlds/footage2robot_single_arm/1.0.0"))
    episode = next(iter(builder.as_dataset(split="train")))
    steps = list(episode["steps"])
    assert len(steps) == 45
    assert steps[0]["observation"]["state"].shape == (7,) and steps[0]["observation"]["image"].shape == (32, 32, 3)
    assert bool(steps[0]["is_first"]) and bool(steps[-1]["is_last"]) and not bool(steps[1]["is_last"])
    assert steps[0]["language_instruction"].numpy() == b"pick up the cup"


def test_delta_actions_and_no_images(tmp_path):
    cfg = PipelineConfig(extractor="synthetic", embodiments=["single_arm"], formats=["json"],
                         action_mode="delta", include_images=False)
    summary = run_pipeline(["missing.mp4"], tmp_path, cfg)
    d = summary["datasets"][0]
    assert d["action_names"][0] == "delta_x" and d["action_names"][-1] == "gripper" and not d["has_images"]
    ep = json.loads((tmp_path / "datasets/single_arm/json/dataset.json").read_text())["episodes"][0]
    s, a = np.array(ep["state"]), np.array(ep["action"])
    assert np.allclose(a[:-1, :6], s[1:, :6] - s[:-1, :6], atol=1e-5) and np.allclose(a[:-1, 6], s[1:, 6])


def test_config_validation():
    with pytest.raises(ValueError):
        PipelineConfig(formats=["nope"]).validate()
    with pytest.raises(ValueError):
        PipelineConfig(embodiments=[]).validate()
