import numpy as np
import pytest

from footage2robot.dataset import UntrackedError, build_episode, compute_actions
from footage2robot.process import fill_gaps, smooth
from footage2robot.retarget import EMBODIMENTS, RetargetConfig, retarget
from footage2robot.retarget.geometry import ROTATION_DIMS, encode_rotation, matrix_to_quat
from footage2robot.schema import HumanMotion


def test_fill_gaps_interpolates_short_runs_only():
    x = np.arange(10, dtype=np.float32)[:, None].copy()
    x[3:5] = np.nan      # short gap
    y = fill_gaps(x, max_gap=2)
    assert np.allclose(y[:, 0], np.arange(10))
    x[3:8] = np.nan      # long gap stays
    assert np.isnan(fill_gaps(x, max_gap=2)[5, 0])
    x2 = np.arange(5, dtype=np.float32)[:, None].copy()
    x2[0] = np.nan       # never extrapolate
    assert np.isnan(fill_gaps(x2, 3)[0, 0])


def test_smooth_preserves_nans_and_reduces_noise():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 2)).astype(np.float32)
    x[50] = np.nan
    y = smooth(x, 2.0)
    assert np.isnan(y[50]).all() and np.isfinite(y[49]).all()
    assert np.nanstd(y) < 0.6 * np.nanstd(x)


def test_motion_roundtrip(motion, tmp_path):
    motion.save(tmp_path / "m.npz")
    back = HumanMotion.load(tmp_path / "m.npz")
    assert back.fps == motion.fps and np.allclose(back.body_3d, motion.body_3d, equal_nan=True)
    overlay = back.to_overlay_json()
    assert len(overlay["body"]) == motion.num_frames and len(overlay["hands"][0]) == 2


@pytest.mark.parametrize("emb", list(EMBODIMENTS))
def test_retarget_shapes(motion, emb):
    traj = retarget(motion, emb)
    assert traj.state.shape == (motion.num_frames, len(traj.names))
    assert len(set(traj.names)) == len(traj.names)
    assert np.isfinite(traj.state).all()


def test_single_arm_semantics(motion):
    """Synthetic clip: the right hand reaches forward, closes, lifts, and opens."""
    traj = retarget(motion, "single_arm")
    assert traj.info["hand"] == "right"
    x, y, grip = traj.state[:, 0], traj.state[:, 1], traj.state[:, 6]
    mid = len(x) // 2
    assert x[mid] > x[0] + 0.2            # reaches forward (+x) in the torso frame
    assert (y < 0).all()                  # right hand stays on the person's right (-y)
    assert grip[0] < 0.2 and grip[mid] > 0.8 and grip[-1] < 0.2
    assert retarget(motion, "single_arm", RetargetConfig(side="left")).info["hand"] == "left"


def test_retarget_is_camera_invariant(motion):
    """Rotating the whole scene about the camera's vertical axis must not change the result."""
    import copy
    a = np.deg2rad(35)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]], np.float32)
    rot = copy.deepcopy(motion)
    rot.body_3d = motion.body_3d @ R.T
    rot.hands_3d = motion.hands_3d @ R.T
    for emb in ("bimanual", "humanoid", "dex_hand"):
        assert np.allclose(retarget(motion, emb).state, retarget(rot, emb).state, atol=2e-3), emb


def test_scale_and_rotation_encodings(motion):
    base = retarget(motion, "single_arm").state
    scaled = retarget(motion, "single_arm", RetargetConfig(scale=0.5)).state
    assert np.allclose(scaled[:, :3], 0.5 * base[:, :3], atol=1e-5)
    for kind, dim in ROTATION_DIMS.items():
        assert retarget(motion, "single_arm", RetargetConfig(rotation=kind)).state.shape[1] == 3 + dim + 1


def test_rotation_helpers():
    a = 0.7
    Rz = np.array([[[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]])
    assert np.allclose(encode_rotation(Rz, "euler"), [[0, 0, a]])
    assert np.allclose(encode_rotation(Rz, "rotvec"), [[0, 0, a]])
    assert np.allclose(matrix_to_quat(Rz), [[0, 0, np.sin(a / 2), np.cos(a / 2)]])


def test_humanoid_elbow_and_knee(motion):
    traj = retarget(motion, "humanoid")
    s = dict(zip(traj.names, traj.state.T))
    assert (s["right_elbow"] >= 0).all() and s["right_elbow"].max() < np.pi
    assert s["left_knee"].max() > s["left_knee"].min() + 0.1     # knees bend in the clip
    assert abs(s["waist_yaw"]).max() < 0.2


def test_actions():
    s = np.array([[0, 0.0], [1, 1.0], [3, 0.0]], np.float32)
    assert np.allclose(compute_actions(s, "absolute", []), [[1, 1], [3, 0], [3, 0]])
    assert np.allclose(compute_actions(s, "delta", [1]), [[1, 1], [2, 0], [0, 0]])


def test_build_episode_trims_and_holds(motion):
    traj = retarget(motion, "single_arm")
    traj.state[:5] = np.nan
    traj.state[40:45] = np.nan
    traj.state[-3:] = np.nan
    ep = build_episode(traj, task="t", video_path="x.mp4", start_frame=10, stride=2)
    assert len(ep) == motion.num_frames - 8 and np.isfinite(ep.state).all()
    assert ep.source_frames[0] == 10 + 5 * 2
    assert not ep.tracked[35:40].any() and np.allclose(ep.state[35:40], ep.state[34])
    traj.state[:] = np.nan
    with pytest.raises(UntrackedError):
        build_episode(traj, task="t")


def test_optional_dims_default_to_zero(motion):
    import copy
    m = copy.deepcopy(motion)
    m.body_3d[:, 25:] = np.nan           # no legs in frame (upper-body footage)
    traj = retarget(m, "humanoid")
    ep = build_episode(traj, task="t")
    assert len(ep) == m.num_frames and np.isfinite(ep.state).all()
    assert "left_knee" in ep.meta["defaulted_dims"]
