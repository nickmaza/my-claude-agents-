import numpy as np
import pytest

from footage2robot.retarget import RetargetConfig, retarget
from footage2robot.retarget.geometry import body_frames, to_frame, unit
from footage2robot.robots import ROBOTS, custom_profile
from footage2robot.robots.ik import solve_ik
from footage2robot.robots.urdf import Chain
from footage2robot.schema import B

ARMS = ["so101", "franka_panda", "ur5e"]


def test_fk_matches_published_poses():
    franka = ROBOTS["franka_panda"].chain
    T = franka.fk(np.array(ROBOTS["franka_panda"].rest))
    assert np.allclose(T[:3, 3], [0.307, 0.0, 0.487], atol=2e-3)      # Franka's documented home pose
    assert np.allclose(T[:3, 2], [0, 0, -1], atol=2e-3)                # gripper pointing down
    ur = ROBOTS["ur5e"].chain
    assert np.allclose(ur.fk(np.zeros(6))[:3, 3], [-0.8172, -0.2329, 0.0628], atol=1e-4)   # UR5e zero pose


@pytest.mark.parametrize("key", ARMS)
def test_jacobian_matches_finite_differences(key):
    chain = ROBOTS[key].chain
    q = np.random.default_rng(1).uniform(-0.8, 0.8, chain.dof)
    T, J = chain.jacobian(q)
    eps = 1e-6
    num = np.stack([(chain.fk(q + eps * np.eye(chain.dof)[i])[:3, 3] - T[:3, 3]) / eps for i in range(chain.dof)], 1)
    assert np.abs(num - J[:3]).max() < 1e-5


def test_ik_reaches_target_within_limits():
    p = ROBOTS["franka_panda"]
    q, err = solve_ik(p.chain, np.array([0.45, 0.15, 0.3]), np.array([0.0, 0.0, -1.0]),
                      np.array(p.rest), p.geometry["approach_axis"])
    assert err < 1e-3
    assert (q >= p.chain.lower - 1e-9).all() and (q <= p.chain.upper + 1e-9).all()
    assert np.dot(p.chain.fk(q)[:3, 2], [0, 0, -1]) > 0.99          # approach axis aligned
    # unreachable target: stays within limits and reports the miss
    q, err = solve_ik(p.chain, np.array([3.0, 0.0, 0.3]), None, np.array(p.rest), p.geometry["approach_axis"])
    assert err > 1.0 and (q >= p.chain.lower - 1e-9).all() and (q <= p.chain.upper + 1e-9).all()


@pytest.mark.parametrize("key", ARMS)
def test_arm_retarget(motion, key):
    p = ROBOTS[key]
    traj = retarget(motion, key)
    assert traj.names == p.chain.names + [p.gripper_joint or "gripper"]
    q = traj.state[:, :-1]
    assert np.isfinite(traj.state).all()
    assert (q >= p.chain.lower - 1e-6).all() and (q <= p.chain.upper + 1e-6).all()
    assert traj.info["ik_error_max_m"] < 0.005 and traj.info["ik_reachable_fraction"] == 1.0
    assert np.abs(np.diff(q, axis=0)).max() <= p.max_joint_speed / motion.fps + 1e-6   # within the speed limit
    assert traj.info["max_joint_speed_rad_s"] <= p.max_joint_speed + 1e-6
    # the tool actually follows the hand: forward reach in the clip moves the tool away from the base
    tool = np.array([p.chain.fk(row)[:3, 3] for row in q.astype(np.float64)])
    assert (tool[:, 2] >= p.min_z - 1e-6).all()
    reach = np.linalg.norm(tool[:, :2] - p.geometry["anchor"][:2], axis=1)
    assert reach[len(reach) // 2] > reach[0] + 0.05
    # gripper follows the hand: open at the start, closed mid-clip
    lo, hi = sorted([p.gripper_open, p.gripper_closed]) if p.gripper_joint else (0.0, 1.0)
    grip = traj.state[:, -1]
    assert grip.min() >= lo - 1e-6 and grip.max() <= hi + 1e-6
    closed_value = p.gripper_closed if p.gripper_joint else 1.0
    assert abs(grip[len(grip) // 2] - closed_value) < abs(grip[0] - closed_value)


def test_g1_matches_human_limb_directions(motion):
    """Feeding the G1 angles through the G1's own kinematics must reproduce the person's arm directions."""
    p = ROBOTS["unitree_g1"]
    traj = retarget(motion, "unitree_g1")
    assert traj.state.shape[1] == 31 and np.isfinite(traj.state).all()
    s = dict(zip(traj.names, traj.state.astype(np.float64).T))
    fr = body_frames(motion.body_3d.astype(np.float64))
    zero = np.zeros((motion.num_frames, 3))
    for side in ("left", "right"):
        chain = Chain(p.urdf, "torso_link", f"{side}_wrist_yaw_link")
        human_u = unit(to_frame(motion.body_3d[:, B[f"{side}_elbow"]] - motion.body_3d[:, B[f"{side}_shoulder"]],
                                fr["torso_R"], zero))
        human_f = unit(to_frame(motion.body_3d[:, B[f"{side}_wrist"]] - motion.body_3d[:, B[f"{side}_elbow"]],
                                fr["torso_R"], zero))
        for t in range(0, motion.num_frames, 10):
            q = np.array([s[n][t] for n in chain.names])
            _, frames = chain.fk(q, with_frames=True)
            origins = [f[0] for f in frames]
            upper, fore = unit(origins[3] - origins[0]), unit(origins[6] - origins[3])
            assert np.arccos(np.clip(upper @ human_u[t], -1, 1)) < 0.3, (side, t)
            assert np.arccos(np.clip(fore @ human_f[t], -1, 1)) < 0.3, (side, t)
    limits = {j.name: (j.lower, j.upper) for j in Chain(p.urdf, "pelvis", "left_ankle_roll_link").movable}
    for name, (lo, hi) in limits.items():
        assert (s[name] >= lo - 1e-6).all() and (s[name] <= hi + 1e-6).all()


def test_custom_urdf_and_side(motion):
    profile = custom_profile(str(ROBOTS["ur5e"].urdf), "base", "tool0")
    assert profile.chain.dof == 6 and profile.key == "custom"
    with pytest.raises(ValueError):
        custom_profile(str(ROBOTS["ur5e"].urdf), "base", "no_such_link")
    assert retarget(motion, "so101", RetargetConfig(side="left")).info["hand"] == "left"
