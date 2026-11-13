"""Robot profiles and human -> robot joint-space retargeting.

Arms are driven by inverse kinematics on the robot's URDF: the hand's motion
relative to the person's shoulder is scaled to the robot's reach and replayed
relative to the robot's own shoulder, so a comfortable human reach maps to a
comfortable robot reach. With a workspace marker (``frame="marker"``) targets
are instead placed metrically relative to the marker, with no scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from ..schema import B, SIDES, HumanMotion
from .ik import solve_trajectory
from .urdf import Chain

DATA = Path(__file__).parent / "data"
PALM_OFFSET_M = 0.08          # wrist -> palm centre, added to the tracked arm length


@dataclass
class RobotProfile:
    key: str
    label: str
    urdf: Path
    base_link: str
    ee_link: str
    rest: list[float]
    kind: str = "arm"                         # arm | humanoid
    reach_fraction: float = 0.65              # share of the kinematic reach a full human reach maps to
    finger_axis: tuple[float, float, float] | None = None   # EE-frame axis the fingers open along
    gripper_joint: str | None = None          # extra state dim; None -> "gripper" closedness in [0, 1]
    gripper_open: float = 0.0
    gripper_closed: float = 1.0
    min_z: float = 0.0                        # floor for targets in the base frame (the table)
    max_joint_speed: float = 3.0              # rad/s; caps how fast any joint may move between frames
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @cached_property
    def chain(self) -> Chain:
        return Chain(self.urdf, self.base_link, self.ee_link)

    @cached_property
    def geometry(self) -> dict:
        """Shoulder anchor, forward yaw and tool approach axis, derived from the URDF."""
        chain = self.chain
        rest = np.array(self.rest, np.float64)
        T, frames = chain.fk(rest, with_frames=True)
        anchor = frames[min(1, len(frames) - 1)][0]
        fwd = T[:3, 3] - anchor
        yaw = float(np.arctan2(fwd[1], fwd[0])) if np.hypot(fwd[0], fwd[1]) > 0.02 else 0.0
        # approach = direction from the last moving joint to the tool point, in the EE frame
        tip = T[:3, :3].T @ (T[:3, 3] - frames[-1][0])
        if np.linalg.norm(tip) < 1e-3:
            tip = np.array([0.0, 0.0, 1.0])
        axis = np.zeros(3)
        axis[np.argmax(np.abs(tip))] = np.sign(tip[np.argmax(np.abs(tip))])
        return {"anchor": anchor, "yaw": yaw, "approach_axis": axis,
                "reach": self.reach_fraction * chain.reach(1)}

    @property
    def state_names(self) -> list[str]:
        return self.chain.names + [self.gripper_joint or "gripper"]


ROBOTS: dict[str, RobotProfile] = {p.key: p for p in [
    RobotProfile(
        "so101", "SO-101 / SO-100 arm (5 joints + gripper, LeRobot)", DATA / "so101.urdf",
        "base_link", "gripper_frame_link", rest=[0.0, -0.6, 0.9, 0.6, 0.0],
        gripper_joint="gripper", gripper_open=1.7453, gripper_closed=-0.1745,
        notes="5-DoF: position and approach direction are matched, not roll about the approach axis."),
    RobotProfile(
        "franka_panda", "Franka Emika Panda / FR3 (7 joints + gripper)", DATA / "franka_panda.urdf",
        "panda_link0", "panda_hand_tcp", rest=[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        finger_axis=(0.0, 1.0, 0.0), gripper_joint="panda_finger_width", gripper_open=0.08, gripper_closed=0.0),
    RobotProfile(
        "ur5e", "Universal Robots UR5e (6 joints, generic gripper)", DATA / "ur5e.urdf",
        "base", "tool0", rest=[0.0, -1.571, 1.571, -1.571, -1.571, 0.0],
        notes="Joint angles follow the UR DH convention; gripper is a 0..1 closedness for any tool."),
    RobotProfile(
        "unitree_g1", "Unitree G1 humanoid (29 joints + hands)", DATA / "unitree_g1.urdf",
        "pelvis", "left_wrist_yaw_link", rest=[], kind="humanoid",
        notes="Joint angles are mapped directly from the person's limb angles and clipped to G1 limits. "
              "The pelvis pose and balance are not estimated."),
]}


def custom_profile(urdf: str, base_link: str, ee_link: str, rest: list[float] | None = None) -> RobotProfile:
    """Profile for any serial arm from a user-supplied URDF."""
    chain = Chain(urdf, base_link, ee_link)
    rest = rest if rest is not None else (0.5 * (chain.lower + chain.upper)).tolist()
    return RobotProfile("custom", f"Custom URDF ({Path(urdf).name})", Path(urdf), base_link, ee_link, rest=rest)


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _smooth_directions(v: np.ndarray, fps: float, sigma_s: float = 0.2, max_rate: float = 2.5) -> np.ndarray:
    """Low-pass unit vectors and cap how fast they may turn (rad/s).

    Orientation is the noisiest thing a single camera recovers; a wrist that
    snaps round in one frame is a tracking artefact, and chasing it would make
    the arm thrash. Positions are followed exactly; orientation follows calmly.
    """
    from ..process.filters import fill_gaps, smooth
    from ..retarget.geometry import unit
    out = unit(smooth(fill_gaps(v, max_gap=len(v)), sigma_s * fps).astype(np.float64))
    step = max_rate / fps
    prev = None
    for t in range(len(out)):
        if not np.isfinite(out[t]).all():
            continue
        if prev is not None:
            angle = np.arccos(np.clip(prev @ out[t], -1, 1))
            if angle > step:                                   # turn only `step` of the way towards it
                axis = np.cross(prev, out[t])
                n = np.linalg.norm(axis)
                if n > 1e-9:
                    k = axis / n
                    out[t] = prev * np.cos(step) + np.cross(k, prev) * np.sin(step)
                else:
                    out[t] = prev
        prev = out[t]
    return out


def retarget_robot(motion: HumanMotion, profile: RobotProfile, cfg) -> "tuple[np.ndarray, list[str], dict]":
    """Joint-space trajectory for ``profile``. Returns ``(state, names, info)``."""
    if profile.kind == "humanoid":
        from .g1 import g1_joint_angles
        return g1_joint_angles(motion, profile)

    from ..retarget import pick_side, reference_frame
    from ..retarget.geometry import body_frames, to_frame, unit
    from ..retarget.hands import ee_orientation_cam, ee_position_cam, gripper_closedness

    side = pick_side(motion, cfg)
    geo, chain = profile.geometry, profile.chain
    pos_cam = ee_position_cam(motion, side).astype(np.float64)
    R_hand = ee_orientation_cam(motion, side).astype(np.float64)       # columns: palm normal, knuckles, approach
    info: dict = {"hand": SIDES[side], "robot": profile.key, "units": "radians (prismatic joints in metres)"}

    ref_R, ref_origin, frame = reference_frame(motion, cfg)
    if frame == "marker":
        pos = to_frame(pos_cam, ref_R, ref_origin) + np.asarray(cfg.base_offset)
        to_base = np.swapaxes(ref_R, -1, -2)
        info["mapping"] = "metric, relative to the workspace marker (+ base_offset)"
    else:
        fr = body_frames(motion.body_3d.astype(np.float64))
        hand = to_frame(pos_cam, fr["torso_R"], fr["torso_origin"])
        shoulder = to_frame(motion.body_3d[:, B[f"{SIDES[side]}_shoulder"]].astype(np.float64),
                            fr["torso_R"], fr["torso_origin"])
        s = f"{SIDES[side]}"
        b = motion.body_3d
        human_reach = float(np.nanmedian(
            np.linalg.norm(b[:, B[f"{s}_shoulder"]] - b[:, B[f"{s}_elbow"]], axis=-1)
            + np.linalg.norm(b[:, B[f"{s}_elbow"]] - b[:, B[f"{s}_wrist"]], axis=-1))) + PALM_OFFSET_M
        k = geo["reach"] / human_reach if np.isfinite(human_reach) and human_reach > 0.2 else 1.0
        k *= cfg.scale
        yaw = _rot_z(geo["yaw"])
        pos = geo["anchor"] + (hand - shoulder) * k @ yaw.T
        to_base = yaw @ np.swapaxes(fr["torso_R"], -1, -2)
        info.update(mapping="shoulder-relative, scaled to the robot's reach", workspace_scale=round(k, 3))

    # Work-surface alignment: a tabletop arm's shoulder sits lower above the table than a
    # person's, so lift the whole trajectory just enough to keep it above the robot's floor.
    finite = np.isfinite(pos[:, 2])
    shift = 0.0
    if finite.any() and frame != "marker":
        shift = max(0.0, profile.min_z + 0.03 - float(np.percentile(pos[finite, 2], 2)))
        pos[:, 2] += shift
    info["vertical_shift_m"] = round(shift, 3)
    below = pos[:, 2] < profile.min_z
    info["floor_clipped_fraction"] = round(float(np.mean(below[finite])) if finite.any() else 0.0, 3)
    pos[below, 2] = profile.min_z

    approach = unit(np.squeeze(to_base @ R_hand[:, :, 2:3], -1))
    seconds = None
    second_axis = None
    if profile.finger_axis is not None:        # fingers close along the palm normal
        seconds = unit(np.squeeze(to_base @ R_hand[:, :, 0:1], -1))
        second_axis = np.array(profile.finger_axis)

    # Hand orientation is the noisiest signal (it switches source when the hand tracker drops out),
    # so smooth the direction targets; positions were already smoothed with the rest of the motion.
    approach = _smooth_directions(approach, motion.fps)
    if seconds is not None:
        seconds = _smooth_directions(seconds, motion.fps)

    q, err = solve_trajectory(chain, pos, approach, np.array(profile.rest), geo["approach_axis"],
                              seconds, second_axis, max_step=profile.max_joint_speed / motion.fps)
    closed = gripper_closedness(motion, side).astype(np.float64)
    grip = profile.gripper_open + closed * (profile.gripper_closed - profile.gripper_open) \
        if profile.gripper_joint else closed
    state = np.concatenate([q, grip[:, None]], axis=-1)
    state[~np.isfinite(q).all(axis=1)] = np.nan

    good = np.isfinite(err)
    if good.any():
        info.update(ik_error_mean_m=round(float(err[good].mean()), 4), ik_error_max_m=round(float(err[good].max()), 4),
                    ik_reachable_fraction=round(float((err[good] < 0.02).mean()), 3))
        at_limit = (np.abs(q[good] - chain.lower) < 1e-3) | (np.abs(q[good] - chain.upper) < 1e-3)
        info["at_joint_limit_fraction"] = round(float(at_limit.any(axis=1).mean()), 3)
        steps = np.abs(np.diff(q[good], axis=0))
        info["max_joint_speed_rad_s"] = round(float(steps.max() * motion.fps), 2) if len(steps) else 0.0
    if profile.notes:
        info["notes"] = profile.notes
    return state.astype(np.float32), profile.state_names, info
