"""Minimal URDF kinematics: parse a serial chain, forward kinematics, Jacobian."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _floats(text: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    return np.array([float(v) for v in text.split()], np.float64) if text else np.array(default, np.float64)


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s, v = np.cos(angle), np.sin(angle), 1 - np.cos(angle)
    return np.array([[c + x * x * v, x * y * v - z * s, x * z * v + y * s],
                     [y * x * v + z * s, c + y * y * v, y * z * v - x * s],
                     [z * x * v - y * s, z * y * v + x * s, c + z * z * v]])


@dataclass
class Joint:
    name: str
    kind: str                 # revolute | continuous | prismatic | fixed
    parent: str
    child: str
    origin: np.ndarray        # 4x4 parent-link -> joint frame
    axis: np.ndarray
    lower: float
    upper: float

    @property
    def movable(self) -> bool:
        return self.kind != "fixed"


def parse_urdf(path: str | Path) -> dict[str, Joint]:
    """All joints of a URDF keyed by child link name."""
    root = ET.parse(path).getroot()
    joints = {}
    for j in root.findall("joint"):
        origin = j.find("origin")
        T = np.eye(4)
        if origin is not None:
            T[:3, :3] = rpy_matrix(_floats(origin.get("rpy")))
            T[:3, 3] = _floats(origin.get("xyz"))
        axis = _floats(j.find("axis").get("xyz") if j.find("axis") is not None else None, (0, 0, 1))
        n = np.linalg.norm(axis)
        limit = j.find("limit")
        kind = j.get("type")
        lower = float(limit.get("lower", -np.pi)) if limit is not None and kind != "continuous" else -np.pi
        upper = float(limit.get("upper", np.pi)) if limit is not None and kind != "continuous" else np.pi
        joints[j.find("child").get("link")] = Joint(
            j.get("name"), kind, j.find("parent").get("link"), j.find("child").get("link"),
            T, axis / n if n > 0 else np.array([0.0, 0.0, 1.0]), lower, upper)
    return joints


class Chain:
    """Serial kinematic chain between two links of a URDF."""

    def __init__(self, path: str | Path, base_link: str, ee_link: str):
        by_child = parse_urdf(path)
        joints: list[Joint] = []
        link = ee_link
        while link != base_link:
            if link not in by_child:
                raise ValueError(f"Link '{ee_link}' is not a descendant of '{base_link}' in {path}")
            joints.append(by_child[link])
            link = by_child[link].parent
        self.joints = joints[::-1]
        self.movable = [j for j in self.joints if j.movable]
        self.names = [j.name for j in self.movable]
        self.lower = np.array([j.lower for j in self.movable])
        self.upper = np.array([j.upper for j in self.movable])
        self.dof = len(self.movable)

    def fk(self, q: np.ndarray, with_frames: bool = False):
        """End-effector pose (4x4) in the base frame; optionally every joint frame."""
        T = np.eye(4)
        frames = []      # (origin, world axis, kind) of each movable joint
        k = 0
        for j in self.joints:
            T = T @ j.origin
            if j.movable:
                frames.append((T[:3, 3].copy(), T[:3, :3] @ j.axis, j.kind))
                M = np.eye(4)
                if j.kind == "prismatic":
                    M[:3, 3] = j.axis * q[k]
                else:
                    M[:3, :3] = axis_rotation(j.axis, q[k])
                T = T @ M
                k += 1
        return (T, frames) if with_frames else T

    def jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(T_ee, J) with J the 6 x dof geometric Jacobian (linear on top, angular below)."""
        T, frames = self.fk(q, with_frames=True)
        J = np.zeros((6, self.dof))
        p = T[:3, 3]
        for i, (o, z, kind) in enumerate(frames):
            if kind == "prismatic":
                J[:3, i] = z
            else:
                J[:3, i] = np.cross(z, p - o)
                J[3:, i] = z
        return T, J

    def reach(self, from_joint: int = 1) -> float:
        """Sum of link lengths from movable joint ``from_joint`` to the end-effector."""
        total, seen = 0.0, -1
        for j in self.joints:
            if j.movable:
                seen += 1
            if seen >= from_joint and not (j.movable and seen == from_joint):
                total += float(np.linalg.norm(j.origin[:3, 3]))
        return total
