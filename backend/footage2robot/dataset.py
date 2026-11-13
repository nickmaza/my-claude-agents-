"""Format-agnostic robot dataset: episodes of (observation.state, action, images, task)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from .retarget import Trajectory

ACTION_MODES = {
    "absolute": "action[t] = state[t+1] (next target pose / joint positions)",
    "delta": "action[t] = state[t+1] - state[t] (grippers & finger joints stay absolute)",
}


@dataclass
class Episode:
    state: np.ndarray                 # (T, D) float32, no NaNs
    action: np.ndarray                # (T, D) float32
    task: str = ""
    video_path: str | None = None
    source_frames: np.ndarray | None = None   # (T,) index into the source video per step
    tracked: np.ndarray | None = None  # (T,) False where the state was hold-filled
    env_state: np.ndarray | None = None  # (T, E) e.g. tracked object box; None if not available
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.state.shape[0])

    @property
    def has_video(self) -> bool:
        return bool(self.video_path) and self.source_frames is not None

    def iter_images(self, size: tuple[int, int] | None = None) -> Iterator[np.ndarray]:
        """Yield one RGB uint8 frame per step, optionally resized to (width, height)."""
        import cv2

        if not self.has_video:
            return
        wanted = [int(i) for i in self.source_frames]
        cap = cv2.VideoCapture(self.video_path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, wanted[0])
            cur, last = wanted[0], None
            for target in wanted:
                while cur <= target:
                    ok, bgr = cap.read()
                    cur += 1
                    if not ok:
                        break
                    last = bgr
                if last is None:
                    raise ValueError(f"Could not read frame {target} of {self.video_path}")
                rgb = cv2.cvtColor(last, cv2.COLOR_BGR2RGB)
                if size is not None and (rgb.shape[1], rgb.shape[0]) != tuple(size):
                    rgb = cv2.resize(rgb, tuple(size), interpolation=cv2.INTER_AREA)
                yield rgb
        finally:
            cap.release()

    def images(self, size: tuple[int, int]) -> np.ndarray:
        return np.stack(list(self.iter_images(size)))


@dataclass
class RobotDataset:
    name: str
    embodiment: str
    fps: float
    state_names: list[str]
    action_names: list[str]
    episodes: list[Episode] = field(default_factory=list)
    action_mode: str = "absolute"
    image_size: tuple[int, int] = (224, 224)     # (width, height) for array-based formats
    include_images: bool = True
    camera: str = "cam"
    info: dict = field(default_factory=dict)
    env_state_names: list[str] = field(default_factory=list)

    @property
    def state_dim(self) -> int:
        return len(self.state_names)

    @property
    def total_frames(self) -> int:
        return sum(len(e) for e in self.episodes)

    @property
    def has_images(self) -> bool:
        return self.include_images and bool(self.episodes) and all(e.has_video for e in self.episodes)

    @property
    def has_env_state(self) -> bool:
        return bool(self.env_state_names) and bool(self.episodes) and \
            all(e.env_state is not None for e in self.episodes)

    def describe(self) -> dict:
        return {
            "name": self.name, "embodiment": self.embodiment, "fps": self.fps,
            "num_episodes": len(self.episodes), "total_frames": self.total_frames,
            "state_dim": self.state_dim, "state_names": self.state_names,
            "action_names": self.action_names, "action_mode": self.action_mode,
            "has_images": self.has_images, "image_size": list(self.image_size),
            "env_state_names": self.env_state_names if self.has_env_state else [],
            "tasks": sorted({e.task for e in self.episodes}), "info": self.info,
        }


def compute_actions(state: np.ndarray, mode: str, absolute_dims: list[int]) -> np.ndarray:
    nxt = np.concatenate([state[1:], state[-1:]], axis=0)
    if mode == "absolute":
        return nxt.copy()
    if mode == "delta":
        act = nxt - state
        act[:, absolute_dims] = nxt[:, absolute_dims]
        return act
    raise KeyError(f"Unknown action mode '{mode}'. Options: {sorted(ACTION_MODES)}")


def action_names(names: list[str], mode: str, absolute_dims: list[int]) -> list[str]:
    if mode == "absolute":
        return list(names)
    return [n if i in absolute_dims else f"delta_{n}" for i, n in enumerate(names)]


class UntrackedError(ValueError):
    """Raised when a clip has too little usable tracking for an embodiment."""


def build_episode(traj: Trajectory, *, task: str, action_mode: str = "absolute",
                  video_path: str | None = None, start_frame: int = 0, stride: int = 1,
                  min_tracked: float = 0.5, min_length: int = 5,
                  frame_range: tuple[int, int] | None = None,
                  env_state: np.ndarray | None = None) -> Episode:
    """Turn a (possibly gappy) trajectory into a clean episode.

    Leading/trailing untracked frames are trimmed; interior gaps (already too
    long for interpolation) are filled by holding the last tracked state and
    flagged in ``tracked``. ``frame_range`` restricts the episode to
    ``[start, end)`` frames of the clip; ``env_state`` is a clip-length array
    cut to the same frames.
    """
    a, b = frame_range if frame_range is not None else (0, len(traj.state))
    full = traj.state
    traj_state = full[a:b]
    required = np.setdiff1d(np.arange(traj.state.shape[1]), traj.optional_dims)
    ok = np.isfinite(traj_state[:, required]).all(axis=1)
    if traj.valid is not None:
        ok &= traj.valid[a:b]
    idx = np.where(ok)[0]
    if len(idx) < min_length:
        raise UntrackedError(f"Only {len(idx)} usable frames for embodiment '{traj.embodiment}'")
    lo, hi = idx[0], idx[-1] + 1
    state, ok = traj_state[lo:hi].copy(), ok[lo:hi]
    if ok.mean() < min_tracked:
        raise UntrackedError(
            f"Only {ok.mean():.0%} of frames tracked for embodiment '{traj.embodiment}' (need {min_tracked:.0%})")
    last = np.maximum.accumulate(np.where(ok, np.arange(len(ok)), 0))
    state = state[last]
    defaulted = []
    for d in traj.optional_dims:       # hold-fill optional dims; zero if never tracked
        col = state[:, d]
        good = np.where(np.isfinite(col))[0]
        if len(good) == 0:
            col[:] = 0.0
            defaulted.append(traj.names[d])
        elif len(good) < len(col):
            nearest = good[np.abs(np.arange(len(col))[:, None] - good[None]).argmin(axis=1)]
            col[:] = col[nearest]
    return Episode(
        state=state.astype(np.float32),
        action=compute_actions(state, action_mode, traj.absolute_dims).astype(np.float32),
        task=task, video_path=video_path,
        source_frames=start_frame + (a + lo + np.arange(hi - lo)) * stride if video_path else None,
        tracked=ok,
        env_state=env_state[a + lo:a + hi].astype(np.float32) if env_state is not None else None,
        meta={"tracked_fraction": float(ok.mean()), "trimmed_start": int(a + lo),
              "trimmed_end": int(len(full) - (a + hi)),
              "defaulted_dims": _summarise(defaulted), **traj.info},
    )


def _summarise(names: list[str], limit: int = 12) -> list[str]:
    return names if len(names) <= limit else names[:limit] + [f"... +{len(names) - limit} more"]


QUANTILES = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def feature_stats(x: np.ndarray, quantiles: bool = False) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    out = {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(),
           "std": x.std(0).tolist(), "count": [int(x.shape[0])]}
    if quantiles:
        out.update({k: np.quantile(x, q, axis=0).tolist() for k, q in QUANTILES.items()})
    return out
