"""End-to-end pipeline: videos -> human motion -> robot trajectories -> datasets.

Two stages, so edits never repeat the expensive part:

``track``   video -> tracked, cleaned, calibrated human motion (saved per clip),
            plus proposed episode boundaries and an object prompt.
``build``   saved motion + episode boundaries -> retargeted episodes -> exports.
            Re-run on its own after the user trims episodes or picks an object.
"""

from __future__ import annotations

import json
import shutil
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import calibrate
from .dataset import ACTION_MODES, RobotDataset, UntrackedError, action_names, build_episode
from .export import FORMATS, export
from .extract import get_extractor
from .process import clean_motion
from .process.filters import fill_gaps, smooth
from .retarget import EMBODIMENTS, GENERIC_EMBODIMENTS, RetargetConfig, retarget
from .schema import HumanMotion
from .segment import auto_split, validate_segments

ProgressFn = Callable[[float, str], None]


@dataclass
class PipelineConfig:
    task: str = "do the demonstrated task"
    name: str = "footage2robot"
    extractor: str = "mediapipe"            # mediapipe | synthetic
    pose_model: str = "full"                # lite | full | heavy
    target_fps: float | None = 30.0         # None = keep source fps
    start_s: float = 0.0
    end_s: float | None = None
    embodiments: list[str] = field(default_factory=lambda: list(GENERIC_EMBODIMENTS))
    formats: list[str] = field(default_factory=lambda: list(FORMATS))
    action_mode: str = "absolute"
    rotation: str = "euler"
    side: str = "auto"
    scale: float = 1.0
    smooth_s: float = 0.05
    max_gap_s: float = 0.25
    min_tracked: float = 0.5
    include_images: bool = True
    image_size: tuple[int, int] = (224, 224)
    # calibration
    person_height_m: float | None = None    # scale the skeleton to the person's real height ...
    arm_length_m: float | None = None       # ... or (more direct) their shoulder-to-wrist length
    marker_size_m: float | None = None      # look for the printed workspace marker of this edge length
    fov_deg: float = calibrate.DEFAULT_FOV_DEG
    base_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)   # marker (or torso) origin in the robot base frame
    # episodes and objects
    split: str = "none"                     # none: one episode per clip | auto: split on pauses
    track_objects: bool = False
    object_backend: str = "flow"            # flow | sam2

    def validate(self) -> None:
        for e in self.embodiments:
            if e not in EMBODIMENTS:
                raise ValueError(f"Unknown embodiment '{e}'. Options: {sorted(EMBODIMENTS)}")
        for f in self.formats:
            if f not in FORMATS:
                raise ValueError(f"Unknown format '{f}'. Options: {sorted(FORMATS)}")
        if self.action_mode not in ACTION_MODES:
            raise ValueError(f"Unknown action mode '{self.action_mode}'. Options: {sorted(ACTION_MODES)}")
        if not self.embodiments or not self.formats:
            raise ValueError("Pick at least one embodiment and one format")
        if self.split not in ("none", "auto"):
            raise ValueError("split must be 'none' or 'auto'")
        if self.object_backend not in ("flow", "sam2"):
            raise ValueError("object_backend must be 'flow' or 'sam2'")
        for name, lo, hi in (("person_height_m", 0.8, 2.5), ("arm_length_m", 0.25, 1.0), ("marker_size_m", 0.02, 1.0)):
            v = getattr(self, name)
            if v is not None and not lo <= float(v) <= hi:
                raise ValueError(f"{name} must be within {lo}..{hi} metres")


def _motion_path(out_dir: Path, i: int) -> Path:
    return out_dir / "motion" / f"episode_{i:03d}.npz"


def _fill_all(x: np.ndarray, fps: float) -> np.ndarray:
    """Interpolate every interior gap and smooth (camera-space body position is slow-moving)."""
    return smooth(fill_gaps(x, max_gap=len(x)), 0.15 * fps)


# --- stage 1 ------------------------------------------------------------------

def track(videos: list[str], out_dir: str | Path, cfg: PipelineConfig,
          progress: ProgressFn | None = None) -> dict:
    """Track every clip and save its motion. Returns (and writes) ``tracking.json``."""
    cfg.validate()
    out_dir = Path(out_dir)
    (out_dir / "motion").mkdir(parents=True, exist_ok=True)
    report = progress or (lambda p, msg: None)
    kwargs = {"pose_model": cfg.pose_model} if cfg.extractor == "mediapipe" else {}
    extractor = get_extractor(cfg.extractor, **kwargs)
    clips, warnings = [], []
    for i, video in enumerate(videos):
        def sub(p, msg, i=i):
            report((i + 0.95 * p) / len(videos), f"[{i + 1}/{len(videos)}] {msg}")
        raw = extractor.extract(video, target_fps=cfg.target_fps, start_s=cfg.start_s,
                                end_s=cfg.end_s, progress=sub)
        motion = clean_motion(raw, max_gap_s=cfg.max_gap_s, smooth_s=cfg.smooth_s)
        has_video = Path(video).is_file()
        name = Path(video).name

        scale, calibration = calibrate.body_scale(motion, cfg.person_height_m, cfg.arm_length_m)
        calibrate.apply_scale(motion, scale)
        if (cfg.person_height_m or cfg.arm_length_m) and "method" not in calibration:
            warnings.append(f"{name}: could not measure the body, scale left at 1.0")

        calibration["marker"] = False
        if cfg.marker_size_m and has_video:
            sub(0.97, "Looking for the workspace marker")
            cam = calibrate.Camera(motion.width, motion.height, cfg.fov_deg)
            marker = calibrate.find_marker(video, cfg.marker_size_m, cam)
            root = calibrate.root_translation(motion, cam)
            if marker is None:
                warnings.append(f"{name}: workspace marker not found, using the body frame instead")
            elif np.isfinite(root[:, 0]).sum() < 2:
                warnings.append(f"{name}: body not visible enough to place it relative to the marker")
            else:
                root = _fill_all(root, motion.fps)
                motion.world_R = np.array(marker["R"], np.float32)
                motion.world_origin = (np.array(marker["t"]) - root).astype(np.float32)
                calibration.update(marker=True, marker_detections=marker["detections"],
                                   marker_distance_m=round(float(np.linalg.norm(marker["t"])), 3))

        motion.save(_motion_path(out_dir, i))
        cov = raw.coverage()
        if cov["body"] < 0.5:
            warnings.append(f"{name}: body tracked in only {cov['body']:.0%} of frames")
        clip = {
            "video": str(video), "has_video": has_video, "frames": motion.num_frames, "fps": motion.fps,
            "duration_s": motion.num_frames / motion.fps, "coverage": cov, "calibration": calibration,
            "segments": auto_split(motion) if cfg.split == "auto" else [[0, motion.num_frames]],
            "object_prompt": None,
        }
        if cfg.track_objects and has_video:
            from .objects import auto_prompt
            prompt = auto_prompt(motion, cfg.side)
            clip["object_prompt"] = list(prompt) if prompt else None
            if prompt is None:
                warnings.append(f"{name}: no grasp found to locate the object — click it in the editor to track it")
        clips.append(clip)
    tracking = {"clips": clips, "warnings": warnings}
    (out_dir / "tracking.json").write_text(json.dumps(tracking, indent=2))
    report(1.0, "Tracking complete")
    return tracking


# --- stage 2 ------------------------------------------------------------------

def _object_track(out_dir: Path, i: int, clip: dict, motion: HumanMotion, cfg: PipelineConfig,
                  warnings: list[str]) -> np.ndarray | None:
    """Tracked object box for a clip, cached on disk per prompt."""
    prompt = clip.get("object_prompt")
    if not (cfg.track_objects and prompt and clip["has_video"]):
        return None
    cache = out_dir / "motion" / f"object_{i:03d}.npz"
    key = json.dumps([prompt, cfg.object_backend])
    if cache.exists():
        with np.load(cache) as z:
            if str(z["key"]) == key:
                return z["boxes"]
    from .objects import track_object
    start = int(round(motion.meta.get("start_s", 0.0) * motion.meta.get("source_fps", motion.fps)))
    frames = start + np.arange(motion.num_frames) * int(motion.meta.get("stride", 1))
    try:
        boxes = track_object(clip["video"], frames, tuple(prompt), cfg.object_backend)
    except Exception as e:
        warnings.append(f"{Path(clip['video']).name}: object tracking failed — {type(e).__name__}: {e}")
        return None
    if boxes[:, 4].mean() < 0.2:
        warnings.append(f"{Path(clip['video']).name}: the object could not be followed — pick a clearer point on it")
        return None
    np.savez_compressed(cache, key=np.array(key), boxes=boxes)
    return boxes


def build(out_dir: str | Path, cfg: PipelineConfig, progress: ProgressFn | None = None) -> dict:
    """Retarget and export from saved tracking. Returns (and writes) ``summary.json``."""
    cfg.validate()
    out_dir = Path(out_dir)
    report = progress or (lambda p, msg: None)
    t0 = time.time()
    tracking = json.loads((out_dir / "tracking.json").read_text())
    clips = tracking["clips"]
    warnings = list(tracking.get("warnings", []))
    shutil.rmtree(out_dir / "datasets", ignore_errors=True)

    motions = [HumanMotion.load(_motion_path(out_dir, i)) for i in range(len(clips))]
    objects = []
    for i, (clip, motion) in enumerate(zip(clips, motions)):
        if cfg.track_objects and clip.get("object_prompt"):
            report(0.1 * i / len(clips), f"Tracking the object in clip {i + 1}")
        objects.append(_object_track(out_dir, i, clip, motion, cfg, warnings))
        clip["object_tracked"] = objects[-1] is not None
    use_objects = cfg.track_objects and any(o is not None for o in objects)

    rcfg = RetargetConfig(rotation=cfg.rotation, scale=cfg.scale, side=cfg.side, base_offset=tuple(cfg.base_offset))
    datasets = []
    total, done = len(cfg.embodiments) * (len(cfg.formats) + 1), 0
    for emb in cfg.embodiments:
        report(0.1 + 0.9 * done / total, f"Retargeting to {emb}")
        ds = None
        for i, (clip, motion) in enumerate(zip(clips, motions)):
            name = Path(clip["video"]).name
            try:
                traj = retarget(motion, emb, rcfg)
            except Exception as e:
                warnings.append(f"{name}: retargeting to {emb} failed — {type(e).__name__}: {e}")
                continue
            if ds is None:
                from .objects import OBJECT_STATE_NAMES
                ds = RobotDataset(
                    name=f"{cfg.name}_{emb}", embodiment=emb, fps=motion.fps, state_names=traj.names,
                    action_names=action_names(traj.names, cfg.action_mode, traj.absolute_dims),
                    action_mode=cfg.action_mode, image_size=tuple(cfg.image_size),
                    include_images=cfg.include_images,
                    env_state_names=OBJECT_STATE_NAMES if use_objects else [],
                    info={"rotation": cfg.rotation, "scale": cfg.scale, "source": "human_video"})
            env = objects[i]
            if use_objects and env is None:      # keep the feature present in every episode
                env = np.zeros((motion.num_frames, 5), np.float32)
            for k, (a, b) in enumerate(clip["segments"]):
                try:
                    ep = build_episode(
                        traj, task=cfg.task, action_mode=cfg.action_mode,
                        video_path=clip["video"] if clip["has_video"] else None,
                        start_frame=int(round(motion.meta.get("start_s", 0.0) * motion.meta.get("source_fps", motion.fps))),
                        stride=int(motion.meta.get("stride", 1)), min_tracked=cfg.min_tracked,
                        frame_range=(a, b), env_state=env if use_objects else None)
                except UntrackedError as e:
                    label = name if len(clip["segments"]) == 1 else f"{name} episode {k + 1}"
                    warnings.append(f"{label}: skipped for {emb} — {e}")
                    continue
                ep.meta.update(clip=i, segment=k)
                ds.episodes.append(ep)
            reach = traj.info.get("ik_reachable_fraction")
            if reach is not None and reach < 0.9:
                warnings.append(f"{name}: {emb} can only reach {reach:.0%} of this motion — "
                                "the hand moves outside this robot's workspace")
        done += 1
        entry = {"embodiment": emb, "description": EMBODIMENTS[emb], "exports": {}, "errors": {}}
        if ds is None or not ds.episodes:
            entry["errors"]["all"] = "No clip had enough tracking for this embodiment"
            done += len(cfg.formats)
        else:
            entry.update(ds.describe())
            entry["episodes"] = [{"length": len(e), "task": e.task, **e.meta} for e in ds.episodes]
            preview = {"state_names": ds.state_names, "fps": ds.fps, "episodes": [
                {"clip": e.meta["clip"], "offset": e.meta.get("trimmed_start", 0),
                 "state": [[round(float(v), 4) for v in row] for row in e.state]}
                for e in ds.episodes]}
            (out_dir / "datasets" / emb).mkdir(parents=True, exist_ok=True)
            (out_dir / "datasets" / emb / "preview.json").write_text(json.dumps(preview))
            for fmt in cfg.formats:
                report(0.1 + 0.9 * done / total, f"Exporting {emb} → {fmt}")
                target = out_dir / "datasets" / emb / fmt
                try:
                    export(ds, fmt, target)
                    entry["exports"][fmt] = str(target.relative_to(out_dir))
                except Exception as e:  # one failing format must not sink the rest
                    entry["errors"][fmt] = f"{type(e).__name__}: {e}"
                    warnings.append(f"{emb}/{fmt} export failed: {type(e).__name__}: {e}")
                    target.mkdir(parents=True, exist_ok=True)
                    (target / "error.txt").write_text(traceback.format_exc())
                done += 1
        datasets.append(entry)

    summary = {"config": asdict(cfg), "clips": clips, "datasets": datasets, "warnings": warnings,
               "elapsed_s": round(time.time() - t0, 2)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    report(1.0, "Done")
    return summary


def apply_edits(out_dir: str | Path, edits: dict) -> None:
    """Store user edits ({clip index: {"segments": [...], "object_prompt": [f, x, y] | None}})."""
    out_dir = Path(out_dir)
    tracking = json.loads((out_dir / "tracking.json").read_text())
    for key, edit in edits.items():
        i = int(key)
        if not 0 <= i < len(tracking["clips"]):
            raise ValueError(f"No clip {i}")
        clip = tracking["clips"][i]
        if "segments" in edit:
            clip["segments"] = validate_segments(edit["segments"], clip["frames"])
        if "object_prompt" in edit:
            p = edit["object_prompt"]
            if p is not None:
                if not (isinstance(p, (list, tuple)) and len(p) == 3 and 0 <= int(p[0]) < clip["frames"]
                        and 0 <= float(p[1]) <= 1 and 0 <= float(p[2]) <= 1):
                    raise ValueError("object_prompt must be [frame, x, y] with x, y in 0..1")
                p = [int(p[0]), float(p[1]), float(p[2])]
            clip["object_prompt"] = p
    (out_dir / "tracking.json").write_text(json.dumps(tracking, indent=2))


def run_pipeline(videos: list[str], out_dir: str | Path, cfg: PipelineConfig | None = None,
                 progress: ProgressFn | None = None) -> dict:
    """Track ``videos`` and write every requested dataset.

    Output layout::

        out_dir/tracking.json, summary.json
        out_dir/motion/episode_000.npz                 tracked human motion per clip
        out_dir/datasets/<embodiment>/<format>/...     exported datasets
    """
    cfg = cfg or PipelineConfig()
    report = progress or (lambda p, msg: None)
    t0 = time.time()
    track(videos, out_dir, cfg, lambda p, msg: report(0.7 * p, msg))
    summary = build(out_dir, cfg, lambda p, msg: report(0.7 + 0.3 * p, msg))
    summary["elapsed_s"] = round(time.time() - t0, 2)
    (Path(out_dir) / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
