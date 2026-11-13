"""End-to-end pipeline: videos -> human motion -> robot trajectories -> datasets."""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from .dataset import ACTION_MODES, RobotDataset, UntrackedError, action_names, build_episode
from .export import FORMATS, export
from .extract import get_extractor
from .process import clean_motion
from .retarget import EMBODIMENTS, RetargetConfig, retarget
from .schema import HumanMotion

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
    embodiments: list[str] = field(default_factory=lambda: list(EMBODIMENTS))
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


def run_pipeline(videos: list[str], out_dir: str | Path, cfg: PipelineConfig | None = None,
                 progress: ProgressFn | None = None, tasks: list[str] | None = None) -> dict:
    """Process ``videos`` (one episode each) and write every requested dataset.

    Returns a JSON-serialisable summary, also written to ``out_dir/summary.json``.
    Output layout::

        out_dir/motion/episode_000.npz                 tracked human motion
        out_dir/datasets/<embodiment>/<format>/...     exported datasets
    """
    cfg = cfg or PipelineConfig()
    cfg.validate()
    out_dir = Path(out_dir)
    (out_dir / "motion").mkdir(parents=True, exist_ok=True)
    report = progress or (lambda p, msg: None)
    t0 = time.time()
    warnings: list[str] = []

    # 1) tracking -------------------------------------------------------------
    kwargs = {"pose_model": cfg.pose_model} if cfg.extractor == "mediapipe" else {}
    extractor = get_extractor(cfg.extractor, **kwargs)
    motions: list[HumanMotion] = []
    clips = []
    for i, video in enumerate(videos):
        def sub(p, msg, i=i):
            report(0.7 * (i + p) / len(videos), f"[{i + 1}/{len(videos)}] {msg}")
        raw = extractor.extract(video, target_fps=cfg.target_fps, start_s=cfg.start_s,
                                end_s=cfg.end_s, progress=sub)
        motion = clean_motion(raw, max_gap_s=cfg.max_gap_s, smooth_s=cfg.smooth_s)
        motion.save(out_dir / "motion" / f"episode_{i:03d}.npz")
        motions.append(motion)
        cov = raw.coverage()
        clips.append({"video": str(video), "frames": motion.num_frames, "fps": motion.fps,
                      "duration_s": motion.num_frames / motion.fps, "coverage": cov})
        if cov["body"] < 0.5:
            warnings.append(f"{Path(video).name}: body tracked in only {cov['body']:.0%} of frames")

    # 2) retarget + 3) export -------------------------------------------------
    rcfg = RetargetConfig(rotation=cfg.rotation, scale=cfg.scale, side=cfg.side)
    datasets = []
    jobs = [(e, f) for e in cfg.embodiments for f in cfg.formats]
    done = 0
    for emb in cfg.embodiments:
        ds = None
        for i, (video, motion) in enumerate(zip(videos, motions)):
            traj = retarget(motion, emb, rcfg)
            if ds is None:
                ds = RobotDataset(
                    name=f"{cfg.name}_{emb}", embodiment=emb, fps=motion.fps, state_names=traj.names,
                    action_names=action_names(traj.names, cfg.action_mode, traj.absolute_dims),
                    action_mode=cfg.action_mode, image_size=tuple(cfg.image_size),
                    include_images=cfg.include_images,
                    info={"rotation": cfg.rotation, "scale": cfg.scale, "source": "human_video"})
            has_video = cfg.extractor != "synthetic" or Path(video).exists()
            try:
                ds.episodes.append(build_episode(
                    traj, task=(tasks[i] if tasks else cfg.task), action_mode=cfg.action_mode,
                    video_path=str(video) if has_video else None,
                    start_frame=int(round(motion.meta.get("start_s", 0.0) * motion.meta.get("source_fps", motion.fps))),
                    stride=int(motion.meta.get("stride", 1)), min_tracked=cfg.min_tracked))
                ds.episodes[-1].meta["clip"] = i
            except UntrackedError as e:
                warnings.append(f"{Path(video).name}: skipped for {emb} — {e}")
        entry = {"embodiment": emb, "description": EMBODIMENTS[emb], "exports": {}, "errors": {}}
        if ds is None or not ds.episodes:
            entry["errors"]["all"] = "No clip had enough tracking for this embodiment"
            done += len(cfg.formats)
        else:
            entry.update(ds.describe())
            entry["episodes"] = [{"length": len(e), "task": e.task, **e.meta} for e in ds.episodes]
            preview = {"state_names": ds.state_names, "fps": ds.fps, "episodes": [
                {"clip": e.meta["clip"],
                 "offset": e.meta.get("trimmed_start", 0),
                 "state": [[round(float(v), 4) for v in row] for row in e.state]}
                for e in ds.episodes]}
            (out_dir / "datasets" / emb).mkdir(parents=True, exist_ok=True)
            (out_dir / "datasets" / emb / "preview.json").write_text(json.dumps(preview))
            for fmt in cfg.formats:
                report(0.7 + 0.3 * done / len(jobs), f"Exporting {emb} → {fmt}")
                target = out_dir / "datasets" / emb / fmt
                try:
                    export(ds, fmt, target)
                    entry["exports"][fmt] = str(target.relative_to(out_dir))
                except Exception as e:  # one failing format must not sink the rest
                    entry["errors"][fmt] = f"{type(e).__name__}: {e}"
                    warnings.append(f"{emb}/{fmt} export failed: {type(e).__name__}: {e}")
                    (target / "error.txt").parent.mkdir(parents=True, exist_ok=True)
                    (target / "error.txt").write_text(traceback.format_exc())
                done += 1
        datasets.append(entry)

    summary = {"config": asdict(cfg), "clips": clips, "datasets": datasets, "warnings": warnings,
               "elapsed_s": round(time.time() - t0, 2)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    report(1.0, "Done")
    return summary
