"""Command line interface: ``footage2robot convert|serve|formats``."""

from __future__ import annotations

import argparse
import json
import sys

from .export import FORMATS
from .pipeline import PipelineConfig, run_pipeline
from .retarget import EMBODIMENTS


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="footage2robot", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="Convert videos into robot datasets")
    c.add_argument("videos", nargs="+", help="Video files (one demonstration episode each)")
    c.add_argument("-o", "--out", required=True, help="Output directory")
    c.add_argument("--task", default="do the demonstrated task", help="Language instruction for the episodes")
    c.add_argument("--name", default="footage2robot")
    c.add_argument("--embodiments", type=_csv, default=list(EMBODIMENTS), help=f"Comma list of: {', '.join(EMBODIMENTS)}")
    c.add_argument("--formats", type=_csv, default=list(FORMATS), help=f"Comma list of: {', '.join(FORMATS)}")
    c.add_argument("--fps", type=float, default=30.0, help="Target fps (0 = keep source)")
    c.add_argument("--action-mode", default="absolute", choices=["absolute", "delta"])
    c.add_argument("--rotation", default="euler", choices=["euler", "rotvec", "quat", "rot6d"])
    c.add_argument("--side", default="auto", choices=["auto", "left", "right"])
    c.add_argument("--scale", type=float, default=1.0)
    c.add_argument("--start", type=float, default=0.0)
    c.add_argument("--end", type=float, default=None)
    c.add_argument("--pose-model", default="full", choices=["lite", "full", "heavy"])
    c.add_argument("--extractor", default="mediapipe", choices=["mediapipe", "synthetic"])
    c.add_argument("--no-images", action="store_true", help="Export states/actions only")
    c.add_argument("--image-size", type=int, nargs=2, default=(224, 224), metavar=("W", "H"))

    s = sub.add_parser("serve", help="Run the web app / API server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--data-dir", default=None, help="Where jobs are stored (default ./f2r_data)")

    sub.add_parser("formats", help="List embodiments and export formats")
    args = p.parse_args(argv)

    if args.cmd == "formats":
        print("Embodiments:")
        for k, v in EMBODIMENTS.items():
            print(f"  {k:<11} {v}")
        print("Formats:")
        for k, v in FORMATS.items():
            print(f"  {k:<11} {v}")
        return 0

    if args.cmd == "serve":
        import os

        import uvicorn
        if args.data_dir:
            os.environ["FOOTAGE2ROBOT_DATA_DIR"] = args.data_dir
        uvicorn.run("footage2robot.api.app:app", host=args.host, port=args.port)
        return 0

    cfg = PipelineConfig(
        task=args.task, name=args.name, extractor=args.extractor, pose_model=args.pose_model,
        target_fps=args.fps or None, start_s=args.start, end_s=args.end,
        embodiments=args.embodiments, formats=args.formats, action_mode=args.action_mode,
        rotation=args.rotation, side=args.side, scale=args.scale,
        include_images=not args.no_images, image_size=tuple(args.image_size))

    def progress(pct: float, msg: str) -> None:
        print(f"\r[{pct:6.1%}] {msg:<60}", end="", file=sys.stderr, flush=True)

    summary = run_pipeline(args.videos, args.out, cfg, progress)
    print(file=sys.stderr)
    for d in summary["datasets"]:
        ok = ", ".join(d["exports"]) or "none"
        print(f"{d['embodiment']:<11} {d.get('num_episodes', 0)} episode(s), "
              f"{d.get('total_frames', 0)} frames, state_dim={d.get('state_dim', '-')}: {ok}")
    for w in summary["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    print(json.dumps({"out": args.out, "elapsed_s": summary["elapsed_s"]}))
    return 0 if any(d["exports"] for d in summary["datasets"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
