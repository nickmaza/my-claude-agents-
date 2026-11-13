"""Command line interface: ``footage2robot convert|serve|worker|marker|formats``."""

from __future__ import annotations

import argparse
import json
import sys

from .export import FORMATS
from .pipeline import PipelineConfig, run_pipeline
from .retarget import EMBODIMENTS, GENERIC_EMBODIMENTS


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _self_signed(data_dir: str) -> dict:
    """Create (once) a self-signed certificate so phones on the LAN can use the camera."""
    import os
    import shutil
    import subprocess

    cert, key = os.path.join(data_dir, "selfsigned.crt"), os.path.join(data_dir, "selfsigned.key")
    if not (os.path.exists(cert) and os.path.exists(key)):
        if not shutil.which("openssl"):
            raise SystemExit("--https needs the openssl command, or pass --ssl-certfile/--ssl-keyfile")
        os.makedirs(data_dir, exist_ok=True)
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "365",
                        "-subj", "/CN=footage2robot", "-keyout", key, "-out", cert],
                       check=True, capture_output=True)
        os.chmod(key, 0o600)
    return {"ssl_certfile": cert, "ssl_keyfile": key}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="footage2robot", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="Convert videos into robot datasets")
    c.add_argument("videos", nargs="+", help="Video files (one demonstration episode each)")
    c.add_argument("-o", "--out", required=True, help="Output directory")
    c.add_argument("--task", default="do the demonstrated task", help="Language instruction for the episodes")
    c.add_argument("--name", default="footage2robot")
    c.add_argument("--embodiments", type=_csv, default=list(GENERIC_EMBODIMENTS),
                   help=f"Comma list of: {', '.join(EMBODIMENTS)}, or 'all'")
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
    c.add_argument("--height", type=float, default=None, help="Person's height in metres (metric scale)")
    c.add_argument("--arm-length", type=float, default=None, help="Shoulder-to-wrist length in metres (metric scale)")
    c.add_argument("--marker-size", type=float, default=None,
                   help="Edge length in metres of the printed workspace marker (`footage2robot marker`)")
    c.add_argument("--base-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
                   help="Marker origin in the robot base frame, metres")
    c.add_argument("--split", default="none", choices=["none", "auto"], help="Split clips into episodes on pauses")
    c.add_argument("--track-objects", action="store_true", help="Follow the handled object in the image")
    c.add_argument("--object-backend", default="flow", choices=["flow", "sam2"])
    c.add_argument("--urdf", default=None, help="Custom robot arm URDF (adds the 'custom' embodiment)")
    c.add_argument("--base-link", default=None)
    c.add_argument("--ee-link", default=None)

    s = sub.add_parser("serve", help="Run the web app / API server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--data-dir", default=None, help="Where jobs are stored (default ./f2r_data)")
    s.add_argument("--workers", type=int, default=1, help="Worker processes running conversions")
    s.add_argument("--https", action="store_true",
                   help="Serve over HTTPS with a self-signed certificate (lets phones on your network use the camera)")
    s.add_argument("--ssl-certfile", default=None)
    s.add_argument("--ssl-keyfile", default=None)

    w = sub.add_parser("worker", help="Run a conversion worker (pulls jobs from the queue)")
    w.add_argument("--data-dir", default=None)

    m = sub.add_parser("marker", help="Write the printable workspace marker")
    m.add_argument("-o", "--out", default="footage2robot_marker.png")

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

    if args.cmd == "marker":
        import cv2

        from .calibrate import marker_image
        cv2.imwrite(args.out, marker_image())
        print(f"Wrote {args.out}. Print it, measure the black square's edge, and pass that as --marker-size.")
        return 0

    if args.cmd in ("serve", "worker"):
        import os
        if args.data_dir:
            os.environ["FOOTAGE2ROBOT_DATA_DIR"] = os.path.abspath(args.data_dir)

    if args.cmd == "worker":
        from pathlib import Path

        from .api.store import JobStore
        from .api.worker import run_worker
        run_worker(JobStore(Path(os.environ.get("FOOTAGE2ROBOT_DATA_DIR", "f2r_data")).resolve()))
        return 0

    if args.cmd == "serve":
        import subprocess

        import uvicorn
        procs = []
        if args.workers > 0:
            os.environ["FOOTAGE2ROBOT_EMBEDDED_WORKER"] = "0"
            procs = [subprocess.Popen([sys.executable, "-m", "footage2robot.cli", "worker"])
                     for _ in range(args.workers)]
        ssl = {}
        if args.ssl_certfile and args.ssl_keyfile:
            ssl = {"ssl_certfile": args.ssl_certfile, "ssl_keyfile": args.ssl_keyfile}
        elif args.https:
            ssl = _self_signed(os.environ.get("FOOTAGE2ROBOT_DATA_DIR", "f2r_data"))
        try:
            uvicorn.run("footage2robot.api.app:app", host=args.host, port=args.port, **ssl)
        finally:
            for proc in procs:
                proc.terminate()
        return 0

    if args.embodiments == ["all"]:
        args.embodiments = list(EMBODIMENTS)
    if args.urdf:
        if not (args.base_link and args.ee_link):
            p.error("--urdf needs --base-link and --ee-link")
        from .retarget import register_robot
        from .robots import custom_profile
        register_robot(custom_profile(args.urdf, args.base_link, args.ee_link))
        args.embodiments = [*args.embodiments, "custom"]

    cfg = PipelineConfig(
        task=args.task, name=args.name, extractor=args.extractor, pose_model=args.pose_model,
        target_fps=args.fps or None, start_s=args.start, end_s=args.end,
        embodiments=args.embodiments, formats=args.formats, action_mode=args.action_mode,
        rotation=args.rotation, side=args.side, scale=args.scale,
        include_images=not args.no_images, image_size=tuple(args.image_size),
        person_height_m=args.height, arm_length_m=args.arm_length, marker_size_m=args.marker_size,
        base_offset=tuple(args.base_offset), split=args.split, track_objects=args.track_objects,
        object_backend=args.object_backend)

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
