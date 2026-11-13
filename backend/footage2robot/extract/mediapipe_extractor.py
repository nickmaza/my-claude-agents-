"""MediaPipe Tasks based body + hand extractor."""

from __future__ import annotations

import numpy as np

from ..schema import B, HumanMotion
from ..video import frame_stride, iter_frames, probe
from .base import Extractor, ProgressFn
from .models import ensure_model


class MediaPipeExtractor(Extractor):
    """Runs the MediaPipe PoseLandmarker and HandLandmarker on every frame.

    Hands are assigned to the person's anatomical left/right by matching each
    detected hand's wrist to the body model's wrists in the image. That is more
    reliable than MediaPipe's handedness label, which assumes a mirrored
    (selfie) image.
    """

    name = "mediapipe"

    def __init__(self, pose_model: str = "full", min_confidence: float = 0.5,
                 mirrored: bool = False):
        self.pose_model = f"pose_{pose_model}"
        self.min_confidence = min_confidence
        self.mirrored = mirrored

    def _make_landmarkers(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        running = vision.RunningMode.VIDEO
        pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(self.pose_model))),
            running_mode=running, num_poses=1,
            min_pose_detection_confidence=self.min_confidence,
            min_pose_presence_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
        ))
        hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model("hand"))),
            running_mode=running, num_hands=2,
            min_hand_detection_confidence=self.min_confidence,
            min_hand_presence_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
        ))
        return mp, pose, hand

    def extract(self, video_path: str, *, target_fps: float | None = None,
                start_s: float = 0.0, end_s: float | None = None,
                progress: ProgressFn | None = None) -> HumanMotion:
        info = probe(video_path)
        stride = frame_stride(info.fps, target_fps)
        out_fps = info.fps / stride
        end = min(end_s, info.duration) if end_s is not None else info.duration
        expected = max(1, int((end - start_s) * info.fps / stride) + 1)

        mp, pose_lm, hand_lm = self._make_landmarkers()
        frames: list[dict] = []
        last_ts = -1
        try:
            for i, (_, t, rgb) in enumerate(iter_frames(video_path, stride, start_s, end_s)):
                ts_ms = max(int(round(t * 1000)), last_ts + 1)  # must be strictly increasing
                last_ts = ts_ms
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                frames.append(self._read_frame(pose_lm.detect_for_video(image, ts_ms),
                                               hand_lm.detect_for_video(image, ts_ms)))
                if progress and i % 10 == 0:
                    progress(min(0.99, (i + 1) / expected), f"Tracking frame {i + 1}/~{expected}")
        finally:
            pose_lm.close()
            hand_lm.close()

        motion = HumanMotion.empty(len(frames), out_fps, info.width, info.height, source=str(video_path))
        for i, f in enumerate(frames):
            if f["body_3d"] is not None:
                motion.body_3d[i] = f["body_3d"]
                motion.body_2d[i] = f["body_2d"]
                motion.body_conf[i] = f["body_conf"]
            for side, hand in f["hands"].items():
                motion.hands_3d[i, side] = hand["3d"]
                motion.hands_2d[i, side] = hand["2d"]
                motion.hands_conf[i, side] = hand["conf"]
        motion.meta.update({"extractor": self.name, "pose_model": self.pose_model,
                            "source_fps": info.fps, "stride": stride, "start_s": start_s})
        if progress:
            progress(1.0, "Tracking complete")
        return motion

    def _read_frame(self, pose_res, hand_res) -> dict:
        out: dict = {"body_3d": None, "hands": {}}
        if pose_res.pose_world_landmarks:
            world = pose_res.pose_world_landmarks[0]
            img = pose_res.pose_landmarks[0]
            out["body_3d"] = np.array([[p.x, p.y, p.z] for p in world], np.float32)
            out["body_2d"] = np.array([[p.x, p.y] for p in img], np.float32)
            out["body_conf"] = np.array([_vis(p) for p in img], np.float32)

        detections = []
        for k, lms in enumerate(hand_res.hand_landmarks or []):
            label = hand_res.handedness[k][0].category_name.lower() if hand_res.handedness else ""
            score = hand_res.handedness[k][0].score if hand_res.handedness else 1.0
            detections.append({
                "2d": np.array([[p.x, p.y] for p in lms], np.float32),
                "3d": np.array([[p.x, p.y, p.z] for p in hand_res.hand_world_landmarks[k]], np.float32),
                "conf": float(score), "label": label,
            })
        for side, det in self._assign_hands(detections, out.get("body_2d")).items():
            out["hands"][side] = det
        return out

    def _assign_hands(self, dets: list[dict], body_2d: np.ndarray | None) -> dict[int, dict]:
        if not dets:
            return {}
        if body_2d is not None:
            wrists = body_2d[[B["left_wrist"], B["right_wrist"]]]
            cost = np.array([[np.linalg.norm(d["2d"][0] - w) for w in wrists] for d in dets])
            if len(dets) == 1:
                return {int(np.argmin(cost[0])): dets[0]}
            a = cost[0, 0] + cost[1, 1]
            b = cost[0, 1] + cost[1, 0]
            return {0: dets[0], 1: dets[1]} if a <= b else {0: dets[1], 1: dets[0]}
        # No body: fall back to MediaPipe's label (which assumes a mirrored image).
        result = {}
        for d in dets:
            is_left = d["label"] == "left"
            side = 0 if (is_left == self.mirrored) else 1
            if side not in result or d["conf"] > result[side]["conf"]:
                result[side] = d
        return result


def _vis(p) -> float:
    v = getattr(p, "visibility", None)
    pr = getattr(p, "presence", None)
    vals = [x for x in (v, pr) if x is not None]
    return float(min(vals)) if vals else 1.0
