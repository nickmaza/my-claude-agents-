"""Track the object being handled, in image space.

Given one point on the object (clicked by the user, or guessed from where the
fingers close), the object is segmented and followed through the clip. The
result is a per-frame box ``[cx, cy, w, h, visible]`` in normalised image
coordinates, exported as ``observation.environment_state``.

Backends
--------
``flow`` (default, CPU)
    MediaPipe's interactive segmenter re-run on every frame, with the prompt
    point carried along by optical flow. Good for distinct, rigid objects;
    it can drift under heavy occlusion by the hand.
``sam2`` (optional, GPU recommended)
    Meta's SAM 2 video predictor (``pip install sam2``; weights download from
    Hugging Face on first use). Much more robust to occlusion.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .retarget import RetargetConfig, pick_side
from .retarget.hands import gripper_closedness
from .schema import H, HumanMotion

OBJECT_STATE_NAMES = ["object_cx", "object_cy", "object_w", "object_h", "object_visible"]
MAX_SIDE = 384


def auto_prompt(motion: HumanMotion, side: str = "auto") -> tuple[int, float, float] | None:
    """Guess ``(frame, x, y)`` on the object: between the fingertips just before the first grasp."""
    s = pick_side(motion, RetargetConfig(side=side))
    closed = np.nan_to_num(gripper_closedness(motion, s), nan=0.0)
    onsets = np.where((closed[1:] >= 0.5) & (closed[:-1] < 0.5))[0]
    if len(onsets) == 0:
        return None
    frame = max(0, int(onsets[0]) - int(round(0.3 * motion.fps)))
    hand = motion.hands_2d[frame, s]
    if not np.isfinite(hand).all():
        return None
    mid = 0.5 * (hand[H["thumb_tip"]] + hand[H["index_tip"]])
    point = mid + 0.15 * (mid - hand[H["wrist"]])        # a little beyond the fingertips
    if not ((0 <= point) & (point <= 1)).all():
        return None
    return frame, float(point[0]), float(point[1])


def _load_frames(video_path: str, source_frames: np.ndarray) -> list[np.ndarray]:
    import cv2

    from .dataset import Episode
    ep = Episode(state=np.zeros((len(source_frames), 1), np.float32), action=np.zeros((len(source_frames), 1), np.float32),
                 video_path=video_path, source_frames=source_frames)
    frames = []
    for rgb in ep.iter_images():
        h, w = rgb.shape[:2]
        k = min(1.0, MAX_SIDE / max(h, w))
        frames.append(cv2.resize(rgb, (max(2, int(w * k)), max(2, int(h * k))), interpolation=cv2.INTER_AREA)
                      if k < 1 else rgb)
    return frames


def _box(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < 12:
        return None
    h, w = mask.shape
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return np.array([xs.mean() / w, ys.mean() / h, (x1 - x0) / w, (y1 - y0) / h, 1.0], np.float32)


def _finish(boxes: list[np.ndarray | None]) -> np.ndarray:
    """Hold the last known box through gaps, with visible = 0."""
    out = np.zeros((len(boxes), 5), np.float32)
    known = [i for i, b in enumerate(boxes) if b is not None]
    if not known:
        return out
    for i in range(len(boxes)):
        if boxes[i] is not None:
            out[i] = boxes[i]
        else:
            j = max([k for k in known if k < i], default=known[0])
            out[i, :4] = boxes[j][:4]
    return out


class FlowTracker:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        from .extract.models import ensure_model
        self.mp = mp
        # MediaPipe renamed the point-prompt segmenter to "...Legacy" in newer releases
        legacy = hasattr(vision, "InteractiveSegmenterLegacy")
        cls = vision.InteractiveSegmenterLegacy if legacy else vision.InteractiveSegmenter
        opts = vision.InteractiveSegmenterLegacyOptions if legacy else vision.InteractiveSegmenterOptions
        self.roi_cls = vision.InteractiveSegmenterLegacyRegionOfInterest if legacy \
            else vision.InteractiveSegmenterRegionOfInterest
        self.segmenter = cls.create_from_options(opts(
            base_options=BaseOptions(model_asset_path=str(ensure_model("interactive_segmenter"))),
            output_confidence_masks=False, output_category_mask=True))

    def segment(self, rgb: np.ndarray, x: float, y: float) -> np.ndarray:
        from mediapipe.tasks.python.components.containers.keypoint import NormalizedKeypoint
        roi = self.roi_cls(format=self.roi_cls.Format.KEYPOINT,
                           keypoint=NormalizedKeypoint(x=float(np.clip(x, 0, 1)), y=float(np.clip(y, 0, 1))))
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        mask = np.squeeze(self.segmenter.segment(image, roi).category_mask.numpy_view())
        h, w = mask.shape
        px, py = min(w - 1, int(x * w)), min(h - 1, int(y * h))
        return mask == mask[py, px]          # the region the prompt point belongs to

    def track(self, frames: list[np.ndarray], frame: int, x: float, y: float) -> np.ndarray:
        import cv2

        boxes: list[np.ndarray | None] = [None] * len(frames)
        first = self.segment(frames[frame], x, y)
        if first.mean() > 0.6:               # the "object" is most of the image: the prompt hit background
            return _finish(boxes)
        boxes[frame] = _box(first)
        if boxes[frame] is None:
            return _finish(boxes)
        area0 = float(first.sum())
        grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
        for step in (1, -1):
            mask, point, lost = first, np.array([x, y], np.float64), 0
            i = frame + step
            while 0 <= i < len(frames) and lost < 15:
                h, w = mask.shape
                pts = cv2.goodFeaturesToTrack(grays[i - step], 40, 0.01, 3, mask=mask.astype(np.uint8) * 255)
                if pts is not None:
                    nxt, ok, _ = cv2.calcOpticalFlowPyrLK(grays[i - step], grays[i], pts, None)
                    good = ok.reshape(-1) == 1
                    if good.sum() >= 3:
                        point = point + np.median((nxt - pts).reshape(-1, 2)[good], axis=0) / [w, h]
                cand = self.segment(frames[i], point[0], point[1])
                box = _box(cand)
                ratio = cand.sum() / max(area0, 1.0)
                if box is not None and 0.35 < ratio < 2.8:
                    boxes[i], mask, lost = box, cand, 0
                    point = 0.5 * point + 0.5 * box[:2]       # re-centre on the object
                else:
                    lost += 1
                i += step
        self.segmenter.close()
        return _finish(boxes)


def _track_sam2(frames: list[np.ndarray], frame: int, x: float, y: float, model: str) -> np.ndarray:
    """SAM 2 video predictor. Not exercised by the test-suite (needs downloaded weights)."""
    import cv2
    import torch
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = SAM2VideoPredictor.from_pretrained(model, device=device)
    boxes: list[np.ndarray | None] = [None] * len(frames)
    h, w = frames[0].shape[:2]
    with tempfile.TemporaryDirectory() as tmp, torch.inference_mode():
        for i, f in enumerate(frames):
            cv2.imwrite(str(Path(tmp) / f"{i:05d}.jpg"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        state = predictor.init_state(video_path=tmp)
        predictor.add_new_points_or_box(state, frame_idx=frame, obj_id=1,
                                        points=np.array([[x * w, y * h]], np.float32), labels=np.array([1], np.int32))
        for reverse in (False, True):
            for idx, _, logits in predictor.propagate_in_video(state, reverse=reverse):
                boxes[idx] = _box((logits[0, 0] > 0).cpu().numpy())
    return _finish(boxes)


def track_object(video_path: str, source_frames: np.ndarray, prompt: tuple[int, float, float],
                 backend: str = "flow", sam2_model: str = "facebook/sam2.1-hiera-small") -> np.ndarray:
    """Follow the object under ``prompt = (frame, x, y)``. Returns (T, 5) float32."""
    frames = _load_frames(video_path, source_frames)
    frame, x, y = int(prompt[0]), float(prompt[1]), float(prompt[2])
    frame = int(np.clip(frame, 0, len(frames) - 1))
    if backend == "sam2":
        return _track_sam2(frames, frame, x, y, sam2_model)
    if backend == "flow":
        return FlowTracker().track(frames, frame, x, y)
    raise KeyError(f"Unknown object tracking backend '{backend}'. Options: flow, sam2")
