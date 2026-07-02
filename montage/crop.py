"""Smart 9:16 reframe.

For each EDL clip, decide a crop window. Strategy:
  1. Sample frames; detect faces (MediaPipe primary, OpenCV Haar fallback).
  2. If a dominant face is found, center crop horizontally on its centroid.
  3. Else, use the crop_hint from EDLClip ('face_left' etc.) or center.
  4. Emit an ffmpeg filter expression per clip.

Output target: 1080 x 1920 (9:16) regardless of source aspect.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from montage.schemas import EDLClip, MediaAsset

TARGET_W = 1080
TARGET_H = 1920
TARGET_AR = TARGET_W / TARGET_H  # 0.5625


def _load_mediapipe():
    try:
        import mediapipe as mp  # type: ignore
        return mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.4
        )
    except (ImportError, AttributeError):
        return None


_MP_DETECTOR = _load_mediapipe()
_HAAR_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


@dataclass
class CropPlan:
    src_w: int
    src_h: int
    crop_x: int
    crop_y: int
    crop_w: int
    crop_h: int

    def ffmpeg_filter(self) -> str:
        return (
            f"crop={self.crop_w}:{self.crop_h}:{self.crop_x}:{self.crop_y},"
            f"scale={TARGET_W}:{TARGET_H}:flags=lanczos,setsar=1"
        )


def _sample_frame(
    video_path: Path, t_s: float, rotation: int = 0
) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    # OpenCV does not honor stream rotation. Apply manually.
    if rotation == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == -90 or rotation == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif abs(rotation) == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def _detect_face_box_mediapipe(frame: np.ndarray) -> tuple[int, int] | None:
    if _MP_DETECTOR is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    results = _MP_DETECTOR.process(rgb)
    if not results.detections:
        return None
    best = max(results.detections, key=lambda d: d.score[0] if d.score else 0)
    bbox = best.location_data.relative_bounding_box
    cx = (bbox.xmin + bbox.width / 2) * w
    cy = (bbox.ymin + bbox.height / 2) * h
    return int(cx), int(cy)


def _detect_face_box_haar(frame: np.ndarray) -> tuple[int, int] | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _HAAR_CASCADE.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(48, 48)
    )
    if len(faces) == 0:
        return None
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    return int(fx + fw / 2), int(fy + fh / 2)


def _detect_face_box(frame: np.ndarray) -> tuple[int, int] | None:
    """Try MediaPipe; fall back to Haar. Returns (cx, cy) of dominant face."""
    box = _detect_face_box_mediapipe(frame)
    if box is not None:
        return box
    return _detect_face_box_haar(frame)


def _hint_anchor_x(src_w: int, hint: str) -> int:
    if hint == "face_left":
        return int(src_w * 0.30)
    if hint == "face_right":
        return int(src_w * 0.70)
    return src_w // 2


def plan_crop_for_clip(
    clip: EDLClip, asset: MediaAsset, samples: int = 3
) -> CropPlan:
    src_w, src_h = asset.width, asset.height

    # Compute crop dimensions to hit 9:16
    src_ar = src_w / src_h
    if src_ar > TARGET_AR:
        crop_h = src_h
        crop_w = int(round(src_h * TARGET_AR))
    else:
        crop_w = src_w
        crop_h = int(round(src_w / TARGET_AR))

    anchor_x = _hint_anchor_x(src_w, clip.crop_hint)
    anchor_y = src_h // 2

    if asset.type == "video":
        in_s, out_s = clip.in_s, clip.out_s
        ts = np.linspace(in_s, max(in_s, out_s - 0.01), samples)
        # Probe rotation once
        from montage.ingest import _ffprobe, _rotation
        try:
            info = _ffprobe(asset.path)
            vs = next(s for s in info["streams"] if s["codec_type"] == "video")
            rotation = _rotation(vs)
        except Exception:
            rotation = 0

        xs: list[int] = []
        ys: list[int] = []
        for t in ts:
            frame = _sample_frame(asset.path, float(t), rotation=rotation)
            if frame is None:
                continue
            face = _detect_face_box(frame)
            if face is not None:
                fcx, fcy = face
                fh, fw = frame.shape[:2]
                xs.append(int(fcx * src_w / fw))
                ys.append(int(fcy * src_h / fh))
        if xs:
            anchor_x = int(np.median(xs))
        if ys:
            # Faces typically belong in the upper third for portrait framing.
            # Use the detected y centroid directly.
            anchor_y = int(np.median(ys))

    crop_x = max(0, min(src_w - crop_w, anchor_x - crop_w // 2))
    crop_y = max(0, min(src_h - crop_h, anchor_y - crop_h // 2))

    return CropPlan(
        src_w=src_w, src_h=src_h,
        crop_x=crop_x, crop_y=crop_y, crop_w=crop_w, crop_h=crop_h,
    )


def plan_crops(
    clips: list[EDLClip], assets_by_path: dict[Path, MediaAsset]
) -> list[CropPlan]:
    return [plan_crop_for_clip(c, assets_by_path[c.asset_path]) for c in clips]
