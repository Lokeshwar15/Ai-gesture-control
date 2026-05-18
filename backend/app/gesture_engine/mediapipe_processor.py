"""MediaPipe Hands: frame bytes or BGR image to structured landmarks."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from app.schemas.gesture import HandLandmarks, LandmarkPoint

logger = logging.getLogger(__name__)

_mp_hands = None


def _get_hands():
    global _mp_hands
    if _mp_hands is None:
        _mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _mp_hands


def decode_jpeg_to_bgr(frame_bytes: bytes) -> np.ndarray | None:
    buf = np.frombuffer(frame_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def process_bgr_image(image_bgr: np.ndarray) -> list[HandLandmarks]:
    """
    Run MediaPipe Hands on a BGR uint8 image.
    Returns 0-2 HandLandmarks in normalized image coordinates.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []
    h, w = image_bgr.shape[:2]
    if h < 2 or w < 2:
        return []

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hands = _get_hands()
    result = hands.process(rgb)
    out: list[HandLandmarks] = []
    if not result.multi_hand_landmarks:
        return out

    handedness_list = result.multi_handedness or []
    for idx, hand_lms in enumerate(result.multi_hand_landmarks):
        label = "Unknown"
        if idx < len(handedness_list) and handedness_list[idx].classification:
            label = handedness_list[idx].classification[0].label
        pts: list[LandmarkPoint] = []
        for lm in hand_lms.landmark:
            pts.append(LandmarkPoint(x=float(lm.x), y=float(lm.y), z=float(lm.z)))
        out.append(HandLandmarks(points=pts, handedness=label))
    return out


def process_jpeg_bytes(frame_bytes: bytes) -> tuple[list[HandLandmarks], tuple[int, int] | None]:
    """Decode JPEG and return hands + (width, height)."""
    img = decode_jpeg_to_bgr(frame_bytes)
    if img is None:
        return [], None
    h, w = img.shape[:2]
    return process_bgr_image(img), (w, h)


def landmarks_to_serializable(hands: list[HandLandmarks]) -> list[dict[str, Any]]:
    """Flatten for JSON WebSocket payload."""
    serialized = []
    for hand in hands:
        serialized.append(
            {
                "handedness": hand.handedness,
                "landmarks": [{"x": p.x, "y": p.y, "z": p.z} for p in hand.points],
            }
        )
    return serialized


def close_mediapipe() -> None:
    global _mp_hands
    if _mp_hands is not None:
        _mp_hands.close()
        _mp_hands = None
