"""Rule-based static gesture classification with hold-timer debounce."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from app.gesture_engine import finger_state
from app.schemas.gesture import GestureResult, HandLandmarks

# Gesture label -> hold duration (seconds)
HOLD_SECONDS: dict[str, float] = {
    "thumbs_up": 0.30,
    "thumbs_down": 0.30,
    "open_palm": 0.20,
    "closed_fist": 0.30,
    "peace_sign": 0.20,
    "pinch": 0.20,
}

DEFAULT_CONFIDENCE = 0.85
PINCH_PX_THRESHOLD = 30.0


def _match_raw_candidate(
    hand: HandLandmarks,
    image_wh: tuple[int, int] | None,
) -> tuple[str, float] | tuple[None, float]:
    fs = finger_state.get_finger_states(hand)
    lm = finger_state.hand_to_numpy(hand)
    w, h = (image_wh if image_wh else (640, 480))

    # Pinch first (can overlap with partial fist)
    pinch_px = finger_state.pinch_distance_px(lm, w, h)
    if pinch_px < PINCH_PX_THRESHOLD:
        conf = max(0.0, 1.0 - pinch_px / PINCH_PX_THRESHOLD) * 0.9 + 0.1
        return "pinch", min(0.99, conf)

    all_extended = fs.thumb and fs.index and fs.middle and fs.ring and fs.pinky
    none_extended = not (fs.thumb or fs.index or fs.middle or fs.ring or fs.pinky)

    # Peace: index + middle extended, ring + pinky curled
    if fs.index and fs.middle and not fs.ring and not fs.pinky:
        return "peace_sign", DEFAULT_CONFIDENCE

    # Thumbs up / down before fist (thumb out, other fingers curled)
    thumb_down = finger_state.is_thumb_pointing_down(lm)
    thumb_ext = finger_state.is_thumb_extended(lm)
    index_middle_curled = not fs.index and not fs.middle
    if thumb_ext and index_middle_curled and not fs.ring and not fs.pinky:
        if thumb_down:
            return "thumbs_down", DEFAULT_CONFIDENCE
        return "thumbs_up", DEFAULT_CONFIDENCE

    # Open palm
    if all_extended:
        return "open_palm", DEFAULT_CONFIDENCE

    # Closed fist
    if none_extended or (not fs.index and not fs.middle and not fs.ring and not fs.pinky):
        return "closed_fist", DEFAULT_CONFIDENCE

    return None, 0.0


def raw_static_gesture(hand: HandLandmarks, image_wh: tuple[int, int] | None) -> GestureResult:
    """Single-frame static label (no hold timer) — for offline validation datasets."""
    label, conf = _match_raw_candidate(hand, image_wh)
    if label is None:
        return GestureResult(gesture="UNKNOWN", confidence=float(conf))
    return GestureResult(gesture=label, confidence=float(conf))


class StaticClassifier:
    """Emits a gesture only after it holds continuously for configured duration."""

    def __init__(
        self,
        confidence_threshold: float = 0.75,
        get_time: Callable[[], float] | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self._get_time = get_time or time.monotonic
        self._hold_label: str | None = None
        self._hold_start: float | None = None

    @property
    def in_progress_static(self) -> str | None:
        """Gesture label being held but not yet confirmed (suppress dynamic)."""
        if self._hold_label is None or self._hold_start is None:
            return None
        now = self._get_time()
        hold = HOLD_SECONDS.get(self._hold_label, 0.2)
        if now - self._hold_start < hold:
            return self._hold_label
        return None

    def reset(self) -> None:
        self._hold_label = None
        self._hold_start = None

    def detect(
        self,
        hand: HandLandmarks | None,
        image_wh: tuple[int, int] | None,
    ) -> GestureResult:
        if hand is None:
            self.reset()
            return GestureResult(gesture="UNKNOWN", confidence=0.0)

        label, conf = _match_raw_candidate(hand, image_wh)
        if label is None or conf < self.confidence_threshold:
            self.reset()
            return GestureResult(gesture="UNKNOWN", confidence=float(conf))

        now = self._get_time()
        hold = HOLD_SECONDS.get(label, 0.2)

        if label != self._hold_label:
            self._hold_label = label
            self._hold_start = now
            return GestureResult(gesture="UNKNOWN", confidence=float(conf))

        assert self._hold_start is not None
        if now - self._hold_start >= hold:
            return GestureResult(gesture=label, confidence=float(conf))
        return GestureResult(gesture="UNKNOWN", confidence=float(conf))

    def detect_with_landmarks(
        self,
        hand: HandLandmarks | None,
        image_wh: tuple[int, int] | None,
        serializable_landmarks: list[dict[str, Any]] | None,
    ) -> GestureResult:
        r = self.detect(hand, image_wh)
        return r.model_copy(update={"landmarks": serializable_landmarks})
