"""Unified static + dynamic gesture classification."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from app.gesture_engine.dynamic_classifier import WINDOW_SIZE, DynamicClassifier
from app.gesture_engine import finger_state
from app.gesture_engine.static_classifier import StaticClassifier
from app.schemas.gesture import GestureResult, HandLandmarks


class GestureClassifierSession:
    """Per-connection session: hold-timer static + LSTM on landmark history."""

    def __init__(
        self,
        static: StaticClassifier | None = None,
        dynamic: DynamicClassifier | None = None,
        static_threshold: float = 0.75,
    ) -> None:
        self.static = static or StaticClassifier(confidence_threshold=static_threshold)
        self.dynamic = dynamic or DynamicClassifier(confidence_threshold=static_threshold)
        self._history: deque[np.ndarray] = deque(maxlen=WINDOW_SIZE * 2)

    def reset(self) -> None:
        self.static.reset()
        self._history.clear()

    def _primary_hand(self, hands: list[HandLandmarks]) -> HandLandmarks | None:
        if not hands:
            return None
        return hands[0]

    def classify(
        self,
        hands: list[HandLandmarks],
        image_wh: tuple[int, int] | None,
        serializable_landmarks: list[dict[str, Any]] | None,
    ) -> GestureResult:
        hand = self._primary_hand(hands)
        if hand is None:
            self.static.reset()
            self._history.clear()
            return GestureResult(gesture="UNKNOWN", confidence=0.0, landmarks=serializable_landmarks)

        lm = finger_state.hand_to_numpy(hand)
        self._history.append(lm.copy())

        static_res = self.static.detect_with_landmarks(hand, image_wh, serializable_landmarks)
        if static_res.gesture != "UNKNOWN":
            return static_res

        if self.static.in_progress_static is not None:
            return GestureResult(
                gesture="UNKNOWN",
                confidence=static_res.confidence,
                landmarks=serializable_landmarks,
            )

        dyn_res = self.dynamic.predict(list(self._history))
        if dyn_res.gesture != "UNKNOWN":
            return dyn_res.model_copy(update={"landmarks": serializable_landmarks})
        return GestureResult(
            gesture="UNKNOWN",
            confidence=dyn_res.confidence,
            landmarks=serializable_landmarks,
        )
