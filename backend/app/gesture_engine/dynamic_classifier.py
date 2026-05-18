"""LSTM dynamic gesture inference."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from app.gesture_engine.sequence_features import WINDOW_SIZE, window_to_tensor
from app.schemas.gesture import GestureResult

logger = logging.getLogger(__name__)

# Re-export for callers
__all__ = ["DynamicClassifier", "WINDOW_SIZE"]


def _default_model_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "ml" / "models"


class DynamicClassifier:
    def __init__(
        self,
        model_path: Path | None = None,
        label_map_path: Path | None = None,
        confidence_threshold: float = 0.75,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self._model = None
        self._labels: list[str] = ["none", "swipe_left", "swipe_right", "swipe_up", "wave"]
        base = _default_model_dir()
        self._model_path = Path(model_path) if model_path else base / "gesture_lstm.keras"
        self._label_map_path = Path(label_map_path) if label_map_path else base / "label_map.json"
        self._load_label_map()
        self._try_load_model()

    def _load_label_map(self) -> None:
        if self._label_map_path.is_file():
            try:
                data = json.loads(self._label_map_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._labels = data
                elif isinstance(data, dict) and "labels" in data:
                    self._labels = list(data["labels"])
                elif isinstance(data, dict):
                    # index -> name
                    keys = sorted(data.keys(), key=lambda k: int(k) if str(k).isdigit() else 0)
                    self._labels = [data[k] for k in keys]
            except (OSError, json.JSONDecodeError, ValueError) as e:
                logger.warning("Could not read label map %s: %s", self._label_map_path, e)

    def _try_load_model(self) -> None:
        if not self._model_path.is_file():
            logger.warning("LSTM model not found at %s — dynamic gestures disabled", self._model_path)
            return
        try:
            import tensorflow as tf  # noqa: PLC0415

            self._model = tf.keras.models.load_model(self._model_path)
            logger.info("Loaded dynamic gesture model from %s", self._model_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load LSTM model: %s", e)
            self._model = None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def predict(self, frames: list[np.ndarray]) -> GestureResult:
        """
        frames: last WINDOW_SIZE frames of (21,3) numpy arrays (single hand).
        """
        if len(frames) < WINDOW_SIZE:
            return GestureResult(gesture="UNKNOWN", confidence=0.0)
        window = frames[-WINDOW_SIZE:]
        if self._model is None:
            return GestureResult(gesture="UNKNOWN", confidence=0.0)
        x = window_to_tensor(window)
        probs = self._model.predict(x, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        if idx >= len(self._labels):
            return GestureResult(gesture="UNKNOWN", confidence=0.0)
        label = self._labels[idx]
        if label == "none" or conf < self.confidence_threshold:
            return GestureResult(gesture="UNKNOWN", confidence=conf)
        return GestureResult(gesture=label, confidence=conf)
