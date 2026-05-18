from unittest.mock import MagicMock

import numpy as np

from app.gesture_engine.dynamic_classifier import DynamicClassifier


def test_dynamic_predict_with_mock_model():
    dyn = DynamicClassifier()
    mock = MagicMock()
    mock.predict.return_value = np.array([[0.1, 0.1, 0.1, 0.1, 0.6]])
    dyn._model = mock
    dyn._labels = ["none", "swipe_left", "swipe_right", "swipe_up", "wave"]
    dyn.confidence_threshold = 0.5
    frames = [np.zeros((21, 3), dtype=np.float32) for _ in range(30)]
    r = dyn.predict(frames)
    assert r.gesture == "wave"


def test_dynamic_short_history_returns_unknown():
    dyn = DynamicClassifier()
    dyn._model = MagicMock()
    frames = [np.zeros((21, 3), dtype=np.float32) for _ in range(5)]
    r = dyn.predict(frames)
    assert r.gesture == "UNKNOWN"
