import numpy as np

from app.gesture_engine.classifier import GestureClassifierSession
from app.gesture_engine.dynamic_classifier import DynamicClassifier
from app.gesture_engine import finger_state
from tests.fixtures.landmark_builders import pose_open_palm


def test_session_unknown_when_no_hands():
    session = GestureClassifierSession(dynamic=DynamicClassifier())
    r = session.classify([], (640, 480), None)
    assert r.gesture == "UNKNOWN"


def test_session_populates_history():
    dyn = DynamicClassifier()
    dyn._model = None
    session = GestureClassifierSession(dynamic=dyn)
    hand = finger_state.hand_landmarks_from_array(pose_open_palm())
    for _ in range(5):
        session.classify([hand], (640, 480), [])
    assert len(session._history) == 5
