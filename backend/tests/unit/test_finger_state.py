import numpy as np

from app.gesture_engine import finger_state
from app.gesture_engine.finger_state import FingerStates
from tests.fixtures.landmark_builders import pose_open_palm


def test_hand_to_numpy_roundtrip():
    hand = finger_state.hand_landmarks_from_array(pose_open_palm())
    arr = finger_state.hand_to_numpy(hand)
    assert arr.shape == (21, 3)


def test_get_finger_states_open_palm():
    hand = finger_state.hand_landmarks_from_array(pose_open_palm())
    fs = finger_state.get_finger_states(hand)
    assert isinstance(fs, FingerStates)
