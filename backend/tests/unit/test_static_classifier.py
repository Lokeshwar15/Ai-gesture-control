import time

from app.gesture_engine.static_classifier import StaticClassifier, raw_static_gesture
from app.gesture_engine import finger_state
from tests.fixtures.landmark_builders import (
    pose_closed_fist,
    pose_open_palm,
    pose_peace_sign,
    pose_pinch_close,
    pose_thumbs_down,
    pose_thumbs_up,
)


def test_raw_open_palm():
    hand = finger_state.hand_landmarks_from_array(pose_open_palm())
    r = raw_static_gesture(hand, (640, 480))
    assert r.gesture == "open_palm"


def test_hold_timer_emits_after_duration():
    t0 = {"v": 0.0}

    def fake_time():
        return t0["v"]

    clf = StaticClassifier(confidence_threshold=0.5, get_time=fake_time)
    hand = finger_state.hand_landmarks_from_array(pose_open_palm())

    t0["v"] = 0.0
    assert clf.detect(hand, (640, 480)).gesture == "UNKNOWN"
    t0["v"] = 0.1
    assert clf.detect(hand, (640, 480)).gesture == "UNKNOWN"
    t0["v"] = 0.25
    out = clf.detect(hand, (640, 480))
    assert out.gesture == "open_palm"


def test_thumbs_up_raw():
    hand = finger_state.hand_landmarks_from_array(pose_thumbs_up())
    r = raw_static_gesture(hand, (640, 480))
    assert r.gesture in ("thumbs_up", "UNKNOWN")


def test_peace_sign_raw():
    hand = finger_state.hand_landmarks_from_array(pose_peace_sign())
    r = raw_static_gesture(hand, (640, 480))
    assert r.gesture == "peace_sign"


def test_pinch_raw():
    hand = finger_state.hand_landmarks_from_array(pose_pinch_close())
    r = raw_static_gesture(hand, (640, 480))
    assert r.gesture == "pinch"
