"""
Finger extended/curl detection from 21 MediaPipe hand landmarks.

Landmark indices (MediaPipe Hands):
0 WRIST, 1-4 THUMB, 5-8 INDEX, 9-12 MIDDLE, 13-16 RING, 17-20 PINKY
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.schemas.gesture import HandLandmarks, LandmarkPoint


@dataclass
class FingerStates:
    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool


def hand_to_numpy(hand: HandLandmarks) -> np.ndarray:
    """Return (21, 3) landmark array."""
    return _to_np(hand)


def _to_np(hand: HandLandmarks) -> np.ndarray:
    """Shape (21, 3) in x,y,z."""
    arr = np.zeros((21, 3), dtype=np.float64)
    for i, p in enumerate(hand.points):
        arr[i, 0] = p.x
        arr[i, 1] = p.y
        arr[i, 2] = p.z
    return arr


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at b between ba and bc."""
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba)
    nbc = np.linalg.norm(bc)
    if nba < 1e-9 or nbc < 1e-9:
        return 180.0
    cosang = float(np.clip(np.dot(ba, bc) / (nba * nbc), -1.0, 1.0))
    return math.degrees(math.acos(cosang))


def is_finger_extended(
    lm: np.ndarray,
    mcp_idx: int,
    pip_idx: int,
    dip_idx: int,
    tip_idx: int,
    *,
    straight_deg_max: float = 160.0,
    tip_beyond_ratio: float = 0.95,
) -> bool:
    """
    Finger extended if PIP angle is near straight and tip lies past pip
    along the ray from MCP toward tip (2D x,y).
    """
    wrist = lm[0, :2]
    mcp = lm[mcp_idx, :2]
    pip = lm[pip_idx, :2]
    dip = lm[dip_idx, :2]
    tip = lm[tip_idx, :2]

    ang_mcp = _angle_deg(wrist, mcp, pip)
    ang_pip = _angle_deg(mcp, pip, dip)
    ang_dip = _angle_deg(pip, dip, tip)
    # Extended finger: joint angles near straight (MediaPipe collinear chains ~180°)
    if ang_pip < 150.0 or ang_dip < 150.0:
        return False

    # tip should be farther from wrist than pip along finger direction
    v = pip - mcp
    vn = np.linalg.norm(v)
    if vn < 1e-9:
        return False
    u = v / vn
    proj_tip = float(np.dot(tip - mcp, u))
    proj_pip = float(np.dot(pip - mcp, u))
    return proj_tip >= tip_beyond_ratio * max(proj_pip, 1e-6)


def is_thumb_extended(lm: np.ndarray, *, tip_ip_dist_min: float = 0.06) -> bool:
    """
    Thumb extended: distance between thumb tip and index MCP is large
    relative to hand scale (wrist to middle MCP).
    """
    wrist = lm[0]
    middle_mcp = lm[9]
    thumb_tip = lm[4]
    index_mcp = lm[5]
    scale = float(np.linalg.norm(middle_mcp[:2] - wrist[:2]))
    if scale < 1e-6:
        scale = 1.0
    dist = float(np.linalg.norm(thumb_tip[:2] - index_mcp[:2]))
    return (dist / scale) >= tip_ip_dist_min


def is_thumb_pointing_down(lm: np.ndarray) -> bool:
    """Thumbs down: thumb tip below index MCP in image y (y grows downward)."""
    thumb_tip = lm[4, 1]
    index_mcp = lm[5, 1]
    wrist_y = lm[0, 1]
    # tip clearly below index mcp and below wrist
    return thumb_tip > index_mcp + 0.02 and thumb_tip > wrist_y + 0.01


def get_finger_states(hand: HandLandmarks) -> FingerStates:
    lm = _to_np(hand)
    thumb_ext = is_thumb_extended(lm)
    index_ext = is_finger_extended(lm, 5, 6, 7, 8)
    middle_ext = is_finger_extended(lm, 9, 10, 11, 12)
    ring_ext = is_finger_extended(lm, 13, 14, 15, 16)
    pinky_ext = is_finger_extended(lm, 17, 18, 19, 20)
    return FingerStates(
        thumb=thumb_ext,
        index=index_ext,
        middle=middle_ext,
        ring=ring_ext,
        pinky=pinky_ext,
    )


def pinch_distance_px(lm: np.ndarray, image_width: int, image_height: int) -> float:
    """Euclidean distance thumb tip to index tip in pixels."""
    t = lm[4, :2]
    i = lm[8, :2]
    dx = (t[0] - i[0]) * image_width
    dy = (t[1] - i[1]) * image_height
    return float(math.hypot(dx, dy))


def hand_landmarks_from_array(arr: np.ndarray, handedness: str = "Right") -> HandLandmarks:
    """Build HandLandmarks from (21,3) float array."""
    pts = [LandmarkPoint(x=float(arr[i, 0]), y=float(arr[i, 1]), z=float(arr[i, 2])) for i in range(21)]
    return HandLandmarks(points=pts, handedness=handedness)
