"""Synthetic (21,3) landmark arrays for tests and static validation NPZ generation."""

from __future__ import annotations

import numpy as np


def neutral_wrist_center() -> np.ndarray:
    lm = np.zeros((21, 3), dtype=np.float64)
    lm[0] = (0.5, 0.5, 0.0)
    return lm


def _set_extended_finger(lm: np.ndarray, mcp: int, x: float, y0: float) -> None:
    """Collinear extended finger chain (straight PIP/DIP angles)."""
    lm[mcp] = (x, y0, 0.0)
    lm[mcp + 1] = (x, y0 + 0.06, 0.0)
    lm[mcp + 2] = (x, y0 + 0.12, 0.0)
    lm[mcp + 3] = (x, y0 + 0.18, 0.0)


def _set_curled_finger(lm: np.ndarray, mcp: int, x: float, y0: float) -> None:
    """Curled finger: tips stay near MCP (not extended)."""
    lm[mcp] = (x, y0, 0.0)
    lm[mcp + 1] = (x + 0.01, y0 + 0.02, 0.0)
    lm[mcp + 2] = (x + 0.02, y0 + 0.03, 0.0)
    lm[mcp + 3] = (x + 0.02, y0 + 0.04, 0.0)


def pose_open_palm() -> np.ndarray:
    lm = neutral_wrist_center()
    _set_extended_finger(lm, 5, 0.46, 0.55)
    _set_extended_finger(lm, 9, 0.50, 0.55)
    _set_extended_finger(lm, 13, 0.54, 0.55)
    _set_extended_finger(lm, 17, 0.58, 0.55)
    _set_extended_finger(lm, 1, 0.42, 0.55)  # thumb chain uses indices 1-4
    lm[1] = (0.42, 0.55, 0.0)
    lm[2] = (0.40, 0.50, 0.0)
    lm[3] = (0.38, 0.46, 0.0)
    lm[4] = (0.36, 0.42, 0.0)
    return lm


def pose_closed_fist() -> np.ndarray:
    lm = neutral_wrist_center()
    for mcp, x in ((5, 0.46), (9, 0.50), (13, 0.54), (17, 0.58)):
        _set_curled_finger(lm, mcp, x, 0.55)
    lm[1:5] = [
        (0.48, 0.55, 0.0),
        (0.49, 0.56, 0.0),
        (0.50, 0.57, 0.0),
        (0.51, 0.58, 0.0),
    ]
    return lm


def pose_peace_sign() -> np.ndarray:
    lm = pose_closed_fist()
    _set_extended_finger(lm, 5, 0.46, 0.55)
    _set_extended_finger(lm, 9, 0.50, 0.55)
    # thumb tucked (not extended)
    lm[1:5] = [
        (0.52, 0.58, 0.0),
        (0.53, 0.59, 0.0),
        (0.54, 0.60, 0.0),
        (0.55, 0.61, 0.0),
    ]
    return lm


def pose_pinch_far() -> np.ndarray:
    lm = pose_open_palm()
    lm[4] = (0.8, 0.5, 0.0)
    lm[8] = (0.2, 0.5, 0.0)
    return lm


def pose_pinch_close() -> np.ndarray:
    lm = pose_open_palm()
    lm[4] = (0.51, 0.73, 0.0)
    lm[8] = (0.49, 0.73, 0.0)
    return lm


def pose_thumbs_up() -> np.ndarray:
    lm = pose_closed_fist()
    lm[1:5] = [
        (0.48, 0.55, 0.0),
        (0.46, 0.50, 0.0),
        (0.44, 0.44, 0.0),
        (0.42, 0.38, 0.0),
    ]
    return lm


def pose_thumbs_down() -> np.ndarray:
    lm = pose_closed_fist()
    lm[1:5] = [
        (0.48, 0.58, 0.0),
        (0.46, 0.64, 0.0),
        (0.44, 0.72, 0.0),
        (0.42, 0.82, 0.0),
    ]
    return lm
