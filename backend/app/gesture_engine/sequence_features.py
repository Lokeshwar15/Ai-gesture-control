"""Shared landmark sequence normalization for LSTM (must match training)."""

from __future__ import annotations

import numpy as np

WINDOW_SIZE = 30


def normalize_frame(lm: np.ndarray) -> np.ndarray:
    """
    lm: (21, 3) single frame MediaPipe landmarks (normalized image coords).
    Translate to wrist; scale xy (and z) by palm size (wrist to middle MCP).
    Returns flattened (63,) vector.
    """
    out = lm.astype(np.float64).copy()
    wrist = out[0].copy()
    out -= wrist
    palm = float(np.linalg.norm(out[9, :2]))
    scale = max(palm, 1e-6)
    out[:, 0] /= scale
    out[:, 1] /= scale
    out[:, 2] /= scale
    return out.reshape(-1)


def window_to_tensor(frames: list[np.ndarray]) -> np.ndarray:
    """frames: list of (21,3), length must be WINDOW_SIZE. Returns (1, T, 63)."""
    if len(frames) != WINDOW_SIZE:
        raise ValueError(f"Expected {WINDOW_SIZE} frames, got {len(frames)}")
    feats = np.stack([normalize_frame(f) for f in frames], axis=0)
    return feats.reshape(1, WINDOW_SIZE, -1)
