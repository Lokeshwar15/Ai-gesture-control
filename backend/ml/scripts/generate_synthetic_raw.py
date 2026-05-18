"""Generate synthetic raw landmark clips for LSTM pipeline (dev/CI bootstrap)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.fixtures.landmark_builders import (  # noqa: E402
    neutral_wrist_center,
    pose_open_palm,
)

LABELS = ["none", "swipe_left", "swipe_right", "swipe_up", "wave"]


def _motion_sequence(label: str, frames: int = 45) -> np.ndarray:
    base = pose_open_palm()
    seq = []
    for t in range(frames):
        lm = base.copy()
        phase = t / frames
        if label == "swipe_left":
            lm[:, 0] -= 0.02 * phase
        elif label == "swipe_right":
            lm[:, 0] += 0.02 * phase
        elif label == "swipe_up":
            lm[:, 1] -= 0.02 * phase
        elif label == "wave":
            lm[:, 0] += 0.01 * np.sin(phase * 8 * np.pi)
        else:
            lm = neutral_wrist_center()
        seq.append(lm.astype(np.float32))
    return np.stack(seq, axis=0)


def main() -> None:
    out = ROOT / "ml" / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    per_class = 30
    for label in LABELS:
        for i in range(per_class):
            seq = _motion_sequence(label)
            path = out / f"{label}_{ts}_{i:03d}.npz"
            np.savez_compressed(path, landmarks=seq, label=np.array(label))
    print(f"Wrote {per_class * len(LABELS)} clips to {out}")


if __name__ == "__main__":
    main()
