"""Write ml/data/processed/static_val.npz from synthetic landmark poses."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.fixtures.landmark_builders import (  # noqa: E402
    pose_closed_fist,
    pose_open_palm,
    pose_peace_sign,
    pose_pinch_close,
    pose_thumbs_down,
    pose_thumbs_up,
)


def main() -> None:
    out = ROOT / "ml" / "data" / "processed" / "static_val.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs = [
        ("open_palm", pose_open_palm()),
        ("closed_fist", pose_closed_fist()),
        ("peace_sign", pose_peace_sign()),
        ("pinch", pose_pinch_close()),
        ("thumbs_up", pose_thumbs_up()),
        ("thumbs_down", pose_thumbs_down()),
    ]
    rng = np.random.default_rng(42)
    arrs = []
    labels = []
    for lab, arr in pairs:
        for _ in range(12):
            noise = rng.normal(0, 0.002, arr.shape)
            arrs.append(arr + noise)
            labels.append(lab)
    X = np.stack(arrs, axis=0)
    y = np.array(labels, dtype=object)
    np.savez_compressed(out, landmarks=X, labels=y)
    print("Wrote", out, X.shape)


if __name__ == "__main__":
    main()
