"""Merge single-frame static captures from ml/data/raw into static_val.npz."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STATIC_LABELS = {
    "thumbs_up",
    "thumbs_down",
    "open_palm",
    "closed_fist",
    "peace_sign",
    "pinch",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "ml" / "data" / "raw")
    parser.add_argument("--out", type=Path, default=ROOT / "ml" / "data" / "processed" / "static_val.npz")
    args = parser.parse_args()

    arrs: list[np.ndarray] = []
    labels: list[str] = []

    for path in sorted(args.raw.glob("*.npz")):
        label_prefix = path.stem.split("_")[0]
        if label_prefix not in STATIC_LABELS:
            continue
        data = np.load(path, allow_pickle=True)
        seq = data["landmarks"]
        # use middle frame if clip, else single frame
        frame = seq[len(seq) // 2] if seq.ndim == 3 else seq
        arrs.append(frame.astype(np.float64))
        labels.append(label_prefix)

    if not arrs:
        raise SystemExit(f"No static-labeled files in {args.raw}. Run: collect_landmarks.py --mode static")

    X = np.stack(arrs, axis=0)
    y = np.array(labels, dtype=object)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, landmarks=X, labels=y)
    print(f"Wrote {args.out} — {len(y)} samples")


if __name__ == "__main__":
    main()
