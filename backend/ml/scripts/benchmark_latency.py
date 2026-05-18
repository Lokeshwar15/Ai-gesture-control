"""Measure MediaPipe + classifier inference latency."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.gesture_engine.classifier import GestureClassifierSession  # noqa: E402
from app.gesture_engine.mediapipe_processor import process_bgr_image  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=ROOT / "tests" / "fixtures" / "blank_640x480.jpg")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--out", type=Path, default=ROOT / "ml" / "models" / "benchmark.json")
    args = parser.parse_args()

    import cv2  # noqa: PLC0415

    img = cv2.imread(str(args.fixture))
    if img is None:
        img = np.zeros((480, 640, 3), dtype=np.uint8)

    times: list[float] = []
    session = GestureClassifierSession()
    h, w = img.shape[:2]
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        hands = process_bgr_image(img)
        session.classify(hands, (w, h), [])
        times.append((time.perf_counter() - t0) * 1000.0)

    report = {
        "iterations": args.iterations,
        "p50_ms": statistics.median(times),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": float(np.mean(times)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
