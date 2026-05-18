"""
Record labeled hand-landmark data from your webcam.

Modes
-----
dynamic (default): record 2–3s motion clips for LSTM training (swipes, wave, none)
static:            save one labeled frame per keypress for static-gesture validation/tuning

Output: ml/data/raw/{label}_{timestamp}.npz with keys landmarks (T,21,3), label

Usage
-----
  cd backend
  python ml/scripts/collect_landmarks.py                  # dynamic clips
  python ml/scripts/collect_landmarks.py --mode static  # single-frame labels
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mediapipe as mp  # noqa: E402

DYNAMIC_LABELS = ["none", "swipe_left", "swipe_right", "swipe_up", "wave"]
STATIC_LABELS = [
    "thumbs_up",
    "thumbs_down",
    "open_palm",
    "closed_fist",
    "peace_sign",
    "pinch",
]


def _landmarks_from_result(res, frame_shape) -> np.ndarray | None:
    if not res.multi_hand_landmarks:
        return None
    arr = np.zeros((21, 3), dtype=np.float32)
    for i, lm in enumerate(res.multi_hand_landmarks[0].landmark):
        arr[i, 0] = lm.x
        arr[i, 1] = lm.y
        arr[i, 2] = lm.z
    return arr


def _draw_hand(vis, res, frame_shape) -> None:
    if not res.multi_hand_landmarks:
        return
    h, w, _ = frame_shape
    hand_lms = res.multi_hand_landmarks[0]
    for connection in mp.solutions.hands.HAND_CONNECTIONS:
        a = hand_lms.landmark[connection[0]]
        b = hand_lms.landmark[connection[1]]
        cv2.line(
            vis,
            (int(a.x * w), int(a.y * h)),
            (int(b.x * w), int(b.y * h)),
            (255, 0, 0),
            2,
        )


def _save_clip(out_dir: Path, label: str, buffer: list[np.ndarray]) -> None:
    if len(buffer) < 5:
        print("Too few frames — hold the gesture longer or move slower.")
        return
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out_path = out_dir / f"{label}_{ts}.npz"
    stacked = np.stack(buffer, axis=0)
    np.savez_compressed(out_path, landmarks=stacked, label=np.array(label))
    print(f"Saved clip {out_path.name} shape={stacked.shape}")


def _save_static_frame(out_dir: Path, label: str, frame_lm: np.ndarray) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out_path = out_dir / f"{label}_{ts}.npz"
    # (1, 21, 3) so build_dataset can ingest if needed
    stacked = frame_lm.reshape(1, 21, 3)
    np.savez_compressed(out_path, landmarks=stacked, label=np.array(label))
    print(f"Saved static frame {out_path.name}")


def _print_help(mode: str, labels: list[str]) -> None:
    print("\n=== Webcam label collector ===")
    print(f"Mode: {mode}")
    for i, lab in enumerate(labels):
        print(f"  [{i}] {lab}")
    if mode == "dynamic":
        print("  SPACE — start/stop recording a clip for the selected label")
    else:
        print("  Press 0-5 — save ONE labeled frame immediately (hand must be visible)")
    print("  q — quit\n")


def run_dynamic(args: argparse.Namespace, hands, cap) -> None:
    labels = DYNAMIC_LABELS
    _print_help("dynamic", labels)
    label_idx = 0
    recording = False
    buffer: list[np.ndarray] = []
    counts: dict[str, int] = {l: 0 for l in labels}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        vis = frame.copy()
        label_name = labels[label_idx]
        lm_arr = _landmarks_from_result(res, frame.shape)
        _draw_hand(vis, res, frame.shape)

        status = "REC" if recording else "idle"
        cv2.putText(
            vis,
            f"[{label_idx}] {label_name} | {status} | frames={len(buffer)} | saved={counts[label_name]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0) if recording else (200, 200, 200),
            2,
        )
        cv2.putText(
            vis,
            "0-4: label | SPACE: record | q: quit",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
        )

        if recording and lm_arr is not None:
            buffer.append(lm_arr.copy())
        elif recording and lm_arr is None:
            cv2.putText(vis, "No hand!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("collect_landmarks", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            if recording and buffer:
                _save_clip(args.out, label_name, buffer)
                counts[label_name] += 1
            recording = not recording
            buffer = []
        for i in range(len(labels)):
            if key == ord(str(i)):
                label_idx = i
                print("Label ->", labels[label_idx])


def run_static(args: argparse.Namespace, hands, cap) -> None:
    labels = STATIC_LABELS
    _print_help("static", labels)
    counts: dict[str, int] = {l: 0 for l in labels}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        vis = frame.copy()
        lm_arr = _landmarks_from_result(res, frame.shape)
        _draw_hand(vis, res, frame.shape)

        cv2.putText(
            vis,
            "Press 0-5 to save labeled STILL pose | q: quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        y = 60
        for i, lab in enumerate(labels):
            cv2.putText(vis, f"{i}: {lab} ({counts[lab]})", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y += 22

        if lm_arr is None:
            cv2.putText(vis, "Show your hand to label", (10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("collect_landmarks", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        for i in range(len(labels)):
            if key == ord(str(i)) and lm_arr is not None:
                lab = labels[i]
                _save_static_frame(args.out, lab, lm_arr)
                counts[lab] += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect labeled landmark data from webcam")
    parser.add_argument("--mode", choices=["dynamic", "static"], default="dynamic")
    parser.add_argument("--out", type=Path, default=ROOT / "ml" / "data" / "raw")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    try:
        if args.mode == "static":
            run_static(args, hands, cap)
        else:
            run_dynamic(args, hands, cap)
    finally:
        cap.release()
        hands.close()
        cv2.destroyAllWindows()

    print(f"\nFiles written to: {args.out.resolve()}")
    print("Next (dynamic/LSTM):")
    print("  python ml/scripts/build_dataset.py")
    print("  python ml/scripts/train_lstm.py")


if __name__ == "__main__":
    main()
