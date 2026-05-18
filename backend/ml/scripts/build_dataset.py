"""Build windowed dataset from raw .npz landmark clips."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.gesture_engine.sequence_features import WINDOW_SIZE, normalize_frame  # noqa: E402

CLASS_NAMES = ["none", "swipe_left", "swipe_right", "swipe_up", "wave"]
NAME_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}


def load_raw_dir(raw_dir: Path) -> tuple[list[np.ndarray], list[int]]:
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    for path in sorted(raw_dir.glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        if "landmarks" not in data or "label" not in data:
            continue
        seq = data["landmarks"]
        label = str(data["label"].item()) if hasattr(data["label"], "item") else str(data["label"])
        if label not in NAME_TO_ID:
            continue
        yid = NAME_TO_ID[label]
        T = seq.shape[0]
        if T < WINDOW_SIZE:
            continue
        stride = 5
        for start in range(0, T - WINDOW_SIZE + 1, stride):
            window = seq[start : start + WINDOW_SIZE]
            feats = np.stack([normalize_frame(window[t]) for t in range(WINDOW_SIZE)], axis=0)
            X_list.append(feats)
            y_list.append(yid)
    if not X_list:
        raise SystemExit("No samples found. Add .npz files under ml/data/raw with keys landmarks,label")
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=ROOT / "ml" / "data" / "raw")
    parser.add_argument("--out", type=Path, default=ROOT / "ml" / "data" / "splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    X, y = load_raw_dir(args.raw)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=args.seed, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp
    )

    np.savez_compressed(args.out / "train.npz", X=X_train, y=y_train)
    np.savez_compressed(args.out / "val.npz", X=X_val, y=y_val)
    np.savez_compressed(args.out / "test.npz", X=X_test, y=y_test)
    print("Saved splits:", X_train.shape, X_val.shape, X_test.shape)


if __name__ == "__main__":
    main()
