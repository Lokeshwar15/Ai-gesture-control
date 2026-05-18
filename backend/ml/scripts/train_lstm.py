"""Train LSTM on split .npz files; optional synthetic bootstrap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CLASS_NAMES = ["none", "swipe_left", "swipe_right", "swipe_up", "wave"]


def _synthetic_splits(out_dir: Path, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    def make_split(n_per_class: int) -> tuple[np.ndarray, np.ndarray]:
        X_list = []
        y_list = []
        for cls in range(len(CLASS_NAMES)):
            for _ in range(n_per_class):
                X_list.append(rng.normal(size=(30, 63)).astype(np.float32))
                y_list.append(cls)
        X = np.stack(X_list, axis=0)
        y = np.array(y_list, dtype=np.int32)
        p = rng.permutation(len(y))
        return X[p], y[p]

    for name, n in [("train", 80), ("val", 20), ("test", 20)]:
        X, y = make_split(n)
        np.savez_compressed(out_dir / f"{name}.npz", X=X, y=y)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, default=ROOT / "ml" / "data" / "splits")
    parser.add_argument("--out-model", type=Path, default=ROOT / "ml" / "models" / "gesture_lstm.keras")
    parser.add_argument("--out-labels", type=Path, default=ROOT / "ml" / "models" / "label_map.json")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--synthetic", action="store_true", help="Create random splits then train (dev only)")
    args = parser.parse_args()

    if args.synthetic:
        _synthetic_splits(args.splits)

    import tensorflow as tf  # noqa: PLC0415
    from tensorflow.keras import Sequential  # noqa: PLC0415
    from tensorflow.keras.callbacks import EarlyStopping  # noqa: PLC0415
    from tensorflow.keras.layers import Dense, Dropout, LSTM  # noqa: PLC0415

    train = np.load(args.splits / "train.npz")
    val = np.load(args.splits / "val.npz")
    X_train, y_train = train["X"], train["y"]
    X_val, y_val = val["X"], val["y"]

    model = Sequential(
        [
            LSTM(64, return_sequences=True, input_shape=(30, 63)),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dense(len(CLASS_NAMES), activation="softmax"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=32,
        callbacks=[es],
        verbose=1,
    )

    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out_model)
    args.out_labels.write_text(json.dumps({"labels": CLASS_NAMES}, indent=2), encoding="utf-8")
    print("Saved", args.out_model)


if __name__ == "__main__":
    main()
