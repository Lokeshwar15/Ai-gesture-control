"""Evaluate LSTM on val/test splits; optional static rule eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "ml" / "data" / "splits")
    parser.add_argument("--model", type=Path, default=ROOT / "ml" / "models" / "gesture_lstm.keras")
    parser.add_argument("--label-map", type=Path, default=ROOT / "ml" / "models" / "label_map.json")
    parser.add_argument("--out-metrics", type=Path, default=ROOT / "ml" / "models" / "metrics.json")
    parser.add_argument("--min-accuracy", type=float, default=0.88)
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--static-val", type=Path, default=ROOT / "ml" / "data" / "processed" / "static_val.npz")
    parser.add_argument("--min-static-acc", type=float, default=0.95)
    args = parser.parse_args()

    metrics: dict = {}

    if args.static_val.is_file():
        from app.gesture_engine.static_classifier import raw_static_gesture  # noqa: E402
        from app.gesture_engine import finger_state  # noqa: E402

        data = np.load(args.static_val, allow_pickle=True)
        Xs, ys = data["landmarks"], data["labels"]
        correct = 0
        total = 0
        for i in range(len(ys)):
            hand = finger_state.hand_landmarks_from_array(Xs[i])
            r = raw_static_gesture(hand, (640, 480))
            pred = r.gesture
            exp = str(ys[i])
            if pred == exp:
                correct += 1
            total += 1
        s_acc = correct / max(total, 1)
        metrics["static_accuracy"] = s_acc
        if not args.no_gate and s_acc < args.min_static_acc:
            raise SystemExit(f"Static accuracy {s_acc:.3f} < {args.min_static_acc}")

    if not args.model.is_file():
        print("Model missing; skipping LSTM eval:", args.model)
        args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
        args.out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return

    import tensorflow as tf  # noqa: PLC0415

    labels = json.loads(args.label_map.read_text(encoding="utf-8"))
    if isinstance(labels, dict) and "labels" in labels:
        label_names = labels["labels"]
    else:
        label_names = labels

    split_path = args.splits_dir / f"{args.split}.npz"
    if not split_path.is_file():
        print("Split missing:", split_path)
        args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
        args.out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return

    d = np.load(split_path)
    X, y = d["X"], d["y"]
    model = tf.keras.models.load_model(args.model)
    probs = model.predict(X, verbose=0)
    pred = np.argmax(probs, axis=1)
    acc = float(np.mean(pred == y))
    metrics[f"lstm_{args.split}_accuracy"] = acc
    metrics["classification_report"] = classification_report(
        y, pred, target_names=list(label_names), output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y, pred).tolist()
    metrics["confusion_matrix"] = cm

    none_idx = list(label_names).index("none") if "none" in label_names else None
    if none_idx is not None:
        none_mask = y == none_idx
        fp = float(np.mean(pred[~none_mask] == none_idx)) if (~none_mask).any() else 0.0
        metrics["none_false_positive_rate"] = fp

    args.out_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, ax = plt.subplots()
        ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title("Confusion matrix")
        fig.savefig(args.out_metrics.with_name("confusion_matrix.png"), dpi=150)
        plt.close(fig)
    except Exception:  # noqa: BLE001
        pass

    if not args.no_gate and acc < args.min_accuracy:
        raise SystemExit(f"LSTM accuracy {acc:.3f} < {args.min_accuracy}")


if __name__ == "__main__":
    main()
