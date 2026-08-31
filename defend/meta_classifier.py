from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from common import evaluate_probabilities


def stack_probabilities(xgb_probs: np.ndarray, tcn_probs: np.ndarray, gnn_probs: np.ndarray) -> np.ndarray:
    return np.column_stack([xgb_probs, tcn_probs, gnn_probs])


def train_meta_classifier(base_probabilities: np.ndarray, y_val) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(base_probabilities, y_val)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PhantomGuard meta-classifier from saved base-model probability arrays."
    )
    parser.add_argument("--xgb-probs", required=True)
    parser.add_argument("--tcn-probs", required=True)
    parser.add_argument("--gnn-probs", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out-dir", default="artifacts/meta_classifier")
    args = parser.parse_args()

    xgb_probs = np.load(args.xgb_probs)
    tcn_probs = np.load(args.tcn_probs)
    gnn_probs = np.load(args.gnn_probs)
    labels = np.load(args.labels)
    base_probabilities = stack_probabilities(xgb_probs, tcn_probs, gnn_probs)
    model = train_meta_classifier(base_probabilities, labels)
    metrics = evaluate_probabilities(labels, model.predict_proba(base_probabilities)[:, 1])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "meta_classifier.joblib")
    (out_dir / "metrics.txt").write_text(str(metrics), encoding="utf-8")
    print(f"Meta-classifier F1: {metrics['f1']:.4f}")
    print(f"Saved: {out_dir / 'meta_classifier.joblib'}")


if __name__ == "__main__":
    main()
