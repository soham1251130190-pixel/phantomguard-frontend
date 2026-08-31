from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from common import build_preprocessor, evaluate_probabilities, load_dataset, split_features


DEFAULT_DATA_DIR = "data/phantomguard_tcn_dataset/phantomguard_generate/data"


def train_xgboost(x_train: np.ndarray, y_train, seed: int = 42) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=220,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PhantomGuard XGBoost model.")
    parser.add_argument("--train-csv", default=f"{DEFAULT_DATA_DIR}/phantomguard_synthetic_dataset.csv")
    parser.add_argument("--out-dir", default="artifacts/xgboost")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.train_csv)
    x_raw, y = split_features(df)
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x_raw, y, test_size=0.2, random_state=args.random_state, stratify=y
    )

    preprocessor = build_preprocessor(x_train_raw)
    x_train = preprocessor.fit_transform(x_train_raw).astype(np.float32)
    x_test = preprocessor.transform(x_test_raw).astype(np.float32)
    model = train_xgboost(x_train, y_train, args.random_state)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = evaluate_probabilities(y_test, probabilities)

    joblib.dump({"preprocessor": preprocessor, "model": model, "feature_columns": list(x_raw.columns)}, out_dir / "xgboost_model.joblib")
    (out_dir / "metrics.txt").write_text(str(metrics), encoding="utf-8")
    print(f"XGBoost F1: {metrics['f1']:.4f}")
    print(f"Saved: {out_dir / 'xgboost_model.joblib'}")


if __name__ == "__main__":
    main()
