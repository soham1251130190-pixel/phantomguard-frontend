from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from common import (
    build_preprocessor,
    build_sequences,
    evaluate_probabilities,
    load_dataset,
    predict_torch,
    set_seed,
    split_features,
    train_torch_model,
)


DEFAULT_DATA_DIR = "data/phantomguard_tcn_dataset/phantomguard_generate/data"


class TCNClassifier(nn.Module):
    def __init__(self, feature_count: int, hidden_channels: int = 48) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(feature_count, hidden_channels, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=2, padding=2, dilation=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.network(x).squeeze(-1)
        return self.classifier(features).squeeze(-1)


def train_tcn(x_train_seq: np.ndarray, y_train, feature_count: int, epochs: int, batch_size: int) -> TCNClassifier:
    model = TCNClassifier(feature_count)
    return train_torch_model(model, (x_train_seq,), y_train, epochs, batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PhantomGuard TCN model.")
    parser.add_argument("--train-csv", default=f"{DEFAULT_DATA_DIR}/phantomguard_tcn_sequences.csv")
    parser.add_argument("--out-dir", default="artifacts/tcn")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=4)
    args = parser.parse_args()
    set_seed(args.random_state)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.train_csv)
    x_raw, y = split_features(df)
    train_df, test_df, x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        df, x_raw, y, test_size=0.2, random_state=args.random_state, stratify=y
    )
    preprocessor = build_preprocessor(x_train_raw)
    x_train = preprocessor.fit_transform(x_train_raw).astype(np.float32)
    x_test = preprocessor.transform(x_test_raw).astype(np.float32)

    train_seq = build_sequences(train_df, x_train, args.sequence_length)
    test_seq = build_sequences(test_df, x_test, args.sequence_length)
    model = train_tcn(train_seq, y_train, x_train.shape[1], args.epochs, args.batch_size)
    probabilities = predict_torch(model, (test_seq,), args.batch_size)
    metrics = evaluate_probabilities(y_test, probabilities)

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "state_dict": model.state_dict(),
            "feature_count": x_train.shape[1],
            "feature_columns": list(x_raw.columns),
            "sequence_length": args.sequence_length,
        },
        out_dir / "tcn_model.joblib",
    )
    (out_dir / "metrics.txt").write_text(str(metrics), encoding="utf-8")
    print(f"TCN F1: {metrics['f1']:.4f}")
    print(f"Saved: {out_dir / 'tcn_model.joblib'}")


if __name__ == "__main__":
    main()
