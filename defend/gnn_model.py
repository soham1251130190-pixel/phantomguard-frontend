from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from common import (
    build_neighbor_means,
    build_preprocessor,
    evaluate_probabilities,
    load_dataset,
    predict_torch,
    set_seed,
    split_features,
    train_torch_model,
)


DEFAULT_DATA_DIR = "data/phantomguard_tcn_dataset/phantomguard_generate/data"


class GraphSAGEClassifier(nn.Module):
    def __init__(self, feature_count: int, hidden_units: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count * 2, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, self_features: torch.Tensor, neighbor_features: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([self_features, neighbor_features], dim=1)).squeeze(-1)


def train_gnn(
    x_train: np.ndarray,
    train_neighbor_means: np.ndarray,
    y_train,
    epochs: int,
    batch_size: int,
) -> GraphSAGEClassifier:
    model = GraphSAGEClassifier(x_train.shape[1])
    return train_torch_model(model, (x_train, train_neighbor_means), y_train, epochs, batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PhantomGuard GNN model.")
    parser.add_argument("--train-csv", default=f"{DEFAULT_DATA_DIR}/phantomguard_synthetic_dataset.csv")
    parser.add_argument("--out-dir", default="artifacts/gnn")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-neighbors", type=int, default=8)
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

    train_neighbors = build_neighbor_means(train_df, x_train, args.max_neighbors, args.random_state)
    test_neighbors = build_neighbor_means(test_df, x_test, args.max_neighbors, args.random_state)
    model = train_gnn(x_train, train_neighbors, y_train, args.epochs, args.batch_size)
    probabilities = predict_torch(model, (x_test, test_neighbors), args.batch_size)
    metrics = evaluate_probabilities(y_test, probabilities)

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "state_dict": model.state_dict(),
            "feature_count": x_train.shape[1],
            "feature_columns": list(x_raw.columns),
            "max_neighbors": args.max_neighbors,
        },
        out_dir / "gnn_model.joblib",
    )
    (out_dir / "metrics.txt").write_text(str(metrics), encoding="utf-8")
    print(f"GNN F1: {metrics['f1']:.4f}")
    print(f"Saved: {out_dir / 'gnn_model.joblib'}")


if __name__ == "__main__":
    main()
