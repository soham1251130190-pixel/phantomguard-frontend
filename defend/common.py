from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


LABEL_COLUMN = "is_fraud"

LEAKAGE_COLUMNS = {
    "transaction_id",
    "user_id",
    "is_fraud",
    "attack_type",
    "attack_category",
    "genai_capability",
    "fraud_severity",
    "is_blended_attack",
    "ground_truth_source",
}

HIGH_CARDINALITY_COLUMNS = {
    "timestamp",
    "merchant_id",
    "device_id",
    "community_cluster_id",
    "sequence_id",
    "seq_position",
    "seq_length",
    "temporal_pattern",
}

GRAPH_COLUMNS = [
    "merchant_id",
    "device_id",
    "community_cluster_id",
    "ip_country",
    "billing_country",
    "merchant_category_code",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Dataset must contain target column: {LABEL_COLUMN}")
    return df


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    excluded = LEAKAGE_COLUMNS | HIGH_CARDINALITY_COLUMNS
    feature_columns = [column for column in df.columns if column not in excluded]
    return df[feature_columns], df[LABEL_COLUMN].astype(int)


def build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = [
        column
        for column in x_train.columns
        if not is_numeric_dtype(x_train[column]) or is_bool_dtype(x_train[column])
    ]
    numeric_columns = [column for column in x_train.columns if column not in categorical_columns]

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("clean", FunctionTransformer(_clean_categorical_values, validate=False)),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )


def _clean_categorical_values(values: pd.DataFrame) -> pd.DataFrame:
    return values.fillna("missing").astype(str)


def build_sequences(df: pd.DataFrame, transformed: np.ndarray, sequence_length: int) -> np.ndarray:
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    group_column = "sequence_id" if "sequence_id" in df.columns else "user_id"
    order_column = "seq_position" if "seq_position" in df.columns else "_timestamp"
    work = df[[group_column]].copy()
    work["_timestamp"] = timestamps
    if "seq_position" in df.columns:
        work["seq_position"] = pd.to_numeric(df["seq_position"], errors="coerce").fillna(0)
    work["_position"] = np.arange(len(df))
    sequences = np.zeros((len(df), sequence_length, transformed.shape[1]), dtype=np.float32)

    for _, group in work.sort_values(order_column).groupby(group_column, sort=False):
        positions = group["_position"].to_numpy()
        for offset, position in enumerate(positions):
            start = max(0, offset - sequence_length + 1)
            history = positions[start : offset + 1]
            sequences[position, -len(history) :, :] = transformed[history]

    return np.transpose(sequences, (0, 2, 1))


def build_neighbor_means(df: pd.DataFrame, transformed: np.ndarray, max_neighbors: int, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    neighbors: list[set[int]] = [set() for _ in range(len(df))]
    usable_graph_columns = [column for column in GRAPH_COLUMNS if column in df.columns]

    for column in usable_graph_columns:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, value in enumerate(df[column].fillna("missing").astype(str)):
            groups[value].append(index)
        for indices in groups.values():
            if len(indices) <= 1:
                continue
            for index in indices:
                candidates = [candidate for candidate in indices if candidate != index]
                sampled = rng.sample(candidates, min(max_neighbors, len(candidates)))
                neighbors[index].update(sampled)

    neighbor_means = np.zeros_like(transformed, dtype=np.float32)
    for index, index_neighbors in enumerate(neighbors):
        if index_neighbors:
            neighbor_means[index] = transformed[list(index_neighbors)].mean(axis=0)
        else:
            neighbor_means[index] = transformed[index]
    return neighbor_means


def train_torch_model(
    model: nn.Module,
    feature_arrays: tuple[np.ndarray, ...],
    labels: pd.Series,
    epochs: int,
    batch_size: int,
) -> nn.Module:
    tensors = [torch.tensor(array, dtype=torch.float32) for array in feature_arrays]
    y_tensor = torch.tensor(labels.to_numpy(), dtype=torch.float32)
    loader = DataLoader(TensorDataset(*tensors, y_tensor), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        for batch in loader:
            *features, y_batch = batch
            optimizer.zero_grad()
            logits = model(*features)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
    return model


def predict_torch(model: nn.Module, feature_arrays: tuple[np.ndarray, ...], batch_size: int) -> np.ndarray:
    tensors = [torch.tensor(array, dtype=torch.float32) for array in feature_arrays]
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=False)
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(*batch)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities)


def evaluate_probabilities(y_true: pd.Series, probabilities: np.ndarray) -> dict:
    predictions = (probabilities >= 0.5).astype(int)
    roc_auc = roc_auc_score(y_true, probabilities) if y_true.nunique() > 1 else None
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc,
        "false_positive_rate": false_positive_rate(y_true, predictions),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_true, predictions, labels=[0, 1], zero_division=0),
    }


def false_positive_rate(y_true: pd.Series, y_pred: np.ndarray) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) else 0.0


def format_optional_float(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "not defined for one-class set"
