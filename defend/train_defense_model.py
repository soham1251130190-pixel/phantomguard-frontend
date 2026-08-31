from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PhantomGuard DEFEND fraud model.")
    parser.add_argument(
        "--train-csv",
        default="data/phantomguard_dataset/phantomguard_synthetic_dataset.csv",
        help="Main labeled training dataset.",
    )
    parser.add_argument(
        "--stress-csv",
        default="data/phantomguard_dataset/blended_ai_phishing_account_takeover.csv",
        help="Optional blended-attack stress-test dataset.",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/defend",
        help="Directory for trained model and metrics.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
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

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("clean", FunctionTransformer(_clean_categorical_values, validate=False)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def _clean_categorical_values(values: pd.DataFrame) -> pd.DataFrame:
    return values.fillna("missing").astype(str)


def build_model(x_train: pd.DataFrame) -> Pipeline:
    preprocessor = build_preprocessor(x_train)

    ensemble = VotingClassifier(
        estimators=[
            (
                "hist_gradient_boosting",
                HistGradientBoostingClassifier(
                    learning_rate=0.08,
                    max_iter=180,
                    l2_regularization=0.01,
                    random_state=42,
                ),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=180,
                    max_depth=14,
                    min_samples_leaf=3,
                    n_jobs=-1,
                    random_state=42,
                    class_weight="balanced_subsample",
                ),
            ),
            (
                "logistic_regression",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ],
        voting="soft",
        weights=[3, 2, 1],
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", ensemble)])


def evaluate(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict:
    predictions = model.predict(x)
    probabilities = model.predict_proba(x)[:, 1]
    roc_auc = roc_auc_score(y, probabilities) if y.nunique() > 1 else None

    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc,
        "false_positive_rate": _false_positive_rate(y, predictions),
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y, predictions, labels=[0, 1], zero_division=0),
    }


def _false_positive_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn) if (fp + tn) else 0.0


def evaluate_stress_set(model: Pipeline, stress_path: Path, feature_columns: list[str]) -> dict | None:
    if not stress_path.exists():
        return None

    df = load_dataset(stress_path)
    x = df[feature_columns]
    y = df[LABEL_COLUMN].astype(int)
    metrics = evaluate(model, x, y)
    metrics["rows"] = int(len(df))
    metrics["attack_types"] = sorted(df["attack_type"].dropna().unique().tolist())
    return metrics


def main() -> None:
    args = parse_args()
    train_path = Path(args.train_csv)
    stress_path = Path(args.stress_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(train_path)
    x, y = split_features(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = build_model(x_train)
    model.fit(x_train, y_train)

    train_metrics = evaluate(model, x_train, y_train)
    test_metrics = evaluate(model, x_test, y_test)
    stress_metrics = evaluate_stress_set(model, stress_path, list(x.columns))

    summary = {
        "dataset": str(train_path),
        "rows": int(len(df)),
        "features": list(x.columns),
        "excluded_columns": sorted((LEAKAGE_COLUMNS | HIGH_CARDINALITY_COLUMNS) & set(df.columns)),
        "class_balance": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "stress_metrics": stress_metrics,
    }

    model_path = out_dir / "phantomguard_defense_model.joblib"
    metrics_path = out_dir / "metrics.json"
    report_path = out_dir / "metrics_summary.txt"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.write_text(format_summary(summary), encoding="utf-8")

    print(format_summary(summary))
    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")


def format_summary(summary: dict) -> str:
    test = summary["test_metrics"]
    lines = [
        "PhantomGuard DEFEND Training Summary",
        "=" * 40,
        f"Dataset: {summary['dataset']}",
        f"Rows: {summary['rows']}",
        f"Features used: {len(summary['features'])}",
        f"Class balance: {summary['class_balance']}",
        "",
        "Holdout Metrics",
        f"Accuracy: {test['accuracy']:.4f}",
        f"Precision: {test['precision']:.4f}",
        f"Recall: {test['recall']:.4f}",
        f"F1-score: {test['f1']:.4f}",
        f"ROC-AUC: {_format_optional_float(test['roc_auc'])}",
        f"False positive rate: {test['false_positive_rate']:.4f}",
        f"Confusion matrix [[TN, FP], [FN, TP]]: {test['confusion_matrix']}",
    ]

    if summary["stress_metrics"]:
        stress = summary["stress_metrics"]
        lines.extend(
            [
                "",
                "Blended Attack Stress Test",
                f"Rows: {stress['rows']}",
                f"Attack types: {', '.join(stress['attack_types'])}",
                f"Recall: {stress['recall']:.4f}",
                f"F1-score: {stress['f1']:.4f}",
                f"ROC-AUC: {_format_optional_float(stress['roc_auc'])}",
                f"Confusion matrix [[TN, FP], [FN, TP]]: {stress['confusion_matrix']}",
            ]
        )

    return "\n".join(lines)


def _format_optional_float(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "not defined for one-class set"


if __name__ == "__main__":
    main()
