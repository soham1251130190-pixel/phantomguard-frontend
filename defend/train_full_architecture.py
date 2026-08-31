from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from common import (
    build_neighbor_means,
    build_preprocessor,
    build_sequences,
    evaluate_probabilities,
    format_optional_float,
    load_dataset,
    predict_torch,
    set_seed,
    split_features,
)
from gnn_model import train_gnn
from meta_classifier import stack_probabilities, train_meta_classifier
from tcn_model import train_tcn
from xgboost_model import train_xgboost


DEFAULT_DATA_DIR = "data/phantomguard_tcn_dataset/phantomguard_generate/data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PhantomGuard XGBoost + TCN + GNN + meta-classifier.")
    parser.add_argument("--train-csv", default=f"{DEFAULT_DATA_DIR}/phantomguard_synthetic_dataset.csv")
    parser.add_argument("--tcn-csv", default=f"{DEFAULT_DATA_DIR}/phantomguard_tcn_sequences.csv")
    parser.add_argument("--holdout-csv", default=f"{DEFAULT_DATA_DIR}/phantomguard_realistic_holdout.csv")
    parser.add_argument("--stress-csv", default=f"{DEFAULT_DATA_DIR}/blended_ai_phishing_account_takeover.csv")
    parser.add_argument("--out-dir", default="artifacts/defend_full_tcn")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=10)
    parser.add_argument("--max-neighbors", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    iid_df = load_dataset(args.train_csv)
    tcn_df = load_dataset(args.tcn_csv)
    x_raw, y = split_features(iid_df)

    train_df, temp_df, x_train_raw, x_temp_raw, y_train, y_temp = train_test_split(
        iid_df, x_raw, y, test_size=0.4, random_state=args.random_state, stratify=y
    )
    val_df, test_df, x_val_raw, x_test_raw, y_val, y_test = train_test_split(
        temp_df, x_temp_raw, y_temp, test_size=0.5, random_state=args.random_state, stratify=y_temp
    )

    preprocessor = build_preprocessor(x_train_raw)
    x_train = preprocessor.fit_transform(x_train_raw).astype(np.float32)
    x_val = preprocessor.transform(x_val_raw).astype(np.float32)
    x_test = preprocessor.transform(x_test_raw).astype(np.float32)

    xgb_model = train_xgboost(x_train, y_train, args.random_state)

    tcn_train_df, tcn_val_df, tcn_test_df = split_sequence_dataset(tcn_df, args.random_state)
    tcn_x_train = preprocessor.transform(tcn_train_df[list(x_raw.columns)]).astype(np.float32)
    tcn_x_val = preprocessor.transform(tcn_val_df[list(x_raw.columns)]).astype(np.float32)
    tcn_x_test = preprocessor.transform(tcn_test_df[list(x_raw.columns)]).astype(np.float32)
    tcn_y_train = tcn_train_df["is_fraud"].astype(int)
    tcn_y_val = tcn_val_df["is_fraud"].astype(int)
    tcn_y_test = tcn_test_df["is_fraud"].astype(int)

    tcn_train_seq = build_sequences(tcn_train_df, tcn_x_train, args.sequence_length)
    tcn_val_seq = build_sequences(tcn_val_df, tcn_x_val, args.sequence_length)
    tcn_test_seq = build_sequences(tcn_test_df, tcn_x_test, args.sequence_length)
    tcn_model = train_tcn(tcn_train_seq, tcn_y_train, x_train.shape[1], args.epochs, args.batch_size)

    train_neighbors = build_neighbor_means(train_df, x_train, args.max_neighbors, args.random_state)
    val_neighbors = build_neighbor_means(val_df, x_val, args.max_neighbors, args.random_state)
    test_neighbors = build_neighbor_means(test_df, x_test, args.max_neighbors, args.random_state)
    gnn_model = train_gnn(x_train, train_neighbors, y_train, args.epochs, args.batch_size)

    iid_val_base_probs = stack_probabilities(
        xgb_model.predict_proba(x_val)[:, 1],
        predict_torch(tcn_model, (build_sequences(val_df, x_val, args.sequence_length),), args.batch_size),
        predict_torch(gnn_model, (x_val, val_neighbors), args.batch_size),
    )
    tcn_val_neighbors = build_neighbor_means(tcn_val_df, tcn_x_val, args.max_neighbors, args.random_state)
    tcn_val_base_probs = stack_probabilities(
        xgb_model.predict_proba(tcn_x_val)[:, 1],
        predict_torch(tcn_model, (tcn_val_seq,), args.batch_size),
        predict_torch(gnn_model, (tcn_x_val, tcn_val_neighbors), args.batch_size),
    )
    meta_model = train_meta_classifier(
        np.vstack([iid_val_base_probs, tcn_val_base_probs]),
        np.concatenate([y_val.to_numpy(), tcn_y_val.to_numpy()]),
    )

    test_neighbors = build_neighbor_means(test_df, x_test, args.max_neighbors, args.random_state)
    iid_test_base_probs = stack_probabilities(
        xgb_model.predict_proba(x_test)[:, 1],
        predict_torch(tcn_model, (build_sequences(test_df, x_test, args.sequence_length),), args.batch_size),
        predict_torch(gnn_model, (x_test, test_neighbors), args.batch_size),
    )
    iid_test_meta_probs = meta_model.predict_proba(iid_test_base_probs)[:, 1]

    tcn_test_neighbors = build_neighbor_means(tcn_test_df, tcn_x_test, args.max_neighbors, args.random_state)
    tcn_test_base_probs = stack_probabilities(
        xgb_model.predict_proba(tcn_x_test)[:, 1],
        predict_torch(tcn_model, (tcn_test_seq,), args.batch_size),
        predict_torch(gnn_model, (tcn_x_test, tcn_test_neighbors), args.batch_size),
    )
    tcn_test_meta_probs = meta_model.predict_proba(tcn_test_base_probs)[:, 1]

    metrics = {
        "xgboost_iid_test": evaluate_probabilities(y_test, iid_test_base_probs[:, 0]),
        "tcn_iid_test": evaluate_probabilities(y_test, iid_test_base_probs[:, 1]),
        "gnn_iid_test": evaluate_probabilities(y_test, iid_test_base_probs[:, 2]),
        "meta_iid_test": evaluate_probabilities(y_test, iid_test_meta_probs),
        "tcn_sequence_test": evaluate_probabilities(tcn_y_test, tcn_test_base_probs[:, 1]),
        "meta_sequence_test": evaluate_probabilities(tcn_y_test, tcn_test_meta_probs),
    }

    holdout_metrics = evaluate_external_set(
        Path(args.holdout_csv), preprocessor, xgb_model, tcn_model, gnn_model, meta_model, list(x_raw.columns), args
    )
    stress_metrics = evaluate_external_set(
        Path(args.stress_csv), preprocessor, xgb_model, tcn_model, gnn_model, meta_model, list(x_raw.columns), args
    )

    artifact = {
        "preprocessor": preprocessor,
        "xgboost": xgb_model,
        "tcn_state_dict": tcn_model.state_dict(),
        "tcn_feature_count": x_train.shape[1],
        "gnn_state_dict": gnn_model.state_dict(),
        "gnn_feature_count": x_train.shape[1],
        "meta_classifier": meta_model,
        "feature_columns": list(x_raw.columns),
        "sequence_length": args.sequence_length,
        "max_neighbors": args.max_neighbors,
    }
    summary = {
        "iid_dataset": args.train_csv,
        "tcn_dataset": args.tcn_csv,
        "holdout_dataset": args.holdout_csv,
        "iid_rows": int(len(iid_df)),
        "tcn_rows": int(len(tcn_df)),
        "features": list(x_raw.columns),
        "iid_class_balance": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "tcn_class_balance": {
            str(k): int(v) for k, v in tcn_df["is_fraud"].astype(int).value_counts().sort_index().items()
        },
        "split_rows": {
            "iid_base_train": int(len(train_df)),
            "iid_meta_validation": int(len(val_df)),
            "iid_final_test": int(len(test_df)),
            "tcn_train": int(len(tcn_train_df)),
            "tcn_validation": int(len(tcn_val_df)),
            "tcn_test": int(len(tcn_test_df)),
        },
        "metrics": metrics,
        "holdout_metrics": holdout_metrics,
        "stress_metrics": stress_metrics,
        "notes": [
            "XGBoost and GNN train on the i.i.d. synthetic dataset.",
            "TCN trains on phantomguard_tcn_sequences.csv using sequence_id and seq_position.",
            "Meta-classifier trains on both i.i.d. validation rows and sequence validation rows.",
            "Realistic holdout gives the best false-positive-rate read because it has live-like fraud prevalence.",
        ],
    }

    joblib.dump(artifact, out_dir / "phantomguard_full_architecture.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "metrics_summary.txt").write_text(format_summary(summary), encoding="utf-8")
    print(format_summary(summary))
    print(f"\nSaved full architecture: {out_dir / 'phantomguard_full_architecture.joblib'}")
    print(f"Saved metrics: {out_dir / 'metrics.json'}")


def split_sequence_dataset(df, seed):
    if "sequence_id" not in df.columns:
        train_df, temp_df = train_test_split(df, test_size=0.4, random_state=seed, stratify=df["is_fraud"])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=temp_df["is_fraud"])
        return train_df.copy(), val_df.copy(), test_df.copy()

    labels = df.groupby("sequence_id")["is_fraud"].max().astype(int)
    sequence_ids = labels.index.to_numpy()
    train_ids, temp_ids = train_test_split(sequence_ids, test_size=0.4, random_state=seed, stratify=labels)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=seed, stratify=labels.loc[temp_ids])
    return (
        df[df["sequence_id"].isin(train_ids)].copy(),
        df[df["sequence_id"].isin(val_ids)].copy(),
        df[df["sequence_id"].isin(test_ids)].copy(),
    )


def evaluate_external_set(path, preprocessor, xgb_model, tcn_model, gnn_model, meta_model, feature_columns, args):
    if not path.exists():
        return None
    df = load_dataset(str(path))
    x_raw = df[feature_columns]
    y = df["is_fraud"].astype(int)
    x = preprocessor.transform(x_raw).astype(np.float32)
    sequences = build_sequences(df, x, args.sequence_length)
    neighbors = build_neighbor_means(df, x, args.max_neighbors, args.random_state)
    base_probs = stack_probabilities(
        xgb_model.predict_proba(x)[:, 1],
        predict_torch(tcn_model, (sequences,), args.batch_size),
        predict_torch(gnn_model, (x, neighbors), args.batch_size),
    )
    meta_probs = meta_model.predict_proba(base_probs)[:, 1]
    return {
        "rows": int(len(df)),
        "class_balance": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "xgboost": evaluate_probabilities(y, base_probs[:, 0]),
        "tcn": evaluate_probabilities(y, base_probs[:, 1]),
        "gnn": evaluate_probabilities(y, base_probs[:, 2]),
        "meta_classifier": evaluate_probabilities(y, meta_probs),
    }


def format_summary(summary: dict) -> str:
    lines = [
        "PhantomGuard Full DEFEND Architecture",
        "=" * 44,
        f"IID dataset: {summary['iid_dataset']}",
        f"TCN dataset: {summary['tcn_dataset']}",
        f"Realistic holdout: {summary['holdout_dataset']}",
        f"IID rows: {summary['iid_rows']}",
        f"TCN rows: {summary['tcn_rows']}",
        f"Features used: {len(summary['features'])}",
        f"IID class balance: {summary['iid_class_balance']}",
        f"TCN class balance: {summary['tcn_class_balance']}",
        f"Split rows: {summary['split_rows']}",
        "",
        "Internal Test Metrics",
    ]
    for name, metrics in summary["metrics"].items():
        lines.extend(format_metric_block(name, metrics))

    if summary["holdout_metrics"]:
        lines.extend(["", "Realistic Holdout Metrics", f"Rows: {summary['holdout_metrics']['rows']}"])
        for name in ["xgboost", "tcn", "gnn", "meta_classifier"]:
            lines.extend(format_metric_block(name, summary["holdout_metrics"][name]))

    if summary["stress_metrics"]:
        lines.extend(["", "Blended Attack Stress Test", f"Rows: {summary['stress_metrics']['rows']}"])
        for name in ["xgboost", "tcn", "gnn", "meta_classifier"]:
            lines.extend(format_metric_block(name, summary["stress_metrics"][name]))
    return "\n".join(lines)


def format_metric_block(name: str, metrics: dict) -> list[str]:
    return [
        f"{name}:",
        f"  Accuracy: {metrics['accuracy']:.4f}",
        f"  Precision: {metrics['precision']:.4f}",
        f"  Recall: {metrics['recall']:.4f}",
        f"  F1-score: {metrics['f1']:.4f}",
        f"  ROC-AUC: {format_optional_float(metrics['roc_auc'])}",
        f"  False positive rate: {metrics['false_positive_rate']:.4f}",
        f"  Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}",
    ]


if __name__ == "__main__":
    main()
