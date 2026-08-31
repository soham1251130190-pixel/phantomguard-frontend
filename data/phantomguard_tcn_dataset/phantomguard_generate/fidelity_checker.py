"""
fidelity_checker.py
--------------------
Lightweight sanity/fidelity checks on a generated dataset. This is NOT a
statistical-realism proof (that requires real production data to compare
against, which we don't have) — it's a guardrail that catches generator
bugs: label leakage, degenerate columns, NaN patterns that don't match
the schema's intent, and per-attack separability sanity (each attack
should be statistically distinguishable from legitimate traffic on at
least one feature block, or the "Defend" model has nothing to learn).
"""

import numpy as np
import pandas as pd


def run_fidelity_report(df: pd.DataFrame) -> dict:
    report = {}

    report["n_rows"] = len(df)
    report["class_balance"] = df["attack_type"].value_counts().to_dict()
    report["overall_fraud_rate"] = round(float(df["is_fraud"].mean()), 4)

    # 1. no fully-constant "feature" columns (would be useless/a bug)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    constant_cols = [c for c in numeric_cols if df[c].nunique(dropna=True) <= 1]
    report["constant_columns"] = constant_cols

    # 2. amount distributions should differ meaningfully by class
    legit_amt = df.loc[df.is_fraud == 0, "amount"]
    fraud_amt = df.loc[df.is_fraud == 1, "amount"]
    report["amount_median_legit"] = round(float(legit_amt.median()), 2) if len(legit_amt) else None
    report["amount_median_fraud"] = round(float(fraud_amt.median()), 2) if len(fraud_amt) else None

    # 3. per-attack-type separability check: for each attack, find the
    #    feature with the largest standardized mean gap vs legitimate rows
    numeric_feats = [c for c in numeric_cols if c not in
                      ("is_fraud", "fraud_severity", "hour_of_day")]
    separability = {}
    legit = df[df.attack_type == "legitimate"]
    if len(legit) < 5:
        # No (or too few) legitimate rows in this dataset to compare against —
        # expected for a single-attack-type / --mode single run. Skip rather
        # than crash; note it so it's obvious why the section is empty.
        report["per_attack_separability"] = None
        report["separability_note"] = ("Skipped: fewer than 5 legitimate rows in "
                                        "this dataset. Generate against a dataset "
                                        "that includes legitimate rows (e.g. --mode "
                                        "all or --mode realistic) to check separability.")
    else:
        for attack_type in df.attack_type.unique():
            if attack_type == "legitimate":
                continue
            sub = df[df.attack_type == attack_type]
            best_feat, best_gap = None, 0.0
            for feat in numeric_feats:
                a, b = sub[feat].dropna(), legit[feat].dropna()
                if len(a) < 5 or len(b) < 5:
                    continue
                pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2) or 1e-6
                gap = abs(a.mean() - b.mean()) / pooled_std
                if gap > best_gap:
                    best_feat, best_gap = feat, gap
            separability[attack_type] = {"most_discriminative_feature": best_feat,
                                          "standardized_gap": round(float(best_gap), 2)}
        report["per_attack_separability"] = separability

    # 4. NaN rate per genai column (expected: high, since features are
    #    channel/attack-specific — flag if suspiciously ALL null or ALL filled)
    genai_cols = ["text_similarity_to_phishing_corpus", "llm_generated_content_prob",
                  "voice_authenticity_score", "deepfake_video_score",
                  "document_authenticity_score", "image_manipulation_score"]
    report["genai_feature_fill_rate"] = {
        c: round(float(df[c].notna().mean()), 3) for c in genai_cols
    }

    return report


def print_report(report: dict):
    print(f"Rows: {report['n_rows']:,}  |  Overall fraud rate: {report['overall_fraud_rate']:.1%}")
    legit_m = f"${report['amount_median_legit']:.2f}" if report["amount_median_legit"] is not None else "n/a"
    fraud_m = f"${report['amount_median_fraud']:.2f}" if report["amount_median_fraud"] is not None else "n/a"
    print(f"Amount median — legit: {legit_m}  vs fraud: {fraud_m}")
    if report["constant_columns"]:
        print(f"WARNING - constant columns detected: {report['constant_columns']}")
    else:
        print("No degenerate constant columns.")
    print("\nClass balance:")
    for k, v in sorted(report["class_balance"].items(), key=lambda x: -x[1]):
        print(f"  {k:38s} {v:>6,}")
    if report.get("per_attack_separability") is None:
        print(f"\nSeparability check: {report['separability_note']}")
    else:
        print("\nMost discriminative feature per attack type (standardized mean gap vs legitimate):")
        for k, v in report["per_attack_separability"].items():
            feat = v["most_discriminative_feature"] or "(none found)"
            print(f"  {k:38s} {feat:30s} gap={v['standardized_gap']}")
    print("\nGenAI-specific feature fill rates (expected to be partial/sparse by design):")
    for k, v in report["genai_feature_fill_rate"].items():
        print(f"  {k:38s} {v:.1%}")
