"""
evaluator.py
------------
Bridge between the trained PhantomGuard DEFEND models and the ScreenITright
frontend.  Exposes the exact function signature that monitoring.py expects:

    get_live_scored_feed(n: int) -> pd.DataFrame

This module loads the trained four-model architecture once at import time,
then scores randomly generated transactions on demand.
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

# ── Make sibling modules importable regardless of cwd ──────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common import build_neighbor_means, build_sequences  # noqa: E402
from gnn_model import GraphSAGEClassifier  # noqa: E402
from meta_classifier import stack_probabilities  # noqa: E402
from tcn_model import TCNClassifier  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────
ARTIFACT_PATH = _SCRIPT_DIR.parent / "artifacts" / "defend_full_tcn" / "phantomguard_full_architecture.joblib"

CHANNELS = ["card_present", "mobile_app", "card_not_present_online", "p2p_transfer", "atm_withdrawal"]
PAYMENT_METHODS = ["credit_card", "debit_card", "upi", "digital_wallet", "bank_transfer", "prepaid_card"]
DEVICE_TYPES = ["android_mobile", "ios_mobile", "windows_desktop", "mac_desktop", "linux_desktop"]
COUNTRIES = ["US", "GB", "IN", "JP", "DE", "BR", "SG", "NL", "AU", "CA"]
ACCOUNT_CHANNELS = ["web_remote", "in_branch", "agent_assisted", "api_partner_onboarding"]
KYC_LEVELS = ["unverified", "basic_document", "enhanced_verification", "video_kyc"]
CREDIT_BANDS = ["thin_file", "sub_prime", "near_prime", "prime", "super_prime"]
MERCHANT_CATEGORIES = ["Electronics", "Travel", "Groceries", "Gaming", "Crypto/Wallet", "Fashion", "Subscriptions"]
MERCHANT_DESCS = [
    "Grocery Stores", "Department Stores", "Electronics Stores", "Restaurants",
    "Gas Stations", "Insurance", "Liquor Stores", "Hotels/Motels",
]

# Profiles: 70% legitimate, 30% various fraud types
PROFILES: dict[str, dict] = {
    "legitimate": {
        "weight": 0.70,
        "amount": (5, 500), "velocity_1h": (0, 1), "velocity_24h": (0, 3),
        "avg_ratio": (0.5, 2.0), "device_fp": (0.70, 0.99),
        "behavioral": (0.60, 0.99), "login_anomaly": (0.0, 0.15),
        "cross_border": 0.05, "password_reset": 0.02, "account_age": (90, 3000),
        "text_sim": None, "llm_prob": None,
    },
    "ai_phishing": {
        "weight": 0.07,
        "amount": (50, 2000), "velocity_1h": (1, 5), "velocity_24h": (1, 6),
        "avg_ratio": (3.0, 20.0), "device_fp": (0.02, 0.30),
        "behavioral": (0.30, 0.65), "login_anomaly": (0.50, 0.95),
        "cross_border": 0.60, "password_reset": 0.45, "account_age": (10, 800),
        "text_sim": (0.70, 0.97), "llm_prob": (0.70, 0.99),
    },
    "account_takeover": {
        "weight": 0.07,
        "amount": (100, 5000), "velocity_1h": (2, 8), "velocity_24h": (3, 8),
        "avg_ratio": (5.0, 50.0), "device_fp": (0.01, 0.15),
        "behavioral": (0.10, 0.50), "login_anomaly": (0.60, 0.98),
        "cross_border": 0.70, "password_reset": 0.80, "account_age": (200, 2500),
        "text_sim": None, "llm_prob": None,
    },
    "synthetic_identity_fraud": {
        "weight": 0.05,
        "amount": (200, 8000), "velocity_1h": (0, 2), "velocity_24h": (0, 3),
        "avg_ratio": (1.0, 5.0), "device_fp": (0.40, 0.80),
        "behavioral": (0.50, 0.80), "login_anomaly": (0.05, 0.30),
        "cross_border": 0.15, "password_reset": 0.05, "account_age": (0, 90),
        "text_sim": None, "llm_prob": None,
    },
    "voice_cloning_fraud": {
        "weight": 0.05,
        "amount": (500, 15000), "velocity_1h": (0, 2), "velocity_24h": (1, 4),
        "avg_ratio": (8.0, 80.0), "device_fp": (0.05, 0.40),
        "behavioral": (0.20, 0.55), "login_anomaly": (0.40, 0.85),
        "cross_border": 0.50, "password_reset": 0.35, "account_age": (100, 1500),
        "text_sim": None, "llm_prob": None,
    },
    "merchant_fraud": {
        "weight": 0.06,
        "amount": (10, 3000), "velocity_1h": (0, 3), "velocity_24h": (2, 8),
        "avg_ratio": (1.0, 8.0), "device_fp": (0.50, 0.90),
        "behavioral": (0.60, 0.90), "login_anomaly": (0.0, 0.10),
        "cross_border": 0.10, "password_reset": 0.01, "account_age": (30, 1000),
        "text_sim": None, "llm_prob": None,
    },
}


# ── Lazy singleton model loader ───────────────────────────────────────
class _ModelBundle:
    """Loads the trained artifact once and caches everything."""

    _instance: _ModelBundle | None = None

    def __init__(self, path: Path):
        bundle = joblib.load(path)
        self.preprocessor = bundle["preprocessor"]
        self.xgb = bundle["xgboost"]
        self.feature_columns: list[str] = bundle["feature_columns"]
        self.seq_len: int = bundle["sequence_length"]
        self.max_neighbors: int = bundle["max_neighbors"]

        self.tcn = TCNClassifier(bundle["tcn_feature_count"])
        self.tcn.load_state_dict(bundle["tcn_state_dict"])
        self.tcn.eval()

        self.gnn = GraphSAGEClassifier(bundle["gnn_feature_count"])
        self.gnn.load_state_dict(bundle["gnn_state_dict"])
        self.gnn.eval()

        self.meta = bundle["meta_classifier"]

    @classmethod
    def get(cls, path: Path | None = None) -> _ModelBundle:
        if cls._instance is None:
            cls._instance = cls(path or ARTIFACT_PATH)
        return cls._instance


def _rand(lo: float, hi: float) -> float:
    return round(random.uniform(lo, hi), 3)


def _generate_row(profile_name: str, profile: dict, feature_columns: list[str]) -> dict:
    """Generate a single fake transaction matching the model's feature schema."""
    hour = random.randint(0, 23)
    billing = random.choice(COUNTRIES)
    is_cross = random.random() < profile["cross_border"]
    ip_country = random.choice([c for c in COUNTRIES if c != billing]) if is_cross else billing

    row: dict = {
        "amount": round(random.uniform(*profile["amount"]), 2),
        "currency": "USD",
        "channel": random.choice(CHANNELS),
        "payment_method": random.choice(PAYMENT_METHODS),
        "merchant_category_code": random.choice([4111, 5311, 5411, 5732, 5921, 6012, 6300, 7011, 8299]),
        "merchant_category_desc": random.choice(MERCHANT_DESCS),
        "merchant_risk_score": _rand(0.01, 0.98) if profile_name != "legitimate" else _rand(0.01, 0.40),
        "device_type": random.choice(DEVICE_TYPES),
        "device_fingerprint_score": _rand(*profile["device_fp"]),
        "ip_country": ip_country,
        "billing_country": billing,
        "is_cross_border": is_cross,
        "hour_of_day": hour,
        "is_night_time": hour <= 5,
        "time_since_last_txn_sec": round(random.uniform(10, 200000), 2),
        "txn_velocity_1h": random.randint(*profile["velocity_1h"]),
        "txn_velocity_24h": random.randint(*profile["velocity_24h"]),
        "amount_vs_user_avg_ratio": _rand(*profile["avg_ratio"]),
        "account_age_days": random.randint(*profile["account_age"]),
        "account_creation_channel": random.choice(ACCOUNT_CHANNELS),
        "kyc_verification_level": random.choice(KYC_LEVELS),
        "credit_score_band": random.choice(CREDIT_BANDS),
        "historical_avg_amount": round(random.uniform(15, 200), 2),
        "historical_txn_count": random.randint(0, 1200),
        "num_devices_used": random.randint(1, 3),
        "num_linked_accounts": random.randint(0, 4),
        "behavioral_score": _rand(*profile["behavioral"]),
        "login_anomaly_score": _rand(*profile["login_anomaly"]),
        "password_reset_recent": random.random() < profile["password_reset"],
        "mfa_enabled": random.choice([True, False]),
        "prior_fraud_flags": random.choices([0, 1, 2], weights=[0.85, 0.10, 0.05])[0],
        "shared_device_n_accounts": random.randint(1, 5),
        "shared_ip_n_accounts": random.randint(1, 5),
        "num_beneficiaries_30d": random.randint(0, 6),
        "beneficiary_account_age_days": round(random.uniform(0, 900), 2),
        "text_similarity_to_phishing_corpus": _rand(*profile["text_sim"]) if profile.get("text_sim") else np.nan,
        "llm_generated_content_prob": _rand(*profile["llm_prob"]) if profile.get("llm_prob") else np.nan,
        "voice_authenticity_score": np.nan,
        "deepfake_video_score": np.nan,
        "document_authenticity_score": np.nan,
        "image_manipulation_score": np.nan,
        "refund_count_30d": random.randint(0, 3),
        "refund_to_purchase_ratio": np.nan,
        "structuring_score": _rand(0.0, 0.20) if profile_name == "legitimate" else _rand(0.0, 0.95),
    }
    return {col: row.get(col, np.nan) for col in feature_columns}


def score_transaction(row_dict: dict) -> dict[str, float]:
    """Score a single transaction through all four models. Returns per-model probabilities."""
    m = _ModelBundle.get()
    df = pd.DataFrame([row_dict])
    df["timestamp"] = datetime.now(timezone.utc).isoformat()
    df["user_id"] = "SIM_USER"

    x_raw = df[m.feature_columns]
    x = m.preprocessor.transform(x_raw).astype(np.float32)

    # XGBoost
    xgb_p = float(m.xgb.predict_proba(x)[:, 1][0])

    # TCN — single-row, zero-padded sequence
    seq = np.zeros((1, x.shape[1], m.seq_len), dtype=np.float32)
    seq[:, :, -1] = x
    with torch.no_grad():
        tcn_p = float(torch.sigmoid(m.tcn(torch.tensor(seq, dtype=torch.float32))).item())

    # GNN — self = neighbor for isolated transaction
    x_t = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        gnn_p = float(torch.sigmoid(m.gnn(x_t, x_t)).item())

    # Meta-classifier
    base = stack_probabilities(np.array([xgb_p]), np.array([tcn_p]), np.array([gnn_p]))
    meta_p = float(m.meta.predict_proba(base)[:, 1][0])

    return {"xgboost": xgb_p, "tcn": tcn_p, "gnn": gnn_p, "meta": meta_p}


def _classify(meta_prob: float) -> tuple[str, str]:
    """Return (risk_tier, recommended_action) based on meta-classifier score."""
    if meta_prob >= 0.85:
        return "Critical", "Block"
    if meta_prob >= 0.60:
        return "High", "Flag for Review"
    if meta_prob >= 0.30:
        return "Medium", "Monitor"
    return "Low", "Allow"


# ── Public API — the function monitoring.py imports ───────────────────

def get_live_scored_feed(n: int = 40) -> pd.DataFrame:
    """
    Generate *n* random transactions, score each through the trained
    DEFEND ensemble, and return a DataFrame matching the shape that
    monitoring.py expects.

    Columns: timestamp, txn_id, amount, merchant_category, channel,
             xgb_score, tcn_score, gnn_score, fraud_probability,
             risk_tier, recommended_action
    """
    m = _ModelBundle.get()
    profile_names = list(PROFILES.keys())
    profile_weights = [PROFILES[p]["weight"] for p in profile_names]
    now = datetime.now()

    rows: list[dict] = []
    for i in range(n):
        pname = random.choices(profile_names, weights=profile_weights, k=1)[0]
        profile = PROFILES[pname]
        raw = _generate_row(pname, profile, m.feature_columns)
        scores = score_transaction(raw)

        tier, action = _classify(scores["meta"])
        rows.append({
            "timestamp": (now - timedelta(seconds=int((n - i) * 3))).strftime("%H:%M:%S"),
            "txn_id": f"TXN-{random.randint(100000, 999999)}",
            "amount": round(raw.get("amount", 0), 2),
            "merchant_category": random.choice(MERCHANT_CATEGORIES),
            "channel": raw.get("channel", "unknown"),
            "xgb_score": round(scores["xgboost"], 3),
            "tcn_score": round(scores["tcn"], 3),
            "gnn_score": round(scores["gnn"], 3),
            "fraud_probability": round(scores["meta"], 3),
            "risk_tier": tier,
            "recommended_action": action,
        })

    return pd.DataFrame(rows)


# ── Quick self-test ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading models...")
    feed = get_live_scored_feed(5)
    print(feed.to_string())
    print(f"\nScored {len(feed)} transactions successfully.")
