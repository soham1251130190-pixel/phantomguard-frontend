"""
mock_data.py
------------
Standalone data generators so the web_app can be built, run, and demoed
RIGHT NOW, before Members 1-4 finish their real modules.

Each function here mimics the OUTPUT SHAPE of a real pillar:
    generate_attack_taxonomy()        <-> identify/attack_database.py
    generate_synthetic_transactions() <-> generate/transaction_sim.py
    generate_realtime_feed()          <-> defend/evaluator.py (live scoring)
    generate_feedback_history()       <-> feedback/loop_controller.py

HOW TO SWITCH TO REAL DATA LATER:
    In dashboard.py / monitoring.py / analytics.py you'll see blocks like:

        try:
            from identify.attack_database import get_all_attacks as generate_attack_taxonomy
        except ImportError:
            from web_app.mock_data import generate_attack_taxonomy

    Once a teammate's file exists on the same interface, the real one is
    used automatically and this mock is ignored. No other code changes needed.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)

CATEGORIES = [
    "Synthetic Identity", "Social Engineering", "Deepfake Voice",
    "Deepfake Video/KYC Bypass", "AI-Generated Phishing",
    "Account Takeover", "Coordinated Multi-Account", "Synthetic Merchant Network",
]

GENAI_CAPABILITIES = [
    "Text Generation (LLM)", "Voice Cloning", "Image/Video Deepfake",
    "Behavioral Mimicry Model", "Synthetic Document Generation",
]

SEVERITIES = ["Low", "Medium", "High", "Critical"]
MERCHANT_CATEGORIES = ["Electronics", "Travel", "Groceries", "Gaming", "Crypto/Wallet", "Fashion", "Subscriptions"]
GEOS = ["Mumbai", "Delhi", "Bengaluru", "Pune", "New York", "London", "Singapore", "Lagos", "Sao Paulo"]


def generate_attack_taxonomy(n_attacks: int = 47) -> pd.DataFrame:
    """Simulates the 47+ attack vector database from Pillar 1 (Identify)."""
    rows = []
    for i in range(n_attacks):
        category = rng.choice(CATEGORIES)
        capability = rng.choice(GENAI_CAPABILITIES)
        severity = rng.choice(SEVERITIES, p=[0.15, 0.35, 0.35, 0.15])
        feasibility = round(rng.uniform(0.2, 0.98), 2)
        impact = round(rng.uniform(0.3, 0.99), 2)
        discovered = datetime.now() - timedelta(days=int(rng.integers(0, 60)))
        rows.append({
            "attack_id": f"ATK-{i+1:03d}",
            "name": f"{category} via {capability}",
            "category": category,
            "genai_capability": capability,
            "severity": severity,
            "feasibility": feasibility,
            "impact_score": impact,
            "priority": round((feasibility * 0.5 + impact * 0.5), 2),
            "discovered_on": discovered.strftime("%Y-%m-%d"),
            "description": (
                f"Combines {capability.lower()} with known {category.lower()} "
                f"patterns to bypass conventional rule-based fraud filters."
            ),
        })
    return pd.DataFrame(rows).sort_values("priority", ascending=False).reset_index(drop=True)


def generate_synthetic_transactions(attack_type: str, volume: int = 500, sophistication: float = 0.5) -> pd.DataFrame:
    """
    Simulates Pillar 2 (Generate) output: synthetic fraud transactions
    for a chosen attack type. `sophistication` in [0,1] controls how well
    the fraud blends in with legitimate behavior (higher = harder to catch).
    """
    n = int(volume)
    base_amount = np.exp(rng.normal(4.2, 1.0, n))  # log-normal amounts
    # Higher sophistication -> amounts/timing look more "normal"
    noise = rng.normal(0, 1 - sophistication, n) * 50
    amount = np.clip(base_amount + noise, 1, None).round(2)

    hour = rng.integers(0, 24, n)
    if sophistication > 0.6:
        hour = np.clip(rng.normal(14, 4, n).astype(int) % 24, 0, 23)  # mimic normal daytime activity

    is_fraud_signature = rng.random(n) > (0.15 + sophistication * 0.3)  # subtler signatures at high sophistication

    df = pd.DataFrame({
        "txn_id": [f"TXN-{attack_type[:3].upper()}-{i:05d}" for i in range(n)],
        "attack_type": attack_type,
        "amount": amount,
        "hour_of_day": hour,
        "merchant_category": rng.choice(MERCHANT_CATEGORIES, n),
        "geo": rng.choice(GEOS, n),
        "device_fingerprint_reused": rng.random(n) < (0.4 - sophistication * 0.3),
        "velocity_flag": rng.random(n) < (0.3 - sophistication * 0.2),
        "carries_fraud_signature": is_fraud_signature,
        "sophistication": sophistication,
    })
    return df


def generate_realtime_feed(n: int = 40) -> pd.DataFrame:
    """Simulates Pillar 3 (Defend) live scoring output for the Defense Monitor."""
    now = datetime.now()
    fraud_prob = np.clip(rng.beta(2, 6, n), 0, 1)  # mostly low, some high
    rows = []
    for i in range(n):
        p = float(fraud_prob[i])
        if p > 0.85:
            action, tier = "Block", "Critical"
        elif p > 0.6:
            action, tier = "Flag for Review", "High"
        elif p > 0.3:
            action, tier = "Monitor", "Medium"
        else:
            action, tier = "Allow", "Low"
        rows.append({
            "timestamp": (now - timedelta(seconds=int((n - i) * 3))).strftime("%H:%M:%S"),
            "txn_id": f"TXN-{100000+i}",
            "amount": round(float(np.exp(rng.normal(4.2, 1.0))), 2),
            "merchant_category": rng.choice(MERCHANT_CATEGORIES),
            "fraud_probability": round(p, 3),
            "risk_tier": tier,
            "recommended_action": action,
        })
    return pd.DataFrame(rows)


def generate_feedback_history() -> pd.DataFrame:
    """
    Simulates the closed feedback loop timeline described in the solution doc:
    Day 1 -> 95%, Day 5 attack evolves -> Day 7 drops to 70%, Day 10 retrain,
    Day 12 -> 94%, Day 15 attack evolves again, etc.
    """
    days = list(range(1, 31))
    detection_acc = []
    attack_sophistication = []
    acc = 0.95
    soph = 0.3
    for d in days:
        if d in (5, 15, 22):
            soph += rng.uniform(0.1, 0.2)
            acc -= rng.uniform(0.2, 0.3)
        elif d in (10, 20, 27):
            acc += rng.uniform(0.15, 0.25)
        else:
            acc += rng.normal(0, 0.01)
            soph += rng.normal(0, 0.005)
        acc = float(np.clip(acc, 0.55, 0.99))
        soph = float(np.clip(soph, 0.2, 0.95))
        detection_acc.append(round(acc, 3))
        attack_sophistication.append(round(soph, 3))

    return pd.DataFrame({
        "day": days,
        "detection_accuracy": detection_acc,
        "attack_sophistication": attack_sophistication,
    })
