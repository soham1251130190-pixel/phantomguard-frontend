"""
transaction_sim.py
-------------------
The generic transaction generator. This is the piece the task explicitly
asked for: attack_type is a *parameter*, not a hard-coded branch tree.

    generate_transactions(attack_type="voice_cloning_fraud", n=500)
    generate_transactions(attack_type=None, n=500)   # legitimate baseline

Adding attack #11 requires zero changes to this file — only a new entry
in config.ATTACK_REGISTRY and a new function in pattern_injector.py.

Flow per row:
  1. pick/create a Persona whose profile matches what this attack_type
     realistically targets (thin-file, established-trust, victim, etc.)
  2. build a plausible *legitimate* baseline transaction for that persona
     (this is what "fidelity" means: fraud rows start from a realistic
     transaction, not a hand-wavy template)
  3. if attack_type is set, look up its injector in the registry and let
     it mutate the specific fields that actually carry the fraud signal
  4. attach labels (is_fraud, attack_type, attack_category, genai
     capability, fraud_severity) and provenance
"""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import fake_data as fd
from persona_generator import generate_persona
from config import get_attack_config, ATTACK_REGISTRY
import pattern_injector as pi
from schema import ALL_COLUMNS

BASE_DATE = datetime(2026, 6, 1)

# which persona profile each attack type realistically needs
ATTACK_PERSONA_PROFILE = {
    "synthetic_identity_fraud": "new_thin",
    "account_takeover": "established_trust",
    "ai_phishing": "mule_or_victim",
    "voice_cloning_fraud": "mule_or_victim",
    "deepfake_identity_fraud": "new_thin",
    "synthetic_document_fraud": "new_thin",
    "payment_diversion_invoice_fraud": "established_trust",
    "merchant_fraud": "normal",
    "refund_fraud": "normal",
    "coordinated_multi_account_fraud": "new_thin",
}


def build_baseline_transaction(rng: np.random.Generator, persona, channel_bias=None) -> dict:
    """A realistic, non-fraudulent transaction for this persona."""
    merchant_id, mcc, mcc_desc = fd.sample_merchant(rng, bias_high_risk_prob=0.08)
    channel = str(rng.choice(channel_bias)) if channel_bias else str(rng.choice(fd.CHANNELS))
    hour = int(rng.integers(0, 24))
    ts = BASE_DATE + timedelta(
        days=int(rng.integers(0, 90)), hours=hour,
        minutes=int(rng.integers(0, 60)), seconds=int(rng.integers(0, 60)),
    )
    ip_country = persona.billing_country if rng.random() < 0.92 else str(fd.sample_country(rng))

    amount = round(float(rng.lognormal(mean=np.log(max(persona.historical_avg_amount, 5)), sigma=0.35)), 2)

    return {
        "transaction_id": None,  # filled by caller
        "user_id": persona.user_id,
        "timestamp": ts,
        "amount": amount,
        "currency": "USD",
        "channel": channel,
        "payment_method": str(rng.choice(fd.PAYMENT_METHODS)),
        "merchant_id": merchant_id,
        "merchant_category_code": mcc,
        "merchant_category_desc": mcc_desc,
        "merchant_risk_score": round(float(rng.beta(2, 8)), 3),
        "device_id": persona.home_device_id,
        "device_type": persona.home_device_type,
        "device_fingerprint_score": round(float(rng.uniform(0.75, 0.99)), 3),
        "ip_country": ip_country,
        "billing_country": persona.billing_country,
        "is_cross_border": ip_country != persona.billing_country,
        "hour_of_day": hour,
        "is_night_time": hour < 5,
        "time_since_last_txn_sec": float(rng.exponential(3600 * 12)),
        "txn_velocity_1h": int(rng.poisson(0.15)),
        "txn_velocity_24h": int(rng.poisson(1.2)),
        "amount_vs_user_avg_ratio": round(amount / max(persona.historical_avg_amount, 5), 2),

        "account_age_days": persona.account_age_days,
        "account_creation_channel": persona.account_creation_channel,
        "kyc_verification_level": persona.kyc_verification_level,
        "credit_score_band": persona.credit_score_band,
        "historical_avg_amount": persona.historical_avg_amount,
        "historical_txn_count": persona.historical_txn_count,
        "num_devices_used": persona.num_devices_used,
        "num_linked_accounts": persona.num_linked_accounts,
        "behavioral_score": persona.behavioral_score,
        "login_anomaly_score": round(float(rng.uniform(0.0, 0.15)), 3),
        "password_reset_recent": False,
        "mfa_enabled": persona.mfa_enabled,
        "prior_fraud_flags": persona.prior_fraud_flags,

        "shared_device_n_accounts": 1,
        "shared_ip_n_accounts": int(rng.integers(1, 3)),
        "community_cluster_id": f"CLUSTER{rng.integers(1000, 9999)}",
        "num_beneficiaries_30d": int(rng.poisson(0.5)),
        "beneficiary_account_age_days": float(rng.uniform(30, 900)),

        "text_similarity_to_phishing_corpus": np.nan,
        "llm_generated_content_prob": np.nan,
        "voice_authenticity_score": np.nan,
        "deepfake_video_score": np.nan,
        "document_authenticity_score": np.nan,
        "image_manipulation_score": np.nan,

        "refund_count_30d": 0,
        "refund_to_purchase_ratio": np.nan,
        "structuring_score": round(float(rng.uniform(0.0, 0.15)), 3),
    }


def _apply_injector(rng, persona, txn, attack_type, cfg, ring_ctx):
    injector = pi.INJECTORS[cfg["injector"]]
    if attack_type == "coordinated_multi_account_fraud":
        return injector(rng, persona, txn, ring_id=ring_ctx["id"],
                         ring_device_id=ring_ctx["device"], ring_ip_country=ring_ctx["ip"])
    return injector(rng, persona, txn)


def generate_transactions(attack_type, n, seed=None, ring_size=6):
    """
    Core generic entry point.

    attack_type : one of —
                  * None                          -> legitimate transactions only
                  * str, e.g. "account_takeover"   -> single attack vector
                  * list/tuple of str, e.g.
                    ["ai_phishing", "account_takeover"] -> a BLENDED/CHAINED
                    attack: each row runs through every listed injector in
                    sequence (order randomized per-row for diversity), so
                    the row carries BOTH attacks' feature signatures at
                    once. Any registered attack_type can be combined with
                    any other — no extra registration needed.
    n           : number of rows to generate.
    seed        : rng seed for reproducibility.
    ring_size   : only used when coordinated_multi_account_fraud is one of
                  the attack types, controls the shared device/IP "ring".

    Returns a pandas.DataFrame with columns == schema.ALL_COLUMNS.
    """
    rng = np.random.default_rng(seed)
    rows = []

    if attack_type is None:
        for i in range(n):
            persona = generate_persona(rng, persona_id=rng.integers(1, 10_000_000), profile="normal")
            txn = build_baseline_transaction(rng, persona)
            txn["transaction_id"] = f"TXN{rng.integers(10**11, 10**12)}"
            txn.update({
                "is_fraud": 0, "attack_type": "legitimate", "attack_category": "legitimate",
                "genai_capability": "none", "fraud_severity": 0.0, "is_blended_attack": False,
                "ground_truth_source": "synthetic",
            })
            rows.append(txn)
        return pd.DataFrame(rows)[ALL_COLUMNS]

    # normalize to a list so single-attack and combo-attack share one code path
    attack_list = [attack_type] if isinstance(attack_type, str) else list(attack_type)
    cfgs = [get_attack_config(a) for a in attack_list]
    is_combo = len(attack_list) > 1

    # persona profile: for a combo, use the profile of the FIRST attack listed
    # (e.g. ai_phishing+account_takeover -> "mule_or_victim", since that's who
    # gets phished; the ATO signature still gets injected on top regardless)
    profile = ATTACK_PERSONA_PROFILE.get(attack_list[0], "normal")
    channel_bias = cfgs[0]["default_channel_bias"]

    ring_ctx = {
        "device": f"DEV{rng.integers(10_000_000, 99_999_999)}",
        "ip": str(fd.sample_country(rng, bias="high_risk")),
        "id": f"CLUSTER{rng.integers(1000, 9999)}",
    }

    for i in range(n):
        persona = generate_persona(rng, persona_id=rng.integers(1, 10_000_000), profile=profile)
        txn = build_baseline_transaction(rng, persona, channel_bias=channel_bias)
        txn["transaction_id"] = f"TXN{rng.integers(10**11, 10**12)}"

        # apply each injector in (per-row randomized) order so that when two
        # injectors both write the same field, which one "wins" isn't a fixed
        # artifact of registry order — it varies across rows like real chained
        # attacks would (sometimes the ATO signature dominates, sometimes phishing does)
        order = rng.permutation(len(attack_list))
        severities = []
        for idx in order:
            a, cfg = attack_list[idx], cfgs[idx]
            txn = _apply_injector(rng, persona, txn, a, cfg, ring_ctx)
            severities.append(txn.get("fraud_severity", 0.5))

        # combined severity = probabilistic OR of each individual severity,
        # not just whichever injector happened to write last
        combined_severity = 1.0
        for s in severities:
            combined_severity *= (1.0 - s)
        combined_severity = round(min(0.99, 1.0 - combined_severity), 3)

        txn.update({
            "is_fraud": 1,
            "attack_type": "+".join(a for a in attack_list) if is_combo else attack_list[0],
            "attack_category": "+".join(dict.fromkeys(c["category"] for c in cfgs)),
            "genai_capability": "+".join(dict.fromkeys(c["genai_capability"] for c in cfgs)),
            "fraud_severity": combined_severity,
            "ground_truth_source": "synthetic",
        })
        rows.append(txn)

    df = pd.DataFrame(rows)
    df["is_blended_attack"] = is_combo
    return df[ALL_COLUMNS]


def generate_balanced_dataset(n_per_attack=1200, n_legit=None, seed=42):
    """
    Convenience wrapper: generates all 10 attack types + a legitimate
    baseline and concatenates into one training-ready DataFrame.
    n_legit defaults to n_per_attack * (num_attack_types) so classes are
    roughly balanced fraud-vs-legit overall (typical for a *training* set;
    real-world deployment eval should instead use a realistic, imbalanced
    holdout — see fidelity_checker.py).
    """
    frames = []
    for idx, attack_type in enumerate(ATTACK_REGISTRY.keys()):
        frames.append(generate_transactions(attack_type, n_per_attack, seed=seed + idx + 1))

    if n_legit is None:
        n_legit = n_per_attack * len(ATTACK_REGISTRY)
    frames.append(generate_transactions(None, n_legit, seed=seed))

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle
    return df


def generate_realistic_holdout(n_total=50_000, fraud_rate=0.006, seed=7):
    """
    A second, deliberately IMBALANCED dataset that mimics live traffic
    (real payment networks see well under 1% fraud). Use the balanced
    dataset from generate_balanced_dataset() to *train*, and this one to
    *evaluate* — that's the honest way to report F1 / false-positive
    rate, and matches how the challenge brief frames 'real-world
    feasibility in live payments'.

    Attack mix is weighted by each attack's severity_weight so rarer/
    higher-impact attack types are proportionally less common, which is
    closer to real fraud-type prevalence than a uniform split.
    """
    n_fraud = int(n_total * fraud_rate)
    n_legit = n_total - n_fraud

    weights = np.array([ATTACK_REGISTRY[a]["severity_weight"] for a in ATTACK_REGISTRY])
    weights = weights / weights.sum()
    per_attack_counts = np.random.default_rng(seed).multinomial(n_fraud, weights)

    frames = [generate_transactions(None, n_legit, seed=seed)]
    for (attack_type, _), count in zip(ATTACK_REGISTRY.items(), per_attack_counts):
        if count > 0:
            frames.append(generate_transactions(attack_type, int(count), seed=seed + hash(attack_type) % 1000))

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df
