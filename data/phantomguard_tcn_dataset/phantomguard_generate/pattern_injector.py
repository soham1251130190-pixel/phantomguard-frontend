"""
pattern_injector.py
--------------------
One function per attack type. Each injector receives the rng, the
persona, and a *baseline legitimate* transaction dict (already populated
by transaction_sim.build_baseline_transaction), and returns a mutated
dict where the fields that a real fraud/GenAI-defense feature pipeline
would actually move are pushed into that attack's realistic range.

Design rule: injectors only ever *overwrite specific keys* — they never
need to know the full schema, which keeps them short and makes adding
attack #11 a ~30-line function.

Every injector also sets `fraud_severity` (0-1) — a soft label so
Member 3 can do cost-sensitive / ranked training, not just binary.
"""

import numpy as np
import fake_data as fd


def _bump(rng, low, high):
    return float(rng.uniform(low, high))


# ---------------------------------------------------------------- 1 -----
def inject_synthetic_identity_fraud(rng, persona, txn):
    """
    New, thin-file identity built specifically to earn trust then 'bust out'.
    Signature: young account, fabricated-but-plausible documents, a sudden
    large transaction relative to a short, clean history.
    """
    txn["account_age_days"] = int(rng.integers(1, 60))
    txn["kyc_verification_level"] = str(rng.choice(["unverified", "basic_document"], p=[0.4, 0.6]))
    txn["credit_score_band"] = "thin_file"
    txn["historical_txn_count"] = int(rng.integers(0, 6))
    txn["num_linked_accounts"] = int(rng.integers(0, 3))
    txn["document_authenticity_score"] = _bump(rng, 0.35, 0.68)
    txn["image_manipulation_score"] = _bump(rng, 0.25, 0.6)
    txn["llm_generated_content_prob"] = _bump(rng, 0.4, 0.85)  # synthetic bio/profile text
    # bust-out: amount far above the thin history
    txn["amount"] = round(float(rng.lognormal(mean=6.3, sigma=0.5)), 2)
    txn["amount_vs_user_avg_ratio"] = round(txn["amount"] / max(txn["historical_avg_amount"], 10), 2)
    txn["device_fingerprint_score"] = _bump(rng, 0.4, 0.7)
    txn["behavioral_score"] = _bump(rng, 0.25, 0.55)
    txn["fraud_severity"] = _bump(rng, 0.6, 0.9)
    return txn


# ---------------------------------------------------------------- 2 -----
def inject_account_takeover(rng, persona, txn):
    """
    Established, trusted account suddenly accessed from a new device/geo,
    right after a credential reset, with abnormal login behavior.
    """
    txn["device_id"] = f"DEV{rng.integers(10_000_000, 99_999_999)}"  # NOT persona.home_device_id
    txn["device_type"] = str(rng.choice(fd.DEVICE_TYPES[:5]))
    txn["device_fingerprint_score"] = _bump(rng, 0.02, 0.2)
    txn["ip_country"] = str(fd.sample_country(rng, bias="high_risk"))
    txn["is_cross_border"] = txn["ip_country"] != txn["billing_country"]
    txn["login_anomaly_score"] = _bump(rng, 0.7, 0.98)
    txn["password_reset_recent"] = bool(rng.random() < 0.75)
    txn["mfa_enabled"] = bool(rng.random() < 0.2)  # often disabled/bypassed
    txn["txn_velocity_1h"] = int(rng.integers(2, 9))
    txn["time_since_last_txn_sec"] = float(rng.uniform(30, 600))
    txn["amount"] = round(float(rng.lognormal(mean=5.8, sigma=0.6)), 2)
    txn["amount_vs_user_avg_ratio"] = round(txn["amount"] / max(txn["historical_avg_amount"], 10), 2)
    txn["behavioral_score"] = _bump(rng, 0.1, 0.4)
    txn["fraud_severity"] = _bump(rng, 0.65, 0.95)
    return txn


# ---------------------------------------------------------------- 3 -----
def inject_ai_phishing(rng, persona, txn):
    """
    Victim was socially engineered by an LLM-crafted message into
    self-initiating a payment (or handing over credentials used
    downstream). Signature lives mostly in the text/genai features plus
    urgency-shaped transaction timing.
    """
    txn["channel"] = str(rng.choice(["card_not_present_online", "p2p_transfer"]))
    txn["text_similarity_to_phishing_corpus"] = _bump(rng, 0.7, 0.97)
    txn["llm_generated_content_prob"] = _bump(rng, 0.75, 0.99)
    txn["num_beneficiaries_30d"] = int(rng.integers(1, 4))
    txn["beneficiary_account_age_days"] = _bump(rng, 0, 3)  # brand-new payee
    txn["is_night_time"] = bool(rng.random() < 0.35)
    txn["amount"] = round(float(rng.lognormal(mean=5.5, sigma=0.7)), 2)
    txn["structuring_score"] = _bump(rng, 0.0, 0.3)
    txn["behavioral_score"] = _bump(rng, 0.55, 0.85)  # victim behaves "normally" — hard case
    txn["fraud_severity"] = _bump(rng, 0.45, 0.75)
    return txn


# ---------------------------------------------------------------- 4 -----
def inject_voice_cloning_fraud(rng, persona, txn):
    """
    Cloned voice used against phone banking / call-center authentication
    or to socially engineer a P2P transfer ('this is your CEO/relative').
    """
    txn["channel"] = str(rng.choice(["phone_banking", "p2p_transfer"], p=[0.6, 0.4]))
    txn["voice_authenticity_score"] = _bump(rng, 0.05, 0.35)
    txn["llm_generated_content_prob"] = _bump(rng, 0.5, 0.9)  # scripted dialogue
    txn["login_anomaly_score"] = _bump(rng, 0.4, 0.8)
    txn["beneficiary_account_age_days"] = _bump(rng, 0, 5)
    txn["is_night_time"] = bool(rng.random() < 0.3)
    txn["amount"] = round(float(rng.lognormal(mean=6.5, sigma=0.55)), 2)  # urgency -> large amounts
    txn["mfa_enabled"] = bool(rng.random() < 0.4)
    txn["fraud_severity"] = _bump(rng, 0.7, 0.97)
    return txn


# ---------------------------------------------------------------- 5 -----
def inject_deepfake_identity_fraud(rng, persona, txn):
    """
    Deepfake video used to pass liveness/video-KYC during remote
    onboarding or a high-risk step-up authentication.
    """
    txn["account_creation_channel"] = "mobile_app_remote"
    txn["kyc_verification_level"] = "video_verified"  # falsely "passed"
    txn["deepfake_video_score"] = _bump(rng, 0.6, 0.95)
    txn["image_manipulation_score"] = _bump(rng, 0.5, 0.9)
    txn["document_authenticity_score"] = _bump(rng, 0.3, 0.65)
    txn["account_age_days"] = int(rng.integers(0, 30))
    txn["historical_txn_count"] = int(rng.integers(0, 4))
    txn["amount"] = round(float(rng.lognormal(mean=6.1, sigma=0.5)), 2)
    txn["fraud_severity"] = _bump(rng, 0.65, 0.92)
    return txn


# ---------------------------------------------------------------- 6 -----
def inject_synthetic_document_fraud(rng, persona, txn):
    """
    GenAI-fabricated ID / proof-of-address / bank statement submitted to
    pass a document check (onboarding, limit increase, dispute evidence).
    """
    txn["document_authenticity_score"] = _bump(rng, 0.05, 0.4)
    txn["image_manipulation_score"] = _bump(rng, 0.55, 0.95)
    txn["kyc_verification_level"] = str(rng.choice(["basic_document", "unverified"], p=[0.7, 0.3]))
    txn["llm_generated_content_prob"] = _bump(rng, 0.3, 0.7)  # fabricated statement text
    txn["account_age_days"] = int(rng.integers(0, 120))
    txn["amount"] = round(float(rng.lognormal(mean=5.9, sigma=0.55)), 2)
    txn["fraud_severity"] = _bump(rng, 0.55, 0.85)
    return txn


# ---------------------------------------------------------------- 7 -----
def inject_payment_diversion_fraud(rng, persona, txn):
    """
    Business Email Compromise style: LLM-written fake invoice / vendor
    email redirects a legitimate-looking payment to a new beneficiary.
    """
    txn["channel"] = "card_not_present_online"
    txn["payment_method"] = "bank_transfer"
    txn["text_similarity_to_phishing_corpus"] = _bump(rng, 0.6, 0.9)
    txn["llm_generated_content_prob"] = _bump(rng, 0.8, 0.99)
    txn["beneficiary_account_age_days"] = _bump(rng, 0, 2)
    txn["num_beneficiaries_30d"] = int(rng.integers(3, 12))  # vendor list churn
    txn["amount"] = round(float(rng.lognormal(mean=8.0, sigma=0.6)), 2)  # invoice-sized, large
    txn["merchant_category_code"] = "4829"
    txn["merchant_category_desc"] = "Money Transfer"
    txn["structuring_score"] = _bump(rng, 0.0, 0.2)
    txn["fraud_severity"] = _bump(rng, 0.75, 0.98)
    return txn


# ---------------------------------------------------------------- 8 -----
def inject_merchant_fraud(rng, persona, txn):
    """
    Fraud on the merchant side: bust-out merchant, fake storefront, or
    laundering via a network of small 'test' charges before a large pull.
    """
    txn["merchant_risk_score"] = _bump(rng, 0.7, 0.98)
    txn["merchant_category_code"], txn["merchant_category_desc"] = str(rng.choice(
        ["6051", "4829", "7995"])), rng.choice(["Crypto / Digital Currency", "Money Transfer", "Betting / Gambling"])
    txn["txn_velocity_1h"] = int(rng.integers(3, 15))  # burst of test charges
    is_bust_out = rng.random() < 0.3
    txn["amount"] = round(float(rng.lognormal(mean=8.3, sigma=0.4)) if is_bust_out
                           else float(rng.uniform(0.5, 5.0)), 2)
    txn["structuring_score"] = _bump(rng, 0.4, 0.9) if not is_bust_out else _bump(rng, 0.0, 0.2)
    txn["fraud_severity"] = _bump(rng, 0.6, 0.95)
    return txn


# ---------------------------------------------------------------- 9 -----
def inject_refund_fraud(rng, persona, txn):
    """
    Serial refund/chargeback abuse: goods kept, refund claimed repeatedly,
    often via a fabricated ('this arrived damaged') AI-written complaint.
    """
    txn["refund_count_30d"] = int(rng.integers(3, 14))
    txn["refund_to_purchase_ratio"] = _bump(rng, 0.85, 1.0)
    txn["text_similarity_to_phishing_corpus"] = _bump(rng, 0.1, 0.4)
    txn["llm_generated_content_prob"] = _bump(rng, 0.5, 0.9)  # AI-drafted complaint text
    txn["amount"] = round(float(rng.lognormal(mean=4.3, sigma=0.5)), 2)
    txn["merchant_category_code"] = str(rng.choice(["5999", "5691", "5732"]))
    txn["fraud_severity"] = _bump(rng, 0.35, 0.65)
    return txn


# --------------------------------------------------------------- 10 -----
def inject_coordinated_multi_account_fraud(rng, persona, txn, ring_id=None,
                                            ring_device_id=None, ring_ip_country=None):
    """
    A ring of synthetic/controlled accounts transacting in a synchronized,
    structured way (device/IP sharing, amounts kept under review
    thresholds) — classic GNN-catchable pattern.
    """
    txn["device_id"] = ring_device_id or txn["device_id"]
    txn["shared_device_n_accounts"] = int(rng.integers(4, 25))
    txn["shared_ip_n_accounts"] = int(rng.integers(4, 25))
    txn["ip_country"] = ring_ip_country or txn["ip_country"]
    txn["community_cluster_id"] = ring_id or f"CLUSTER{rng.integers(1000, 9999)}"
    txn["account_age_days"] = int(rng.integers(1, 90))
    txn["amount"] = round(float(rng.uniform(180, 480)), 2)  # kept under common reporting thresholds
    txn["structuring_score"] = _bump(rng, 0.6, 0.95)
    txn["txn_velocity_24h"] = int(rng.integers(2, 6))
    txn["fraud_severity"] = _bump(rng, 0.55, 0.9)
    return txn


INJECTORS = {
    "inject_synthetic_identity_fraud": inject_synthetic_identity_fraud,
    "inject_account_takeover": inject_account_takeover,
    "inject_ai_phishing": inject_ai_phishing,
    "inject_voice_cloning_fraud": inject_voice_cloning_fraud,
    "inject_deepfake_identity_fraud": inject_deepfake_identity_fraud,
    "inject_synthetic_document_fraud": inject_synthetic_document_fraud,
    "inject_payment_diversion_fraud": inject_payment_diversion_fraud,
    "inject_merchant_fraud": inject_merchant_fraud,
    "inject_refund_fraud": inject_refund_fraud,
    "inject_coordinated_multi_account_fraud": inject_coordinated_multi_account_fraud,
}
