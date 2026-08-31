"""
schema.py
---------
Single source of truth for the dataset's column schema. This is what
Member 3 (Defend / feature engineering) will consume directly, and what
Member 5 will render in the "data dictionary" section of the Solution
Walkthrough doc.

Each feature is tagged with:
  - dtype        : pandas/ML dtype
  - block         : which feature block it belongs to (matches the
                    "Feature Engineering (200+ features)" categories in
                    the PhantomGuard doc: transaction / user / network / genai)
  - description   : human-readable meaning
  - ml_role       : "feature" (model input) or "label" (target / metadata,
                    must be excluded from model input to avoid leakage)

FEATURE_COLUMNS / LABEL_COLUMNS give ready-made lists so training code can
do  X = df[FEATURE_COLUMNS]; y = df["is_fraud"]  without hand-picking columns.
"""

SCHEMA = {
    # ---------- identifiers (not features, not labels — keys) ----------
    "transaction_id":        {"dtype": "string",  "block": "key",         "ml_role": "key",     "description": "Unique synthetic transaction identifier."},
    "user_id":                {"dtype": "string",  "block": "key",         "ml_role": "key",     "description": "Unique synthetic account/user identifier."},

    # ---------- transaction-level block ----------
    "timestamp":               {"dtype": "datetime", "block": "transaction", "ml_role": "feature", "description": "Transaction timestamp (UTC)."},
    "amount":                  {"dtype": "float",   "block": "transaction", "ml_role": "feature", "description": "Transaction amount in USD-equivalent."},
    "currency":                {"dtype": "category","block": "transaction", "ml_role": "feature", "description": "ISO currency code."},
    "channel":                 {"dtype": "category","block": "transaction", "ml_role": "feature", "description": "Transaction channel (card_present, mobile_app, p2p_transfer, etc.)."},
    "payment_method":          {"dtype": "category","block": "transaction", "ml_role": "feature", "description": "Instrument used (credit_card, upi, bank_transfer, etc.)."},
    "merchant_id":              {"dtype": "string",  "block": "transaction", "ml_role": "feature", "description": "Synthetic merchant identifier."},
    "merchant_category_code":  {"dtype": "category","block": "transaction", "ml_role": "feature", "description": "4-digit MCC of the merchant."},
    "merchant_category_desc":  {"dtype": "category","block": "transaction", "ml_role": "feature", "description": "Human-readable merchant category."},
    "merchant_risk_score":     {"dtype": "float",   "block": "transaction", "ml_role": "feature", "description": "0-1 prior risk score of the merchant (chargeback history proxy)."},
    "device_id":                {"dtype": "string",  "block": "transaction", "ml_role": "feature", "description": "Synthetic device fingerprint identifier."},
    "device_type":             {"dtype": "category","block": "transaction", "ml_role": "feature", "description": "Device OS/form factor."},
    "device_fingerprint_score":{"dtype": "float",   "block": "transaction", "ml_role": "feature", "description": "0-1 trust score for this device on this account (1 = long-trusted device)."},
    "ip_country":               {"dtype": "category","block": "transaction", "ml_role": "feature", "description": "Country inferred from transaction IP."},
    "billing_country":         {"dtype": "category","block": "transaction", "ml_role": "feature", "description": "Country on file for the account."},
    "is_cross_border":         {"dtype": "bool",    "block": "transaction", "ml_role": "feature", "description": "True if ip_country != billing_country."},
    "hour_of_day":              {"dtype": "int",     "block": "transaction", "ml_role": "feature", "description": "0-23, local-normalized hour of transaction."},
    "is_night_time":           {"dtype": "bool",    "block": "transaction", "ml_role": "feature", "description": "True if hour_of_day in [0,5]."},
    "time_since_last_txn_sec": {"dtype": "float",   "block": "transaction", "ml_role": "feature", "description": "Seconds since this user's previous transaction."},
    "txn_velocity_1h":          {"dtype": "int",     "block": "transaction", "ml_role": "feature", "description": "Count of transactions by this user in the trailing 1 hour."},
    "txn_velocity_24h":         {"dtype": "int",     "block": "transaction", "ml_role": "feature", "description": "Count of transactions by this user in the trailing 24 hours."},
    "amount_vs_user_avg_ratio":{"dtype": "float",   "block": "transaction", "ml_role": "feature", "description": "amount / user's historical average amount."},

    # ---------- user-level block ----------
    "account_age_days":        {"dtype": "int",     "block": "user", "ml_role": "feature", "description": "Days since account creation."},
    "account_creation_channel":{"dtype": "category","block": "user", "ml_role": "feature", "description": "How the account was opened."},
    "kyc_verification_level":  {"dtype": "category","block": "user", "ml_role": "feature", "description": "Strength of identity verification on file."},
    "credit_score_band":       {"dtype": "category","block": "user", "ml_role": "feature", "description": "Coarse credit bureau band (thin_file..super_prime)."},
    "historical_avg_amount":   {"dtype": "float",   "block": "user", "ml_role": "feature", "description": "User's trailing 90-day average transaction amount."},
    "historical_txn_count":    {"dtype": "int",     "block": "user", "ml_role": "feature", "description": "User's trailing 90-day transaction count."},
    "num_devices_used":        {"dtype": "int",     "block": "user", "ml_role": "feature", "description": "Distinct devices seen on this account, lifetime."},
    "num_linked_accounts":     {"dtype": "int",     "block": "user", "ml_role": "feature", "description": "Other accounts linked to this user's payment instrument/identity."},
    "behavioral_score":        {"dtype": "float",   "block": "user", "ml_role": "feature", "description": "0-1 composite behavioral-biometrics consistency score."},
    "login_anomaly_score":     {"dtype": "float",   "block": "user", "ml_role": "feature", "description": "0-1 anomaly score for the login preceding this transaction."},
    "password_reset_recent":   {"dtype": "bool",    "block": "user", "ml_role": "feature", "description": "True if password/credentials reset within last 48h."},
    "mfa_enabled":             {"dtype": "bool",    "block": "user", "ml_role": "feature", "description": "Whether MFA is enabled on the account."},
    "prior_fraud_flags":       {"dtype": "int",     "block": "user", "ml_role": "feature", "description": "Count of previous confirmed/suspected fraud flags on this account."},

    # ---------- network-level block ----------
    "shared_device_n_accounts":{"dtype": "int",     "block": "network", "ml_role": "feature", "description": "Number of distinct accounts that have used this exact device fingerprint."},
    "shared_ip_n_accounts":    {"dtype": "int",     "block": "network", "ml_role": "feature", "description": "Number of distinct accounts transacting from this IP in the last 24h."},
    "community_cluster_id":    {"dtype": "string",  "block": "network", "ml_role": "feature", "description": "Graph-community id from the GNN's account/device/merchant graph (synthetic)."},
    "num_beneficiaries_30d":   {"dtype": "int",     "block": "network", "ml_role": "feature", "description": "Distinct payees/beneficiaries this account paid in trailing 30 days."},
    "beneficiary_account_age_days": {"dtype": "float", "block": "network", "ml_role": "feature", "description": "Age in days of the receiving account/beneficiary relationship (low = newly added payee)."},

    # ---------- GenAI-specific block ----------
    "text_similarity_to_phishing_corpus": {"dtype": "float", "block": "genai", "ml_role": "feature", "description": "0-1 similarity of associated message/email text to known phishing/BEC corpus (NaN if no text channel involved)."},
    "llm_generated_content_prob":         {"dtype": "float", "block": "genai", "ml_role": "feature", "description": "0-1 probability the associated text content was LLM-generated (NaN if not applicable)."},
    "voice_authenticity_score":           {"dtype": "float", "block": "genai", "ml_role": "feature", "description": "0-1 score from a voice-liveness/anti-spoofing model, 1 = confidently human/genuine (NaN if not a voice channel)."},
    "deepfake_video_score":               {"dtype": "float", "block": "genai", "ml_role": "feature", "description": "0-1 deepfake-likelihood score from video/liveness check during onboarding or step-up auth (NaN if not applicable)."},
    "document_authenticity_score":        {"dtype": "float", "block": "genai", "ml_role": "feature", "description": "0-1 score from document forensics model on ID/proof-of-address docs, 1 = genuine (NaN if no document submitted)."},
    "image_manipulation_score":           {"dtype": "float", "block": "genai", "ml_role": "feature", "description": "0-1 score of detected GenAI image manipulation/synthesis artifacts (NaN if not applicable)."},

    # ---------- attack-specific auxiliary features (sparse, attack-dependent) ----------
    "refund_count_30d":        {"dtype": "float", "block": "aux", "ml_role": "feature", "description": "Refund requests by this user in trailing 30 days (populated mainly for refund_fraud / merchant_fraud rows, else 0)."},
    "refund_to_purchase_ratio":{"dtype": "float", "block": "aux", "ml_role": "feature", "description": "Ratio of refunded amount to original purchase amount for this transaction (NaN if not a refund)."},
    "structuring_score":       {"dtype": "float", "block": "aux", "ml_role": "feature", "description": "0-1 score reflecting how close the amount sits just under a common reporting/review threshold."},

    # ---------- labels & metadata (exclude from model input!) ----------
    "is_fraud":                {"dtype": "int(0/1)",  "block": "label", "ml_role": "label",    "description": "Primary binary label. 1 = fraudulent/attack transaction, 0 = legitimate."},
    "attack_type":              {"dtype": "category",  "block": "label", "ml_role": "label",    "description": "Fine-grained label: one of the 10 attack_type ids, or 'legitimate'."},
    "attack_category":          {"dtype": "category",  "block": "label", "ml_role": "label",    "description": "Coarse taxonomy bucket (identity/access/social_engineering/document/merchant/network), or 'legitimate'."},
    "genai_capability":         {"dtype": "category",  "block": "label", "ml_role": "label",    "description": "GenAI capability that powers the attack (e.g. voice_synthesis), or 'none' for legitimate rows."},
    "fraud_severity":           {"dtype": "float",     "block": "label", "ml_role": "label",    "description": "0-1 soft severity/confidence score for the attack instance (0 for legitimate rows). For blended attacks this is a probabilistic combination of each component attack's severity, not just the last injector's value. Useful for cost-sensitive training or ranking review queues."},
    "is_blended_attack":        {"dtype": "bool",      "block": "label", "ml_role": "label",    "description": "True if this row was generated from more than one attack_type combined (e.g. 'ai_phishing+account_takeover'), False for single-vector or legitimate rows."},
    "ground_truth_source":      {"dtype": "category",  "block": "label", "ml_role": "metadata", "description": "Always 'synthetic' — flags this row as generator-produced, for provenance/audit."},
}

FEATURE_COLUMNS = [c for c, m in SCHEMA.items() if m["ml_role"] == "feature"]
LABEL_COLUMNS = [c for c, m in SCHEMA.items() if m["ml_role"] == "label"]
KEY_COLUMNS = [c for c, m in SCHEMA.items() if m["ml_role"] == "key"]
ALL_COLUMNS = KEY_COLUMNS + FEATURE_COLUMNS + LABEL_COLUMNS + ["ground_truth_source"]
