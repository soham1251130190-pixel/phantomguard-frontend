# PhantomGuard Synthetic Dataset — Data Dictionary

## Identifiers

| Column | Type | Description |
|---|---|---|
| `transaction_id` | string | Unique synthetic transaction identifier. |
| `user_id` | string | Unique synthetic account/user identifier. |

## Transaction-level features

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | Transaction timestamp (UTC). |
| `amount` | float | Transaction amount in USD-equivalent. |
| `currency` | category | ISO currency code. |
| `channel` | category | Transaction channel (card_present, mobile_app, p2p_transfer, etc.). |
| `payment_method` | category | Instrument used (credit_card, upi, bank_transfer, etc.). |
| `merchant_id` | string | Synthetic merchant identifier. |
| `merchant_category_code` | category | 4-digit MCC of the merchant. |
| `merchant_category_desc` | category | Human-readable merchant category. |
| `merchant_risk_score` | float | 0-1 prior risk score of the merchant (chargeback history proxy). |
| `device_id` | string | Synthetic device fingerprint identifier. |
| `device_type` | category | Device OS/form factor. |
| `device_fingerprint_score` | float | 0-1 trust score for this device on this account (1 = long-trusted device). |
| `ip_country` | category | Country inferred from transaction IP. |
| `billing_country` | category | Country on file for the account. |
| `is_cross_border` | bool | True if ip_country != billing_country. |
| `hour_of_day` | int | 0-23, local-normalized hour of transaction. |
| `is_night_time` | bool | True if hour_of_day in [0,5]. |
| `time_since_last_txn_sec` | float | Seconds since this user's previous transaction. |
| `txn_velocity_1h` | int | Count of transactions by this user in the trailing 1 hour. |
| `txn_velocity_24h` | int | Count of transactions by this user in the trailing 24 hours. |
| `amount_vs_user_avg_ratio` | float | amount / user's historical average amount. |

## User-level features

| Column | Type | Description |
|---|---|---|
| `account_age_days` | int | Days since account creation. |
| `account_creation_channel` | category | How the account was opened. |
| `kyc_verification_level` | category | Strength of identity verification on file. |
| `credit_score_band` | category | Coarse credit bureau band (thin_file..super_prime). |
| `historical_avg_amount` | float | User's trailing 90-day average transaction amount. |
| `historical_txn_count` | int | User's trailing 90-day transaction count. |
| `num_devices_used` | int | Distinct devices seen on this account, lifetime. |
| `num_linked_accounts` | int | Other accounts linked to this user's payment instrument/identity. |
| `behavioral_score` | float | 0-1 composite behavioral-biometrics consistency score. |
| `login_anomaly_score` | float | 0-1 anomaly score for the login preceding this transaction. |
| `password_reset_recent` | bool | True if password/credentials reset within last 48h. |
| `mfa_enabled` | bool | Whether MFA is enabled on the account. |
| `prior_fraud_flags` | int | Count of previous confirmed/suspected fraud flags on this account. |

## Network-level features

| Column | Type | Description |
|---|---|---|
| `shared_device_n_accounts` | int | Number of distinct accounts that have used this exact device fingerprint. |
| `shared_ip_n_accounts` | int | Number of distinct accounts transacting from this IP in the last 24h. |
| `community_cluster_id` | string | Graph-community id from the GNN's account/device/merchant graph (synthetic). |
| `num_beneficiaries_30d` | int | Distinct payees/beneficiaries this account paid in trailing 30 days. |
| `beneficiary_account_age_days` | float | Age in days of the receiving account/beneficiary relationship (low = newly added payee). |

## GenAI-specific features

| Column | Type | Description |
|---|---|---|
| `text_similarity_to_phishing_corpus` | float | 0-1 similarity of associated message/email text to known phishing/BEC corpus (NaN if no text channel involved). |
| `llm_generated_content_prob` | float | 0-1 probability the associated text content was LLM-generated (NaN if not applicable). |
| `voice_authenticity_score` | float | 0-1 score from a voice-liveness/anti-spoofing model, 1 = confidently human/genuine (NaN if not a voice channel). |
| `deepfake_video_score` | float | 0-1 deepfake-likelihood score from video/liveness check during onboarding or step-up auth (NaN if not applicable). |
| `document_authenticity_score` | float | 0-1 score from document forensics model on ID/proof-of-address docs, 1 = genuine (NaN if no document submitted). |
| `image_manipulation_score` | float | 0-1 score of detected GenAI image manipulation/synthesis artifacts (NaN if not applicable). |

## Attack-specific auxiliary features

| Column | Type | Description |
|---|---|---|
| `refund_count_30d` | float | Refund requests by this user in trailing 30 days (populated mainly for refund_fraud / merchant_fraud rows, else 0). |
| `refund_to_purchase_ratio` | float | Ratio of refunded amount to original purchase amount for this transaction (NaN if not a refund). |
| `structuring_score` | float | 0-1 score reflecting how close the amount sits just under a common reporting/review threshold. |

## Labels & metadata (exclude from model input, except is_fraud as target)

| Column | Type | Description |
|---|---|---|
| `is_fraud` | int(0/1) | Primary binary label. 1 = fraudulent/attack transaction, 0 = legitimate. |
| `attack_type` | category | Fine-grained label: one of the 10 attack_type ids, or 'legitimate'. |
| `attack_category` | category | Coarse taxonomy bucket (identity/access/social_engineering/document/merchant/network), or 'legitimate'. |
| `genai_capability` | category | GenAI capability that powers the attack (e.g. voice_synthesis), or 'none' for legitimate rows. |
| `fraud_severity` | float | 0-1 soft severity/confidence score for the attack instance (0 for legitimate rows). For blended attacks this is a probabilistic combination of each component attack's severity, not just the last injector's value. Useful for cost-sensitive training or ranking review queues. |
| `is_blended_attack` | bool | True if this row was generated from more than one attack_type combined (e.g. 'ai_phishing+account_takeover'), False for single-vector or legitimate rows. |
| `ground_truth_source` | category | Always 'synthetic' — flags this row as generator-produced, for provenance/audit. |

