"""
config.py
---------
Central registry of attack types. This is THE extension point of the
whole generator: adding attack vector #11, #12, ... #47+ later means
adding one entry here (+ one injector function in pattern_injector.py)
and NOTHING else changes. transaction_sim.py never hard-codes attack
names — it always looks them up in ATTACK_REGISTRY.

Each entry declares:
  id                 -> stable snake_case key used everywhere (labels, files)
  display_name       -> human-readable name (for UI / docs)
  category           -> broad bucket (identity, access, social_engineering,
                         document, network, merchant) — useful as a coarse
                         label for the classifier / for the taxonomy pillar
  genai_capability    -> which GenAI capability powers this attack (maps to
                         the "GenAI-specific" feature block in schema.py)
  default_channel_bias-> which transaction channels this attack skews toward
  severity_weight     -> relative typical impact (used to sample amount /
                         priority when generating attack hypotheses)
  injector            -> name of the function in pattern_injector.py that
                         mutates a base transaction dict into this attack's
                         signature pattern
"""

ATTACK_REGISTRY = {
    "synthetic_identity_fraud": {
        "id": "synthetic_identity_fraud",
        "display_name": "Synthetic Identity Fraud",
        "category": "identity",
        "genai_capability": "text_generation",
        "default_channel_bias": ["card_not_present_online", "mobile_app", "digital_wallet"],
        "severity_weight": 0.8,
        "injector": "inject_synthetic_identity_fraud",
    },
    "account_takeover": {
        "id": "account_takeover",
        "display_name": "Account Takeover (ATO)",
        "category": "access",
        "genai_capability": "credential_stuffing_automation",
        "default_channel_bias": ["mobile_app", "card_not_present_online", "digital_wallet"],
        "severity_weight": 0.85,
        "injector": "inject_account_takeover",
    },
    "ai_phishing": {
        "id": "ai_phishing",
        "display_name": "AI-Powered Phishing",
        "category": "social_engineering",
        "genai_capability": "text_generation",
        "default_channel_bias": ["card_not_present_online", "p2p_transfer"],
        "severity_weight": 0.65,
        "injector": "inject_ai_phishing",
    },
    "voice_cloning_fraud": {
        "id": "voice_cloning_fraud",
        "display_name": "AI Voice-Cloning Fraud",
        "category": "social_engineering",
        "genai_capability": "voice_synthesis",
        "default_channel_bias": ["phone_banking", "p2p_transfer"],
        "severity_weight": 0.9,
        "injector": "inject_voice_cloning_fraud",
    },
    "deepfake_identity_fraud": {
        "id": "deepfake_identity_fraud",
        "display_name": "Deepfake Identity Fraud",
        "category": "identity",
        "genai_capability": "video_synthesis",
        "default_channel_bias": ["mobile_app", "card_not_present_online"],
        "severity_weight": 0.85,
        "injector": "inject_deepfake_identity_fraud",
    },
    "synthetic_document_fraud": {
        "id": "synthetic_document_fraud",
        "display_name": "Synthetic Document Fraud",
        "category": "document",
        "genai_capability": "image_generation",
        "default_channel_bias": ["mobile_app", "card_not_present_online"],
        "severity_weight": 0.7,
        "injector": "inject_synthetic_document_fraud",
    },
    "payment_diversion_invoice_fraud": {
        "id": "payment_diversion_invoice_fraud",
        "display_name": "Payment Diversion / Invoice Fraud (BEC)",
        "category": "social_engineering",
        "genai_capability": "text_generation",
        "default_channel_bias": ["bank_transfer_b2b", "card_not_present_online"],
        "severity_weight": 0.95,
        "injector": "inject_payment_diversion_fraud",
    },
    "merchant_fraud": {
        "id": "merchant_fraud",
        "display_name": "Merchant Fraud",
        "category": "merchant",
        "genai_capability": "text_generation",
        "default_channel_bias": ["card_present", "card_not_present_online"],
        "severity_weight": 0.75,
        "injector": "inject_merchant_fraud",
    },
    "refund_fraud": {
        "id": "refund_fraud",
        "display_name": "Refund Fraud",
        "category": "merchant",
        "genai_capability": "text_generation",
        "default_channel_bias": ["card_not_present_online", "mobile_app"],
        "severity_weight": 0.5,
        "injector": "inject_refund_fraud",
    },
    "coordinated_multi_account_fraud": {
        "id": "coordinated_multi_account_fraud",
        "display_name": "Coordinated Multi-Account Fraud",
        "category": "network",
        "genai_capability": "identity_generation_at_scale",
        "default_channel_bias": ["mobile_app", "digital_wallet", "p2p_transfer"],
        "severity_weight": 0.8,
        "injector": "inject_coordinated_multi_account_fraud",
    },
}

ATTACK_TYPES = list(ATTACK_REGISTRY.keys())


def get_attack_config(attack_type: str) -> dict:
    if attack_type not in ATTACK_REGISTRY:
        raise ValueError(
            f"Unknown attack_type '{attack_type}'. "
            f"Valid options: {ATTACK_TYPES}. "
            f"To add a new attack vector, register it in config.ATTACK_REGISTRY "
            f"and add a matching injector in pattern_injector.py."
        )
    return ATTACK_REGISTRY[attack_type]
