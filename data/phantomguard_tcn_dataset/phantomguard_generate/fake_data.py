"""
fake_data.py
------------
Lightweight, dependency-free reference pools used to synthesize realistic
identity / merchant / device / geo attributes. Faker is not available in
this environment (no network to pip install), so we hand-roll curated
pools that are large and varied enough to avoid obvious repetition in a
dataset of tens of thousands of rows.

All pools are intentionally regionally diverse (mirrors a global card
network like Mastercard) and include both "high trust" and "high risk"
geographies so attack generators can bias sampling appropriately.
"""

import numpy as np

RNG_SEED = 42

FIRST_NAMES = [
    "Aarav", "Vivian", "Noah", "Priya", "Liam", "Sofia", "Mateo", "Aisha",
    "Kenji", "Elena", "Omar", "Grace", "Diego", "Mei", "Lucas", "Amara",
    "Ivan", "Fatima", "Ethan", "Chidi", "Anika", "Marcus", "Yuki", "Zara",
    "Nikhil", "Isabella", "Hassan", "Olivia", "Kwame", "Valentina", "Ravi",
    "Camila", "Josef", "Naledi", "Arjun", "Emma", "Tariq", "Freya", "Sven",
    "Leila",
]

LAST_NAMES = [
    "Sharma", "Johnson", "Garcia", "Kumar", "Smith", "Rossi", "Chen",
    "Okafor", "Petrov", "Silva", "Nguyen", "Muller", "Khan", "Andersson",
    "Dubois", "Patel", "Kim", "Alvarez", "Novak", "Haddad", "Yamamoto",
    "Costa", "Ivanova", "Reddy", "Cohen", "Osei", "Tanaka", "Fernandez",
    "Mbeki", "Larsen", "Rao", "Mensah", "Volkov", "Santos", "Iqbal",
]

COUNTRIES = [
    "IN", "US", "GB", "AE", "SG", "DE", "FR", "BR", "NG", "ZA", "PH",
    "ID", "MX", "CN", "RU", "VN", "PK", "BD", "EG", "TR", "CA", "AU",
    "JP", "KR", "IT", "ES", "NL", "SE", "KE", "AR",
]

# Countries with elevated baseline fraud/AML risk scoring in most card
# network risk engines (used to bias attack geographies, not a real list)
HIGH_RISK_COUNTRIES = ["NG", "RU", "PK", "BD", "VN", "EG", "ID"]
LOW_RISK_COUNTRIES = ["US", "GB", "DE", "CA", "AU", "SG", "JP", "NL", "SE"]

MERCHANT_CATEGORIES = [
    ("5411", "Grocery Stores"), ("5812", "Restaurants"),
    ("5732", "Electronics Stores"), ("5941", "Sporting Goods"),
    ("4829", "Money Transfer"), ("6051", "Crypto / Digital Currency"),
    ("5999", "Misc Retail"), ("4111", "Transit / Transportation"),
    ("5311", "Department Stores"), ("7995", "Betting / Gambling"),
    ("5964", "Direct Marketing / Catalog"), ("5967", "Direct Marketing / Adult"),
    ("4899", "Cable / Streaming Services"), ("6300", "Insurance"),
    ("5541", "Fuel / Gas Stations"), ("8299", "Educational Services"),
    ("5921", "Liquor Stores"), ("7011", "Hotels / Lodging"),
    ("4722", "Travel Agencies"), ("6011", "ATM / Cash Disbursement"),
    ("5661", "Shoe Stores"), ("5691", "Apparel"), ("8011", "Healthcare"),
    ("6012", "Financial Institutions - Merchandise/Services"),
]

HIGH_RISK_MCC = {"6051", "4829", "7995", "5967", "6011"}

CHANNELS = ["card_present", "card_not_present_online", "mobile_app",
            "digital_wallet", "p2p_transfer", "phone_banking", "atm"]

PAYMENT_METHODS = ["credit_card", "debit_card", "prepaid_card",
                    "bank_transfer", "digital_wallet", "upi"]

DEVICE_TYPES = ["ios_mobile", "android_mobile", "windows_desktop",
                 "mac_desktop", "linux_desktop", "pos_terminal", "atm_kiosk"]

KYC_LEVELS = ["unverified", "basic_document", "video_verified", "in_branch_verified"]

CREDIT_BANDS = ["thin_file", "subprime", "near_prime", "prime", "super_prime"]

ACCOUNT_CREATION_CHANNELS = ["mobile_app_remote", "web_remote", "in_branch",
                              "agent_assisted", "api_partner_onboarding"]


def _rng(seed=None):
    return np.random.default_rng(seed if seed is not None else RNG_SEED)


def sample_name(rng):
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def sample_country(rng, bias=None):
    """bias: None (uniform over all), 'high_risk', or 'low_risk'"""
    if bias == "high_risk":
        pool = HIGH_RISK_COUNTRIES
    elif bias == "low_risk":
        pool = LOW_RISK_COUNTRIES
    else:
        pool = COUNTRIES
    return rng.choice(pool)


def sample_merchant(rng, bias_high_risk_prob=0.15):
    if rng.random() < bias_high_risk_prob:
        pool = [m for m in MERCHANT_CATEGORIES if m[0] in HIGH_RISK_MCC]
    else:
        pool = MERCHANT_CATEGORIES
    idx = rng.integers(0, len(pool))
    mcc, desc = pool[idx]
    merchant_id = f"MID{rng.integers(100000, 999999)}"
    return merchant_id, mcc, desc
