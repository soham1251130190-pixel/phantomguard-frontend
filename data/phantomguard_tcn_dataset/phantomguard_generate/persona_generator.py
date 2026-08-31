"""
persona_generator.py
---------------------
Generates a synthetic *user* (persona): the stable identity + behavioral
profile that a stream of transactions is then generated against. Kept
separate from transaction_sim.py so it can be reused / imported by
Member 1's attack_database (personas can be attached to attack
hypotheses) without pulling in the transaction engine.

A persona is deliberately "thin" — it holds only what a payments network
would plausibly know about an account, not full PII (no synthetic SSNs /
card numbers are generated; this is fraud *pattern* data, not fake real
credentials).
"""

from dataclasses import dataclass, field
import numpy as np

import fake_data as fd


@dataclass
class Persona:
    user_id: str
    name: str
    account_age_days: int
    account_creation_channel: str
    kyc_verification_level: str
    credit_score_band: str
    billing_country: str
    historical_avg_amount: float
    historical_txn_count: int
    num_devices_used: int
    num_linked_accounts: int
    behavioral_score: float
    mfa_enabled: bool
    prior_fraud_flags: int
    home_device_id: str
    home_device_type: str


def generate_persona(rng: np.random.Generator, persona_id: int,
                      profile: str = "normal") -> Persona:
    """
    profile: 'normal'            -> typical legitimate long-standing user
             'new_thin'          -> brand-new, thin-file account (synth-ID risk)
             'established_trust' -> older, higher-trust account (ATO target)
             'mule_or_victim'    -> mid-tenure, variable trust (phishing/BEC victim or mule)
    """
    user_id = f"U{persona_id:07d}"
    name = fd.sample_name(rng)

    if profile == "new_thin":
        account_age_days = int(rng.integers(0, 45))
        kyc = rng.choice(fd.KYC_LEVELS[:2], p=[0.55, 0.45])
        credit_band = "thin_file"
        hist_count = int(rng.integers(0, 5))
        hist_avg = float(rng.uniform(15, 80))
        devices = 1
        behavioral = float(rng.uniform(0.3, 0.6))
    elif profile == "established_trust":
        account_age_days = int(rng.integers(400, 3000))
        kyc = rng.choice(fd.KYC_LEVELS[1:], p=[0.5, 0.3, 0.2])
        credit_band = rng.choice(["near_prime", "prime", "super_prime"], p=[0.3, 0.45, 0.25])
        hist_count = int(rng.integers(80, 1200))
        hist_avg = float(rng.lognormal(mean=4.0, sigma=0.6))
        devices = int(rng.integers(1, 4))
        behavioral = float(rng.uniform(0.75, 0.98))
    elif profile == "mule_or_victim":
        account_age_days = int(rng.integers(60, 900))
        kyc = rng.choice(fd.KYC_LEVELS)
        credit_band = rng.choice(fd.CREDIT_BANDS, p=[0.15, 0.25, 0.3, 0.2, 0.1])
        hist_count = int(rng.integers(5, 300))
        hist_avg = float(rng.lognormal(mean=3.6, sigma=0.7))
        devices = int(rng.integers(1, 3))
        behavioral = float(rng.uniform(0.5, 0.85))
    else:  # normal
        account_age_days = int(rng.integers(90, 2500))
        kyc = rng.choice(fd.KYC_LEVELS, p=[0.05, 0.35, 0.35, 0.25])
        credit_band = rng.choice(fd.CREDIT_BANDS, p=[0.05, 0.15, 0.3, 0.35, 0.15])
        hist_count = int(rng.integers(10, 900))
        hist_avg = float(rng.lognormal(mean=3.8, sigma=0.65))
        devices = int(rng.integers(1, 4))
        behavioral = float(rng.uniform(0.7, 0.99))

    device_type = rng.choice(fd.DEVICE_TYPES[:5])
    device_id = f"DEV{rng.integers(10_000_000, 99_999_999)}"

    return Persona(
        user_id=user_id,
        name=name,
        account_age_days=account_age_days,
        account_creation_channel=str(rng.choice(fd.ACCOUNT_CREATION_CHANNELS)),
        kyc_verification_level=str(kyc),
        credit_score_band=str(credit_band),
        billing_country=str(fd.sample_country(rng, bias="low_risk" if profile != "new_thin" else None)),
        historical_avg_amount=round(hist_avg, 2),
        historical_txn_count=hist_count,
        num_devices_used=devices,
        num_linked_accounts=int(max(0, rng.poisson(0.3))),
        behavioral_score=round(behavioral, 3),
        mfa_enabled=bool(rng.random() < (0.75 if profile == "established_trust" else 0.5)),
        prior_fraud_flags=int(rng.poisson(0.05)),
        home_device_id=device_id,
        home_device_type=str(device_type),
    )
