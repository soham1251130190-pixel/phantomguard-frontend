"""
sequence_generator.py
----------------------
Extension of the `generate/` module for Member 3's TCN branch.

WHY THIS FILE EXISTS
---------------------
`transaction_sim.generate_transactions()` draws a *brand-new random
persona for every row*. That is correct and sufficient for XGBoost / GNN
/ meta-classifier training (i.i.d. rows are fine there), but it means
the main dataset has ~1 transaction per user_id — there is no real
"same user over time" thread for a Temporal Convolutional Network to
learn from. Padding length-1 sequences up to a TCN's input window just
teaches it to ignore padding, not to recognize temporal fraud patterns
(card-testing bursts, gradual account-takeover drift, escalating
bust-outs, dormant-then-reactivate rings).

This module fixes that at the source: it keeps ONE persona fixed and
generates a genuine, time-ordered sequence of transactions for them,
with temporal features (time_since_last_txn_sec, txn_velocity_1h/24h,
historical_avg_amount, historical_txn_count, amount_vs_user_avg_ratio)
computed causally from the sequence itself instead of sampled
independently per row.

It re-uses everything from the existing module — `generate_persona`,
`build_baseline_transaction`, and the same attack injectors in
`pattern_injector.py` — so a sequence's fraud rows carry the exact same
per-attack feature signatures Member 3 already trains XGBoost/GNN on.
Nothing in transaction_sim.py, pattern_injector.py, or config.py needs
to change.

OUTPUT SCHEMA
-------------
Same columns as `schema.ALL_COLUMNS`, plus four additive columns
(appended, never overwriting existing ones — see SEQUENCE_EXTRA_COLUMNS
below) so this file's output can be concatenated underneath the
existing i.i.d. dataset without breaking `schema.FEATURE_COLUMNS`-based
code Member 3 already wrote:

    sequence_id        -> groups the rows belonging to one synthetic
                           user-episode (NOT the same as user_id when a
                           dormant/reactivation gap resets identity-ish
                           behavior — see temporal_pattern docstrings)
    seq_position        -> 0-indexed order of this transaction within
                           its sequence (what a TCN's positional axis is)
    seq_length          -> total length of the sequence this row
                           belongs to (needed to build the padding mask)
    temporal_pattern    -> which temporal generator produced this
                           sequence: 'legitimate_sequence',
                           'card_testing_burst', 'gradual_account_takeover',
                           'bustout_escalation', or 'dormant_reactivation'

`is_fraud` is still set PER TRANSACTION (0 for the legitimate lead-in
rows, 1 only once the pattern actually turns fraudulent), so the TCN
learns *when in a sequence* fraud onset happens, not just "this whole
sequence is bad".
"""

from datetime import timedelta
import numpy as np
import pandas as pd

import fake_data as fd
from persona_generator import generate_persona
from transaction_sim import build_baseline_transaction, BASE_DATE
import pattern_injector as pi
from schema import ALL_COLUMNS

SEQUENCE_EXTRA_COLUMNS = ["sequence_id", "seq_position", "seq_length", "temporal_pattern"]
SEQUENCE_ALL_COLUMNS = ALL_COLUMNS + SEQUENCE_EXTRA_COLUMNS

TEMPORAL_PATTERNS = [
    "legitimate_sequence",
    "card_testing_burst",
    "gradual_account_takeover",
    "bustout_escalation",
    "dormant_reactivation",
    "ai_phishing_escalation",
]


# --------------------------------------------------------------------- #
# Causal (sequence-consistent) temporal feature bookkeeping
# --------------------------------------------------------------------- #

class _RunningUserState:
    """
    Tracks what a real feature pipeline would know ONLY from transactions
    seen so far for this user — mirrors how these same columns are
    computed at inference time in a production system, where you never
    get to peek at a transaction's own future.
    """

    def __init__(self, persona):
        self.avg_amount = persona.historical_avg_amount
        self.txn_count = persona.historical_txn_count
        self.last_ts = None
        self.ts_in_last_hour = []
        self.ts_in_last_24h = []

    def snapshot_before(self, ts):
        """Feature values as they'd be computed just BEFORE this txn."""
        if self.last_ts is None:
            time_since_last = float(np.random.default_rng().exponential(3600 * 12))
        else:
            time_since_last = max(0.0, (ts - self.last_ts).total_seconds())

        v1h = sum(1 for t in self.ts_in_last_hour if (ts - t).total_seconds() <= 3600)
        v24h = sum(1 for t in self.ts_in_last_24h if (ts - t).total_seconds() <= 86400)
        return {
            "time_since_last_txn_sec": round(time_since_last, 1),
            "txn_velocity_1h": v1h,
            "txn_velocity_24h": v24h,
            "historical_avg_amount": round(self.avg_amount, 2),
            "historical_txn_count": self.txn_count,
        }

    def update_after(self, ts, amount):
        """Roll the running stats forward after this txn is generated."""
        n = self.txn_count
        self.avg_amount = (self.avg_amount * n + amount) / (n + 1)
        self.txn_count = n + 1
        self.ts_in_last_hour = [t for t in self.ts_in_last_hour if (ts - t).total_seconds() <= 3600] + [ts]
        self.ts_in_last_24h = [t for t in self.ts_in_last_24h if (ts - t).total_seconds() <= 86400] + [ts]
        self.last_ts = ts


def _apply_causal_features(txn, state, ts):
    """Overwrite the random per-row temporal fields with causal ones."""
    snap = state.snapshot_before(ts)
    txn.update(snap)
    txn["amount_vs_user_avg_ratio"] = round(txn["amount"] / max(snap["historical_avg_amount"], 5), 2)
    txn["timestamp"] = ts
    state.update_after(ts, txn["amount"])
    return txn


def _finalize_row(txn, rng, is_fraud, attack_type, attack_category, genai_capability, fraud_severity):
    txn["transaction_id"] = f"TXN{rng.integers(10**11, 10**12)}"
    txn.update({
        "is_fraud": int(is_fraud),
        "attack_type": attack_type,
        "attack_category": attack_category,
        "genai_capability": genai_capability,
        "fraud_severity": round(float(fraud_severity), 3),
        "is_blended_attack": False,
        "ground_truth_source": "synthetic",
    })
    return txn


# --------------------------------------------------------------------- #
# Pattern 1 — legitimate sequence (the majority class the TCN also needs)
# --------------------------------------------------------------------- #

def generate_legitimate_sequence(rng, seq_len, persona=None):
    """A normal user transacting normally over time. No fraud onset."""
    persona = persona or generate_persona(rng, persona_id=int(rng.integers(1, 10_000_000)), profile="normal")
    state = _RunningUserState(persona)
    ts = BASE_DATE + timedelta(days=int(rng.integers(0, 30)), hours=int(rng.integers(0, 24)))
    rows = []
    for i in range(seq_len):
        gap_hours = float(rng.gamma(shape=2.0, scale=10.0))  # typical multi-day-ish gaps, right-skewed
        ts = ts + timedelta(hours=gap_hours)
        txn = build_baseline_transaction(rng, persona)
        txn = _apply_causal_features(txn, state, ts)
        txn = _finalize_row(txn, rng, is_fraud=0, attack_type="legitimate",
                             attack_category="legitimate", genai_capability="none", fraud_severity=0.0)
        rows.append(txn)
    return rows


# --------------------------------------------------------------------- #
# Pattern 2 — card-testing burst
# Many tiny transactions seconds/minutes apart (validating a stolen
# card), then either the burst stops (declined/abandoned) or one larger
# transaction succeeds. This is a PURELY temporal signature — no single
# row looks obviously fraudulent, only the inter-arrival pattern does.
# --------------------------------------------------------------------- #

def generate_card_testing_burst_sequence(rng, seq_len, lead_in_frac=0.5):
    persona = generate_persona(rng, persona_id=int(rng.integers(1, 10_000_000)), profile="normal")
    state = _RunningUserState(persona)
    ts = BASE_DATE + timedelta(days=int(rng.integers(0, 30)), hours=int(rng.integers(0, 24)))
    rows = []

    lead_in = max(1, int(seq_len * lead_in_frac))
    burst_len = seq_len - lead_in

    for i in range(lead_in):
        gap_hours = float(rng.gamma(shape=2.0, scale=10.0))
        ts = ts + timedelta(hours=gap_hours)
        txn = build_baseline_transaction(rng, persona)
        txn = _apply_causal_features(txn, state, ts)
        txn = _finalize_row(txn, rng, 0, "legitimate", "legitimate", "none", 0.0)
        rows.append(txn)

    for j in range(burst_len):
        ts = ts + timedelta(seconds=float(rng.uniform(5, 90)))  # seconds apart, not hours
        txn = build_baseline_transaction(rng, persona)
        txn["amount"] = round(float(rng.uniform(0.5, 4.0)), 2)  # small test charges
        txn["merchant_category_code"] = str(rng.choice(["6051", "5999", "5732"]))
        txn = _apply_causal_features(txn, state, ts)
        txn["structuring_score"] = round(float(rng.uniform(0.5, 0.9)), 3)
        severity = min(0.95, 0.4 + 0.08 * j)  # confidence in the label rises deeper into the burst
        txn = _finalize_row(txn, rng, 1, "card_testing_burst", "access", "none", severity)
        rows.append(txn)

    return rows


# --------------------------------------------------------------------- #
# Pattern 3 — gradual account takeover
# A previously normal, established sequence that DRIFTS: behavioral
# score slides down, then device/geo change, culminating in the
# existing inject_account_takeover() signature for the final rows.
# Teaches the TCN "trust eroding over several transactions", which a
# single-row model literally cannot see.
# --------------------------------------------------------------------- #

def generate_gradual_ato_sequence(rng, seq_len, pivot_frac=0.7):
    persona = generate_persona(rng, persona_id=int(rng.integers(1, 10_000_000)), profile="established_trust")
    state = _RunningUserState(persona)
    ts = BASE_DATE + timedelta(days=int(rng.integers(0, 30)), hours=int(rng.integers(0, 24)))
    rows = []

    pivot = max(1, int(seq_len * pivot_frac))
    drift_zone = max(1, pivot // 3)  # last few "legit" rows show early erosion

    for i in range(pivot):
        gap_hours = float(rng.gamma(shape=2.0, scale=14.0))
        ts = ts + timedelta(hours=gap_hours)
        txn = build_baseline_transaction(rng, persona)
        if i >= pivot - drift_zone:
            # early erosion: behavioral/device trust quietly sliding before the overt takeover
            decay = (i - (pivot - drift_zone) + 1) / drift_zone
            txn["behavioral_score"] = round(persona.behavioral_score * (1 - 0.35 * decay), 3)
            txn["login_anomaly_score"] = round(float(rng.uniform(0.15, 0.4)) * decay, 3)
        txn = _apply_causal_features(txn, state, ts)
        txn = _finalize_row(txn, rng, 0, "legitimate", "legitimate", "none", 0.0)
        rows.append(txn)

    for k in range(seq_len - pivot):
        ts = ts + timedelta(minutes=float(rng.uniform(2, 40)))  # rapid follow-on activity after takeover
        txn = build_baseline_transaction(rng, persona)
        txn = pi.inject_account_takeover(rng, persona, txn)
        txn = _apply_causal_features(txn, state, ts)
        severity = txn.get("fraud_severity", 0.8)
        txn = _finalize_row(txn, rng, 1, "account_takeover", "access",
                             "credential_stuffing_automation", severity)
        rows.append(txn)

    return rows


# --------------------------------------------------------------------- #
# Pattern 4 — bust-out escalation
# A thin-file account behaves normally with slowly increasing amounts
# to build trust/limit, then a final bust-out transaction. Reuses
# inject_synthetic_identity_fraud() for the pivot row.
# --------------------------------------------------------------------- #

def generate_bustout_escalation_sequence(rng, seq_len, pivot_frac=0.85):
    persona = generate_persona(rng, persona_id=int(rng.integers(1, 10_000_000)), profile="new_thin")
    state = _RunningUserState(persona)
    ts = BASE_DATE + timedelta(days=int(rng.integers(0, 30)), hours=int(rng.integers(0, 24)))
    rows = []

    pivot = max(1, int(seq_len * pivot_frac))
    base_amount = float(rng.uniform(15, 40))

    for i in range(pivot):
        gap_hours = float(rng.gamma(shape=1.8, scale=18.0))
        ts = ts + timedelta(hours=gap_hours)
        txn = build_baseline_transaction(rng, persona)
        # amounts creep up ~10-20% per transaction — building a clean-looking limit history
        txn["amount"] = round(base_amount * (1.12 ** i) * float(rng.uniform(0.9, 1.1)), 2)
        txn = _apply_causal_features(txn, state, ts)
        txn = _finalize_row(txn, rng, 0, "legitimate", "legitimate", "none", 0.0)
        rows.append(txn)

    for k in range(seq_len - pivot):
        ts = ts + timedelta(hours=float(rng.uniform(1, 6)))
        txn = build_baseline_transaction(rng, persona)
        txn = pi.inject_synthetic_identity_fraud(rng, persona, txn)
        txn = _apply_causal_features(txn, state, ts)
        severity = txn.get("fraud_severity", 0.85)
        txn = _finalize_row(txn, rng, 1, "synthetic_identity_fraud", "identity",
                             "text_generation", severity)
        rows.append(txn)

    return rows


# --------------------------------------------------------------------- #
# Pattern 5 — dormant then reactivation
# A ring-style account (or a stolen dormant account) goes quiet for a
# long stretch, then reactivates with coordinated-fraud-style behavior.
# The LONG GAP itself is the temporal signal a single-row model cannot
# represent at all — this pattern exists specifically to give the TCN
# something to learn from time_since_last_txn_sec's tail.
# --------------------------------------------------------------------- #

def generate_dormant_reactivation_sequence(rng, seq_len, dormant_days_range=(45, 180)):
    persona = generate_persona(rng, persona_id=int(rng.integers(1, 10_000_000)), profile="normal")
    state = _RunningUserState(persona)
    ts = BASE_DATE + timedelta(days=int(rng.integers(0, 30)), hours=int(rng.integers(0, 24)))
    rows = []

    lead_in = max(1, seq_len // 2)
    for i in range(lead_in):
        gap_hours = float(rng.gamma(shape=2.0, scale=10.0))
        ts = ts + timedelta(hours=gap_hours)
        txn = build_baseline_transaction(rng, persona)
        txn = _apply_causal_features(txn, state, ts)
        txn = _finalize_row(txn, rng, 0, "legitimate", "legitimate", "none", 0.0)
        rows.append(txn)

    dormant_days = float(rng.uniform(*dormant_days_range))
    ts = ts + timedelta(days=dormant_days)

    ring_ctx = dict(
        device=f"DEV{rng.integers(10_000_000, 99_999_999)}",
        ip=str(fd.sample_country(rng, bias="high_risk")),
        cid=f"CLUSTER{rng.integers(1000, 9999)}",
    )
    for k in range(seq_len - lead_in):
        ts = ts + timedelta(minutes=float(rng.uniform(3, 25)))  # rapid reactivation burst
        txn = build_baseline_transaction(rng, persona)
        txn = pi.inject_coordinated_multi_account_fraud(
            rng, persona, txn, ring_id=ring_ctx["cid"],
            ring_device_id=ring_ctx["device"], ring_ip_country=ring_ctx["ip"],
        )
        txn = _apply_causal_features(txn, state, ts)
        severity = txn.get("fraud_severity", 0.75)
        txn = _finalize_row(txn, rng, 1, "coordinated_multi_account_fraud", "network",
                             "identity_generation_at_scale", severity)
        rows.append(txn)

    return rows


# --------------------------------------------------------------------- #
# Pattern 6 — escalating AI-phishing campaign
# A social-engineering campaign rarely succeeds on the first message.
# The attacker (or an LLM agent automating the attacker) makes repeated
# contact attempts, each more convincing than the last — rapport-building
# messages, a small "test" payment, then a large diversion once trust is
# established. This is the pattern that finally gives the TCN temporal
# signal in the GenAI-specific feature block (text_similarity_to_
# phishing_corpus, llm_generated_content_prob), which none of the other
# five patterns route through.
# --------------------------------------------------------------------- #

def generate_ai_phishing_escalation_sequence(rng, seq_len, lead_in_frac=0.4):
    persona = generate_persona(rng, persona_id=int(rng.integers(1, 10_000_000)), profile="mule_or_victim")
    state = _RunningUserState(persona)
    ts = BASE_DATE + timedelta(days=int(rng.integers(0, 30)), hours=int(rng.integers(0, 24)))
    rows = []

    lead_in = max(1, int(seq_len * lead_in_frac))
    campaign_len = max(2, seq_len - lead_in)  # need at least a "test" + a "climax" row

    # ---- lead-in: victim transacting normally, campaign hasn't started ----
    for i in range(lead_in):
        gap_hours = float(rng.gamma(shape=2.0, scale=12.0))
        ts = ts + timedelta(hours=gap_hours)
        txn = build_baseline_transaction(rng, persona)
        txn = _apply_causal_features(txn, state, ts)
        txn = _finalize_row(txn, rng, 0, "legitimate", "legitimate", "none", 0.0)
        rows.append(txn)

    # ---- escalating contact attempts: each more convincing than the last ----
    for j in range(campaign_len - 1):
        # attacker's messages get more targeted/convincing over the campaign,
        # measured by rising text_similarity + llm_generated_content_prob
        progress = (j + 1) / campaign_len
        ts = ts + timedelta(hours=float(rng.uniform(3, 30)))  # follow-up contact within days
        txn = build_baseline_transaction(rng, persona)
        txn = pi.inject_ai_phishing(rng, persona, txn)
        txn["text_similarity_to_phishing_corpus"] = round(min(0.97, 0.4 + 0.5 * progress
                                                                + float(rng.uniform(-0.05, 0.05))), 3)
        txn["llm_generated_content_prob"] = round(min(0.99, 0.5 + 0.45 * progress
                                                        + float(rng.uniform(-0.05, 0.05))), 3)
        # small "test the water" payment first, growing toward the real ask
        txn["amount"] = round(float(rng.uniform(5, 50)) * (1 + 3 * progress), 2)
        txn = _apply_causal_features(txn, state, ts)
        severity = round(min(0.9, 0.35 + 0.5 * progress), 3)
        txn = _finalize_row(txn, rng, 1, "ai_phishing", "social_engineering", "text_generation", severity)
        rows.append(txn)

    # ---- climax: victim fully convinced, large BEC-style diversion payment ----
    ts = ts + timedelta(hours=float(rng.uniform(1, 12)))
    txn = build_baseline_transaction(rng, persona)
    txn = pi.inject_payment_diversion_fraud(rng, persona, txn)
    txn["text_similarity_to_phishing_corpus"] = round(float(rng.uniform(0.85, 0.98)), 3)
    txn["llm_generated_content_prob"] = round(float(rng.uniform(0.9, 0.99)), 3)
    txn = _apply_causal_features(txn, state, ts)
    severity = txn.get("fraud_severity", 0.9)
    txn = _finalize_row(txn, rng, 1, "payment_diversion_invoice_fraud", "social_engineering",
                         "text_generation", severity)
    rows.append(txn)

    return rows


_GENERATORS = {
    "legitimate_sequence": generate_legitimate_sequence,
    "card_testing_burst": generate_card_testing_burst_sequence,
    "gradual_account_takeover": generate_gradual_ato_sequence,
    "bustout_escalation": generate_bustout_escalation_sequence,
    "dormant_reactivation": generate_dormant_reactivation_sequence,
    "ai_phishing_escalation": generate_ai_phishing_escalation_sequence,
}


def _rows_to_labeled_df(rows, temporal_pattern, sequence_id):
    for pos, r in enumerate(rows):
        r["sequence_id"] = sequence_id
        r["seq_position"] = pos
        r["seq_length"] = len(rows)
        r["temporal_pattern"] = temporal_pattern
    return rows


def generate_sequence_dataset(n_per_pattern=200, seq_len_range=(6, 25), seed=42,
                               patterns=None):
    """
    Top-level entry point — the sequence-data equivalent of
    transaction_sim.generate_balanced_dataset().

    n_per_pattern : number of independent sequences to generate for EACH
                    temporal_pattern (including 'legitimate_sequence').
    seq_len_range : (min, max) transactions per sequence, sampled per
                    sequence — deliberately variable length, since a
                    real TCN pipeline must handle variable-length input
                    (that's what seq_length / padding masks are for).
    patterns      : subset of TEMPORAL_PATTERNS to generate; defaults to
                    all 5.

    Returns a DataFrame with columns == SEQUENCE_ALL_COLUMNS, sorted by
    (sequence_id, seq_position) — i.e. already in the order a TCN
    windowing function should consume it.
    """
    rng = np.random.default_rng(seed)
    patterns = patterns or TEMPORAL_PATTERNS
    all_rows = []
    seq_counter = 0

    for pattern in patterns:
        gen_fn = _GENERATORS[pattern]
        for _ in range(n_per_pattern):
            seq_len = int(rng.integers(seq_len_range[0], seq_len_range[1] + 1))
            rows = gen_fn(rng, seq_len)
            sequence_id = f"SEQ{seq_counter:07d}"
            seq_counter += 1
            all_rows.extend(_rows_to_labeled_df(rows, pattern, sequence_id))

    df = pd.DataFrame(all_rows)[SEQUENCE_ALL_COLUMNS]
    return df.sort_values(["sequence_id", "seq_position"]).reset_index(drop=True)


# --------------------------------------------------------------------- #
# Windowing helper — directly usable by Member 3's TCN training code
# --------------------------------------------------------------------- #

def build_tcn_windows(df, feature_columns, window_size=10, min_history=3):
    """
    Groups df by sequence_id (already time-ordered by seq_position) and
    produces fixed-size, left-padded windows + a padding mask + the
    label of the LAST transaction in each window — the standard
    "predict this transaction given its preceding history" framing.

    Returns:
      X        : float32 array, shape (num_windows, window_size, num_features)
      mask     : float32 array, shape (num_windows, window_size) — 1 = real, 0 = padding
      y        : int array, shape (num_windows,) — is_fraud of the LAST row in the window
      n_real   : int array, shape (num_windows,) — how many real (non-padded) steps
                 are in this window; use this to implement the
                 "only trust the TCN branch when n_real >= min_history"
                 safeguard discussed for the meta-classifier.

    min_history is NOT applied as a filter here (all windows are
    returned) — it's surfaced via n_real so Member 3's meta-classifier
    can decide per-window how much to weight the TCN's output, exactly
    as flagged as a required safeguard for this dataset's short
    real-world sequences.
    """
    X_list, mask_list, y_list, n_real_list = [], [], [], []

    for _, g in df.sort_values(["sequence_id", "seq_position"]).groupby("sequence_id"):
        feats = g[feature_columns].to_numpy(dtype="float32", na_value=0.0)
        labels = g["is_fraud"].to_numpy()
        n = len(g)
        for end in range(n):
            start = max(0, end - window_size + 1)
            window = feats[start:end + 1]
            real_len = window.shape[0]
            pad_len = window_size - real_len
            if pad_len > 0:
                pad = np.zeros((pad_len, feats.shape[1]), dtype="float32")
                window = np.vstack([pad, window])
            mask = np.array([0.0] * pad_len + [1.0] * real_len, dtype="float32")

            X_list.append(window)
            mask_list.append(mask)
            y_list.append(labels[end])
            n_real_list.append(real_len)

    return (
        np.stack(X_list),
        np.stack(mask_list),
        np.array(y_list),
        np.array(n_real_list),
    )
