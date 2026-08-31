# TCN Sequence Extension — `sequence_generator.py`

## The problem this solves

The original synthetic dataset (`phantomguard_synthetic_dataset.csv`, 32,000
rows) has **31,941 unique users** — almost every user has exactly one
transaction. That's fine for XGBoost, the GNN, and the meta-classifier
(they treat rows independently), but it means the TCN branch has no real
per-user history to learn from. Padding length-1 "sequences" up to a
fixed window just teaches the TCN to ignore padding, not to recognize
genuine temporal fraud patterns.

## What this extension adds

A new, additive module — `sequence_generator.py` — that generates
**genuine time-ordered sequences per synthetic user**, with temporal
features computed *causally* (only from transactions seen so far, the
way a real production feature pipeline would compute them), covering
five patterns:

| `temporal_pattern` | What it teaches the TCN |
|---|---|
| `legitimate_sequence` | Normal spacing/amounts over time — the majority class |
| `card_testing_burst` | Many tiny transactions seconds apart (stolen-card validation) |
| `gradual_account_takeover` | Trust quietly eroding (behavioral score, login anomaly) over several transactions *before* the overt takeover signature hits |
| `bustout_escalation` | Slowly increasing amounts building a clean-looking history, then a bust-out |
| `dormant_reactivation` | A long inactivity gap (weeks/months) followed by a coordinated-fraud reactivation burst |
| `ai_phishing_escalation` | A social-engineering campaign escalating over repeated contact attempts — text_similarity_to_phishing_corpus and llm_generated_content_prob climb monotonically, small "test" payments grow into a large BEC-style diversion climax |

Five of these six patterns are signatures that **only exist across
multiple transactions** — a single-row model (XGBoost) cannot see them
even in principle, which is exactly the gap a TCN is meant to fill.
`ai_phishing_escalation` is also the only pattern that gives the TCN
temporal signal in the **GenAI-specific** feature block — none of the
other five route through phishing/text channels, so those columns sat
at 0% fill rate before this pattern was added.

## Why it reuses the existing module instead of replacing it

- Same persona generator (`persona_generator.generate_persona`)
- Same baseline transaction builder (`transaction_sim.build_baseline_transaction`)
- Same fraud injectors (`pattern_injector.inject_account_takeover`,
  `inject_synthetic_identity_fraud`, `inject_coordinated_multi_account_fraud`)
- Same output schema (`schema.ALL_COLUMNS`), plus 4 new additive columns

Nothing in `config.py`, `schema.py`, `transaction_sim.py`, or
`pattern_injector.py` was changed. This is a pure addition — the
existing 32K i.i.d. dataset and pipeline keep working exactly as before.

## New columns (additive only)

| Column | Purpose |
|---|---|
| `sequence_id` | Groups rows belonging to one synthetic user-episode |
| `seq_position` | 0-indexed order within the sequence (the TCN's time axis) |
| `seq_length` | Total length of that sequence (for building padding masks) |
| `temporal_pattern` | Which of the 5 generators produced this sequence |

`is_fraud` is still set **per transaction**, not per sequence — the
legitimate lead-in rows are labeled 0, and only the actual fraud onset
rows are labeled 1. This lets the TCN learn *when* in a sequence fraud
starts, not just "this whole sequence is bad."

## Fixing temporal features to be causally consistent

In the original per-row generator, `time_since_last_txn_sec`,
`txn_velocity_1h/24h`, `historical_avg_amount`, and
`amount_vs_user_avg_ratio` are sampled independently per row — fine for
i.i.d. training, meaningless within a sequence. This extension replaces
that with a small running-state tracker (`_RunningUserState`) that
computes each of these fields from the actual prior transactions in the
sequence, exactly as a live feature pipeline would at inference time
(it never looks at a transaction's own future).

## Quickstart

```bash
# Default: 200 sequences per pattern (1,000 total sequences)
python generate_tcn_dataset.py --out data/phantomguard_tcn_sequences.csv

# Larger run used for the delivered sample (2,500 sequences, ~49K rows)
python generate_tcn_dataset.py --n_per_pattern 500 --min_len 5 --max_len 35 \
    --out data/phantomguard_tcn_sequences.csv

# Only some patterns
python generate_tcn_dataset.py --patterns legitimate_sequence,card_testing_burst \
    --out data/tcn_subset.csv

# List available patterns
python generate_tcn_dataset.py --list
```

## Training helper: `build_tcn_windows()`

Also included: a windowing utility that turns the sequence dataset into
fixed-size, left-padded arrays ready for a TCN, without Member 3 needing
to write windowing/padding code from scratch:

```python
from sequence_generator import generate_sequence_dataset, build_tcn_windows
from schema import FEATURE_COLUMNS

df = generate_sequence_dataset(n_per_pattern=500, seq_len_range=(5, 35), seed=42)
numeric_feats = [c for c in FEATURE_COLUMNS if df[c].dtype.kind in "if"]

X, mask, y, n_real = build_tcn_windows(df, numeric_feats, window_size=10)
# X:      (num_windows, window_size, num_features)
# mask:   (num_windows, window_size)  — 1 = real step, 0 = padding
# y:      (num_windows,)              — is_fraud of the last (predicted) row
# n_real: (num_windows,)              — how many real (non-padded) steps
```

`n_real` is the important one: it's the exact signal needed for the
ensemble safeguard already discussed for this project — **only trust the
TCN branch when a window has enough real history** (e.g. `n_real >= 4`);
otherwise let XGBoost + GNN carry the decision. Feed `n_real` into the
meta-classifier as a feature (or use it to gate the TCN's contribution
directly) so the ensemble doesn't blindly trust a TCN prediction built
mostly from padding — which is exactly the failure mode flagged when the
1-transaction-per-user finding first came up.

## Recommended usage split

- **Train the TCN** on this sequence dataset (`phantomguard_tcn_sequences.csv`),
  windowed via `build_tcn_windows()`.
- **Train XGBoost / GNN / meta-classifier** on the original i.i.d. dataset
  (`phantomguard_synthetic_dataset.csv`) as already planned — that data
  is well-suited to them as-is.
- **At meta-classifier training time**, include both real single-transaction
  cases (n_real=1) and longer-history cases so it learns to weight the
  TCN's output down when history is thin — matching what real production
  traffic will actually look like (mostly short histories, occasionally deep ones).

## Fidelity check

Running the existing `fidelity_checker.py` against a 3,000-sequence run
(~59K rows, all 6 patterns) shows clean class separation with no
accidental leakage:

| Pattern | Most discriminative feature | Standardized effect size |
|---|---|---|
| `card_testing_burst` | `structuring_score` | 7.18 |
| `account_takeover` (pivot rows) | `device_fingerprint_score` | 12.36 |
| `synthetic_identity_fraud` (pivot rows) | `device_fingerprint_score` | 4.06 |
| `coordinated_multi_account_fraud` (pivot rows) | `structuring_score` | 8.94 |
| `ai_phishing` (escalation rows) | `beneficiary_account_age_days` | 2.60 |
| `payment_diversion_invoice_fraud` (climax row) | `num_beneficiaries_30d` | 3.40 |

Median transaction amount: legitimate $49.16 vs. fraud (pivot/climax rows) $134.10.

GenAI-specific columns now show non-trivial fill rates thanks to
`ai_phishing_escalation`: `text_similarity_to_phishing_corpus` at 10.5%
and `llm_generated_content_prob` at 13.4% (previously both 0% with only
the first 5 patterns). `voice_authenticity_score` and
`deepfake_video_score` remain at 0% — no pattern here routes through a
voice or video-KYC channel yet; that's the natural next pattern to add
if Member 3 wants full GenAI-channel coverage (see below).

## Extending to a new temporal pattern

1. Write a new `generate_<name>_sequence(rng, seq_len, ...)` function
   following the same shape as the five above: build a lead-in with
   `build_baseline_transaction` + `_apply_causal_features`, then a pivot
   portion using whichever existing `pattern_injector` fits (or a new one).
2. Add it to `_GENERATORS` and `TEMPORAL_PATTERNS`.
3. No other file changes needed — `generate_sequence_dataset()` and
   `generate_tcn_dataset.py` are pattern-agnostic.
