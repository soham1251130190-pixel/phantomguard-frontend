"""
generate_tcn_dataset.py
-------------------------
CLI entry point for the TCN-focused sequence dataset (sequence_generator.py).

Usage
-----
# Default: 200 sequences per pattern (5 patterns), lengths 6-25
python generate_tcn_dataset.py --out data/phantomguard_tcn_sequences.csv

# Larger run, only 3 of the 5 temporal patterns
python generate_tcn_dataset.py --n_per_pattern 500 \
    --patterns legitimate_sequence,card_testing_burst,gradual_account_takeover \
    --out data/tcn_subset.csv

# List available temporal patterns
python generate_tcn_dataset.py --list
"""

import argparse
import os
import sys

from sequence_generator import generate_sequence_dataset, TEMPORAL_PATTERNS


def main():
    parser = argparse.ArgumentParser(description="PhantomGuard TCN sequence dataset generator")
    parser.add_argument("--n_per_pattern", type=int, default=200,
                         help="Number of independent sequences per temporal_pattern")
    parser.add_argument("--min_len", type=int, default=6)
    parser.add_argument("--max_len", type=int, default=25)
    parser.add_argument("--patterns", type=str, default=None,
                         help="Comma-separated subset of temporal patterns (default: all 5)")
    parser.add_argument("--out", type=str, default="data/phantomguard_tcn_sequences.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("Available temporal_pattern values:")
        for p in TEMPORAL_PATTERNS:
            print(f"  {p}")
        sys.exit(0)

    patterns = args.patterns.split(",") if args.patterns else None
    if patterns:
        for p in patterns:
            if p not in TEMPORAL_PATTERNS:
                parser.error(f"Unknown pattern '{p}'. Valid options: {TEMPORAL_PATTERNS}")

    df = generate_sequence_dataset(
        n_per_pattern=args.n_per_pattern,
        seq_len_range=(args.min_len, args.max_len),
        seed=args.seed,
        patterns=patterns,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    n_sequences = df["sequence_id"].nunique()
    print(f"Wrote {len(df):,} rows across {n_sequences:,} sequences -> {args.out}\n")
    print("Rows per temporal_pattern:")
    print(df.drop_duplicates("sequence_id")["temporal_pattern"].value_counts().to_string())
    print(f"\nSequence length: min={df['seq_length'].min()}, "
          f"max={df['seq_length'].max()}, mean={df['seq_length'].mean():.1f}")
    print(f"Fraud rate (row-level): {df['is_fraud'].mean():.1%}")


if __name__ == "__main__":
    main()
