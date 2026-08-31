"""
generate_dataset.py
--------------------
CLI entry point for Member 2's deliverable.

Usage examples
--------------
# Full balanced training dataset, all 10 attack types + legitimate baseline
python generate_dataset.py --mode all --n_per_attack 1200 --out data/phantomguard_synth.csv

# Just one attack type (e.g. to hand to Member 3 for a quick model test)
python generate_dataset.py --mode single --attack_type voice_cloning_fraud --n 2000 --out data/voice_cloning.csv

# List available attack types
python generate_dataset.py --list
"""

import argparse
import os
import sys

from config import ATTACK_TYPES
from transaction_sim import generate_transactions, generate_balanced_dataset, generate_realistic_holdout
from fidelity_checker import run_fidelity_report, print_report


def main():
    parser = argparse.ArgumentParser(description="PhantomGuard synthetic fraud data generator")
    parser.add_argument("--mode", choices=["all", "single", "realistic"], default="all")
    parser.add_argument("--attack_type", type=str, default=None,
                         help="Required when --mode single. A single type (e.g. "
                              "'account_takeover'), or a '+'-joined combo for a "
                              "blended attack (e.g. 'ai_phishing+account_takeover'). "
                              "Use --list to see valid individual types.")
    parser.add_argument("--n", type=int, default=2000, help="Rows to generate for --mode single")
    parser.add_argument("--n_per_attack", type=int, default=1200,
                         help="Rows per attack type for --mode all")
    parser.add_argument("--out", type=str, default="phantomguard_synthetic_dataset.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list", action="store_true", help="List attack types and exit")
    args = parser.parse_args()

    if args.list:
        print("Available attack_type values:")
        for t in ATTACK_TYPES:
            print(f"  {t}")
        sys.exit(0)

    if args.mode == "single":
        if not args.attack_type:
            parser.error("--attack_type is required with --mode single")
        requested = args.attack_type.split("+")
        for a in requested:
            if a not in ATTACK_TYPES:
                parser.error(f"Unknown attack_type '{a}'. Valid options: {ATTACK_TYPES}")
        attack_arg = requested[0] if len(requested) == 1 else requested
        df = generate_transactions(attack_arg, args.n, seed=args.seed)
    elif args.mode == "realistic":
        df = generate_realistic_holdout(n_total=args.n, seed=args.seed)
    else:
        df = generate_balanced_dataset(n_per_attack=args.n_per_attack, seed=args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} rows -> {args.out}\n")

    report = run_fidelity_report(df)
    print_report(report)


if __name__ == "__main__":
    main()
