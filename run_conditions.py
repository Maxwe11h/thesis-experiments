#!/usr/bin/env python3
"""Run a subset of experiment conditions.

Usage:
    python run_conditions.py vanilla dispersion success_rate avg_improvement
    python run_conditions.py --list          # print all 12 condition names
"""

import argparse
import sys

from experiments.run_experiment import CONDITIONS, run_condition


def main():
    parser = argparse.ArgumentParser(description="Run selected experiment conditions")
    parser.add_argument(
        "conditions", nargs="*",
        help="Condition names to run (see --list for options)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print all available condition names and exit",
    )
    parser.add_argument(
        "--no-worker-pool", action="store_true",
        help="Disable worker pool (run evaluations in-process)",
    )
    args = parser.parse_args()

    if args.list:
        for i, name in enumerate(CONDITIONS, 1):
            print(f"  {i:2d}. {name}")
        return

    if not args.conditions:
        parser.error("provide at least one condition name (or --list)")

    for name in args.conditions:
        if name not in CONDITIONS:
            print(f"ERROR: unknown condition '{name}'", file=sys.stderr)
            print(f"Valid conditions: {', '.join(CONDITIONS)}", file=sys.stderr)
            sys.exit(1)

    for name in args.conditions:
        print(f"\n{'#'*60}")
        print(f"  CONDITION: {name}")
        print(f"{'#'*60}")
        run_condition(
            name,
            show_stdout=True,
            log_stdout=True,
            use_worker_pool=not args.no_worker_pool,
        )

    print("\nAll conditions complete.")


if __name__ == "__main__":
    main()
