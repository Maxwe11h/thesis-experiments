#!/usr/bin/env python
"""Run the vanilla baseline twice: subprocess-per-eval vs worker pool.

Runs all seeds for each mode using BLADE's Experiment, then compares
total wall-clock times.

Usage:
    python run_baseline_comparison.py
"""

import sys
import os
import time
import json
from datetime import timedelta

ROOT = os.path.dirname(os.path.abspath(__file__))
for p in [ROOT, os.path.join(ROOT, "LLaMEA"), os.path.join(ROOT, "BLADE")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.run_experiment import run_condition

SEEDS = [0, 1, 2]


def fmt(seconds):
    return str(timedelta(seconds=int(seconds)))


def main():
    results = {}

    for mode_name, use_pool in [("subprocess", False), ("worker_pool", True)]:
        print(f"\n{'#'*60}")
        print(f"  MODE: {mode_name}")
        print(f"{'#'*60}")

        t0 = time.perf_counter()
        logger = run_condition(
            "vanilla", seeds=SEEDS, show_stdout=True,
            use_worker_pool=use_pool,
        )
        elapsed = time.perf_counter() - t0

        results[mode_name] = {
            "elapsed_s": elapsed,
            "log_dir": logger.dirname,
        }
        print(f"\n  {mode_name} total: {fmt(elapsed)}")

    # --- Comparison ---
    sub = results["subprocess"]["elapsed_s"]
    pool = results["worker_pool"]["elapsed_s"]
    speedup = sub / pool if pool > 0 else float("inf")

    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    print(f"  Subprocess:  {fmt(sub)}")
    print(f"  Worker pool: {fmt(pool)}")
    print(f"  Speedup:     {speedup:.2f}x")
    print(f"  Time saved:  {fmt(sub - pool)}")

    out_path = os.path.join(ROOT, "experiments", "baseline_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
