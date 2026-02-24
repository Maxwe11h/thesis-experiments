#!/usr/bin/env python
"""Run the vanilla baseline experiment.

Uses BLADE's Experiment for seed management, structured logging,
and progress display. Results are saved to results/vanilla/.

Usage:
    python run_vanilla_baseline.py          # all 3 seeds
    python run_vanilla_baseline.py 0        # seed 0 only
    python run_vanilla_baseline.py 0 1      # seeds 0 and 1
"""

import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
for p in [ROOT, os.path.join(ROOT, "LLaMEA"), os.path.join(ROOT, "BLADE")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.run_experiment import run_condition

if len(sys.argv) > 1:
    seeds = [int(s) for s in sys.argv[1:]]
else:
    seeds = [0, 1, 2]

print(f"Running vanilla baseline with seeds: {seeds}")
logger = run_condition("vanilla", seeds=seeds, show_stdout=True)
print(f"\nDone! Results saved to {logger.dirname}/")
