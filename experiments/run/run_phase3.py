#!/usr/bin/env python3
"""Run the Phase 3 behavioural feature screening experiment.

Usage:
    python run_phase3.py --list
    python run_phase3.py vanilla
    python run_phase3.py neutral-avg_improvement
    python run_phase3.py all
    python run_phase3.py neutral          # all neutral conditions
    python run_phase3.py directional      # all directional conditions
    python run_phase3.py vanilla --sanity

    # Generate summary CSVs for existing results:
    python run_phase3.py --summarise
"""

from experiments.phase3_experiment import main

if __name__ == "__main__":
    main()
