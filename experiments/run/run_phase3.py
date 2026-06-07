#!/usr/bin/env python3
"""Run the Phase 3 behavioural feature screening experiment.

Usage:
    python -m experiments.run.run_phase3 --list
    python -m experiments.run.run_phase3 vanilla
    python -m experiments.run.run_phase3 neutral-avg_improvement
    python -m experiments.run.run_phase3 all
    python -m experiments.run.run_phase3 neutral          # all neutral conditions
    python -m experiments.run.run_phase3 directional      # all directional conditions
    python -m experiments.run.run_phase3 vanilla --sanity

    # Generate summary CSVs for existing results:
    python -m experiments.run.run_phase3 --summarise
"""

from experiments.phase3_experiment import main

if __name__ == "__main__":
    main()
