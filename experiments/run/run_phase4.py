#!/usr/bin/env python3
"""Run the Phase 4 full benchmark comparison experiment.

Usage:
    python -m experiments.run.run_phase4 --list
    python -m experiments.run.run_phase4 vanilla
    python -m experiments.run.run_phase4 all
    python -m experiments.run.run_phase4 all --sanity
    python -m experiments.run.run_phase4 --summarise
"""

from experiments.phase4_experiment import main

if __name__ == "__main__":
    main()
