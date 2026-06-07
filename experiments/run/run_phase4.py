#!/usr/bin/env python3
"""Run the Phase 4 full benchmark comparison experiment.

Usage:
    python run_phase4.py --list
    python run_phase4.py vanilla
    python run_phase4.py all
    python run_phase4.py all --sanity
    python run_phase4.py --summarise
"""

from experiments.phase4_experiment import main

if __name__ == "__main__":
    main()
