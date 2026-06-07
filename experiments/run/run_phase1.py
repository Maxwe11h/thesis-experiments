#!/usr/bin/env python3
"""Run the Phase 1 LLM screening experiment.

Usage:
    python -m experiments.run.run_phase1 --list
    python -m experiments.run.run_phase1 qwen3.5-4b
    python -m experiments.run.run_phase1 qwen3.5-4b --seeds 0 1 2
    python -m experiments.run.run_phase1 all --budget 50
    python -m experiments.run.run_phase1 qwen3.5-4b --sanity
    python -m experiments.run.run_phase1 my-tag --custom-ollama "mistral:7b"
    GOOGLE_API_KEY=... python -m experiments.run.run_phase1 gemini-3-pro

    # On a specific GPU/port:
    OLLAMA_PORT=11435 python -m experiments.run.run_phase1 qwen3.5-27b --seeds 0 1

    # Generate summary CSVs for existing results:
    python -m experiments.run.run_phase1 --summarise
"""

from experiments.phase1_experiment import main

if __name__ == "__main__":
    main()
