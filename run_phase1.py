#!/usr/bin/env python3
"""Run the Phase 1 LLM screening experiment.

Usage:
    python run_phase1.py --list
    python run_phase1.py qwen3.5-4b
    python run_phase1.py qwen3.5-4b --seeds 0 1 2
    python run_phase1.py all --budget 50
    python run_phase1.py qwen3.5-4b --sanity
    python run_phase1.py my-tag --custom-ollama "mistral:7b"
    GOOGLE_API_KEY=... python run_phase1.py gemini-3-pro

    # On a specific GPU/port:
    OLLAMA_PORT=11435 python run_phase1.py qwen3.5-27b --seeds 0 1

    # Generate summary CSVs for existing results:
    python run_phase1.py --summarise
"""

from experiments.phase1_experiment import main

if __name__ == "__main__":
    main()
