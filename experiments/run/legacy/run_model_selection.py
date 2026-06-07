#!/usr/bin/env python3
"""Run the model selection experiment.

Usage:
    python -m experiments.run.legacy.run_model_selection --list
    python -m experiments.run.legacy.run_model_selection qwen3-8b llama3.1-8b qwen3-14b
    python -m experiments.run.legacy.run_model_selection all --budget 50
    python -m experiments.run.legacy.run_model_selection my-model --custom-model "mistral:7b" --budget 30

    # Reduced eval config for faster turnaround:
    python -m experiments.run.legacy.run_model_selection all --budget 50 --training-instances 5 --eval-seeds 3

    # On a specific GPU/port:
    OLLAMA_PORT=11435 python -m experiments.run.legacy.run_model_selection qwen3-14b
"""

from experiments.model_selection import main

if __name__ == "__main__":
    main()
