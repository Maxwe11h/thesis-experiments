#!/usr/bin/env python3
"""Run the model selection experiment.

Usage:
    python run_model_selection.py --list
    python run_model_selection.py qwen3-8b llama3.1-8b qwen3-14b
    python run_model_selection.py all --budget 50
    python run_model_selection.py my-model --custom-model "mistral:7b" --budget 30

    # Reduced eval config for faster turnaround:
    python run_model_selection.py all --budget 50 --training-instances 5 --eval-seeds 3

    # On a specific GPU/port:
    OLLAMA_PORT=11435 python run_model_selection.py qwen3-14b
"""

from experiments.model_selection import main

if __name__ == "__main__":
    main()
