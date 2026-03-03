"""Phase 1 configuration: LLM screening on MA-BBOB with (1+1)-ES.

This experiment screens multiple LLMs for their baseline algorithm discovery
capability using LLaMEA's evolutionary loop on the MA-BBOB benchmark.

Key settings:
  - Population strategy: (1+1) — 1 parent, 1 offspring, elitism
  - Mutation: 90% refinement, 10% exploration (new algorithm)
  - 5 independent runs per LLM (seeds 0–4)
  - Shared initial algorithm across all LLMs for comparability
"""

import os

# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------
# 10 MA-BBOB training instances selected for balanced BBOB function coverage.
# Greedy selection from 1000-instance pool: all 24 functions covered,
# each of the 5 BBOB groups within 0.6% of the ideal 20% share.
# See analysis/mabbob_instance_selection.ipynb for derivation.
TRAINING_INSTANCES = [191, 277, 300, 412, 455, 635, 648, 744, 760, 843]

EVAL_SEEDS = 5            # random seeds per instance in inner evaluation loop
DIMS = [5]                # problem dimensionalities
BUDGET_FACTOR = 2000      # evaluation budget = BUDGET_FACTOR * dim = 10 000

BBOB_BOUNDS = [(-5.0, 5.0)]
ALLOWED_IMPORTS = ["numpy"]

# ---------------------------------------------------------------------------
# Evolution settings — (1+1)-ES
# ---------------------------------------------------------------------------
N_PARENTS = 1
N_OFFSPRING = 1
ELITISM = True            # (mu + lambda) strategy

# LLaMEA budget: total candidates evaluated per run (including 1 initial)
# With (1+1)-ES this means 1 initial + 99 generations = 100 total candidates
LLAMEA_BUDGET = 100

# ---------------------------------------------------------------------------
# Mutation prompts — 90% refinement, 10% exploration
# ---------------------------------------------------------------------------
# random.choice() picks one per mutation, so 9 refine + 1 explore ≈ 10% explore
MUTATION_PROMPTS = [
    # 9 x refinement (exploitation)
    "Refine the strategy of the selected solution to improve it.",
    "Refine the strategy of the selected solution to improve it.",
    "Refine the strategy of the selected solution to improve it.",
    "Refine the strategy of the selected solution to improve it.",
    "Refine the strategy of the selected solution to improve it.",
    "Refine the strategy of the selected solution to improve it.",
    "Refine the strategy of the selected solution to improve it.",
    "Refine the strategy of the selected solution to improve it.",
    "Refine the strategy of the selected solution to improve it.",
    # 1 x exploration (mutation)
    "Generate a new algorithm that is different from the algorithms you have "
    "tried before. Try a fundamentally different search strategy.",
]

# ---------------------------------------------------------------------------
# LLaMEA run seeds — 5 independent runs per model
# ---------------------------------------------------------------------------
RUN_SEEDS = list(range(5))   # [0, 1, 2, 3, 4]

# ---------------------------------------------------------------------------
# Ollama settings
# ---------------------------------------------------------------------------
OLLAMA_PORT = int(os.environ.get("OLLAMA_PORT", 11434))

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Each entry: tag -> dict with 'type' ("ollama" or "gemini") and model config.
# Ollama entries have 'model' (Ollama model name).
# Gemini entries have 'model' (Gemini model ID) and use GOOGLE_API_KEY env var.
#
# Adjust exact model names to match what's available in your Ollama instance
# or the Gemini API at the time of running.
CANDIDATE_MODELS = {
    # --- Qwen 3.5 family ---
    "qwen3.5-4b":   {"type": "ollama", "model": "qwen3.5:4b"},
    "qwen3.5-9b":   {"type": "ollama", "model": "qwen3.5:9b"},
    "qwen3.5-27b":  {"type": "ollama", "model": "qwen3.5:27b"},
    # --- rnj-1 ---
    "rnj-1-8b":     {"type": "ollama", "model": "rnj-1:8b"},
    # --- Devstral small 2 ---
    "devstral-small-2-24b": {"type": "ollama", "model": "devstral-small-2:24b"},
    # --- OLMo 3 family ---
    "olmo3-7b":     {"type": "ollama", "model": "olmo3:7b"},
    "olmo3-30b":    {"type": "ollama", "model": "olmo3:30b"},
    # --- Granite 4 ---
    "granite4-3b":  {"type": "ollama", "model": "granite4:3b"},
    # --- Gemini 3 (API) ---
    "gemini-3-pro":   {"type": "gemini", "model": "gemini-3.0-pro"},
    "gemini-3-flash": {"type": "gemini", "model": "gemini-3.0-flash"},
}

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
RESULTS_DIR = "results_phase1"
