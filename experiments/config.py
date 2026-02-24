"""Shared constants for the feature-selection experiment (MA-BBOB)."""

# 10 MA-BBOB training instances (indices into the CSV data)
TRAINING_INSTANCES = list(range(10))

EVAL_SEEDS = 5  # random seeds per instance in the inner evaluation loop
# Total per candidate: 10 instances × 5 seeds = 50 algorithm runs

DIMS = [5]
BUDGET_FACTOR = 2000
LLAMEA_BUDGET = 100

BBOB_BOUNDS = [(-5.0, 5.0)]

# (1+1)-ES
N_PARENTS = 1
N_OFFSPRING = 1

OLLAMA_MODEL = "qwen3:8b"

ALLOWED_IMPORTS = ["numpy"]

BEHAVIORAL_FEATURES = [
    "avg_nearest_neighbor_distance",
    "dispersion",
    "avg_exploration_pct",
    "avg_distance_to_best",
    "intensification_ratio",
    "avg_exploitation_pct",
    "average_convergence_rate",
    "avg_improvement",
    "success_rate",
    "longest_no_improvement_streak",
    "last_improvement_fraction",
]
