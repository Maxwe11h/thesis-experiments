"""Shared constants for the feasibility-stage experiment (MA-BBOB)."""

TRAINING_INSTANCES = [0, 7, 14]
DIMS = [5]

BUDGET_FACTOR = 2000
LLAMEA_BUDGET = 100

BBOB_BOUNDS = [(-5.0, 5.0)]

N_PARENTS = 5
N_OFFSPRING = 5

OLLAMA_MODEL = "qwen2.5-coder:7b"
SEEDS = range(3)

ALLOWED_IMPORTS = ["numpy"]
