"""Phase 4 configuration: full benchmark comparison of feedback conditions.

This experiment compares four feedback conditions head-to-head on a larger
MA-BBOB benchmark (20 instances, 500 candidates):
  - vanilla:      AOCC only
  - behavioural:  AOCC + 5 neutral behavioral features
  - sage:         AOCC + structural code guidance (SAGE)
  - combined:     AOCC + 5 neutral features + SAGE

All conditions use gemini-3-flash, (1+1)-ES with 90/10 refine/explore,
and 10 independent seeds per condition.

Total: 4 conditions x 10 seeds = 40 runs.
"""

import os
from .phase1_config import (
    EVAL_SEEDS,
    DIMS,
    BUDGET_FACTOR,
    BBOB_BOUNDS,
    ALLOWED_IMPORTS,
    N_PARENTS,
    N_OFFSPRING,
    ELITISM,
    MUTATION_PROMPTS,
)

# Override Phase 1 timeout: 20 instances (2x Phase 1) + CPU contention headroom
EVAL_TIMEOUT = 1200

# ---------------------------------------------------------------------------
# Model — gemini-3-flash (best from Phase 1)
# ---------------------------------------------------------------------------
MODEL_TAG = "gemini-3-flash"
MODEL_CFG = {"type": "gemini", "model": "gemini-3-flash-preview"}

# ---------------------------------------------------------------------------
# Benchmark — 20 MA-BBOB instances selected for per-function uniformity
# ---------------------------------------------------------------------------
# Selected by analysis/select_instances_phase4.py: greedy + SA (20 restarts)
# + local search. Scoring = -missing*100 - func_cv*10 - group_std.
# Coverage: 24/24 functions, CV=0.067, all groups within 3% of ideal 20%.
TRAINING_INSTANCES = [35, 48, 50, 288, 396, 445, 514, 590, 605, 642,
                      680, 721, 725, 770, 780, 805, 816, 831, 916, 968]

# ---------------------------------------------------------------------------
# Evolution settings
# ---------------------------------------------------------------------------
LLAMEA_BUDGET = 500
RUN_SEEDS = list(range(10))  # 10 independent seeds per condition

# ---------------------------------------------------------------------------
# Top 5 neutral features (from Phase 3 screening)
# ---------------------------------------------------------------------------
NEUTRAL_FEATURES = [
    "intensification_ratio",
    "dimension_convergence_heterogeneity",
    "fitness_plateau_fraction",
    "avg_improvement",
    "improvement_spatial_correlation",
]

# ---------------------------------------------------------------------------
# Condition registry
# ---------------------------------------------------------------------------
# Each condition maps to a dict with:
#   - feedback: "vanilla" or "behavioural" (which feedback function to use)
#   - sage: bool (whether to enable feature_guided_mutation in LLaMEA)
CONDITIONS = {
    "vanilla":     {"feedback": "vanilla",     "sage": False},
    "behavioural": {"feedback": "behavioural", "sage": False},
    "sage":        {"feedback": "vanilla",     "sage": True},
    "combined":    {"feedback": "behavioural", "sage": True},
}

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
RESULTS_DIR = os.environ.get("PHASE4_RESULTS_DIR", "results_phase4")
