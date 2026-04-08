"""Phase 4 configuration: full benchmark comparison of feedback conditions.

This experiment compares six feedback conditions head-to-head on a larger
MA-BBOB benchmark (20 instances, 500 candidates):
  - vanilla:              AOCC only
  - neutral:              AOCC + 5 neutral behavioral features
  - directional:          AOCC + 5 directional behavioral features
  - sage:                 AOCC + structural code guidance (SAGE)
  - combined_neutral:     AOCC + 5 neutral features + SAGE
  - combined_directional: AOCC + 5 directional features + SAGE

All conditions use gemini-3-flash with thinking disabled (thinking_budget=0),
(1+1)-ES with 90/10 refine/explore, and 10 independent seeds per condition.

Total: 6 conditions x 10 seeds = 60 runs.
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
# Model — gemini-3-flash (best from Phase 1), thinking disabled
# ---------------------------------------------------------------------------
MODEL_TAG = "gemini-3-flash"
MODEL_CFG = {
    "type": "gemini",
    "model": "gemini-3-flash-preview",
    "generation_config": {"thinking_config": {"thinking_budget": 0}},
}

# ---------------------------------------------------------------------------
# Benchmark — 20 MA-BBOB instances selected by group-stratified greedy + swap
# ---------------------------------------------------------------------------
# Selected by analysis/select_instances.py methodology (greedy + local search)
# with K=20, excluding Phase 1 instances. Scoring = coverage_penalty
# - 3.0*func_cv - 5.0*group_std.
# Coverage: 24/24 functions, func CV=0.080, group std=0.011
# Group shares: Sep 20.0%, Low/mod 18.0%, High/uni 20.7%, Multi-adeq 21.3%, Multi-weak 20.0%
TRAINING_INSTANCES = [22, 93, 166, 196, 203, 288, 321, 408, 480, 513,
                      528, 598, 697, 781, 784, 803, 894, 947, 951, 999]

# ---------------------------------------------------------------------------
# Evolution settings
# ---------------------------------------------------------------------------
LLAMEA_BUDGET = 500
RUN_SEEDS = list(range(10))  # 10 independent seeds per condition

# ---------------------------------------------------------------------------
# Top 5 neutral features (from Phase 3 screening, ranked by mean best AOCC)
# ---------------------------------------------------------------------------
NEUTRAL_FEATURES = [
    "intensification_ratio",
    "dimension_convergence_heterogeneity",
    "fitness_plateau_fraction",
    "avg_improvement",
    "improvement_spatial_correlation",
]

# ---------------------------------------------------------------------------
# Top 5 directional features (from Phase 3 screening, ranked by mean best AOCC)
# ---------------------------------------------------------------------------
DIRECTIONAL_FEATURES = [
    "fitness_plateau_fraction",
    "avg_improvement",
    "half_convergence_time",
    "longest_no_improvement_streak",
    "improvement_spatial_correlation",
]

# ---------------------------------------------------------------------------
# Condition registry
# ---------------------------------------------------------------------------
# Each condition maps to a dict with:
#   - feedback: "vanilla" or "behavioural" (which feedback function to use)
#   - sage: bool (whether to enable feature_guided_mutation in LLaMEA)
CONDITIONS = {
    "vanilla":              {"feedback": "vanilla",      "sage": False},
    "neutral":              {"feedback": "neutral",      "sage": False},
    "directional":          {"feedback": "directional",  "sage": False},
    "sage":                 {"feedback": "vanilla",      "sage": True},
    "combined_neutral":     {"feedback": "neutral",      "sage": True},
    "combined_directional": {"feedback": "directional",  "sage": True},
}

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
RESULTS_DIR = os.environ.get("PHASE4_RESULTS_DIR", "results_phase4")
