"""Phase 3 configuration: behavioural feature screening with neutral vs directional feedback.

This experiment tests each of the 10 selected behavioural features individually as
LLM feedback, in two formats (neutral and directional), against a vanilla AOCC-only
baseline. All conditions use the same model (gemini-3-flash), benchmark, and (1+1)-ES
strategy from Phase 1.

Conditions:
  - neutral-{feature}    : AOCC + feature value (no interpretation)
  - directional-{feature}: AOCC + feature value + directional guidance
  - comparative-{feature}: AOCC + feature value + comparison against top-performing reference

Vanilla baseline reused from Phase 1 gemini-3-flash results.
Total: 29 conditions × 5 seeds = 145 runs (longest_no_improvement_streak
excluded from comparative due to non-monotonic AOCC relationship).
"""

import os
from .phase1_config import (
    TRAINING_INSTANCES,
    EVAL_SEEDS,
    DIMS,
    BUDGET_FACTOR,
    BBOB_BOUNDS,
    ALLOWED_IMPORTS,
    EVAL_TIMEOUT,
    N_PARENTS,
    N_OFFSPRING,
    ELITISM,
    LLAMEA_BUDGET,
    MUTATION_PROMPTS,
    RUN_SEEDS,
)

# ---------------------------------------------------------------------------
# Model — use gemini-3-flash only
# ---------------------------------------------------------------------------
MODEL_TAG = "gemini-3-flash"
MODEL_CFG = {"type": "gemini", "model": "gemini-3-flash-preview"}

# ---------------------------------------------------------------------------
# Selected features from Phase 2 (Borda-count consensus + greedy de-dup)
# ---------------------------------------------------------------------------
# NOTE: "fitness_autocorrelation_lag1" was renamed to "fitness_autocorrelation"
# on 2026-03-11 to simplify the name (lag-1 is the only lag we compute).
# Phase 1 result data still uses the old key; the analysis notebook maps it
# on load via backward-compat logic in load_all().
SELECTED_FEATURES = [
    "avg_improvement",
    "intensification_ratio",
    "fitness_plateau_fraction",
    "step_size_autocorrelation",
    "improvement_spatial_correlation",
    "half_convergence_time",
    "fitness_autocorrelation",
    "x_spread_early",
    "longest_no_improvement_streak",
    "dimension_convergence_heterogeneity",
]

# ---------------------------------------------------------------------------
# Feedback formats
# ---------------------------------------------------------------------------
FEEDBACK_FORMATS = ["neutral", "directional", "comparative"]

# Features excluded from comparative feedback due to non-monotonic relationship
# with AOCC. longest_no_improvement_streak has a U-shaped curve: both top and
# bottom algorithms have high streaks (early convergence vs stagnation), so
# "distance from top-10% reference" is misleading.
COMPARATIVE_EXCLUDE = {"longest_no_improvement_streak"}

# ---------------------------------------------------------------------------
# Build condition registry: condition_tag -> feedback format info
# ---------------------------------------------------------------------------
def get_conditions():
    """Return dict of condition_tag -> (feature_name, format).

    Returns:
        dict mapping condition tag strings to tuples of (feature_name, format_name).
        Vanilla baseline is NOT included — reuse Phase 1 gemini-3-flash results.
    """
    conditions = {}
    for feature in SELECTED_FEATURES:
        for fmt in FEEDBACK_FORMATS:
            if fmt == "comparative" and feature in COMPARATIVE_EXCLUDE:
                continue
            tag = f"{fmt}-{feature}"
            conditions[tag] = (feature, fmt)
    return conditions


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
RESULTS_DIR = os.environ.get("PHASE3_RESULTS_DIR", "results_phase3")
