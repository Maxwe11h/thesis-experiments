"""Phase 4 full-suite generalisation: 4 LLM winners + CMA-ES on all 1 000
MA-BBOB functions x 5 instances x {5, 10, 20} dims under 2 000d budget.

Per-feedback-rule (`feedback_dont_modify_existing`), this is a NEW config
file rather than an edit of `phase4_config.py`.
"""
import os

from .phase1_config import BBOB_BOUNDS  # noqa: F401  (re-exported for runner)

EVAL_TIMEOUT = int(os.environ.get('EVAL_TIMEOUT_SECONDS', 1800))  # per-shard wall budget

# Test set
ALL_INSTANCES = list(range(1000))
EVAL_SEEDS = 5

# Dimensions and budget
DIMS = [5, 10, 20]
BUDGET_FACTOR = 2000  # FEs = 2000 * dim, BBOB convention

# Algorithm shards
ALGORITHMS = {
    'vanilla_winner':          'docs/stage4_winners/vanilla_winner.py',
    'neutral_winner':          'docs/stage4_winners/neutral_winner.py',
    'sage_winner':             'docs/stage4_winners/sage_winner.py',
    'combined_neutral_winner': 'docs/stage4_winners/combined_neutral_winner.py',
    'cma_es':                  'BUILTIN:cma_es',  # served by the runner directly
    'neighborhood_adaptive_de': 'baselines/neighborhood_adaptive_de.py',  # external: 2025 MA-BBOB competition winner
}

RESULTS_DIR = os.environ.get('PHASE4_FULL_SUITE_DIR', 'results_phase4_full_suite')
