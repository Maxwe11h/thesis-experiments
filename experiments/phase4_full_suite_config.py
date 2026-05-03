"""Phase 4 full-suite generalisation: 4 LLM winners + CMA-ES on all 1 000
MA-BBOB functions x 5 instances x {5, 10, 20} dims under 2 000d budget.

Per-feedback-rule (`feedback_dont_modify_existing`), this is a NEW config
file rather than an edit of `phase4_config.py`.
"""
import os

from .phase1_config import BBOB_BOUNDS  # noqa: F401  (re-exported for runner)

EVAL_TIMEOUT = 1800  # 30 min per algorithm x dim x instance shard

# Test set
ALL_INSTANCES = list(range(1000))
EVAL_SEEDS = 5

# Dimensions and budget
DIMS = [5, 10, 20]
BUDGET_FACTOR = 2000  # FEs = 2000 * dim, BBOB convention

# Algorithm shards
ALGORITHMS = {
    'vanilla_winner':          'analysis/figs_phase4/p4_winners/vanilla_winner.py',
    'neutral_winner':          'analysis/figs_phase4/p4_winners/neutral_winner.py',
    'sage_winner':             'analysis/figs_phase4/p4_winners/sage_winner.py',
    'combined_neutral_winner': 'analysis/figs_phase4/p4_winners/combined_neutral_winner.py',
    'cma_es':                  'BUILTIN:cma_es',  # served by the runner directly
}

RESULTS_DIR = os.environ.get('PHASE4_FULL_SUITE_DIR', 'results_phase4_full_suite')
