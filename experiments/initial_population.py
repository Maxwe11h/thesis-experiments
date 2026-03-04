"""Shared initial algorithm for Phase 1: one hand-crafted RandomSearch.

All LLMs and all seeds start from the same algorithm so that differences
in evolutionary outcomes are attributable to the LLM, not to initialization
luck.

The algorithm is deliberately simple so every LLM has room to improve:
  - RandomSearch — uniform sampling, no adaptation

Usage:
    from experiments.initial_population import get_initial_solutions
    solutions = get_initial_solutions()   # list of 1 unevaluated Solution
"""

import json
from pathlib import Path

from iohblade.solution import Solution

# ---------------------------------------------------------------------------
# Algorithm code string
# ---------------------------------------------------------------------------

ALGORITHM_1_NAME = "RandomSearch"
ALGORITHM_1_DESC = (
    "Uniform random sampling across the search space; returns the best sample."
)
ALGORITHM_1_CODE = """\
import numpy as np

class RandomSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        for i in range(self.budget):
            x = np.random.uniform(-5.0, 5.0, self.dim)
            f = func(x)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x.copy()
        return self.f_opt, self.x_opt
"""

# Packed list so callers can iterate (single entry for 1+1 strategy)
_INITIAL_ALGORITHMS = [
    (ALGORITHM_1_NAME, ALGORITHM_1_DESC, ALGORITHM_1_CODE),
]


def get_initial_solutions():
    """Return a list of unevaluated Solution objects for the initial population.

    Returns 1 solution for the (1+1)-ES strategy. The Solution has code, name,
    and description set, but fitness is NaN (must be evaluated by the problem
    before use in the evolutionary loop).
    """
    solutions = []
    for name, desc, code in _INITIAL_ALGORITHMS:
        sol = Solution(
            name=name,
            description=desc,
            code=code,
            generation=0,
        )
        sol.task_prompt = ""  # will be overwritten by LLaMEA from the problem
        solutions.append(sol)
    return solutions


def save_initial_population(path="initial_population.json"):
    """Persist the initial algorithm(s) to a JSON file for auditing."""
    records = []
    for name, desc, code in _INITIAL_ALGORITHMS:
        records.append({"name": name, "description": desc, "code": code})
    Path(path).write_text(json.dumps(records, indent=2))
    return path


def load_initial_population(path="initial_population.json"):
    """Load initial algorithms from a JSON file and return Solution objects."""
    records = json.loads(Path(path).read_text())
    solutions = []
    for rec in records:
        sol = Solution(
            name=rec["name"],
            description=rec["description"],
            code=rec["code"],
            generation=0,
        )
        sol.task_prompt = ""
        solutions.append(sol)
    return solutions
