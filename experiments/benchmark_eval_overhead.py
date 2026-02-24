#!/usr/bin/env python
"""Benchmark: compare subprocess-per-eval vs worker-pool evaluation overhead.

Uses a known-good simple RandomSearch algorithm so we measure infrastructure
overhead rather than algorithm quality.
"""

import time
import sys
from pathlib import Path

import numpy as np

# Ensure the thesis root is on sys.path so imports resolve
_THESIS_ROOT = Path(__file__).resolve().parents[1]
for p in [str(_THESIS_ROOT), str(_THESIS_ROOT / "LLaMEA"), str(_THESIS_ROOT / "BLADE")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from iohblade.solution import Solution
from experiments.mabbob_problem import MaBBOBProblem
from experiments.feedback import vanilla_feedback
from experiments.config import TRAINING_INSTANCES, DIMS, BUDGET_FACTOR, BBOB_BOUNDS, ALLOWED_IMPORTS

# A simple RandomSearch algorithm that is known to work
_RANDOM_SEARCH_CODE = '''
import numpy as np

class RandomSearch:
    """Simple random search baseline."""
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        best_x = None
        best_f = float("inf")
        for _ in range(self.budget):
            x = np.random.uniform(-5.0, 5.0, self.dim)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x
        return best_f, best_x
'''

N_EVALS = 5  # number of evaluations per mode


def make_solution():
    sol = Solution(code=_RANDOM_SEARCH_CODE, name="RandomSearch")
    return sol


def benchmark_mode(use_worker_pool, n=N_EVALS):
    mode = "worker_pool" if use_worker_pool else "subprocess"
    print(f"\n{'='*50}")
    print(f"  Mode: {mode}  ({n} evaluations)")
    print(f"{'='*50}")

    problem = MaBBOBProblem(
        make_feedback=vanilla_feedback,
        training_instances=TRAINING_INSTANCES,
        dims=DIMS,
        budget_factor=BUDGET_FACTOR,
        bbob_bounds=BBOB_BOUNDS,
        allowed_imports=ALLOWED_IMPORTS,
        use_worker_pool=use_worker_pool,
    )

    times = []
    try:
        for i in range(n):
            sol = make_solution()
            t0 = time.perf_counter()
            result = problem(sol)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            status = f"fitness={result.fitness:.4f}" if result.fitness > float("-inf") else f"ERROR: {result.error[:80]}"
            print(f"  [{i+1}/{n}] {elapsed:.2f}s — {status}")
    finally:
        problem.cleanup()

    return times


def main():
    print("Benchmark: subprocess-per-eval vs persistent worker pool")
    print(f"Training instances: {TRAINING_INSTANCES}")
    print(f"Dims: {DIMS}, Budget factor: {BUDGET_FACTOR}")

    # Subprocess per eval
    t_sub = benchmark_mode(use_worker_pool=False)

    # Worker pool
    t_pool = benchmark_mode(use_worker_pool=True)

    # Report
    mean_sub = np.mean(t_sub)
    mean_pool = np.mean(t_pool)
    speedup = mean_sub / mean_pool if mean_pool > 0 else float("inf")

    print(f"\n{'='*50}")
    print(f"  RESULTS")
    print(f"{'='*50}")
    print(f"  Subprocess mean: {mean_sub:.2f}s  (std {np.std(t_sub):.2f}s)")
    print(f"  Worker pool mean: {mean_pool:.2f}s  (std {np.std(t_pool):.2f}s)")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Per-eval savings: {mean_sub - mean_pool:.2f}s")
    print(f"  Over 100 evals: ~{(mean_sub - mean_pool) * 100:.0f}s saved")


if __name__ == "__main__":
    main()
