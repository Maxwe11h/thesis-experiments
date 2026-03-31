#!/usr/bin/env python3
"""Select 20 MA-BBOB instances optimising per-function uniformity.

Phase 4 uses 20 instances (double Phase 1's 10) with the primary goal of
uniform per-function coverage rather than per-group balance.

Scoring:  score = -missing * 100 - func_cv * 10 - group_std
  - missing: number of BBOB functions with zero weight (heaviest penalty)
  - func_cv: coefficient of variation of per-function weight shares (primary)
  - group_std: std of group weight shares (secondary tiebreaker)

Method: greedy seed + simulated annealing with multi-restart.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Load weights
# ---------------------------------------------------------------------------
WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "BLADE" / "iohblade" / "benchmarks" / "BBOB" / "mabbob" / "weights.csv"
weights = pd.read_csv(WEIGHTS_PATH, index_col=0)
W = weights.values  # 1000 x 24
N_INSTANCES = W.shape[0]

GROUPS = {
    "Separable (f1-f5)": list(range(0, 5)),
    "Low/mod conditioning (f6-f9)": list(range(5, 9)),
    "High cond / unimodal (f10-f14)": list(range(9, 14)),
    "Multimodal adequate (f15-f19)": list(range(14, 19)),
    "Multimodal weak (f20-f24)": list(range(19, 24)),
}

K = 20


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_subset(indices):
    """Score a subset of instance indices. Higher is better."""
    sub = W[indices]
    func_totals = sub.sum(axis=0)
    total = func_totals.sum()
    if total == 0:
        return -9999

    missing = (func_totals == 0).sum()

    nonzero = func_totals[func_totals > 0]
    func_cv = np.std(nonzero) / np.mean(nonzero) if len(nonzero) > 0 else 999

    group_shares = [func_totals[cols].sum() / total for cols in GROUPS.values()]
    group_std = np.std(group_shares)

    return -missing * 100 - func_cv * 10 - group_std


# ---------------------------------------------------------------------------
# Greedy forward selection (seed for SA)
# ---------------------------------------------------------------------------

print(f"Pool: {N_INSTANCES} instances, {W.shape[1]} BBOB functions")
print(f"Selecting K={K} instances\n")
print("--- Greedy forward selection ---")

selected = []
remaining = list(range(N_INSTANCES))

for step in range(K):
    best_score = -9999
    best_idx = -1
    for idx in remaining:
        s = score_subset(selected + [idx])
        if s > best_score:
            best_score = s
            best_idx = idx
    selected.append(best_idx)
    remaining.remove(best_idx)
    n_covered = (W[selected].sum(axis=0) > 0).sum()
    print(f"  Step {step+1:2d}: added instance {best_idx:>3d} "
          f"({n_covered}/24 covered, score={best_score:.4f})")

greedy_score = score_subset(selected)
print(f"\nGreedy score: {greedy_score:.6f}")


# ---------------------------------------------------------------------------
# Local search: swap refinement (polish the greedy solution)
# ---------------------------------------------------------------------------

def local_search(solution):
    """Exhaustive single-swap local search. Returns improved solution."""
    sol = solution.copy()
    improved = True
    while improved:
        improved = False
        current_score = score_subset(sol)
        sol_set = set(sol)
        for i in range(K):
            old_idx = sol[i]
            best_swap_score = current_score
            best_swap_idx = old_idx
            for candidate in range(N_INSTANCES):
                if candidate in sol_set:
                    continue
                sol[i] = candidate
                s = score_subset(sol)
                if s > best_swap_score:
                    best_swap_score = s
                    best_swap_idx = candidate
            sol[i] = old_idx
            if best_swap_idx != old_idx:
                sol_set.discard(old_idx)
                sol_set.add(best_swap_idx)
                sol[i] = best_swap_idx
                current_score = best_swap_score
                improved = True
    return sol


print("\n--- Local search (polish greedy) ---")
selected = local_search(selected)
ls_score = score_subset(selected)
print(f"  Score after local search: {ls_score:.6f}")


# ---------------------------------------------------------------------------
# Simulated annealing with multi-restart
# ---------------------------------------------------------------------------

def simulated_annealing(init_solution, rng, n_iter=50000, t_start=0.05, t_end=1e-5):
    """SA with geometric cooling, single-swap neighbourhood."""
    sol = init_solution.copy()
    sol_score = score_subset(sol)
    best = sol.copy()
    best_score = sol_score
    sol_set = set(sol)
    pool = [i for i in range(N_INSTANCES) if i not in sol_set]

    cooling = (t_end / t_start) ** (1.0 / n_iter)
    temp = t_start

    for step in range(n_iter):
        # Pick a random position and a random replacement
        pos = rng.integers(K)
        pool_idx = rng.integers(len(pool))

        old_val = sol[pos]
        new_val = pool[pool_idx]

        sol[pos] = new_val
        new_score = score_subset(sol)
        delta = new_score - sol_score

        if delta > 0 or rng.random() < np.exp(delta / temp):
            # Accept
            sol_set.discard(old_val)
            sol_set.add(new_val)
            pool[pool_idx] = old_val
            sol_score = new_score
            if sol_score > best_score:
                best = sol.copy()
                best_score = sol_score
        else:
            # Reject
            sol[pos] = old_val

        temp *= cooling

    return best, best_score


N_RESTARTS = 20
SA_ITERS = 100000

print(f"\n--- Simulated annealing ({N_RESTARTS} restarts x {SA_ITERS} iterations) ---")

global_best = selected.copy()
global_best_score = ls_score

for restart in range(N_RESTARTS):
    rng = np.random.default_rng(seed=restart)

    if restart == 0:
        # First restart: use the greedy+LS solution
        init = selected.copy()
    else:
        # Random initial solution (but ensure full coverage)
        init = list(rng.choice(N_INSTANCES, size=K, replace=False))

    sa_best, sa_score = simulated_annealing(init, rng, n_iter=SA_ITERS)

    # Polish with local search
    sa_best = local_search(sa_best)
    sa_score = score_subset(sa_best)

    if sa_score > global_best_score:
        global_best = sa_best.copy()
        global_best_score = sa_score
        marker = " ***"
    else:
        marker = ""

    print(f"  Restart {restart+1:2d}: score={sa_score:.6f}{marker}")

selected = [int(x) for x in global_best]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

selected.sort()
print(f"\n{'='*60}")
print(f"Selected {K} instances: {selected}")
print(f"{'='*60}")

sub = W[selected]
func_totals = sub.sum(axis=0)
total = func_totals.sum()
func_shares = func_totals / total
n_covered = (func_totals > 0).sum()

print(f"\nCoverage: {n_covered}/24 BBOB functions")
print(f"Per-function share: min={func_shares.min()*100:.1f}%, "
      f"max={func_shares.max()*100:.1f}%, "
      f"CV={func_shares.std()/func_shares.mean():.3f}")

print("\nPer-function breakdown:")
for i in range(24):
    n_inst = (sub[:, i] > 0).sum()
    print(f"  f{i+1:2d}: share={func_shares[i]*100:5.1f}%, "
          f"in {n_inst:2d}/{K} instances")

print("\nGroup balance:")
for gname, cols in GROUPS.items():
    gshare = func_totals[cols].sum() / total
    print(f"  {gname:40s} {gshare*100:5.1f}% (ideal 20%)")

# Python-ready output for config
print(f"\n# For phase4_config.py:")
print(f"TRAINING_INSTANCES = {selected}")
