"""
Select 10 MA-BBOB instances from 1000 that give the most balanced coverage
of all 24 BBOB functions across 5 function groups.
"""

import numpy as np
import pandas as pd
from itertools import combinations

# ── Load data ────────────────────────────────────────────────────────────────
CSV = "/Users/maxharell/repos/thesis/BLADE/iohblade/problems/mabbob/weights.csv"
df = pd.read_csv(CSV, index_col=0)
W = df.values  # (1000, 24)
N, F = W.shape
K = 10  # number of instances to select

print(f"Loaded weights: {N} instances × {F} functions\n")

# ── Group definitions ────────────────────────────────────────────────────────
groups = {
    "Separable (f1-f5)":        list(range(0, 5)),
    "Low/mod cond (f6-f9)":     list(range(5, 9)),
    "High cond/uni (f10-f14)":  list(range(9, 14)),
    "Multi-modal adeq (f15-f19)": list(range(14, 19)),
    "Multi-modal weak (f20-f24)": list(range(19, 24)),
}

# ── Helper: evaluate a subset ────────────────────────────────────────────────
def evaluate(indices):
    """Return a dict of quality metrics for a selection of instance indices."""
    sub = W[indices]                      # (K, 24)
    total_per_func = sub.sum(axis=0)      # (24,) total weight per function

    # Coverage: how many instances have weight > 0.01 for each function
    coverage = (sub > 0.01).sum(axis=0)   # (24,)

    # Group weight shares
    group_weights = {}
    total_weight = total_per_func.sum()
    for gname, cols in groups.items():
        group_weights[gname] = total_per_func[cols].sum() / total_weight

    # Metrics
    n_covered = (total_per_func > 0.01).sum()         # functions with meaningful weight
    func_uniformity = total_per_func.min() / (total_per_func.max() + 1e-12)  # min/max ratio
    group_shares = np.array(list(group_weights.values()))
    group_balance = 1.0 - np.std(group_shares)        # higher = more balanced (ideal std=0)

    return {
        "indices": list(indices),
        "n_covered": n_covered,
        "func_uniformity": func_uniformity,
        "group_balance": group_balance,
        "group_weights": group_weights,
        "total_per_func": total_per_func,
        "coverage": coverage,
        "group_shares_std": np.std(group_shares),
    }

def print_eval(name, ev):
    """Pretty-print evaluation results."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  Selected indices: {ev['indices']}")
    print(f"  Functions covered (weight > 0.01): {ev['n_covered']}/24")
    print(f"  Function uniformity (min/max weight): {ev['func_uniformity']:.4f}")
    print(f"  Group balance (1-std): {ev['group_balance']:.4f}")
    print(f"  Group shares std: {ev['group_shares_std']:.4f}")

    print(f"\n  Per-function coverage:")
    print(f"  {'Func':<6} {'Coverage':>8} {'Total Weight':>12}")
    print(f"  {'-'*28}")
    for f_idx in range(F):
        fname = f"f{f_idx+1}"
        print(f"  {fname:<6} {ev['coverage'][f_idx]:>8d} {ev['total_per_func'][f_idx]:>12.4f}")

    print(f"\n  Per-group weight shares (target: 20% each):")
    for gname, share in ev['group_weights'].items():
        n_funcs = len(groups[gname])
        print(f"  {gname:<30s} {share*100:6.2f}%  ({n_funcs} functions)")


# ── Approach 1: Greedy with multi-objective scoring ──────────────────────────
print("\n" + "~"*70)
print("  OPTIMIZATION: Greedy + Local Search")
print("~"*70)

def score_subset(indices):
    """Combined score: coverage + uniformity + group balance."""
    sub = W[list(indices)]
    total_per_func = sub.sum(axis=0)

    # Penalty for uncovered functions
    n_uncovered = (total_per_func < 0.01).sum()
    coverage_penalty = -10.0 * n_uncovered

    # Function uniformity: minimize coefficient of variation
    mean_w = total_per_func.mean()
    std_w = total_per_func.std()
    func_cv = std_w / (mean_w + 1e-12)

    # Group balance: minimize deviation from 20%
    total_weight = total_per_func.sum()
    group_shares = []
    for gname, cols in groups.items():
        group_shares.append(total_per_func[cols].sum() / total_weight)
    group_std = np.std(group_shares)

    # Combined score (higher is better)
    score = coverage_penalty - 3.0 * func_cv - 5.0 * group_std
    return score


# Greedy construction
selected = []
remaining = set(range(N))

for step in range(K):
    best_score = -np.inf
    best_idx = None
    for candidate in remaining:
        trial = selected + [candidate]
        s = score_subset(trial)
        if s > best_score:
            best_score = s
            best_idx = candidate
    selected.append(best_idx)
    remaining.remove(best_idx)
    print(f"  Step {step+1}: added index {best_idx}, score = {best_score:.4f}")

print(f"\n  Greedy solution: {sorted(selected)}")
greedy_eval = evaluate(selected)
print_eval("Greedy Solution", greedy_eval)


# ── Approach 2: Local search (swap optimization) ────────────────────────────
print("\n" + "~"*70)
print("  LOCAL SEARCH: Swap optimization (starting from greedy)")
print("~"*70)

best_set = set(selected)
best_score = score_subset(best_set)
improved = True
iteration = 0

while improved:
    improved = False
    iteration += 1
    for idx_in in list(best_set):
        for idx_out in range(N):
            if idx_out in best_set:
                continue
            trial = (best_set - {idx_in}) | {idx_out}
            s = score_subset(trial)
            if s > best_score + 1e-8:
                best_set = trial
                best_score = s
                improved = True
                print(f"  Iter {iteration}: swapped {idx_in} -> {idx_out}, score = {s:.4f}")
                break
        if improved:
            break

optimized = sorted(best_set)
print(f"\n  Optimized solution: {optimized}")
opt_eval = evaluate(optimized)
print_eval("Optimized Solution (Greedy + Swap)", opt_eval)


# ── Approach 3: ILP / scipy optimization ─────────────────────────────────────
print("\n" + "~"*70)
print("  ILP APPROACH: Mixed-integer programming")
print("~"*70)

try:
    from scipy.optimize import milp, LinearConstraint, Bounds

    # Decision variables: x_i ∈ {0,1} for each instance, plus slack variable t
    # Objective: maximize t (the minimum per-group weight share)
    # But milp minimizes, so we minimize -t

    # Actually, let's use a simpler LP relaxation approach.
    # Instead, use a different formulation:
    # Maximize minimum per-function total weight across all functions

    # Variables: x_0..x_{N-1} (binary), t (continuous, min function weight)
    n_vars = N + 1  # last var is t

    # Objective: minimize -t (i.e., maximize t)
    c = np.zeros(n_vars)
    c[-1] = -1.0  # minimize -t

    # Constraint 1: sum(x_i) = K
    A_sum = np.zeros((1, n_vars))
    A_sum[0, :N] = 1.0

    # Constraint 2: for each function f, sum(W[i,f] * x_i) >= t
    # i.e., -sum(W[i,f] * x_i) + t <= 0
    A_func = np.zeros((F, n_vars))
    for f_idx in range(F):
        A_func[f_idx, :N] = -W[:, f_idx]
        A_func[f_idx, -1] = 1.0  # +t

    # Stack constraints
    A = np.vstack([A_sum, A_func])

    # Bounds for constraints
    # sum(x_i) = K  →  K <= sum <= K
    # -sum(W*x) + t <= 0
    lb = np.concatenate([[K], -np.inf * np.ones(F)])
    ub = np.concatenate([[K], np.zeros(F)])

    constraints = LinearConstraint(A, lb, ub)

    # Variable bounds
    var_lb = np.concatenate([np.zeros(N), [0.0]])
    var_ub = np.concatenate([np.ones(N), [np.inf]])
    bounds = Bounds(var_lb, var_ub)

    # Integrality: x_i are binary (1), t is continuous (0)
    integrality = np.concatenate([np.ones(N), [0]])

    print("  Solving MIP (maximize minimum per-function weight)...")
    result = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)

    if result.success:
        x_sol = result.x[:N]
        ilp_indices = sorted(np.where(x_sol > 0.5)[0])
        print(f"  ILP solution: {ilp_indices}")
        print(f"  Min per-function weight (t): {-result.fun:.4f}")
        ilp_eval = evaluate(ilp_indices)
        print_eval("ILP Solution (max min-function-weight)", ilp_eval)
    else:
        print(f"  ILP failed: {result.message}")
        ilp_eval = None

except ImportError:
    print("  scipy not available, skipping ILP approach")
    ilp_eval = None


# ── Approach 4: ILP with group balance ───────────────────────────────────────
print("\n" + "~"*70)
print("  ILP APPROACH 2: Maximize min-function-weight + group balance")
print("~"*70)

try:
    # Variables: x_0..x_{N-1} (binary), t (min func weight), s (slack for group balance)
    # Minimize: -t + lambda * sum_of_group_deviations
    #
    # Alternative: add group balance constraints
    # For each group g: |group_share_g - 0.2| <= s
    # Minimize -t + alpha * s

    n_vars2 = N + 1 + 1  # x_i, t, s

    # Objective: minimize -t + 5*s
    c2 = np.zeros(n_vars2)
    c2[N] = -1.0    # -t
    c2[N+1] = 5.0   # +5*s (penalize group imbalance)

    constraints_list = []

    # Constraint: sum(x_i) = K
    A_sum2 = np.zeros((1, n_vars2))
    A_sum2[0, :N] = 1.0
    constraints_list.append(LinearConstraint(A_sum2, K, K))

    # Constraint: for each function f, sum(W[i,f] * x_i) >= t
    A_func2 = np.zeros((F, n_vars2))
    for f_idx in range(F):
        A_func2[f_idx, :N] = -W[:, f_idx]
        A_func2[f_idx, N] = 1.0
    constraints_list.append(LinearConstraint(A_func2, -np.inf, 0))

    # Group balance constraints:
    # For each group g with columns C_g:
    #   sum_{i} sum_{f in C_g} W[i,f] * x_i  ≈ 0.2 * sum_{i} sum_f W[i,f] * x_i
    # Linearized: group_weight_g - 0.2 * total_weight <= s * total_weight
    #            -group_weight_g + 0.2 * total_weight <= s * total_weight
    # But total_weight depends on x, making this nonlinear.
    #
    # Approximation: since each row sums to 1, total_weight = K = 10
    # So group_weight_g = sum_{i} sum_{f in C_g} W[i,f] * x_i
    # Target: group_weight_g = 0.2 * 10 = 2.0
    # |group_weight_g - 2.0| <= s

    for gname, cols in groups.items():
        # group_weight - 2.0 <= s  →  group_weight - s <= 2.0
        A_g_upper = np.zeros((1, n_vars2))
        A_g_upper[0, :N] = W[:, cols].sum(axis=1)
        A_g_upper[0, N+1] = -1.0
        constraints_list.append(LinearConstraint(A_g_upper, -np.inf, 2.0))

        # -(group_weight - 2.0) <= s  →  -group_weight - s <= -2.0
        A_g_lower = np.zeros((1, n_vars2))
        A_g_lower[0, :N] = -W[:, cols].sum(axis=1)
        A_g_lower[0, N+1] = -1.0
        constraints_list.append(LinearConstraint(A_g_lower, -np.inf, -2.0))

    # Variable bounds
    var_lb2 = np.concatenate([np.zeros(N), [0.0], [0.0]])
    var_ub2 = np.concatenate([np.ones(N), [np.inf], [np.inf]])
    bounds2 = Bounds(var_lb2, var_ub2)

    integrality2 = np.concatenate([np.ones(N), [0], [0]])

    print("  Solving MIP (max min-function-weight + group balance)...")
    result2 = milp(c2, constraints=constraints_list, integrality=integrality2, bounds=bounds2)

    if result2.success:
        x_sol2 = result2.x[:N]
        ilp2_indices = sorted(np.where(x_sol2 > 0.5)[0])
        print(f"  ILP2 solution: {ilp2_indices}")
        print(f"  t = {result2.x[N]:.4f}, s = {result2.x[N+1]:.4f}")
        ilp2_eval = evaluate(ilp2_indices)
        print_eval("ILP2 Solution (balanced)", ilp2_eval)
    else:
        print(f"  ILP2 failed: {result2.message}")
        ilp2_eval = None

except Exception as e:
    print(f"  ILP2 failed with error: {e}")
    ilp2_eval = None


# ── Baseline: first 10 instances (indices 0-9) ──────────────────────────────
baseline_eval = evaluate(list(range(10)))
print_eval("Baseline (indices 0-9)", baseline_eval)


# ── Final comparison ─────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  COMPARISON SUMMARY")
print("="*70)

results = [
    ("Baseline (0-9)", baseline_eval),
    ("Greedy", greedy_eval),
    ("Greedy+Swap", opt_eval),
]
if ilp_eval:
    results.append(("ILP (max-min)", ilp_eval))
if ilp2_eval:
    results.append(("ILP2 (balanced)", ilp2_eval))

print(f"\n  {'Method':<22s} {'Covered':>8s} {'Uniformity':>11s} {'Grp Std':>8s} {'Grp Balance':>11s}")
print(f"  {'-'*62}")
for name, ev in results:
    print(f"  {name:<22s} {ev['n_covered']:>5d}/24 {ev['func_uniformity']:>11.4f} {ev['group_shares_std']:>8.4f} {ev['group_balance']:>11.4f}")

# Pick the best overall
print("\n" + "="*70)
print("  RECOMMENDED SELECTION")
print("="*70)

# Pick the one with best coverage first, then best group balance
all_results = results[1:]  # exclude baseline
# Sort: first by coverage (desc), then group_shares_std (asc), then func_uniformity (desc)
all_results.sort(key=lambda x: (-x[1]['n_covered'], x[1]['group_shares_std'], -x[1]['func_uniformity']))
best_name, best_ev = all_results[0]

print(f"\n  Best method: {best_name}")
print(f"  Indices: {best_ev['indices']}")
print(f"  Coverage: {best_ev['n_covered']}/24 functions")
print(f"  Group balance std: {best_ev['group_shares_std']:.4f}")
print(f"  Function uniformity: {best_ev['func_uniformity']:.4f}")
