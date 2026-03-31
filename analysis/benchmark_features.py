"""Benchmark per-category feature extraction time on a synthetic trajectory."""

import sys, time
sys.path.insert(0, "/Users/maxharell/repos/thesis")
sys.path.insert(0, "/Users/maxharell/repos/thesis/BLADE")

import numpy as np
import pandas as pd
from iohblade.behaviour_metrics import *

# --- Create a realistic synthetic trajectory ---
# Matches real structure: T=10,000, d=5, bounds [-5, 5]
# Simulates a simple (1+1)-ES with Gaussian mutation + selection
T = 10_000
d = 5
rng = np.random.default_rng(42)

x = np.zeros(d)
positions = [x.copy()]
fitnesses = []

# Simple sphere function for realistic fitness values
for t in range(T):
    if t == 0:
        fitnesses.append(float(np.sum(x**2)))
    else:
        x_new = x + rng.normal(0, 0.5, d)
        x_new = np.clip(x_new, -5, 5)
        f_new = float(np.sum(x_new**2)) + rng.normal(0, 0.1)
        if f_new < fitnesses[-1]:
            x = x_new
        positions.append(x.copy())
        fitnesses.append(f_new if f_new < fitnesses[-1] else fitnesses[-1] + rng.normal(0, 0.01))

# Fix: generate proper trajectory with raw_y (not best-so-far)
positions_arr = np.array(positions[:T])
df = pd.DataFrame({
    "evaluations": np.arange(T),
    "raw_y": fitnesses[:T],
    **{f"x{j}": positions_arr[:, j] for j in range(d)},
})

bounds = [(-5.0, 5.0)] * d
radius = 0.1 * 10  # 0.1 * (ub - lb)

N_RUNS = 50
print(f"Benchmarking {N_RUNS} runs on synthetic trajectory (T={T}, d={d})\n")

# --- Define feature groups matching the thesis categories ---
categories = {
    "Existing (BLADE)": lambda: {
        "avg_nearest_neighbor_distance": average_nearest_neighbor_distance(df),
        "dispersion": coverage_dispersion(df, bounds, 10_000),
        "avg_exploration_pct": avg_exploration_exploitation_chunked(df)[0],
        "avg_exploitation_pct": avg_exploration_exploitation_chunked(df)[1],
        "avg_distance_to_best": average_distance_to_best_so_far(df),
        "intensification_ratio": intensification_ratio(df, radius),
        "average_convergence_rate": average_convergence_rate(df),
        "avg_improvement": improvement_statistics(df)[0],
        "success_rate": improvement_statistics(df)[1],
        "longest_no_improvement_streak": longest_no_improvement_streak(df),
        "last_improvement_fraction": last_improvement_fraction(df),
    },
    "Step-size dynamics": lambda: {
        "step_size_mean": step_size_mean(df),
        "step_size_std": step_size_std(df),
        "step_size_trend": step_size_trend(df),
        "directional_persistence": directional_persistence(df),
    },
    "Information-theoretic": lambda: {
        "fitness_sample_entropy": fitness_sample_entropy(df),
        "fitness_permutation_entropy": fitness_permutation_entropy(df),
        "fitness_autocorrelation": fitness_autocorrelation(df),
        "fitness_lempel_ziv_complexity": fitness_lempel_ziv_complexity(df),
    },
    "Adapted pop. dynamics": lambda: {
        "x_spread_early": x_spread_early(df),
        "x_spread_late": x_spread_late(df),
        "spread_ratio": spread_ratio(df),
        "centroid_drift": centroid_drift(df),
        "f_range_early": f_range_early(df),
        "f_range_late": f_range_late(df),
        "f_range_ratio": f_range_ratio(df),
    },
    "Novel features": lambda: {
        "improvement_spatial_correlation": improvement_spatial_correlation(df),
        "improvement_burstiness": improvement_burstiness(df),
        "dimension_convergence_heterogeneity": dimension_convergence_heterogeneity(df),
        "step_size_autocorrelation": step_size_autocorrelation(df),
        "fitness_plateau_fraction": fitness_plateau_fraction(df),
        "half_convergence_time": half_convergence_time(df),
    },
}

# --- Warmup run ---
print("Warmup run...")
for name, func in categories.items():
    func()
print()

# --- Timed runs ---
results = {name: [] for name in categories}

for i in range(N_RUNS):
    for name, func in categories.items():
        t0 = time.perf_counter()
        func()
        elapsed = time.perf_counter() - t0
        results[name].append(elapsed)

# --- Report ---
print(f"{'Category':<28} {'#':>3} {'Mean (ms)':>10} {'Std (ms)':>10} {'Per-cand':>12}")
print("-" * 70)
total_mean = 0
total_count = 0
for name, timings in results.items():
    arr = np.array(timings) * 1000  # to ms
    n_features = len(categories[name]())
    per_cand = arr.mean() * 50 / 1000  # 50 trajectories per candidate, convert to seconds
    total_mean += arr.mean()
    total_count += n_features
    print(f"{name:<28} {n_features:>3} {arr.mean():>10.2f} {arr.std():>10.2f} {per_cand:>10.2f} s")

per_cand_total = total_mean * 50 / 1000
print("-" * 70)
print(f"{'Total':<28} {total_count:>3} {total_mean:>10.2f} {'':>10} {per_cand_total:>10.2f} s")
