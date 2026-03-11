# Selected Behavioral Features for Phase 1 Screening

10 features selected via greedy Borda-count ranking with correlation-based de-duplication (|ρ| > 0.8 threshold). Features are ranked by consensus across three criteria: Spearman correlation with AOCC, Random Forest permutation importance, and KS distributional effect size.

**Source code**: `BLADE/iohblade/behaviour_metrics.py`
**Selection notebook**: `analysis/phase1_behavior_analysis.ipynb` (Cell 11)

---

## 1. `avg_improvement` (Borda: 3)

**Category**: Convergence · **Source**: BLADE (Niki van Stein) · **ρ = −0.782**

Mean improvement magnitude on steps that improve the best-so-far fitness.

**Calculation** (`improvement_statistics`, line 235):
- Walk the raw objective sequence `y[0], y[1], ...`
- Track best-so-far; when `y[t] < best`, record improvement `best − y[t]`
- Return mean of recorded improvements

**Interpretation**: Higher values mean the algorithm makes larger jumps when it does improve. Negatively correlated with AOCC because in BLADE, fitness is *negated* AOCC (lower raw_y = better), so large improvements in raw_y correspond to algorithms that are still far from optimal — good algorithms make small, precise refinements.

**Rejected redundant feature**: `success_rate` (|ρ| = 0.917 with `avg_improvement`)

---

## 2. `intensification_ratio` (Borda: 13)

**Category**: Exploitation · **Source**: BLADE (Niki van Stein) · **ρ = +0.725**

Fraction of all evaluations lying within a radius of the final best point.

**Calculation** (`intensification_ratio`, line 202):
```
radius = 0.1 × (upper_bound − lower_bound)    # default: 0.1 × 10 = 1.0
ratio  = count(||x_t − x_best|| < radius) / N
```

**Interpretation**: Measures how much the search concentrates near the best solution. High values indicate strong exploitation / local search behavior. Positively correlated with AOCC — better algorithms spend more time refining near the optimum.

**Rejected redundant features**: `avg_distance_to_best` (|ρ| = 0.961), `step_size_mean` (|ρ| = 0.829), `avg_exploitation_pct` (|ρ| = 0.869), `avg_exploration_pct` (|ρ| = 0.869)

---

## 3. `fitness_plateau_fraction` (Borda: 15)

**Category**: Novel Features · **Source**: This thesis · **ρ = +0.685**

Fraction of consecutive evaluations where fitness barely changes.

**Calculation** (`fitness_plateau_fraction`, line 557):
```
eps = 1e-8 × range(y)
plateau_frac = mean(|y[t+1] − y[t]| < eps)
```

**Interpretation**: Measures how much of the search traverses flat fitness regions. Uses raw fitness (not best-so-far), so it captures the actual landscape structure the algorithm encounters. Positively correlated with AOCC — better algorithms that exploit near the optimum naturally see small fitness changes, while worse algorithms jumping around see large changes.

**Rejected redundant feature**: `fitness_sample_entropy` (|ρ| = 0.873)

---

## 4. `step_size_autocorrelation` (Borda: 18)

**Category**: Novel Features · **Source**: This thesis · **ρ = +0.766**

Lag-1 autocorrelation of consecutive step sizes.

**Calculation** (`step_size_autocorrelation`, line 544):
```
steps[t] = ||x[t+1] − x[t]||
s_centered = steps − mean(steps)
autocorr = dot(s_centered[:-1], s_centered[1:]) / dot(s_centered, s_centered)
```

**Interpretation**: Captures search momentum — whether large steps tend to follow large steps (positive autocorrelation) or alternate with small steps (negative). Positive correlation with AOCC suggests that good algorithms have consistent step-size behavior (e.g., a CMA-ES-like mechanism that smoothly adapts step size) rather than erratic jumps.

---

## 5. `improvement_spatial_correlation` (Borda: 28)

**Category**: Novel Features · **Source**: This thesis · **ρ = +0.757**

Pearson correlation between step size and improvement magnitude, computed only on improving steps.

**Calculation** (`improvement_spatial_correlation`, line 489):
- Identify improving steps: where `y[t] < best_so_far[t-1]`
- For those steps, compute step size `||x[t] − x[t-1]||` and improvement `best_so_far[t-1] − y[t]`
- Return Pearson correlation between the two vectors

**Interpretation**: Positive values mean "big jumps yield big improvements" — the algorithm explores effectively and finds distant good solutions. Near-zero means improvement magnitude is independent of distance traveled. This separates algorithms that improve through directed long-range moves from those that improve through local refinement.

---

## 6. `half_convergence_time` (Borda: 37)

**Category**: Novel Features · **Source**: This thesis · **ρ = −0.597**

Normalized budget fraction at which best-so-far reaches 50% of its total improvement.

**Calculation** (`half_convergence_time`, line 570):
```
total_improvement = best_so_far[0] − best_so_far[-1]
target = best_so_far[0] − 0.5 × total_improvement
half_time = first index where best_so_far ≤ target / (N − 1)
```

**Interpretation**: Low values indicate fast early convergence (the algorithm finds good solutions quickly then refines). High values mean slow or late convergence. Negatively correlated with AOCC — better algorithms converge faster. Returns 1.0 if no improvement occurs.

---

## 7. `fitness_autocorrelation` (Borda: 38)

**Category**: Information-Theoretic · **Source**: This thesis · **ρ = +0.645**

Lag-1 autocorrelation of the raw fitness sequence.

**Calculation** (`fitness_autocorrelation`, line 367):
```
y_centered = y − mean(y)
autocorr = dot(y_centered[:-1], y_centered[1:]) / dot(y_centered, y_centered)
```

Based on Weinberger (1990) autocorrelation for fitness landscape analysis.

**Interpretation**: High positive autocorrelation means the fitness values change smoothly between consecutive evaluations — the algorithm samples nearby points with similar fitness (local search). Low or negative autocorrelation means the algorithm jumps between very different fitness regions. Better algorithms tend to have higher autocorrelation, indicating structured, local search behavior.

---

## 8. `x_spread_early` (Borda: 40)

**Category**: Early/Late Dynamics · **Source**: This thesis (adapted from DynamoRep, Cenikj et al. 2023) · **ρ = −0.599**

Mean per-dimension standard deviation of x-coordinates in the first 25% of evaluations.

**Calculation** (`x_spread_early`, line 431):
```
cutoff = N // 4
spread = mean over dimensions j of: std(X[:cutoff, j])
```

**Interpretation**: Measures how spread out the initial search is across the decision space. Negatively correlated with AOCC — algorithms that start with a more focused search (lower spread) tend to perform better. This may reflect that the shared initial population (RandomSearch + SimpleES) provides a decent starting region, and algorithms that immediately narrow down outperform those that keep exploring broadly.

**Rejected redundant feature**: `avg_nearest_neighbor_distance` (|ρ| = 0.820)

---

## 9. `longest_no_improvement_streak` (Borda: 43)

**Category**: Stagnation · **Source**: BLADE (Niki van Stein) · **ρ = −0.469**

Length of the longest consecutive sequence of evaluations without improving best-so-far.

**Calculation** (`longest_no_improvement_streak`, line 260):
- Walk the objective sequence, tracking best-so-far
- Count consecutive evaluations where `y[t] ≥ best`
- Return the maximum such count

**Interpretation**: Directly measures stagnation. Longer streaks mean the algorithm gets stuck. Negatively correlated with AOCC — algorithms with shorter stagnation periods achieve better final performance. With a 10,000-evaluation budget per trajectory, a streak of thousands indicates the algorithm has effectively stopped improving.

---

## 10. `dimension_convergence_heterogeneity` (Borda: 55)

**Category**: Novel Features · **Source**: This thesis · **ρ = +0.471**

Standard deviation across dimensions of per-dimension range shrinkage ratios.

**Calculation** (`dimension_convergence_heterogeneity`, line 528):
```
For each dimension j:
    shrinkage[j] = range(X[-N//4:, j]) / range(X[:N//4, j])
heterogeneity = std(shrinkage)
```

**Interpretation**: Measures whether the algorithm converges evenly across all dimensions or unevenly (some dimensions converge tightly while others remain spread). Positively correlated with AOCC — better algorithms exhibit *more* heterogeneous convergence, suggesting they identify and exploit separability structure in the problem (converging fast on easy dimensions while continuing to search hard ones).

---

## Summary

| # | Feature | Category | Source | ρ with AOCC |
|---|---------|----------|--------|-------------|
| 1 | `avg_improvement` | Convergence | BLADE | −0.782 |
| 2 | `intensification_ratio` | Exploitation | BLADE | +0.725 |
| 3 | `fitness_plateau_fraction` | Novel | Thesis | +0.685 |
| 4 | `step_size_autocorrelation` | Novel | Thesis | +0.766 |
| 5 | `improvement_spatial_correlation` | Novel | Thesis | +0.757 |
| 6 | `half_convergence_time` | Novel | Thesis | −0.597 |
| 7 | `fitness_autocorrelation` | Info-Theoretic | Thesis | +0.645 |
| 8 | `x_spread_early` | Early/Late | Thesis | −0.599 |
| 9 | `longest_no_improvement_streak` | Stagnation | BLADE | −0.469 |
| 10 | `dimension_convergence_heterogeneity` | Novel | Thesis | +0.471 |

**Attribution**: 3 features from BLADE (Niki van Stein), 7 new features from this thesis.

**Redundant features removed**: `success_rate`, `fitness_sample_entropy`, `avg_distance_to_best`, `step_size_mean`, `avg_exploitation_pct`, `avg_exploration_pct`, `avg_nearest_neighbor_distance` — all had |ρ| > 0.8 with a higher-ranked selected feature.

---

## Value Ranges

Statistics from 3,071 valid (non-failed) candidates across all 10 models and 5 seeds. Note: `x_spread_early` and `dimension_convergence_heterogeneity` have extreme outliers from degenerate algorithms with exploding coordinates; the 99th percentile is shown as a practical upper bound.

### Overall Distribution

| Feature | Min | 25th | Median | 75th | Max | Unit |
|---------|-----|------|--------|------|-----|------|
| `avg_improvement` | 0.00 | 0.11 | 0.24 | 1.17 | 209.3 | fitness units |
| `intensification_ratio` | 0.0001 | 0.016 | 0.69 | 0.85 | 1.0 | fraction [0, 1] |
| `fitness_plateau_fraction` | 0.0 | 0.0 | 0.15 | 0.52 | 1.0 | fraction [0, 1] |
| `step_size_autocorrelation` | −0.81 | 0.30 | 0.84 | 0.89 | 1.0 | correlation [−1, 1] |
| `improvement_spatial_correlation` | −0.39 | 0.18 | 0.58 | 0.66 | 0.94 | correlation [−1, 1] |
| `half_convergence_time` | 0.0002 | 0.0014 | 0.0024 | 0.005 | 1.0 | fraction of budget [0, 1] |
| `fitness_autocorrelation` | −0.86 | 0.04 | 0.66 | 0.74 | 1.0 | correlation [−1, 1] |
| `x_spread_early` | 0.0 | 0.78 | 1.59 | 2.87 | ~3.5* | std per dimension |
| `longest_no_improvement_streak` | 1 | 1,859 | 5,171 | 6,113 | 9,999 | evaluations (budget=10,000) |
| `dimension_convergence_heterogeneity` | 0.0 | 0.0006 | 0.013 | 0.077 | ~0.25* | std of shrinkage ratios |

*\*Practical upper bound (99th percentile); extreme outliers from degenerate algorithms excluded.*

### Top 25% vs Bottom 25% AOCC

Typical values for high-performing (AOCC ≥ 0.73) vs low-performing (AOCC ≤ 0.23) algorithms:

| Feature | Top 25% (median) | Bottom 25% (median) | Direction |
|---------|-------------------|----------------------|-----------|
| `avg_improvement` | 0.10 | 1.75 | lower is better |
| `intensification_ratio` | 0.87 | 0.0002 | higher is better |
| `fitness_plateau_fraction` | 0.57 | 0.0 | higher is better |
| `step_size_autocorrelation` | 0.90 | 0.13 | higher is better |
| `improvement_spatial_correlation` | 0.67 | 0.06 | higher is better |
| `half_convergence_time` | 0.0013 | 0.0065 | lower is better |
| `fitness_autocorrelation` | 0.76 | 0.003 | higher is better |
| `x_spread_early` | 0.74 | 2.89 | lower is better |
| `longest_no_improvement_streak` | 5,170 | 6,264 | lower is better |
| `dimension_convergence_heterogeneity` | 0.072 | 0.0006 | higher is better |
