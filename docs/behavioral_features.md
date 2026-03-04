# Behavioral Features for Algorithm Trajectory Characterization

This document catalogs all behavioral features computed during Phase 1 experiments,
organized by category. Each feature includes its source, formula, computational cost,
and what it measures about algorithm behavior.

**Data format:** Each per-evaluation trajectory is a DataFrame with columns:
`evaluations`, `raw_y` (objective, lower=better), `x0`..`x{d-1}` (decision variables).
Typical trajectory: T=10,000 evaluations, d=5 dimensions, bounds [-5, 5].

---

## Existing Features (11 metrics)

Already implemented in `BLADE/iohblade/behaviour_metrics.py` by Niki van Stein (2025).

| Feature | What it measures | Cost |
|---------|-----------------|------|
| `avg_nearest_neighbor_distance` | Sequential novelty of evaluated points | O(T) |
| `dispersion` | Max gap in search space coverage (KD-tree) | O(T + S) |
| `avg_exploration_pct` | Chunk-based diversity vs random baseline | O(T*d) |
| `avg_distance_to_best` | How close evaluations stay to best-so-far | O(T*d) |
| `intensification_ratio` | Fraction of evaluations near final best | O(T*d) |
| `avg_exploitation_pct` | 100 - avg_exploration_pct | free |
| `average_convergence_rate` | Geometric mean of successive error ratios | O(T) |
| `avg_improvement` | Mean improvement magnitude on improving steps | O(T) |
| `success_rate` | Fraction of evaluations that improved best-so-far | O(T) |
| `longest_no_improvement_streak` | Longest consecutive plateau | O(T) |
| `last_improvement_fraction` | How far back the last improvement was | O(T) |

---

## Category 1: Step-Size & Movement Dynamics

Basic trajectory mechanics characterizing how the algorithm moves through search space.
These are standard trajectory analysis features, referenced across multiple works
including the meta-feature survey by Cenikj et al. (2024).

> Cenikj, G., Nikolikj, A., Petelin, G., van Stein, N., Doerr, C., & Eftimov, T.
> (2024). "A Survey of Meta-features Used for Automated Selection of Algorithms
> for Black-box Single-objective Continuous Optimization." arXiv:2406.06629.

### 1.1 `step_size_mean`

Mean Euclidean distance between consecutive evaluations.

    step_size_mean = mean(||x_{t+1} - x_t||)  for t = 1..T-1

- **Measures:** Average jump distance. Large = broad exploration. Small = local exploitation.
- **Input:** x-coordinates only
- **Cost:** O(T*d), ~0.1ms per trajectory

### 1.2 `step_size_std`

Standard deviation of consecutive step sizes.

    step_size_std = std(||x_{t+1} - x_t||)

- **Measures:** Variability in jumps. High = mixed exploration/exploitation phases.
  Low = consistent step size throughout.
- **Input:** x-coordinates only
- **Cost:** O(T*d), ~0.1ms per trajectory

### 1.3 `step_size_trend`

Slope of linear regression of step sizes over evaluation index.

    step_sizes = [||x_{t+1} - x_t|| for t = 1..T-1]
    step_size_trend = LinearRegression().fit(range(T-1), step_sizes).coef_

- **Measures:** Whether the algorithm contracts (negative slope) or expands (positive)
  over time. Negative = classic convergence behavior.
- **Input:** x-coordinates only
- **Cost:** O(T*d), ~0.2ms per trajectory

### 1.4 `directional_persistence`

Mean cosine similarity between consecutive displacement vectors.

    d_t = x_{t+1} - x_t
    directional_persistence = mean(cos(d_t, d_{t-1}))  for t = 2..T-1

    where cos(a, b) = (a . b) / (||a|| * ||b||)

- **Measures:** Whether the algorithm moves in a consistent direction (high, ~1) or
  zigzags/random-walks (low, ~0). Negative values indicate anti-persistent reversal.
- **Input:** x-coordinates only
- **Cost:** O(T*d), ~0.2ms per trajectory

**Category 1 total cost: ~0.6ms per trajectory, ~30ms per candidate (50 trajectories)**

---

## Category 2: Information-Theoretic Features

Time-series analysis features applied to the fitness sequence. These are established
in signal processing and physics but have **never been applied to algorithm fitness
trajectories on BBOB** (see novelty notes per feature).

### 2.1 `fitness_sample_entropy`

Sample Entropy (SampEn) of the raw_y fitness time series.

> Richman, J. S. & Moorman, J. R. (2000). "Physiological time-series analysis using
> approximate entropy and sample entropy." Am. J. Physiology -- Heart and Circulatory
> Physiology, 278(6), H2039-H2049.

    Given {u(1)..u(N)}, embedding dimension m, tolerance r:
    1. Form templates: x_m(i) = [u(i), u(i+1), ..., u(i+m-1)]
    2. B = count of pairs (i!=j) where max-norm d(x_m(i), x_m(j)) <= r
    3. A = count of pairs (i!=j) where max-norm d(x_{m+1}(i), x_{m+1}(j)) <= r
    4. SampEn = -ln(A / B)

- **Params:** m=2, r=0.2*std(subsampled raw_y) (antropy default)
- **Measures:** Regularity/predictability of fitness trajectory. Low = repetitive,
  self-similar convergence. High = complex, unpredictable dynamics.
- **Input:** raw_y only
- **Novelty:** Never applied to optimization fitness trajectories on BBOB.
- **Package:** `antropy.sample_entropy()`
- **Cost:** O(N^2 * m). With N=10000 this is ~10^8, taking 5-30s.
  **Mitigation:** Subsample every 10th point (N=1000) -> ~0.1-0.5s.

### 2.2 `fitness_permutation_entropy`

Normalized Permutation Entropy (PE) of the fitness time series.

> Bandt, C. & Pompe, B. (2002). "Permutation entropy: A natural complexity measure
> for time series." Physical Review Letters, 88(17), 174102.

    1. Form overlapping vectors of dimension D with delay tau:
       v(t) = [u(t), u(t+tau), ..., u(t+(D-1)*tau)]
    2. Map each v(t) to its ordinal pattern (ranking permutation)
    3. Compute relative frequency p(pi) of each of D! patterns
    4. PE = -sum(p(pi) * log2(p(pi))) / log2(D!)    (normalized to [0,1])

- **Params:** D=5, tau=1 (need N >> D!=120; we have T=10000)
- **Measures:** Ordinal complexity of fitness trajectory. Invariant to monotone
  transforms. PE~0 = perfectly monotone convergence. PE~1 = completely random.
- **Input:** raw_y only
- **Novelty:** Never applied to BBOB algorithm trajectories.
- **Package:** `antropy.perm_entropy()`
- **Cost:** O(T * D!), ~1ms per trajectory

### 2.3 `fitness_autocorrelation_lag1`

Lag-1 autocorrelation of the raw fitness time series.

> Weinberger, E. (1990). "Correlated and Uncorrelated Fitness Landscapes and How to
> Tell the Difference." Biological Cybernetics, 63, 325-336.

    rho(s) = Cov(f(t), f(t+s)) / Var(f(t))
    fitness_autocorrelation_lag1 = rho(1)

- **Measures:** Short-range fitness correlation. rho(1) near 1 = smooth, exploitative.
  Near 0 = uncorrelated. Negative = anti-persistent oscillation.
- **Input:** raw_y only
- **Novelty:** Autocorrelation has been used on SEPARATE random walks to characterize
  landscapes (Weinberger 1990; Munoz et al. 2015; flacco/pflacco). But has never been
  computed on the algorithm's OWN trajectory on BBOB.
- **Package:** numpy or statsmodels
- **Cost:** O(T), ~0.1ms per trajectory

### 2.4 `fitness_lempel_ziv_complexity`

Normalized Lempel-Ziv complexity of the symbolized fitness time series.

> Lempel, A. & Ziv, J. (1976). "On the Complexity of Finite Sequences."
> IEEE Trans. Information Theory, 22(1), 75-81.

    1. Binarize: s(t) = 1 if f(t+1) < f(t) else 0  (improvement vs not)
    2. LZ76 parsing: count distinct subpatterns c(N)
    3. Normalize: LZ_n = c(N) / (N / log2(N))

- **Measures:** Pattern novelty rate. LZ~1 = random (many new patterns).
  LZ~0 = regular (repetitive improvement/stagnation pattern).
- **Input:** raw_y only (after binarization)
- **Novelty:** LZC applied to PSO particle positions on simple benchmarks
  (Vantuch et al. 2019, Natural Computing), but never to fitness trajectories on BBOB.
- **Package:** `antropy.lziv_complexity()`
- **Cost:** O(T log T), ~10ms per trajectory

### 2.5 `fitness_hurst_exponent`

Hurst exponent estimated via Rescaled Range (R/S) analysis.

> Hurst, H. E. (1951). "Long-term storage capacity of reservoirs."
> Trans. ASCE, 116, 770-808.

    For subseries of length n:
    1. Y_t = X_t - mean(X)       (mean-adjusted)
    2. Z_t = cumsum(Y)            (cumulative deviation)
    3. R(n) = max(Z) - min(Z)     (range)
    4. S(n) = std(X)              (standard deviation)
    5. E[R(n)/S(n)] ~ C * n^H     (power law)
    6. H = slope of log(R/S) vs log(n)

- **Measures:** Long-range dependence. H=0.5 = random walk. H>0.5 = persistent
  (trends continue). H<0.5 = anti-persistent (mean-reverting).
- **Input:** raw_y only
- **Novelty:** Never applied to optimization trajectories on BBOB or CEC.
- **Package:** `nolds.hurst_rs()`
- **Cost:** O(T * log T), ~50ms per trajectory

### 2.6 `fitness_dfa_alpha`

Detrended Fluctuation Analysis scaling exponent.

> Peng, C.-K. et al. (1994). "Mosaic organization of DNA nucleotide sequences."
> Physical Review E, 49(2), 1685-1689.

    1. Y(i) = cumsum(x - mean(x))          (integrated profile)
    2. Divide Y into windows of size n
    3. In each window, fit linear trend, compute residuals
    4. F(n) = RMS of residuals, averaged over windows
    5. alpha = slope of log(F(n)) vs log(n)

- **Key advantage:** Handles non-stationary series (optimization trajectories
  trend downward, which breaks standard R/S analysis).
- **Measures:** Fractal scaling. alpha~0.5 = uncorrelated noise. alpha~1.0 = 1/f noise.
  alpha~1.5 = Brownian/smooth monotone convergence.
- **Input:** raw_y only
- **Novelty:** Never applied to BBOB or CEC optimization trajectories.
- **Package:** `nolds.dfa()`
- **Cost:** O(T * n_windows), ~50ms per trajectory

**Category 2 total cost (with SampEn subsampling): ~0.6s per trajectory, ~30s per candidate**

---

## Category 4: Adapted DynamoRep — Population Dynamics

Inspired by DynamoRep, adapted for (1+1)-ES where population size is 1.
Instead of per-generation population statistics, we compare early vs. late
search behavior using trajectory windows.

> Cenikj, G., Petelin, G., Doerr, C., Korosec, P. & Eftimov, T. (2023).
> "DynamoRep: Trajectory-Based Population Dynamics for Classification of
> Black-box Optimization Problems." GECCO 2023. arXiv:2306.05438.

### 4.1 `x_spread_early`

Mean per-dimension standard deviation of x-coordinates in the first 25% of evaluations.

    X_early = X[:T//4]
    x_spread_early = mean([std(X_early[:, j]) for j in range(d)])

- **Measures:** How broadly the algorithm explores in its early phase.
- **Cost:** O(T*d), ~0.1ms

### 4.2 `x_spread_late`

Same as above for the last 25% of evaluations.

    X_late = X[3*T//4:]
    x_spread_late = mean([std(X_late[:, j]) for j in range(d)])

- **Measures:** How focused (or not) the algorithm is in its late phase.
- **Cost:** O(T*d), ~0.1ms

### 4.3 `spread_ratio`

Ratio of late to early spread.

    spread_ratio = x_spread_late / x_spread_early

- **Measures:** Convergence behavior. <1 = algorithm narrows search (exploitation).
  >1 = algorithm diversifies over time (unusual). ~1 = no change in search breadth.
- **Cost:** free (derived)

### 4.4 `centroid_drift`

Euclidean distance between centroid of first 25% and last 25% of evaluated points.

    centroid_drift = ||mean(X_early) - mean(X_late)||

- **Measures:** How far the algorithm's search center moves over the run.
  Large drift = significant relocation (found a new region). Small = stayed local.
- **Cost:** O(T*d), ~0.1ms

### 4.5 `f_range_early` / `f_range_late` / `f_range_ratio`

Range (max-min) of fitness values in first/last 25%, and their ratio.

    f_range_early = max(y[:T//4]) - min(y[:T//4])
    f_range_late  = max(y[3*T//4:]) - min(y[3*T//4:])
    f_range_ratio = f_range_late / f_range_early

- **Measures:** Whether fitness variance shrinks (exploitation near optimum) or
  remains high (still sampling widely different fitness values).
- **Cost:** O(T), trivial

**Category 4 total cost: ~0.5ms per trajectory, ~25ms per candidate**

---

## Category 5: Novel Features (Not from Prior Work)

These are features we define specifically for this thesis. They capture aspects of
algorithm behavior not addressed by any existing metric in the BBOB literature.

### 5.1 `improvement_spatial_correlation`

Pearson correlation between step size and fitness improvement magnitude, computed
only on improving steps.

    improving = {t : f(t) < f_best(t-1)}
    steps = [||x_t - x_{t-1}|| for t in improving]
    improvements = [f_best(t-1) - f(t) for t in improving]
    improvement_spatial_correlation = Pearson(steps, improvements)

- **Measures:** Whether big jumps yield big improvements (positive = exploration is
  rewarded) or small steps yield improvements (negative = exploitation is rewarded).
  Near zero = no relationship between jump distance and improvement size.
- **Novelty:** No prior work correlates step size with improvement magnitude.
- **Cost:** O(T*d), ~0.2ms

### 5.2 `improvement_burstiness`

Coefficient of variation of inter-improvement intervals, inspired by burstiness
analysis in network science.

    improving_indices = [t for t in range(T) if f(t) < f_best(t-1)]
    intervals = diff(improving_indices)
    improvement_burstiness = std(intervals) / mean(intervals)

- **Measures:** Temporal clustering of improvements. High CV = improvements come in
  bursts (alternating productive and stagnant phases). Low CV = steady improvement
  rate (metronomic progress). CV=1 for Poisson-like random improvements.
- **Novelty:** Burstiness (Barabasi 2005, Nature 435) is studied in human dynamics
  and network science but has never been applied to characterize optimization
  algorithm improvement patterns.
- **Cost:** O(T), ~0.1ms

### 5.3 `dimension_convergence_heterogeneity`

Standard deviation across dimensions of per-dimension range shrinkage.

    For each dimension j:
        range_early_j = max(X[:T//4, j]) - min(X[:T//4, j])
        range_late_j  = max(X[3*T//4:, j]) - min(X[3*T//4:, j])
        shrinkage_j = range_late_j / range_early_j
    dimension_convergence_heterogeneity = std([shrinkage_j for j in range(d)])

- **Measures:** Whether all dimensions converge uniformly or some converge while
  others keep exploring. High = uneven convergence (algorithm "locks in" on some
  dimensions before others). Low = uniform convergence across all dimensions.
- **Novelty:** Per-dimension convergence analysis exists conceptually but this
  specific heterogeneity metric is new.
- **Cost:** O(T*d), ~0.2ms

### 5.4 `step_size_autocorrelation`

Lag-1 autocorrelation of the step size time series (not fitness).

    steps = [||x_{t+1} - x_t|| for t in range(T-1)]
    step_size_autocorrelation = autocorrelation(steps, lag=1)

- **Measures:** Search momentum. High = the algorithm takes similarly-sized steps
  consecutively (smooth acceleration or deceleration). Low = step sizes vary
  randomly (erratic search). Captures movement dynamics that fitness autocorrelation
  misses entirely.
- **Novelty:** Autocorrelation of fitness sequences is well-studied (Weinberger 1990),
  but autocorrelation of STEP SIZES has not been applied.
- **Cost:** O(T*d), ~0.2ms

### 5.5 `fitness_plateau_fraction`

Fraction of consecutive evaluations where the absolute fitness change is negligible.

    eps = 1e-8 * (max(y) - min(y))   # relative threshold
    plateau_steps = sum(|f(t+1) - f(t)| < eps for t in range(T-1))
    fitness_plateau_fraction = plateau_steps / (T - 1)

- **Measures:** How much time the algorithm spends on flat regions of the landscape.
  Different from stagnation (which measures best-so-far plateaus) — this measures
  the RAW fitness values, capturing whether the algorithm is sampling from a
  genuinely flat region vs. finding worse-but-different points.
- **Novelty:** Existing stagnation metrics use best-so-far. This uses raw evaluations.
- **Cost:** O(T), ~0.1ms

### 5.6 `half_convergence_time`

Fraction of the budget at which 50% of total best-so-far improvement is reached.

    bsf = cumulative_min(y)
    total_improvement = bsf[0] - bsf[-1]
    half_target = bsf[0] - 0.5 * total_improvement
    half_convergence_time = first t where bsf[t] <= half_target, normalized to [0, 1]

- **Measures:** How front-loaded vs. back-loaded convergence is. Low values (~0.1-0.2)
  = rapid early improvement followed by slow refinement. High values (~0.8-0.9) = slow
  start with late breakthroughs. Complements `average_convergence_rate` by capturing
  the temporal distribution of improvement rather than just the geometric mean.
- **Novelty:** Simple but not used as a behavioral feature in prior BBOB work.
- **Cost:** O(T), ~0.1ms

**Category 5 total cost: ~0.9ms per trajectory, ~45ms per candidate**

---

## Computational Cost Summary

Per trajectory (T=10,000 evaluations, d=5):

| Category | # Features | Time per trajectory | Time per candidate (x50) |
|----------|-----------|--------------------|-----------------------|
| Existing | 11 | ~1-2s | ~1-2min |
| 1: Step-Size Dynamics | 4 | ~0.6ms | ~30ms |
| 2: Info-Theoretic | 6 | ~0.6s (subsampled SampEn) | ~30s |
| 4: Adapted DynamoRep | 7 | ~0.5ms | ~25ms |
| 5: Novel Features | 6 | ~0.9ms | ~45ms |
| **Total** | **34** | **~1.6-2.6s** | **~1.5-2.5min** |

### Bottlenecks and mitigations

- **SampEn (Cat 2):** O(N^2). At N=10000 takes 5-30s. Subsample every 10th point
  (N=1000) to get ~0.1-0.5s with minimal information loss.

### Impact on experiment runtime

Current per-candidate cost: ~30-120s for MA-BBOB evaluation (running the algorithm
50 times). Adding 23 new features adds ~30-90s per candidate, dominated by
Category 2's SampEn subsampling.

With 100 candidates per run, 5 seeds, 10 models: overhead is modest (~1-2 min per
candidate). Categories 1, 4, and 5 are essentially free (<50ms). Category 2 is
~30s per candidate with subsampling.

---

## New Dependencies Required

| Package | Version | Features it provides | Size |
|---------|---------|---------------------|------|
| `antropy` | >=0.1.6 | SampEn, PE, LZC | ~50KB (pure Python/NumPy) |
| `nolds` | >=0.6.1 | Hurst exponent, DFA | ~30KB (pure Python/NumPy) |

Both are lightweight pure-Python packages with no dependencies beyond NumPy.

---

## References

### Category 1
- Cenikj et al. (2024). "A Survey of Meta-features Used for Automated Selection of
  Algorithms for Black-box Single-objective Continuous Optimization." arXiv:2406.06629.

### Category 2
- Richman & Moorman (2000). "Physiological time-series analysis using approximate
  entropy and sample entropy." Am. J. Physiology, 278(6), H2039-H2049.
- Bandt & Pompe (2002). "Permutation entropy: A natural complexity measure for time
  series." Physical Review Letters, 88(17), 174102.
- Weinberger (1990). "Correlated and Uncorrelated Fitness Landscapes and How to Tell
  the Difference." Biological Cybernetics, 63, 325-336.
- Lempel & Ziv (1976). "On the Complexity of Finite Sequences." IEEE Trans.
  Information Theory, 22(1), 75-81.
- Hurst (1951). "Long-term storage capacity of reservoirs." Trans. ASCE, 116, 770-808.
- Peng et al. (1994). "Mosaic organization of DNA nucleotide sequences." Physical
  Review E, 49(2), 1685-1689.

### Category 4
- Cenikj, Petelin, Doerr, Korosec & Eftimov (2023). "DynamoRep: Trajectory-Based
  Population Dynamics for Classification of Black-box Optimization Problems."
  GECCO 2023. arXiv:2306.05438.

### Category 5
- Barabasi, A.-L. (2005). "The origin of bursts and heavy tails in human dynamics."
  Nature, 435, 207-211. (Inspiration for improvement_burstiness)

### Related work on trajectory features for BBOB

The following papers use trajectory-based features on BBOB benchmarks. We reviewed
each for additional hand-crafted features to adopt; in all cases the features are
either already captured by the categories above, not applicable to our (1+1)-ES
setting, or better suited as post-hoc analysis tools.

- **Kostovska et al. (2022).** "Per-Run Algorithm Selection with Warm-Starting Using
  Trajectory-Based Features." PPSN 2022. arXiv:2204.09483.
  - *What it does:* Combines ELA landscape features with CMA-ES internal state
    variables (step-size sigma, condition number of covariance matrix) to predict
    algorithm performance mid-run for warm-started algorithm selection.
  - *Why not adopted:* The novel features are CMA-ES internals (sigma, eigenvalues)
    that only exist for CMA-ES variants. The ELA features they use are standard
    pflacco features computed on the trajectory — we considered these (Category 3)
    but dropped them as they characterize the landscape-as-seen rather than algorithm
    behavior, and add significant computational overhead.

- **Renau & Hart (2024).** "On the Utility of Probing Trajectories for Algorithm
  Selection." EvoApplications 2024. arXiv:2401.12745.
  - *What it does:* Uses raw algorithm trajectories (sequences of fitness values from
    short "probing" runs) as input to deep learning models for algorithm selection.
    Compares feeding raw trajectories to neural networks vs. hand-crafted ELA features.
  - *Why not adopted:* This paper deliberately avoids hand-crafted features — the
    contribution is showing that raw trajectories fed to LSTMs/CNNs can replace them.
    No new scalar metrics are defined that we could incorporate.

- **Stein, van Stein, Back & Ochoa (2025).** "Behaviour Space Analysis of LLM-Driven
  Meta-Heuristic Discovery." arXiv:2507.03605.
  - *What it does:* Directly relevant prior work. Uses LLaMEA to evolve optimization
    algorithms on BBOB, then characterizes discovered algorithms using BLADE's
    existing 11 behavioral metrics (the same ones in our "Existing Features" section).
    Applies MAP-Elites-style behavior space visualization.
  - *Why not adopted:* The behavioral features used are exactly the 11 metrics already
    in BLADE that we inherit. No additional metrics are defined. This paper validates
    our baseline feature set and motivates extending it (which is what Categories 1-5
    do). **Recommended reading** as the most closely related work to this thesis.

- **Vermetten et al. (2023).** "To Switch or Not to Switch: Predicting the Benefit of
  Switching Between Algorithms Based on Trajectory Features." EvoApplications 2023.
  - *What it does:* Computes ELA features in a sliding window over algorithm
    trajectories to detect when the landscape-as-seen changes, triggering algorithm
    switching decisions. Features include ela_meta, ela_distr, information content,
    and nearest-better clustering, all from pflacco.
  - *Why not adopted:* All features are standard ELA features applied in temporal
    windows — same Category 3 features we dropped. The sliding-window approach is
    interesting but adds complexity without clear benefit for characterizing overall
    algorithm behavior (it targets online switching decisions, not offline profiling).

- **Ochoa, Malan & Blum (2021).** "Search trajectory networks: A tool for analysing
  and visualising the behaviour of metaheuristics." Applied Soft Computing, 109,
  107492.
  - *What it does:* Introduces Search Trajectory Networks (STNs): discretize the
    search space into a grid, map algorithm trajectories to graph nodes/edges, then
    compute graph metrics (clustering coefficient, path length, connected components)
    and produce visualizations.
  - *Why not adopted:* STNs are primarily a visualization tool. The graph metrics
    require discretizing continuous search space into a grid, which scales poorly
    to d=5 dimensions (grid cells grow as O(bins^d)). The reference implementation
    is R-only (igraph). STNs have never been used for algorithm selection or automated
    profiling. We plan to use the stn-analytics.com web tool for post-hoc
    visualization of selected algorithms, but not as computed features during
    experiments.
