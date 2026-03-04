# Feature-Selection Experiment: Results Write-Up

## 1. Experiment Overview

The feature-selection experiment ran 12 conditions on two LIACS compute servers (vibranium, duranium). Each condition used LLaMEA with a (1+1)-ES strategy and Qwen3 8B (via Ollama) to generate 100 candidate optimization algorithms, each evaluated on 50 runs (10 MA-BBOB instances x 5 seeds). The only variable between conditions was the feedback string: vanilla received AOCC only, while each single-feature condition received AOCC plus one behavioral metric value.

All 12 conditions completed their 100 evaluations successfully.

## 2. Failure Rate Analysis

### 2.1 Overall

Of the 1,200 total candidates generated across all conditions, **922 (76.8%) failed** to produce a valid fitness score. Only 278 candidates (23.2%) executed successfully.

| Condition | Total | Valid | Failed | Failure Rate |
|-----------|-------|-------|--------|-------------|
| average_convergence_rate | 100 | 50 | 50 | 50% |
| avg_exploration_pct | 100 | 45 | 55 | 55% |
| avg_improvement | 100 | 37 | 63 | 63% |
| vanilla | 100 | 33 | 67 | 67% |
| longest_no_improvement_streak | 100 | 32 | 68 | 68% |
| dispersion | 100 | 26 | 74 | 74% |
| avg_exploitation_pct | 100 | 15 | 85 | 85% |
| avg_nearest_neighbor_distance | 100 | 14 | 86 | 86% |
| last_improvement_fraction | 100 | 13 | 87 | 87% |
| avg_distance_to_best | 100 | 6 | 94 | 94% |
| intensification_ratio | 100 | 6 | 94 | 94% |
| success_rate | 100 | 1 | 99 | 99% |

### 2.2 Root Causes

The 922 failures fall into three root cause categories:

| Root Cause | Count | % of Failures |
|-----------|-------|--------------|
| Interface mismatch (wrong `__init__`/`__call__` signature) | 459 | 49.8% |
| Code generation failure (no valid Python class produced) | 286 | 31.0% |
| Runtime errors (correct structure, bugs in logic) | 177 | 19.2% |

### 2.3 Detailed Breakdown

**Interface mismatch (459 failures, 49.8%)** — The LLM produced syntactically valid classes, but with the wrong constructor or call signature. LLaMEA instantiates candidates as `ClassName(budget=100, dim=5)` and calls them as `instance(func)`. The three sub-types:

| Sub-type | Count | Description |
|----------|-------|-------------|
| `__init__` rejects `budget` kwarg | 303 | The LLM defined `__init__(self)` or `__init__(self, other_args)` without accepting `budget` as a keyword argument. This is the single largest failure mode. |
| `__call__` expects `bounds` arg | 107 | The LLM wrote `__call__(self, func, bounds)` instead of `__call__(self, func)`. The bounds should be accessed via `func.bounds.lb`/`func.bounds.ub`. |
| `__init__` rejects `dim` kwarg | 31 | Similar to the `budget` issue but for the `dim` parameter. |
| Other arg mismatches | 18 | Miscellaneous signature problems. |

**Code generation failure (286 failures, 31.0%)** — The LLM failed to produce a Python class at all:

| Sub-type | Count | Description |
|----------|-------|-------------|
| Wrote standalone function instead of class | 224 | The LLM produced `def optimize(...)` rather than wrapping it in a class with `__init__` and `__call__`. LLaMEA's regex `r"(?:def\|class)\s*(\w*).*\:"` found a function name but no class, and the evaluation framework could not instantiate it. |
| Produced pseudocode or text | 43 | The LLM output natural language descriptions, pseudocode, or algorithm outlines instead of executable Python. |
| No code block found | 18 | The LLM's response did not contain a markdown code block (`` ```python ... ``` ``), so the extraction regex found nothing. |
| Class present but name extraction failed | 1 | Edge case. |

**Runtime errors (177 failures, 19.2%)** — The LLM produced a correctly structured class, but the algorithm logic contained bugs:

| Sub-type | Count | Description |
|----------|-------|-------------|
| NameError (undefined variable) | 24 | Referenced variables not defined in scope (e.g., `initial_exploration`). |
| AttributeError (missing attribute) | 20 | Accessed `self.attr` that was never set in `__init__`. |
| `object is not callable` | 20 | Tried to call something that isn't a function (e.g., calling bounds as `bounds()` instead of indexing). |
| `RealBounds not subscriptable` | 16 | Used `func.bounds[0]` instead of `func.bounds.lb` — misunderstood the IOH bounds API. |
| `a must be 1-dimensional` / shape errors | 28 | NumPy array shape mismatches, typically from incorrect use of `np.random.choice` or matrix operations. |
| `Truth value ambiguous` | 17 | Used `if array:` instead of `if array.any():` — standard NumPy pitfall. |
| Syntax/indentation error | 3 | Minor formatting issues in generated code. |
| Other (ZeroDivision, overflow, convergence, etc.) | 49 | Miscellaneous logic bugs. |

### 2.4 Failure Mode by Condition

The `success_rate` condition is a notable outlier: 85 of its 99 failures were `__call__` signature mismatches (`missing 1 required positional argument: 'bounds'`). The LLM appears to have gotten stuck in a loop generating the same wrong pattern (AdaptiveDirectionalSearch/AdaptiveDirectionalRandomSearch with `__call__(self, func, bounds)`), never escaping because the (1+1)-ES retained the one valid candidate and kept mutating from it — but the LLM kept reproducing the same structural mistake.

Several conditions show elevated `no_name` (wrote function instead of class) rates: `avg_distance_to_best` (67), `intensification_ratio` (53), `last_improvement_fraction` (39), `avg_exploitation_pct` (38). These are also among the highest-failure-rate conditions.

### 2.5 Implications

The dominant failure mode (49.8% of all failures) is **interface mismatch** — the LLM generated valid Python classes but with the wrong `__init__` or `__call__` signature. This is not a fundamental code generation problem; the LLM understood the algorithmic task but failed to comply with the required API. The task prompt and example provided by LLaMEA clearly specify `__init__(self, budget, dim)` and `__call__(self, func)`, and an example RandomSearch class demonstrates the correct pattern. Qwen3 8B appears to frequently deviate from this template despite the instructions.

The second largest category (31.0%) — the LLM not producing a class at all — suggests that Qwen3 8B sometimes fails to maintain the output format across 100 generations of evolutionary prompting. This may worsen as the conversation grows longer.

## 3. Performance Results

### 3.1 Final Best AOCC

Each condition's best-so-far AOCC after 100 evaluations:

| Rank | Condition | Final Best AOCC | Delta vs Vanilla | % Change |
|------|-----------|----------------|-----------------|----------|
| 1 | avg_improvement | 0.705 | +0.289 | +69.5% |
| 2 | avg_exploitation_pct | 0.660 | +0.244 | +58.7% |
| 3 | longest_no_improvement_streak | 0.581 | +0.165 | +39.6% |
| 4 | last_improvement_fraction | 0.518 | +0.102 | +24.5% |
| 5 | dispersion | 0.476 | +0.060 | +14.5% |
| 6 | avg_exploration_pct | 0.429 | +0.013 | +3.1% |
| 7 | **vanilla (baseline)** | **0.416** | **—** | **—** |
| 8 | average_convergence_rate | 0.344 | -0.072 | -17.4% |
| 9 | avg_nearest_neighbor_distance | 0.340 | -0.076 | -18.2% |
| 10 | success_rate | 0.285 | -0.131 | -31.6% |
| 11 | intensification_ratio | 0.218 | -0.198 | -47.6% |
| 12 | avg_distance_to_best | 0.217 | -0.199 | -47.9% |

Six conditions produced a better final algorithm than vanilla; five produced worse.

### 3.2 Convergence Behavior

The best-so-far convergence curves show distinct patterns:

- **avg_improvement** and **avg_exploitation_pct** found strong candidates early (around evaluation 20-30) and continued improving throughout the budget, reaching 0.66-0.70.
- **vanilla** shows a characteristic staircase with a large jump around evaluation 35, then plateaus at ~0.42.
- **no_improv** and **last_improv** (Stagnation & Reliability category) showed late improvements, with major jumps after evaluation 40-50, suggesting these metrics may take longer to guide the search.
- **dist_best** and **intens_ratio** flatlined early at ~0.22, suggesting these metrics may have confused rather than helped the LLM.
- **success_rate** found its single valid candidate early and never improved (99% failure rate).

When grouped by category:
- **Convergence Progress**: avg_improvement strongly outperforms vanilla; conv_rate underperforms; success_rate is essentially non-functional.
- **Stagnation & Reliability**: Both no_improv and last_improv outperform vanilla, with late-stage improvements.
- **Exploration & Diversity**: explore_% and dispersion outperform vanilla; nn_dist slightly underperforms.
- **Exploitation & Intensification**: exploit_% strongly outperforms; dist_best and intens_ratio strongly underperform.

### 3.3 Candidate-Level Distribution

The boxplot of valid candidate AOCC distributions shows:
- **exploit_%** has the highest median and widest interquartile range among conditions that beat vanilla (median ~0.32, IQR 0.20-0.55), but with only 15 valid candidates.
- **explore_%** has the tightest distribution around a moderate mean (n=45 valid, median ~0.30).
- **conv_rate** has 50 valid candidates but a low median (~0.14) — many candidates, mostly poor.
- Conditions with very few valid candidates (dist_best n=6, intens_ratio n=6, success_% n=1) have distributions that are not meaningful to compare.

### 3.4 Statistical Tests

**Mann-Whitney U** (comparing all valid candidates of each condition vs vanilla's 33 valid candidates):

| Condition | n | Mean Diff | p-value | Cliff's d | Effect |
|-----------|---|-----------|---------|-----------|--------|
| avg_exploitation_pct | 15 | +0.134 | 0.008 | +0.49 | large |
| avg_improvement | 37 | +0.094 | 0.006 | +0.38 | medium |
| avg_exploration_pct | 45 | +0.060 | 0.003 | +0.40 | medium |
| dispersion | 26 | +0.056 | 0.007 | +0.41 | medium |
| longest_no_improvement_streak | 32 | +0.023 | 0.778 | +0.04 | negligible |
| last_improvement_fraction | 13 | +0.014 | 0.826 | +0.04 | negligible |
| avg_nearest_neighbor_distance | 14 | -0.017 | 0.422 | -0.15 | small |
| intensification_ratio | 6 | -0.037 | 0.159 | -0.37 | medium |
| avg_distance_to_best | 6 | -0.053 | 0.018 | -0.61 | large |
| average_convergence_rate | 50 | -0.084 | <0.001 | -0.72 | large |
| success_rate | — | skipped (n=1) | — | — | — |

Four conditions showed statistically significant improvement over vanilla (p < 0.01): exploit_%, avg_improv, explore_%, dispersion. Two showed significant degradation: conv_rate, dist_best.

**Wilcoxon signed-rank** (paired comparison of each condition's best algorithm vs vanilla's best on 50 instance-seed AUC scores):

| Condition | Best Algorithm | Mean Diff | p-value | Cliff's d |
|-----------|---------------|-----------|---------|-----------|
| avg_improvement | DynamicExploreAdaptivePerturbation | +0.289 | <0.001 | +0.61 (large) |
| avg_exploitation_pct | AdaptiveHybridDifferentialEvolution | +0.244 | <0.001 | +0.55 (large) |
| longest_no_improvement_streak | HybridPhaseDiversityOptimizer | +0.165 | <0.001 | +0.32 (small) |
| last_improvement_fraction | AdaptiveMomentumWithDynamicExploration | +0.102 | <0.001 | +0.26 (small) |
| dispersion | HybridAdaptiveOptimizer | +0.060 | <0.001 | +0.21 (small) |
| avg_exploration_pct | AdaptiveDiversityDrivenSearch | +0.013 | 0.443 | +0.07 (negligible) |
| average_convergence_rate | Optimizer | -0.072 | 0.011 | -0.26 (small) |
| avg_nearest_neighbor_distance | HybridAdaptiveDirectionalSampling | -0.076 | <0.001 | -0.12 (negligible) |
| success_rate | AdaptiveDirectionalRandomSearch | -0.131 | <0.001 | -0.34 (medium) |
| intensification_ratio | AdaptiveGaussianIntensification | -0.198 | <0.001 | -0.41 (medium) |
| avg_distance_to_best | HybridAdaptiveOptimizer | -0.199 | <0.001 | -0.43 (medium) |

### 3.5 Per-Instance Analysis

The heatmap of best-algorithm AOCC by instance reveals clear instance-dependent difficulty. Instances 0 and 9 are the hardest (AOCC 0.10-0.20 for vanilla), while instances 3 and 5 are the easiest (AOCC 0.47-0.73 for vanilla).

Per-instance wins (how many of the 10 instances does each condition's best algorithm beat vanilla's best):

| Condition | Instances Better | Instances Worse |
|-----------|-----------------|----------------|
| avg_improvement | 10 | 0 |
| longest_no_improvement_streak | 10 | 0 |
| dispersion | 9 | 1 |
| avg_exploitation_pct | 9 | 1 |
| last_improvement_fraction | 7 | 3 |
| avg_exploration_pct | 6 | 4 |
| avg_nearest_neighbor_distance | 3 | 7 |
| average_convergence_rate | 3 | 7 |
| avg_distance_to_best | 0 | 10 |
| intensification_ratio | 0 | 10 |
| success_rate | 0 | 10 |

avg_improvement and longest_no_improvement_streak beat vanilla on all 10 instances. avg_distance_to_best, intensification_ratio, and success_rate lost on all 10.

### 3.6 Correlation Analysis

Pearson correlations between behavioral metrics and AOCC (fitness), pooled across all valid candidates:

| Metric | Correlation with AOCC |
|--------|----------------------|
| avg_exploration_pct | +0.10 |
| intensification_ratio | +0.07 |
| average_convergence_rate | +0.06 |
| avg_improvement | -0.05 |
| avg_nearest_neighbor_distance | -0.05 |
| avg_distance_to_best | -0.06 |
| avg_exploitation_pct | -0.10 |
| success_rate | -0.16 |
| last_improvement_fraction | -0.17 |
| longest_no_improvement_streak | -0.19 |
| dispersion | -0.44 |

Most metrics show weak correlation with fitness. Dispersion has the strongest relationship (r = -0.44): lower dispersion (better coverage of the search space) is associated with higher AOCC. Note that the sign of correlation does not determine whether showing the metric in feedback helps — avg_improvement has a near-zero raw correlation (-0.05) yet produced the best condition by far.

Notable inter-metric correlations: explore_% and exploit_% are perfectly anti-correlated (r = -1.00, they are complements); no_improv and last_improv are strongly correlated (r = 0.86, both measure stagnation); intens_ratio and exploit_% are strongly correlated (r = 0.82).

### 3.7 Behavioral Metric Evolution

The best-so-far metric tracking plots show how the behavioral profile of the leading candidate evolves over evaluations. When a metric is shown in feedback, some conditions exhibit divergent behavior:
- **dispersion**: The shown condition's best-so-far candidate achieves much lower dispersion (~4) compared to vanilla's best (~10), indicating the LLM actively steered algorithm design toward better search space coverage.
- **exploit_%**: The shown condition's best candidate exploits much more heavily (~90%) vs vanilla (~60%).
- **conv_rate**: Minimal difference between shown and vanilla, suggesting this metric did not provide actionable guidance.

## 4. Limitations

### 4.1 Single-Run Design

Each condition was run exactly once (one evolutionary run of 100 candidates). The (1+1)-ES is inherently stochastic — different runs of the same condition would produce different candidate algorithms and potentially different outcomes. Without multiple independent runs per condition, the observed differences between conditions could be due to random variation rather than the effect of the feedback metric. The statistical tests applied (Mann-Whitney U, Wilcoxon signed-rank) compare candidates *within* a single run but do not account for run-to-run variance.

### 4.2 High Failure Rates Reduce Effective Sample Size

With 50-99% failure rates, many conditions have very few valid candidates (e.g., success_rate: 1, dist_best: 6, intens_ratio: 6). Statistical tests on these conditions have low power. The failure rate itself is confounded with the condition — it is unclear whether certain feedback metrics cause higher failure rates or whether this variation is due to run-level randomness.

### 4.3 Failure Rates May Confound Performance

Conditions with fewer valid candidates have fewer opportunities to discover good algorithms, but also face less selection pressure in the (1+1)-ES (the elite is retained through long stretches of failures). The relationship between failure rate and final performance is not straightforward and warrants further investigation.
