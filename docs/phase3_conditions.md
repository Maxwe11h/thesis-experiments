# Phase 3: Behavioral Feedback Conditions

Phase 3 screens whether adding a single behavioral feature to the LLM's fitness feedback improves algorithm design, and whether the framing of that feedback matters.

## Experimental Design

**Model**: Gemini 3 Flash (`gemini-3-flash-preview`)
**Strategy**: (1+1)-ES with 90% refine / 10% explore mutation mix
**Benchmark**: MA-BBOB, 10 training instances, 5 eval seeds, 5 dimensions, budget factor 2000
**Budget**: 100 LLM-generated candidates per run
**Seeds**: 5 independent runs per condition
**Baseline**: Vanilla (AOCC-only) results reused from Phase 1 gemini-3-flash runs

## Feedback Formats

Each condition adds exactly one behavioral feature to the standard AOCC feedback sentence. Three formats are tested:

### 1. Neutral

Reports the feature value and its definition, with no interpretation of whether high or low is desirable.

> The algorithm X got an average Area over the convergence curve (AOCC, 1.0 is the best) score of 0.7234 with standard deviation 0.0891. It achieved a step_size_autocorrelation of 0.8421 with standard deviation 0.0532, which measures whether large steps tend to follow large steps, capturing momentum in the search.

### 2. Directional

Adds a sentence telling the LLM which direction is associated with better performance, derived from Phase 2 Spearman correlation analysis.

> ...which measures whether large steps tend to follow large steps, capturing momentum in the search. Higher values are associated with better performance, indicating consistent, smooth step-size adaptation rather than erratic jumps.

### 3. Comparative

Replaces directional guidance with a distance-aware comparison against the median feature value of top-10% AOCC algorithms from Phase 1 (n=280, AOCC >= 0.847). The language adapts based on how far the candidate's value is from the reference.

The distance is normalised by the range between top-10% and bottom-25% medians, giving a meaningful scale (bottom-25% algorithms have AOCC ~0.21, essentially RandomSearch-level). Four tiers:

| Normalised distance | Tier | Example language |
|---------------------|------|------------------|
| <= 0 | Already there | "This already meets or exceeds the top-performing reference of 0.9070. Maintain this." |
| 0 - 0.15 | Close | "This is close to the top-performing reference of 0.9070. Small refinements could bring it in line with the best algorithms." |
| 0.15 - 0.50 | Moderate | "The top-performing algorithms achieve 0.9070 for this metric. Increasing this value would move the algorithm towards that level." |
| > 0.50 | Far | "This is far from the top-performing reference of 0.9070. Increasing this value is a priority..." |

## Selected Features

10 features selected via greedy Borda-count ranking with |rho| > 0.8 correlation de-duplication (see `docs/selected_behavioral_features.md` for full details).

| # | Feature | Direction | rho | Category |
|---|---------|-----------|-----|----------|
| 1 | `avg_improvement` | lower | -0.782 | Convergence |
| 2 | `intensification_ratio` | higher | +0.725 | Exploitation |
| 3 | `fitness_plateau_fraction` | higher | +0.685 | Novel |
| 4 | `step_size_autocorrelation` | higher | +0.766 | Novel |
| 5 | `improvement_spatial_correlation` | higher | +0.757 | Novel |
| 6 | `half_convergence_time` | lower | -0.597 | Novel |
| 7 | `fitness_autocorrelation` | higher | +0.645 | Info-Theoretic |
| 8 | `x_spread_early` | lower | -0.599 | Early/Late |
| 9 | `longest_no_improvement_streak` | lower | -0.469 | Stagnation |
| 10 | `dimension_convergence_heterogeneity` | higher | +0.471 | Novel |

Note: `fitness_autocorrelation` was previously named `fitness_autocorrelation_lag1` (renamed 2026-03-11; lag-1 is the only lag computed).

## Condition List

29 total conditions (10 neutral + 10 directional + 9 comparative):

| Condition tag | Feature | Format |
|---------------|---------|--------|
| `neutral-avg_improvement` | avg_improvement | neutral |
| `directional-avg_improvement` | avg_improvement | directional |
| `comparative-avg_improvement` | avg_improvement | comparative |
| `neutral-intensification_ratio` | intensification_ratio | neutral |
| `directional-intensification_ratio` | intensification_ratio | directional |
| `comparative-intensification_ratio` | intensification_ratio | comparative |
| `neutral-fitness_plateau_fraction` | fitness_plateau_fraction | neutral |
| `directional-fitness_plateau_fraction` | fitness_plateau_fraction | directional |
| `comparative-fitness_plateau_fraction` | fitness_plateau_fraction | comparative |
| `neutral-step_size_autocorrelation` | step_size_autocorrelation | neutral |
| `directional-step_size_autocorrelation` | step_size_autocorrelation | directional |
| `comparative-step_size_autocorrelation` | step_size_autocorrelation | comparative |
| `neutral-improvement_spatial_correlation` | improvement_spatial_correlation | neutral |
| `directional-improvement_spatial_correlation` | improvement_spatial_correlation | directional |
| `comparative-improvement_spatial_correlation` | improvement_spatial_correlation | comparative |
| `neutral-half_convergence_time` | half_convergence_time | neutral |
| `directional-half_convergence_time` | half_convergence_time | directional |
| `comparative-half_convergence_time` | half_convergence_time | comparative |
| `neutral-fitness_autocorrelation` | fitness_autocorrelation | neutral |
| `directional-fitness_autocorrelation` | fitness_autocorrelation | directional |
| `comparative-fitness_autocorrelation` | fitness_autocorrelation | comparative |
| `neutral-x_spread_early` | x_spread_early | neutral |
| `directional-x_spread_early` | x_spread_early | directional |
| `comparative-x_spread_early` | x_spread_early | comparative |
| `neutral-longest_no_improvement_streak` | longest_no_improvement_streak | neutral |
| `directional-longest_no_improvement_streak` | longest_no_improvement_streak | directional |
| `neutral-dimension_convergence_heterogeneity` | dimension_convergence_heterogeneity | neutral |
| `directional-dimension_convergence_heterogeneity` | dimension_convergence_heterogeneity | directional |
| `comparative-dimension_convergence_heterogeneity` | dimension_convergence_heterogeneity | comparative |

### Why no comparative condition for `longest_no_improvement_streak`?

This feature has a U-shaped (non-monotonic) relationship with AOCC. Top-performing algorithms have high streaks (~5,327) because they converge early and then coast through the remaining budget. Bottom-performing algorithms also have high streaks (~6,264) because they stagnate. Mid-quality algorithms actually have the shortest streaks. The top-10% to bottom-25% range is only 937 evaluations, and 53% of all algorithms appear "better than reference" under this metric. This makes comparative distance feedback misleading, so it is excluded. Neutral and directional feedback still work because they do not reference an absolute target.

## Comparative Reference Values

Median feature values for top-10% AOCC algorithms (AOCC >= 0.847, n=280) and bottom-25% AOCC algorithms (AOCC <= 0.233, n=768). The bottom-25% anchor represents RandomSearch-level behavior (mean AOCC ~0.21).

| Feature | Top-10% median | Bottom-25% median | Range |
|---------|---------------|-------------------|-------|
| `avg_improvement` | 0.1040 | 1.7500 | 1.646 |
| `intensification_ratio` | 0.8654 | 0.0002 | 0.865 |
| `fitness_plateau_fraction` | 0.5873 | 0.0000 | 0.587 |
| `step_size_autocorrelation` | 0.9070 | 0.1300 | 0.777 |
| `improvement_spatial_correlation` | 0.6739 | 0.0600 | 0.614 |
| `half_convergence_time` | 0.0012 | 0.0065 | 0.005 |
| `fitness_autocorrelation` | 0.7851 | 0.0030 | 0.782 |
| `x_spread_early` | 0.7105 | 2.8900 | 2.180 |
| `dimension_convergence_heterogeneity` | 0.0841 | 0.0006 | 0.084 |

## Rationale

### Why screen features individually?

Testing features one at a time isolates the effect of each feature on LLM-guided algorithm design. Multi-feature feedback risks confounding effects and makes it harder to attribute improvements to specific features.

### Why three feedback formats?

The formats test increasing levels of interpretive guidance:
- **Neutral** tests whether the LLM can figure out on its own what the feature value means
- **Directional** tests whether telling the LLM "higher/lower is better" helps
- **Comparative** tests whether anchoring against a concrete reference value helps further

This lets us disentangle "is the feature informative?" from "does the LLM need help interpreting it?"

### Why bottom-25% as normalisation anchor?

We need a scale to decide whether a gap is "close" or "far" from the top-10% reference. The bottom-25% median represents the spread between good and bad algorithms for each feature. We evaluated alternatives:
- **Bottom-10%**: Worse coverage for most features due to outlier contamination
- **Median**: Too narrow, only 30-39% of algorithms fall in the normalised [0, 1] range
- **Min/max**: Too wide, washes out meaningful differences

Bottom-25% gives 66-83% of algorithms in [0, 1] for 9 of 10 features.

### Why reuse Phase 1 vanilla as baseline?

Phase 3 uses the same model (gemini-3-flash), benchmark, strategy, initial population, and seeds as Phase 1. Running vanilla again would duplicate existing data. The Phase 1 gemini-3-flash results across 5 seeds provide a direct comparison point.

## Running

```bash
# List all conditions
python run_phase3.py --list

# Run one condition
python run_phase3.py neutral-avg_improvement

# Run all neutral conditions
python run_phase3.py neutral

# Run everything
python run_phase3.py all

# Sanity check (2 instances, 1 seed, 10 candidates)
python run_phase3.py neutral-avg_improvement --sanity

# Launch all conditions in parallel on server
nohup bash run_phase3.sh > logs/phase3_all.log 2>&1 &
```

Results are written to `results_phase3/{condition_tag}/seed-{seed}/`.
