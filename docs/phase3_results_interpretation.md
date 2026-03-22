# Phase 3 Results Interpretation

## Experiment Setup

- **Model**: gemini-3-flash (Phase 1 winner on efficiency)
- **Strategy**: (1+1)-ES, 100 candidates, 5 seeds per condition
- **Top-5 Borda features**: avg_improvement, intensification_ratio, fitness_plateau_fraction, step_size_autocorrelation, improvement_spatial_correlation
- **3 feedback formats** per feature = 15 conditions, 75 runs total
- **Vanilla baseline**: Phase 1 gemini-3-flash AOCC-only (0.862 +/- 0.097)

### Feedback Formats

| Format | What the LLM sees |
|---|---|
| Neutral | AOCC score + feature value + description |
| Directional | Neutral + "higher/lower values are associated with better performance" |
| Comparative | Neutral + "top-performing algorithms achieve X; you are close/far" |

## Key Findings

### 1. More guidance = worse performance

| Format | Mean Best AOCC | Mean Rank (1=best) | Wins (out of 25) |
|---|---|---|---|
| Neutral | 0.869 | 1.52 | 17 |
| Directional | 0.840 | 1.88 | 6 |
| Comparative | 0.788 | 2.60 | 2 |

Overall Kruskal-Wallis p=0.003. Neutral dominates across 4/5 features. The strongest effect is on intensification_ratio (KW p=0.002, Cliff's d=1.0 for all pairwise comparisons).

### 2. Directional advice is correct but steering is unreliable

Directional feedback gives correct advice (verified: coded direction matches empirical Spearman correlation for all 5 features).

| Feature | AOCC Correlation | Advice | Directional shift | Comparative shift |
|---|---|---|---|---|
| avg_improvement | -0.146 (lower=better) | "Lower values are better" | +0.012 (WRONG) | +0.010 (WRONG) |
| intensification_ratio | +0.464 (higher=better) | "Higher values are better" | -0.030 (WRONG) | +0.038 (correct) |
| fitness_plateau_fraction | +0.578 (higher=better) | "Higher values are better" | -0.085 (WRONG) | +0.105 (correct) |
| step_size_autocorrelation | +0.462 (higher=better) | "Higher values are better" | -0.034 (WRONG) | -0.020 (WRONG) |
| improvement_spatial_corr | +0.409 (higher=better) | "Higher values are better" | -0.019 (WRONG) | -0.020 (WRONG) |

When measured as aggregate median shift vs neutral, directional pushes the wrong way for 5/5 features and comparative for 3/5. However, this aggregate view masks important nuance revealed by the AOCC-matched analysis (Finding 5).

### 3. Neutral feedback produces algorithms with naturally good behavioral profiles

Performance tier analysis (bottom-25%, middle-50%, top-25% by AOCC) shows:

| Feature | Bot 25% | Mid 50% | Top 25% | Neutral | Directional | Comparative |
|---|---|---|---|---|---|---|
| avg_improvement | 0.133 | 0.096 | 0.078 | 0.094 | 0.105 | 0.104 |
| intensification_ratio | 0.730 | 0.792 | 0.868 | 0.813 | 0.783 | 0.852 |
| fitness_plateau_fraction | 0.199 | 0.366 | 0.594 | 0.448 | 0.364 | 0.553 |
| step_size_autocorrelation | 0.845 | 0.881 | 0.900 | 0.897 | 0.863 | 0.877 |
| improvement_spatial_corr | 0.622 | 0.630 | 0.683 | 0.649 | 0.630 | 0.629 |

- Neutral sits between mid-50% and top-25% naturally
- Directional gets pulled down toward mid-50% or worse
- Comparative is mixed: reaches near top-25% for 2 features but falls back for 3

### 4. No feedback format significantly outperforms vanilla

Top conditions show positive deltas vs vanilla baseline but none reach significance (all p>0.09, 5 seeds):

| Condition | AOCC | vs Vanilla |
|---|---|---|
| neutral-intensification_ratio | 0.907 | +0.046 |
| neutral-fitness_plateau_fraction | 0.891 | +0.030 |
| neutral-avg_improvement | 0.883 | +0.021 |
| comparative-intensification_ratio | 0.721 | -0.140 |

### 5. AOCC-matched analysis: feedback changes behavior but not predictably

When controlling for algorithm quality by comparing within AOCC bands (low 0.3-0.5, mid 0.5-0.7, high 0.7-0.85, elite 0.85+), 16/19 feature x band combinations show significant behavioral differences between formats (KW p<0.05). This confirms the LLM is responding to the feedback -- it produces behaviorally different code. But the direction of the shift depends on both the feature and the quality band.

**Direction tally within AOCC bands:**
- Directional vs Neutral: 9/19 right, 10/19 wrong (47% -- essentially coin-flip)
- Comparative vs Neutral: 8/19 right, 11/19 wrong (42%)

**Feature-by-feature patterns:**

**avg_improvement** (lower = better):
- Directional nails it in the low-AOCC band (massive -0.19 shift) but fails in mid/high/elite
- Comparative always pushes the wrong way
- Neutral produces the best values at higher quality levels
- Interpretation: among weak algorithms, directional feedback helps reduce avg_improvement. Among strong algorithms the values naturally converge and guidance adds noise.

**intensification_ratio** (higher = better):
- Both directional and comparative push RIGHT in low and mid bands
- Comparative is best in low/mid but flips to worst in the high band
- Directional is best in the high band
- Interpretation: prescriptive feedback helps at lower quality, but at high AOCC the relationship between intensification_ratio and code structure becomes more subtle and guidance misfires.

**fitness_plateau_fraction** (higher = better):
- Comparative consistently pushes right across ALL 4 bands -- the one feature where comparative genuinely works as intended
- Directional pushes right in low/mid but flips to wrong in high/elite
- Interpretation: the comparative reference target (0.5873) for plateau fraction may be concrete enough for the LLM to act on. This feature may be more directly mappable to code patterns (e.g., convergence tolerance).

**step_size_autocorrelation** (higher = better):
- Mostly wrong for both formats
- Comparative gets it right only in the elite band
- Neutral consistently produces the best values in mid/high bands
- Interpretation: step size autocorrelation is a deeply emergent property that resists direct steering.

**improvement_spatial_correlation** (higher = better):
- Both formats push wrong in low/mid bands
- Both push right in the high band
- Directional is best in the elite band
- Interpretation: at high quality the LLM may start to understand the relationship between step size and improvement, but this doesn't translate to AOCC gains.

**Key insight:** The aggregate analysis (Finding 2) showing 0/5 and 2/5 correct directions is misleading because it conflates quality levels. Within AOCC bands, steering accuracy is closer to 50/50 -- the LLM sometimes follows the guidance, sometimes doesn't, depending on the feature and the quality level of algorithms it's producing. The inconsistency itself is the finding: **behavioral steering via natural language is unreliable, not systematically wrong**.

### 6. Why neutral works best

The relationship between code changes and emergent behavioral metrics is highly indirect. Telling the LLM "increase step_size_autocorrelation" is like telling a chef "increase the Maillard reaction" -- they know it's desirable but the mapping to concrete actions (temperature, timing, moisture) is non-obvious and easy to get wrong.

Neutral feedback works because:
1. It gives the LLM a richer picture of what happened without prescribing action
2. The evolutionary selection pressure (keeping high-AOCC algorithms) implicitly teaches which behavioral patterns matter
3. The LLM can use the behavioral information as context for understanding its algorithms without being pulled toward a potentially counterproductive secondary objective
4. It avoids the multi-objective tension between "improve AOCC" and "move feature X in direction Y"

## Interpretation

Behavioral features provide useful observational context that slightly improves or maintains LLM-driven algorithm design when presented neutrally. Prescriptive feedback -- whether directional guidance or comparative targets -- degrades overall performance despite the LLM clearly responding to it (16/19 AOCC-matched comparisons significant).

The AOCC-matched analysis reveals that steering is not systematically wrong but unpredictable: roughly coin-flip accuracy across feature x band combinations. The LLM changes the behavioral profile of its generated algorithms in response to feedback, but it cannot reliably translate high-level behavioral objectives into the right code changes. Some features (fitness_plateau_fraction) are more steerable than others (step_size_autocorrelation), suggesting that steerability depends on how directly the behavioral metric maps to identifiable code patterns.

This suggests that in LLM-guided evolutionary algorithm design, **the selection mechanism should remain the primary driver of behavioral improvement**, with feedback serving an informational rather than prescriptive role.

## Open Questions for Remaining 14 Conditions

- Does the neutral > directional > comparative pattern hold for the bottom-5 features?
- Are some features more "steerable" than others? (Early signs: fitness_plateau_fraction is the most steerable)
- Do features with stronger AOCC correlation show stronger format effects?
- Does the AOCC-matched steering accuracy improve for features with more direct code-to-behavior mappings?

## Condition Rankings (Full)

| Rank | Condition | AOCC | Fail% |
|---|---|---|---|
| 1 | neutral-intensification_ratio | 0.907 +/- 0.021 | 4.2% |
| 2 | neutral-fitness_plateau_fraction | 0.891 +/- 0.005 | 7.6% |
| 3 | neutral-avg_improvement | 0.883 +/- 0.055 | 2.2% |
| 4 | directional-fitness_plateau_fraction | 0.874 +/- 0.055 | 3.0% |
| 5 | directional-avg_improvement | 0.866 +/- 0.060 | 3.4% |
| 6 | comparative-avg_improvement | 0.860 +/- 0.076 | 2.8% |
| 7 | directional-improvement_spatial_corr | 0.857 +/- 0.044 | 1.8% |
| 8 | neutral-improvement_spatial_corr | 0.853 +/- 0.089 | 1.8% |
| 9 | comparative-fitness_plateau_fraction | 0.811 +/- 0.077 | 1.4% |
| 10 | neutral-step_size_autocorrelation | 0.810 +/- 0.111 | 1.8% |
| 11 | directional-intensification_ratio | 0.806 +/- 0.044 | 2.0% |
| 12 | directional-step_size_autocorrelation | 0.798 +/- 0.075 | 2.0% |
| 13 | comparative-improvement_spatial_corr | 0.786 +/- 0.100 | 1.2% |
| 14 | comparative-step_size_autocorrelation | 0.761 +/- 0.103 | 0.8% |
| 15 | comparative-intensification_ratio | 0.721 +/- 0.046 | 0.6% |
