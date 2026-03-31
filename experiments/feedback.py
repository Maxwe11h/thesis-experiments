"""Feedback formatters for different experimental conditions."""

# ---------------------------------------------------------------------------
# Feature descriptions (neutral — no directional guidance)
# ---------------------------------------------------------------------------

FEATURE_DESCRIPTIONS = {
    "avg_nearest_neighbor_distance": "the average distance between consecutive evaluated points and their nearest predecessors, measuring how spread out the search is",
    "dispersion": "how well the evaluated points cover the search domain",
    "avg_exploration_pct": "the percentage of search effort spent exploring vs exploiting",
    "avg_distance_to_best": "how far each evaluation is from the current best solution",
    "intensification_ratio": "the fraction of evaluations that sample near the best-so-far solution",
    "avg_exploitation_pct": "the percentage of search effort spent exploiting vs exploring",
    "average_convergence_rate": "the rate at which the best-so-far fitness error decreases per step",
    "avg_improvement": "the average magnitude of improvement when a new best solution is found",
    "success_rate": "how often an evaluation improves on the best-so-far solution",
    "longest_no_improvement_streak": "the longest run of consecutive evaluations with no improvement",
    "last_improvement_fraction": "how much of the budget has passed since the last improvement",
    "step_size_mean": "the average distance between consecutive evaluated points",
    "step_size_std": "the variability in step sizes between consecutive evaluations",
    "step_size_trend": "whether the algorithm's step sizes are shrinking or growing over time",
    "directional_persistence": "whether the algorithm moves in consistent directions or zigzags",
    "fitness_sample_entropy": "the complexity and irregularity of the fitness trajectory",
    "fitness_permutation_entropy": "the randomness in the ordering of fitness values",
    "fitness_autocorrelation": "how similar consecutive fitness values are to each other",
    "fitness_lempel_ziv_complexity": "the compressibility of the improvement pattern",
    "x_spread_early": "how spread out the search is across dimensions in the first quarter of evaluations",
    "x_spread_late": "how spread out the search is across dimensions in the last quarter of evaluations",
    "spread_ratio": "whether the spatial spread of the search contracts or expands over time",
    "centroid_drift": "how far the centre of the search has moved from early to late evaluations",
    "f_range_early": "the range of fitness values observed in the first quarter of evaluations",
    "f_range_late": "the range of fitness values observed in the last quarter of evaluations",
    "f_range_ratio": "whether the fitness range narrows or widens from early to late evaluations",
    "improvement_spatial_correlation": "whether larger steps tend to produce larger improvements",
    "improvement_burstiness": "whether improvements come in bursts or at a steady rate",
    "dimension_convergence_heterogeneity": "whether the algorithm converges evenly across all dimensions or focuses on some more than others",
    "step_size_autocorrelation": "whether large steps tend to follow large steps, capturing momentum in the search",
    "fitness_plateau_fraction": "the fraction of consecutive evaluations where fitness barely changes",
    "half_convergence_time": "how quickly the algorithm reaches half of its total improvement",
}

# ---------------------------------------------------------------------------
# Directional guidance (derived from Phase 2 correlation analysis)
# ---------------------------------------------------------------------------
# Maps feature_name -> (direction_str, guidance_str)

FEATURE_DIRECTIONS = {
    "avg_improvement": ("lower", "Lower values are associated with better performance, indicating small precise refinements rather than large jumps."),
    "intensification_ratio": ("higher", "Higher values are associated with better performance, indicating the algorithm concentrates search near the best solution found."),
    "fitness_plateau_fraction": ("higher", "Higher values are associated with better performance, indicating the algorithm exploits regions with similar fitness near the optimum."),
    "step_size_autocorrelation": ("higher", "Higher values are associated with better performance, indicating consistent, smooth step-size adaptation rather than erratic jumps."),
    "improvement_spatial_correlation": ("higher", "Higher values are associated with better performance, indicating that larger exploratory steps yield proportionally larger improvements."),
    "half_convergence_time": ("lower", "Lower values are associated with better performance, indicating fast early convergence towards good solutions."),
    "fitness_autocorrelation": ("higher", "Higher values are associated with better performance, indicating structured local search that samples nearby points with similar fitness."),
    "x_spread_early": ("lower", "Lower values are associated with better performance, indicating a more focused initial search rather than broad exploration."),
    "longest_no_improvement_streak": ("lower", "Lower values are associated with better performance, indicating the algorithm avoids long periods of stagnation."),
    "dimension_convergence_heterogeneity": ("higher", "Higher values are associated with better performance, indicating the algorithm exploits separability by converging faster on easier dimensions."),
}

# ---------------------------------------------------------------------------
# Comparative reference values (Phase 1, top-10% AOCC median)
# ---------------------------------------------------------------------------
# Median values of candidates with AOCC >= 0.8465 (90th percentile, n=280).
# Computed from results_phase1/ summary CSVs.  bottom-25% medians used to
# normalise distance (gives a meaningful scale across features).
#
# Format: feature_name -> (top10_median, bottom25_median)

FEATURE_REFERENCES = {
    "avg_improvement":                      (0.1040, 1.75),
    "intensification_ratio":                (0.8654, 0.0002),
    "fitness_plateau_fraction":             (0.5873, 0.0),
    "step_size_autocorrelation":            (0.9070, 0.13),
    "improvement_spatial_correlation":      (0.6739, 0.06),
    "half_convergence_time":                (0.0012, 0.0065),
    "fitness_autocorrelation":         (0.7851, 0.003),
    "x_spread_early":                       (0.7105, 2.89),
    "longest_no_improvement_streak":        (5327,   6264),
    "dimension_convergence_heterogeneity":  (0.0841, 0.0006),
}


# ---------------------------------------------------------------------------
# Integer-valued features (format as int, not float)
# ---------------------------------------------------------------------------
INTEGER_FEATURES = {"longest_no_improvement_streak"}


# ---------------------------------------------------------------------------
# Feedback formatters
# ---------------------------------------------------------------------------

def _fmt_value(feature_name, value):
    """Format a metric value: integer features as int, others as 4-decimal float."""
    if feature_name in INTEGER_FEATURES:
        return f"{int(round(value))}"
    return f"{value:.4f}"


def vanilla_feedback(name, auc_mean, auc_std, _metrics, _metrics_std=None):
    """AOCC-only feedback (matches existing LLaMEA format)."""
    return (
        f"The algorithm {name} got an average Area over the convergence curve "
        f"(AOCC, 1.0 is the best) score of {auc_mean:0.4f} with standard "
        f"deviation {auc_std:0.4f}."
    )


def _metric_sentence(feature_name, value, std, description):
    """Build the common metric sentence: 'It achieved a {name} of {val} (std {std}), which measures {desc}.'"""
    fmt_val = _fmt_value(feature_name, value)
    if std is not None:
        fmt_std = _fmt_value(feature_name, std)
        return f"It achieved a {feature_name} of {fmt_val} with standard deviation {fmt_std}, which measures {description}."
    return f"It achieved a {feature_name} of {fmt_val}, which measures {description}."


def make_single_feature_feedback(feature_name):
    """Factory: returns a feedback function that reports AOCC + one behavioral metric (neutral)."""
    description = FEATURE_DESCRIPTIONS[feature_name]

    def feedback_fn(name, auc_mean, auc_std, metrics, metrics_std=None):
        base = vanilla_feedback(name, auc_mean, auc_std, metrics)
        value = metrics.get(feature_name)
        if value is None:
            return base
        std = metrics_std.get(feature_name) if metrics_std else None
        return f"{base} {_metric_sentence(feature_name, value, std, description)}"

    return feedback_fn


def make_directional_feature_feedback(feature_name):
    """Factory: AOCC + one behavioral metric with directional guidance."""
    description = FEATURE_DESCRIPTIONS[feature_name]
    _direction, guidance = FEATURE_DIRECTIONS[feature_name]

    def feedback_fn(name, auc_mean, auc_std, metrics, metrics_std=None):
        base = vanilla_feedback(name, auc_mean, auc_std, metrics)
        value = metrics.get(feature_name)
        if value is None:
            return base
        std = metrics_std.get(feature_name) if metrics_std else None
        sentence = _metric_sentence(feature_name, value, std, description)
        return f"{base} {sentence} {guidance}"

    return feedback_fn


def make_multi_feature_neutral_feedback(feature_names):
    """Factory: returns a feedback function that reports AOCC + multiple behavioral metrics (neutral format)."""
    descriptions = {name: FEATURE_DESCRIPTIONS[name] for name in feature_names}

    def feedback_fn(name, auc_mean, auc_std, metrics, metrics_std=None):
        base = vanilla_feedback(name, auc_mean, auc_std, metrics)
        parts = [base]
        for feat in feature_names:
            value = metrics.get(feat)
            if value is None:
                continue
            std = metrics_std.get(feat) if metrics_std else None
            parts.append(_metric_sentence(feat, value, std, descriptions[feat]))
        return " ".join(parts)

    return feedback_fn


def make_comparative_feature_feedback(feature_name):
    """Factory: AOCC + one behavioral metric compared against top-performing reference.

    Handles four regimes:
      - Already better than (or matching) the reference -> maintain
      - Close to the reference -> small refinements
      - Moderately worse -> clear action needed
      - Far worse -> priority to improve
    """
    description = FEATURE_DESCRIPTIONS[feature_name]
    direction, _guidance = FEATURE_DIRECTIONS[feature_name]
    top_ref, bot_ref = FEATURE_REFERENCES[feature_name]
    ref_range = abs(top_ref - bot_ref)
    verb = "Reducing" if direction == "lower" else "Increasing"

    def feedback_fn(name, auc_mean, auc_std, metrics, metrics_std=None):
        base = vanilla_feedback(name, auc_mean, auc_std, metrics)
        value = metrics.get(feature_name)
        if value is None:
            return base

        std = metrics_std.get(feature_name) if metrics_std else None
        sentence = _metric_sentence(feature_name, value, std, description)
        fmt_ref = _fmt_value(feature_name, top_ref)

        # Signed gap: positive = worse than reference, negative = better
        if direction == "higher":
            gap = top_ref - value
        else:
            gap = value - top_ref

        normalised = gap / ref_range if ref_range > 0 else 0.0

        if normalised <= 0.0:
            comparison = f"This already meets or exceeds the top-performing reference of {fmt_ref}. Maintain this."
        elif normalised <= 0.15:
            comparison = f"This is close to the top-performing reference of {fmt_ref}. Small refinements could bring it in line with the best algorithms."
        elif normalised <= 0.5:
            comparison = f"The top-performing algorithms achieve {fmt_ref} for this metric. {verb} this value would move the algorithm towards that level."
        else:
            comparison = (
                f"This is far from the top-performing reference of {fmt_ref}. "
                f"{verb} this value is a priority — the best algorithms achieve "
                f"significantly {'lower' if direction == 'lower' else 'higher'} values."
            )

        return f"{base} {sentence} {comparison}"

    return feedback_fn
