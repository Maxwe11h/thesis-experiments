"""Feedback formatters for different experimental conditions."""

FEATURE_DESCRIPTIONS = {
    "avg_nearest_neighbor_distance": "How spread out the evaluated points are from each other",
    "dispersion": "How well the evaluated points cover the search domain",
    "avg_exploration_pct": "Percentage of search effort spent exploring vs exploiting",
    "avg_distance_to_best": "How far each evaluation is from the current best solution",
    "intensification_ratio": "How often the algorithm samples near the best-so-far solution",
    "avg_exploitation_pct": "Percentage of search effort spent exploiting vs exploring",
    "average_convergence_rate": "Rate at which the best-so-far fitness error decreases per step",
    "avg_improvement": "Average size of improvement when a new best solution is found",
    "success_rate": "How often an evaluation improves on the best-so-far solution",
    "longest_no_improvement_streak": "Longest run of consecutive evaluations with no improvement",
    "last_improvement_fraction": "How much of the budget has passed since the last improvement",
}


def vanilla_feedback(name, auc_mean, auc_std, _metrics):
    """AOCC-only feedback (matches existing LLaMEA format)."""
    return (
        f"The algorithm {name} got an average Area over the convergence curve "
        f"(AOCC, 1.0 is the best) score of {auc_mean:0.4f} with standard "
        f"deviation {auc_std:0.4f}."
    )


def make_single_feature_feedback(feature_name):
    """Factory: returns a feedback function that reports AOCC + one behavioral metric."""
    description = FEATURE_DESCRIPTIONS[feature_name]

    def feedback_fn(name, auc_mean, auc_std, metrics):
        base = vanilla_feedback(name, auc_mean, auc_std, metrics)
        value = metrics.get(feature_name)
        if value is None:
            return base
        return (
            f"{base}\n"
            f"Behavioral metric — {feature_name}: {value:.4f}\n"
            f"  ({description})"
        )

    return feedback_fn
