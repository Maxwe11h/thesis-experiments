"""Feedback formatters for different experimental conditions."""


def vanilla_feedback(name, auc_mean, auc_std, metrics):
    """AOCC-only feedback (matches existing LLaMEA format)."""
    return (
        f"The algorithm {name} got an average Area over the convergence curve "
        f"(AOCC, 1.0 is the best) score of {auc_mean:0.4f} with standard "
        f"deviation {auc_std:0.4f}."
    )


def behavioral_feedback(name, auc_mean, auc_std, metrics):
    """AOCC + all 11 behavioral metrics."""
    lines = [vanilla_feedback(name, auc_mean, auc_std, metrics)]

    lines.append("\nBehavioral profile:")
    lines.append("  Exploration & diversity:")
    lines.append(f"    avg_nearest_neighbor_distance: {metrics['avg_nearest_neighbor_distance']:.4f}")
    lines.append(f"    dispersion: {metrics['dispersion']:.4f}")
    lines.append(f"    avg_exploration_pct: {metrics['avg_exploration_pct']:.4f}")

    lines.append("  Exploitation:")
    lines.append(f"    avg_distance_to_best: {metrics['avg_distance_to_best']:.4f}")
    lines.append(f"    intensification_ratio: {metrics['intensification_ratio']:.4f}")
    lines.append(f"    avg_exploitation_pct: {metrics['avg_exploitation_pct']:.4f}")

    lines.append("  Convergence:")
    lines.append(f"    average_convergence_rate: {metrics['average_convergence_rate']:.4f}")
    lines.append(f"    avg_improvement: {metrics['avg_improvement']:.4f}")
    lines.append(f"    success_rate: {metrics['success_rate']:.4f}")

    lines.append("  Stagnation:")
    lines.append(f"    longest_no_improvement_streak: {metrics['longest_no_improvement_streak']}")
    lines.append(f"    last_improvement_fraction: {metrics['last_improvement_fraction']:.4f}")

    return "\n".join(lines)
