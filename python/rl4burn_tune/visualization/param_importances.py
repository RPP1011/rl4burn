"""Plot parameter importances as a horizontal bar chart."""


def plot_param_importances(study, importances=None):
    """Plot parameter importances for a study.

    Args:
        study: An rl4burn_tune_py.Study instance.
        importances: Dict mapping param names to importance scores (0-1).
            If None, computes a simple variance-based importance.

    Returns:
        A matplotlib Figure object.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")

    if importances is None:
        importances = _compute_simple_importance(study)

    if not importances:
        raise ValueError("No parameter importances to plot.")

    # Sort by importance
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5)))
    bars = ax.barh(range(len(names)), scores, color="steelblue")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Importance")
    ax.set_title("Parameter Importances")
    ax.invert_yaxis()  # Most important at top
    ax.set_xlim(0, max(scores) * 1.1 if scores else 1)

    # Add value labels
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    return fig


def _compute_simple_importance(study):
    """Compute a simple variance-based parameter importance."""
    trials = study.get_trials()
    completed = [t for t in trials if t["state"] == "Complete"]

    if len(completed) < 2:
        return {}

    # Collect param values and objective values
    param_names = set()
    for t in completed:
        param_names.update(t["params"].keys())

    importances = {}
    total_var = _variance([t["value"] for t in completed])

    if total_var < 1e-15:
        return {name: 1.0 / len(param_names) for name in param_names}

    for name in param_names:
        # Compute correlation between param and objective
        param_vals = [t["params"].get(name, 0.0) for t in completed]
        obj_vals = [t["value"] for t in completed]
        corr = abs(_correlation(param_vals, obj_vals))
        importances[name] = corr

    # Normalize
    total = sum(importances.values())
    if total > 0:
        importances = {k: v / total for k, v in importances.items()}

    return importances


def _variance(values):
    """Compute variance of a list of values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / (len(values) - 1)


def _correlation(x, y):
    """Compute Pearson correlation between two lists."""
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / (n - 1)
    std_x = _variance(x) ** 0.5
    std_y = _variance(y) ** 0.5
    if std_x < 1e-15 or std_y < 1e-15:
        return 0.0
    return cov / (std_x * std_y)
