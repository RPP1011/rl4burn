"""Plot optimization history (objective value vs trial number)."""


def plot_optimization_history(study, target_name="Objective Value"):
    """Plot the optimization history of a study.

    Args:
        study: An rl4burn_tune_py.Study instance.
        target_name: Label for the y-axis.

    Returns:
        A matplotlib Figure object.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")

    trials = study.get_trials()
    completed = [t for t in trials if t["state"] == "Complete"]

    if not completed:
        raise ValueError("No completed trials to plot.")

    numbers = [t["number"] for t in completed]
    values = [t["value"] for t in completed]

    # Compute running best
    best_values = []
    current_best = None
    for v in values:
        if current_best is None or v < current_best:
            current_best = v
        best_values.append(current_best)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(numbers, values, alpha=0.5, label="Objective Value", s=20)
    ax.plot(numbers, best_values, color="red", linewidth=2, label="Best Value")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel(target_name)
    ax.set_title("Optimization History")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig
