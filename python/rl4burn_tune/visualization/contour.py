"""Plot objective contour for two parameters."""


def plot_contour(study, params=None):
    """Plot a contour plot of objective values for two parameters.

    Args:
        study: An rl4burn_tune_py.Study instance.
        params: Tuple of two parameter names. If None, picks the first two.

    Returns:
        A matplotlib Figure object.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError("matplotlib and numpy are required for visualization.")

    trials = study.get_trials()
    completed = [t for t in trials if t["state"] == "Complete"]

    if len(completed) < 3:
        raise ValueError("Need at least 3 completed trials for contour plot.")

    # Determine parameters
    if params is None:
        all_params = set()
        for t in completed:
            all_params.update(t["params"].keys())
        param_list = sorted(all_params)
        if len(param_list) < 2:
            raise ValueError("Need at least 2 parameters for contour plot.")
        params = (param_list[0], param_list[1])

    param_x, param_y = params

    # Extract data
    x_vals = [t["params"].get(param_x) for t in completed if param_x in t["params"] and param_y in t["params"]]
    y_vals = [t["params"].get(param_y) for t in completed if param_x in t["params"] and param_y in t["params"]]
    z_vals = [t["value"] for t in completed if param_x in t["params"] and param_y in t["params"]]

    if len(x_vals) < 3:
        raise ValueError("Not enough trials with both parameters.")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot with color representing objective value
    scatter = ax.scatter(x_vals, y_vals, c=z_vals, cmap="viridis", s=40, edgecolors="white", linewidth=0.5)
    plt.colorbar(scatter, ax=ax, label="Objective Value")

    # Try to create a contour from the scattered data
    try:
        from matplotlib.tri import Triangulation
        triang = Triangulation(x_vals, y_vals)
        ax.tricontourf(triang, z_vals, levels=20, cmap="viridis", alpha=0.3)
    except Exception:
        pass  # Contour may fail for degenerate point sets

    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title(f"Objective Contour: {param_x} vs {param_y}")

    fig.tight_layout()
    return fig
