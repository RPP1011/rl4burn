"""Shared fixtures for cross-verification tests between rl4burn_tune_py and optuna."""

import math
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Seeded RNG fixtures
# ---------------------------------------------------------------------------

CROSS_SEED = 42


@pytest.fixture
def seed():
    """Canonical seed shared across both libraries."""
    return CROSS_SEED


@pytest.fixture
def rng():
    """Seeded NumPy RNG for deterministic test data generation."""
    return np.random.RandomState(CROSS_SEED)


# ---------------------------------------------------------------------------
# Canonical objective functions
# ---------------------------------------------------------------------------


def sphere(x):
    """Sphere function: f(x) = sum(x_i^2). Minimum at origin."""
    return sum(xi ** 2 for xi in x)


def rosenbrock(x):
    """Rosenbrock function. Minimum at (1, 1, ..., 1)."""
    total = 0.0
    for i in range(len(x) - 1):
        total += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
    return total


def styblinski_tang(x):
    """Styblinski-Tang function. Minimum ~= -39.16599 * d at x_i ~= -2.903534."""
    return sum(xi ** 4 - 16.0 * xi ** 2 + 5.0 * xi for xi in x) / 2.0


def zdt1(x):
    """ZDT1 multi-objective test function. x in [0,1]^n, returns (f1, f2)."""
    n = len(x)
    f1 = x[0]
    g = 1.0 + 9.0 * sum(x[1:]) / (n - 1)
    f2 = g * (1.0 - math.sqrt(f1 / g))
    return f1, f2


@pytest.fixture
def sphere_fn():
    return sphere


@pytest.fixture
def rosenbrock_fn():
    return rosenbrock


@pytest.fixture
def styblinski_tang_fn():
    return styblinski_tang


@pytest.fixture
def zdt1_fn():
    return zdt1


# ---------------------------------------------------------------------------
# Helpers for building comparable studies
# ---------------------------------------------------------------------------


def make_optuna_study(direction="minimize", seed=CROSS_SEED, sampler=None):
    """Create an Optuna study with a seeded sampler."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if sampler is None:
        sampler = optuna.samplers.TPESampler(seed=seed)
    return optuna.create_study(direction=direction, sampler=sampler)


def make_rl4burn_study(direction="minimize"):
    """Create an rl4burn_tune_py study."""
    import rl4burn_tune_py as rl

    return rl.Study(direction)


def run_optuna_ask_tell(study, objective, n_trials, search_space):
    """Run n_trials on an Optuna study using the ask/tell interface.

    search_space: list of (name, low, high) tuples for float params.
    objective: callable that takes a dict of params and returns a value.
    """
    for _ in range(n_trials):
        trial = study.ask()
        params = {}
        for name, low, high in search_space:
            params[name] = trial.suggest_float(name, low, high)
        value = objective(list(params.values()))
        study.tell(trial, value)
    return study


def run_rl4burn_ask_tell(study, sampler, objective, n_trials, search_space):
    """Run n_trials on an rl4burn study using the ask/tell interface.

    search_space: list of (name, low, high) tuples for float params.
    objective: callable that takes a list of param values and returns a value.
    """
    import rl4burn_tune_py as rl

    for _ in range(n_trials):
        trial = study.ask()
        values = []
        for name, low, high in search_space:
            v = trial.suggest_float(name, low, high, sampler, study)
            values.append(v)
        value = objective(values)
        study.tell(trial, value)
    return study


# ---------------------------------------------------------------------------
# Tolerance constants
# ---------------------------------------------------------------------------

FLOAT_ATOL = 1e-12
FLOAT_RTOL = 1e-6
LOG_PDF_ATOL = 1e-6
