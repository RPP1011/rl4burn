#!/usr/bin/env python3
"""Differential tests: compare Rust TPE implementation against Optuna.

Each test calls the same function in both Optuna and our Rust bindings
with identical inputs and asserts the outputs match.

Usage:
    maturin build -m crates/rl4burn-tune-py/Cargo.toml --release
    pip install target/wheels/rl4burn_tune_py-*.whl --force-reinstall
    python3 scripts/test_tpe_differential.py
"""

import numpy as np
import pytest
from scipy.stats import norm as scipy_norm

# Optuna internals
from optuna.samplers._tpe.sampler import (
    default_gamma as optuna_default_gamma,
    default_weights as optuna_default_weights,
    hyperopt_default_gamma as optuna_hyperopt_default_gamma,
)
from optuna.samplers._tpe.parzen_estimator import (
    _ParzenEstimator,
    _ParzenEstimatorParameters,
)
from optuna.distributions import FloatDistribution, CategoricalDistribution

# Rust bindings
import rl4burn_tune_py as rust

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
EXACT = 0  # integer comparisons
TOL = 1e-12  # simple float arithmetic
SIGMA_TOL = 1e-10  # divisions / powf
LOG_PDF_TOL = 1e-6  # log-sum-exp accumulation


def assert_close(actual, expected, tol, msg=""):
    # Handle infinities: both -inf or both +inf are equal
    if actual == expected:
        return
    diff = abs(actual - expected)
    denom = max(abs(expected), 1.0)
    assert diff / denom <= tol, (
        f"{msg}: expected {expected}, got {actual}, "
        f"rel_diff={diff/denom:.2e}"
    )


def assert_vec_close(actual, expected, tol, msg=""):
    assert len(actual) == len(expected), (
        f"{msg}: length mismatch {len(actual)} vs {len(expected)}"
    )
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert_close(a, e, tol, f"{msg}[{i}]")


# ===================================================================
# 1. default_gamma
# ===================================================================
@pytest.mark.parametrize("n", [0, 1, 5, 9, 10, 11, 24, 25, 50, 100, 250, 1000])
def test_default_gamma(n):
    expected = optuna_default_gamma(n)
    actual = rust.default_gamma(n)
    assert actual == expected, f"default_gamma({n}): rust={actual}, optuna={expected}"


# ===================================================================
# 2. hyperopt_default_gamma
# ===================================================================
@pytest.mark.parametrize("n", [0, 1, 4, 9, 16, 17, 25, 100, 400, 1000, 10000])
def test_hyperopt_default_gamma(n):
    expected = optuna_hyperopt_default_gamma(n)
    actual = rust.hyperopt_default_gamma(n)
    assert actual == expected, (
        f"hyperopt_default_gamma({n}): rust={actual}, optuna={expected}"
    )


# ===================================================================
# 3. default_weights
# ===================================================================
@pytest.mark.parametrize("n", [0, 1, 5, 10, 24, 25, 26, 30, 50, 100])
def test_default_weights(n):
    expected = optuna_default_weights(n).tolist()
    actual = rust.default_weights(n)
    assert_vec_close(actual, expected, TOL, f"default_weights({n})")


# ===================================================================
# 4. gaussian_log_pdf vs scipy
# ===================================================================
GAUSSIAN_CASES = [
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (3.0, 0.0, 1.0),
    (0.5, 0.5, 0.1),
    (0.0, 0.0, 0.001),
    (100.0, 0.0, 1.0),
    (-2.0, 1.0, 0.5),
    (0.5, 0.0, 10.0),
]


@pytest.mark.parametrize("x,mu,sigma", GAUSSIAN_CASES)
def test_gaussian_log_pdf(x, mu, sigma):
    expected = scipy_norm.logpdf(x, loc=mu, scale=sigma)
    actual = rust.gaussian_log_pdf(x, mu, sigma)
    assert_close(actual, expected, SIGMA_TOL, f"gaussian_log_pdf({x},{mu},{sigma})")


# ===================================================================
# 5. ParzenEstimator — compare against Optuna's _ParzenEstimator
# ===================================================================

def _make_optuna_pe(observations, low=0.0, high=1.0, prior_weight=1.0,
                    consider_magic_clip=True, consider_endpoints=False,
                    weights_fn=None):
    """Build an Optuna _ParzenEstimator for a FloatDistribution."""
    if weights_fn is None:
        weights_fn = lambda n: np.ones(n)
    params = _ParzenEstimatorParameters(
        prior_weight=prior_weight,
        consider_magic_clip=consider_magic_clip,
        consider_endpoints=consider_endpoints,
        weights=weights_fn,
        multivariate=False,
        categorical_distance_func={},
    )
    search_space = {"x": FloatDistribution(low, high)}
    obs = {"x": np.array(observations, dtype=float)}
    pe = _ParzenEstimator(obs, search_space, params)
    md = pe._mixture_distribution
    d = md.distributions[0]
    return {
        "mus": d.mu.tolist(),
        "sigmas": d.sigma.tolist(),
        "weights": md.weights.tolist(),
        "pe": pe,
    }


def _make_rust_pe(observations, low=0.0, high=1.0, prior_weight=1.0,
                  consider_magic_clip=True, consider_endpoints=False,
                  weights_fn=None):
    """Build a Rust ParzenEstimator with the same inputs."""
    n = len(observations)
    if weights_fn is None:
        weights = [1.0] * n
    else:
        weights = weights_fn(n).tolist() if hasattr(weights_fn(n), 'tolist') else list(weights_fn(n))
    mus, sigmas, ws = rust.parzen_estimator_new(
        observations, low, high, prior_weight,
        consider_magic_clip, consider_endpoints, weights,
    )
    return {"mus": mus, "sigmas": sigmas, "weights": ws}


PE_CASES = [
    {"name": "five_basic", "obs": [0.1, 0.3, 0.5, 0.7, 0.9]},
    {"name": "single", "obs": [0.3]},
    {"name": "three_even", "obs": [0.1, 0.5, 0.9]},
    {"name": "prior_at_start", "obs": [0.6, 0.7, 0.8, 0.9]},
    {"name": "prior_at_end", "obs": [0.1, 0.2, 0.3]},
    {"name": "clustered", "obs": [0.49, 0.50, 0.51]},
    {"name": "boundary_low", "obs": [0.0, 0.01, 0.02]},
    {"name": "boundary_high", "obs": [0.98, 0.99, 1.0]},
]


@pytest.mark.parametrize("case", PE_CASES, ids=lambda c: c["name"])
def test_parzen_estimator_mus(case):
    optuna_result = _make_optuna_pe(case["obs"])
    rust_result = _make_rust_pe(case["obs"])
    assert_vec_close(
        rust_result["mus"], optuna_result["mus"], TOL,
        f"PE mus ({case['name']})"
    )


@pytest.mark.parametrize("case", PE_CASES, ids=lambda c: c["name"])
def test_parzen_estimator_sigmas(case):
    optuna_result = _make_optuna_pe(case["obs"])
    rust_result = _make_rust_pe(case["obs"])
    assert_vec_close(
        rust_result["sigmas"], optuna_result["sigmas"], SIGMA_TOL,
        f"PE sigmas ({case['name']})"
    )


@pytest.mark.parametrize("case", PE_CASES, ids=lambda c: c["name"])
def test_parzen_estimator_weights(case):
    optuna_result = _make_optuna_pe(case["obs"])
    rust_result = _make_rust_pe(case["obs"])
    assert_vec_close(
        rust_result["weights"], optuna_result["weights"], SIGMA_TOL,
        f"PE weights ({case['name']})"
    )


# ===================================================================
# 6. ParzenEstimator log_pdf — compare against Optuna
# ===================================================================
X_TEST = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


@pytest.mark.parametrize("case", PE_CASES, ids=lambda c: c["name"])
def test_parzen_estimator_log_pdf(case):
    optuna_result = _make_optuna_pe(case["obs"])
    rust_result = _make_rust_pe(case["obs"])

    optuna_log_pdfs = optuna_result["pe"].log_pdf({"x": np.array(X_TEST)}).tolist()
    rust_log_pdfs = rust.parzen_estimator_log_pdf(
        rust_result["mus"], rust_result["sigmas"], rust_result["weights"],
        0.0, 1.0, X_TEST,
    )

    assert_vec_close(
        rust_log_pdfs, optuna_log_pdfs, LOG_PDF_TOL,
        f"PE log_pdf ({case['name']})"
    )


# ===================================================================
# 7. ParzenEstimator with non-uniform weights
# ===================================================================
def test_parzen_estimator_ramp_weights():
    """Test with ramp weights (n=30, using default_weights)."""
    obs = np.linspace(0.0, 1.0, 30).tolist()
    optuna_result = _make_optuna_pe(
        obs, weights_fn=lambda n: optuna_default_weights(n)
    )
    rust_result = _make_rust_pe(
        obs, weights_fn=lambda n: optuna_default_weights(n)
    )
    # Compare log_pdf at test points
    optuna_log_pdfs = optuna_result["pe"].log_pdf({"x": np.array(X_TEST)}).tolist()
    rust_log_pdfs = rust.parzen_estimator_log_pdf(
        rust_result["mus"], rust_result["sigmas"], rust_result["weights"],
        0.0, 1.0, X_TEST,
    )
    assert_vec_close(
        rust_log_pdfs, optuna_log_pdfs, LOG_PDF_TOL,
        "PE log_pdf (ramp_weights)"
    )


# ===================================================================
# 8. ParzenEstimator without magic clip
# ===================================================================
def test_parzen_estimator_no_magic_clip():
    obs = [0.1, 0.3, 0.5, 0.7, 0.9]
    optuna_result = _make_optuna_pe(obs, consider_magic_clip=False)
    rust_result = _make_rust_pe(obs, consider_magic_clip=False)
    optuna_log_pdfs = optuna_result["pe"].log_pdf({"x": np.array(X_TEST)}).tolist()
    rust_log_pdfs = rust.parzen_estimator_log_pdf(
        rust_result["mus"], rust_result["sigmas"], rust_result["weights"],
        0.0, 1.0, X_TEST,
    )
    assert_vec_close(
        rust_log_pdfs, optuna_log_pdfs, LOG_PDF_TOL,
        "PE log_pdf (no_magic_clip)"
    )


# ===================================================================
# 9. ParzenEstimator with consider_endpoints=True
# ===================================================================
def test_parzen_estimator_with_endpoints():
    obs = [0.1, 0.3, 0.5, 0.7, 0.9]
    optuna_result = _make_optuna_pe(obs, consider_endpoints=True)
    rust_result = _make_rust_pe(obs, consider_endpoints=True)
    assert_vec_close(
        rust_result["sigmas"], optuna_result["sigmas"], SIGMA_TOL,
        "PE sigmas (consider_endpoints)"
    )


# ===================================================================
# 10. CategoricalParzenEstimator — compare against Optuna
# ===================================================================

def _make_optuna_cpe(observations, n_choices, prior_weight=1.0, weights_fn=None):
    """Build an Optuna _ParzenEstimator for a CategoricalDistribution."""
    if weights_fn is None:
        weights_fn = lambda n: np.ones(n)
    params = _ParzenEstimatorParameters(
        prior_weight=prior_weight,
        consider_magic_clip=True,
        consider_endpoints=False,
        weights=weights_fn,
        multivariate=False,
        categorical_distance_func={},
    )
    choices = [f"c{i}" for i in range(n_choices)]
    search_space = {"x": CategoricalDistribution(choices)}
    obs = {"x": np.array(observations, dtype=float)}
    pe = _ParzenEstimator(obs, search_space, params)
    return pe


CPE_CASES = [
    {"name": "empty_3", "obs": [], "n": 3, "pw": 1.0},
    {"name": "single_0", "obs": [0], "n": 3, "pw": 1.0},
    {"name": "uniform_3", "obs": [0, 1, 2], "n": 3, "pw": 1.0},
    {"name": "concentrated", "obs": [0, 0, 0, 0, 0], "n": 3, "pw": 1.0},
    {"name": "binary", "obs": [0, 0, 1, 1, 0], "n": 2, "pw": 1.0},
    {"name": "many_choices", "obs": [0, 3, 7, 9], "n": 10, "pw": 1.0},
    {"name": "no_prior", "obs": [0, 0, 1], "n": 3, "pw": 0.0},
    {"name": "heavy_prior", "obs": [0, 0, 1], "n": 3, "pw": 10.0},
]


@pytest.mark.parametrize("case", CPE_CASES, ids=lambda c: c["name"])
def test_categorical_log_pdf(case):
    """Compare categorical log_pdf between Optuna and Rust."""
    optuna_pe = _make_optuna_cpe(case["obs"], case["n"], case["pw"])
    indices = list(range(case["n"]))
    optuna_log_pdfs = optuna_pe.log_pdf(
        {"x": np.array(indices, dtype=float)}
    ).tolist()

    weights = [1.0] * len(case["obs"])
    comp_w, mix_w = rust.categorical_parzen_estimator_new(
        case["obs"], case["n"], case["pw"], weights,
    )
    rust_log_pdfs = rust.categorical_parzen_estimator_log_pdf(
        comp_w, mix_w, case["n"], indices,
    )

    assert_vec_close(
        rust_log_pdfs, optuna_log_pdfs, LOG_PDF_TOL,
        f"CPE log_pdf ({case['name']})"
    )


# ===================================================================
# 11. calculate_order — basic correctness
# ===================================================================
ORDER_CASES = [
    ([], []),
    ([5.0], [0]),
    ([1.0, 2.0, 3.0], [0, 1, 2]),
    ([3.0, 1.0, 2.0], [1, 2, 0]),
    ([0.5, 0.1, 0.9, 0.3], [1, 3, 0, 2]),
]


@pytest.mark.parametrize("values,expected", ORDER_CASES)
def test_calculate_order(values, expected):
    actual = rust.calculate_order(values)
    sorted_values = [values[i] for i in actual]
    assert sorted_values == sorted(values), (
        f"calculate_order({values}): permutation {actual} doesn't sort correctly"
    )
    if expected:
        no_dupes = len(set(values)) == len(values)
        if no_dupes:
            assert actual == expected, (
                f"calculate_order({values}): {actual} != {expected}"
            )


# ===================================================================
# 12. Transform functions — roundtrip consistency
# ===================================================================
FLOAT_TRANSFORM_CASES = [
    (5.0, 0.0, 10.0, False),
    (0.0, 0.0, 10.0, False),
    (10.0, 0.0, 10.0, False),
    (10.0, 1.0, 100.0, True),
    (1.0, 1.0, 100.0, True),
    (100.0, 1.0, 100.0, True),
    (0.01, 0.001, 1.0, True),
]


@pytest.mark.parametrize("value,low,high,log", FLOAT_TRANSFORM_CASES)
def test_float_transform_roundtrip(value, low, high, log):
    internal = rust.float_transform_to_internal(value, low, high, log)
    assert 0.0 <= internal <= 1.0, f"internal {internal} not in [0,1]"
    back = rust.float_transform_from_internal(internal, low, high, log)
    assert_close(back, value, SIGMA_TOL, f"float roundtrip({value},{low},{high},{log})")


INT_TRANSFORM_CASES = [
    (5, 0, 10, False),
    (0, 0, 10, False),
    (10, 0, 10, False),
    (50, 0, 100, False),
    (10, 1, 1000, True),
    (1, 1, 1000, True),
    (1000, 1, 1000, True),
]


@pytest.mark.parametrize("value,low,high,log", INT_TRANSFORM_CASES)
def test_int_transform_roundtrip(value, low, high, log):
    internal = rust.int_transform_to_internal_py(value, low, high, log)
    assert 0.0 <= internal <= 1.0, f"internal {internal} not in [0,1]"
    back = rust.int_transform_from_internal_py(internal, low, high, log)
    assert back == value, f"int roundtrip({value},{low},{high},{log}): got {back}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
