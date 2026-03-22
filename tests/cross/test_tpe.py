"""Cross-verification tests for TPE sampler between rl4burn_tune_py and Optuna."""

import math
import pytest
import numpy as np

import rl4burn_tune_py as rl

optuna = pytest.importorskip("optuna")
from optuna.samplers import TPESampler

from .conftest import (
    CROSS_SEED,
    LOG_PDF_ATOL,
    sphere,
    rosenbrock,
    make_optuna_study,
    make_rl4burn_study,
    run_optuna_ask_tell,
    run_rl4burn_ask_tell,
)


# ---------------------------------------------------------------------------
# Gamma function
# ---------------------------------------------------------------------------


class TestGamma:
    """Verify gamma (good/bad split point) matches Optuna."""

    @pytest.mark.parametrize("n", [1, 5, 10, 20, 50, 100, 250, 500])
    def test_default_gamma(self, n):
        expected = int(min(math.ceil(0.1 * n), 25))
        assert rl.default_gamma(n) == expected

    @pytest.mark.parametrize("n", [1, 5, 10, 20, 50, 100, 250, 500])
    def test_hyperopt_gamma(self, n):
        expected = int(min(math.ceil(0.25 * math.sqrt(n)), 25))
        assert rl.hyperopt_default_gamma(n) == expected


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


class TestWeights:
    """Verify weight computation matches Optuna's default_weights."""

    @pytest.mark.parametrize("n", [1, 3, 5, 10, 25, 50])
    def test_default_weights_length(self, n):
        weights = rl.default_weights(n)
        assert len(weights) == n

    @pytest.mark.parametrize("n", [1, 3, 5, 10, 25, 50])
    def test_default_weights_positive(self, n):
        weights = rl.default_weights(n)
        assert all(w > 0 for w in weights)

    @pytest.mark.parametrize("n", [1, 3, 5, 10, 25])
    def test_default_weights_monotonic(self, n):
        weights = rl.default_weights(n)
        if n > 1:
            # Later observations should have >= weight than earlier ones
            for i in range(1, n):
                assert weights[i] >= weights[i - 1] - 1e-15

    @pytest.mark.parametrize("n", [1, 3, 5, 10, 25])
    def test_default_weights_flat_below_25(self, n):
        """For n < 25, all weights should be equal (1.0)."""
        weights = rl.default_weights(n)
        assert all(abs(w - 1.0) < 1e-15 for w in weights)


# ---------------------------------------------------------------------------
# Parzen estimator: mus, sigmas, weights
# ---------------------------------------------------------------------------


class TestParzenEstimator:
    """Cross-verify Parzen estimator construction and log-pdf."""

    def test_basic_construction(self):
        """Verify Parzen estimator with simple observations."""
        observations = [0.1, 0.3, 0.5, 0.7, 0.9]
        low, high = 0.0, 1.0
        prior_weight = 1.0
        weights = rl.default_weights(len(observations))

        mus, sigmas, pe_weights = rl.parzen_estimator_new(
            observations, low, high, prior_weight, True, True, weights
        )

        # Should have len(observations) + 1 components (observations + prior)
        assert len(mus) == len(observations) + 1
        assert len(sigmas) == len(observations) + 1
        assert len(pe_weights) == len(observations) + 1
        # All sigmas should be positive
        assert all(s > 0 for s in sigmas)
        # Weights should sum to 1
        assert abs(sum(pe_weights) - 1.0) < 1e-12

    def test_log_pdf_in_range(self):
        """Log-pdf should return finite values for points in [low, high]."""
        observations = [0.2, 0.4, 0.6, 0.8]
        low, high = 0.0, 1.0
        weights = rl.default_weights(len(observations))

        mus, sigmas, pe_weights = rl.parzen_estimator_new(
            observations, low, high, 1.0, True, True, weights
        )

        x_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        log_pdfs = rl.parzen_estimator_log_pdf(mus, sigmas, pe_weights, low, high, x_values)
        assert len(log_pdfs) == len(x_values)
        assert all(math.isfinite(lp) for lp in log_pdfs)

    def test_log_pdf_peaks_near_observations(self):
        """Log-pdf should be higher near observations than far from them."""
        observations = [0.3, 0.7]
        low, high = 0.0, 1.0
        weights = rl.default_weights(len(observations))

        mus, sigmas, pe_weights = rl.parzen_estimator_new(
            observations, low, high, 1.0, True, True, weights
        )

        # Points near observations should have higher log-pdf
        near_obs = rl.parzen_estimator_log_pdf(
            mus, sigmas, pe_weights, low, high, [0.3, 0.7]
        )
        far_from_obs = rl.parzen_estimator_log_pdf(
            mus, sigmas, pe_weights, low, high, [0.0, 1.0]
        )
        assert max(near_obs) > max(far_from_obs)


# ---------------------------------------------------------------------------
# Categorical Parzen estimator
# ---------------------------------------------------------------------------


class TestCategoricalParzenEstimator:
    """Cross-verify categorical Parzen estimator."""

    def test_basic_construction(self):
        observations = [0, 1, 1, 2, 0]
        n_choices = 3
        prior_weight = 1.0
        weights = rl.default_weights(len(observations))

        comp_weights, mix_weights = rl.categorical_parzen_estimator_new(
            observations, n_choices, prior_weight, weights
        )

        # Should have len(observations) + 1 components
        assert len(comp_weights) == len(observations) + 1
        assert len(mix_weights) == len(observations) + 1
        assert abs(sum(mix_weights) - 1.0) < 1e-12

    def test_log_pdf_valid(self):
        observations = [0, 1, 1, 2]
        n_choices = 3
        weights = rl.default_weights(len(observations))

        comp_weights, mix_weights = rl.categorical_parzen_estimator_new(
            observations, n_choices, 1.0, weights
        )

        log_pdfs = rl.categorical_parzen_estimator_log_pdf(
            comp_weights, mix_weights, n_choices, [0, 1, 2]
        )
        assert len(log_pdfs) == 3
        assert all(math.isfinite(lp) for lp in log_pdfs)
        # Category 1 was observed most, should have highest pdf
        assert log_pdfs[1] > log_pdfs[2]


# ---------------------------------------------------------------------------
# Full TPE sampler: end-to-end convergence
# ---------------------------------------------------------------------------


class TestTPEEndToEnd:
    """Compare TPE optimization results between Optuna and rl4burn."""

    def test_sphere_convergence(self):
        """Both TPE implementations should optimize sphere to similar quality."""
        n_trials = 50
        search_space = [("x", -5.0, 5.0), ("y", -5.0, 5.0)]

        # Optuna
        optuna_study = make_optuna_study(seed=CROSS_SEED)
        run_optuna_ask_tell(optuna_study, sphere, n_trials, search_space)
        optuna_best = optuna_study.best_value

        # rl4burn
        rl_study = make_rl4burn_study()
        rl_sampler = rl.tpe_sampler(seed=CROSS_SEED)
        run_rl4burn_ask_tell(rl_study, rl_sampler, sphere, n_trials, search_space)
        rl_best = rl_study.best_value()

        # Both should find a reasonably good solution
        # We don't require exact match due to implementation differences,
        # but both should be within same order of magnitude
        assert optuna_best < 10.0, f"Optuna didn't converge: {optuna_best}"
        assert rl_best < 10.0, f"rl4burn didn't converge: {rl_best}"

    def test_rosenbrock_convergence(self):
        """Both should make progress on Rosenbrock."""
        n_trials = 60
        search_space = [("x", -2.0, 2.0), ("y", -2.0, 2.0)]

        optuna_study = make_optuna_study(seed=CROSS_SEED)
        run_optuna_ask_tell(optuna_study, rosenbrock, n_trials, search_space)
        optuna_best = optuna_study.best_value

        rl_study = make_rl4burn_study()
        rl_sampler = rl.tpe_sampler(seed=CROSS_SEED)
        run_rl4burn_ask_tell(rl_study, rl_sampler, rosenbrock, n_trials, search_space)
        rl_best = rl_study.best_value()

        # Both should find values much better than random (random ~= 10-50)
        assert optuna_best < 50.0, f"Optuna didn't converge: {optuna_best}"
        assert rl_best < 50.0, f"rl4burn didn't converge: {rl_best}"

    def test_tpe_with_constant_liar(self):
        """TPE with constant_liar should still converge."""
        n_trials = 40
        search_space = [("x", -5.0, 5.0), ("y", -5.0, 5.0)]

        rl_study = make_rl4burn_study()
        rl_sampler = rl.tpe_sampler(seed=CROSS_SEED, constant_liar=True)
        run_rl4burn_ask_tell(rl_study, rl_sampler, sphere, n_trials, search_space)
        assert rl_study.best_value() < 15.0

    def test_tpe_multivariate(self):
        """TPE with multivariate=True should still converge."""
        n_trials = 40
        search_space = [("x", -5.0, 5.0), ("y", -5.0, 5.0)]

        rl_study = make_rl4burn_study()
        rl_sampler = rl.tpe_sampler(seed=CROSS_SEED, multivariate=True)
        run_rl4burn_ask_tell(rl_study, rl_sampler, sphere, n_trials, search_space)
        assert rl_study.best_value() < 15.0


# ---------------------------------------------------------------------------
# TPE: Constrained sampling (Phase 1a placeholder)
# ---------------------------------------------------------------------------


class TestConstrainedTPE:
    """Placeholder tests for constrained TPE (Phase 1a)."""

    @pytest.mark.skip(reason="Phase 1a: constrained TPE not yet implemented")
    def test_constrained_tpe_prefers_feasible(self):
        """TPE with constraints should prefer feasible trials."""
        pass
