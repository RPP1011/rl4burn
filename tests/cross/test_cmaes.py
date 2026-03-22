"""Cross-verification tests for CMA-ES sampler between rl4burn_tune_py and Optuna."""

import pytest
import numpy as np

import rl4burn_tune_py as rl

optuna = pytest.importorskip("optuna")
from optuna.samplers import CmaEsSampler as OptunaCmaEs

from .conftest import CROSS_SEED, sphere, rosenbrock, make_rl4burn_study, run_rl4burn_ask_tell


# ---------------------------------------------------------------------------
# Basic CMA-ES convergence
# ---------------------------------------------------------------------------


class TestCmaESConvergence:
    """Compare CMA-ES optimization results."""

    @pytest.mark.slow
    def test_sphere_convergence(self):
        """Both CMA-ES implementations should optimize sphere reasonably."""
        n_trials = 50
        search_space = [("x", -5.0, 5.0), ("y", -5.0, 5.0)]

        # Optuna CMA-ES
        try:
            optuna_sampler = OptunaCmaEs(seed=CROSS_SEED)
            optuna_study = optuna.create_study(direction="minimize", sampler=optuna_sampler)
            for _ in range(n_trials):
                trial = optuna_study.ask()
                x = trial.suggest_float("x", -5.0, 5.0)
                y = trial.suggest_float("y", -5.0, 5.0)
                optuna_study.tell(trial, sphere([x, y]))
            optuna_best = optuna_study.best_value
            assert optuna_best < 5.0, f"Optuna CMA-ES didn't converge: {optuna_best}"
        except ImportError:
            pytest.skip("cmaes package not installed")

        # rl4burn CMA-ES
        rl_study = make_rl4burn_study()
        rl_sampler = rl.cmaes_sampler(seed=CROSS_SEED)
        run_rl4burn_ask_tell(rl_study, rl_sampler, sphere, n_trials, search_space)
        rl_best = rl_study.best_value()
        assert rl_best < 5.0, f"rl4burn CMA-ES didn't converge: {rl_best}"

    @pytest.mark.slow
    def test_rosenbrock_convergence(self):
        """Both should make progress on Rosenbrock."""
        n_trials = 80
        search_space = [("x", -2.0, 2.0), ("y", -2.0, 2.0)]

        # Optuna CMA-ES
        try:
            optuna_sampler = OptunaCmaEs(seed=CROSS_SEED)
            optuna_study = optuna.create_study(direction="minimize", sampler=optuna_sampler)
            for _ in range(n_trials):
                trial = optuna_study.ask()
                x = trial.suggest_float("x", -2.0, 2.0)
                y = trial.suggest_float("y", -2.0, 2.0)
                optuna_study.tell(trial, rosenbrock([x, y]))
            optuna_best = optuna_study.best_value
            assert optuna_best < 50.0, f"Optuna CMA-ES didn't converge: {optuna_best}"
        except ImportError:
            pytest.skip("cmaes package not installed")

        # rl4burn CMA-ES
        rl_study = make_rl4burn_study()
        rl_sampler = rl.cmaes_sampler(seed=CROSS_SEED)
        run_rl4burn_ask_tell(rl_study, rl_sampler, rosenbrock, n_trials, search_space)
        rl_best = rl_study.best_value()
        assert rl_best < 50.0, f"rl4burn CMA-ES didn't converge: {rl_best}"

    def test_parameter_bounds_respected(self):
        """CMA-ES should always suggest values within bounds."""
        n_trials = 30
        rl_study = make_rl4burn_study()
        rl_sampler = rl.cmaes_sampler(seed=CROSS_SEED)

        for _ in range(n_trials):
            trial = rl_study.ask()
            x = trial.suggest_float("x", -3.0, 3.0, rl_sampler, rl_study)
            y = trial.suggest_float("y", -3.0, 3.0, rl_sampler, rl_study)
            assert -3.0 <= x <= 3.0, f"x out of bounds: {x}"
            assert -3.0 <= y <= 3.0, f"y out of bounds: {y}"
            rl_study.tell(trial, sphere([x, y]))


# ---------------------------------------------------------------------------
# CMA-ES: Full covariance (Phase 1c placeholder)
# ---------------------------------------------------------------------------


class TestCmaESFullCovariance:
    """Placeholder for full-covariance CMA-ES (Phase 1c)."""

    @pytest.mark.skip(reason="Phase 1c: full covariance CMA-ES not yet implemented")
    def test_cmaes_correlated_params(self):
        """Full CMA-ES should handle correlated parameters better than separable."""
        pass
