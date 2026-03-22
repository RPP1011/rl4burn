"""Cross-verification tests for GP sampler between rl4burn_tune_py and Optuna."""

import pytest
import numpy as np

import rl4burn_tune_py as rl

optuna = pytest.importorskip("optuna")
from optuna.samplers import GPSampler as OptunaGPSampler

from .conftest import CROSS_SEED, sphere, rosenbrock, make_rl4burn_study, run_rl4burn_ask_tell


# ---------------------------------------------------------------------------
# GP sampler convergence
# ---------------------------------------------------------------------------


class TestGPConvergence:
    """Compare GP optimization results."""

    @pytest.mark.slow
    def test_sphere_convergence(self):
        """Both GP implementations should optimize sphere."""
        n_trials = 40
        search_space = [("x", -5.0, 5.0), ("y", -5.0, 5.0)]

        # Optuna GP (requires torch)
        try:
            optuna_sampler = OptunaGPSampler(seed=CROSS_SEED)
            optuna_study = optuna.create_study(direction="minimize", sampler=optuna_sampler)
            for _ in range(n_trials):
                trial = optuna_study.ask()
                x = trial.suggest_float("x", -5.0, 5.0)
                y = trial.suggest_float("y", -5.0, 5.0)
                optuna_study.tell(trial, sphere([x, y]))
            optuna_best = optuna_study.best_value
            assert optuna_best < 5.0, f"Optuna GP didn't converge: {optuna_best}"
        except (ImportError, ModuleNotFoundError):
            pytest.skip("torch not installed (required for Optuna GPSampler)")

        # rl4burn GP
        rl_study = make_rl4burn_study()
        rl_sampler = rl.gp_sampler(seed=CROSS_SEED)
        run_rl4burn_ask_tell(rl_study, rl_sampler, sphere, n_trials, search_space)
        rl_best = rl_study.best_value()
        assert rl_best < 5.0, f"rl4burn GP didn't converge: {rl_best}"

    @pytest.mark.slow
    def test_rosenbrock_convergence(self):
        """Both should make progress on Rosenbrock."""
        n_trials = 40
        search_space = [("x", -2.0, 2.0), ("y", -2.0, 2.0)]

        # Optuna GP (requires torch)
        try:
            optuna_sampler = OptunaGPSampler(seed=CROSS_SEED)
            optuna_study = optuna.create_study(direction="minimize", sampler=optuna_sampler)
            for _ in range(n_trials):
                trial = optuna_study.ask()
                x = trial.suggest_float("x", -2.0, 2.0)
                y = trial.suggest_float("y", -2.0, 2.0)
                optuna_study.tell(trial, rosenbrock([x, y]))
            optuna_best = optuna_study.best_value
            assert optuna_best < 50.0, f"Optuna GP didn't converge: {optuna_best}"
        except (ImportError, ModuleNotFoundError):
            pytest.skip("torch not installed (required for Optuna GPSampler)")

        # rl4burn GP
        rl_study = make_rl4burn_study()
        rl_sampler = rl.gp_sampler(seed=CROSS_SEED)
        run_rl4burn_ask_tell(rl_study, rl_sampler, rosenbrock, n_trials, search_space)
        rl_best = rl_study.best_value()
        assert rl_best < 50.0, f"rl4burn GP didn't converge: {rl_best}"

    def test_parameter_bounds_respected(self):
        """GP should always suggest values within bounds."""
        n_trials = 25
        rl_study = make_rl4burn_study()
        rl_sampler = rl.gp_sampler(seed=CROSS_SEED)

        for _ in range(n_trials):
            trial = rl_study.ask()
            x = trial.suggest_float("x", -3.0, 3.0, rl_sampler, rl_study)
            y = trial.suggest_float("y", -3.0, 3.0, rl_sampler, rl_study)
            assert -3.0 <= x <= 3.0, f"x out of bounds: {x}"
            assert -3.0 <= y <= 3.0, f"y out of bounds: {y}"
            rl_study.tell(trial, sphere([x, y]))


# ---------------------------------------------------------------------------
# GP: Multi-dimensional ARD (Phase 1d placeholder)
# ---------------------------------------------------------------------------


class TestGPMultiDimARD:
    """Placeholder for multi-dimensional GP with ARD (Phase 1d)."""

    @pytest.mark.skip(reason="Phase 1d: multi-dimensional GP not yet implemented")
    def test_gp_2d_correlated(self):
        """GP with ARD should handle correlated 2D functions."""
        pass
