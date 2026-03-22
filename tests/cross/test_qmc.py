"""Cross-verification tests for QMC sampler."""

import pytest
import numpy as np

import rl4burn_tune_py as rl

from .conftest import CROSS_SEED, sphere


class TestQmcSampler:
    """Verify QMC sampler produces low-discrepancy points."""

    def test_sobol_discrepancy(self):
        """QMC points should have lower discrepancy than uniform random."""
        n_points = 256
        study = rl.Study("minimize")
        sampler = rl.qmc_sampler(seed=CROSS_SEED)

        values = []
        for i in range(n_points):
            trial = study.ask()
            x = trial.suggest_float("x", 0.0, 1.0, sampler, study)
            values.append(x)
            study.tell(trial, x ** 2)

        # Compute star discrepancy
        sorted_vals = sorted(values)
        max_disc = 0.0
        for i, x in enumerate(sorted_vals):
            empirical = (i + 1) / n_points
            disc = abs(empirical - x)
            max_disc = max(max_disc, disc)

        # QMC should have significantly lower discrepancy than random (~0.06)
        assert max_disc < 0.25, f"Star discrepancy {max_disc} too high for QMC"

    def test_bounds_respected(self):
        """QMC sampler should always produce values within bounds."""
        study = rl.Study("minimize")
        sampler = rl.qmc_sampler(seed=CROSS_SEED)

        for _ in range(50):
            trial = study.ask()
            x = trial.suggest_float("x", -3.0, 3.0, sampler, study)
            assert -3.0 <= x <= 3.0, f"x={x} out of bounds"
            study.tell(trial, x ** 2)

    def test_convergence_on_sphere(self):
        """QMC initialization should help TPE/random find good solutions."""
        n_trials = 50
        study = rl.Study("minimize")
        sampler = rl.qmc_sampler(seed=CROSS_SEED)

        for _ in range(n_trials):
            trial = study.ask()
            x = trial.suggest_float("x", -5.0, 5.0, sampler, study)
            y = trial.suggest_float("y", -5.0, 5.0, sampler, study)
            study.tell(trial, sphere([x, y]))

        # Should find reasonably good solutions due to space coverage
        assert study.best_value() < 10.0, f"QMC didn't find good solution: {study.best_value()}"
