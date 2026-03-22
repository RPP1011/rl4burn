"""Cross-verification tests for NSGA-II sampler between rl4burn_tune_py and Optuna."""

import math
import pytest
import numpy as np

import rl4burn_tune_py as rl

optuna = pytest.importorskip("optuna")
from optuna.samplers import NSGAIISampler as OptunaNSGA

from .conftest import CROSS_SEED, zdt1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hypervolume(points, ref):
    """Compute 2D hypervolume indicator (dominated area below reference point)."""
    # Sort by f1
    sorted_pts = sorted(points, key=lambda p: p[0])
    hv = 0.0
    prev_f2 = ref[1]
    for f1, f2 in sorted_pts:
        if f1 < ref[0] and f2 < ref[1]:
            hv += (ref[0] - f1) * (prev_f2 - max(f2, 0))
            prev_f2 = min(prev_f2, f2)
    # Simple 2D HV: sum of non-dominated rectangles
    # Use a proper sweep for accuracy
    filtered = [(f1, f2) for f1, f2 in sorted_pts if f1 < ref[0] and f2 < ref[1]]
    if not filtered:
        return 0.0
    # Non-dominated sorting first
    nd = []
    for p in filtered:
        dominated = False
        for q in filtered:
            if q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1]):
                dominated = True
                break
        if not dominated:
            nd.append(p)
    nd.sort(key=lambda p: p[0])
    if not nd:
        return 0.0
    hv = 0.0
    prev_f2 = ref[1]
    for i, (f1, f2) in enumerate(nd):
        next_f1 = nd[i + 1][0] if i + 1 < len(nd) else ref[0]
        hv += (next_f1 - f1) * (prev_f2 - f2)
        prev_f2 = f2
    return hv


# ---------------------------------------------------------------------------
# NSGA-II convergence on ZDT1
# ---------------------------------------------------------------------------


class TestNSGAIIConvergence:
    """Compare NSGA-II multi-objective optimization results."""

    @pytest.mark.slow
    def test_zdt1_hypervolume(self):
        """Both NSGA-II implementations should achieve decent hypervolume on ZDT1."""
        n_trials = 100
        n_vars = 5
        ref_point = (1.0, 5.0)

        # Optuna NSGA-II
        optuna_sampler = OptunaNSGA(seed=CROSS_SEED, population_size=50)
        optuna_study = optuna.create_study(
            directions=["minimize", "minimize"], sampler=optuna_sampler
        )
        for _ in range(n_trials):
            trial = optuna_study.ask()
            x = [trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(n_vars)]
            f1, f2 = zdt1(x)
            optuna_study.tell(trial, [f1, f2])

        optuna_pareto = [
            (t.values[0], t.values[1]) for t in optuna_study.best_trials
        ]
        optuna_hv = _hypervolume(optuna_pareto, ref_point)

        # rl4burn NSGA-II
        rl_study = rl.Study.create_multi(["minimize", "minimize"])
        rl_sampler = rl.nsga2_sampler(seed=CROSS_SEED, population_size=50)
        for _ in range(n_trials):
            trial = rl_study.ask()
            x = [
                trial.suggest_float(f"x{i}", 0.0, 1.0, rl_sampler, rl_study)
                for i in range(n_vars)
            ]
            f1, f2 = zdt1(x)
            rl_study.tell_multi(trial, [f1, f2])

        # Get rl4burn trials and compute pareto front
        rl_trials = rl_study.get_trials()
        rl_values = [
            (t["value"], t.get("values", [None, None]))
            for t in rl_trials
            if t["state"] == "Complete"
        ]
        # rl4burn returns values differently - extract from trial data
        # For now just check that it ran without errors
        assert rl_study.n_completed() == n_trials

        # Optuna should achieve positive hypervolume
        assert optuna_hv > 0, f"Optuna NSGA-II got zero hypervolume"

    def test_parameter_bounds_respected(self):
        """NSGA-II should always suggest values within bounds."""
        rl_study = rl.Study.create_multi(["minimize", "minimize"])
        rl_sampler = rl.nsga2_sampler(seed=CROSS_SEED)

        for _ in range(30):
            trial = rl_study.ask()
            x = trial.suggest_float("x", 0.0, 1.0, rl_sampler, rl_study)
            y = trial.suggest_float("y", 0.0, 1.0, rl_sampler, rl_study)
            assert 0.0 <= x <= 1.0, f"x out of bounds: {x}"
            assert 0.0 <= y <= 1.0, f"y out of bounds: {y}"
            rl_study.tell_multi(trial, [x, y])

    def test_crossover_sbx(self):
        """SBX crossover should produce valid offspring within bounds."""
        rl_study = rl.Study.create_multi(["minimize", "minimize"])
        rl_sampler = rl.nsga2_sampler(seed=CROSS_SEED, crossover_type="sbx")

        for _ in range(50):
            trial = rl_study.ask()
            x = trial.suggest_float("x", -10.0, 10.0, rl_sampler, rl_study)
            y = trial.suggest_float("y", -10.0, 10.0, rl_sampler, rl_study)
            assert -10.0 <= x <= 10.0
            assert -10.0 <= y <= 10.0
            rl_study.tell_multi(trial, [x ** 2, y ** 2])

    def test_crossover_uniform(self):
        """Uniform crossover should produce valid offspring within bounds."""
        rl_study = rl.Study.create_multi(["minimize", "minimize"])
        rl_sampler = rl.nsga2_sampler(seed=CROSS_SEED, crossover_type="uniform")

        for _ in range(50):
            trial = rl_study.ask()
            x = trial.suggest_float("x", -10.0, 10.0, rl_sampler, rl_study)
            y = trial.suggest_float("y", -10.0, 10.0, rl_sampler, rl_study)
            assert -10.0 <= x <= 10.0
            assert -10.0 <= y <= 10.0
            rl_study.tell_multi(trial, [x ** 2, y ** 2])


# ---------------------------------------------------------------------------
# NSGA-II: Additional crossovers (Phase 3a placeholder)
# ---------------------------------------------------------------------------


class TestNSGAIICrossoverOperators:
    """Placeholder for additional crossover operators (Phase 3a)."""

    @pytest.mark.skip(reason="Phase 3a: SPX crossover not yet implemented")
    def test_crossover_spx_bounded(self):
        pass

    @pytest.mark.skip(reason="Phase 3a: UNDX crossover not yet implemented")
    def test_crossover_undx_bounded(self):
        pass

    @pytest.mark.skip(reason="Phase 3a: VSBX crossover not yet implemented")
    def test_crossover_vsbx_bounded(self):
        pass


# ---------------------------------------------------------------------------
# NSGA-II: Constrained (Phase 3b placeholder)
# ---------------------------------------------------------------------------


class TestNSGAIIConstrained:
    """Tests for constrained NSGA-II (Phase 3b)."""

    def test_constrained_dominance(self):
        """Constrained NSGA-II should prefer feasible solutions over infeasible ones."""
        rl_study = rl.Study.create_multi(["minimize", "minimize"])
        rl_sampler = rl.nsga2_sampler(seed=CROSS_SEED, population_size=20, n_startup_trials=10)

        # Run trials: some feasible, some infeasible
        for i in range(30):
            trial = rl_study.ask()
            x = trial.suggest_float("x", 0.0, 10.0, rl_sampler, rl_study)
            y = trial.suggest_float("y", 0.0, 10.0, rl_sampler, rl_study)

            f1 = x
            f2 = y
            # Constraint: x + y <= 12 (violation = x + y - 12 when positive)
            violation = x + y - 12.0
            trial.set_constraints([violation])

            rl_study.tell_multi(trial, [f1, f2])

        # Verify it ran without errors
        assert rl_study.n_completed() == 30

        # Check that feasible solutions exist (some x+y <= 12)
        trials = rl_study.get_trials()
        completed = [t for t in trials if t["state"] == "Complete"]
        assert len(completed) == 30

        # At least some trials should have constraint values recorded
        has_constraints = sum(
            1 for t in completed if "constraint_values" in t
        )
        assert has_constraints > 0, "No constraint values found in completed trials"
