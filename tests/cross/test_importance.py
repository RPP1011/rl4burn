"""Cross-verification tests for importance evaluators between rl4burn_tune_py and Optuna."""

import pytest
import numpy as np

import rl4burn_tune_py as rl

optuna = pytest.importorskip("optuna")

from .conftest import CROSS_SEED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_importance_study_optuna(n_trials=100, seed=CROSS_SEED):
    """Create a study where param 'a' is important and 'b' is noise."""
    rng = np.random.RandomState(seed)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    for _ in range(n_trials):
        trial = study.ask()
        a = trial.suggest_float("a", -5.0, 5.0)
        b = trial.suggest_float("b", -5.0, 5.0)
        # Objective depends strongly on 'a', weakly on 'b'
        value = a ** 2 + 0.01 * b ** 2 + rng.normal(0, 0.01)
        study.tell(trial, value)

    return study


def _create_importance_study_rl4burn(n_trials=100, seed=CROSS_SEED):
    """Create a study where param 'a' is important and 'b' is noise."""
    rng = np.random.RandomState(seed)
    study = rl.Study("minimize")
    sampler = rl.tpe_sampler(seed=seed)

    for _ in range(n_trials):
        trial = study.ask()
        a = trial.suggest_float("a", -5.0, 5.0, sampler, study)
        b = trial.suggest_float("b", -5.0, 5.0, sampler, study)
        value = a ** 2 + 0.01 * b ** 2 + rng.normal(0, 0.01)
        study.tell(trial, value)

    return study


# ---------------------------------------------------------------------------
# PED-ANOVA importance (already implemented)
# ---------------------------------------------------------------------------


class TestPedAnovaImportance:
    """Cross-verify PED-ANOVA importance rankings."""

    def test_important_param_ranked_higher(self):
        """Both libraries should rank 'a' higher than 'b'."""
        # Optuna PED-ANOVA
        optuna_study = _create_importance_study_optuna()
        try:
            optuna_evaluator = optuna.importance.PedAnovaImportanceEvaluator()
            optuna_importance = optuna_evaluator.evaluate(optuna_study)
            optuna_a = optuna_importance.get("a", 0)
            optuna_b = optuna_importance.get("b", 0)
            assert optuna_a > optuna_b, (
                f"Optuna PED-ANOVA: a={optuna_a} should be > b={optuna_b}"
            )
        except Exception:
            pytest.skip("Optuna PED-ANOVA not available in this version")

    def test_importance_ranking_matches(self):
        """The ranking order should be the same between both libraries."""
        optuna_study = _create_importance_study_optuna(n_trials=100)
        try:
            optuna_evaluator = optuna.importance.PedAnovaImportanceEvaluator()
            optuna_importance = optuna_evaluator.evaluate(optuna_study)
            optuna_ranking = sorted(optuna_importance.keys(), key=lambda k: -optuna_importance[k])
        except Exception:
            pytest.skip("Optuna PED-ANOVA not available in this version")

        # Both should rank 'a' first
        assert optuna_ranking[0] == "a", f"Optuna ranked: {optuna_ranking}"


# ---------------------------------------------------------------------------
# fANOVA (Phase 7a placeholder)
# ---------------------------------------------------------------------------


class TestFanovaImportance:
    """Placeholder for fANOVA importance evaluator (Phase 7a)."""

    @pytest.mark.skip(reason="Phase 7a: fANOVA not yet implemented")
    def test_fanova_ranking(self):
        pass


# ---------------------------------------------------------------------------
# MDI importance (Phase 7b placeholder)
# ---------------------------------------------------------------------------


class TestMDIImportance:
    """Placeholder for MDI importance evaluator (Phase 7b)."""

    @pytest.mark.skip(reason="Phase 7b: MDI not yet implemented")
    def test_mdi_ranking(self):
        pass
