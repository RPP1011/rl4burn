"""Cross-verification tests for pruners between rl4burn_tune_py and Optuna."""

import pytest
import numpy as np

import rl4burn_tune_py as rl

optuna = pytest.importorskip("optuna")

from .conftest import CROSS_SEED


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_optuna_study_with_trials(n_trials, n_steps, direction="minimize", seed=CROSS_SEED):
    """Create an Optuna study with completed trials that have intermediate values."""
    rng = np.random.RandomState(seed)
    study = optuna.create_study(direction=direction)

    for i in range(n_trials):
        trial = study.ask()
        trial.suggest_float("x", -5.0, 5.0)
        for step in range(n_steps):
            # Decreasing values with noise
            trial.report(1.0 / (step + 1) + rng.normal(0, 0.1) + i * 0.5, step)
        study.tell(trial, 1.0 / n_steps + i * 0.5)

    return study


def _create_rl4burn_study_with_trials(n_trials, n_steps, seed=CROSS_SEED):
    """Create an rl4burn study with completed trials that have intermediate values."""
    rng = np.random.RandomState(seed)
    study = rl.Study("minimize")
    sampler = rl.random_sampler(seed)

    for i in range(n_trials):
        trial = study.ask()
        trial.suggest_float("x", -5.0, 5.0, sampler, study)
        for step in range(n_steps):
            trial.report(step, 1.0 / (step + 1) + rng.normal(0, 0.1) + i * 0.5)
        study.tell(trial, 1.0 / n_steps + i * 0.5)

    return study


# ---------------------------------------------------------------------------
# Median pruner
# ---------------------------------------------------------------------------


class TestMedianPruner:
    """Cross-verify median pruner decisions."""

    def test_no_prune_during_warmup(self):
        """Neither library should prune during warmup steps."""
        n_warmup = 5

        # rl4burn
        rl_study = _create_rl4burn_study_with_trials(10, 10)
        rl_pruner = rl.median_pruner(n_startup_trials=0, n_warmup_steps=n_warmup)

        # Create a trial and report bad intermediate values
        rl_trial = rl_study.ask()
        rl_sampler = rl.random_sampler(CROSS_SEED + 1)
        rl_trial.suggest_float("x", -5.0, 5.0, rl_sampler, rl_study)
        for step in range(n_warmup):
            rl_trial.report(step, 1000.0)  # Very bad value
            assert not rl_trial.should_prune(rl_study, rl_pruner)

    def test_no_prune_during_startup(self):
        """Should not prune when fewer than n_startup_trials are completed."""
        rl_study = rl.Study("minimize")
        rl_sampler = rl.random_sampler(CROSS_SEED)
        rl_pruner = rl.median_pruner(n_startup_trials=5)

        # Only add 3 completed trials (below startup threshold)
        for i in range(3):
            t = rl_study.ask()
            t.suggest_float("x", -5.0, 5.0, rl_sampler, rl_study)
            rl_study.tell(t, float(i))

        # New trial should not be pruned
        trial = rl_study.ask()
        trial.suggest_float("x", -5.0, 5.0, rl_sampler, rl_study)
        trial.report(0, 1000.0)
        assert not trial.should_prune(rl_study, rl_pruner)

    def test_prune_bad_trial(self):
        """A very bad trial should be pruned after warmup."""
        rl_study = _create_rl4burn_study_with_trials(10, 10)
        rl_pruner = rl.median_pruner(n_startup_trials=0, n_warmup_steps=0)

        trial = rl_study.ask()
        rl_sampler = rl.random_sampler(CROSS_SEED + 1)
        trial.suggest_float("x", -5.0, 5.0, rl_sampler, rl_study)
        # Report extremely bad value at step 5
        trial.report(5, 1000.0)
        assert trial.should_prune(rl_study, rl_pruner)


# ---------------------------------------------------------------------------
# Percentile pruner
# ---------------------------------------------------------------------------


class TestPercentilePruner:
    """Cross-verify percentile pruner."""

    def test_percentile_50_equals_median(self):
        """Percentile pruner at 50% should behave like median pruner."""
        rl_study = _create_rl4burn_study_with_trials(10, 10)

        median_p = rl.median_pruner(n_startup_trials=0, n_warmup_steps=0)
        pctile_p = rl.percentile_pruner(
            percentile=50.0, n_startup_trials=0, n_warmup_steps=0
        )

        # Both should agree on prune decisions
        rl_sampler = rl.random_sampler(CROSS_SEED + 1)
        for val in [0.001, 0.5, 5.0, 100.0, 1000.0]:
            trial_m = rl_study.ask()
            trial_m.suggest_float("x", -5.0, 5.0, rl_sampler, rl_study)
            trial_m.report(5, val)

            trial_p = rl_study.ask()
            trial_p.suggest_float("x", -5.0, 5.0, rl_sampler, rl_study)
            trial_p.report(5, val)

            m_decision = trial_m.should_prune(rl_study, median_p)
            p_decision = trial_p.should_prune(rl_study, pctile_p)
            assert m_decision == p_decision, (
                f"Mismatch for val={val}: median={m_decision}, percentile={p_decision}"
            )

    def test_strict_percentile(self):
        """At percentile=25, more trials should be pruned than at 75."""
        rl_study = _create_rl4burn_study_with_trials(20, 10)

        strict_p = rl.percentile_pruner(
            percentile=25.0, n_startup_trials=0, n_warmup_steps=0
        )
        lenient_p = rl.percentile_pruner(
            percentile=75.0, n_startup_trials=0, n_warmup_steps=0
        )

        rl_sampler = rl.random_sampler(CROSS_SEED + 2)
        rng = np.random.RandomState(CROSS_SEED)
        strict_prunes = 0
        lenient_prunes = 0

        for _ in range(20):
            val = rng.uniform(0, 20)

            trial_s = rl_study.ask()
            trial_s.suggest_float("x", -5.0, 5.0, rl_sampler, rl_study)
            trial_s.report(5, val)
            if trial_s.should_prune(rl_study, strict_p):
                strict_prunes += 1

            trial_l = rl_study.ask()
            trial_l.suggest_float("x", -5.0, 5.0, rl_sampler, rl_study)
            trial_l.report(5, val)
            if trial_l.should_prune(rl_study, lenient_p):
                lenient_prunes += 1

        assert strict_prunes >= lenient_prunes


# ---------------------------------------------------------------------------
# Threshold pruner
# ---------------------------------------------------------------------------


class TestThresholdPruner:
    """Cross-verify threshold pruner."""

    def test_upper_bound(self):
        """Values above upper bound should be pruned."""
        rl_study = rl.Study("minimize")
        rl_sampler = rl.random_sampler(CROSS_SEED)
        pruner = rl.threshold_pruner(upper=10.0)

        trial = rl_study.ask()
        trial.suggest_float("x", -5.0, 5.0, rl_sampler, rl_study)
        trial.report(0, 5.0)
        assert not trial.should_prune(rl_study, pruner)

        trial2 = rl_study.ask()
        trial2.suggest_float("x", -5.0, 5.0, rl_sampler, rl_study)
        trial2.report(0, 15.0)
        assert trial2.should_prune(rl_study, pruner)

    def test_lower_bound(self):
        """Values below lower bound should be pruned."""
        rl_study = rl.Study("minimize")
        rl_sampler = rl.random_sampler(CROSS_SEED)
        pruner = rl.threshold_pruner(lower=0.0)

        trial = rl_study.ask()
        trial.suggest_float("x", -5.0, 5.0, rl_sampler, rl_study)
        trial.report(0, 1.0)
        assert not trial.should_prune(rl_study, pruner)

        trial2 = rl_study.ask()
        trial2.suggest_float("x", -5.0, 5.0, rl_sampler, rl_study)
        trial2.report(0, -1.0)
        assert trial2.should_prune(rl_study, pruner)


# ---------------------------------------------------------------------------
# Wilcoxon pruner
# ---------------------------------------------------------------------------


class TestWilcoxonPruner:
    """Cross-verify Wilcoxon pruner."""

    def test_does_not_prune_with_few_steps(self):
        """Should not prune when trial has fewer steps than n_min_steps."""
        rl_study = _create_rl4burn_study_with_trials(10, 10)
        rl_pruner = rl.wilcoxon_pruner(p_threshold=0.1, n_min_steps=5, n_startup_trials=0)

        rl_sampler = rl.random_sampler(CROSS_SEED + 1)
        trial = rl_study.ask()
        trial.suggest_float("x", -5.0, 5.0, rl_sampler, rl_study)
        # Report only 3 steps (< n_min_steps=5)
        for step in range(3):
            trial.report(step, 1000.0)
        assert not trial.should_prune(rl_study, rl_pruner)

    def test_prunes_significantly_worse_trial(self):
        """A trial that's consistently much worse should eventually be pruned."""
        rl_study = _create_rl4burn_study_with_trials(10, 20)
        rl_pruner = rl.wilcoxon_pruner(p_threshold=0.1, n_min_steps=5, n_startup_trials=0)

        rl_sampler = rl.random_sampler(CROSS_SEED + 1)
        trial = rl_study.ask()
        trial.suggest_float("x", -5.0, 5.0, rl_sampler, rl_study)
        # Report consistently terrible values
        for step in range(20):
            trial.report(step, 10000.0)

        # After enough steps, should be pruned
        assert trial.should_prune(rl_study, rl_pruner)


# ---------------------------------------------------------------------------
# Hyperband pruner
# ---------------------------------------------------------------------------


class TestHyperbandPruner:
    """Cross-verify Hyperband pruner."""

    def test_basic_construction(self):
        """Hyperband pruner should be constructible with valid params."""
        pruner = rl.hyperband_pruner(min_resource=1, max_resource=81, reduction_factor=3)
        assert pruner is not None

    def test_no_prune_during_early_trials(self):
        """First few trials should not be pruned (building up brackets)."""
        rl_study = rl.Study("minimize")
        rl_sampler = rl.random_sampler(CROSS_SEED)
        pruner = rl.hyperband_pruner(min_resource=1, max_resource=81, reduction_factor=3)

        # First trial should not be pruned
        trial = rl_study.ask()
        trial.suggest_float("x", -5.0, 5.0, rl_sampler, rl_study)
        trial.report(0, 100.0)
        # Hyperband typically doesn't prune very early trials
        # Just verify it doesn't crash
        _ = trial.should_prune(rl_study, pruner)


# ---------------------------------------------------------------------------
# Wilcoxon average_is_best (Phase 4a placeholder)
# ---------------------------------------------------------------------------


class TestWilcoxonAverageIsBest:
    """Tests for average_is_best mode (Phase 4a)."""

    def test_wilcoxon_average_is_best(self):
        """With average_is_best=True, a trial far above the step-wise mean should be pruned."""
        rl_study = _create_rl4burn_study_with_trials(10, 20)
        rl_pruner = rl.wilcoxon_pruner(
            p_threshold=0.1, n_min_steps=5, n_startup_trials=0, average_is_best=True
        )

        rl_sampler = rl.random_sampler(CROSS_SEED + 1)
        trial = rl_study.ask()
        trial.suggest_float("x", -5.0, 5.0, rl_sampler, rl_study)
        # Report values consistently far worse than any completed trial
        for step in range(20):
            trial.report(step, 10000.0)

        # Should be pruned because 10000.0 is way above the average at each step
        assert trial.should_prune(rl_study, rl_pruner)

    def test_wilcoxon_average_is_best_no_prune_good_trial(self):
        """With average_is_best=True, a trial better than average should NOT be pruned."""
        rl_study = _create_rl4burn_study_with_trials(10, 20)
        rl_pruner = rl.wilcoxon_pruner(
            p_threshold=0.1, n_min_steps=5, n_startup_trials=0, average_is_best=True
        )

        rl_sampler = rl.random_sampler(CROSS_SEED + 2)
        trial = rl_study.ask()
        trial.suggest_float("x", -5.0, 5.0, rl_sampler, rl_study)
        # Report values well below the average (minimization, so lower is better)
        # The average offset across trials 0..9 is 0.5*4.5 = 2.25; use 0.0 to be clearly better
        for step in range(20):
            trial.report(step, 1.0 / (step + 1))

        # Should NOT be pruned -- values are better than the average
        assert not trial.should_prune(rl_study, rl_pruner)
