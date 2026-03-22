# Re-export native module for convenience
import time as _time

import rl4burn_tune_py as _native

# Re-export everything from native module
from rl4burn_tune_py import (
    Study as _NativeStudy,
    Trial,
    Sampler,
    Pruner,
    Storage,
    random_sampler,
    tpe_sampler,
    cmaes_sampler,
    nsga2_sampler,
    gp_sampler,
    qmc_sampler,
    brute_force_sampler,
    partial_fixed_sampler,
    nop_pruner,
    median_pruner,
    percentile_pruner,
    threshold_pruner,
    patient_pruner,
    successive_halving_pruner,
    hyperband_pruner,
    wilcoxon_pruner,
    in_memory_storage,
    journal_storage,
)


class TrialPruned(Exception):
    """Raise this inside an objective function to indicate that the trial should be pruned."""
    pass


class Study(_NativeStudy):
    """Extended Study with a high-level ``optimize`` method."""

    def optimize(self, objective_fn, n_trials, sampler, pruner=None, timeout=None):
        """Run the ask/tell optimization loop.

        Parameters
        ----------
        objective_fn : callable
            A function that takes a :class:`Trial` and returns a float value.
            Raise :class:`TrialPruned` to prune the trial.
        n_trials : int
            Number of trials to run.
        sampler : Sampler
            The sampler to pass through to the objective (unused directly
            here but kept for API symmetry; the objective is expected to
            use it when calling ``trial.suggest_*``).
        pruner : Pruner, optional
            A pruner instance (reserved for use inside the objective).
        timeout : float, optional
            Time limit in seconds.  The loop stops once the elapsed wall-clock
            time meets or exceeds this value.
        """
        start = _time.monotonic()
        for _ in range(n_trials):
            if self.is_stopped():
                break
            if timeout is not None and (_time.monotonic() - start) >= timeout:
                break

            trial = self.ask()
            try:
                value = objective_fn(trial)
                self.tell(trial, value, "complete")
            except TrialPruned:
                self.tell(trial, None, "pruned")
            except Exception as exc:
                if "TrialPruned" in str(exc):
                    self.tell(trial, None, "pruned")
                else:
                    self.tell(trial, None, "fail")
                    raise


def create_study(direction="minimize", name=None):
    """Factory function that returns a :class:`Study` (with ``optimize``)."""
    return Study(direction, name)
