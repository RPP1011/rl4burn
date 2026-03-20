use crate::study::Study;
use crate::trial::{FrozenTrial, TrialState};

use super::Pruner;

/// Prunes a trial if its intermediate value is below the median of
/// completed trials at the same step.
pub struct MedianPruner {
    /// Number of completed trials before pruning kicks in.
    pub n_startup_trials: usize,
    /// Number of steps before pruning kicks in for a given trial.
    pub n_warmup_steps: usize,
    /// Only consider pruning every `interval_steps` steps.
    pub interval_steps: usize,
}

impl MedianPruner {
    pub fn new(n_startup_trials: usize, n_warmup_steps: usize, interval_steps: usize) -> Self {
        assert!(interval_steps > 0, "interval_steps must be > 0");
        Self {
            n_startup_trials,
            n_warmup_steps,
            interval_steps,
        }
    }
}

impl Default for MedianPruner {
    fn default() -> Self {
        Self {
            n_startup_trials: 5,
            n_warmup_steps: 0,
            interval_steps: 1,
        }
    }
}

impl Pruner for MedianPruner {
    fn prune(&self, study: &Study, trial: &FrozenTrial) -> bool {
        let step = trial.last_step();
        if step < self.n_warmup_steps {
            return false;
        }

        if step % self.interval_steps != 0 {
            return false;
        }

        let completed: Vec<_> = study
            .trials()
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .collect();

        if completed.len() < self.n_startup_trials {
            return false;
        }

        let current_value = match trial.intermediate_values.get(&step) {
            Some(&v) => v,
            None => return false,
        };

        // Gather values at this step from completed trials
        let mut values_at_step: Vec<f64> = completed
            .iter()
            .filter_map(|t| t.intermediate_values.get(&step).copied())
            .collect();

        if values_at_step.is_empty() {
            return false;
        }

        values_at_step.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = median_of_sorted(&values_at_step);

        match study.direction() {
            crate::study::Direction::Minimize => current_value > median,
            crate::study::Direction::Maximize => current_value < median,
        }
    }
}

/// Compute the median of a sorted slice.
pub(crate) fn median_of_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median_of_sorted() {
        assert_eq!(median_of_sorted(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(median_of_sorted(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(median_of_sorted(&[5.0]), 5.0);
    }

    #[test]
    fn test_no_prune_during_warmup() {
        let pruner = MedianPruner::new(0, 10, 1);
        let study = Study::new_default();
        let mut trial = FrozenTrial::new(0);
        trial.report(5, 0.5);
        assert!(!pruner.prune(&study, &trial));
    }

    #[test]
    fn test_no_prune_during_startup() {
        let pruner = MedianPruner::new(5, 0, 1);
        let study = Study::new_default();
        // No completed trials yet
        let mut trial = FrozenTrial::new(0);
        trial.report(0, 0.5);
        assert!(!pruner.prune(&study, &trial));
    }

    #[test]
    fn test_prune_below_median() {
        let pruner = MedianPruner::new(0, 0, 1);
        let mut study = Study::new(crate::study::Direction::Minimize);

        // Add completed trials with intermediate values at step 0
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            let mut t = FrozenTrial::new(study.trials().len());
            t.report(0, v);
            t.state = TrialState::Complete;
            t.value = Some(v);
            study.add_trial(t);
        }

        // Current trial has value 4.0 at step 0 — above median (3.0), should prune
        let mut trial = FrozenTrial::new(10);
        trial.report(0, 4.0);
        assert!(pruner.prune(&study, &trial));

        // Current trial has value 1.0 — below median, should not prune
        let mut trial = FrozenTrial::new(11);
        trial.report(0, 1.0);
        assert!(!pruner.prune(&study, &trial));
    }
}
