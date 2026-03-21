use crate::study::Study;
use crate::trial::{FrozenTrial, TrialState};

use super::Pruner;

/// Prunes a trial if its intermediate value is below the given percentile of
/// completed trials at the same step.
pub struct PercentilePruner {
    /// Percentile threshold (0.0 to 100.0). E.g., 50.0 is equivalent to MedianPruner.
    pub percentile: f64,
    /// Number of completed trials before pruning kicks in.
    pub n_startup_trials: usize,
    /// Number of steps before pruning kicks in for a given trial.
    pub n_warmup_steps: usize,
    /// Only consider pruning every `interval_steps` steps.
    pub interval_steps: usize,
}

impl PercentilePruner {
    pub fn new(
        percentile: f64,
        n_startup_trials: usize,
        n_warmup_steps: usize,
        interval_steps: usize,
    ) -> Self {
        assert!(
            (0.0..=100.0).contains(&percentile),
            "percentile must be in [0, 100], got {percentile}"
        );
        assert!(interval_steps > 0, "interval_steps must be > 0");
        Self {
            percentile,
            n_startup_trials,
            n_warmup_steps,
            interval_steps,
        }
    }
}

impl Pruner for PercentilePruner {
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

        let mut values_at_step: Vec<f64> = completed
            .iter()
            .filter_map(|t| t.intermediate_values.get(&step).copied())
            .collect();

        if values_at_step.is_empty() {
            return false;
        }

        values_at_step.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = percentile_of_sorted(&values_at_step, self.percentile);

        match study.direction() {
            crate::study::Direction::Minimize => current_value > threshold,
            crate::study::Direction::Maximize => current_value < threshold,
        }
    }
}

/// Compute the given percentile (0-100) of a sorted slice using linear interpolation.
fn percentile_of_sorted(sorted: &[f64], percentile: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }

    let rank = percentile / 100.0 * (n - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;

    if lower == upper {
        sorted[lower]
    } else {
        let frac = rank - lower as f64;
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile_of_sorted() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile_of_sorted(&v, 0.0) - 1.0).abs() < 1e-12);
        assert!((percentile_of_sorted(&v, 50.0) - 3.0).abs() < 1e-12);
        assert!((percentile_of_sorted(&v, 100.0) - 5.0).abs() < 1e-12);
        assert!((percentile_of_sorted(&v, 25.0) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_percentile_pruner_at_50_matches_median() {
        let pruner = PercentilePruner::new(50.0, 0, 0, 1);
        let mut study = Study::new(crate::study::Direction::Minimize);

        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            let mut t = FrozenTrial::new(study.trials().len());
            t.report(0, v);
            t.state = TrialState::Complete;
            t.value = Some(v);
            study.add_trial(t);
        }

        // Value 4.0 > median 3.0 → prune
        let mut trial = FrozenTrial::new(10);
        trial.report(0, 4.0);
        assert!(pruner.prune(&study, &trial));

        // Value 2.0 < median → don't prune
        let mut trial = FrozenTrial::new(11);
        trial.report(0, 2.0);
        assert!(!pruner.prune(&study, &trial));
    }
}
