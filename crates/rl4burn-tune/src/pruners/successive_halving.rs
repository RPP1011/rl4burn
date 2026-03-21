//! Successive Halving pruner.
//!
//! Implements the Successive Halving Algorithm (SHA) for early stopping.
//! At each rung (resource level), keeps only the top 1/reduction_factor fraction.

use crate::study::Study;
use crate::trial::{FrozenTrial, TrialState};

use super::Pruner;

/// Successive Halving pruner.
///
/// Trials report intermediate values at integer steps. At each "rung"
/// (steps that are powers of `reduction_factor` times `min_resource`),
/// trials whose value is not in the top 1/`reduction_factor` are pruned.
pub struct SuccessiveHalvingPruner {
    /// Minimum resource (first rung step).
    pub min_resource: usize,
    /// Fraction of trials to keep at each rung (1/reduction_factor).
    pub reduction_factor: usize,
    /// Number of warmup steps before pruning begins.
    pub n_warmup_steps: usize,
}

impl SuccessiveHalvingPruner {
    /// Create a new Successive Halving pruner.
    pub fn new(min_resource: usize, reduction_factor: usize, n_warmup_steps: usize) -> Self {
        assert!(min_resource > 0);
        assert!(reduction_factor >= 2);
        Self {
            min_resource,
            reduction_factor,
            n_warmup_steps,
        }
    }
}

impl Pruner for SuccessiveHalvingPruner {
    fn prune(&self, study: &Study, trial: &FrozenTrial) -> bool {
        let step = trial.last_step();
        if step < self.n_warmup_steps {
            return false;
        }

        // Find the current rung
        let mut rung = self.min_resource;
        while rung * self.reduction_factor <= step {
            rung *= self.reduction_factor;
        }

        // If we haven't reached min_resource, don't prune
        if step < self.min_resource {
            return false;
        }

        // Get the trial's value at the current rung
        let trial_value = match trial.intermediate_values.get(&rung) {
            Some(&v) => v,
            None => return false,
        };

        // Collect values from all other trials at this rung
        let mut rung_values: Vec<f64> = study
            .trials()
            .iter()
            .filter(|t| {
                (t.state == TrialState::Complete || t.state == TrialState::Running)
                    && t.number != trial.number
            })
            .filter_map(|t| t.intermediate_values.get(&rung).copied())
            .collect();

        if rung_values.is_empty() {
            return false;
        }

        rung_values.push(trial_value);
        rung_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Keep top 1/reduction_factor (for minimize: keep lowest values)
        let n_keep = (rung_values.len() + self.reduction_factor - 1) / self.reduction_factor;
        let threshold = rung_values[n_keep.min(rung_values.len()) - 1];

        // Prune if trial's value is worse than the threshold
        match study.direction() {
            crate::study::Direction::Minimize => trial_value > threshold,
            crate::study::Direction::Maximize => trial_value < {
                // For maximize, take top values
                let rev_idx = rung_values.len().saturating_sub(n_keep);
                rung_values[rev_idx]
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study::{Direction, Study};
    use crate::trial::{FrozenTrial, TrialState};

    #[test]
    fn test_sha_basic() {
        let pruner = SuccessiveHalvingPruner::new(1, 3, 0);
        let mut study = Study::new(Direction::Minimize);

        // Add 6 completed trials with values at step 1
        for i in 0..6 {
            let mut t = FrozenTrial::new(i);
            t.state = TrialState::Complete;
            t.value = Some(i as f64);
            t.intermediate_values.insert(1, i as f64);
            study.add_trial(t);
        }

        // Trial with value 1.0 at step 1 should survive (top 1/3 = top 2)
        let mut trial = FrozenTrial::new(6);
        trial.state = TrialState::Running;
        trial.intermediate_values.insert(1, 1.0);
        assert!(!pruner.prune(&study, &trial));

        // Trial with value 5.0 at step 1 should be pruned
        let mut trial = FrozenTrial::new(7);
        trial.state = TrialState::Running;
        trial.intermediate_values.insert(1, 5.0);
        assert!(pruner.prune(&study, &trial));
    }
}
