//! Threshold pruner — prunes if intermediate value exceeds a hard bound.

use crate::study::Study;
use crate::trial::FrozenTrial;

use super::Pruner;

/// Prune trials whose intermediate values exceed a hard threshold.
pub struct ThresholdPruner {
    /// If set, prune when value exceeds this upper bound.
    pub upper: Option<f64>,
    /// If set, prune when value falls below this lower bound.
    pub lower: Option<f64>,
    /// Number of warmup steps before pruning begins.
    pub n_warmup_steps: usize,
}

impl ThresholdPruner {
    /// Create a threshold pruner with the given bounds.
    pub fn new(upper: Option<f64>, lower: Option<f64>, n_warmup_steps: usize) -> Self {
        Self {
            upper,
            lower,
            n_warmup_steps,
        }
    }
}

impl Pruner for ThresholdPruner {
    fn prune(&self, _study: &Study, trial: &FrozenTrial) -> bool {
        let step = trial.last_step();
        if step < self.n_warmup_steps {
            return false;
        }

        if let Some(&value) = trial.intermediate_values.get(&step) {
            if let Some(upper) = self.upper {
                if value > upper {
                    return true;
                }
            }
            if let Some(lower) = self.lower {
                if value < lower {
                    return true;
                }
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study::{Direction, Study};
    use crate::trial::{FrozenTrial, TrialState};

    #[test]
    fn test_threshold_upper() {
        let pruner = ThresholdPruner::new(Some(10.0), None, 0);
        let study = Study::new(Direction::Minimize);

        let mut trial = FrozenTrial::new(0);
        trial.state = TrialState::Running;
        trial.report(0, 5.0);
        assert!(!pruner.prune(&study, &trial));

        trial.report(1, 15.0);
        assert!(pruner.prune(&study, &trial));
    }

    #[test]
    fn test_threshold_lower() {
        let pruner = ThresholdPruner::new(None, Some(0.0), 0);
        let study = Study::new(Direction::Minimize);

        let mut trial = FrozenTrial::new(0);
        trial.state = TrialState::Running;
        trial.report(0, 1.0);
        assert!(!pruner.prune(&study, &trial));

        trial.report(1, -1.0);
        assert!(pruner.prune(&study, &trial));
    }

    #[test]
    fn test_threshold_warmup() {
        let pruner = ThresholdPruner::new(Some(10.0), None, 5);
        let study = Study::new(Direction::Minimize);

        let mut trial = FrozenTrial::new(0);
        trial.state = TrialState::Running;
        trial.report(3, 100.0);
        assert!(!pruner.prune(&study, &trial)); // Still in warmup
    }
}
