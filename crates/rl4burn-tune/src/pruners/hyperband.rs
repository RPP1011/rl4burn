use crate::study::Study;
use crate::trial::FrozenTrial;

use super::percentile::PercentilePruner;
use super::Pruner;

/// Hyperband pruner implementing successive halving with multiple brackets.
///
/// Uses the Hyperband algorithm which runs multiple brackets of successive
/// halving with different resource budgets. Each bracket has a different
/// minimum resource and number of rungs.
///
/// Trials are assigned to brackets based on `trial.number % n_brackets`.
/// Each bracket i (0-indexed from the most aggressive) has:
/// - `min_resource_i = min_resource * reduction_factor^i`
/// - Rungs at `min_resource_i * reduction_factor^k` for k = 0, 1, ...
pub struct HyperbandPruner {
    /// Minimum resource (e.g., number of epochs).
    pub min_resource: usize,
    /// Maximum resource.
    pub max_resource: usize,
    /// Reduction factor (typically 3).
    pub reduction_factor: usize,
    /// Number of brackets.
    n_brackets: usize,
    /// Percentile threshold for pruning at each rung (1/reduction_factor * 100).
    percentile: f64,
    /// Internal percentile pruners, one per bracket.
    /// Each bracket has its own pruner with appropriate warmup.
    bracket_pruners: Vec<Vec<PercentilePruner>>,
}

impl HyperbandPruner {
    pub fn new(min_resource: usize, max_resource: usize, reduction_factor: usize) -> Self {
        assert!(min_resource > 0, "min_resource must be > 0");
        assert!(
            max_resource >= min_resource,
            "max_resource must be >= min_resource"
        );
        assert!(reduction_factor >= 2, "reduction_factor must be >= 2");

        let percentile = 100.0 / reduction_factor as f64;
        let n_brackets =
            ((max_resource as f64 / min_resource as f64).log(reduction_factor as f64)).floor()
                as usize
                + 1;

        // Build per-bracket pruner configs.
        // Bracket i has min_resource_i = min_resource * reduction_factor^i
        // and rungs at min_resource_i, min_resource_i * rf, min_resource_i * rf^2, ...
        let mut bracket_pruners = Vec::with_capacity(n_brackets);
        for bracket in 0..n_brackets {
            let bracket_min = min_resource * reduction_factor.pow(bracket as u32);
            let mut rung_pruners = Vec::new();
            let mut rung_resource = bracket_min;
            while rung_resource <= max_resource {
                // Each rung uses a percentile pruner with n_startup_trials=1
                // and warmup at the rung resource level
                rung_pruners.push(PercentilePruner::new(percentile, 1, 0, 1));
                rung_resource *= reduction_factor;
            }
            bracket_pruners.push(rung_pruners);
        }

        Self {
            min_resource,
            max_resource,
            reduction_factor,
            n_brackets,
            percentile,
            bracket_pruners,
        }
    }

    /// Get the number of brackets.
    pub fn n_brackets(&self) -> usize {
        self.n_brackets
    }

    /// Get the bracket assignment for a trial.
    pub fn bracket_for_trial(&self, trial_number: usize) -> usize {
        trial_number % self.n_brackets
    }

    /// Get the rung resources for a given bracket.
    pub fn rung_resources(&self, bracket: usize) -> Vec<usize> {
        let bracket_min = self.min_resource * self.reduction_factor.pow(bracket as u32);
        let mut rungs = Vec::new();
        let mut r = bracket_min;
        while r <= self.max_resource {
            rungs.push(r);
            r *= self.reduction_factor;
        }
        rungs
    }
}

impl Default for HyperbandPruner {
    fn default() -> Self {
        Self::new(1, 100, 3)
    }
}

impl Pruner for HyperbandPruner {
    fn prune(&self, study: &Study, trial: &FrozenTrial) -> bool {
        let step = trial.last_step();
        let bracket = self.bracket_for_trial(trial.number);

        // Get the rung resources for this bracket
        let rungs = self.rung_resources(bracket);

        // Check if the current step matches any rung in this bracket
        for (rung_idx, &rung_resource) in rungs.iter().enumerate() {
            if step == rung_resource {
                if let Some(pruner) = self.bracket_pruners.get(bracket).and_then(|bp| bp.get(rung_idx)) {
                    return pruner.prune(study, trial);
                }
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperband_creation() {
        let hb = HyperbandPruner::new(1, 81, 3);
        // log_3(81/1) + 1 = 5 brackets
        assert_eq!(hb.n_brackets(), 5);
    }

    #[test]
    fn test_hyperband_default() {
        let hb = HyperbandPruner::default();
        assert_eq!(hb.min_resource, 1);
        assert_eq!(hb.max_resource, 100);
        assert_eq!(hb.reduction_factor, 3);
    }

    #[test]
    fn test_bracket_assignment() {
        let hb = HyperbandPruner::new(1, 81, 3);
        assert_eq!(hb.bracket_for_trial(0), 0);
        assert_eq!(hb.bracket_for_trial(1), 1);
        assert_eq!(hb.bracket_for_trial(4), 4);
        assert_eq!(hb.bracket_for_trial(5), 0); // wraps around
        assert_eq!(hb.bracket_for_trial(7), 2);
    }

    #[test]
    fn test_rung_resources() {
        let hb = HyperbandPruner::new(1, 81, 3);
        // Bracket 0: min=1, rungs at 1, 3, 9, 27, 81
        assert_eq!(hb.rung_resources(0), vec![1, 3, 9, 27, 81]);
        // Bracket 1: min=3, rungs at 3, 9, 27, 81
        assert_eq!(hb.rung_resources(1), vec![3, 9, 27, 81]);
        // Bracket 2: min=9, rungs at 9, 27, 81
        assert_eq!(hb.rung_resources(2), vec![9, 27, 81]);
        // Bracket 3: min=27, rungs at 27, 81
        assert_eq!(hb.rung_resources(3), vec![27, 81]);
        // Bracket 4: min=81, rungs at 81
        assert_eq!(hb.rung_resources(4), vec![81]);
    }
}
