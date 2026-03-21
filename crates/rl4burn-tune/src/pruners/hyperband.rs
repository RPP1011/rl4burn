use crate::study::Study;
use crate::trial::FrozenTrial;

use super::percentile::PercentilePruner;
use super::Pruner;

/// Hyperband pruner implementing successive halving with multiple brackets.
///
/// Uses the Hyperband algorithm which runs multiple brackets of successive
/// halving with different resource budgets.
pub struct HyperbandPruner {
    /// Minimum resource (e.g., number of epochs).
    pub min_resource: usize,
    /// Maximum resource.
    pub max_resource: usize,
    /// Reduction factor (typically 3).
    pub reduction_factor: usize,
    /// Internal percentile pruners, one per bracket rung.
    pruners: Vec<PercentilePruner>,
}

impl HyperbandPruner {
    pub fn new(min_resource: usize, max_resource: usize, reduction_factor: usize) -> Self {
        assert!(min_resource > 0, "min_resource must be > 0");
        assert!(
            max_resource >= min_resource,
            "max_resource must be >= min_resource"
        );
        assert!(reduction_factor >= 2, "reduction_factor must be >= 2");

        // Compute the number of brackets and set up rung-level pruners.
        // Each rung uses a percentile = 1/reduction_factor * 100 threshold.
        let percentile = 100.0 / reduction_factor as f64;
        let n_brackets =
            ((max_resource as f64 / min_resource as f64).log(reduction_factor as f64)).floor()
                as usize
                + 1;

        let mut pruners = Vec::with_capacity(n_brackets);
        for _bracket in 0..n_brackets {
            pruners.push(PercentilePruner::new(percentile, 1, 0, 1));
        }

        Self {
            min_resource,
            max_resource,
            reduction_factor,
            pruners,
        }
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

        // Determine which rung the trial is at.
        // Rung k corresponds to resource = min_resource * reduction_factor^k.
        // Only prune at rung boundaries.
        let mut rung = 0;
        let mut rung_resource = self.min_resource;
        let mut should_prune = false;

        while rung_resource <= self.max_resource && rung < self.pruners.len() {
            if step == rung_resource {
                should_prune = self.pruners[rung].prune(study, trial);
                break;
            }
            rung += 1;
            rung_resource *= self.reduction_factor;
        }

        should_prune
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperband_creation() {
        let hb = HyperbandPruner::new(1, 81, 3);
        // log_3(81/1) + 1 = 5 brackets
        assert_eq!(hb.pruners.len(), 5);
    }

    #[test]
    fn test_hyperband_default() {
        let hb = HyperbandPruner::default();
        assert_eq!(hb.min_resource, 1);
        assert_eq!(hb.max_resource, 100);
        assert_eq!(hb.reduction_factor, 3);
    }
}
