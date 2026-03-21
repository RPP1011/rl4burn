mod median;
mod percentile;
mod hyperband;

pub use median::MedianPruner;
pub use percentile::PercentilePruner;
pub use hyperband::HyperbandPruner;

use crate::study::Study;
use crate::trial::FrozenTrial;

/// Trait for trial pruners.
pub trait Pruner: Send + Sync {
    /// Decide whether the trial should be pruned at its current step.
    fn prune(&self, study: &Study, trial: &FrozenTrial) -> bool;
}
