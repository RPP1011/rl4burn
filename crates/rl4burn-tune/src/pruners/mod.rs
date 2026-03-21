mod hyperband;
mod median;
mod nop;
mod patient;
mod percentile;
mod successive_halving;
mod threshold;
mod wilcoxon;

pub use hyperband::HyperbandPruner;
pub use median::MedianPruner;
pub use nop::NopPruner;
pub use patient::PatientPruner;
pub use percentile::PercentilePruner;
pub use successive_halving::SuccessiveHalvingPruner;
pub use threshold::ThresholdPruner;
pub use wilcoxon::WilcoxonPruner;

use crate::study::Study;
use crate::trial::FrozenTrial;

/// Trait for trial pruners.
pub trait Pruner: Send + Sync {
    /// Decide whether the trial should be pruned at its current step.
    fn prune(&self, study: &Study, trial: &FrozenTrial) -> bool;
}
