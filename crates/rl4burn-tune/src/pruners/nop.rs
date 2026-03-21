//! No-operation pruner that never prunes.

use crate::study::Study;
use crate::trial::FrozenTrial;

use super::Pruner;

/// A pruner that never prunes. Useful as a default or placeholder.
pub struct NopPruner;

impl Pruner for NopPruner {
    fn prune(&self, _study: &Study, _trial: &FrozenTrial) -> bool {
        false
    }
}
