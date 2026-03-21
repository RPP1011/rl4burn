//! Patient pruner — wraps another pruner with a patience counter.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::study::Study;
use crate::trial::FrozenTrial;

use super::Pruner;

/// Wraps another pruner and requires N consecutive prune signals before
/// actually pruning a trial.
pub struct PatientPruner {
    /// The wrapped pruner.
    inner: Box<dyn Pruner>,
    /// Number of consecutive prune signals required.
    patience: usize,
    /// Track consecutive prune counts per trial.
    counts: Mutex<HashMap<usize, usize>>,
}

impl PatientPruner {
    /// Create a patient pruner wrapping the given pruner with the specified patience.
    pub fn new(inner: Box<dyn Pruner>, patience: usize) -> Self {
        Self {
            inner,
            patience,
            counts: Mutex::new(HashMap::new()),
        }
    }
}

impl Pruner for PatientPruner {
    fn prune(&self, study: &Study, trial: &FrozenTrial) -> bool {
        if self.inner.prune(study, trial) {
            let mut counts = self.counts.lock().unwrap();
            let count = counts.entry(trial.number).or_insert(0);
            *count += 1;
            *count >= self.patience
        } else {
            // Reset counter on non-prune
            let mut counts = self.counts.lock().unwrap();
            counts.insert(trial.number, 0);
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study::{Direction, Study};
    use crate::trial::FrozenTrial;

    /// A pruner that always prunes (for testing).
    struct AlwaysPruner;
    impl Pruner for AlwaysPruner {
        fn prune(&self, _study: &Study, _trial: &FrozenTrial) -> bool {
            true
        }
    }

    #[test]
    fn test_patient_requires_consecutive() {
        let patient = PatientPruner::new(Box::new(AlwaysPruner), 3);
        let study = Study::new(Direction::Minimize);
        let trial = FrozenTrial::new(0);

        // First two should not prune
        assert!(!patient.prune(&study, &trial));
        assert!(!patient.prune(&study, &trial));
        // Third should prune
        assert!(patient.prune(&study, &trial));
    }
}
