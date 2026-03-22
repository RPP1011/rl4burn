//! Convergence-based terminators.

use crate::study::{Direction, Study};
use crate::trial::TrialState;

use super::Terminator;

/// Terminates when the expected improvement (regret bound) is small enough.
///
/// The terminator tracks the running best value across completed trials.
/// Once `min_trials` have finished, it checks whether the best value has
/// improved by at least `threshold` in the last `patience` trials.
/// If no such improvement is observed, the study is considered converged.
pub struct RegretBoundTerminator {
    /// Minimum number of completed trials before convergence checking begins.
    min_trials: usize,
    /// Number of consecutive trials without sufficient improvement before
    /// declaring convergence.
    patience: usize,
    /// Minimum absolute improvement to count as progress.
    threshold: f64,
}

impl RegretBoundTerminator {
    /// Create a new `RegretBoundTerminator`.
    ///
    /// * `min_trials` - do not consider termination until this many trials
    ///   have completed.
    /// * `patience` - how many consecutive trials without improvement are
    ///   tolerated.
    /// * `threshold` - the minimum absolute improvement to reset the patience
    ///   counter.
    pub fn new(min_trials: usize, patience: usize, threshold: f64) -> Self {
        Self {
            min_trials,
            patience,
            threshold,
        }
    }
}

impl Terminator for RegretBoundTerminator {
    fn should_terminate(&self, study: &Study) -> bool {
        let direction = study.direction();
        let completed: Vec<f64> = study
            .trials()
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .filter_map(|t| t.value)
            .collect();

        if completed.len() < self.min_trials {
            return false;
        }

        // Walk through the completed values and track the running best.
        // Count how many consecutive trials at the tail had no improvement.
        let mut best = completed[0];
        let mut trials_since_improvement: usize = 0;

        for &v in &completed[1..] {
            let improved = match direction {
                Direction::Minimize => best - v > self.threshold,
                Direction::Maximize => v - best > self.threshold,
            };
            if improved {
                best = v;
                trials_since_improvement = 0;
            } else {
                trials_since_improvement += 1;
            }
        }

        trials_since_improvement >= self.patience
    }
}

// ---------------------------------------------------------------------------

/// A simple terminator that stops once the best value crosses a fixed target.
///
/// For `Minimize` studies, termination happens when the best value is at or
/// below `target_value`.  For `Maximize` studies, when it is at or above.
pub struct StaticErrorTerminator {
    target_value: f64,
    direction: Direction,
}

impl StaticErrorTerminator {
    pub fn new(target_value: f64, direction: Direction) -> Self {
        Self {
            target_value,
            direction,
        }
    }
}

impl Terminator for StaticErrorTerminator {
    fn should_terminate(&self, study: &Study) -> bool {
        let best = match study.best_trial() {
            Some(t) => match t.value {
                Some(v) => v,
                None => return false,
            },
            None => return false,
        };

        match self.direction {
            Direction::Minimize => best <= self.target_value,
            Direction::Maximize => best >= self.target_value,
        }
    }
}
