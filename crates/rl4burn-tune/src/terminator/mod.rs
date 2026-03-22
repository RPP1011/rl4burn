//! Automatic convergence detection for studies.
//!
//! Terminators inspect the trial history and decide whether further
//! optimization is unlikely to yield meaningful improvements.

mod regret_bound;

pub use regret_bound::{RegretBoundTerminator, StaticErrorTerminator};

use crate::study::Study;

/// Decides whether a study should stop accepting new trials.
pub trait Terminator: Send + Sync {
    /// Returns true if the study should be terminated.
    fn should_terminate(&self, study: &Study) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study::Direction;
    use std::collections::HashMap;

    /// Helper: build a study with the given completed trial values.
    fn study_with_values(direction: Direction, values: &[f64]) -> Study {
        let mut study = Study::new(direction);
        for &v in values {
            study.add_completed_trial(HashMap::new(), v);
        }
        study
    }

    // -- RegretBoundTerminator tests ------------------------------------------

    #[test]
    fn regret_bound_does_not_terminate_before_min_trials() {
        let term = RegretBoundTerminator::new(10, 5, 1e-4);
        let study = study_with_values(Direction::Minimize, &[5.0, 4.0, 3.0]);
        assert!(!term.should_terminate(&study));
    }

    #[test]
    fn regret_bound_terminates_on_flat_minimize() {
        // All values identical after min_trials => no improvement => should terminate.
        let term = RegretBoundTerminator::new(5, 3, 1e-4);
        let vals: Vec<f64> = vec![1.0; 10];
        let study = study_with_values(Direction::Minimize, &vals);
        assert!(term.should_terminate(&study));
    }

    #[test]
    fn regret_bound_terminates_on_flat_maximize() {
        let term = RegretBoundTerminator::new(5, 3, 1e-4);
        let vals: Vec<f64> = vec![9.0; 10];
        let study = study_with_values(Direction::Maximize, &vals);
        assert!(term.should_terminate(&study));
    }

    #[test]
    fn regret_bound_does_not_terminate_when_improving() {
        let term = RegretBoundTerminator::new(3, 5, 1e-4);
        // Steadily improving values (minimizing).
        let vals: Vec<f64> = (0..10).map(|i| 10.0 - i as f64).collect();
        let study = study_with_values(Direction::Minimize, &vals);
        assert!(!term.should_terminate(&study));
    }

    // -- StaticErrorTerminator tests ------------------------------------------

    #[test]
    fn static_error_terminates_when_target_reached_minimize() {
        let term = StaticErrorTerminator::new(2.0, Direction::Minimize);
        let study = study_with_values(Direction::Minimize, &[5.0, 3.0, 1.5]);
        assert!(term.should_terminate(&study));
    }

    #[test]
    fn static_error_does_not_terminate_when_target_not_reached_minimize() {
        let term = StaticErrorTerminator::new(2.0, Direction::Minimize);
        let study = study_with_values(Direction::Minimize, &[5.0, 3.0]);
        assert!(!term.should_terminate(&study));
    }

    #[test]
    fn static_error_terminates_when_target_reached_maximize() {
        let term = StaticErrorTerminator::new(8.0, Direction::Maximize);
        let study = study_with_values(Direction::Maximize, &[5.0, 7.0, 9.0]);
        assert!(term.should_terminate(&study));
    }

    #[test]
    fn static_error_does_not_terminate_when_target_not_reached_maximize() {
        let term = StaticErrorTerminator::new(8.0, Direction::Maximize);
        let study = study_with_values(Direction::Maximize, &[5.0, 7.0]);
        assert!(!term.should_terminate(&study));
    }

    #[test]
    fn static_error_no_trials() {
        let term = StaticErrorTerminator::new(1.0, Direction::Minimize);
        let study = Study::new(Direction::Minimize);
        assert!(!term.should_terminate(&study));
    }
}
