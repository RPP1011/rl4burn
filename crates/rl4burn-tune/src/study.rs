use std::collections::HashMap;

use crate::trial::{FrozenTrial, TrialState};

/// Optimization direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Minimize,
    Maximize,
}

/// Configuration for creating a study.
#[derive(Debug, Clone)]
pub struct StudyConfig {
    pub direction: Direction,
}

/// A study holds the trial history and manages optimization.
pub struct Study {
    direction: Direction,
    trials: Vec<FrozenTrial>,
}

impl Study {
    /// Create a new study with the given direction.
    pub fn new(direction: Direction) -> Self {
        Self {
            direction,
            trials: Vec::new(),
        }
    }

    /// Create a study with default settings (minimize).
    pub fn new_default() -> Self {
        Self::new(Direction::Minimize)
    }

    /// Get the optimization direction.
    pub fn direction(&self) -> Direction {
        self.direction
    }

    /// Get all trials.
    pub fn trials(&self) -> &[FrozenTrial] {
        &self.trials
    }

    /// Add a raw trial.
    pub fn add_trial(&mut self, trial: FrozenTrial) {
        self.trials.push(trial);
    }

    /// Add a completed trial with known parameters and objective value.
    pub fn add_completed_trial(&mut self, params: HashMap<String, f64>, value: f64) {
        let trial_number = self.trials.len();
        let mut trial = FrozenTrial::new(trial_number);
        trial.params = params;
        trial.value = Some(value);
        trial.state = TrialState::Complete;
        self.trials.push(trial);
    }

    /// Get the best trial (lowest value for minimize, highest for maximize).
    pub fn best_trial(&self) -> Option<&FrozenTrial> {
        self.trials
            .iter()
            .filter(|t| t.state == TrialState::Complete && t.value.is_some())
            .min_by(|a, b| {
                let va = a.value.unwrap();
                let vb = b.value.unwrap();
                match self.direction {
                    Direction::Minimize => {
                        va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                    }
                    Direction::Maximize => {
                        vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
                    }
                }
            })
    }

    /// Get the best objective value found so far.
    pub fn best_value(&self) -> Option<f64> {
        self.best_trial().and_then(|t| t.value)
    }

    /// Get the number of completed trials.
    pub fn n_completed(&self) -> usize {
        self.trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .count()
    }

    /// Run the optimization loop.
    ///
    /// `objective` receives a mutable `Trial` and should return the objective value.
    /// `sampler` is used to suggest parameter values.
    /// `pruner` is optionally used to decide early stopping.
    pub fn optimize<F>(
        &mut self,
        n_trials: usize,
        sampler: &dyn crate::samplers::Sampler,
        pruner: Option<&dyn crate::pruners::Pruner>,
        mut objective: F,
    ) where
        F: FnMut(&mut crate::trial::Trial, &dyn crate::samplers::Sampler, &Self) -> f64,
    {
        for _ in 0..n_trials {
            let trial_number = self.trials.len();
            let mut trial = crate::trial::Trial::new(trial_number);

            let value = objective(&mut trial, sampler, self);

            let mut frozen = FrozenTrial::new(trial_number);
            frozen.params = trial.params.clone();
            frozen.value = Some(value);
            frozen.intermediate_values = trial.intermediate_values.clone();

            // Check pruning
            if let Some(p) = pruner {
                if p.prune(self, &frozen) {
                    frozen.state = TrialState::Pruned;
                } else {
                    frozen.state = TrialState::Complete;
                }
            } else {
                frozen.state = TrialState::Complete;
            }

            self.trials.push(frozen);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_study_best_trial_minimize() {
        let mut study = Study::new(Direction::Minimize);
        let mut params = HashMap::new();
        params.insert("x".to_string(), 5.0);
        study.add_completed_trial(params.clone(), 10.0);
        study.add_completed_trial(params.clone(), 5.0);
        study.add_completed_trial(params.clone(), 8.0);

        assert_eq!(study.best_value(), Some(5.0));
    }

    #[test]
    fn test_study_best_trial_maximize() {
        let mut study = Study::new(Direction::Maximize);
        let mut params = HashMap::new();
        params.insert("x".to_string(), 5.0);
        study.add_completed_trial(params.clone(), 10.0);
        study.add_completed_trial(params.clone(), 5.0);
        study.add_completed_trial(params.clone(), 8.0);

        assert_eq!(study.best_value(), Some(10.0));
    }

    #[test]
    fn test_study_optimize() {
        let mut study = Study::new(Direction::Minimize);
        let sampler = crate::samplers::RandomSampler::new(42);

        study.optimize(20, &sampler, None, |trial, sampler, study| {
            let dist = crate::distributions::Distribution::Float(
                crate::distributions::FloatDistribution::new(-5.0, 5.0, false, None),
            );
            let x = trial.suggest("x", &dist, sampler, study);
            x * x // minimize x^2
        });

        assert_eq!(study.n_completed(), 20);
        assert!(study.best_value().unwrap() < 25.0);
    }
}
