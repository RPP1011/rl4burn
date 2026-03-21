use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::trial::{FrozenTrial, TrialState};

/// A callback invoked after each trial completes during `optimize()`.
pub trait Callback: Send + Sync {
    /// Called after a trial is completed or pruned.
    fn on_trial_complete(&self, study: &Study, trial: &FrozenTrial);
}

/// Optimization direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Minimize,
    Maximize,
}

/// Configuration for creating a study.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudyConfig {
    pub direction: Direction,
}

/// A study holds the trial history and manages optimization.
pub struct Study {
    directions: Vec<Direction>,
    trials: Vec<FrozenTrial>,
    callbacks: Vec<Box<dyn Callback>>,
    stopped: bool,
}

impl Study {
    /// Create a new single-objective study with the given direction.
    pub fn new(direction: Direction) -> Self {
        Self {
            directions: vec![direction],
            trials: Vec::new(),
            callbacks: Vec::new(),
            stopped: false,
        }
    }

    /// Create a new multi-objective study with the given directions.
    pub fn new_multi(directions: Vec<Direction>) -> Self {
        assert!(
            !directions.is_empty(),
            "At least one direction is required"
        );
        Self {
            directions,
            trials: Vec::new(),
            callbacks: Vec::new(),
            stopped: false,
        }
    }

    /// Create a study with default settings (minimize).
    pub fn new_default() -> Self {
        Self::new(Direction::Minimize)
    }

    /// Get the optimization direction (for single-objective studies).
    ///
    /// # Panics
    /// Panics if the study has multiple directions.
    pub fn direction(&self) -> Direction {
        assert_eq!(
            self.directions.len(),
            1,
            "Use directions() for multi-objective studies"
        );
        self.directions[0]
    }

    /// Get all optimization directions.
    pub fn directions(&self) -> &[Direction] {
        &self.directions
    }

    /// Check if this is a multi-objective study.
    pub fn is_multi_objective(&self) -> bool {
        self.directions.len() > 1
    }

    /// Get all trials.
    pub fn trials(&self) -> &[FrozenTrial] {
        &self.trials
    }

    /// Add a raw trial.
    ///
    /// # Panics
    /// Panics if the trial fails validation (e.g., Complete without a value).
    pub fn add_trial(&mut self, trial: FrozenTrial) {
        if let Err(msg) = trial.validate() {
            panic!("Invalid trial: {msg}");
        }
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
    ///
    /// For multi-objective studies, use `best_trials()` instead.
    ///
    /// # Panics
    /// Panics if the study has multiple directions.
    pub fn best_trial(&self) -> Option<&FrozenTrial> {
        let dir = self.direction(); // panics if multi-objective
        self.trials
            .iter()
            .filter(|t| t.state == TrialState::Complete && t.value.is_some())
            .min_by(|a, b| {
                let va = a.value.unwrap();
                let vb = b.value.unwrap();
                match dir {
                    Direction::Minimize => {
                        va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                    }
                    Direction::Maximize => {
                        vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
                    }
                }
            })
    }

    /// Get the Pareto-optimal trials for multi-objective studies.
    ///
    /// Returns all non-dominated completed trials. For single-objective studies,
    /// this returns the single best trial.
    pub fn best_trials(&self) -> Vec<&FrozenTrial> {
        let completed: Vec<&FrozenTrial> = self
            .trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .collect();

        if completed.is_empty() {
            return vec![];
        }

        if self.directions.len() == 1 {
            return self.best_trial().into_iter().collect();
        }

        // Multi-objective: get values and do non-dominated sort
        let values: Vec<Vec<f64>> = completed
            .iter()
            .map(|t| {
                t.values
                    .clone()
                    .unwrap_or_else(|| t.value.map(|v| vec![v]).unwrap_or_default())
            })
            .collect();

        let fronts =
            crate::multi_objective::non_dominated_sort(&values, &self.directions);

        if fronts.is_empty() {
            return vec![];
        }

        fronts[0].iter().map(|&i| completed[i]).collect()
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

    /// Register a callback to be invoked after each trial in `optimize()`.
    pub fn add_callback(&mut self, callback: Box<dyn Callback>) {
        self.callbacks.push(callback);
    }

    /// Request the study to stop optimization after the current trial.
    pub fn stop(&mut self) {
        self.stopped = true;
    }

    /// Check if the study has been stopped.
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    /// Create a new trial for the ask/tell interface.
    ///
    /// Returns a `Trial` that can be used to suggest parameters. After evaluation,
    /// call `tell()` to record the result.
    pub fn ask(&self) -> crate::trial::Trial {
        let trial_number = self.trials.len();
        crate::trial::Trial::new(trial_number)
    }

    /// Record the result of a trial from the ask/tell interface.
    ///
    /// `trial` is the trial returned by `ask()`, after parameter suggestions
    /// and evaluation. `state` should be `Complete` or `Pruned`.
    /// `value` is the objective value (required for `Complete`, optional for `Pruned`).
    pub fn tell(
        &mut self,
        trial: crate::trial::Trial,
        state: TrialState,
        value: Option<f64>,
    ) {
        let mut frozen = FrozenTrial::new(trial.number);
        frozen.params = trial.params;
        frozen.intermediate_values = trial.intermediate_values;
        frozen.user_attrs = trial.user_attrs;
        frozen.system_attrs = trial.system_attrs;
        frozen.state = state;
        frozen.value = value;
        if let Err(msg) = frozen.validate() {
            panic!("Invalid trial in tell(): {msg}");
        }
        self.trials.push(frozen);
    }

    /// Enqueue a trial with pre-set parameters.
    ///
    /// The enqueued trial will be in `Waiting` state. When a sampler encounters
    /// an enqueued trial (via `ask()`), it should use the pre-set parameters
    /// instead of sampling new ones.
    pub fn enqueue_trial(&mut self, params: HashMap<String, f64>) {
        let trial_number = self.trials.len();
        let mut frozen = FrozenTrial::new(trial_number);
        frozen.params = params;
        frozen.state = TrialState::Waiting;
        self.trials.push(frozen);
    }

    /// Run the optimization loop.
    ///
    /// `objective` receives a mutable `Trial` and should return the objective value.
    /// The objective can call `trial.report(step, value)` to report intermediate
    /// values and `trial.should_prune(study, pruner)` to check if the trial
    /// should be stopped early. If the objective returns `Err(Pruned)`, the
    /// trial is recorded as pruned.
    ///
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
            if self.stopped {
                break;
            }

            let trial_number = self.trials.len();
            let mut trial = crate::trial::Trial::new(trial_number);

            let value = objective(&mut trial, sampler, self);

            let mut frozen = FrozenTrial::new(trial_number);
            frozen.params = trial.params.clone();
            frozen.intermediate_values = trial.intermediate_values.clone();
            frozen.user_attrs = trial.user_attrs.clone();
            frozen.system_attrs = trial.system_attrs.clone();

            // Check if the trial was pruned mid-objective
            if trial.pruned {
                frozen.state = TrialState::Pruned;
                // Use the last reported intermediate value, not the objective return
                frozen.value = trial
                    .intermediate_values
                    .iter()
                    .max_by_key(|(&step, _)| step)
                    .map(|(_, &v)| v);
            } else {
                frozen.value = Some(value);
                if let Some(p) = pruner {
                    // Post-hoc pruning check
                    if p.prune(self, &frozen) {
                        frozen.state = TrialState::Pruned;
                    } else {
                        frozen.state = TrialState::Complete;
                    }
                } else {
                    frozen.state = TrialState::Complete;
                }
            }

            self.trials.push(frozen);

            // Invoke callbacks
            let last_trial = self.trials.last().unwrap();
            for cb in &self.callbacks {
                cb.on_trial_complete(self, last_trial);
            }
        }
    }

    /// Run the optimization loop with a fallible objective.
    ///
    /// The objective returns `Ok(value)` for success or `Err(())` to signal
    /// pruning (the trial was stopped early). This enables mid-objective pruning:
    ///
    /// ```ignore
    /// study.optimize_prunable(100, &sampler, Some(&pruner), |trial, sampler, study, pruner| {
    ///     let lr = trial.suggest_float("lr", 1e-5, 1e-1, true, None, sampler, study);
    ///     for epoch in 0..100 {
    ///         let loss = train_epoch(lr);
    ///         trial.report(epoch, loss);
    ///         if trial.should_prune(study, pruner) {
    ///             return Err(());
    ///         }
    ///     }
    ///     Ok(final_loss)
    /// });
    /// ```
    pub fn optimize_prunable<F>(
        &mut self,
        n_trials: usize,
        sampler: &dyn crate::samplers::Sampler,
        pruner: Option<&dyn crate::pruners::Pruner>,
        mut objective: F,
    ) where
        F: FnMut(
            &mut crate::trial::Trial,
            &dyn crate::samplers::Sampler,
            &Self,
            Option<&dyn crate::pruners::Pruner>,
        ) -> Result<f64, ()>,
    {
        for _ in 0..n_trials {
            if self.stopped {
                break;
            }

            let trial_number = self.trials.len();
            let mut trial = crate::trial::Trial::new(trial_number);

            let result = objective(&mut trial, sampler, self, pruner);

            let mut frozen = FrozenTrial::new(trial_number);
            frozen.params = trial.params.clone();
            frozen.intermediate_values = trial.intermediate_values.clone();
            frozen.user_attrs = trial.user_attrs.clone();
            frozen.system_attrs = trial.system_attrs.clone();

            match result {
                Ok(value) => {
                    frozen.value = Some(value);
                    frozen.state = TrialState::Complete;
                }
                Err(()) => {
                    // Use the value at the highest reported step
                    frozen.value = trial
                        .intermediate_values
                        .iter()
                        .max_by_key(|(&step, _)| step)
                        .map(|(_, &v)| v);
                    frozen.state = TrialState::Pruned;
                }
            }

            self.trials.push(frozen);

            // Invoke callbacks
            let last_trial = self.trials.last().unwrap();
            for cb in &self.callbacks {
                cb.on_trial_complete(self, last_trial);
            }
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
    fn test_study_optimize_prunable() {
        use crate::pruners::MedianPruner;

        let mut study = Study::new(Direction::Minimize);
        let sampler = crate::samplers::RandomSampler::new(42);
        let pruner = MedianPruner::new(2, 0, 1);

        study.optimize_prunable(20, &sampler, Some(&pruner), |trial, sampler, study, pruner| {
            let dist = crate::distributions::Distribution::Float(
                crate::distributions::FloatDistribution::new(-5.0, 5.0, false, None),
            );
            let x = trial.suggest("x", &dist, sampler, study);

            // Simulate multi-step objective
            for step in 0..5 {
                let value = x * x + step as f64;
                trial.report(step, value);
                if trial.should_prune(study, pruner) {
                    return Err(());
                }
            }
            Ok(x * x)
        });

        assert_eq!(study.trials().len(), 20);
        // Some trials should have been pruned
        // At least some should complete (startup trials won't be pruned)
        let n_completed = study
            .trials()
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .count();
        assert!(n_completed > 0, "some trials should complete");
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
