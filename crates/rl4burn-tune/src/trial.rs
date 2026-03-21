use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::distributions::Distribution;

/// State of a trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrialState {
    /// Trial has been created but not yet started (for ask/tell and enqueue).
    Waiting,
    /// Trial is currently being evaluated.
    Running,
    /// Trial completed successfully.
    Complete,
    /// Trial was pruned (stopped early).
    Pruned,
    /// Trial failed with an error.
    Fail,
}

/// A frozen (immutable) record of a completed or pruned trial.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenTrial {
    pub number: usize,
    pub state: TrialState,
    pub value: Option<f64>,
    /// Multiple objective values for multi-objective optimization.
    /// For single-objective studies, this is `None` and `value` is used instead.
    pub values: Option<Vec<f64>>,
    pub params: HashMap<String, f64>,
    pub intermediate_values: HashMap<usize, f64>,
    /// User-defined attributes for arbitrary metadata.
    pub user_attrs: HashMap<String, String>,
    /// System-defined attributes used internally by samplers/pruners.
    pub system_attrs: HashMap<String, String>,
    /// Constraint values for constrained optimization.
    /// Negative values indicate feasible constraints, positive values indicate violations.
    /// A trial is feasible if all constraint values are <= 0.
    pub constraint_values: Option<Vec<f64>>,
}

impl FrozenTrial {
    pub fn new(number: usize) -> Self {
        Self {
            number,
            state: TrialState::Running,
            value: None,
            values: None,
            params: HashMap::new(),
            intermediate_values: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            constraint_values: None,
        }
    }

    /// Check whether this trial is feasible (all constraints satisfied).
    /// Returns true if no constraints are set.
    pub fn is_feasible(&self) -> bool {
        match &self.constraint_values {
            None => true,
            Some(cv) => cv.iter().all(|&v| v <= 0.0),
        }
    }

    /// Total constraint violation (sum of positive constraint values).
    pub fn total_violation(&self) -> f64 {
        match &self.constraint_values {
            None => 0.0,
            Some(cv) => cv.iter().map(|&v| v.max(0.0)).sum(),
        }
    }

    /// Report an intermediate value at the given step.
    pub fn report(&mut self, step: usize, value: f64) {
        self.intermediate_values.insert(step, value);
    }

    /// Get the last reported step number.
    pub fn last_step(&self) -> usize {
        self.intermediate_values.keys().copied().max().unwrap_or(0)
    }

    /// Validate the trial's internal consistency.
    ///
    /// Returns `Err` with a description if the trial is malformed.
    pub fn validate(&self) -> Result<(), String> {
        match self.state {
            TrialState::Complete => {
                if self.value.is_none() {
                    return Err(format!(
                        "Trial {} is Complete but has no value",
                        self.number
                    ));
                }
            }
            TrialState::Waiting => {
                if self.value.is_some() {
                    return Err(format!(
                        "Trial {} is Waiting but has a value",
                        self.number
                    ));
                }
            }
            TrialState::Running => {
                if self.value.is_some() {
                    return Err(format!(
                        "Trial {} is Running but has a value",
                        self.number
                    ));
                }
            }
            TrialState::Pruned | TrialState::Fail => {}
        }
        Ok(())
    }
}

/// A trial in progress, providing the suggest API.
pub struct Trial {
    pub number: usize,
    pub params: HashMap<String, f64>,
    pub intermediate_values: HashMap<usize, f64>,
    /// Whether this trial has been marked as pruned.
    pub pruned: bool,
    /// User-defined attributes for arbitrary metadata.
    pub user_attrs: HashMap<String, String>,
    /// System-defined attributes used internally by samplers/pruners.
    pub system_attrs: HashMap<String, String>,
    /// Constraint values for constrained optimization.
    pub constraint_values: Option<Vec<f64>>,
}

impl Trial {
    pub fn new(number: usize) -> Self {
        Self {
            number,
            params: HashMap::new(),
            intermediate_values: HashMap::new(),
            pruned: false,
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            constraint_values: None,
        }
    }

    /// Set constraint values for this trial.
    /// Negative values indicate feasible constraints, positive values indicate violations.
    pub fn set_constraints(&mut self, values: Vec<f64>) {
        self.constraint_values = Some(values);
    }

    /// Set a user-defined attribute.
    pub fn set_user_attr(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.user_attrs.insert(key.into(), value.into());
    }

    /// Set a system-defined attribute.
    pub fn set_system_attr(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.system_attrs.insert(key.into(), value.into());
    }

    /// Suggest a parameter value.
    ///
    /// If this parameter has already been suggested in this trial, returns the
    /// cached value. Otherwise, delegates to the sampler.
    pub fn suggest(
        &mut self,
        name: &str,
        distribution: &Distribution,
        sampler: &dyn crate::samplers::Sampler,
        study: &crate::study::Study,
    ) -> f64 {
        if let Some(&cached) = self.params.get(name) {
            return cached;
        }

        let value = sampler.sample(study, self, name, distribution);
        self.params.insert(name.to_string(), value);
        value
    }

    /// Suggest a float parameter.
    pub fn suggest_float(
        &mut self,
        name: &str,
        low: f64,
        high: f64,
        log: bool,
        step: Option<f64>,
        sampler: &dyn crate::samplers::Sampler,
        study: &crate::study::Study,
    ) -> f64 {
        let dist = Distribution::Float(crate::distributions::FloatDistribution::new(
            low, high, log, step,
        ));
        self.suggest(name, &dist, sampler, study)
    }

    /// Suggest an integer parameter.
    pub fn suggest_int(
        &mut self,
        name: &str,
        low: i64,
        high: i64,
        log: bool,
        step: Option<i64>,
        sampler: &dyn crate::samplers::Sampler,
        study: &crate::study::Study,
    ) -> i64 {
        let dist = Distribution::Int(crate::distributions::IntDistribution::new(
            low, high, log, step,
        ));
        self.suggest(name, &dist, sampler, study) as i64
    }

    /// Suggest a categorical parameter (returns the index).
    pub fn suggest_categorical(
        &mut self,
        name: &str,
        choices: Vec<String>,
        sampler: &dyn crate::samplers::Sampler,
        study: &crate::study::Study,
    ) -> usize {
        let dist =
            Distribution::Categorical(crate::distributions::CategoricalDistribution::new(choices));
        self.suggest(name, &dist, sampler, study) as usize
    }

    /// Report an intermediate value at the given step.
    pub fn report(&mut self, step: usize, value: f64) {
        self.intermediate_values.insert(step, value);
    }

    /// Check if the trial should be pruned, using the given study and pruner.
    ///
    /// Creates a temporary FrozenTrial snapshot to pass to the pruner.
    /// If the pruner says to prune, marks this trial as pruned and returns true.
    pub fn should_prune(
        &mut self,
        study: &crate::study::Study,
        pruner: Option<&dyn crate::pruners::Pruner>,
    ) -> bool {
        let pruner = match pruner {
            Some(p) => p,
            None => return false,
        };

        let frozen = FrozenTrial {
            number: self.number,
            state: TrialState::Running,
            value: None,
            values: None,
            params: self.params.clone(),
            intermediate_values: self.intermediate_values.clone(),
            user_attrs: self.user_attrs.clone(),
            system_attrs: self.system_attrs.clone(),
            constraint_values: self.constraint_values.clone(),
        };

        if pruner.prune(study, &frozen) {
            self.pruned = true;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study::Study;

    #[test]
    fn test_trial_suggest_caches() {
        let sampler = crate::samplers::RandomSampler::new(42);
        let study = Study::new_default();
        let mut trial = Trial::new(0);

        let dist = Distribution::Float(crate::distributions::FloatDistribution::new(
            0.0, 1.0, false, None,
        ));

        let v1 = trial.suggest("x", &dist, &sampler, &study);
        let v2 = trial.suggest("x", &dist, &sampler, &study);
        assert_eq!(v1, v2, "suggest should cache parameter values");
    }

    #[test]
    fn test_trial_suggest_float() {
        let sampler = crate::samplers::RandomSampler::new(42);
        let study = Study::new_default();
        let mut trial = Trial::new(0);

        let v = trial.suggest_float("lr", 1e-5, 1e-1, true, None, &sampler, &study);
        assert!(v >= 1e-5 && v <= 1e-1);
    }

    #[test]
    fn test_trial_suggest_int() {
        let sampler = crate::samplers::RandomSampler::new(42);
        let study = Study::new_default();
        let mut trial = Trial::new(0);

        let v = trial.suggest_int("n_layers", 1, 10, false, None, &sampler, &study);
        assert!(v >= 1 && v <= 10);
    }

    #[test]
    fn test_trial_suggest_categorical() {
        let sampler = crate::samplers::RandomSampler::new(42);
        let study = Study::new_default();
        let mut trial = Trial::new(0);

        let v = trial.suggest_categorical(
            "optimizer",
            vec!["adam".into(), "sgd".into(), "rmsprop".into()],
            &sampler,
            &study,
        );
        assert!(v < 3);
    }

    #[test]
    fn test_frozen_trial_report() {
        let mut ft = FrozenTrial::new(0);
        ft.report(0, 0.5);
        ft.report(5, 0.3);
        ft.report(10, 0.1);

        assert_eq!(ft.last_step(), 10);
        assert_eq!(ft.intermediate_values[&5], 0.3);
    }
}
