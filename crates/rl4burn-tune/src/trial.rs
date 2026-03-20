use std::collections::HashMap;

use crate::distributions::Distribution;

/// State of a trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrialState {
    Running,
    Complete,
    Pruned,
    Fail,
}

/// A frozen (immutable) record of a completed or pruned trial.
#[derive(Debug, Clone)]
pub struct FrozenTrial {
    pub number: usize,
    pub state: TrialState,
    pub value: Option<f64>,
    pub params: HashMap<String, f64>,
    pub intermediate_values: HashMap<usize, f64>,
}

impl FrozenTrial {
    pub fn new(number: usize) -> Self {
        Self {
            number,
            state: TrialState::Running,
            value: None,
            params: HashMap::new(),
            intermediate_values: HashMap::new(),
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
}

/// A trial in progress, providing the suggest API.
pub struct Trial {
    pub number: usize,
    pub params: HashMap<String, f64>,
    pub intermediate_values: HashMap<usize, f64>,
    /// Whether this trial has been marked as pruned.
    pub pruned: bool,
}

impl Trial {
    pub fn new(number: usize) -> Self {
        Self {
            number,
            params: HashMap::new(),
            intermediate_values: HashMap::new(),
            pruned: false,
        }
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
            params: self.params.clone(),
            intermediate_values: self.intermediate_values.clone(),
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
