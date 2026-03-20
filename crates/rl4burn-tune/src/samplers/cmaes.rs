use rand::prelude::*;
use std::sync::Mutex;

use crate::distributions::Distribution;
use crate::study::Study;
use crate::trial::{Trial, TrialState};

use super::Sampler;

/// CMA-ES (Covariance Matrix Adaptation Evolution Strategy) sampler.
///
/// This implements a simplified CMA-ES that maintains a multivariate Gaussian
/// search distribution, adapting its mean and covariance based on the best
/// trials. For parameters it hasn't seen enough trials for, it falls back to
/// random sampling.
///
/// This is a separable CMA-ES variant (diagonal covariance) for simplicity,
/// which works well when parameters are roughly independent.
pub struct CmaEsSampler {
    config: CmaEsConfig,
    random_sampler: super::RandomSampler,
    state: Mutex<CmaEsState>,
}

/// Configuration for the CMA-ES sampler.
#[derive(Debug, Clone)]
pub struct CmaEsConfig {
    /// Number of startup trials before CMA-ES kicks in.
    pub n_startup_trials: usize,
    /// Population size (lambda). If 0, uses `4 + floor(3 * ln(n_params))`.
    pub population_size: usize,
    /// Initial step size (sigma0).
    pub sigma0: f64,
}

impl Default for CmaEsConfig {
    fn default() -> Self {
        Self {
            n_startup_trials: 10,
            population_size: 0,
            sigma0: 0.5,
        }
    }
}

/// Internal state of CMA-ES, updated after each generation.
#[derive(Debug)]
struct CmaEsState {
    rng: StdRng,
    /// Mean of the search distribution (one entry per parameter, in [0,1] space).
    mean: Vec<f64>,
    /// Standard deviations (diagonal of the covariance, in [0,1] space).
    sigma: Vec<f64>,
    /// Overall step size.
    step_size: f64,
    /// Parameter names in canonical order.
    param_names: Vec<String>,
    /// Number of generations completed.
    generation: usize,
    /// Whether the state has been initialized with parameter names.
    initialized: bool,
}

impl CmaEsSampler {
    pub fn new(config: CmaEsConfig, seed: u64) -> Self {
        Self {
            config: config.clone(),
            random_sampler: super::RandomSampler::new(seed),
            state: Mutex::new(CmaEsState {
                rng: StdRng::seed_from_u64(seed.wrapping_add(2)),
                mean: Vec::new(),
                sigma: Vec::new(),
                step_size: config.sigma0,
                param_names: Vec::new(),
                generation: 0,
                initialized: false,
            }),
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        Self::new(CmaEsConfig::default(), seed)
    }

    /// Get the effective population size.
    #[allow(dead_code)]
    fn lambda(&self, n_params: usize) -> usize {
        if self.config.population_size > 0 {
            self.config.population_size
        } else {
            4 + (3.0 * (n_params as f64).ln()).floor() as usize
        }
    }

    /// Update the CMA-ES state based on completed trials.
    fn update_state(&self, study: &Study, param_name: &str) {
        let mut state = self.state.lock().unwrap();

        // Initialize parameter tracking on first call
        if !state.initialized || !state.param_names.contains(&param_name.to_string()) {
            if !state.param_names.contains(&param_name.to_string()) {
                state.param_names.push(param_name.to_string());
                state.mean.push(0.5); // Start at center of [0,1]
                state.sigma.push(self.config.sigma0);
            }
            state.initialized = true;
        }

        // Find the parameter index
        let param_idx = state
            .param_names
            .iter()
            .position(|n| n == param_name)
            .unwrap();

        // Gather completed trials with this parameter
        let mut completed: Vec<_> = study
            .trials()
            .iter()
            .filter(|t| t.state == TrialState::Complete && t.params.contains_key(param_name))
            .collect();

        if completed.len() < 2 {
            return;
        }

        // Sort by objective
        completed.sort_by(|a, b| {
            let va = a.value.unwrap_or(f64::INFINITY);
            let vb = b.value.unwrap_or(f64::INFINITY);
            match study.direction() {
                crate::study::Direction::Minimize => {
                    va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                }
                crate::study::Direction::Maximize => {
                    vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
                }
            }
        });

        // Use the best mu_eff trials to update the mean
        let n = completed.len();
        let mu_eff = (n as f64 / 2.0).ceil() as usize;
        let selected = &completed[..mu_eff.min(n)];

        // Compute weights for selected trials (log-linear, as in CMA-ES)
        let weights: Vec<f64> = (0..selected.len())
            .map(|i| ((mu_eff as f64 + 0.5).ln() - (i as f64 + 1.0).ln()).max(0.0))
            .collect();
        let w_sum: f64 = weights.iter().sum();

        if w_sum <= 0.0 {
            return;
        }

        // Weighted mean of the best parameter values (in raw space, then track)
        let new_mean: f64 = selected
            .iter()
            .zip(weights.iter())
            .map(|(t, &w)| {
                let raw = t.params[param_name];
                // We track means in [0,1] approximately
                w * raw
            })
            .sum::<f64>()
            / w_sum;

        // Update sigma based on spread of selected trials
        let variance: f64 = selected
            .iter()
            .zip(weights.iter())
            .map(|(t, &w)| {
                let raw = t.params[param_name];
                let diff = raw - new_mean;
                w * diff * diff
            })
            .sum::<f64>()
            / w_sum;

        let new_sigma = variance.sqrt().max(1e-12);

        state.mean[param_idx] = new_mean;
        state.sigma[param_idx] = new_sigma;
        state.generation += 1;
    }
}

impl Sampler for CmaEsSampler {
    fn sample(
        &self,
        study: &Study,
        trial: &Trial,
        param_name: &str,
        distribution: &Distribution,
    ) -> f64 {
        let completed_count = study
            .trials()
            .iter()
            .filter(|t| t.state == TrialState::Complete && t.params.contains_key(param_name))
            .count();

        if completed_count < self.config.n_startup_trials {
            return self
                .random_sampler
                .sample(study, trial, param_name, distribution);
        }

        // Update the CMA-ES state
        self.update_state(study, param_name);

        let mut state = self.state.lock().unwrap();
        let param_idx = state
            .param_names
            .iter()
            .position(|n| n == param_name);

        let param_idx = match param_idx {
            Some(idx) => idx,
            None => {
                drop(state);
                return self
                    .random_sampler
                    .sample(study, trial, param_name, distribution);
            }
        };

        let mean = state.mean[param_idx];
        let sigma = state.sigma[param_idx];

        // Sample from N(mean, sigma^2 * step_size^2)
        let normal: f64 = state.rng.sample(rand_distr::StandardNormal);
        let raw_sample = mean + sigma * state.step_size * normal;

        match distribution {
            Distribution::Float(d) => {
                raw_sample.clamp(d.low, d.high)
            }
            Distribution::Int(d) => {
                let clamped = raw_sample.clamp(d.low as f64, d.high as f64);
                let step = d.step.unwrap_or(1);
                let shifted = clamped - d.low as f64;
                let quantized = (shifted / step as f64).round() * step as f64 + d.low as f64;
                (quantized.round() as i64).clamp(d.low, d.high) as f64
            }
            Distribution::Categorical(_) => {
                // CMA-ES doesn't naturally handle categoricals — fall back to random
                drop(state);
                self.random_sampler
                    .sample(study, trial, param_name, distribution)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_cmaes_float_in_bounds() {
        let sampler = CmaEsSampler::with_seed(42);
        let dist = Distribution::Float(crate::distributions::FloatDistribution::new(
            0.0, 10.0, false, None,
        ));
        let mut study = Study::new_default();

        // Add startup trials
        for i in 0..15 {
            let value = (i as f64) * 0.5 + 1.0;
            let mut params = HashMap::new();
            params.insert("x".to_string(), value);
            study.add_completed_trial(params, (value - 3.0).powi(2));
        }

        let trial = Trial::new(15);
        let v = sampler.sample(&study, &trial, "x", &dist);
        assert!(v >= 0.0 && v <= 10.0, "CMA-ES suggested {v} outside [0, 10]");
    }

    #[test]
    fn test_cmaes_int_in_bounds() {
        let sampler = CmaEsSampler::with_seed(42);
        let dist = Distribution::Int(crate::distributions::IntDistribution::new(
            1, 50, false, None,
        ));
        let mut study = Study::new_default();

        for i in 0..15 {
            let value = (i * 3 + 1) as f64;
            let mut params = HashMap::new();
            params.insert("n".to_string(), value);
            study.add_completed_trial(params, (value - 20.0).powi(2));
        }

        let trial = Trial::new(15);
        let v = sampler.sample(&study, &trial, "n", &dist) as i64;
        assert!(v >= 1 && v <= 50, "CMA-ES suggested int {v} outside [1, 50]");
    }

    #[test]
    fn test_cmaes_during_startup_uses_random() {
        let sampler = CmaEsSampler::with_seed(42);
        let dist = Distribution::Float(crate::distributions::FloatDistribution::new(
            0.0, 1.0, false, None,
        ));
        let study = Study::new_default(); // No trials
        let trial = Trial::new(0);

        // Should not panic, falls back to random
        let v = sampler.sample(&study, &trial, "x", &dist);
        assert!(v >= 0.0 && v <= 1.0);
    }

    #[test]
    fn test_cmaes_converges_toward_optimum() {
        let sampler = CmaEsSampler::new(
            CmaEsConfig {
                n_startup_trials: 5,
                population_size: 0,
                sigma0: 2.0,
            },
            42,
        );
        let dist = Distribution::Float(crate::distributions::FloatDistribution::new(
            -10.0, 10.0, false, None,
        ));
        let mut study = Study::new_default();

        // Run many trials minimizing (x - 3)^2
        for _ in 0..50 {
            let trial = Trial::new(study.trials().len());
            let x = sampler.sample(&study, &trial, "x", &dist);
            let obj = (x - 3.0).powi(2);
            let mut params = HashMap::new();
            params.insert("x".to_string(), x);
            study.add_completed_trial(params, obj);
        }

        // The best value should be close to 0 (x near 3)
        let best = study.best_value().unwrap();
        assert!(best < 5.0, "CMA-ES should converge, best={best}");
    }
}
