use rand::prelude::*;
use std::sync::Mutex;

use crate::distributions::{
    int_transform_from_internal, int_transform_to_internal, transform_from_internal,
    transform_to_internal, Distribution,
};
use crate::study::Study;
use crate::trial::{Trial, TrialState};

use super::Sampler;

/// CMA-ES (Covariance Matrix Adaptation Evolution Strategy) sampler.
///
/// This implements a simplified separable CMA-ES that maintains a per-parameter
/// Gaussian search distribution in normalized [0,1] internal space. The mean
/// and sigma are adapted based on the weighted mean of the best trials,
/// with CMA-ES-style log-linear weights.
///
/// All parameters are transformed to [0,1] internal space for scale-invariance.
/// Categorical parameters fall back to random sampling.
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
    /// Initial step size (sigma0) in [0,1] space. Default 0.3.
    pub sigma0: f64,
}

impl Default for CmaEsConfig {
    fn default() -> Self {
        Self {
            n_startup_trials: 10,
            sigma0: 0.3,
        }
    }
}

/// Internal state of CMA-ES per parameter.
#[derive(Debug, Clone)]
struct ParamState {
    /// Mean of the search distribution in [0,1] internal space.
    mean: f64,
    /// Standard deviation in [0,1] internal space.
    sigma: f64,
    /// Number of times this parameter's state has been updated.
    n_updates: usize,
}

/// Internal state of CMA-ES.
#[derive(Debug)]
struct CmaEsState {
    rng: StdRng,
    /// Per-parameter state, keyed by parameter name.
    params: std::collections::HashMap<String, ParamState>,
}

impl CmaEsSampler {
    pub fn new(config: CmaEsConfig, seed: u64) -> Self {
        Self {
            config: config.clone(),
            random_sampler: super::RandomSampler::new(seed),
            state: Mutex::new(CmaEsState {
                rng: StdRng::seed_from_u64(seed.wrapping_add(2)),
                params: std::collections::HashMap::new(),
            }),
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        Self::new(CmaEsConfig::default(), seed)
    }

    /// Transform a raw parameter value to [0,1] internal space based on distribution.
    fn to_internal(value: f64, distribution: &Distribution) -> f64 {
        match distribution {
            Distribution::Float(d) => transform_to_internal(value, d),
            Distribution::Int(d) => int_transform_to_internal(value as i64, d),
            Distribution::Categorical(_) => value, // not used
        }
    }

    /// Transform from [0,1] internal space back to raw parameter value.
    fn from_internal(internal: f64, distribution: &Distribution) -> f64 {
        match distribution {
            Distribution::Float(d) => transform_from_internal(internal, d),
            Distribution::Int(d) => int_transform_from_internal(internal, d) as f64,
            Distribution::Categorical(_) => internal, // not used
        }
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
        // Categorical: fall back to random
        if matches!(distribution, Distribution::Categorical(_)) {
            return self
                .random_sampler
                .sample(study, trial, param_name, distribution);
        }

        let completed: Vec<_> = study
            .trials()
            .iter()
            .filter(|t| t.state == TrialState::Complete && t.params.contains_key(param_name))
            .collect();

        if completed.len() < self.config.n_startup_trials {
            return self
                .random_sampler
                .sample(study, trial, param_name, distribution);
        }

        // Single lock scope: update state and sample in one go
        let mut state = self.state.lock().unwrap();

        // Initialize parameter if not seen before
        if !state.params.contains_key(param_name) {
            state.params.insert(
                param_name.to_string(),
                ParamState {
                    mean: 0.5,
                    sigma: self.config.sigma0,
                    n_updates: 0,
                },
            );
        }

        // Sort completed trials by objective (best first)
        let mut sorted_trials = completed.clone();
        sorted_trials.sort_by(|a, b| {
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

        // Use the best mu_eff trials
        let n = sorted_trials.len();
        let mu_eff = (n as f64 / 2.0).ceil() as usize;
        let selected = &sorted_trials[..mu_eff.min(n)];

        // CMA-ES log-linear weights
        let weights: Vec<f64> = (0..selected.len())
            .map(|i| ((mu_eff as f64 + 0.5).ln() - (i as f64 + 1.0).ln()).max(0.0))
            .collect();
        let w_sum: f64 = weights.iter().sum();

        if w_sum > 0.0 {
            // Weighted mean in [0,1] internal space
            let new_mean: f64 = selected
                .iter()
                .zip(weights.iter())
                .map(|(t, &w)| {
                    let internal = Self::to_internal(t.params[param_name], distribution);
                    w * internal
                })
                .sum::<f64>()
                / w_sum;

            // Weighted variance in [0,1] internal space
            let variance: f64 = selected
                .iter()
                .zip(weights.iter())
                .map(|(t, &w)| {
                    let internal = Self::to_internal(t.params[param_name], distribution);
                    let diff = internal - new_mean;
                    w * diff * diff
                })
                .sum::<f64>()
                / w_sum;

            let new_sigma = variance.sqrt().max(1e-6);

            let ps = state.params.get_mut(param_name).unwrap();
            // Exponential moving average to blend with prior state
            let alpha = 0.5_f64;
            ps.mean = alpha * new_mean + (1.0 - alpha) * ps.mean;
            ps.sigma = alpha * new_sigma + (1.0 - alpha) * ps.sigma;
            ps.n_updates += 1;
        }

        let ps = &state.params[param_name];
        let mean = ps.mean;
        let sigma = ps.sigma;

        // Sample from N(mean, sigma^2) in [0,1] space, clamp to [0,1]
        let normal: f64 = state.rng.sample(rand_distr::StandardNormal);
        let internal_sample = (mean + sigma * normal).clamp(0.0, 1.0);

        Self::from_internal(internal_sample, distribution)
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
        let study = Study::new_default();
        let trial = Trial::new(0);

        let v = sampler.sample(&study, &trial, "x", &dist);
        assert!(v >= 0.0 && v <= 1.0);
    }

    #[test]
    fn test_cmaes_converges_toward_optimum() {
        let sampler = CmaEsSampler::new(
            CmaEsConfig {
                n_startup_trials: 5,
                sigma0: 0.3,
            },
            42,
        );
        let dist = Distribution::Float(crate::distributions::FloatDistribution::new(
            -10.0, 10.0, false, None,
        ));
        let mut study = Study::new_default();

        for _ in 0..50 {
            let trial = Trial::new(study.trials().len());
            let x = sampler.sample(&study, &trial, "x", &dist);
            let obj = (x - 3.0).powi(2);
            let mut params = HashMap::new();
            params.insert("x".to_string(), x);
            study.add_completed_trial(params, obj);
        }

        let best = study.best_value().unwrap();
        assert!(best < 5.0, "CMA-ES should converge, best={best}");
    }

    #[test]
    fn test_cmaes_log_scale_float() {
        let sampler = CmaEsSampler::with_seed(42);
        let dist = Distribution::Float(crate::distributions::FloatDistribution::new(
            1e-5, 1.0, true, None,
        ));
        let mut study = Study::new_default();

        for i in 0..15 {
            let v = 1e-5 * (1e5_f64).powf(i as f64 / 14.0);
            let mut params = HashMap::new();
            params.insert("lr".to_string(), v);
            study.add_completed_trial(params, (v - 0.01).powi(2));
        }

        let trial = Trial::new(15);
        let v = sampler.sample(&study, &trial, "lr", &dist);
        assert!(v >= 1e-5 && v <= 1.0, "CMA-ES log-scale suggested {v} outside [1e-5, 1.0]");
    }
}
