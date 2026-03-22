//! Partial fixed sampler — wraps another sampler and fixes a subset of parameters.

use std::collections::HashMap;

use crate::distributions::Distribution;
use crate::study::Study;
use crate::trial::Trial;

use super::Sampler;

/// A sampler that fixes some parameters to specified values and delegates the rest.
///
/// This is useful for fixing certain hyperparameters while exploring others,
/// e.g., when doing a sensitivity analysis on a subset of parameters.
pub struct PartialFixedSampler {
    /// The inner sampler used for non-fixed parameters.
    inner: Box<dyn Sampler>,
    /// Fixed parameter values (param_name -> value).
    fixed_params: HashMap<String, f64>,
}

impl PartialFixedSampler {
    /// Create a new partial fixed sampler.
    ///
    /// `inner` is the sampler used for parameters not in `fixed_params`.
    /// `fixed_params` maps parameter names to their fixed values.
    pub fn new(inner: Box<dyn Sampler>, fixed_params: HashMap<String, f64>) -> Self {
        Self {
            inner,
            fixed_params,
        }
    }
}

impl Sampler for PartialFixedSampler {
    fn sample(
        &self,
        study: &Study,
        trial: &Trial,
        param_name: &str,
        distribution: &Distribution,
    ) -> f64 {
        if let Some(&fixed_value) = self.fixed_params.get(param_name) {
            return fixed_value;
        }
        self.inner.sample(study, trial, param_name, distribution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::samplers::RandomSampler;
    use crate::study::{Direction, Study};

    #[test]
    fn test_partial_fixed_returns_fixed_values() {
        let mut fixed = HashMap::new();
        fixed.insert("x".to_string(), 3.14);
        fixed.insert("y".to_string(), 2.71);

        let inner = Box::new(RandomSampler::new(42));
        let sampler = PartialFixedSampler::new(inner, fixed);

        let study = Study::new(Direction::Minimize);
        let trial = Trial::new(0);
        let dist = Distribution::Float(crate::distributions::FloatDistribution::new(
            0.0, 10.0, false, None,
        ));

        // Fixed params should return exact values
        assert_eq!(sampler.sample(&study, &trial, "x", &dist), 3.14);
        assert_eq!(sampler.sample(&study, &trial, "y", &dist), 2.71);

        // Non-fixed params should delegate to inner sampler
        let z = sampler.sample(&study, &trial, "z", &dist);
        assert!(z >= 0.0 && z <= 10.0);
    }

    #[test]
    fn test_partial_fixed_empty_fixes() {
        let fixed = HashMap::new();
        let inner = Box::new(RandomSampler::new(42));
        let sampler = PartialFixedSampler::new(inner, fixed);

        let study = Study::new(Direction::Minimize);
        let trial = Trial::new(0);
        let dist = Distribution::Float(crate::distributions::FloatDistribution::new(
            0.0, 10.0, false, None,
        ));

        // All params should delegate
        let x = sampler.sample(&study, &trial, "x", &dist);
        assert!(x >= 0.0 && x <= 10.0);
    }
}
