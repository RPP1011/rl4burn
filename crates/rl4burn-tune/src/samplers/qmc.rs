//! Quasi-Monte Carlo (QMC) sampler using Halton sequences.
//!
//! Generates low-discrepancy quasi-random points that fill the search space
//! more uniformly than pure random sampling. Well-suited for moderate-budget
//! optimization where space coverage matters.

use std::sync::Mutex;

use crate::distributions::{
    int_transform_from_internal, transform_from_internal, Distribution,
};
use crate::study::Study;
use crate::trial::Trial;

use super::Sampler;

/// Configuration for the QMC sampler.
#[derive(Debug, Clone)]
pub struct QmcConfig {
    /// Skip the first N points of the Halton sequence (scrambling).
    pub skip: usize,
}

impl Default for QmcConfig {
    fn default() -> Self {
        Self { skip: 0 }
    }
}

/// A sampler using Halton quasi-random sequences for low-discrepancy sampling.
///
/// Uses a different prime base for each parameter dimension, producing
/// points that are more uniformly distributed than pure random sampling.
pub struct QmcSampler {
    config: QmcConfig,
    random_sampler: super::RandomSampler,
    counter: Mutex<usize>,
}

/// First 20 primes for Halton sequence bases.
const PRIMES: [usize; 20] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71];

/// Compute the n-th element of a Halton sequence with given base.
fn halton(n: usize, base: usize) -> f64 {
    let mut result = 0.0;
    let mut f = 1.0 / base as f64;
    let mut i = n;
    while i > 0 {
        result += (i % base) as f64 * f;
        i /= base;
        f /= base as f64;
    }
    result
}

impl QmcSampler {
    pub fn new(config: QmcConfig, seed: u64) -> Self {
        Self {
            config,
            random_sampler: super::RandomSampler::new(seed),
            counter: Mutex::new(0),
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        Self::new(QmcConfig::default(), seed)
    }
}

impl Sampler for QmcSampler {
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

        let mut counter = self.counter.lock().unwrap();
        let n = *counter + self.config.skip + 1; // +1 to skip Halton(0)=0
        *counter += 1;

        // Use parameter name hash to select prime base for dimension independence
        let dim_hash = param_name
            .bytes()
            .enumerate()
            .fold(0usize, |acc, (i, b)| acc.wrapping_add(b as usize * (i + 1)));
        let base = PRIMES[dim_hash % PRIMES.len()];

        let u = halton(n, base); // u in [0, 1)

        match distribution {
            Distribution::Float(d) => transform_from_internal(u, d),
            Distribution::Int(d) => int_transform_from_internal(u, d) as f64,
            Distribution::Categorical(_) => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::FloatDistribution;

    #[test]
    fn test_halton_base2() {
        // Halton(1,2) = 0.5, Halton(2,2) = 0.25, Halton(3,2) = 0.75, etc.
        assert!((halton(1, 2) - 0.5).abs() < 1e-10);
        assert!((halton(2, 2) - 0.25).abs() < 1e-10);
        assert!((halton(3, 2) - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_halton_base3() {
        assert!((halton(1, 3) - 1.0 / 3.0).abs() < 1e-10);
        assert!((halton(2, 3) - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_qmc_in_bounds() {
        let sampler = QmcSampler::with_seed(42);
        let dist = Distribution::Float(FloatDistribution::new(0.0, 10.0, false, None));
        let study = Study::new_default();

        for i in 0..100 {
            let v = sampler.sample(&study, &Trial::new(i), "x", &dist);
            assert!(
                (0.0..=10.0).contains(&v),
                "QMC value {v} out of [0, 10]"
            );
        }
    }

    #[test]
    fn test_qmc_coverage() {
        // QMC should cover the space more uniformly than random
        let sampler = QmcSampler::with_seed(42);
        let dist = Distribution::Float(FloatDistribution::new(0.0, 1.0, false, None));
        let study = Study::new_default();

        let values: Vec<f64> = (0..20)
            .map(|i| sampler.sample(&study, &Trial::new(i), "x", &dist))
            .collect();

        // Check that we have values spread across the range
        let has_low = values.iter().any(|&v| v < 0.3);
        let has_mid = values.iter().any(|&v| v > 0.3 && v < 0.7);
        let has_high = values.iter().any(|&v| v > 0.7);
        assert!(has_low, "QMC should have values in low range");
        assert!(has_mid, "QMC should have values in mid range");
        assert!(has_high, "QMC should have values in high range");
    }

    #[test]
    fn test_qmc_deterministic() {
        let dist = Distribution::Float(FloatDistribution::new(0.0, 1.0, false, None));
        let study = Study::new_default();

        let s1 = QmcSampler::with_seed(42);
        let s2 = QmcSampler::with_seed(42);

        for i in 0..10 {
            let v1 = s1.sample(&study, &Trial::new(i), "x", &dist);
            let v2 = s2.sample(&study, &Trial::new(i), "x", &dist);
            assert_eq!(v1, v2, "QMC should be deterministic");
        }
    }
}
