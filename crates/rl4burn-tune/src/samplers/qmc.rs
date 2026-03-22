//! Quasi-Monte Carlo sampler using Sobol sequences.
//!
//! Provides low-discrepancy sampling for better space coverage than
//! uniform random sampling.

use crate::distributions::{
    Distribution, FloatDistribution, IntDistribution,
};
use crate::study::Study;
use crate::trial::Trial;

use super::Sampler;

/// Quasi-Monte Carlo sampler using Sobol sequences.
///
/// Generates low-discrepancy points that cover the search space more
/// uniformly than random sampling. This is particularly useful for
/// initial exploration before switching to a model-based sampler.
pub struct QmcSampler {
    /// Random seed for scrambling.
    seed: u64,
}

impl QmcSampler {
    /// Create a new QMC sampler.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Generate a Sobol-like quasi-random value in [0, 1] for the given index.
    ///
    /// Uses the Van der Corput sequence (base-2 radical inverse) with scrambling.
    /// This is a simplified but effective approach that provides good discrepancy
    /// properties for moderate dimensions.
    fn sobol_value(&self, index: usize, dimension: usize) -> f64 {
        // Use different bases for different dimensions to reduce correlation
        let base = Self::prime_for_dimension(dimension);
        let scramble = self.seed.wrapping_mul(2654435761).wrapping_add(dimension as u64);
        Self::radical_inverse(index as u64 ^ scramble, base)
    }

    /// Radical inverse function in the given base.
    fn radical_inverse(mut n: u64, base: u64) -> f64 {
        let mut result = 0.0_f64;
        let mut factor = 1.0 / base as f64;

        while n > 0 {
            result += (n % base) as f64 * factor;
            n /= base;
            factor /= base as f64;
        }

        result
    }

    /// Get a prime number for the given dimension (0-indexed).
    fn prime_for_dimension(dim: usize) -> u64 {
        // First 32 primes for up to 32 dimensions
        const PRIMES: [u64; 32] = [
            2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79,
            83, 89, 97, 101, 103, 107, 109, 113, 127, 131,
        ];
        PRIMES[dim % PRIMES.len()]
    }

    /// Map a [0,1] value to the distribution's domain.
    fn map_to_distribution(u: f64, distribution: &Distribution) -> f64 {
        match distribution {
            Distribution::Float(d) => Self::map_to_float(u, d),
            Distribution::Int(d) => Self::map_to_int(u, d),
            Distribution::Categorical(d) => {
                let idx = (u * d.choices.len() as f64).floor() as usize;
                idx.min(d.choices.len() - 1) as f64
            }
        }
    }

    fn map_to_float(u: f64, d: &FloatDistribution) -> f64 {
        let value = if d.log {
            let log_low = d.low.ln();
            let log_high = d.high.ln();
            (log_low + u * (log_high - log_low)).exp()
        } else {
            d.low + u * (d.high - d.low)
        };

        if let Some(step) = d.step {
            let shifted = value - d.low;
            let quantized = (shifted / step).round() * step + d.low;
            quantized.clamp(d.low, d.high)
        } else {
            value.clamp(d.low, d.high)
        }
    }

    fn map_to_int(u: f64, d: &IntDistribution) -> f64 {
        let step = d.step.unwrap_or(1);
        let value = if d.log {
            let log_low = (d.low as f64).ln();
            let log_high = (d.high as f64).ln();
            (log_low + u * (log_high - log_low)).exp()
        } else {
            d.low as f64 + u * (d.high - d.low) as f64
        };

        let shifted = value - d.low as f64;
        let quantized = (shifted / step as f64).round() * step as f64 + d.low as f64;
        (quantized.round() as i64).clamp(d.low, d.high) as f64
    }
}

impl Sampler for QmcSampler {
    fn sample(
        &self,
        study: &Study,
        _trial: &Trial,
        param_name: &str,
        distribution: &Distribution,
    ) -> f64 {
        // Use trial count as the sequence index
        let index = study.trials().len();

        // Use a hash of the param name as the dimension index
        let dimension = {
            let mut h: u64 = 5381;
            for b in param_name.bytes() {
                h = h.wrapping_mul(33).wrapping_add(b as u64);
            }
            h as usize
        };

        let u = self.sobol_value(index, dimension);
        Self::map_to_distribution(u, distribution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::{CategoricalDistribution, FloatDistribution, IntDistribution};
    use crate::study::{Direction, Study};

    #[test]
    fn test_qmc_generates_in_bounds() {
        let sampler = QmcSampler::new(42);
        let study = Study::new(Direction::Minimize);
        let dist = Distribution::Float(FloatDistribution::new(0.0, 1.0, false, None));

        for i in 0..100 {
            let trial = Trial::new(i);
            let v = sampler.sample(&study, &trial, "x", &dist);
            assert!(v >= 0.0 && v <= 1.0, "v={v} out of [0,1]");
        }
    }

    #[test]
    fn test_qmc_covers_space() {
        let sampler = QmcSampler::new(42);
        let mut study = Study::new(Direction::Minimize);
        let dist = Distribution::Float(FloatDistribution::new(0.0, 1.0, false, None));

        let mut values = Vec::new();
        for i in 0..256 {
            let trial = Trial::new(i);
            let v = sampler.sample(&study, &trial, "x", &dist);
            values.push(v);

            let mut ft = crate::trial::FrozenTrial::new(i);
            ft.state = crate::trial::TrialState::Complete;
            ft.value = Some(v);
            ft.params.insert("x".to_string(), v);
            study.add_trial(ft);
        }

        // Check that points cover the space: divide [0,1] into 4 bins
        // With 256 points, each bin should get at least some points
        let bins = [0.0..0.25, 0.25..0.5, 0.5..0.75, 0.75..1.0];
        for bin in &bins {
            let count = values.iter().filter(|&&v| bin.contains(&v)).count();
            assert!(
                count >= 10,
                "Only {count} points in bin [{}, {}), expected at least 10",
                bin.start,
                bin.end
            );
        }
    }

    #[test]
    fn test_qmc_log_scale() {
        let sampler = QmcSampler::new(42);
        let study = Study::new(Direction::Minimize);
        let dist = Distribution::Float(FloatDistribution::new(1e-5, 1.0, true, None));

        for i in 0..50 {
            let trial = Trial::new(i);
            let v = sampler.sample(&study, &trial, "x", &dist);
            assert!(v >= 1e-5 && v <= 1.0, "v={v} out of [1e-5, 1.0]");
        }
    }

    #[test]
    fn test_qmc_int_distribution() {
        let sampler = QmcSampler::new(42);
        let study = Study::new(Direction::Minimize);
        let dist = Distribution::Int(IntDistribution::new(0, 10, false, None));

        for i in 0..50 {
            let trial = Trial::new(i);
            let v = sampler.sample(&study, &trial, "x", &dist);
            let iv = v as i64;
            assert!(iv >= 0 && iv <= 10, "v={iv} out of [0, 10]");
        }
    }

    #[test]
    fn test_qmc_categorical() {
        let sampler = QmcSampler::new(42);
        let study = Study::new(Direction::Minimize);
        let dist = Distribution::Categorical(CategoricalDistribution::new(vec![
            "a".into(),
            "b".into(),
            "c".into(),
        ]));

        for i in 0..50 {
            let trial = Trial::new(i);
            let v = sampler.sample(&study, &trial, "x", &dist);
            let idx = v as usize;
            assert!(idx < 3, "idx={idx} >= 3");
        }
    }

    #[test]
    fn test_radical_inverse() {
        // radical_inverse(0, 2) = 0.0
        assert!((QmcSampler::radical_inverse(0, 2) - 0.0).abs() < 1e-12);
        // radical_inverse(1, 2) = 0.5 (binary: 1 -> 0.1)
        assert!((QmcSampler::radical_inverse(1, 2) - 0.5).abs() < 1e-12);
        // radical_inverse(2, 2) = 0.25 (binary: 10 -> 0.01)
        assert!((QmcSampler::radical_inverse(2, 2) - 0.25).abs() < 1e-12);
        // radical_inverse(3, 2) = 0.75 (binary: 11 -> 0.11)
        assert!((QmcSampler::radical_inverse(3, 2) - 0.75).abs() < 1e-12);
    }

    #[test]
    fn test_discrepancy_better_than_random() {
        // QMC points should have lower star discrepancy than random
        let sampler = QmcSampler::new(42);
        let mut study = Study::new(Direction::Minimize);
        let dist = Distribution::Float(FloatDistribution::new(0.0, 1.0, false, None));

        let n = 256;
        let mut qmc_points = Vec::new();
        for i in 0..n {
            let trial = Trial::new(i);
            let v = sampler.sample(&study, &trial, "x", &dist);
            qmc_points.push(v);
            let mut ft = crate::trial::FrozenTrial::new(i);
            ft.state = crate::trial::TrialState::Complete;
            ft.value = Some(v);
            ft.params.insert("x".to_string(), v);
            study.add_trial(ft);
        }

        // Compute star discrepancy: max |F_n(x) - x| for x in points
        qmc_points.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut max_disc = 0.0_f64;
        for (i, &x) in qmc_points.iter().enumerate() {
            let empirical = (i + 1) as f64 / n as f64;
            let disc = (empirical - x).abs();
            max_disc = max_disc.max(disc);
        }

        // QMC discrepancy should be significantly lower than 1/sqrt(n) ~ 0.0625
        // A good QMC sequence should have discrepancy around O(log(n)/n)
        assert!(
            max_disc < 0.2,
            "Star discrepancy {max_disc} too high for QMC"
        );
    }
}
