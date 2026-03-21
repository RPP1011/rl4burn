//! Brute-force sampler that exhaustively enumerates all parameter combinations.
//!
//! Similar to GridSampler but works with distributions directly by discretizing
//! continuous parameters into a fixed number of points.

use std::sync::Mutex;

use crate::distributions::Distribution;
use crate::study::Study;
use crate::trial::Trial;

use super::Sampler;

/// A sampler that exhaustively evaluates all combinations of discretized parameters.
///
/// Float and int parameters are discretized into `n_points` equally-spaced values.
/// Categorical parameters use all choices. Once all combinations are exhausted,
/// sampling wraps around.
pub struct BruteForceSampler {
    /// Number of discretization points for continuous parameters.
    n_points: usize,
    /// Counter tracking which combination to sample next.
    counter: Mutex<usize>,
}

impl BruteForceSampler {
    /// Create a new brute-force sampler with the given discretization granularity.
    pub fn new(n_points: usize) -> Self {
        assert!(n_points >= 2, "n_points must be at least 2");
        Self {
            n_points,
            counter: Mutex::new(0),
        }
    }
}

impl Default for BruteForceSampler {
    fn default() -> Self {
        Self::new(10)
    }
}

impl Sampler for BruteForceSampler {
    fn sample(
        &self,
        _study: &Study,
        _trial: &Trial,
        _param_name: &str,
        distribution: &Distribution,
    ) -> f64 {
        let mut counter = self.counter.lock().unwrap();
        let idx = *counter;

        // Number of grid points for this parameter
        let n = match distribution {
            Distribution::Float(_) | Distribution::Int(_) => self.n_points,
            Distribution::Categorical(d) => d.choices.len(),
        };

        // Use the counter modulo n for this parameter's index
        let point_idx = idx % n;
        *counter += 1;

        match distribution {
            Distribution::Float(d) => {
                if d.log {
                    let log_low = d.low.ln();
                    let log_high = d.high.ln();
                    let frac = point_idx as f64 / (n - 1).max(1) as f64;
                    let value = (log_low + frac * (log_high - log_low)).exp();
                    if let Some(step) = d.step {
                        let shifted = value - d.low;
                        (shifted / step).round() * step + d.low
                    } else {
                        value
                    }
                } else if let Some(step) = d.step {
                    let n_steps = ((d.high - d.low) / step).round() as usize;
                    let actual_idx = point_idx.min(n_steps);
                    d.low + actual_idx as f64 * step
                } else {
                    let frac = point_idx as f64 / (n - 1).max(1) as f64;
                    d.low + frac * (d.high - d.low)
                }
            }
            Distribution::Int(d) => {
                let step = d.step.unwrap_or(1);
                let n_steps = ((d.high - d.low) / step as i64) as usize;
                let actual_n = (n_steps + 1).min(n);
                let step_idx = if actual_n <= 1 {
                    0
                } else {
                    (point_idx * n_steps) / (actual_n - 1).max(1)
                };
                (d.low + step_idx as i64 * step as i64) as f64
            }
            Distribution::Categorical(_) => point_idx as f64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::{CategoricalDistribution, FloatDistribution, IntDistribution};

    #[test]
    fn test_brute_force_float() {
        let sampler = BruteForceSampler::new(5);
        let dist = Distribution::Float(FloatDistribution::new(0.0, 4.0, false, None));
        let study = Study::new_default();

        let values: Vec<f64> = (0..5)
            .map(|i| sampler.sample(&study, &Trial::new(i), "x", &dist))
            .collect();

        // Should produce 0.0, 1.0, 2.0, 3.0, 4.0
        assert_eq!(values, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_brute_force_categorical() {
        let sampler = BruteForceSampler::new(10);
        let dist = Distribution::Categorical(CategoricalDistribution::new(vec![
            "a".into(),
            "b".into(),
            "c".into(),
        ]));
        let study = Study::new_default();

        let values: Vec<f64> = (0..6)
            .map(|i| sampler.sample(&study, &Trial::new(i), "cat", &dist))
            .collect();

        // Should cycle through 0, 1, 2, 0, 1, 2
        assert_eq!(values, vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_brute_force_int() {
        let sampler = BruteForceSampler::new(3);
        let dist = Distribution::Int(IntDistribution::new(0, 10, false, None));
        let study = Study::new_default();

        let values: Vec<f64> = (0..3)
            .map(|i| sampler.sample(&study, &Trial::new(i), "n", &dist))
            .collect();

        // Should produce 0, 5, 10
        assert_eq!(values, vec![0.0, 5.0, 10.0]);
    }
}
