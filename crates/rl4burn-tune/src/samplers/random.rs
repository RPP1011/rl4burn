use rand::prelude::*;
use std::sync::Mutex;

use crate::distributions::Distribution;
use crate::study::Study;
use crate::trial::Trial;

use super::Sampler;

/// Uniform random sampler.
pub struct RandomSampler {
    rng: Mutex<StdRng>,
}

impl RandomSampler {
    /// Create a new random sampler with a given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        }
    }

    fn sample_float(&self, low: f64, high: f64, log: bool, step: Option<f64>) -> f64 {
        let mut rng = self.rng.lock().unwrap();
        let value = if log {
            let log_low = low.ln();
            let log_high = high.ln();
            let u: f64 = rng.random();
            (log_low + u * (log_high - log_low)).exp()
        } else {
            let u: f64 = rng.random();
            low + u * (high - low)
        };

        if let Some(s) = step {
            let shifted = value - low;
            let quantized = (shifted / s).round() * s + low;
            quantized.clamp(low, high)
        } else {
            value
        }
    }

    fn sample_int(&self, low: i64, high: i64, log: bool, step: Option<i64>) -> f64 {
        let mut rng = self.rng.lock().unwrap();
        let step = step.unwrap_or(1);

        let value = if log {
            let log_low = (low as f64).ln();
            let log_high = (high as f64).ln();
            let u: f64 = rng.random();
            (log_low + u * (log_high - log_low)).exp()
        } else {
            let u: f64 = rng.random();
            low as f64 + u * (high - low) as f64
        };

        let shifted = value - low as f64;
        let quantized = (shifted / step as f64).round() * step as f64 + low as f64;
        (quantized.round() as i64).clamp(low, high) as f64
    }

    fn sample_categorical(&self, n_choices: usize) -> f64 {
        let mut rng = self.rng.lock().unwrap();
        let u: f64 = rng.random();
        let index = (u * n_choices as f64).min(n_choices as f64 - 1.0) as usize;
        index as f64
    }
}

impl Sampler for RandomSampler {
    fn sample(
        &self,
        _study: &Study,
        _trial: &Trial,
        _param_name: &str,
        distribution: &Distribution,
    ) -> f64 {
        match distribution {
            Distribution::Float(d) => self.sample_float(d.low, d.high, d.log, d.step),
            Distribution::Int(d) => self.sample_int(d.low, d.high, d.log, d.step),
            Distribution::Categorical(d) => self.sample_categorical(d.choices.len()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::{CategoricalDistribution, FloatDistribution, IntDistribution};

    #[test]
    fn test_random_float_in_bounds() {
        let sampler = RandomSampler::new(42);
        let dist = Distribution::Float(FloatDistribution::new(0.0, 1.0, false, None));
        let study = Study::new_default();
        let trial = Trial::new(0);
        for _ in 0..1000 {
            let v = sampler.sample(&study, &trial, "x", &dist);
            assert!(v >= 0.0 && v <= 1.0, "out of bounds: {v}");
        }
    }

    #[test]
    fn test_random_float_log_in_bounds() {
        let sampler = RandomSampler::new(42);
        let dist = Distribution::Float(FloatDistribution::new(1e-5, 1.0, true, None));
        let study = Study::new_default();
        let trial = Trial::new(0);
        for _ in 0..1000 {
            let v = sampler.sample(&study, &trial, "x", &dist);
            assert!(v >= 1e-5 && v <= 1.0, "out of bounds: {v}");
        }
    }

    #[test]
    fn test_random_int_in_bounds() {
        let sampler = RandomSampler::new(42);
        let dist = Distribution::Int(IntDistribution::new(0, 10, false, None));
        let study = Study::new_default();
        let trial = Trial::new(0);
        for _ in 0..1000 {
            let v = sampler.sample(&study, &trial, "x", &dist) as i64;
            assert!(v >= 0 && v <= 10, "out of bounds: {v}");
        }
    }

    #[test]
    fn test_random_categorical_valid_index() {
        let sampler = RandomSampler::new(42);
        let dist = Distribution::Categorical(CategoricalDistribution::new(vec![
            "a".into(),
            "b".into(),
            "c".into(),
        ]));
        let study = Study::new_default();
        let trial = Trial::new(0);
        for _ in 0..1000 {
            let v = sampler.sample(&study, &trial, "x", &dist) as usize;
            assert!(v < 3, "invalid index: {v}");
        }
    }
}
