//! Brute force sampler — systematically enumerates all parameter combinations.
//!
//! Only works with discrete distributions (IntDistribution with step,
//! FloatDistribution with step, CategoricalDistribution).

use crate::distributions::Distribution;
use crate::study::Study;
use crate::trial::{Trial, TrialState};

use super::Sampler;

/// A sampler that systematically tries every combination of parameter values.
///
/// Unlike `GridSampler`, this sampler automatically generates the grid from
/// distributions rather than requiring explicit value lists. It works with:
/// - `IntDistribution`: enumerates `low, low+step, ..., high`
/// - `FloatDistribution` with `step`: enumerates `low, low+step, ..., high`
/// - `CategoricalDistribution`: enumerates `0, 1, ..., n_choices-1`
/// - `FloatDistribution` without `step`: falls back to random sampling
pub struct BruteForceSampler {
    /// Fallback seed for continuous distributions that can't be enumerated.
    seed: u64,
}

impl BruteForceSampler {
    /// Create a new brute force sampler.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Generate all possible values for a distribution.
    fn enumerate_values(distribution: &Distribution) -> Option<Vec<f64>> {
        match distribution {
            Distribution::Float(d) => {
                let step = d.step?;
                let n_steps = ((d.high - d.low) / step).round() as usize;
                let values: Vec<f64> = (0..=n_steps).map(|i| d.low + i as f64 * step).collect();
                Some(values)
            }
            Distribution::Int(d) => {
                let step = d.step.unwrap_or(1);
                let mut values = Vec::new();
                let mut v = d.low;
                while v <= d.high {
                    values.push(v as f64);
                    v += step;
                }
                Some(values)
            }
            Distribution::Categorical(d) => {
                let values: Vec<f64> = (0..d.choices.len()).map(|i| i as f64).collect();
                Some(values)
            }
        }
    }
}

impl Sampler for BruteForceSampler {
    fn sample(
        &self,
        study: &Study,
        _trial: &Trial,
        param_name: &str,
        distribution: &Distribution,
    ) -> f64 {
        let grid_values = match Self::enumerate_values(distribution) {
            Some(v) if !v.is_empty() => v,
            _ => {
                // Can't enumerate: fall back to random sampling
                let fallback = super::RandomSampler::new(
                    self.seed.wrapping_add(study.trials().len() as u64),
                );
                return fallback.sample(study, _trial, param_name, distribution);
            }
        };

        // Count how many completed/running trials have used each value
        let mut used_counts = vec![0usize; grid_values.len()];
        for t in study.trials() {
            if t.state == TrialState::Complete || t.state == TrialState::Running {
                if let Some(&v) = t.params.get(param_name) {
                    for (i, &gv) in grid_values.iter().enumerate() {
                        if (v - gv).abs() < 1e-12 {
                            used_counts[i] += 1;
                            break;
                        }
                    }
                }
            }
        }

        // Pick the least-used value (first one if tied)
        let min_count = *used_counts.iter().min().unwrap_or(&0);
        let idx = used_counts
            .iter()
            .position(|&c| c == min_count)
            .unwrap_or(0);

        grid_values[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::{CategoricalDistribution, FloatDistribution, IntDistribution};
    use crate::study::{Direction, Study};
    use crate::trial::FrozenTrial;

    #[test]
    fn test_enumerate_int() {
        let dist = Distribution::Int(IntDistribution::new(0, 4, false, Some(2)));
        let values = BruteForceSampler::enumerate_values(&dist).unwrap();
        assert_eq!(values, vec![0.0, 2.0, 4.0]);
    }

    #[test]
    fn test_enumerate_float_step() {
        let dist = Distribution::Float(FloatDistribution::new(0.0, 1.0, false, Some(0.25)));
        let values = BruteForceSampler::enumerate_values(&dist).unwrap();
        assert_eq!(values, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn test_enumerate_categorical() {
        let dist = Distribution::Categorical(CategoricalDistribution::new(vec![
            "a".into(),
            "b".into(),
            "c".into(),
        ]));
        let values = BruteForceSampler::enumerate_values(&dist).unwrap();
        assert_eq!(values, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_enumerate_continuous_returns_none() {
        let dist = Distribution::Float(FloatDistribution::new(0.0, 1.0, false, None));
        assert!(BruteForceSampler::enumerate_values(&dist).is_none());
    }

    #[test]
    fn test_brute_force_cycles() {
        let sampler = BruteForceSampler::new(42);
        let mut study = Study::new(Direction::Minimize);
        let dist = Distribution::Int(IntDistribution::new(0, 2, false, None));

        // Should return 0, 1, 2 in order
        let trial = Trial::new(0);
        assert_eq!(sampler.sample(&study, &trial, "x", &dist), 0.0);

        let mut ft = FrozenTrial::new(0);
        ft.state = TrialState::Complete;
        ft.value = Some(1.0);
        ft.params.insert("x".to_string(), 0.0);
        study.add_trial(ft);

        let trial = Trial::new(1);
        assert_eq!(sampler.sample(&study, &trial, "x", &dist), 1.0);

        let mut ft = FrozenTrial::new(1);
        ft.state = TrialState::Complete;
        ft.value = Some(1.0);
        ft.params.insert("x".to_string(), 1.0);
        study.add_trial(ft);

        let trial = Trial::new(2);
        assert_eq!(sampler.sample(&study, &trial, "x", &dist), 2.0);
    }
}
