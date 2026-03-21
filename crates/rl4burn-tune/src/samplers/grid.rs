//! Grid sampler — exhaustive enumeration of parameter combinations.

use std::collections::HashMap;

use crate::distributions::Distribution;
use crate::study::Study;
use crate::trial::{Trial, TrialState};

use super::Sampler;

/// Grid sampler that enumerates all parameter combinations.
///
/// Specify the search space as a map from parameter names to lists of values.
/// The sampler cycles through all combinations, picking the least-used value
/// for each parameter.
pub struct GridSampler {
    /// Map from parameter name to the list of values to try.
    values: HashMap<String, Vec<f64>>,
}

impl GridSampler {
    /// Create a new grid sampler with explicit parameter values.
    ///
    /// `values` maps parameter names to the list of values to enumerate.
    pub fn new(values: HashMap<String, Vec<f64>>) -> Self {
        Self { values }
    }
}

impl Sampler for GridSampler {
    fn sample(
        &self,
        study: &Study,
        _trial: &Trial,
        param_name: &str,
        _distribution: &Distribution,
    ) -> f64 {
        let grid_values = match self.values.get(param_name) {
            Some(v) if !v.is_empty() => v,
            _ => {
                // No grid values for this parameter; return low bound as fallback
                return match _distribution {
                    Distribution::Float(d) => d.low,
                    Distribution::Int(d) => d.low as f64,
                    Distribution::Categorical(_) => 0.0,
                };
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
    use crate::study::{Direction, Study};
    use std::collections::HashMap;

    #[test]
    fn test_grid_sampler_cycles() {
        let mut values = HashMap::new();
        values.insert("x".to_string(), vec![1.0, 2.0, 3.0]);
        let sampler = GridSampler::new(values);

        let mut study = Study::new(Direction::Minimize);
        let dist = Distribution::Float(crate::distributions::FloatDistribution::new(
            0.0, 10.0, false, None,
        ));

        // First sample should be 1.0 (least used = all at 0)
        let trial = Trial::new(0);
        let v = sampler.sample(&study, &trial, "x", &dist);
        assert_eq!(v, 1.0);

        // Add trial with x=1.0
        let mut ft = crate::trial::FrozenTrial::new(0);
        ft.state = TrialState::Complete;
        ft.value = Some(1.0);
        ft.params.insert("x".to_string(), 1.0);
        study.add_trial(ft);

        // Next should be 2.0
        let trial = Trial::new(1);
        let v = sampler.sample(&study, &trial, "x", &dist);
        assert_eq!(v, 2.0);
    }
}
