//! NSGA-II (Non-dominated Sorting Genetic Algorithm II) sampler.
//!
//! A multi-objective evolutionary sampler that uses non-dominated sorting
//! and crowding distance for selection, with uniform crossover and
//! random mutation.

use std::sync::Mutex;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::distributions::Distribution;
use crate::multi_objective::{crowding_distance, non_dominated_sort};
use crate::samplers::random::RandomSampler;
use crate::samplers::Sampler;
use crate::study::Study;
use crate::trial::{FrozenTrial, Trial, TrialState};

/// Configuration for the NSGA-II sampler.
#[derive(Debug, Clone)]
pub struct NsgaIIConfig {
    /// Number of individuals in each generation.
    pub population_size: usize,
    /// Probability of crossing over each parameter (uniform crossover).
    pub crossover_prob: f64,
    /// Probability of mutating each parameter.
    pub mutation_prob: f64,
    /// Standard deviation for Gaussian mutation (in normalized [0,1] space).
    pub mutation_sigma: f64,
    /// Number of random startup trials before evolutionary sampling begins.
    pub n_startup_trials: usize,
}

impl Default for NsgaIIConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            crossover_prob: 0.5,
            mutation_prob: 0.2,
            mutation_sigma: 0.1,
            n_startup_trials: 50,
        }
    }
}

/// NSGA-II sampler for multi-objective optimization.
pub struct NsgaIISampler {
    config: NsgaIIConfig,
    random_sampler: RandomSampler,
    rng: Mutex<StdRng>,
}

impl NsgaIISampler {
    /// Create a new NSGA-II sampler with the given configuration and seed.
    pub fn new(config: NsgaIIConfig, seed: u64) -> Self {
        Self {
            config,
            random_sampler: RandomSampler::new(seed),
            rng: Mutex::new(StdRng::seed_from_u64(seed.wrapping_add(1))),
        }
    }

    /// Create a new NSGA-II sampler with default configuration and given seed.
    pub fn with_seed(seed: u64) -> Self {
        Self::new(NsgaIIConfig::default(), seed)
    }

    /// Select a parent trial using NSGA-II tournament selection.
    ///
    /// Picks two random completed trials and returns the one with better
    /// (lower) non-dominated rank, or higher crowding distance if tied.
    fn select_parent<'a>(
        &self,
        completed: &[&'a FrozenTrial],
        ranks: &[usize],
        distances: &[f64],
        rng: &mut StdRng,
    ) -> &'a FrozenTrial {
        let i = rng.random_range(0..completed.len());
        let j = rng.random_range(0..completed.len());

        // Prefer lower rank, then higher crowding distance
        if ranks[i] < ranks[j] {
            completed[i]
        } else if ranks[j] < ranks[i] {
            completed[j]
        } else if distances[i] >= distances[j] {
            completed[i]
        } else {
            completed[j]
        }
    }
}

impl Sampler for NsgaIISampler {
    fn sample(
        &self,
        study: &Study,
        _trial: &Trial,
        param_name: &str,
        distribution: &Distribution,
    ) -> f64 {
        let completed: Vec<&FrozenTrial> = study
            .trials()
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .collect();

        // Use random sampling during startup
        if completed.len() < self.config.n_startup_trials {
            return self
                .random_sampler
                .sample(study, _trial, param_name, distribution);
        }

        let mut rng = self.rng.lock().unwrap();

        // Compute objective values for non-dominated sorting
        let directions = study.directions();
        let values: Vec<Vec<f64>> = completed
            .iter()
            .map(|t| {
                t.values
                    .clone()
                    .unwrap_or_else(|| t.value.map(|v| vec![v]).unwrap_or_default())
            })
            .collect();

        // Non-dominated sorting
        let fronts = non_dominated_sort(&values, directions);

        // Assign ranks
        let mut ranks = vec![0usize; completed.len()];
        for (rank, front) in fronts.iter().enumerate() {
            for &idx in front {
                ranks[idx] = rank;
            }
        }

        // Compute crowding distances within each front
        let mut distances = vec![0.0f64; completed.len()];
        for front in &fronts {
            let front_values: Vec<Vec<f64>> =
                front.iter().map(|&i| values[i].clone()).collect();
            let front_distances = crowding_distance(&front_values, directions);
            for (fi, &idx) in front.iter().enumerate() {
                distances[idx] = front_distances[fi];
            }
        }

        // Select two parents via tournament selection
        let parent1 = self.select_parent(&completed, &ranks, &distances, &mut rng);
        let parent2 = self.select_parent(&completed, &ranks, &distances, &mut rng);

        // Get parent values for this parameter
        let p1_val = parent1.params.get(param_name);
        let p2_val = parent2.params.get(param_name);

        match (p1_val, p2_val) {
            (Some(&v1), Some(&v2)) => {
                // Uniform crossover
                let child_val = if rng.random_bool(self.config.crossover_prob) {
                    v2
                } else {
                    v1
                };

                // Mutation
                if rng.random_bool(self.config.mutation_prob) {
                    match distribution {
                        Distribution::Float(d) => {
                            let range = if d.log {
                                d.high.ln() - d.low.ln()
                            } else {
                                d.high - d.low
                            };
                            let noise: f64 =
                                rng.random_range(-1.0..1.0) * self.config.mutation_sigma * range;
                            let mutated = if d.log {
                                (child_val.ln() + noise).exp()
                            } else {
                                child_val + noise
                            };
                            let clamped = mutated.clamp(d.low, d.high);
                            if let Some(step) = d.step {
                                let shifted = clamped - d.low;
                                (shifted / step).round() * step + d.low
                            } else {
                                clamped
                            }
                        }
                        Distribution::Int(d) => {
                            let range = (d.high - d.low) as f64;
                            let noise =
                                rng.random_range(-1.0..1.0) * self.config.mutation_sigma * range;
                            let mutated = (child_val + noise).round();
                            let step = d.step.unwrap_or(1) as f64;
                            let shifted = mutated - d.low as f64;
                            let quantized =
                                (shifted / step).round() * step + d.low as f64;
                            quantized.clamp(d.low as f64, d.high as f64)
                        }
                        Distribution::Categorical(d) => {
                            // Random category mutation
                            rng.random_range(0..d.choices.len()) as f64
                        }
                    }
                } else {
                    child_val
                }
            }
            _ => {
                // Parent doesn't have this parameter; fall back to random
                self.random_sampler
                    .sample(study, _trial, param_name, distribution)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study::Direction;

    #[test]
    fn test_nsga2_startup_random() {
        let sampler = NsgaIISampler::with_seed(42);
        let study = Study::new_multi(vec![Direction::Minimize, Direction::Minimize]);
        let trial = Trial::new(0);
        let dist = Distribution::Float(crate::distributions::FloatDistribution::new(
            0.0, 10.0, false, None,
        ));

        let v = sampler.sample(&study, &trial, "x", &dist);
        assert!((0.0..=10.0).contains(&v));
    }

    #[test]
    fn test_nsga2_after_startup() {
        let config = NsgaIIConfig {
            n_startup_trials: 5,
            population_size: 10,
            ..Default::default()
        };
        let sampler = NsgaIISampler::new(config, 42);
        let mut study =
            Study::new_multi(vec![Direction::Minimize, Direction::Minimize]);

        // Add enough completed trials
        for i in 0..10 {
            let mut trial = FrozenTrial::new(i);
            trial.state = TrialState::Complete;
            trial.value = Some(i as f64);
            trial.values = Some(vec![i as f64, (10 - i) as f64]);
            trial.params.insert("x".to_string(), i as f64);
            trial.params.insert("y".to_string(), (i * 2) as f64);
            study.add_trial(trial);
        }

        let trial = Trial::new(10);
        let dist = Distribution::Float(crate::distributions::FloatDistribution::new(
            0.0, 20.0, false, None,
        ));

        let v = sampler.sample(&study, &trial, "x", &dist);
        assert!((0.0..=20.0).contains(&v));
    }
}
