//! NSGA-II (Non-dominated Sorting Genetic Algorithm II) sampler.
//!
//! A multi-objective evolutionary sampler that uses non-dominated sorting
//! and crowding distance for selection, with SBX crossover and polynomial
//! mutation (matching Optuna's canonical NSGA-II implementation).

use std::sync::Mutex;

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::distributions::Distribution;
use crate::multi_objective::{crowding_distance, non_dominated_sort, non_dominated_sort_constrained};
use crate::samplers::random::RandomSampler;
use crate::samplers::Sampler;
use crate::study::Study;
use crate::trial::{FrozenTrial, Trial, TrialState};

/// Crossover operator type for NSGA-II.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CrossoverType {
    /// Simple uniform crossover (swap parameters with probability).
    Uniform,
    /// Simulated Binary Crossover — the canonical NSGA-II operator.
    SBX,
    /// Simplex Crossover (SPX) — generates offspring inside the simplex
    /// formed by the parents, expanded by a factor epsilon.
    SPX,
    /// Unimodal Normal Distribution Crossover (UNDX) — generates offspring
    /// along the line between parents with Gaussian perturbation.
    UNDX,
    /// Velocity-based SBX — SBX variant that uses the improvement direction.
    VSBX,
}

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
    /// Used only with Uniform crossover; SBX uses polynomial mutation.
    pub mutation_sigma: f64,
    /// Number of random startup trials before evolutionary sampling begins.
    pub n_startup_trials: usize,
    /// Crossover operator type.
    pub crossover_type: CrossoverType,
    /// Distribution index for SBX crossover. Higher values produce children
    /// closer to parents. Default 20 (matches canonical NSGA-II).
    pub eta_c: f64,
    /// Distribution index for polynomial mutation. Higher values produce
    /// smaller perturbations. Default 20 (matches canonical NSGA-II).
    pub eta_m: f64,
}

impl Default for NsgaIIConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            crossover_prob: 0.9,
            mutation_prob: 0.2,
            mutation_sigma: 0.1,
            n_startup_trials: 50,
            crossover_type: CrossoverType::SBX,
            eta_c: 20.0,
            eta_m: 20.0,
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

    /// Simplex Crossover (SPX) for a single parameter.
    /// Generates a child near the midpoint of parents, expanded by epsilon.
    fn spx_crossover(
        p1: f64,
        p2: f64,
        low: f64,
        high: f64,
        epsilon: f64,
        rng: &mut StdRng,
    ) -> f64 {
        let midpoint = (p1 + p2) / 2.0;
        let diff = (p2 - p1).abs();
        // Expand around midpoint by epsilon factor
        let r: f64 = rng.random();
        let child = midpoint + epsilon * (r - 0.5) * diff;
        child.clamp(low, high)
    }

    /// Unimodal Normal Distribution Crossover (UNDX) for a single parameter.
    /// Generates offspring along the line between parents with Gaussian noise.
    fn undx_crossover(
        p1: f64,
        p2: f64,
        low: f64,
        high: f64,
        sigma_xi: f64,
        sigma_eta: f64,
        rng: &mut StdRng,
    ) -> f64 {
        let midpoint = (p1 + p2) / 2.0;
        let diff = (p2 - p1).abs().max(1e-14);
        // Primary component along parent direction
        let xi: f64 = Self::rand_normal(rng) * sigma_xi;
        // Perpendicular perturbation
        let eta: f64 = Self::rand_normal(rng) * sigma_eta;
        let child = midpoint + xi * diff + eta * diff;
        child.clamp(low, high)
    }

    /// Velocity-based SBX (VSBX) for a single parameter.
    /// Like SBX but biased in the improvement direction.
    fn vsbx_crossover(
        p1: f64,
        p2: f64,
        low: f64,
        high: f64,
        eta_c: f64,
        rng: &mut StdRng,
    ) -> f64 {
        // VSBX applies SBX with a velocity bias toward the better parent.
        // For simplicity we use SBX with a shifted center.
        if (p1 - p2).abs() < 1e-14 {
            return p1;
        }
        // Apply standard SBX as the base
        let child = Self::sbx_crossover(p1, p2, low, high, eta_c, rng);
        // Add a small bias toward the midpoint (velocity term)
        let mid = (p1 + p2) / 2.0;
        let biased = child * 0.8 + mid * 0.2;
        biased.clamp(low, high)
    }

    /// Generate a standard normal random variate (Box-Muller).
    fn rand_normal(rng: &mut StdRng) -> f64 {
        let u1: f64 = rng.random::<f64>().max(1e-15);
        let u2: f64 = rng.random();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Simulated Binary Crossover (SBX) for a single parameter.
    /// Returns the child value.
    fn sbx_crossover(
        p1: f64,
        p2: f64,
        low: f64,
        high: f64,
        eta_c: f64,
        rng: &mut StdRng,
    ) -> f64 {
        if (p1 - p2).abs() < 1e-14 {
            return p1;
        }

        let (y1, y2) = if p1 < p2 { (p1, p2) } else { (p2, p1) };

        let u: f64 = rng.random();

        // Bounded SBX
        let diff = (y2 - y1).max(1e-14);
        let beta1 = 1.0 + 2.0 * (y1 - low) / diff;
        let beta2 = 1.0 + 2.0 * (high - y2) / diff;

        let alpha1 = 2.0 - beta1.powf(-(eta_c + 1.0));
        let alpha2 = 2.0 - beta2.powf(-(eta_c + 1.0));

        let betaq1 = if u <= 1.0 / alpha1 {
            (u * alpha1).powf(1.0 / (eta_c + 1.0))
        } else {
            (1.0 / (2.0 - u * alpha1)).powf(1.0 / (eta_c + 1.0))
        };

        let betaq2 = if u <= 1.0 / alpha2 {
            (u * alpha2).powf(1.0 / (eta_c + 1.0))
        } else {
            (1.0 / (2.0 - u * alpha2)).powf(1.0 / (eta_c + 1.0))
        };

        let c1 = 0.5 * ((y1 + y2) - betaq1 * diff);
        let c2 = 0.5 * ((y1 + y2) + betaq2 * diff);

        // Randomly pick one child
        let child = if rng.random_bool(0.5) { c1 } else { c2 };
        child.clamp(low, high)
    }

    /// Polynomial mutation for a single parameter.
    fn polynomial_mutation(
        value: f64,
        low: f64,
        high: f64,
        eta_m: f64,
        rng: &mut StdRng,
    ) -> f64 {
        let u: f64 = rng.random();
        let range = high - low;
        if range < 1e-14 {
            return value;
        }

        let delta1 = (value - low) / range;
        let delta2 = (high - value) / range;

        let deltaq = if u < 0.5 {
            let xy = 1.0 - delta1;
            let val = 2.0 * u + (1.0 - 2.0 * u) * xy.powf(eta_m + 1.0);
            val.powf(1.0 / (eta_m + 1.0)) - 1.0
        } else {
            let xy = 1.0 - delta2;
            let val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * xy.powf(eta_m + 1.0);
            1.0 - val.powf(1.0 / (eta_m + 1.0))
        };

        let mutated = value + deltaq * range;
        mutated.clamp(low, high)
    }

    /// Get the elite population: top `population_size` individuals by rank + crowding.
    fn select_elite<'a>(
        &self,
        completed: &[&'a FrozenTrial],
        ranks: &[usize],
        distances: &[f64],
    ) -> Vec<usize> {
        let n = completed.len();
        let mut indices: Vec<usize> = (0..n).collect();

        // Sort by rank (ascending), then crowding distance (descending)
        indices.sort_by(|&a, &b| {
            ranks[a]
                .cmp(&ranks[b])
                .then_with(|| {
                    distances[b]
                        .partial_cmp(&distances[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        indices.truncate(self.config.population_size);
        indices
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

        // Non-dominated sorting (constraint-aware if any trial has constraints)
        let has_constraints = completed.iter().any(|t| t.constraint_values.is_some());
        let fronts = if has_constraints {
            let violations: Vec<(f64, bool)> = completed
                .iter()
                .map(|t| (t.total_violation(), t.is_feasible()))
                .collect();
            non_dominated_sort_constrained(&values, &violations, directions)
        } else {
            non_dominated_sort(&values, directions)
        };

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

        // Select elite population
        let elite_indices = self.select_elite(&completed, &ranks, &distances);

        // Build elite-only vectors for parent selection
        let elite_completed: Vec<&FrozenTrial> =
            elite_indices.iter().map(|&i| completed[i]).collect();
        let elite_ranks: Vec<usize> = elite_indices.iter().map(|&i| ranks[i]).collect();
        let elite_distances: Vec<f64> = elite_indices.iter().map(|&i| distances[i]).collect();

        // Select two parents via tournament selection from elite
        let parent1 =
            self.select_parent(&elite_completed, &elite_ranks, &elite_distances, &mut rng);
        let parent2 =
            self.select_parent(&elite_completed, &elite_ranks, &elite_distances, &mut rng);

        // Get parent values for this parameter
        let p1_val = parent1.params.get(param_name);
        let p2_val = parent2.params.get(param_name);

        match (p1_val, p2_val) {
            (Some(&v1), Some(&v2)) => {
                match &self.config.crossover_type {
                    CrossoverType::SBX => {
                        // SBX crossover + polynomial mutation
                        let child_val = match distribution {
                            Distribution::Float(d) => {
                                let child = if rng.random_bool(self.config.crossover_prob) {
                                    Self::sbx_crossover(v1, v2, d.low, d.high, self.config.eta_c, &mut rng)
                                } else {
                                    if rng.random_bool(0.5) { v1 } else { v2 }
                                };
                                let mutated = if rng.random_bool(self.config.mutation_prob) {
                                    Self::polynomial_mutation(child, d.low, d.high, self.config.eta_m, &mut rng)
                                } else {
                                    child
                                };
                                if let Some(step) = d.step {
                                    let shifted = mutated - d.low;
                                    (shifted / step).round() * step + d.low
                                } else {
                                    mutated
                                }
                            }
                            Distribution::Int(d) => {
                                let low = d.low as f64;
                                let high = d.high as f64;
                                let child = if rng.random_bool(self.config.crossover_prob) {
                                    Self::sbx_crossover(v1, v2, low, high, self.config.eta_c, &mut rng)
                                } else {
                                    if rng.random_bool(0.5) { v1 } else { v2 }
                                };
                                let mutated = if rng.random_bool(self.config.mutation_prob) {
                                    Self::polynomial_mutation(child, low, high, self.config.eta_m, &mut rng)
                                } else {
                                    child
                                };
                                let step = d.step.unwrap_or(1) as f64;
                                let shifted = mutated - low;
                                let quantized = (shifted / step).round() * step + low;
                                quantized.clamp(low, high)
                            }
                            Distribution::Categorical(d) => {
                                // Categorical: uniform crossover + random mutation
                                let child = if rng.random_bool(0.5) { v1 } else { v2 };
                                if rng.random_bool(self.config.mutation_prob) {
                                    rng.random_range(0..d.choices.len()) as f64
                                } else {
                                    child
                                }
                            }
                        };
                        child_val
                    }
                    CrossoverType::Uniform => {
                        // Original uniform crossover + Gaussian mutation
                        let child_val = if rng.random_bool(self.config.crossover_prob) {
                            v2
                        } else {
                            v1
                        };

                        if rng.random_bool(self.config.mutation_prob) {
                            match distribution {
                                Distribution::Float(d) => {
                                    let range = if d.log {
                                        d.high.ln() - d.low.ln()
                                    } else {
                                        d.high - d.low
                                    };
                                    let noise: f64 = rng.random_range(-1.0..1.0)
                                        * self.config.mutation_sigma
                                        * range;
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
                                    let noise = rng.random_range(-1.0..1.0)
                                        * self.config.mutation_sigma
                                        * range;
                                    let mutated = (child_val + noise).round();
                                    let step = d.step.unwrap_or(1) as f64;
                                    let shifted = mutated - d.low as f64;
                                    let quantized =
                                        (shifted / step).round() * step + d.low as f64;
                                    quantized.clamp(d.low as f64, d.high as f64)
                                }
                                Distribution::Categorical(d) => {
                                    rng.random_range(0..d.choices.len()) as f64
                                }
                            }
                        } else {
                            child_val
                        }
                    }
                    CrossoverType::SPX => {
                        // Simplex crossover + polynomial mutation
                        let epsilon = 3.0_f64.sqrt(); // Standard SPX expansion
                        match distribution {
                            Distribution::Float(d) => {
                                let child = if rng.random_bool(self.config.crossover_prob) {
                                    Self::spx_crossover(v1, v2, d.low, d.high, epsilon, &mut rng)
                                } else {
                                    if rng.random_bool(0.5) { v1 } else { v2 }
                                };
                                let mutated = if rng.random_bool(self.config.mutation_prob) {
                                    Self::polynomial_mutation(child, d.low, d.high, self.config.eta_m, &mut rng)
                                } else {
                                    child
                                };
                                if let Some(step) = d.step {
                                    let shifted = mutated - d.low;
                                    (shifted / step).round() * step + d.low
                                } else {
                                    mutated
                                }
                            }
                            Distribution::Int(d) => {
                                let (low, high) = (d.low as f64, d.high as f64);
                                let child = if rng.random_bool(self.config.crossover_prob) {
                                    Self::spx_crossover(v1, v2, low, high, epsilon, &mut rng)
                                } else {
                                    if rng.random_bool(0.5) { v1 } else { v2 }
                                };
                                let mutated = if rng.random_bool(self.config.mutation_prob) {
                                    Self::polynomial_mutation(child, low, high, self.config.eta_m, &mut rng)
                                } else {
                                    child
                                };
                                let step = d.step.unwrap_or(1) as f64;
                                let shifted = mutated - low;
                                ((shifted / step).round() * step + low).clamp(low, high)
                            }
                            Distribution::Categorical(d) => {
                                let child = if rng.random_bool(0.5) { v1 } else { v2 };
                                if rng.random_bool(self.config.mutation_prob) {
                                    rng.random_range(0..d.choices.len()) as f64
                                } else {
                                    child
                                }
                            }
                        }
                    }
                    CrossoverType::UNDX => {
                        // UNDX crossover + polynomial mutation
                        let sigma_xi = 0.5;
                        let sigma_eta = 0.35;
                        match distribution {
                            Distribution::Float(d) => {
                                let child = if rng.random_bool(self.config.crossover_prob) {
                                    Self::undx_crossover(v1, v2, d.low, d.high, sigma_xi, sigma_eta, &mut rng)
                                } else {
                                    if rng.random_bool(0.5) { v1 } else { v2 }
                                };
                                let mutated = if rng.random_bool(self.config.mutation_prob) {
                                    Self::polynomial_mutation(child, d.low, d.high, self.config.eta_m, &mut rng)
                                } else {
                                    child
                                };
                                if let Some(step) = d.step {
                                    let shifted = mutated - d.low;
                                    (shifted / step).round() * step + d.low
                                } else {
                                    mutated
                                }
                            }
                            Distribution::Int(d) => {
                                let (low, high) = (d.low as f64, d.high as f64);
                                let child = if rng.random_bool(self.config.crossover_prob) {
                                    Self::undx_crossover(v1, v2, low, high, sigma_xi, sigma_eta, &mut rng)
                                } else {
                                    if rng.random_bool(0.5) { v1 } else { v2 }
                                };
                                let mutated = if rng.random_bool(self.config.mutation_prob) {
                                    Self::polynomial_mutation(child, low, high, self.config.eta_m, &mut rng)
                                } else {
                                    child
                                };
                                let step = d.step.unwrap_or(1) as f64;
                                let shifted = mutated - low;
                                ((shifted / step).round() * step + low).clamp(low, high)
                            }
                            Distribution::Categorical(d) => {
                                let child = if rng.random_bool(0.5) { v1 } else { v2 };
                                if rng.random_bool(self.config.mutation_prob) {
                                    rng.random_range(0..d.choices.len()) as f64
                                } else {
                                    child
                                }
                            }
                        }
                    }
                    CrossoverType::VSBX => {
                        // Velocity-based SBX + polynomial mutation
                        match distribution {
                            Distribution::Float(d) => {
                                let child = if rng.random_bool(self.config.crossover_prob) {
                                    Self::vsbx_crossover(v1, v2, d.low, d.high, self.config.eta_c, &mut rng)
                                } else {
                                    if rng.random_bool(0.5) { v1 } else { v2 }
                                };
                                let mutated = if rng.random_bool(self.config.mutation_prob) {
                                    Self::polynomial_mutation(child, d.low, d.high, self.config.eta_m, &mut rng)
                                } else {
                                    child
                                };
                                if let Some(step) = d.step {
                                    let shifted = mutated - d.low;
                                    (shifted / step).round() * step + d.low
                                } else {
                                    mutated
                                }
                            }
                            Distribution::Int(d) => {
                                let (low, high) = (d.low as f64, d.high as f64);
                                let child = if rng.random_bool(self.config.crossover_prob) {
                                    Self::vsbx_crossover(v1, v2, low, high, self.config.eta_c, &mut rng)
                                } else {
                                    if rng.random_bool(0.5) { v1 } else { v2 }
                                };
                                let mutated = if rng.random_bool(self.config.mutation_prob) {
                                    Self::polynomial_mutation(child, low, high, self.config.eta_m, &mut rng)
                                } else {
                                    child
                                };
                                let step = d.step.unwrap_or(1) as f64;
                                let shifted = mutated - low;
                                ((shifted / step).round() * step + low).clamp(low, high)
                            }
                            Distribution::Categorical(d) => {
                                let child = if rng.random_bool(0.5) { v1 } else { v2 };
                                if rng.random_bool(self.config.mutation_prob) {
                                    rng.random_range(0..d.choices.len()) as f64
                                } else {
                                    child
                                }
                            }
                        }
                    }
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

    #[test]
    fn test_sbx_crossover_bounded() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let child = NsgaIISampler::sbx_crossover(3.0, 7.0, 0.0, 10.0, 20.0, &mut rng);
            assert!(
                (0.0..=10.0).contains(&child),
                "SBX child {child} out of [0, 10]"
            );
        }
    }

    #[test]
    fn test_sbx_crossover_produces_children_near_parents() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut near_parent_count = 0;
        let n = 200;
        for _ in 0..n {
            let child = NsgaIISampler::sbx_crossover(4.0, 6.0, 0.0, 10.0, 20.0, &mut rng);
            if child >= 3.0 && child <= 7.0 {
                near_parent_count += 1;
            }
        }
        // With eta_c=20 (high), most children should be near parents
        assert!(
            near_parent_count > n / 2,
            "SBX with high eta should produce children near parents: {near_parent_count}/{n}"
        );
    }

    #[test]
    fn test_polynomial_mutation_bounded() {
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let mutated = NsgaIISampler::polynomial_mutation(5.0, 0.0, 10.0, 20.0, &mut rng);
            assert!(
                (0.0..=10.0).contains(&mutated),
                "Polynomial mutation {mutated} out of [0, 10]"
            );
        }
    }

    #[test]
    fn test_nsga2_uniform_crossover_backward_compat() {
        let config = NsgaIIConfig {
            n_startup_trials: 5,
            population_size: 10,
            crossover_type: CrossoverType::Uniform,
            crossover_prob: 0.5,
            ..Default::default()
        };
        let sampler = NsgaIISampler::new(config, 42);
        let mut study =
            Study::new_multi(vec![Direction::Minimize, Direction::Minimize]);

        for i in 0..10 {
            let mut trial = FrozenTrial::new(i);
            trial.state = TrialState::Complete;
            trial.value = Some(i as f64);
            trial.values = Some(vec![i as f64, (10 - i) as f64]);
            trial.params.insert("x".to_string(), i as f64);
            study.add_trial(trial);
        }

        let trial = Trial::new(10);
        let dist = Distribution::Float(crate::distributions::FloatDistribution::new(
            0.0, 20.0, false, None,
        ));
        let v = sampler.sample(&study, &trial, "x", &dist);
        assert!((0.0..=20.0).contains(&v));
    }

    #[test]
    fn test_nsga2_constrained_sorting() {
        let config = NsgaIIConfig {
            n_startup_trials: 5,
            population_size: 10,
            ..Default::default()
        };
        let sampler = NsgaIISampler::new(config, 42);
        let mut study =
            Study::new_multi(vec![Direction::Minimize, Direction::Minimize]);

        // Add trials with constraint values: some feasible, some infeasible
        for i in 0..10 {
            let mut trial = FrozenTrial::new(i);
            trial.state = TrialState::Complete;
            trial.value = Some(i as f64);
            trial.values = Some(vec![i as f64, (10 - i) as f64]);
            trial.params.insert("x".to_string(), i as f64);
            // Even indices are feasible (constraint <= 0), odd are infeasible
            if i % 2 == 0 {
                trial.constraint_values = Some(vec![-1.0]);
            } else {
                trial.constraint_values = Some(vec![i as f64]); // positive = violation
            }
            study.add_trial(trial);
        }

        let trial = Trial::new(10);
        let dist = Distribution::Float(crate::distributions::FloatDistribution::new(
            0.0, 20.0, false, None,
        ));
        // Should not panic and should produce valid output
        let v = sampler.sample(&study, &trial, "x", &dist);
        assert!((0.0..=20.0).contains(&v));
    }
}
