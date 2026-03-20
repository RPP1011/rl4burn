use rand::prelude::*;
use std::sync::Mutex;

use crate::distributions::{
    int_transform_from_internal, int_transform_to_internal, transform_from_internal,
    transform_to_internal, Distribution, FloatDistribution, IntDistribution,
};
use crate::study::Study;
use crate::trial::{Trial, TrialState};

use super::Sampler;

/// Configuration for the TPE sampler.
#[derive(Debug, Clone)]
pub struct TpeSamplerConfig {
    /// Number of startup trials using random sampling.
    pub n_startup_trials: usize,
    /// Number of EI candidates to evaluate.
    pub n_ei_candidates: usize,
    /// Whether to consider the prior in the Parzen estimator.
    pub consider_prior: bool,
    /// Weight of the prior.
    pub prior_weight: f64,
    /// Whether to apply the "magic clip" heuristic for bandwidth.
    pub consider_magic_clip: bool,
    /// Exponent for magic clip computation.
    pub b_magic_exponent: f64,
    /// Whether to use multivariate TPE (false = independent).
    pub multivariate: bool,
    /// Gamma function variant.
    pub gamma_strategy: GammaStrategy,
}

/// Which gamma function to use for splitting trials into good/bad.
#[derive(Debug, Clone, Copy)]
pub enum GammaStrategy {
    /// `min(ceil(0.1 * n), 25)` — Optuna default.
    Default,
    /// `min(ceil(0.25 * sqrt(n)), 25)` — HyperOpt default.
    Hyperopt,
}

impl Default for TpeSamplerConfig {
    fn default() -> Self {
        Self {
            n_startup_trials: 10,
            n_ei_candidates: 24,
            consider_prior: true,
            prior_weight: 1.0,
            consider_magic_clip: true,
            b_magic_exponent: 1.0,
            multivariate: false,
            gamma_strategy: GammaStrategy::Default,
        }
    }
}

// --- Gamma and weight functions ---

/// Optuna's default gamma: `min(ceil(0.1 * n), 25)`.
pub fn default_gamma(n: usize) -> usize {
    let g = (0.1 * n as f64).ceil() as usize;
    g.clamp(1, 25)
}

/// HyperOpt-compatible gamma: `min(ceil(0.25 * sqrt(n)), 25)`.
pub fn hyperopt_default_gamma(n: usize) -> usize {
    let g = (0.25 * (n as f64).sqrt()).ceil() as usize;
    g.clamp(1, 25)
}

/// Compute weights for the good-distribution trials.
///
/// If `n < 25`, returns all ones. Otherwise, a linear ramp from `1/n` to `1.0`
/// for the first `n-25` entries, then flat ones for the last 25.
pub fn default_weights(n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n < 25 {
        return vec![1.0; n];
    }
    let ramp_len = n - 25;
    let mut weights = Vec::with_capacity(n);
    for i in 0..ramp_len {
        let w = (i as f64 + 1.0) / ramp_len as f64;
        weights.push(w);
    }
    for _ in 0..25 {
        weights.push(1.0);
    }
    weights
}

// --- Order and sorting ---

/// Sort observed values by value (ascending), returning the permutation.
///
/// This implements Optuna's `_calculate_order`: values are sorted by value
/// so that the corresponding weights can be reordered to match. The Parzen
/// estimator requires value-sorted inputs for correct bandwidth computation.
pub fn calculate_order(values: &[f64]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices
}

// --- Parzen Estimator ---

/// A Gaussian Mixture Model for kernel density estimation.
#[derive(Debug, Clone)]
pub struct ParzenEstimator {
    pub mus: Vec<f64>,
    pub sigmas: Vec<f64>,
    pub weights: Vec<f64>,
}

impl ParzenEstimator {
    /// Build a Parzen estimator from observed values.
    ///
    /// `sorted_values`: observation values sorted by objective (best first).
    /// `prior_mu`: prior mean (typically center of the distribution range).
    /// `prior_sigma`: prior bandwidth (typically the range width).
    /// `consider_prior`: whether to include a prior component.
    /// `consider_magic_clip`: whether to apply the magic clip heuristic.
    /// `b_magic_exponent`: exponent for magic clip.
    /// `weights_func`: weights for each observation.
    pub fn new(
        sorted_values: &[f64],
        prior_mu: f64,
        prior_sigma: f64,
        consider_prior: bool,
        consider_magic_clip: bool,
        b_magic_exponent: f64,
        weights_func: &[f64],
    ) -> Self {
        // Find where the prior will be inserted so we can align weights
        let prior_pos = if consider_prior {
            sorted_values
                .iter()
                .position(|&v| v > prior_mu)
                .unwrap_or(sorted_values.len())
        } else {
            0
        };
        let mus = Self::calculate_mus(sorted_values, prior_mu, consider_prior);
        let sigmas =
            Self::calculate_sigmas(&mus, prior_sigma, consider_magic_clip, b_magic_exponent);
        let weights =
            Self::calculate_weights(sorted_values.len(), consider_prior, prior_pos, weights_func);

        ParzenEstimator {
            mus,
            sigmas,
            weights,
        }
    }

    /// Place kernel centers. Inserts the prior mean at the sorted position.
    fn calculate_mus(sorted_values: &[f64], prior_mu: f64, consider_prior: bool) -> Vec<f64> {
        if !consider_prior {
            return sorted_values.to_vec();
        }

        // Insert prior_mu in sorted order
        let pos = sorted_values
            .iter()
            .position(|&v| v > prior_mu)
            .unwrap_or(sorted_values.len());

        let mut mus = Vec::with_capacity(sorted_values.len() + 1);
        mus.extend_from_slice(&sorted_values[..pos]);
        mus.push(prior_mu);
        mus.extend_from_slice(&sorted_values[pos..]);
        mus
    }

    /// Compute bandwidths using adjacent kernel distances and the magic clip heuristic.
    fn calculate_sigmas(
        mus: &[f64],
        prior_sigma: f64,
        consider_magic_clip: bool,
        b_magic_exponent: f64,
    ) -> Vec<f64> {
        let n = mus.len();
        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return vec![prior_sigma];
        }

        // Magic clip: minimum allowed bandwidth
        let magic_clip = if consider_magic_clip {
            prior_sigma / (n as f64).powf(1.0 / b_magic_exponent)
        } else {
            0.0
        };

        let mut sigmas = Vec::with_capacity(n);
        for i in 0..n {
            let sigma = if i == 0 {
                mus[1] - mus[0]
            } else if i == n - 1 {
                mus[n - 1] - mus[n - 2]
            } else {
                (mus[i + 1] - mus[i - 1]) / 2.0
            };

            let sigma = sigma.max(magic_clip).max(1e-12);
            sigmas.push(sigma);
        }

        sigmas
    }

    /// Compute component weights, aligned with the mus vector.
    ///
    /// The prior weight is inserted at `prior_pos` to match where `calculate_mus`
    /// inserted the prior mean.
    fn calculate_weights(
        n_observations: usize,
        consider_prior: bool,
        prior_pos: usize,
        weights_func: &[f64],
    ) -> Vec<f64> {
        debug_assert_eq!(
            weights_func.len(),
            n_observations,
            "weights_func length must match n_observations"
        );
        let mut weights = Vec::with_capacity(n_observations + if consider_prior { 1 } else { 0 });

        if consider_prior {
            let total_obs_weight: f64 = weights_func.iter().sum();
            let normalized_prior = if n_observations > 0 && total_obs_weight > 0.0 {
                total_obs_weight / n_observations as f64
            } else {
                1.0
            };

            // Insert prior weight at the same position as in calculate_mus
            weights.extend_from_slice(&weights_func[..prior_pos.min(n_observations)]);
            weights.push(normalized_prior);
            if prior_pos < n_observations {
                weights.extend_from_slice(&weights_func[prior_pos..]);
            }
        } else {
            weights.extend(weights_func.iter().copied());
        }

        // Normalize
        let total: f64 = weights.iter().sum();
        if total > 0.0 {
            for w in &mut weights {
                *w /= total;
            }
        }

        weights
    }

    /// Evaluate the log-probability of `x` under this GMM.
    pub fn log_pdf(&self, x: f64) -> f64 {
        if self.mus.is_empty() {
            return f64::NEG_INFINITY;
        }
        log_sum_exp_weighted(x, &self.mus, &self.sigmas, &self.weights)
    }

    /// Draw a sample from this GMM.
    pub fn sample(&self, rng: &mut impl Rng) -> f64 {
        if self.mus.is_empty() {
            return 0.0;
        }

        // Pick a component proportionally to weights
        let u: f64 = rng.random();
        let mut cumulative = 0.0;
        let mut chosen = self.weights.len() - 1;
        for (i, &w) in self.weights.iter().enumerate() {
            cumulative += w;
            if u < cumulative {
                chosen = i;
                break;
            }
        }

        // Sample from the chosen Gaussian
        let mu = self.mus[chosen];
        let sigma = self.sigmas[chosen];
        let normal: f64 = rng.sample(rand_distr::StandardNormal);
        mu + sigma * normal
    }
}

/// Categorical Parzen Estimator using the Aitchison-Aitken kernel.
///
/// Instead of a Gaussian mixture, this estimates the probability of each
/// category using a smoothed histogram. The kernel parameter `alpha` controls
/// smoothing: alpha=0 gives the empirical distribution, alpha=1 gives uniform.
#[derive(Debug, Clone)]
pub struct CategoricalParzenEstimator {
    pub probabilities: Vec<f64>,
}

impl CategoricalParzenEstimator {
    /// Build a categorical Parzen estimator from observed category indices.
    ///
    /// `observations`: category indices observed so far.
    /// `n_choices`: total number of categories.
    /// `prior_weight`: weight of the uniform prior.
    /// `weights`: per-observation weights.
    pub fn new(
        observations: &[usize],
        n_choices: usize,
        prior_weight: f64,
        weights: &[f64],
    ) -> Self {
        assert!(n_choices > 0);

        // Aitchison-Aitken kernel bandwidth
        // alpha = 1 / (n_observations + 1) heuristic, clamped
        let n = observations.len() as f64;
        let alpha = if n > 0.0 {
            (1.0 / (n + 1.0)).clamp(1e-12, 1.0 - 1e-12)
        } else {
            1.0 / n_choices as f64
        };

        let mut counts = vec![0.0_f64; n_choices];

        // Weighted counts with Aitchison-Aitken smoothing
        for (i, &cat) in observations.iter().enumerate() {
            let w = if i < weights.len() { weights[i] } else { 1.0 };
            for j in 0..n_choices {
                if j == cat {
                    counts[j] += w * (1.0 - alpha);
                } else {
                    counts[j] += w * alpha / (n_choices as f64 - 1.0);
                }
            }
        }

        // Add uniform prior
        let prior_per_cat = prior_weight / n_choices as f64;
        for c in &mut counts {
            *c += prior_per_cat;
        }

        // Normalize to probabilities
        let total: f64 = counts.iter().sum();
        let probabilities = if total > 0.0 {
            counts.iter().map(|&c| c / total).collect()
        } else {
            vec![1.0 / n_choices as f64; n_choices]
        };

        Self { probabilities }
    }

    /// Log-probability of a given category index.
    pub fn log_pdf(&self, index: usize) -> f64 {
        if index >= self.probabilities.len() {
            return f64::NEG_INFINITY;
        }
        let p = self.probabilities[index];
        if p <= 0.0 {
            f64::NEG_INFINITY
        } else {
            p.ln()
        }
    }

    /// Sample a category index from this estimator.
    #[allow(dead_code)]
    pub fn sample(&self, rng: &mut impl Rng) -> usize {
        let u: f64 = rng.random();
        let mut cumulative = 0.0;
        for (i, &p) in self.probabilities.iter().enumerate() {
            cumulative += p;
            if u < cumulative {
                return i;
            }
        }
        self.probabilities.len() - 1
    }
}

/// Log-pdf of a single Gaussian component.
fn gaussian_log_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    -0.5 * z * z - sigma.ln() - 0.5 * std::f64::consts::TAU.ln()
}

/// Log-sum-exp of weighted Gaussian components.
fn log_sum_exp_weighted(x: f64, mus: &[f64], sigmas: &[f64], weights: &[f64]) -> f64 {
    // Compute log(w_i) + log_pdf(x | mu_i, sigma_i) for each component
    let log_components: Vec<f64> = mus
        .iter()
        .zip(sigmas.iter())
        .zip(weights.iter())
        .filter(|((_, _), &w)| w > 0.0)
        .map(|((&mu, &sigma), &w)| w.ln() + gaussian_log_pdf(x, mu, sigma))
        .collect();

    if log_components.is_empty() {
        return f64::NEG_INFINITY;
    }

    // Log-sum-exp for numerical stability
    let max_val = log_components
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    if max_val == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }

    let sum_exp: f64 = log_components.iter().map(|&lc| (lc - max_val).exp()).sum();

    max_val + sum_exp.ln()
}

/// TPE Sampler implementing the Tree-structured Parzen Estimator.
pub struct TpeSampler {
    config: TpeSamplerConfig,
    random_sampler: super::RandomSampler,
    rng: Mutex<StdRng>,
}

impl TpeSampler {
    /// Create a new TPE sampler with the given config and seed.
    pub fn new(config: TpeSamplerConfig, seed: u64) -> Self {
        Self {
            config,
            random_sampler: super::RandomSampler::new(seed),
            rng: Mutex::new(StdRng::seed_from_u64(seed.wrapping_add(1))),
        }
    }

    /// Create a TPE sampler with default config.
    pub fn with_seed(seed: u64) -> Self {
        Self::new(TpeSamplerConfig::default(), seed)
    }

    /// Get the gamma value (number of "good" trials) for a given trial count.
    fn gamma(&self, n: usize) -> usize {
        match self.config.gamma_strategy {
            GammaStrategy::Default => default_gamma(n),
            GammaStrategy::Hyperopt => hyperopt_default_gamma(n),
        }
    }

    /// Split completed trials into good/bad index groups, sorted by objective.
    /// Returns (good_indices, bad_indices) into the `completed` slice.
    fn split_trials(
        &self,
        study: &Study,
        completed: &[&crate::trial::FrozenTrial],
    ) -> (Vec<usize>, Vec<usize>) {
        let n = completed.len();
        let mut trial_indices: Vec<usize> = (0..n).collect();
        trial_indices.sort_by(|&a, &b| {
            let va = completed[a].value.unwrap_or(f64::INFINITY);
            let vb = completed[b].value.unwrap_or(f64::INFINITY);
            if study.direction() == crate::study::Direction::Maximize {
                vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        let n_good = self.gamma(n);
        let good = trial_indices[..n_good].to_vec();
        let bad = trial_indices[n_good..].to_vec();
        (good, bad)
    }

    /// Sample a float parameter using TPE.
    fn sample_float(
        &self,
        study: &Study,
        dist: &FloatDistribution,
        param_name: &str,
    ) -> f64 {
        let completed: Vec<_> = study
            .trials()
            .iter()
            .filter(|t| t.state == TrialState::Complete && t.params.contains_key(param_name))
            .collect();

        let n = completed.len();
        if n < self.config.n_startup_trials {
            return self.random_sampler.sample(
                study,
                &Trial::new(0),
                param_name,
                &Distribution::Float(dist.clone()),
            );
        }

        let (good_indices, bad_indices) = self.split_trials(study, &completed);

        // Extract parameter values in internal [0, 1] space
        let good_values: Vec<f64> = good_indices
            .iter()
            .map(|&i| transform_to_internal(completed[i].params[param_name], dist))
            .collect();

        let bad_values: Vec<f64> = bad_indices
            .iter()
            .map(|&i| transform_to_internal(completed[i].params[param_name], dist))
            .collect();

        // Apply weight-priority ordering
        let good_w = default_weights(good_values.len());
        let good_order = calculate_order(&good_values);
        let sorted_good: Vec<f64> = good_order.iter().map(|&i| good_values[i]).collect();
        let sorted_good_w: Vec<f64> = good_order.iter().map(|&i| good_w[i]).collect();

        let bad_w = default_weights(bad_values.len());
        let bad_order = calculate_order(&bad_values);
        let sorted_bad: Vec<f64> = bad_order.iter().map(|&i| bad_values[i]).collect();
        let sorted_bad_w: Vec<f64> = bad_order.iter().map(|&i| bad_w[i]).collect();

        // Build Parzen estimators for l(x) and g(x)
        let prior_mu = 0.5;
        let prior_sigma = 1.0;

        let l = ParzenEstimator::new(
            &sorted_good,
            prior_mu,
            prior_sigma,
            self.config.consider_prior,
            self.config.consider_magic_clip,
            self.config.b_magic_exponent,
            &sorted_good_w,
        );

        let g = ParzenEstimator::new(
            &sorted_bad,
            prior_mu,
            prior_sigma,
            self.config.consider_prior,
            self.config.consider_magic_clip,
            self.config.b_magic_exponent,
            &sorted_bad_w,
        );

        // Draw candidates from l(x), evaluate EI = l(x)/g(x), pick best
        let mut rng = self.rng.lock().unwrap();
        let mut best_candidate = 0.5;
        let mut best_ei = f64::NEG_INFINITY;

        for _ in 0..self.config.n_ei_candidates {
            let candidate = l.sample(&mut *rng).clamp(0.0, 1.0);
            let l_log = l.log_pdf(candidate);
            let g_log = g.log_pdf(candidate);
            let ei = l_log - g_log;

            if ei > best_ei {
                best_ei = ei;
                best_candidate = candidate;
            }
        }

        transform_from_internal(best_candidate, dist)
    }

    /// Sample an integer parameter using TPE.
    fn sample_int(
        &self,
        study: &Study,
        dist: &IntDistribution,
        param_name: &str,
    ) -> f64 {
        let completed: Vec<_> = study
            .trials()
            .iter()
            .filter(|t| t.state == TrialState::Complete && t.params.contains_key(param_name))
            .collect();

        let n = completed.len();
        if n < self.config.n_startup_trials {
            return self.random_sampler.sample(
                study,
                &Trial::new(0),
                param_name,
                &Distribution::Int(dist.clone()),
            );
        }

        let (good_indices, bad_indices) = self.split_trials(study, &completed);

        // Extract parameter values in internal [0, 1] space
        let good_values: Vec<f64> = good_indices
            .iter()
            .map(|&i| {
                int_transform_to_internal(completed[i].params[param_name] as i64, dist)
            })
            .collect();

        let bad_values: Vec<f64> = bad_indices
            .iter()
            .map(|&i| {
                int_transform_to_internal(completed[i].params[param_name] as i64, dist)
            })
            .collect();

        // Apply weight-priority ordering
        let good_w = default_weights(good_values.len());
        let good_order = calculate_order(&good_values);
        let sorted_good: Vec<f64> = good_order.iter().map(|&i| good_values[i]).collect();
        let sorted_good_w: Vec<f64> = good_order.iter().map(|&i| good_w[i]).collect();

        let bad_w = default_weights(bad_values.len());
        let bad_order = calculate_order(&bad_values);
        let sorted_bad: Vec<f64> = bad_order.iter().map(|&i| bad_values[i]).collect();
        let sorted_bad_w: Vec<f64> = bad_order.iter().map(|&i| bad_w[i]).collect();

        let prior_mu = 0.5;
        let prior_sigma = 1.0;

        let l = ParzenEstimator::new(
            &sorted_good, prior_mu, prior_sigma,
            self.config.consider_prior, self.config.consider_magic_clip,
            self.config.b_magic_exponent, &sorted_good_w,
        );

        let g = ParzenEstimator::new(
            &sorted_bad, prior_mu, prior_sigma,
            self.config.consider_prior, self.config.consider_magic_clip,
            self.config.b_magic_exponent, &sorted_bad_w,
        );

        let mut rng = self.rng.lock().unwrap();
        let mut best_candidate = 0.5;
        let mut best_ei = f64::NEG_INFINITY;

        for _ in 0..self.config.n_ei_candidates {
            let candidate = l.sample(&mut *rng).clamp(0.0, 1.0);
            let l_log = l.log_pdf(candidate);
            let g_log = g.log_pdf(candidate);
            let ei = l_log - g_log;

            if ei > best_ei {
                best_ei = ei;
                best_candidate = candidate;
            }
        }

        int_transform_from_internal(best_candidate, dist) as f64
    }

    /// Sample a categorical parameter using TPE with Aitchison-Aitken kernels.
    fn sample_categorical(
        &self,
        study: &Study,
        n_choices: usize,
        param_name: &str,
    ) -> f64 {
        let completed: Vec<_> = study
            .trials()
            .iter()
            .filter(|t| t.state == TrialState::Complete && t.params.contains_key(param_name))
            .collect();

        let n = completed.len();
        if n < self.config.n_startup_trials {
            return self.random_sampler.sample(
                study,
                &Trial::new(0),
                param_name,
                &Distribution::Categorical(crate::distributions::CategoricalDistribution::new(
                    (0..n_choices).map(|i| i.to_string()).collect(),
                )),
            );
        }

        let (good_indices, bad_indices) = self.split_trials(study, &completed);

        let good_cats: Vec<usize> = good_indices
            .iter()
            .map(|&i| completed[i].params[param_name] as usize)
            .collect();

        let bad_cats: Vec<usize> = bad_indices
            .iter()
            .map(|&i| completed[i].params[param_name] as usize)
            .collect();

        let good_w = default_weights(good_cats.len());
        let bad_w = default_weights(bad_cats.len());

        let prior_weight = if self.config.consider_prior {
            self.config.prior_weight
        } else {
            0.0
        };

        let l = CategoricalParzenEstimator::new(&good_cats, n_choices, prior_weight, &good_w);
        let g = CategoricalParzenEstimator::new(&bad_cats, n_choices, prior_weight, &bad_w);

        // Evaluate EI for each category and pick the best
        let mut best_cat = 0;
        let mut best_ei = f64::NEG_INFINITY;

        for cat in 0..n_choices {
            let l_log = l.log_pdf(cat);
            let g_log = g.log_pdf(cat);
            let ei = l_log - g_log;

            if ei > best_ei {
                best_ei = ei;
                best_cat = cat;
            }
        }

        best_cat as f64
    }
}

impl Sampler for TpeSampler {
    fn sample(
        &self,
        study: &Study,
        _trial: &Trial,
        param_name: &str,
        distribution: &Distribution,
    ) -> f64 {
        match distribution {
            Distribution::Float(d) => self.sample_float(study, d, param_name),
            Distribution::Int(d) => self.sample_int(study, d, param_name),
            Distribution::Categorical(d) => {
                self.sample_categorical(study, d.choices.len(), param_name)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_gamma() {
        assert_eq!(default_gamma(1), 1);
        assert_eq!(default_gamma(10), 1);
        assert_eq!(default_gamma(11), 2);
        assert_eq!(default_gamma(100), 10);
        assert_eq!(default_gamma(250), 25);
        assert_eq!(default_gamma(1000), 25);
    }

    #[test]
    fn test_gamma_monotonic() {
        for n in 1..1000 {
            assert!(default_gamma(n) <= default_gamma(n + 1));
        }
    }

    #[test]
    fn test_hyperopt_gamma() {
        assert_eq!(hyperopt_default_gamma(1), 1);
        assert_eq!(hyperopt_default_gamma(16), 1);
        assert_eq!(hyperopt_default_gamma(100), 3);
        // At n=10000, ceil(0.25*100)=25
        assert_eq!(hyperopt_default_gamma(10000), 25);
    }

    #[test]
    fn test_default_weights() {
        assert_eq!(default_weights(0), vec![]);
        assert_eq!(default_weights(1), vec![1.0]);
        assert_eq!(default_weights(3), vec![1.0, 1.0, 1.0]);

        let w = default_weights(30);
        assert_eq!(w.len(), 30);
        // First 5 elements are the ramp (n-25 = 5)
        assert!((w[0] - 0.2).abs() < 1e-10);
        assert!((w[4] - 1.0).abs() < 1e-10);
        // Last 25 are all 1.0
        for &wi in &w[5..] {
            assert!((wi - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_weights_all_positive() {
        for n in 1..200 {
            let w = default_weights(n);
            assert_eq!(w.len(), n);
            for &wi in &w {
                assert!(wi > 0.0, "weight must be positive, got {wi} for n={n}");
            }
        }
    }

    #[test]
    fn test_gaussian_log_pdf() {
        // Standard normal at x=0 should give -0.5 * ln(2*pi)
        let lp = gaussian_log_pdf(0.0, 0.0, 1.0);
        let expected = -0.5 * std::f64::consts::TAU.ln();
        assert!((lp - expected).abs() < 1e-12);
    }

    #[test]
    fn test_gaussian_log_pdf_no_nan() {
        let test_cases = [
            (0.0, 0.0, 1.0),
            (100.0, 0.0, 1.0),
            (-100.0, 50.0, 0.1),
            (1e-10, 1e-10, 1e-15),
        ];
        for (x, mu, sigma) in test_cases {
            let lp = gaussian_log_pdf(x, mu, sigma);
            assert!(!lp.is_nan(), "NaN for x={x}, mu={mu}, sigma={sigma}");
            assert!(lp != f64::INFINITY, "+Inf for x={x}, mu={mu}, sigma={sigma}");
        }
    }

    #[test]
    fn test_parzen_estimator_sigmas_positive() {
        let values = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let pe = ParzenEstimator::new(&values, 0.5, 1.0, true, true, 1.0, &[1.0; 5]);
        for &s in &pe.sigmas {
            assert!(s > 0.0, "sigma must be positive, got {s}");
            assert!(s.is_finite(), "sigma must be finite");
        }
    }

    #[test]
    fn test_parzen_estimator_log_pdf_stable() {
        let values = vec![0.2, 0.4, 0.6, 0.8];
        let pe = ParzenEstimator::new(&values, 0.5, 1.0, true, true, 1.0, &[1.0; 4]);

        for i in 0..100 {
            let x = i as f64 / 100.0;
            let lp = pe.log_pdf(x);
            assert!(!lp.is_nan(), "NaN at x={x}");
        }
    }

    #[test]
    fn test_parzen_estimator_sample() {
        let values = vec![0.3, 0.5, 0.7];
        let pe = ParzenEstimator::new(&values, 0.5, 1.0, true, true, 1.0, &[1.0; 3]);

        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let s = pe.sample(&mut rng);
            assert!(s.is_finite(), "sample must be finite");
        }
    }

    #[test]
    fn test_calculate_order() {
        let values = vec![0.5, 0.1, 0.9, 0.3];
        let order = calculate_order(&values);
        // Sort by value ascending
        let sorted: Vec<f64> = order.iter().map(|&i| values[i]).collect();
        assert_eq!(sorted, vec![0.1, 0.3, 0.5, 0.9]);
    }

    #[test]
    fn test_calculate_order_reorders_weights() {
        let values = vec![0.5, 0.1, 0.9];
        let weights = vec![1.0, 3.0, 2.0];
        let order = calculate_order(&values);
        // Values sorted ascending: 0.1 (idx 1), 0.5 (idx 0), 0.9 (idx 2)
        assert_eq!(order, vec![1, 0, 2]);
        // Weights follow: [3.0, 1.0, 2.0]
        let reordered_w: Vec<f64> = order.iter().map(|&i| weights[i]).collect();
        assert_eq!(reordered_w, vec![3.0, 1.0, 2.0]);
    }

    #[test]
    fn test_categorical_parzen_estimator() {
        let obs = vec![0, 0, 0, 1, 2];
        let weights = vec![1.0; 5];
        let pe = CategoricalParzenEstimator::new(&obs, 3, 1.0, &weights);
        assert_eq!(pe.probabilities.len(), 3);
        // Category 0 observed most, should have highest probability
        assert!(pe.probabilities[0] > pe.probabilities[1]);
        assert!(pe.probabilities[0] > pe.probabilities[2]);
        // All positive
        for &p in &pe.probabilities {
            assert!(p > 0.0);
        }
        // Sum to 1
        let total: f64 = pe.probabilities.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_categorical_parzen_sample() {
        let obs = vec![0, 0, 0, 1, 2];
        let pe = CategoricalParzenEstimator::new(&obs, 3, 1.0, &[1.0; 5]);
        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let s = pe.sample(&mut rng);
            assert!(s < 3);
        }
    }

    #[test]
    fn test_tpe_sampler_suggests_in_bounds() {
        let sampler = TpeSampler::with_seed(42);
        let dist = FloatDistribution::new(0.0, 10.0, false, None);
        let mut study = Study::new_default();

        // Add enough trials to pass startup phase
        for i in 0..20 {
            let value = (i as f64) * 0.5;
            let mut params = std::collections::HashMap::new();
            params.insert("x".to_string(), value);
            study.add_completed_trial(params, value);
        }

        let trial = Trial::new(20);
        let v = sampler.sample(
            &study,
            &trial,
            "x",
            &Distribution::Float(dist.clone()),
        );
        assert!(
            v >= 0.0 && v <= 10.0,
            "TPE suggested value {v} outside [0, 10]"
        );
    }

    #[test]
    fn test_tpe_sampler_int_in_bounds() {
        let sampler = TpeSampler::with_seed(42);
        let dist = crate::distributions::IntDistribution::new(1, 100, false, None);
        let mut study = Study::new_default();

        for i in 0..20 {
            let value = (i * 5 + 1) as f64;
            let mut params = std::collections::HashMap::new();
            params.insert("n".to_string(), value);
            study.add_completed_trial(params, value);
        }

        let trial = Trial::new(20);
        let v = sampler.sample(
            &study,
            &trial,
            "n",
            &Distribution::Int(dist),
        ) as i64;
        assert!(
            v >= 1 && v <= 100,
            "TPE suggested int {v} outside [1, 100]"
        );
    }

    #[test]
    fn test_tpe_sampler_categorical() {
        let sampler = TpeSampler::with_seed(42);
        let dist = crate::distributions::CategoricalDistribution::new(vec![
            "a".into(), "b".into(), "c".into(), "d".into(),
        ]);
        let mut study = Study::new_default();

        // Add trials where category 0 ("a") tends to have the best objective
        for i in 0..20 {
            let cat = (i % 4) as f64;
            let value = if cat == 0.0 { 0.1 } else { 5.0 + cat };
            let mut params = std::collections::HashMap::new();
            params.insert("opt".to_string(), cat);
            study.add_completed_trial(params, value);
        }

        let trial = Trial::new(20);
        let v = sampler.sample(
            &study,
            &trial,
            "opt",
            &Distribution::Categorical(dist),
        ) as usize;
        assert!(v < 4, "TPE suggested category {v} outside [0, 4)");
        // With category 0 being consistently best, TPE should favor it
        assert_eq!(v, 0, "TPE should favor category 0 which has the best objective");
    }
}
