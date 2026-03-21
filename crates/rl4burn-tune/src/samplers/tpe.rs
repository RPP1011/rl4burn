use rand::prelude::*;
use std::sync::Mutex;

use crate::distributions::{Distribution, FloatDistribution, IntDistribution};
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
    /// Whether to use endpoints for boundary sigma computation.
    pub consider_endpoints: bool,
    /// Whether to use multivariate TPE (false = independent).
    pub multivariate: bool,
    /// Whether to use the constant liar strategy for parallel sampling.
    /// When enabled, running trials are included with imputed values.
    pub constant_liar: bool,
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
            consider_endpoints: false,
            multivariate: false,
            constant_liar: false,
            gamma_strategy: GammaStrategy::Default,
        }
    }
}

// --- Gamma and weight functions ---

/// Optuna's default gamma: `min(ceil(0.1 * n), 25)`.
pub fn default_gamma(n: usize) -> usize {
    let g = (0.1 * n as f64).ceil() as usize;
    g.min(25)
}

/// HyperOpt-compatible gamma: `min(ceil(0.25 * sqrt(n)), 25)`.
pub fn hyperopt_default_gamma(n: usize) -> usize {
    let g = (0.25 * (n as f64).sqrt()).ceil() as usize;
    g.min(25)
}

/// Compute weights for the good-distribution trials.
///
/// If `n < 25`, returns all ones. Otherwise, `np.linspace(1/n, 1.0, num=n-25)`
/// for the ramp, then flat ones for the last 25. Matches Optuna's `default_weights`.
pub fn default_weights(n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n < 25 {
        return vec![1.0; n];
    }
    let ramp_len = n - 25;
    let mut weights = Vec::with_capacity(n);
    if ramp_len == 0 {
        // No ramp, all flat
    } else if ramp_len == 1 {
        weights.push(1.0 / n as f64);
    } else {
        // linspace(1/n, 1.0, ramp_len)
        let start = 1.0 / n as f64;
        let end = 1.0;
        for i in 0..ramp_len {
            let w = start + (end - start) * i as f64 / (ramp_len - 1) as f64;
            weights.push(w);
        }
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

const EPS: f64 = 1e-12;

/// A truncated Gaussian Mixture Model for kernel density estimation.
///
/// Matches Optuna's `_ParzenEstimator._calculate_numerical_distributions`.
/// Each component is a truncated normal distribution on `[low, high]`.
#[derive(Debug, Clone)]
pub struct ParzenEstimator {
    pub mus: Vec<f64>,
    pub sigmas: Vec<f64>,
    pub weights: Vec<f64>,
    pub low: f64,
    pub high: f64,
}

impl ParzenEstimator {
    /// Build a Parzen estimator from observed values, matching Optuna's algorithm.
    ///
    /// `observations`: parameter values (in original space, not necessarily sorted).
    /// `low`, `high`: distribution bounds.
    /// `prior_weight`: weight of the prior component relative to observations.
    /// `consider_magic_clip`: whether to apply the magic clip heuristic.
    /// `consider_endpoints`: whether to use endpoints for boundary sigma.
    /// `weights_func`: weights for each observation (from `default_weights`).
    pub fn new(
        observations: &[f64],
        low: f64,
        high: f64,
        prior_weight: f64,
        consider_magic_clip: bool,
        consider_endpoints: bool,
        weights_func: &[f64],
    ) -> Self {
        let n_obs = observations.len();
        let prior_mu = 0.5 * (low + high);
        let prior_sigma = high - low;

        // Compute sigmas for observations (Optuna's univariate branch)
        let obs_sigmas = Self::calculate_observation_sigmas(
            observations,
            low,
            high,
            prior_mu,
            consider_magic_clip,
            consider_endpoints,
        );

        // Build mus: observations followed by prior
        let mut mus = Vec::with_capacity(n_obs + 1);
        mus.extend_from_slice(observations);
        mus.push(prior_mu);

        // Build sigmas: observation sigmas followed by prior sigma
        let mut sigmas = Vec::with_capacity(n_obs + 1);
        sigmas.extend_from_slice(&obs_sigmas);
        sigmas.push(prior_sigma);

        // Build weights: observation weights + prior weight, normalized
        let mut weights = Vec::with_capacity(n_obs + 1);
        weights.extend_from_slice(weights_func);
        weights.push(prior_weight);
        let total: f64 = weights.iter().sum();
        if total > 0.0 {
            for w in &mut weights {
                *w /= total;
            }
        }

        ParzenEstimator {
            mus,
            sigmas,
            weights,
            low,
            high,
        }
    }

    /// Compute sigmas for each observation, matching Optuna's univariate branch.
    ///
    /// Algorithm:
    /// 1. Append prior_mu to observations
    /// 2. Sort all mus (observations + prior)
    /// 3. Prepend `low` and append `high` as endpoints
    /// 4. sigma[i] = max(sorted[i+1] - sorted[i], sorted[i] - sorted[i-1])
    /// 5. Optionally adjust boundary sigmas if !consider_endpoints
    /// 6. Unsort sigmas back to original observation order
    /// 7. Clip sigmas to [minsigma, maxsigma]
    fn calculate_observation_sigmas(
        observations: &[f64],
        low: f64,
        high: f64,
        prior_mu: f64,
        consider_magic_clip: bool,
        consider_endpoints: bool,
    ) -> Vec<f64> {
        let n_obs = observations.len();

        // Combine observations with prior
        let mut mus_with_prior = Vec::with_capacity(n_obs + 1);
        mus_with_prior.extend_from_slice(observations);
        mus_with_prior.push(prior_mu);

        // Sort and track original indices
        let mut sorted_indices: Vec<usize> = (0..mus_with_prior.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            mus_with_prior[a]
                .partial_cmp(&mus_with_prior[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let sorted_mus: Vec<f64> = sorted_indices.iter().map(|&i| mus_with_prior[i]).collect();

        // Add endpoints: [low, sorted_mus..., high]
        let n_with_endpoints = sorted_mus.len() + 2;
        let mut with_endpoints = Vec::with_capacity(n_with_endpoints);
        with_endpoints.push(low);
        with_endpoints.extend_from_slice(&sorted_mus);
        with_endpoints.push(high);

        // Compute sigmas: max(right_diff, left_diff) for each sorted mu
        let n_kernels = sorted_mus.len();
        let mut sorted_sigmas = Vec::with_capacity(n_kernels);
        for i in 0..n_kernels {
            let left_diff = with_endpoints[i + 1] - with_endpoints[i];
            let right_diff = with_endpoints[i + 2] - with_endpoints[i + 1];
            sorted_sigmas.push(left_diff.max(right_diff));
        }

        // Adjust endpoint sigmas if !consider_endpoints and enough data
        if !consider_endpoints && n_with_endpoints >= 4 {
            // First kernel: use right_diff only (distance to next)
            sorted_sigmas[0] = with_endpoints[2] - with_endpoints[1];
            // Last kernel: use left_diff only (distance from previous)
            let last = n_kernels - 1;
            sorted_sigmas[last] =
                with_endpoints[n_with_endpoints - 2] - with_endpoints[n_with_endpoints - 3];
        }

        // Unsort: map back to original observation order
        let mut unsorted_sigmas = vec![0.0; n_kernels];
        for (sorted_pos, &original_idx) in sorted_indices.iter().enumerate() {
            unsorted_sigmas[original_idx] = sorted_sigmas[sorted_pos];
        }

        // Only keep observation sigmas (drop prior's sigma, which is index n_obs)
        let obs_sigmas = &unsorted_sigmas[..n_obs];

        // Clip sigmas
        let maxsigma = high - low;
        let minsigma = if consider_magic_clip {
            let n_kernels_total = n_obs + 1; // observations + prior
            (high - low) / (100.0_f64).min((1 + n_kernels_total) as f64)
        } else {
            EPS
        };

        obs_sigmas
            .iter()
            .map(|&s| s.clamp(minsigma, maxsigma))
            .collect()
    }

    /// Evaluate the log-probability of `x` under this truncated GMM.
    pub fn log_pdf(&self, x: f64) -> f64 {
        if self.mus.is_empty() {
            return f64::NEG_INFINITY;
        }
        truncated_gmm_log_pdf(x, &self.mus, &self.sigmas, &self.weights, self.low, self.high)
    }

    /// Draw a sample from this truncated GMM.
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

        // Sample from the chosen truncated Gaussian
        let mu = self.mus[chosen];
        let sigma = self.sigmas[chosen];
        sample_truncated_normal(mu, sigma, self.low, self.high, rng)
    }
}

/// Sample from a truncated normal distribution on [low, high].
fn sample_truncated_normal(mu: f64, sigma: f64, low: f64, high: f64, rng: &mut impl Rng) -> f64 {
    // Simple rejection sampling
    loop {
        let normal: f64 = rng.sample(rand_distr::StandardNormal);
        let x = mu + sigma * normal;
        if x >= low && x <= high {
            return x;
        }
    }
}

/// Categorical Parzen Estimator matching Optuna's `_calculate_categorical_distributions`.
///
/// Each observation and the prior become separate mixture components.
/// Each component has a weight vector over categories. The mixture is weighted
/// by `weights_func` (normalized) and evaluated via log-sum-exp.
#[derive(Debug, Clone)]
pub struct CategoricalParzenEstimator {
    /// Per-component categorical weights: `component_weights[k][j]` is the
    /// probability of category `j` in component `k`. Each row sums to 1.
    pub component_weights: Vec<Vec<f64>>,
    /// Mixture weights (one per component), sum to 1.
    pub mixture_weights: Vec<f64>,
    pub n_choices: usize,
}

impl CategoricalParzenEstimator {
    /// Build a categorical Parzen estimator from observed category indices.
    ///
    /// Matches Optuna: `n_kernels = n_observations + 1` (one prior).
    /// Each kernel starts with base weight `prior_weight / n_kernels` for all categories.
    /// Observation kernels get `+1` at their observed category index.
    /// Prior kernel stays uniform. Each row is then normalized.
    pub fn new(
        observations: &[usize],
        n_choices: usize,
        prior_weight: f64,
        weights: &[f64],
    ) -> Self {
        assert!(n_choices > 0);

        if observations.is_empty() {
            // No observations: single uniform prior component
            let uniform = vec![1.0 / n_choices as f64; n_choices];
            return Self {
                component_weights: vec![uniform],
                mixture_weights: vec![1.0],
                n_choices,
            };
        }

        let n_kernels = observations.len() + 1; // observations + prior
        let base = prior_weight / n_kernels as f64;

        let mut component_weights = Vec::with_capacity(n_kernels);

        // Observation components
        for &cat in observations {
            let mut row = vec![base; n_choices];
            if cat < n_choices {
                row[cat] += 1.0;
            }
            // Normalize row
            let row_sum: f64 = row.iter().sum();
            if row_sum > 0.0 {
                for v in &mut row {
                    *v /= row_sum;
                }
            }
            component_weights.push(row);
        }

        // Prior component (uniform)
        let mut prior_row = vec![base; n_choices];
        let row_sum: f64 = prior_row.iter().sum();
        if row_sum > 0.0 {
            for v in &mut prior_row {
                *v /= row_sum;
            }
        }
        component_weights.push(prior_row);

        // Mixture weights: observation weights + prior weight, normalized
        let mut mixture_weights = Vec::with_capacity(n_kernels);
        mixture_weights.extend_from_slice(weights);
        mixture_weights.push(prior_weight);
        let total: f64 = mixture_weights.iter().sum();
        if total > 0.0 {
            for w in &mut mixture_weights {
                *w /= total;
            }
        }

        Self {
            component_weights,
            mixture_weights,
            n_choices,
        }
    }

    /// Log-probability of a given category index under this mixture.
    pub fn log_pdf(&self, index: usize) -> f64 {
        if index >= self.n_choices {
            return f64::NEG_INFINITY;
        }

        // p(x=index) = sum_k mixture_weights[k] * component_weights[k][index]
        // Use log-sum-exp for numerical stability
        let log_components: Vec<f64> = self
            .mixture_weights
            .iter()
            .zip(self.component_weights.iter())
            .filter(|(&mw, _)| mw > 0.0)
            .filter_map(|(&mw, cw)| {
                let p = cw[index];
                if p > 0.0 {
                    Some(mw.ln() + p.ln())
                } else {
                    None
                }
            })
            .collect();

        if log_components.is_empty() {
            return f64::NEG_INFINITY;
        }

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

    /// Sample a category index from this estimator.
    #[allow(dead_code)]
    pub fn sample(&self, rng: &mut impl Rng) -> usize {
        // First pick a component
        let u: f64 = rng.random();
        let mut cumulative = 0.0;
        let mut chosen_component = self.mixture_weights.len() - 1;
        for (i, &w) in self.mixture_weights.iter().enumerate() {
            cumulative += w;
            if u < cumulative {
                chosen_component = i;
                break;
            }
        }

        // Then sample from that component's categorical distribution
        let u2: f64 = rng.random();
        cumulative = 0.0;
        let cw = &self.component_weights[chosen_component];
        for (i, &p) in cw.iter().enumerate() {
            cumulative += p;
            if u2 < cumulative {
                return i;
            }
        }
        self.n_choices - 1
    }
}

/// Log-pdf of a single Gaussian component.
pub fn gaussian_log_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    -0.5 * z * z - sigma.ln() - 0.5 * std::f64::consts::TAU.ln()
}

/// Log-pdf of a truncated normal distribution.
///
/// `log_pdf_trunc(x) = log_pdf_normal(x) - log(Z)` where
/// `Z = Phi((high - mu)/sigma) - Phi((low - mu)/sigma)` is the normalizing constant.
fn truncated_gaussian_log_pdf(x: f64, mu: f64, sigma: f64, low: f64, high: f64) -> f64 {
    if x < low || x > high {
        return f64::NEG_INFINITY;
    }
    let log_pdf = gaussian_log_pdf(x, mu, sigma);
    let z = normal_cdf((high - mu) / sigma) - normal_cdf((low - mu) / sigma);
    if z <= 0.0 {
        return f64::NEG_INFINITY;
    }
    log_pdf - z.ln()
}

/// Standard normal CDF (approximation via erfc).
fn normal_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Complementary error function (Abramowitz & Stegun approximation).
fn erfc(x: f64) -> f64 {
    // Use the relationship: erfc(x) = 1 - erf(x)
    // For negative x: erfc(x) = 2 - erfc(-x)
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }

    // Horner form of a rational approximation (Abramowitz & Stegun 7.1.26)
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    poly * (-x * x).exp()
}

/// Log-sum-exp of weighted truncated Gaussian components.
fn truncated_gmm_log_pdf(
    x: f64,
    mus: &[f64],
    sigmas: &[f64],
    weights: &[f64],
    low: f64,
    high: f64,
) -> f64 {
    let log_components: Vec<f64> = mus
        .iter()
        .zip(sigmas.iter())
        .zip(weights.iter())
        .filter(|((_, _), &w)| w > 0.0)
        .map(|((&mu, &sigma), &w)| {
            w.ln() + truncated_gaussian_log_pdf(x, mu, sigma, low, high)
        })
        .collect();

    if log_components.is_empty() {
        return f64::NEG_INFINITY;
    }

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
    /// Cached multivariate samples: trial_number -> (param_name -> value).
    /// Used when `multivariate=true` to ensure all parameters are sampled jointly.
    multivariate_cache: Mutex<std::collections::HashMap<usize, std::collections::HashMap<String, f64>>>,
}

impl TpeSampler {
    /// Create a new TPE sampler with the given config and seed.
    pub fn new(config: TpeSamplerConfig, seed: u64) -> Self {
        Self {
            config,
            random_sampler: super::RandomSampler::new(seed),
            rng: Mutex::new(StdRng::seed_from_u64(seed.wrapping_add(1))),
            multivariate_cache: Mutex::new(std::collections::HashMap::new()),
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

    /// Compute a "lie" value for running trials based on completed trial values.
    /// Uses the worst completed value (pessimistic constant liar).
    #[allow(dead_code)]
    fn compute_lie_value(&self, study: &Study) -> f64 {
        let completed_values: Vec<f64> = study
            .trials()
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .filter_map(|t| t.value)
            .collect();

        if completed_values.is_empty() {
            return 0.0;
        }

        // Pessimistic: use the worst value
        match study.direction() {
            crate::study::Direction::Minimize => {
                completed_values
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max)
            }
            crate::study::Direction::Maximize => {
                completed_values
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min)
            }
        }
    }

    /// Get effective trials for TPE, including running trials with imputed values
    /// when constant_liar is enabled.
    #[allow(dead_code)]
    fn get_effective_trials<'a>(
        &self,
        study: &'a Study,
        param_name: &str,
    ) -> Vec<crate::trial::FrozenTrial> {
        let mut trials: Vec<crate::trial::FrozenTrial> = study
            .trials()
            .iter()
            .filter(|t| t.state == TrialState::Complete && t.params.contains_key(param_name))
            .cloned()
            .collect();

        if self.config.constant_liar {
            let lie_value = self.compute_lie_value(study);
            for t in study.trials() {
                if t.state == TrialState::Running && t.params.contains_key(param_name) {
                    let mut imputed = t.clone();
                    imputed.value = Some(lie_value);
                    imputed.state = TrialState::Complete;
                    trials.push(imputed);
                }
            }
        }

        trials
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

    /// Split completed trials into good/bad for multi-objective studies using
    /// Pareto dominance (MOTPE). Fills the "good" group front-by-front until
    /// gamma(n) trials are reached.
    fn split_trials_multi(
        &self,
        completed: &[&crate::trial::FrozenTrial],
        directions: &[crate::study::Direction],
    ) -> (Vec<usize>, Vec<usize>) {
        let n = completed.len();
        let n_good = self.gamma(n);

        // Get objective values
        let values: Vec<Vec<f64>> = completed
            .iter()
            .map(|t| {
                t.values
                    .clone()
                    .unwrap_or_else(|| t.value.map(|v| vec![v]).unwrap_or_default())
            })
            .collect();

        let fronts = crate::multi_objective::non_dominated_sort(&values, directions);

        let mut good = Vec::new();
        for front in &fronts {
            if good.len() >= n_good {
                break;
            }
            for &idx in front {
                if good.len() >= n_good {
                    break;
                }
                good.push(idx);
            }
        }

        let good_set: std::collections::HashSet<usize> = good.iter().copied().collect();
        let bad: Vec<usize> = (0..n).filter(|i| !good_set.contains(i)).collect();

        (good, bad)
    }

    /// Choose the right split strategy based on single/multi-objective.
    fn split_trials_auto(
        &self,
        study: &Study,
        completed: &[&crate::trial::FrozenTrial],
    ) -> (Vec<usize>, Vec<usize>) {
        if study.is_multi_objective() {
            self.split_trials_multi(completed, study.directions())
        } else {
            self.split_trials(study, completed)
        }
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

        let (good_indices, bad_indices) = self.split_trials_auto(study, &completed);

        // Extract parameter values in original space
        let good_values: Vec<f64> = good_indices
            .iter()
            .map(|&i| completed[i].params[param_name])
            .collect();

        let bad_values: Vec<f64> = bad_indices
            .iter()
            .map(|&i| completed[i].params[param_name])
            .collect();

        let good_w = default_weights(good_values.len());
        let bad_w = default_weights(bad_values.len());

        let prior_weight = if self.config.consider_prior {
            self.config.prior_weight
        } else {
            0.0
        };

        let l = ParzenEstimator::new(
            &good_values,
            dist.low,
            dist.high,
            prior_weight,
            self.config.consider_magic_clip,
            self.config.consider_endpoints,
            &good_w,
        );

        let g = ParzenEstimator::new(
            &bad_values,
            dist.low,
            dist.high,
            prior_weight,
            self.config.consider_magic_clip,
            self.config.consider_endpoints,
            &bad_w,
        );

        // Draw candidates from l(x), evaluate EI = l(x)/g(x), pick best
        let mut rng = self.rng.lock().unwrap();
        let mut best_candidate = 0.5 * (dist.low + dist.high);
        let mut best_ei = f64::NEG_INFINITY;

        for _ in 0..self.config.n_ei_candidates {
            let candidate = l.sample(&mut *rng);
            let l_log = l.log_pdf(candidate);
            let g_log = g.log_pdf(candidate);
            let ei = l_log - g_log;

            if ei > best_ei {
                best_ei = ei;
                best_candidate = candidate;
            }
        }

        // Apply step quantization if needed
        if let Some(step) = dist.step {
            best_candidate = (((best_candidate - dist.low) / step).round() * step + dist.low)
                .clamp(dist.low, dist.high);
        }

        best_candidate
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

        let (good_indices, bad_indices) = self.split_trials_auto(study, &completed);

        // Extract parameter values in original space
        let good_values: Vec<f64> = good_indices
            .iter()
            .map(|&i| completed[i].params[param_name])
            .collect();

        let bad_values: Vec<f64> = bad_indices
            .iter()
            .map(|&i| completed[i].params[param_name])
            .collect();

        let good_w = default_weights(good_values.len());
        let bad_w = default_weights(bad_values.len());

        let prior_weight = if self.config.consider_prior {
            self.config.prior_weight
        } else {
            0.0
        };

        let low = dist.low as f64;
        let high = dist.high as f64;

        let l = ParzenEstimator::new(
            &good_values, low, high, prior_weight,
            self.config.consider_magic_clip, self.config.consider_endpoints,
            &good_w,
        );

        let g = ParzenEstimator::new(
            &bad_values, low, high, prior_weight,
            self.config.consider_magic_clip, self.config.consider_endpoints,
            &bad_w,
        );

        let mut rng = self.rng.lock().unwrap();
        let mut best_candidate = 0.5 * (low + high);
        let mut best_ei = f64::NEG_INFINITY;

        for _ in 0..self.config.n_ei_candidates {
            let candidate = l.sample(&mut *rng);
            let l_log = l.log_pdf(candidate);
            let g_log = g.log_pdf(candidate);
            let ei = l_log - g_log;

            if ei > best_ei {
                best_ei = ei;
                best_candidate = candidate;
            }
        }

        // Round to nearest integer, respecting step
        let step = dist.step.unwrap_or(1) as f64;
        let result = (((best_candidate - low) / step).round() * step + low).clamp(low, high);
        result.round()
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

        let (good_indices, bad_indices) = self.split_trials_auto(study, &completed);

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

        let pw = if self.config.consider_prior {
            self.config.prior_weight
        } else {
            0.0
        };

        let l = CategoricalParzenEstimator::new(&good_cats, n_choices, pw, &good_w);
        let g = CategoricalParzenEstimator::new(&bad_cats, n_choices, pw, &bad_w);

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

    /// Sample all parameters jointly for multivariate TPE.
    ///
    /// Builds per-parameter Parzen estimators for good/bad groups, then samples
    /// `n_ei_candidates` joint candidates from the product of good estimators.
    /// Picks the candidate maximizing the sum of log(l(x_i)) - log(g(x_i)).
    fn sample_multivariate(
        &self,
        study: &Study,
        _trial_number: usize,
    ) -> std::collections::HashMap<String, f64> {
        // Collect all parameter names and distributions from completed trials
        let completed: Vec<&crate::trial::FrozenTrial> = study
            .trials()
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .collect();

        if completed.is_empty() {
            return std::collections::HashMap::new();
        }

        // Find all parameter names (union across all completed trials)
        let mut param_names: Vec<String> = completed
            .last()
            .unwrap()
            .params
            .keys()
            .cloned()
            .collect();
        param_names.sort();

        let n = completed.len();
        if n < self.config.n_startup_trials {
            return std::collections::HashMap::new();
        }

        let (good_indices, bad_indices) = self.split_trials_auto(study, &completed);

        let prior_weight = if self.config.consider_prior {
            self.config.prior_weight
        } else {
            0.0
        };

        // Build per-parameter estimators
        let mut l_estimators: Vec<ParzenEstimator> = Vec::new();
        let mut g_estimators: Vec<ParzenEstimator> = Vec::new();
        let mut param_lows: Vec<f64> = Vec::new();
        let mut param_highs: Vec<f64> = Vec::new();

        for name in &param_names {
            let good_values: Vec<f64> = good_indices
                .iter()
                .filter_map(|&i| completed[i].params.get(name).copied())
                .collect();
            let bad_values: Vec<f64> = bad_indices
                .iter()
                .filter_map(|&i| completed[i].params.get(name).copied())
                .collect();

            if good_values.is_empty() || bad_values.is_empty() {
                continue;
            }

            // Infer bounds from data
            let all_values: Vec<f64> = completed
                .iter()
                .filter_map(|t| t.params.get(name).copied())
                .collect();
            let low = all_values.iter().copied().fold(f64::INFINITY, f64::min);
            let high = all_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let range = (high - low).max(1e-10);
            let low = low - 0.1 * range;
            let high = high + 0.1 * range;

            let good_w = default_weights(good_values.len());
            let bad_w = default_weights(bad_values.len());

            l_estimators.push(ParzenEstimator::new(
                &good_values, low, high, prior_weight,
                self.config.consider_magic_clip, self.config.consider_endpoints,
                &good_w,
            ));
            g_estimators.push(ParzenEstimator::new(
                &bad_values, low, high, prior_weight,
                self.config.consider_magic_clip, self.config.consider_endpoints,
                &bad_w,
            ));
            param_lows.push(low);
            param_highs.push(high);
        }

        if l_estimators.is_empty() {
            return std::collections::HashMap::new();
        }

        // Sample n_ei_candidates joint candidates from l(x), pick best joint EI
        let mut rng = self.rng.lock().unwrap();
        let n_params = l_estimators.len();
        let mut best_candidate: Vec<f64> = vec![0.0; n_params];
        let mut best_ei = f64::NEG_INFINITY;

        for _ in 0..self.config.n_ei_candidates {
            let mut candidate = Vec::with_capacity(n_params);
            let mut joint_ei = 0.0;

            for j in 0..n_params {
                let x = l_estimators[j].sample(&mut *rng);
                let l_log = l_estimators[j].log_pdf(x);
                let g_log = g_estimators[j].log_pdf(x);
                joint_ei += l_log - g_log;
                candidate.push(x);
            }

            if joint_ei > best_ei {
                best_ei = joint_ei;
                best_candidate = candidate;
            }
        }

        // Build result map
        let mut result = std::collections::HashMap::new();
        let valid_names: Vec<&String> = param_names.iter()
            .filter(|name| {
                let good_values: Vec<f64> = good_indices
                    .iter()
                    .filter_map(|&i| completed[i].params.get(*name).copied())
                    .collect();
                let bad_values: Vec<f64> = bad_indices
                    .iter()
                    .filter_map(|&i| completed[i].params.get(*name).copied())
                    .collect();
                !good_values.is_empty() && !bad_values.is_empty()
            })
            .collect();

        for (j, name) in valid_names.iter().enumerate() {
            if j < best_candidate.len() {
                result.insert((*name).clone(), best_candidate[j]);
            }
        }

        result
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
        // Multivariate mode: sample all params jointly on first request, cache rest
        if self.config.multivariate {
            // Check cache first
            {
                let cache = self.multivariate_cache.lock().unwrap();
                if let Some(trial_params) = cache.get(&_trial.number) {
                    if let Some(&value) = trial_params.get(param_name) {
                        return value;
                    }
                }
            }

            // Not cached — try to do joint sampling
            let completed_count = study
                .trials()
                .iter()
                .filter(|t| t.state == TrialState::Complete)
                .count();

            if completed_count >= self.config.n_startup_trials {
                let joint_sample = self.sample_multivariate(study, _trial.number);
                if let Some(&value) = joint_sample.get(param_name) {
                    let mut cache = self.multivariate_cache.lock().unwrap();
                    cache.insert(_trial.number, joint_sample);
                    return value;
                }
            }

            // Fall through to univariate if joint sampling didn't produce this param
        }

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
        let empty: Vec<f64> = vec![];
        assert_eq!(default_weights(0), empty);
        assert_eq!(default_weights(1), vec![1.0_f64]);
        assert_eq!(default_weights(3), vec![1.0_f64, 1.0, 1.0]);

        let w = default_weights(30);
        assert_eq!(w.len(), 30);
        // First 5 elements are the ramp (linspace(1/30, 1.0, 5))
        assert!((w[0] - 1.0 / 30.0).abs() < 1e-10);
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
        let pe = ParzenEstimator::new(&values, 0.0, 1.0, 1.0, true, false, &[1.0; 5]);
        for &s in &pe.sigmas {
            assert!(s > 0.0, "sigma must be positive, got {s}");
            assert!(s.is_finite(), "sigma must be finite");
        }
    }

    #[test]
    fn test_parzen_estimator_log_pdf_stable() {
        let values = vec![0.2, 0.4, 0.6, 0.8];
        let pe = ParzenEstimator::new(&values, 0.0, 1.0, 1.0, true, false, &[1.0; 4]);

        for i in 0..100 {
            let x = i as f64 / 100.0;
            let lp = pe.log_pdf(x);
            assert!(!lp.is_nan(), "NaN at x={x}");
        }
    }

    #[test]
    fn test_parzen_estimator_sample() {
        let values = vec![0.3, 0.5, 0.7];
        let pe = ParzenEstimator::new(&values, 0.0, 1.0, 1.0, true, false, &[1.0; 3]);

        let mut rng = StdRng::seed_from_u64(42);
        for _ in 0..100 {
            let s = pe.sample(&mut rng);
            assert!(s.is_finite(), "sample must be finite");
            assert!(s >= 0.0 && s <= 1.0, "sample {s} not in [0,1]");
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
        // 6 components: 5 observations + 1 prior
        assert_eq!(pe.component_weights.len(), 6);
        assert_eq!(pe.mixture_weights.len(), 6);
        // Category 0 observed most, so log_pdf(0) > log_pdf(1)
        assert!(pe.log_pdf(0) > pe.log_pdf(1));
        assert!(pe.log_pdf(0) > pe.log_pdf(2));
        // All log_pdfs are finite
        for i in 0..3 {
            let lp = pe.log_pdf(i);
            assert!(lp.is_finite(), "log_pdf({i}) should be finite");
        }
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
