//! Gaussian Process (GP) sampler for Bayesian optimization.
//!
//! Recommended for low-budget optimization (<200 trials) with continuous parameters.
//! Uses a multi-dimensional Matern 5/2 kernel with Automatic Relevance Determination
//! (ARD) and Expected Improvement (EI) as the acquisition function.
//!
//! The GP operates jointly over all continuous parameters, modeling correlations
//! between dimensions. Categorical parameters fall back to random sampling.

use rand::prelude::*;
use std::collections::HashMap;
use std::sync::Mutex;

use crate::distributions::{
    int_transform_from_internal, int_transform_to_internal, transform_from_internal,
    transform_to_internal, Distribution,
};
use crate::study::Study;
use crate::trial::{Trial, TrialState};

use super::Sampler;

/// Configuration for the GP sampler.
#[derive(Debug, Clone)]
pub struct GpConfig {
    /// Number of startup trials using random sampling.
    pub n_startup_trials: usize,
    /// Number of random candidates for acquisition function optimization.
    pub n_candidates: usize,
}

impl Default for GpConfig {
    fn default() -> Self {
        Self {
            n_startup_trials: 10,
            n_candidates: 2000,
        }
    }
}

/// Cached GP suggestion for a single trial so all parameters come from one joint point.
struct CachedSuggestion {
    /// Trial number this cache is for.
    trial_number: usize,
    /// Suggested values keyed by parameter name (in external/original space).
    values: HashMap<String, f64>,
}

/// Gaussian Process sampler for Bayesian optimization.
///
/// Uses a multi-dimensional Matern 5/2 kernel with ARD (per-dimension length
/// scales) and Expected Improvement (EI) acquisition function. Best suited for
/// optimization with fewer than 200 trials and continuous parameters.
pub struct GpSampler {
    config: GpConfig,
    random_sampler: super::RandomSampler,
    rng: Mutex<StdRng>,
    /// Cache for the current trial's joint suggestion.
    cache: Mutex<Option<CachedSuggestion>>,
}

impl GpSampler {
    pub fn new(config: GpConfig, seed: u64) -> Self {
        Self {
            config,
            random_sampler: super::RandomSampler::new(seed),
            rng: Mutex::new(StdRng::seed_from_u64(seed.wrapping_add(3))),
            cache: Mutex::new(None),
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        Self::new(GpConfig::default(), seed)
    }

    fn to_internal(value: f64, distribution: &Distribution) -> f64 {
        match distribution {
            Distribution::Float(d) => transform_to_internal(value, d),
            Distribution::Int(d) => int_transform_to_internal(value as i64, d),
            Distribution::Categorical(_) => value,
        }
    }

    fn from_internal(internal: f64, distribution: &Distribution) -> f64 {
        match distribution {
            Distribution::Float(d) => transform_from_internal(internal, d),
            Distribution::Int(d) => int_transform_from_internal(internal, d) as f64,
            Distribution::Categorical(_) => internal,
        }
    }

    /// Collect the sorted list of continuous parameter names and their distributions
    /// from completed trials.
    fn collect_param_info(
        completed: &[&crate::trial::FrozenTrial],
    ) -> Vec<(String, Distribution)> {
        // Collect all parameter names that appear in completed trials, excluding categoricals.
        let mut param_set: HashMap<String, Distribution> = HashMap::new();
        for t in completed {
            for (name, dist) in &t.distributions {
                if !matches!(dist, Distribution::Categorical(_)) {
                    param_set.entry(name.clone()).or_insert_with(|| dist.clone());
                }
            }
        }
        // For trials that don't have distributions stored, try to infer from param presence.
        // We need distributions for the transform. If not available, skip.
        let mut params: Vec<(String, Distribution)> = param_set.into_iter().collect();
        params.sort_by(|a, b| a.0.cmp(&b.0));
        params
    }

    /// Build the design matrix X (n x d) and objective vector y (n) from completed trials.
    /// Only includes trials that have ALL the specified parameters.
    fn build_data(
        completed: &[&crate::trial::FrozenTrial],
        param_info: &[(String, Distribution)],
        direction: crate::study::Direction,
    ) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut xs = Vec::new();
        let mut ys = Vec::new();

        for t in completed {
            // Check if trial has all parameters
            let has_all = param_info.iter().all(|(name, _)| t.params.contains_key(name));
            if !has_all {
                continue;
            }

            let row: Vec<f64> = param_info
                .iter()
                .map(|(name, dist)| Self::to_internal(t.params[name], dist))
                .collect();

            let v = t.value.unwrap_or(0.0);
            let y = match direction {
                crate::study::Direction::Minimize => v,
                crate::study::Direction::Maximize => -v,
            };

            xs.push(row);
            ys.push(y);
        }

        (xs, ys)
    }

    /// Run the joint GP optimization and return the best candidate point in internal [0,1]^d space.
    fn optimize_joint(
        &self,
        xs: &[Vec<f64>],
        ys_raw: &[f64],
        d: usize,
    ) -> Option<Vec<f64>> {
        let n = xs.len();
        if n == 0 || d == 0 {
            return None;
        }

        // Standardize ys for numerical stability
        let y_mean: f64 = ys_raw.iter().sum::<f64>() / n as f64;
        let y_std = {
            let var: f64 = ys_raw.iter().map(|y| (y - y_mean).powi(2)).sum::<f64>() / n as f64;
            var.sqrt().max(1e-6)
        };
        let ys: Vec<f64> = ys_raw.iter().map(|y| (y - y_mean) / y_std).collect();

        let best_f = ys.iter().copied().fold(f64::INFINITY, f64::min);

        // Hyperparameter selection: try different length scale configurations.
        // Use heuristic: per-dimension length scales based on data range, with a global multiplier.
        let per_dim_ranges: Vec<f64> = (0..d)
            .map(|j| {
                let col: Vec<f64> = xs.iter().map(|x| x[j]).collect();
                let min = col.iter().copied().fold(f64::INFINITY, f64::min);
                let max = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                (max - min).max(1e-6)
            })
            .collect();

        let multipliers = [0.1, 0.2, 0.5, 1.0, 2.0];
        let noise = 1e-4;

        let mut best_lml = f64::NEG_INFINITY;
        let mut best_alpha: Vec<f64> = Vec::new();
        let mut best_l_mat: Vec<Vec<f64>> = Vec::new();
        let mut best_lengthscales: Vec<f64> = Vec::new();

        for &mult in &multipliers {
            let lengthscales: Vec<f64> = per_dim_ranges.iter().map(|r| r * mult).collect();
            let k = build_kernel_matrix_nd(xs, &lengthscales, noise);
            if let Some(l) = cholesky(&k) {
                let z = forward_solve(&l, &ys);
                let alpha = backward_solve(&l, &z);
                let lml = log_marginal_likelihood(&l, &alpha, &ys);
                if lml > best_lml {
                    best_lml = lml;
                    best_alpha = alpha;
                    best_l_mat = l;
                    best_lengthscales = lengthscales;
                }
            }
        }

        if best_alpha.is_empty() {
            return None;
        }

        // Optimize acquisition function by random search over [0,1]^d
        let mut rng = self.rng.lock().unwrap();
        let mut best_x = vec![0.5; d];
        let mut best_ei = f64::NEG_INFINITY;

        for _ in 0..self.config.n_candidates {
            let x_cand: Vec<f64> = (0..d).map(|_| rng.random::<f64>()).collect();
            let (mean, var) =
                gp_predict_nd(&x_cand, xs, &best_alpha, &best_l_mat, &best_lengthscales);
            let ei = expected_improvement(mean, var, best_f);
            if ei > best_ei {
                best_ei = ei;
                best_x = x_cand;
            }
        }

        Some(best_x)
    }
}

// ============================================================================
// Kernel functions
// ============================================================================

/// Matern 5/2 kernel for a single dimension: matern52(r) where r = |x1 - x2| / length_scale.
fn matern52_1d(x1: f64, x2: f64, length_scale: f64) -> f64 {
    let r = (x1 - x2).abs() / length_scale.max(1e-10);
    let sqrt5_r = 5.0_f64.sqrt() * r;
    (1.0 + sqrt5_r + 5.0 / 3.0 * r * r) * (-sqrt5_r).exp()
}

/// ARD Matern 5/2 kernel: product of per-dimension Matern 5/2 kernels.
/// k(x, x') = product_i matern52(|x_i - x'_i| / l_i)
fn ard_matern52(x1: &[f64], x2: &[f64], lengthscales: &[f64]) -> f64 {
    debug_assert_eq!(x1.len(), x2.len());
    debug_assert_eq!(x1.len(), lengthscales.len());
    let mut product = 1.0;
    for i in 0..x1.len() {
        product *= matern52_1d(x1[i], x2[i], lengthscales[i]);
    }
    product
}

/// Build the n x n kernel matrix K for d-dimensional observations.
fn build_kernel_matrix_nd(
    xs: &[Vec<f64>],
    lengthscales: &[f64],
    noise: f64,
) -> Vec<Vec<f64>> {
    let n = xs.len();
    let mut k = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in i..n {
            let val = ard_matern52(&xs[i], &xs[j], lengthscales);
            k[i][j] = val;
            k[j][i] = val;
        }
        k[i][i] += noise;
    }
    k
}

// ============================================================================
// Linear algebra
// ============================================================================

/// Cholesky decomposition of a positive-definite matrix. Returns lower triangular L.
fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }
            if i == j {
                let diag = a[i][i] - sum;
                if diag <= 0.0 {
                    return None; // Not positive definite
                }
                l[i][j] = diag.sqrt();
            } else {
                l[i][j] = (a[i][j] - sum) / l[j][j].max(1e-20);
            }
        }
    }
    Some(l)
}

/// Solve L * x = b via forward substitution (L is lower triangular).
fn forward_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / l[i][i].max(1e-20);
    }
    x
}

/// Solve L^T * x = b via backward substitution (L is lower triangular).
fn backward_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += l[j][i] * x[j]; // L^T[i][j] = L[j][i]
        }
        x[i] = (b[i] - sum) / l[i][i].max(1e-20);
    }
    x
}

// ============================================================================
// GP prediction
// ============================================================================

/// Compute GP posterior mean and variance at a d-dimensional test point.
fn gp_predict_nd(
    x_star: &[f64],
    xs: &[Vec<f64>],
    alpha: &[f64], // K_inv @ y
    l: &[Vec<f64>],
    lengthscales: &[f64],
) -> (f64, f64) {
    let n = xs.len();

    // k_star[i] = k(x_star, xs[i])
    let k_star: Vec<f64> = (0..n)
        .map(|i| ard_matern52(x_star, &xs[i], lengthscales))
        .collect();

    // Mean: k_star^T @ alpha
    let mean: f64 = k_star.iter().zip(alpha.iter()).map(|(k, a)| k * a).sum();

    // Variance: k(x_star, x_star) - k_star^T @ K_inv @ k_star
    let k_ss = 1.0; // ard_matern52(x_star, x_star, ...) = 1.0
    let v = forward_solve(l, &k_star);
    let var = (k_ss - v.iter().map(|vi| vi * vi).sum::<f64>()).max(1e-10);

    (mean, var)
}

// ============================================================================
// Acquisition function
// ============================================================================

/// Expected Improvement acquisition function.
fn expected_improvement(mean: f64, var: f64, best_f: f64) -> f64 {
    let std = var.sqrt();
    if std < 1e-12 {
        return 0.0;
    }
    let z = (best_f - mean) / std;
    // EI = (best_f - mean) * Phi(z) + std * phi(z)
    let phi = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let big_phi = 0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2));
    (best_f - mean) * big_phi + std * phi
}

/// Error function approximation (Abramowitz & Stegun 7.1.26).
fn erf_approx(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736
                + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Compute marginal log-likelihood for hyperparameter selection.
fn log_marginal_likelihood(l: &[Vec<f64>], alpha: &[f64], ys: &[f64]) -> f64 {
    let n = ys.len();
    // -0.5 * y^T K^{-1} y - sum(log(diag(L))) - n/2 * log(2pi)
    let data_fit: f64 = -0.5 * ys.iter().zip(alpha.iter()).map(|(y, a)| y * a).sum::<f64>();
    let complexity: f64 = -l.iter().enumerate().map(|(i, row)| row[i].ln()).sum::<f64>();
    let constant = -0.5 * n as f64 * (2.0 * std::f64::consts::PI).ln();
    data_fit + complexity + constant
}

// ============================================================================
// Sampler implementation
// ============================================================================

impl Sampler for GpSampler {
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

        let completed: Vec<_> = study
            .trials()
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .collect();

        if completed.len() < self.config.n_startup_trials {
            return self
                .random_sampler
                .sample(study, trial, param_name, distribution);
        }

        // Check if we already have a cached suggestion for this trial.
        {
            let cache = self.cache.lock().unwrap();
            if let Some(ref cached) = *cache {
                if cached.trial_number == trial.number {
                    if let Some(&val) = cached.values.get(param_name) {
                        return val;
                    }
                }
            }
        }

        // Collect parameter info from completed trials.
        let param_info = Self::collect_param_info(&completed);

        // If no continuous parameters found in history (or only current param),
        // ensure the current param's distribution is included.
        let mut param_info_with_current = param_info.clone();
        if !param_info_with_current.iter().any(|(name, _)| name == param_name) {
            param_info_with_current.push((param_name.to_string(), distribution.clone()));
            param_info_with_current.sort_by(|a, b| a.0.cmp(&b.0));
        }

        let d = param_info_with_current.len();

        // Build data matrix using the full parameter set.
        let (xs, ys) = Self::build_data(&completed, &param_info_with_current, study.direction());

        // If not enough data points with all params, fall back to random.
        if xs.len() < self.config.n_startup_trials || d == 0 {
            return self
                .random_sampler
                .sample(study, trial, param_name, distribution);
        }

        // Run the joint GP optimization.
        let best_point = match self.optimize_joint(&xs, &ys, d) {
            Some(p) => p,
            None => {
                return self
                    .random_sampler
                    .sample(study, trial, param_name, distribution);
            }
        };

        // Convert the internal point back to external space and cache it.
        let mut values = HashMap::new();
        for (i, (name, dist)) in param_info_with_current.iter().enumerate() {
            let external = Self::from_internal(best_point[i], dist);
            values.insert(name.clone(), external);
        }

        let result = values.get(param_name).copied().unwrap_or_else(|| {
            self.random_sampler
                .sample(study, trial, param_name, distribution)
        });

        // Store in cache for subsequent parameter requests in the same trial.
        {
            let mut cache = self.cache.lock().unwrap();
            *cache = Some(CachedSuggestion {
                trial_number: trial.number,
                values,
            });
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_gp_float_in_bounds() {
        let sampler = GpSampler::with_seed(42);
        let dist = Distribution::Float(crate::distributions::FloatDistribution::new(
            0.0, 10.0, false, None,
        ));
        let mut study = Study::new_default();

        for i in 0..15 {
            let value = (i as f64) * 0.5 + 1.0;
            let mut params = HashMap::new();
            params.insert("x".to_string(), value);
            let mut trial = crate::trial::FrozenTrial::new(i);
            trial.state = TrialState::Complete;
            trial.value = Some((value - 3.0).powi(2));
            trial.params = params;
            trial.distributions.insert(
                "x".to_string(),
                dist.clone(),
            );
            study.add_trial(trial);
        }

        let trial = Trial::new(15);
        let v = sampler.sample(&study, &trial, "x", &dist);
        assert!(v >= 0.0 && v <= 10.0, "GP suggested {v} outside [0, 10]");
    }

    #[test]
    fn test_gp_during_startup_uses_random() {
        let sampler = GpSampler::with_seed(42);
        let dist = Distribution::Float(crate::distributions::FloatDistribution::new(
            0.0, 1.0, false, None,
        ));
        let study = Study::new_default();
        let trial = Trial::new(0);

        let v = sampler.sample(&study, &trial, "x", &dist);
        assert!(v >= 0.0 && v <= 1.0);
    }

    #[test]
    fn test_gp_converges_toward_optimum() {
        let sampler = GpSampler::new(
            GpConfig {
                n_startup_trials: 5,
                n_candidates: 500,
            },
            42,
        );
        let dist = Distribution::Float(crate::distributions::FloatDistribution::new(
            -5.0, 5.0, false, None,
        ));
        let mut study = Study::new_default();

        for _ in 0..40 {
            let trial = Trial::new(study.trials().len());
            let x = sampler.sample(&study, &trial, "x", &dist);
            let obj = (x - 1.0).powi(2);
            let mut params = HashMap::new();
            params.insert("x".to_string(), x);
            let mut frozen = crate::trial::FrozenTrial::new(study.trials().len());
            frozen.state = TrialState::Complete;
            frozen.value = Some(obj);
            frozen.params = params;
            frozen.distributions
                .insert("x".to_string(), dist.clone());
            study.add_trial(frozen);
        }

        let best = study.best_value().unwrap();
        assert!(best < 2.0, "GP should converge, best={best}");
    }

    #[test]
    fn test_gp_2d_correlated_optimization() {
        // Optimize f(x, y) = (x + y - 1)^2 + (x - y)^2
        // Minimum at x=0.5, y=0.5 with f=0.
        // This tests that the multi-dimensional GP captures correlations.
        let sampler = GpSampler::new(
            GpConfig {
                n_startup_trials: 8,
                n_candidates: 1000,
            },
            123,
        );
        let dist_x = Distribution::Float(crate::distributions::FloatDistribution::new(
            -3.0, 3.0, false, None,
        ));
        let dist_y = Distribution::Float(crate::distributions::FloatDistribution::new(
            -3.0, 3.0, false, None,
        ));
        let mut study = Study::new_default();

        for _ in 0..50 {
            let trial_num = study.trials().len();
            let trial = Trial::new(trial_num);
            let x = sampler.sample(&study, &trial, "x", &dist_x);
            let y = sampler.sample(&study, &trial, "y", &dist_y);
            let obj = (x + y - 1.0).powi(2) + (x - y).powi(2);

            let mut params = HashMap::new();
            params.insert("x".to_string(), x);
            params.insert("y".to_string(), y);
            let mut frozen = crate::trial::FrozenTrial::new(trial_num);
            frozen.state = TrialState::Complete;
            frozen.value = Some(obj);
            frozen.params = params;
            frozen
                .distributions
                .insert("x".to_string(), dist_x.clone());
            frozen
                .distributions
                .insert("y".to_string(), dist_y.clone());
            study.add_trial(frozen);
        }

        let best = study.best_value().unwrap();
        assert!(
            best < 3.0,
            "2D GP should find reasonable minimum, best={best}"
        );
    }

    #[test]
    fn test_cholesky_3x3() {
        let a = vec![
            vec![4.0, 2.0, 0.0],
            vec![2.0, 5.0, 1.0],
            vec![0.0, 1.0, 3.0],
        ];
        let l = cholesky(&a).expect("Cholesky should succeed");
        // Verify L * L^T = A
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += l[i][k] * l[j][k];
                }
                assert!(
                    (sum - a[i][j]).abs() < 1e-10,
                    "Cholesky decomposition incorrect at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_ard_matern52_self() {
        // k(x, x) should be 1.0 for any point
        let x = vec![0.3, 0.7, 0.5];
        let ls = vec![0.2, 0.3, 0.5];
        let val = ard_matern52(&x, &x, &ls);
        assert!(
            (val - 1.0).abs() < 1e-10,
            "ARD kernel at same point should be 1.0, got {val}"
        );
    }

    #[test]
    fn test_ard_matern52_symmetry() {
        let x1 = vec![0.3, 0.7];
        let x2 = vec![0.6, 0.2];
        let ls = vec![0.5, 0.5];
        let k1 = ard_matern52(&x1, &x2, &ls);
        let k2 = ard_matern52(&x2, &x1, &ls);
        assert!(
            (k1 - k2).abs() < 1e-10,
            "ARD kernel should be symmetric"
        );
    }

    #[test]
    fn test_ard_matern52_relevance() {
        // With a very large length scale in dim 1, the kernel should be insensitive
        // to differences in dim 1, but sensitive to dim 0.
        let x1 = vec![0.0, 0.0];
        let x2 = vec![0.0, 1.0]; // differ only in dim 1
        let x3 = vec![1.0, 0.0]; // differ only in dim 0

        let ls_ignore_dim1 = vec![0.1, 100.0]; // large ls for dim 1
        let k12 = ard_matern52(&x1, &x2, &ls_ignore_dim1);
        let k13 = ard_matern52(&x1, &x3, &ls_ignore_dim1);

        // k(x1, x2) should be close to 1 (dim 1 difference ignored)
        assert!(
            k12 > 0.99,
            "With large ls in dim 1, k12 should be ~1.0, got {k12}"
        );
        // k(x1, x3) should be much smaller (dim 0 difference matters)
        assert!(
            k13 < 0.5,
            "With small ls in dim 0, k13 should be small, got {k13}"
        );
    }

    #[test]
    fn test_ei_at_best_is_zero() {
        // When mean equals best_f and variance is 0, EI should be 0
        let ei = expected_improvement(1.0, 1e-20, 1.0);
        assert!(ei.abs() < 1e-6, "EI at best with no var should be ~0, got {ei}");
    }

    #[test]
    fn test_ei_positive_when_improving() {
        // Mean below best_f with positive variance should give positive EI
        let ei = expected_improvement(-1.0, 1.0, 0.0);
        assert!(ei > 0.0, "EI should be positive when mean is below best");
    }

    #[test]
    fn test_matern52_1d_self() {
        // k(x, x) should be 1.0
        assert!((matern52_1d(0.5, 0.5, 0.2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matern52_1d_symmetry() {
        let k1 = matern52_1d(0.3, 0.7, 0.5);
        let k2 = matern52_1d(0.7, 0.3, 0.5);
        assert!((k1 - k2).abs() < 1e-10);
    }
}
