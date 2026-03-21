//! Gaussian Process (GP) sampler for Bayesian optimization.
//!
//! Recommended for low-budget optimization (<200 trials) with continuous parameters.
//! Uses a Matérn 5/2 kernel with automatic relevance determination (ARD) and
//! Expected Improvement (EI) as the acquisition function.

use rand::prelude::*;
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

/// Gaussian Process sampler for Bayesian optimization.
///
/// Uses a Matérn 5/2 kernel with isotropic length scale and Expected
/// Improvement (EI) acquisition function. Best suited for optimization
/// with fewer than 200 trials and continuous parameters.
pub struct GpSampler {
    config: GpConfig,
    random_sampler: super::RandomSampler,
    rng: Mutex<StdRng>,
}

impl GpSampler {
    pub fn new(config: GpConfig, seed: u64) -> Self {
        Self {
            config,
            random_sampler: super::RandomSampler::new(seed),
            rng: Mutex::new(StdRng::seed_from_u64(seed.wrapping_add(3))),
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
}

/// Matérn 5/2 kernel: k(r) = (1 + √5·r + 5/3·r²) · exp(-√5·r)
/// where r = |x1 - x2| / length_scale
fn matern52(x1: f64, x2: f64, length_scale: f64) -> f64 {
    let r = ((x1 - x2).abs()) / length_scale.max(1e-10);
    let sqrt5_r = 5.0_f64.sqrt() * r;
    (1.0 + sqrt5_r + 5.0 / 3.0 * r * r) * (-sqrt5_r).exp()
}

/// Build the kernel matrix K[i,j] for a set of 1D observations.
fn build_kernel_matrix(
    xs: &[f64],
    length_scale: f64,
    noise: f64,
) -> Vec<Vec<f64>> {
    let n = xs.len();
    let mut k = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            k[i][j] = matern52(xs[i], xs[j], length_scale);
            if i == j {
                k[i][j] += noise;
            }
        }
    }
    k
}

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

/// Solve L·x = b via forward substitution (L is lower triangular).
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

/// Solve Lᵀ·x = b via backward substitution (L is lower triangular).
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

/// Compute GP posterior mean and variance at a test point.
fn gp_predict(
    x_star: f64,
    xs: &[f64],
    alpha: &[f64], // K_inv @ y
    l: &[Vec<f64>],
    length_scale: f64,
) -> (f64, f64) {
    let n = xs.len();

    // k_star[i] = k(x_star, xs[i])
    let k_star: Vec<f64> = (0..n)
        .map(|i| matern52(x_star, xs[i], length_scale))
        .collect();

    // Mean: k_star^T @ alpha
    let mean: f64 = k_star.iter().zip(alpha.iter()).map(|(k, a)| k * a).sum();

    // Variance: k(x_star, x_star) - k_star^T @ K_inv @ k_star
    let k_ss = 1.0; // matern52(x_star, x_star) = 1.0
    let v = forward_solve(l, &k_star);
    let var = (k_ss - v.iter().map(|vi| vi * vi).sum::<f64>()).max(1e-10);

    (mean, var)
}

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
    // -0.5 * y^T K^{-1} y - sum(log(diag(L))) - n/2 * log(2π)
    let data_fit: f64 = -0.5 * ys.iter().zip(alpha.iter()).map(|(y, a)| y * a).sum::<f64>();
    let complexity: f64 = -l.iter().enumerate().map(|(i, row)| row[i].ln()).sum::<f64>();
    let constant = -0.5 * n as f64 * (2.0 * std::f64::consts::PI).ln();
    data_fit + complexity + constant
}

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
            .filter(|t| t.state == TrialState::Complete && t.params.contains_key(param_name))
            .collect();

        if completed.len() < self.config.n_startup_trials {
            return self
                .random_sampler
                .sample(study, trial, param_name, distribution);
        }

        // Collect observations in [0,1] internal space
        let xs: Vec<f64> = completed
            .iter()
            .map(|t| Self::to_internal(t.params[param_name], distribution))
            .collect();

        let mut ys: Vec<f64> = completed
            .iter()
            .map(|t| {
                let v = t.value.unwrap_or(0.0);
                match study.direction() {
                    crate::study::Direction::Minimize => v,
                    crate::study::Direction::Maximize => -v,
                }
            })
            .collect();

        // Standardize ys for numerical stability
        let y_mean: f64 = ys.iter().sum::<f64>() / ys.len() as f64;
        let y_std = {
            let var: f64 =
                ys.iter().map(|y| (y - y_mean).powi(2)).sum::<f64>() / ys.len() as f64;
            var.sqrt().max(1e-6)
        };
        for y in &mut ys {
            *y = (*y - y_mean) / y_std;
        }

        let best_f = ys
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);

        // Simple hyperparameter selection: try a few length scales, pick best
        let length_scales = [0.05, 0.1, 0.2, 0.5, 1.0];
        let noise = 1e-4;
        let mut best_ls = 0.2;
        let mut best_lml = f64::NEG_INFINITY;
        let mut best_alpha_vec: Vec<f64> = Vec::new();
        let mut best_l_mat: Vec<Vec<f64>> = Vec::new();

        for &ls in &length_scales {
            let k = build_kernel_matrix(&xs, ls, noise);
            if let Some(l) = cholesky(&k) {
                let z = forward_solve(&l, &ys);
                let alpha = backward_solve(&l, &z);
                let lml = log_marginal_likelihood(&l, &alpha, &ys);
                if lml > best_lml {
                    best_lml = lml;
                    best_ls = ls;
                    best_alpha_vec = alpha;
                    best_l_mat = l;
                }
            }
        }

        // If no Cholesky succeeded, fall back to random
        if best_alpha_vec.is_empty() {
            return self
                .random_sampler
                .sample(study, trial, param_name, distribution);
        }

        // Optimize acquisition function by random search
        let mut rng = self.rng.lock().unwrap();
        let mut best_x = 0.5;
        let mut best_ei = f64::NEG_INFINITY;

        for _ in 0..self.config.n_candidates {
            let x_cand: f64 = rng.random();
            let (mean, var) = gp_predict(x_cand, &xs, &best_alpha_vec, &best_l_mat, best_ls);
            let ei = expected_improvement(mean, var, best_f);
            if ei > best_ei {
                best_ei = ei;
                best_x = x_cand;
            }
        }

        Self::from_internal(best_x, distribution)
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
            study.add_completed_trial(params, (value - 3.0).powi(2));
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
            study.add_completed_trial(params, obj);
        }

        let best = study.best_value().unwrap();
        assert!(best < 2.0, "GP should converge, best={best}");
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
    fn test_matern52_self() {
        // k(x, x) should be 1.0
        assert!((matern52(0.5, 0.5, 0.2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matern52_symmetry() {
        let k1 = matern52(0.3, 0.7, 0.5);
        let k2 = matern52(0.7, 0.3, 0.5);
        assert!((k1 - k2).abs() < 1e-10);
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
}
