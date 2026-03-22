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

/// CMA-ES (Covariance Matrix Adaptation Evolution Strategy) sampler.
///
/// For small dimensions (n <= 50), uses full covariance matrix with
/// eigendecomposition-based sampling. For larger dimensions, falls back
/// to separable (diagonal) CMA-ES for O(n) updates.
///
/// All parameters are transformed to [0,1] internal space for scale-invariance.
/// Categorical parameters fall back to random sampling.
pub struct CmaEsSampler {
    config: CmaEsConfig,
    random_sampler: super::RandomSampler,
    state: Mutex<CmaEsState>,
}

/// Configuration for the CMA-ES sampler.
#[derive(Debug, Clone)]
pub struct CmaEsConfig {
    /// Number of startup trials before CMA-ES kicks in.
    pub n_startup_trials: usize,
    /// Initial step size (sigma0) in [0,1] space. Default 0.3.
    pub sigma0: f64,
    /// Population size (lambda). If None, uses `4 + floor(3 * ln(n_params))`.
    pub population_size: Option<usize>,
}

impl Default for CmaEsConfig {
    fn default() -> Self {
        Self {
            n_startup_trials: 10,
            sigma0: 0.3,
            population_size: None,
        }
    }
}

/// Internal state of CMA-ES.
#[derive(Debug)]
struct CmaEsState {
    rng: StdRng,
    /// Whether the optimizer has been initialized with parameter names.
    initialized: bool,
    /// Ordered parameter names (determines vector layout).
    param_names: Vec<String>,
    /// Mean of the search distribution in [0,1]^n space.
    mean: Vec<f64>,
    /// Global step size.
    sigma: f64,
    /// Whether to use full covariance (false = separable/diagonal only).
    use_full_cov: bool,
    /// Full covariance matrix (n×n). Only used when use_full_cov=true.
    c_full: Vec<Vec<f64>>,
    /// Diagonal covariance (per-parameter variance scaling). Used when use_full_cov=false.
    c_diag: Vec<f64>,
    /// Eigenvectors of C (columns of B). Only used when use_full_cov=true.
    eigvecs: Vec<Vec<f64>>,
    /// Square roots of eigenvalues of C. Only used when use_full_cov=true.
    eigvals_sqrt: Vec<f64>,
    /// Evolution path for step-size control (p_sigma).
    p_sigma: Vec<f64>,
    /// Evolution path for covariance adaptation (p_c).
    p_c: Vec<f64>,
    /// Population size (lambda).
    lambda: usize,
    /// Number of parents (mu).
    mu: usize,
    /// Recombination weights.
    weights: Vec<f64>,
    /// Variance-effective selection mass.
    mu_eff: f64,
    /// Learning rates and damping constants.
    c_sigma: f64,
    d_sigma: f64,
    c_c: f64,
    c_1: f64,
    c_mu_lr: f64,
    /// Expected length of N(0,I) vector.
    chi_n: f64,
    /// Current generation count.
    generation: usize,
    /// Pending population: trial_number -> sampled internal vector.
    pending: HashMap<usize, Vec<f64>>,
    /// Collected (value, internal_vector) pairs for current generation.
    evaluated: Vec<(f64, Vec<f64>)>,
    /// How often to recompute eigendecomposition.
    eigen_update_interval: usize,
    /// Generation at which eigendecomposition was last computed.
    last_eigen_update: usize,
}

impl CmaEsState {
    fn new(rng: StdRng, sigma0: f64) -> Self {
        Self {
            rng,
            initialized: false,
            param_names: Vec::new(),
            mean: Vec::new(),
            sigma: sigma0,
            use_full_cov: false,
            c_full: Vec::new(),
            c_diag: Vec::new(),
            eigvecs: Vec::new(),
            eigvals_sqrt: Vec::new(),
            p_sigma: Vec::new(),
            p_c: Vec::new(),
            lambda: 0,
            mu: 0,
            weights: Vec::new(),
            mu_eff: 0.0,
            c_sigma: 0.0,
            d_sigma: 0.0,
            c_c: 0.0,
            c_1: 0.0,
            c_mu_lr: 0.0,
            chi_n: 0.0,
            generation: 0,
            pending: HashMap::new(),
            evaluated: Vec::new(),
            eigen_update_interval: 1,
            last_eigen_update: 0,
        }
    }

    fn initialize(&mut self, n: usize, sigma0: f64, pop_size: Option<usize>) {
        let lambda = pop_size.unwrap_or_else(|| 4 + (3.0 * (n as f64).ln()).floor() as usize);
        let lambda = lambda.max(4);
        let mu = lambda / 2;

        // Log-linear weights (CMA-ES standard)
        let raw_weights: Vec<f64> = (0..mu)
            .map(|i| ((mu as f64 + 0.5).ln() - ((i + 1) as f64).ln()).max(0.0))
            .collect();
        let w_sum: f64 = raw_weights.iter().sum();
        let weights: Vec<f64> = raw_weights.iter().map(|w| w / w_sum).collect();

        let mu_eff: f64 = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        // Strategy parameter defaults (Hansen 2001)
        let c_sigma = (mu_eff + 2.0) / (n as f64 + mu_eff + 5.0);
        let d_sigma = 1.0
            + 2.0 * (((mu_eff - 1.0) / (n as f64 + 1.0)).sqrt() - 1.0).max(0.0)
            + c_sigma;
        let c_c = (4.0 + mu_eff / n as f64) / (n as f64 + 4.0 + 2.0 * mu_eff / n as f64);

        // For separable CMA-ES: c_1 and c_mu are for diagonal updates
        let c_1 = 2.0 / ((n as f64 + 1.3).powi(2) + mu_eff);
        let c_mu_lr = (2.0 * (mu_eff - 2.0 + 1.0 / mu_eff)
            / ((n as f64 + 2.0).powi(2) + mu_eff))
            .min(1.0 - c_1);

        let chi_n = (n as f64).sqrt() * (1.0 - 1.0 / (4.0 * n as f64) + 1.0 / (21.0 * (n as f64).powi(2)));

        self.mean = vec![0.5; n];
        self.p_sigma = vec![0.0; n];
        self.p_c = vec![0.0; n];
        self.lambda = lambda;
        self.mu = mu;
        self.weights = weights;
        self.mu_eff = mu_eff;
        self.c_sigma = c_sigma;
        self.d_sigma = d_sigma;
        self.c_c = c_c;
        self.c_1 = c_1;
        self.c_mu_lr = c_mu_lr;
        self.chi_n = chi_n;
        self.sigma = sigma0;

        // Use full covariance for small dimensions
        self.use_full_cov = n <= 50;
        if self.use_full_cov {
            // Initialize C = I (identity matrix)
            self.c_full = vec![vec![0.0; n]; n];
            for i in 0..n {
                self.c_full[i][i] = 1.0;
            }
            // Initialize eigendecomposition: B = I, D = I
            self.eigvecs = vec![vec![0.0; n]; n];
            for i in 0..n {
                self.eigvecs[i][i] = 1.0;
            }
            self.eigvals_sqrt = vec![1.0; n];
            self.eigen_update_interval = (n as f64 / 10.0).max(1.0) as usize;
        }
        self.c_diag = vec![1.0; n];

        self.initialized = true;
    }

    /// Jacobi eigenvalue algorithm for symmetric matrices.
    /// Returns (eigenvalues, eigenvectors) where eigenvectors are columns of the returned matrix.
    fn jacobi_eigen(a: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
        let n = a.len();
        let mut a = a.to_vec();
        let mut v = vec![vec![0.0; n]; n];
        for i in 0..n {
            v[i][i] = 1.0;
        }

        let max_iter = 100 * n * n;
        for _ in 0..max_iter {
            // Find largest off-diagonal element
            let mut max_val = 0.0_f64;
            let mut p = 0;
            let mut q = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    if a[i][j].abs() > max_val {
                        max_val = a[i][j].abs();
                        p = i;
                        q = j;
                    }
                }
            }

            if max_val < 1e-15 {
                break;
            }

            // Compute rotation
            let theta = if (a[p][p] - a[q][q]).abs() < 1e-20 {
                std::f64::consts::FRAC_PI_4
            } else {
                0.5 * (2.0 * a[p][q] / (a[p][p] - a[q][q])).atan()
            };

            let cos_t = theta.cos();
            let sin_t = theta.sin();

            // Apply rotation to A
            let mut new_a = a.clone();
            for i in 0..n {
                if i != p && i != q {
                    new_a[i][p] = cos_t * a[i][p] + sin_t * a[i][q];
                    new_a[p][i] = new_a[i][p];
                    new_a[i][q] = -sin_t * a[i][p] + cos_t * a[i][q];
                    new_a[q][i] = new_a[i][q];
                }
            }
            new_a[p][p] = cos_t * cos_t * a[p][p]
                + 2.0 * sin_t * cos_t * a[p][q]
                + sin_t * sin_t * a[q][q];
            new_a[q][q] = sin_t * sin_t * a[p][p]
                - 2.0 * sin_t * cos_t * a[p][q]
                + cos_t * cos_t * a[q][q];
            new_a[p][q] = 0.0;
            new_a[q][p] = 0.0;
            a = new_a;

            // Apply rotation to V
            for i in 0..n {
                let vip = v[i][p];
                let viq = v[i][q];
                v[i][p] = cos_t * vip + sin_t * viq;
                v[i][q] = -sin_t * vip + cos_t * viq;
            }
        }

        let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
        (eigenvalues, v)
    }

    /// Recompute eigendecomposition of the covariance matrix.
    fn update_eigen(&mut self) {
        if !self.use_full_cov {
            return;
        }
        let (eigenvalues, eigvecs) = Self::jacobi_eigen(&self.c_full);
        self.eigvals_sqrt = eigenvalues.iter().map(|&v| v.max(1e-20).sqrt()).collect();
        self.eigvecs = eigvecs;
        self.last_eigen_update = self.generation;
    }

    /// Sample a new candidate vector from the search distribution.
    fn sample_candidate(&mut self) -> Vec<f64> {
        let n = self.mean.len();

        if self.use_full_cov {
            // Full covariance: x = mean + sigma * B * D * z
            let z: Vec<f64> = (0..n)
                .map(|_| self.rng.sample(rand_distr::StandardNormal))
                .collect();

            // Compute B * D * z
            let mut bdz = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    bdz[i] += self.eigvecs[i][j] * self.eigvals_sqrt[j] * z[j];
                }
            }

            let mut x = Vec::with_capacity(n);
            for i in 0..n {
                x.push((self.mean[i] + self.sigma * bdz[i]).clamp(0.0, 1.0));
            }
            x
        } else {
            // Separable: x = mean + sigma * sqrt(c_diag) * z
            let mut x = Vec::with_capacity(n);
            for i in 0..n {
                let z: f64 = self.rng.sample(rand_distr::StandardNormal);
                let xi = self.mean[i] + self.sigma * self.c_diag[i].sqrt() * z;
                x.push(xi.clamp(0.0, 1.0));
            }
            x
        }
    }

    /// Update the search distribution from evaluated solutions.
    /// `solutions` must be sorted best-first.
    fn update(&mut self, solutions: &[(f64, Vec<f64>)]) {
        let n = self.mean.len();
        if solutions.len() < self.mu {
            return;
        }

        let old_mean = self.mean.clone();

        // Weighted recombination: new mean
        let mut new_mean = vec![0.0; n];
        for (wi, (_, x)) in self.weights.iter().zip(solutions.iter()) {
            for j in 0..n {
                new_mean[j] += wi * x[j];
            }
        }
        self.mean = new_mean;

        // Mean displacement
        let mut mean_diff = vec![0.0; n];
        for j in 0..n {
            mean_diff[j] = (self.mean[j] - old_mean[j]) / self.sigma;
        }

        // Update evolution path p_sigma (CSA)
        let sqrt_c_sigma = (self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff).sqrt();
        for j in 0..n {
            let invsqrt_c = 1.0 / self.c_diag[j].sqrt().max(1e-20);
            self.p_sigma[j] = (1.0 - self.c_sigma) * self.p_sigma[j]
                + sqrt_c_sigma * invsqrt_c * mean_diff[j];
        }

        // Norm of p_sigma
        let ps_norm: f64 = self.p_sigma.iter().map(|v| v * v).sum::<f64>().sqrt();

        // h_sigma: indicator for stalling
        let h_sigma_threshold = (1.0 - (1.0 - self.c_sigma).powi(2 * (self.generation as i32 + 1)))
            .sqrt()
            * (1.4 + 2.0 / (n as f64 + 1.0))
            * self.chi_n;
        let h_sigma = if ps_norm < h_sigma_threshold { 1.0 } else { 0.0 };

        // Update evolution path p_c
        let sqrt_c_c = (self.c_c * (2.0 - self.c_c) * self.mu_eff).sqrt();
        for j in 0..n {
            self.p_c[j] = (1.0 - self.c_c) * self.p_c[j]
                + h_sigma * sqrt_c_c * mean_diff[j];
        }

        // Update covariance
        let delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c);

        if self.use_full_cov {
            // Full covariance update: C = (1-c1-cmu)*C + c1*(pc*pc^T + delta_h*C) + cmu*sum(wi*yi*yi^T)
            for i in 0..n {
                for j in 0..n {
                    let rank1 = self.c_1 * (self.p_c[i] * self.p_c[j]
                        + delta_h * self.c_full[i][j]);

                    let mut rank_mu = 0.0;
                    for (wi, (_, x)) in self.weights.iter().zip(solutions.iter()) {
                        let di = (x[i] - old_mean[i]) / self.sigma;
                        let dj = (x[j] - old_mean[j]) / self.sigma;
                        rank_mu += wi * di * dj;
                    }
                    rank_mu *= self.c_mu_lr;

                    self.c_full[i][j] = (1.0 - self.c_1 - self.c_mu_lr) * self.c_full[i][j]
                        + rank1
                        + rank_mu;
                }
            }

            // Update diagonal from full covariance (for compatibility)
            for j in 0..n {
                self.c_diag[j] = self.c_full[j][j].max(1e-20);
            }

            // Periodically recompute eigendecomposition
            if self.generation - self.last_eigen_update >= self.eigen_update_interval {
                self.update_eigen();
            }
        } else {
            // Separable (diagonal-only) update
            for j in 0..n {
                let rank1 = self.c_1 * (self.p_c[j] * self.p_c[j]
                    + delta_h * self.c_diag[j]);

                let mut rank_mu = 0.0;
                for (wi, (_, x)) in self.weights.iter().zip(solutions.iter()) {
                    let diff_j = (x[j] - old_mean[j]) / self.sigma;
                    rank_mu += wi * diff_j * diff_j;
                }
                rank_mu *= self.c_mu_lr;

                self.c_diag[j] = (1.0 - self.c_1 - self.c_mu_lr) * self.c_diag[j]
                    + rank1
                    + rank_mu;

                self.c_diag[j] = self.c_diag[j].max(1e-20);
            }
        }

        // Update sigma via CSA
        self.sigma *= ((self.c_sigma / self.d_sigma) * (ps_norm / self.chi_n - 1.0)).exp();
        self.sigma = self.sigma.clamp(1e-20, 1e5);

        self.generation += 1;
    }
}

impl CmaEsSampler {
    pub fn new(config: CmaEsConfig, seed: u64) -> Self {
        Self {
            config: config.clone(),
            random_sampler: super::RandomSampler::new(seed),
            state: Mutex::new(CmaEsState::new(
                StdRng::seed_from_u64(seed.wrapping_add(2)),
                config.sigma0,
            )),
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        Self::new(CmaEsConfig::default(), seed)
    }

    /// Transform a raw parameter value to [0,1] internal space based on distribution.
    fn to_internal(value: f64, distribution: &Distribution) -> f64 {
        match distribution {
            Distribution::Float(d) => transform_to_internal(value, d),
            Distribution::Int(d) => int_transform_to_internal(value as i64, d),
            Distribution::Categorical(_) => value, // not used
        }
    }

    /// Transform from [0,1] internal space back to raw parameter value.
    fn from_internal(internal: f64, distribution: &Distribution) -> f64 {
        match distribution {
            Distribution::Float(d) => transform_from_internal(internal, d),
            Distribution::Int(d) => int_transform_from_internal(internal, d) as f64,
            Distribution::Categorical(_) => internal, // not used
        }
    }
}

impl Sampler for CmaEsSampler {
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

        let mut state = self.state.lock().unwrap();

        // Check if we have a pending sample for this trial
        if let Some(candidate) = state.pending.get(&trial.number) {
            if let Some(idx) = state.param_names.iter().position(|n| n == param_name) {
                return Self::from_internal(candidate[idx], distribution);
            }
        }

        // Initialize on first real use
        if !state.initialized {
            // Collect all parameter names from the most recent complete trial
            let latest = completed.last().unwrap();
            let mut param_names: Vec<String> = latest.params.keys().cloned().collect();
            param_names.sort();

            // Filter to only non-categorical params
            // For now, use all params found in the latest trial
            let n = param_names.len().max(1);
            state.param_names = param_names;
            state.initialize(n, self.config.sigma0, self.config.population_size);
        }

        // Collect evaluated solutions from completed trials that we sampled
        // and process any complete generation
        let pending_trials: Vec<usize> = state.pending.keys().cloned().collect();
        for trial_num in pending_trials {
            if let Some(ft) = completed.iter().find(|t| t.number == trial_num) {
                if let Some(candidate) = state.pending.remove(&trial_num) {
                    let value = ft.value.unwrap_or(f64::INFINITY);
                    let obj = match study.direction() {
                        crate::study::Direction::Minimize => value,
                        crate::study::Direction::Maximize => -value,
                    };
                    state.evaluated.push((obj, candidate));
                }
            }
        }

        // If we have enough evaluated solutions, update the distribution
        if state.evaluated.len() >= state.lambda {
            let mut solutions = std::mem::take(&mut state.evaluated);
            solutions.sort_by(|a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            state.update(&solutions);
        }

        // Sample a new candidate
        let candidate = state.sample_candidate();
        let idx = state
            .param_names
            .iter()
            .position(|n| n == param_name)
            .unwrap_or(0);

        let result = Self::from_internal(candidate[idx], distribution);
        state.pending.insert(trial.number, candidate);

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_cmaes_float_in_bounds() {
        let sampler = CmaEsSampler::with_seed(42);
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
        assert!(v >= 0.0 && v <= 10.0, "CMA-ES suggested {v} outside [0, 10]");
    }

    #[test]
    fn test_cmaes_int_in_bounds() {
        let sampler = CmaEsSampler::with_seed(42);
        let dist = Distribution::Int(crate::distributions::IntDistribution::new(
            1, 50, false, None,
        ));
        let mut study = Study::new_default();

        for i in 0..15 {
            let value = (i * 3 + 1) as f64;
            let mut params = HashMap::new();
            params.insert("n".to_string(), value);
            study.add_completed_trial(params, (value - 20.0).powi(2));
        }

        let trial = Trial::new(15);
        let v = sampler.sample(&study, &trial, "n", &dist) as i64;
        assert!(v >= 1 && v <= 50, "CMA-ES suggested int {v} outside [1, 50]");
    }

    #[test]
    fn test_cmaes_during_startup_uses_random() {
        let sampler = CmaEsSampler::with_seed(42);
        let dist = Distribution::Float(crate::distributions::FloatDistribution::new(
            0.0, 1.0, false, None,
        ));
        let study = Study::new_default();
        let trial = Trial::new(0);

        let v = sampler.sample(&study, &trial, "x", &dist);
        assert!(v >= 0.0 && v <= 1.0);
    }

    #[test]
    fn test_cmaes_converges_toward_optimum() {
        let sampler = CmaEsSampler::new(
            CmaEsConfig {
                n_startup_trials: 5,
                sigma0: 0.3,
                population_size: None,
            },
            42,
        );
        let dist = Distribution::Float(crate::distributions::FloatDistribution::new(
            -10.0, 10.0, false, None,
        ));
        let mut study = Study::new_default();

        for _ in 0..50 {
            let trial = Trial::new(study.trials().len());
            let x = sampler.sample(&study, &trial, "x", &dist);
            let obj = (x - 3.0).powi(2);
            let mut params = HashMap::new();
            params.insert("x".to_string(), x);
            study.add_completed_trial(params, obj);
        }

        let best = study.best_value().unwrap();
        assert!(best < 5.0, "CMA-ES should converge, best={best}");
    }

    #[test]
    fn test_cmaes_log_scale_float() {
        let sampler = CmaEsSampler::with_seed(42);
        let dist = Distribution::Float(crate::distributions::FloatDistribution::new(
            1e-5, 1.0, true, None,
        ));
        let mut study = Study::new_default();

        for i in 0..15 {
            let v = 1e-5 * (1e5_f64).powf(i as f64 / 14.0);
            let mut params = HashMap::new();
            params.insert("lr".to_string(), v);
            study.add_completed_trial(params, (v - 0.01).powi(2));
        }

        let trial = Trial::new(15);
        let v = sampler.sample(&study, &trial, "lr", &dist);
        assert!(
            v >= 1e-5 && v <= 1.0,
            "CMA-ES log-scale suggested {v} outside [1e-5, 1.0]"
        );
    }

    #[test]
    fn test_cmaes_state_initialization() {
        let state = CmaEsState::new(StdRng::seed_from_u64(42), 0.3);
        assert!(!state.initialized);
    }

    #[test]
    fn test_cmaes_strategy_params() {
        let mut state = CmaEsState::new(StdRng::seed_from_u64(42), 0.3);
        state.param_names = vec!["a".into(), "b".into(), "c".into()];
        state.initialize(3, 0.3, None);

        assert!(state.initialized);
        assert!(state.use_full_cov); // n=3 <= 50, should use full covariance
        assert!(state.lambda >= 4);
        assert!(state.mu > 0);
        assert!(state.mu < state.lambda);
        assert!((state.weights.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        assert!(state.chi_n > 0.0);
        assert!(state.c_sigma > 0.0);
        assert!(state.d_sigma > 0.0);
        assert_eq!(state.c_full.len(), 3);
        assert_eq!(state.eigvecs.len(), 3);
        assert_eq!(state.eigvals_sqrt.len(), 3);
    }

    #[test]
    fn test_jacobi_eigen_identity() {
        let identity = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let (eigenvalues, _) = CmaEsState::jacobi_eigen(&identity);
        for ev in &eigenvalues {
            assert!((ev - 1.0).abs() < 1e-10, "eigenvalue={ev}, expected 1.0");
        }
    }

    #[test]
    fn test_jacobi_eigen_diagonal() {
        let diag = vec![
            vec![4.0, 0.0],
            vec![0.0, 9.0],
        ];
        let (mut eigenvalues, _) = CmaEsState::jacobi_eigen(&diag);
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((eigenvalues[0] - 4.0).abs() < 1e-10);
        assert!((eigenvalues[1] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_jacobi_eigen_symmetric() {
        let a = vec![
            vec![2.0, 1.0],
            vec![1.0, 2.0],
        ];
        let (mut eigenvalues, _) = CmaEsState::jacobi_eigen(&a);
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Eigenvalues of [[2,1],[1,2]] are 1 and 3
        assert!((eigenvalues[0] - 1.0).abs() < 1e-10);
        assert!((eigenvalues[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cmaes_full_cov_correlated() {
        // Test that full CMA-ES can handle correlated parameters
        let sampler = CmaEsSampler::new(
            CmaEsConfig {
                n_startup_trials: 5,
                sigma0: 0.3,
                population_size: Some(8),
            },
            42,
        );
        let dist_x = Distribution::Float(crate::distributions::FloatDistribution::new(
            -5.0, 5.0, false, None,
        ));
        let dist_y = Distribution::Float(crate::distributions::FloatDistribution::new(
            -5.0, 5.0, false, None,
        ));
        let mut study = Study::new_default();

        for _ in 0..40 {
            let trial = Trial::new(study.trials().len());
            let x = sampler.sample(&study, &trial, "x", &dist_x);
            let y = sampler.sample(&study, &trial, "y", &dist_y);
            // Correlated objective: (x+y)^2 + (x-y)^2, minimum at (0,0)
            let obj = (x + y).powi(2) + (x - y).powi(2);
            let mut params = HashMap::new();
            params.insert("x".to_string(), x);
            params.insert("y".to_string(), y);
            study.add_completed_trial(params, obj);
        }

        let best = study.best_value().unwrap();
        assert!(best < 10.0, "Full CMA-ES should converge on correlated problem, best={best}");
    }
}
