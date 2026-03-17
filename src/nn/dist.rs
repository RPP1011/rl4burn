//! Action distribution utilities for discrete, multi-discrete, and continuous spaces.
//!
//! Provides sampling, log-probability, and entropy computation with optional
//! action masking. Works with any Burn backend.

use burn::prelude::*;
use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::TensorData;
use rand::Rng;
use std::f32::consts::PI;

/// Minimum clamping bound for log_std in continuous distributions.
/// exp(-5) ≈ 0.0067 — prevents near-zero standard deviations.
const LOG_STD_MIN: f32 = -5.0;
/// Maximum clamping bound for log_std in continuous distributions.
/// exp(2) ≈ 7.4 — prevents excessively wide exploration.
const LOG_STD_MAX: f32 = 2.0;

/// How log-std is provided for continuous action distributions.
#[derive(Debug, Clone)]
pub enum LogStdMode {
    /// Model outputs `[batch, 2 * action_dim]`: first half = means, second half = log_stds.
    ModelOutput,
    /// log_std is a fixed learnable parameter (state-independent), not part of model output.
    /// This is what CleanRL's continuous PPO uses.
    Separate,
}

/// Describes how to interpret a flat logits tensor as an action distribution.
///
/// For `Discrete(n)`:        logits shape `[batch, n]`, one categorical.
/// For `MultiDiscrete(nvec)`: logits shape `[batch, sum(nvec)]`, independent categoricals.
/// For `Continuous`:          logits shape depends on `LogStdMode`.
#[derive(Debug, Clone)]
pub enum ActionDist {
    /// Single categorical over n actions.
    Discrete(usize),
    /// Multiple independent categoricals. `nvec[i]` = number of choices for dimension i.
    MultiDiscrete(Vec<usize>),
    /// Diagonal Gaussian for continuous control.
    Continuous {
        action_dim: usize,
        log_std_mode: LogStdMode,
    },
}

impl ActionDist {
    /// Total number of logits expected from the model.
    pub fn n_logits(&self) -> usize {
        match self {
            ActionDist::Discrete(n) => *n,
            ActionDist::MultiDiscrete(nvec) => nvec.iter().sum(),
            ActionDist::Continuous {
                action_dim,
                log_std_mode,
            } => match log_std_mode {
                LogStdMode::ModelOutput => 2 * action_dim,
                LogStdMode::Separate => *action_dim,
            },
        }
    }

    /// Number of action dimensions.
    pub fn n_dims(&self) -> usize {
        match self {
            ActionDist::Discrete(_) => 1,
            ActionDist::MultiDiscrete(nvec) => nvec.len(),
            ActionDist::Continuous { action_dim, .. } => *action_dim,
        }
    }

    /// Per-dimension sizes (for discrete variants).
    pub fn nvec(&self) -> Vec<usize> {
        match self {
            ActionDist::Discrete(n) => vec![*n],
            ActionDist::MultiDiscrete(nvec) => nvec.clone(),
            ActionDist::Continuous { action_dim, .. } => vec![1; *action_dim],
        }
    }

    /// Sample actions from logits, optionally masked.
    ///
    /// Returns `[batch_size][n_dims]` action values (integer-valued f32 for discrete).
    ///
    /// - `logits`: `[batch, n_logits]`
    /// - `mask`: `[batch, n_logits]` optional, `1.0`=valid `0.0`=invalid (discrete only)
    /// - `log_std`: separate learnable log_std `[action_dim]` (Continuous/Separate only)
    pub fn sample<B: Backend>(
        &self,
        logits: &Tensor<B, 2>,
        mask: Option<&Tensor<B, 2>>,
        log_std: Option<&Tensor<B, 1>>,
        rng: &mut impl Rng,
    ) -> Vec<Vec<f32>> {
        let batch_size = logits.dims()[0];

        match self {
            ActionDist::Discrete(_) | ActionDist::MultiDiscrete(_) => {
                let nvec = match self {
                    ActionDist::Discrete(n) => vec![*n],
                    ActionDist::MultiDiscrete(v) => v.clone(),
                    _ => unreachable!(),
                };
                let n_logits: usize = nvec.iter().sum();

                let masked_logits = match mask {
                    Some(m) => logits.clone() + (m.clone() - 1.0) * 1e9,
                    None => logits.clone(),
                };

                let all_probs = softmax(masked_logits, 1);
                let probs_data: Vec<f32> = all_probs.into_data().to_vec().unwrap();

                let mut actions = Vec::with_capacity(batch_size);
                for b in 0..batch_size {
                    let row = &probs_data[b * n_logits..(b + 1) * n_logits];
                    let mut batch_actions = Vec::with_capacity(nvec.len());
                    let mut offset = 0;
                    for &dim_size in &nvec {
                        let dim_probs = &row[offset..offset + dim_size];
                        batch_actions.push(sample_categorical(dim_probs, rng) as f32);
                        offset += dim_size;
                    }
                    actions.push(batch_actions);
                }
                actions
            }

            ActionDist::Continuous {
                action_dim,
                log_std_mode,
            } => {
                let ad = *action_dim;
                let logits_data: Vec<f32> = logits.clone().into_data().to_vec().unwrap();
                let total_cols = logits.dims()[1];

                let separate_log_std: Option<Vec<f32>> = match log_std_mode {
                    LogStdMode::Separate => Some(
                        log_std
                            .expect("Continuous/Separate requires log_std")
                            .clone()
                            .into_data()
                            .to_vec()
                            .unwrap(),
                    ),
                    LogStdMode::ModelOutput => None,
                };

                let mut actions = Vec::with_capacity(batch_size);
                for b in 0..batch_size {
                    let row = &logits_data[b * total_cols..(b + 1) * total_cols];
                    let means = &row[0..ad];
                    let log_stds: &[f32] = match log_std_mode {
                        LogStdMode::ModelOutput => &row[ad..2 * ad],
                        LogStdMode::Separate => separate_log_std.as_ref().unwrap(),
                    };

                    let mut batch_actions = Vec::with_capacity(ad);
                    for d in 0..ad {
                        let z = sample_normal(rng);
                        let clamped = log_stds[d].clamp(LOG_STD_MIN, LOG_STD_MAX);
                        batch_actions.push(means[d] + clamped.exp() * z);
                    }
                    actions.push(batch_actions);
                }
                actions
            }
        }
    }

    /// Compute log-probabilities of given actions under the logits distribution.
    ///
    /// Returns `[batch]` — sum of per-dim log-probs.
    ///
    /// - `logits`: `[batch, n_logits]`
    /// - `actions`: `[batch][n_dims]` — integer-valued f32 for discrete, float for continuous
    /// - `mask`: `[batch, n_logits]` optional (discrete only)
    /// - `log_std`: separate learnable log_std `[action_dim]` (Continuous/Separate only)
    pub fn log_prob<B: Backend>(
        &self,
        logits: Tensor<B, 2>,
        actions: &[Vec<f32>],
        mask: Option<&Tensor<B, 2>>,
        log_std: Option<&Tensor<B, 1>>,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        let batch_size = actions.len();

        match self {
            ActionDist::Discrete(_) | ActionDist::MultiDiscrete(_) => {
                let nvec = match self {
                    ActionDist::Discrete(n) => vec![*n],
                    ActionDist::MultiDiscrete(v) => v.clone(),
                    _ => unreachable!(),
                };

                let masked_logits = match mask {
                    Some(m) => logits + (m.clone() - 1.0) * 1e9,
                    None => logits,
                };

                let mut total_lp: Option<Tensor<B, 1>> = None;
                let mut offset = 0;

                for (d, &dim_size) in nvec.iter().enumerate() {
                    let dim_logits =
                        masked_logits
                            .clone()
                            .slice([0..batch_size, offset..offset + dim_size]);
                    let dim_lp = log_softmax(dim_logits, 1);

                    let action_indices: Vec<i32> =
                        actions.iter().map(|a| a[d] as i32).collect();
                    let idx_tensor: Tensor<B, 2, Int> = Tensor::from_data(
                        TensorData::new(action_indices, [batch_size, 1]),
                        device,
                    );
                    let gathered: Tensor<B, 1> =
                        dim_lp.gather(1, idx_tensor).squeeze_dim::<1>(1);

                    total_lp = Some(match total_lp {
                        Some(acc) => acc + gathered,
                        None => gathered,
                    });

                    offset += dim_size;
                }

                total_lp.unwrap()
            }

            ActionDist::Continuous {
                action_dim,
                log_std_mode,
            } => {
                let ad = *action_dim;

                let (means, log_stds_tensor) = match log_std_mode {
                    LogStdMode::ModelOutput => {
                        let m = logits.clone().slice([0..batch_size, 0..ad]);
                        let ls = logits.slice([0..batch_size, ad..2 * ad]);
                        (m, ls)
                    }
                    LogStdMode::Separate => {
                        let ls = log_std
                            .expect("Continuous/Separate requires log_std")
                            .clone()
                            .unsqueeze_dim::<2>(0)
                            .repeat_dim(0, batch_size);
                        (logits, ls)
                    }
                };

                let log_stds_tensor =
                    log_stds_tensor.clamp(LOG_STD_MIN, LOG_STD_MAX);

                let action_flat: Vec<f32> =
                    actions.iter().flat_map(|a| a.iter().copied()).collect();
                let action_tensor: Tensor<B, 2> =
                    Tensor::from_data(TensorData::new(action_flat, [batch_size, ad]), device);

                let stds = log_stds_tensor.clone().exp();
                let normalized = (action_tensor - means) / stds;

                let half_ln_2pi: f32 = 0.5 * (2.0 * PI).ln();
                let log_prob_elements =
                    normalized.powf_scalar(2.0).neg() * 0.5 - log_stds_tensor - half_ln_2pi;

                log_prob_elements.sum_dim(1).squeeze_dim::<1>(1)
            }
        }
    }

    /// Compute entropy of the distribution, optionally masked.
    ///
    /// Returns `[batch]` — sum of per-dim entropies.
    ///
    /// - `logits`: `[batch, n_logits]`
    /// - `mask`: `[batch, n_logits]` optional (discrete only)
    /// - `log_std`: separate learnable log_std `[action_dim]` (Continuous/Separate only)
    pub fn entropy<B: Backend>(
        &self,
        logits: Tensor<B, 2>,
        mask: Option<&Tensor<B, 2>>,
        log_std: Option<&Tensor<B, 1>>,
    ) -> Tensor<B, 1> {
        let batch_size = logits.dims()[0];

        match self {
            ActionDist::Discrete(_) | ActionDist::MultiDiscrete(_) => {
                let nvec = match self {
                    ActionDist::Discrete(n) => vec![*n],
                    ActionDist::MultiDiscrete(v) => v.clone(),
                    _ => unreachable!(),
                };

                let masked_logits = match mask {
                    Some(m) => logits + (m.clone() - 1.0) * 1e9,
                    None => logits,
                };

                let mut total_ent: Option<Tensor<B, 1>> = None;
                let mut offset = 0;

                for &dim_size in &nvec {
                    let dim_logits = masked_logits
                        .clone()
                        .slice([0..batch_size, offset..offset + dim_size]);
                    let probs = softmax(dim_logits.clone(), 1);
                    let log_probs = log_softmax(dim_logits, 1);

                    // H = -sum(p * log_p, dim=1)
                    let ent: Tensor<B, 1> =
                        (probs * log_probs).sum_dim(1).squeeze_dim::<1>(1).neg();

                    total_ent = Some(match total_ent {
                        Some(acc) => acc + ent,
                        None => ent,
                    });

                    offset += dim_size;
                }

                total_ent.unwrap()
            }

            ActionDist::Continuous {
                action_dim,
                log_std_mode,
            } => {
                let ad = *action_dim;

                let log_stds_tensor: Tensor<B, 2> = match log_std_mode {
                    LogStdMode::ModelOutput => logits.slice([0..batch_size, ad..2 * ad]),
                    LogStdMode::Separate => log_std
                        .expect("Continuous/Separate requires log_std")
                        .clone()
                        .unsqueeze_dim::<2>(0)
                        .repeat_dim(0, batch_size),
                };

                let log_stds_tensor =
                    log_stds_tensor.clamp(LOG_STD_MIN, LOG_STD_MAX);

                // H(Normal) = log_std + 0.5 * (1 + ln(2π))
                let half_log_2pi_e: f32 = 0.5 * (1.0 + (2.0 * PI).ln());
                (log_stds_tensor + half_log_2pi_e)
                    .sum_dim(1)
                    .squeeze_dim::<1>(1)
            }
        }
    }
}

fn sample_categorical(probs: &[f32], rng: &mut impl Rng) -> usize {
    let u: f32 = rng.random();
    let mut cum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if u < cum {
            return i;
        }
    }
    probs.len() - 1
}

fn sample_normal(rng: &mut impl Rng) -> f32 {
    // Box-Muller transform
    let u1: f32 = 1.0 - rng.random::<f32>(); // (0, 1] to avoid ln(0)
    let u2: f32 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use rand::SeedableRng;

    type B = NdArray;

    fn device() -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    #[test]
    fn discrete_sample_respects_mask() {
        let dist = ActionDist::Discrete(4);
        let device = device();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

        // Uniform logits
        let logits: Tensor<B, 2> = Tensor::zeros([1000, 4], &device);
        // Mask out actions 0 and 2
        let mask_data = vec![0.0f32, 1.0, 0.0, 1.0];
        let mask_flat: Vec<f32> = (0..1000).flat_map(|_| mask_data.iter().copied()).collect();
        let mask: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(mask_flat, [1000, 4]), &device);

        let actions = dist.sample(&logits, Some(&mask), None, &mut rng);

        for a in &actions {
            let action = a[0] as i32;
            assert!(
                action == 1 || action == 3,
                "masked action {action} was sampled"
            );
        }
    }

    #[test]
    fn discrete_log_prob_matches_manual() {
        let dist = ActionDist::Discrete(3);
        let device = device();

        let logits_data = vec![1.0f32, 2.0, 3.0, 0.5, 0.5, 0.5];
        let logits: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(logits_data.clone(), [2, 3]), &device);

        let actions = vec![vec![1.0f32], vec![2.0]];
        let lp = dist.log_prob(logits, &actions, None, None, &device);
        let lp_data: Vec<f32> = lp.into_data().to_vec().unwrap();

        // Manual: log_softmax of [1,2,3] at index 1
        let manual_lp0 = {
            let max = 3.0f32;
            let lse = (1.0 - max).exp() + (2.0 - max).exp() + (3.0 - max).exp();
            (2.0 - max) - lse.ln()
        };
        // Manual: log_softmax of [0.5,0.5,0.5] at index 2
        let manual_lp1 = -(3.0f32.ln());

        assert!((lp_data[0] - manual_lp0).abs() < 1e-5, "lp[0] mismatch");
        assert!((lp_data[1] - manual_lp1).abs() < 1e-5, "lp[1] mismatch");
    }

    #[test]
    fn multi_discrete_log_prob_sums() {
        let dist = ActionDist::MultiDiscrete(vec![3, 4]);
        let device = device();

        // 7 logits total
        let logits_data = vec![
            1.0f32, 2.0, 3.0, 0.5, 1.5, 2.5, 3.5, // batch item 0
        ];
        let logits: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(logits_data, [1, 7]), &device);

        let actions = vec![vec![0.0f32, 2.0]];
        let lp = dist.log_prob(logits, &actions, None, None, &device);
        let lp_val: f32 = lp.into_data().to_vec().unwrap()[0];

        // Compute per-dim manually
        let _dim0_logits = [1.0f32, 2.0, 3.0];
        let dim0_max = 3.0f32;
        let dim0_lse =
            (1.0f32 - dim0_max).exp() + (2.0f32 - dim0_max).exp() + (3.0f32 - dim0_max).exp();
        let dim0_lp = (1.0 - dim0_max) - dim0_lse.ln(); // action 0

        let dim1_logits = [0.5f32, 1.5, 2.5, 3.5];
        let dim1_max = 3.5;
        let dim1_lse: f32 = dim1_logits.iter().map(|x| (x - dim1_max).exp()).sum();
        let dim1_lp = (2.5 - dim1_max) - dim1_lse.ln(); // action 2

        let expected = dim0_lp + dim1_lp;
        assert!(
            (lp_val - expected).abs() < 1e-5,
            "multi-discrete lp {lp_val} != expected {expected}"
        );
    }

    #[test]
    fn multi_discrete_entropy_sums() {
        let dist = ActionDist::MultiDiscrete(vec![3, 4]);
        let device = device();

        let logits: Tensor<B, 2> = Tensor::zeros([1, 7], &device);
        let ent: f32 = dist
            .entropy(logits, None, None)
            .into_data()
            .to_vec()
            .unwrap()[0];

        // Uniform over 3 choices: ln(3), over 4: ln(4)
        let expected = 3.0f32.ln() + 4.0f32.ln();
        assert!(
            (ent - expected).abs() < 1e-4,
            "entropy {ent} != expected {expected}"
        );
    }

    #[test]
    fn mask_shifts_entropy() {
        let dist = ActionDist::Discrete(4);
        let device = device();

        let logits: Tensor<B, 2> = Tensor::zeros([1, 4], &device);

        // Full: entropy = ln(4)
        let full_ent: f32 = dist
            .entropy(logits.clone(), None, None)
            .into_data()
            .to_vec()
            .unwrap()[0];

        // Mask out 2 actions: entropy = ln(2)
        let mask: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(vec![1.0f32, 1.0, 0.0, 0.0], [1, 4]),
            &device,
        );
        let partial_ent: f32 = dist
            .entropy(logits, Some(&mask), None)
            .into_data()
            .to_vec()
            .unwrap()[0];

        assert!(full_ent > partial_ent, "full entropy should exceed partial");
        assert!(
            (full_ent - 4.0f32.ln()).abs() < 1e-4,
            "full entropy should be ln(4)"
        );
        assert!(
            (partial_ent - 2.0f32.ln()).abs() < 1e-4,
            "partial entropy should be ln(2)"
        );
    }

    #[test]
    fn discrete_is_multi_discrete_1() {
        let disc = ActionDist::Discrete(5);
        let multi = ActionDist::MultiDiscrete(vec![5]);
        let device = device();

        let logits_data = vec![0.1f32, 0.5, 0.3, 0.8, 0.2];
        let logits_d: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(logits_data.clone(), [1, 5]), &device);
        let logits_m: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(logits_data, [1, 5]), &device);

        let actions = vec![vec![3.0f32]];

        let lp_d: f32 = disc
            .log_prob(logits_d.clone(), &actions, None, None, &device)
            .into_data()
            .to_vec()
            .unwrap()[0];
        let lp_m: f32 = multi
            .log_prob(logits_m.clone(), &actions, None, None, &device)
            .into_data()
            .to_vec()
            .unwrap()[0];
        assert!(
            (lp_d - lp_m).abs() < 1e-6,
            "Discrete and MultiDiscrete(1) log_prob differ"
        );

        let ent_d: f32 = disc
            .entropy(logits_d, None, None)
            .into_data()
            .to_vec()
            .unwrap()[0];
        let ent_m: f32 = multi
            .entropy(logits_m, None, None)
            .into_data()
            .to_vec()
            .unwrap()[0];
        assert!(
            (ent_d - ent_m).abs() < 1e-6,
            "Discrete and MultiDiscrete(1) entropy differ"
        );
    }

    #[test]
    fn all_masked_except_one() {
        let dist = ActionDist::Discrete(4);
        let device = device();

        let logits: Tensor<B, 2> = Tensor::zeros([1, 4], &device);
        // Only action 2 is valid
        let mask: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(vec![0.0f32, 0.0, 1.0, 0.0], [1, 4]),
            &device,
        );

        let actions = vec![vec![2.0f32]];
        let lp: f32 = dist
            .log_prob(logits.clone(), &actions, Some(&mask), None, &device)
            .into_data()
            .to_vec()
            .unwrap()[0];
        let ent: f32 = dist
            .entropy(logits, Some(&mask), None)
            .into_data()
            .to_vec()
            .unwrap()[0];

        assert!(lp.abs() < 1e-5, "log_prob should be ~0 (prob=1), got {lp}");
        assert!(ent.abs() < 1e-5, "entropy should be ~0, got {ent}");
    }

    #[test]
    fn continuous_sample_and_log_prob() {
        let dist = ActionDist::Continuous {
            action_dim: 2,
            log_std_mode: LogStdMode::ModelOutput,
        };
        let device = device();
        let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

        // means=[1.0, -1.0], log_stds=[0.0, 0.0] → stds=[1.0, 1.0]
        let logits: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(vec![1.0f32, -1.0, 0.0, 0.0], [1, 4]),
            &device,
        );

        let actions = dist.sample(&logits, None, None, &mut rng);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].len(), 2);

        // Log prob of the sampled action
        let lp: f32 = dist
            .log_prob(
                Tensor::<B, 2>::from_data(
                    TensorData::new(vec![1.0f32, -1.0, 0.0, 0.0], [1, 4]),
                    &device,
                ),
                &actions,
                None,
                None,
                &device,
            )
            .into_data()
            .to_vec()
            .unwrap()[0];

        // Should be finite and negative
        assert!(lp.is_finite(), "log_prob should be finite");
        assert!(lp <= 0.0, "log_prob should be <= 0");
    }

    #[test]
    fn continuous_separate_log_std() {
        let dist = ActionDist::Continuous {
            action_dim: 2,
            log_std_mode: LogStdMode::Separate,
        };
        let device = device();

        // means only
        let logits: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(vec![0.0f32, 0.0], [1, 2]), &device);
        let log_std: Tensor<B, 1> =
            Tensor::from_data(TensorData::new(vec![0.0f32, 0.0], [2]), &device);

        let actions = vec![vec![0.0f32, 0.0]]; // at the mean
        let lp: f32 = dist
            .log_prob(logits.clone(), &actions, None, Some(&log_std), &device)
            .into_data()
            .to_vec()
            .unwrap()[0];

        // log_prob at mean with std=1: -0.5*ln(2π) per dim, 2 dims
        let expected = -((2.0 * PI).ln());
        assert!(
            (lp - expected).abs() < 1e-4,
            "lp {lp} != expected {expected}"
        );

        // Entropy: log_std + 0.5*(1+ln(2π)) per dim
        let ent: f32 = dist
            .entropy(logits, None, Some(&log_std))
            .into_data()
            .to_vec()
            .unwrap()[0];
        let expected_ent = 2.0 * (0.0 + 0.5 * (1.0 + (2.0 * PI).ln()));
        assert!(
            (ent - expected_ent).abs() < 1e-4,
            "entropy {ent} != expected {expected_ent}"
        );
    }
}
