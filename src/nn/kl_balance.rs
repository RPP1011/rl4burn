//! KL balancing with free bits (DreamerV3 style).
//!
//! DreamerV3's KL loss has two terms with stop-gradients applied to different
//! arguments:
//!
//! - **Dynamics loss**: `max(free_bits, KL(sg(posterior) || prior)) * dyn_weight`
//!   — trains the prior to match the posterior.
//! - **Representation loss**: `max(free_bits, KL(posterior || sg(prior))) * rep_weight`
//!   — trains the posterior to be predictable.
//!
//! Where `sg()` means stop-gradient (`detach` in autodiff).
//!
//! Reference: Hafner et al., "Mastering Diverse Domains through World Models"
//! (DreamerV3), 2023.

use burn::prelude::*;
use burn::tensor::activation::{log_softmax, softmax};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for KL balancing with free bits (DreamerV3).
#[derive(Debug, Clone)]
pub struct KlBalanceConfig {
    /// Weight for dynamics loss (trains prior to match posterior). Default: 0.5
    pub dyn_weight: f32,
    /// Weight for representation loss (trains posterior to be predictable). Default: 0.1
    pub rep_weight: f32,
    /// Free bits threshold in nats. KL below this is ignored. Default: 1.0
    pub free_bits: f32,
}

impl Default for KlBalanceConfig {
    fn default() -> Self {
        Self {
            dyn_weight: 0.5,
            rep_weight: 0.1,
            free_bits: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Categorical KL divergence
// ---------------------------------------------------------------------------

/// Compute KL divergence between two categorical distributions.
///
/// Both inputs are logits (pre-softmax) of shape `[batch, n_categories]`.
/// Returns per-sample KL divergence `[batch]`.
pub fn categorical_kl<B: Backend>(
    p_logits: Tensor<B, 2>,
    q_logits: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let p = softmax(p_logits.clone(), 1);
    let log_p = log_softmax(p_logits, 1);
    let log_q = log_softmax(q_logits, 1);

    // KL(P || Q) = sum(p * (log_p - log_q), dim=1)
    let kl_per_category = p * (log_p - log_q);
    kl_per_category.sum_dim(1).squeeze_dim::<1>(1) // [batch]
}

/// Compute KL divergence for DreamerV3-style categorical distributions.
///
/// For RSSM with 32 groups x 32 classes, logits have shape `[batch, 32, 32]`.
/// Computes KL per group and sums across groups.
///
/// * `p_logits` and `q_logits`: `[batch, n_groups, n_classes]`
/// * Returns: `[batch]` (sum of per-group KL)
pub fn categorical_kl_groups<B: Backend>(
    p_logits: Tensor<B, 3>,
    q_logits: Tensor<B, 3>,
) -> Tensor<B, 1> {
    let [batch, n_groups, n_classes] = p_logits.dims();

    // Reshape to [batch * n_groups, n_classes] for softmax
    let p_flat = p_logits.reshape([batch * n_groups, n_classes]);
    let q_flat = q_logits.reshape([batch * n_groups, n_classes]);

    let p = softmax(p_flat.clone(), 1);
    let log_p = log_softmax(p_flat, 1);
    let log_q = log_softmax(q_flat, 1);

    let kl_flat = (p * (log_p - log_q))
        .sum_dim(1)
        .squeeze_dim::<1>(1); // [batch * n_groups]
    let kl_groups = kl_flat.reshape([batch, n_groups]); // [batch, n_groups]
    kl_groups.sum_dim(1).squeeze_dim::<1>(1) // [batch] — sum across groups
}

// ---------------------------------------------------------------------------
// KL-balanced loss
// ---------------------------------------------------------------------------

/// Compute KL-balanced loss with free bits (DreamerV3).
///
/// Uses stop-gradients to ensure:
/// - Dynamics loss only updates the prior (posterior is detached)
/// - Representation loss only updates the posterior (prior is detached)
///
/// Both inputs are logits of shape `[batch, n_categories]`.
///
/// # Arguments
/// * `posterior_logits` - Encoder output logits `[batch, n_categories]`
/// * `prior_logits` - Dynamics predictor output logits `[batch, n_categories]`
/// * `config` - KL balance configuration
///
/// # Returns
/// Scalar loss tensor `[1]`
pub fn kl_balanced_loss<B: Backend>(
    posterior_logits: Tensor<B, 2>,
    prior_logits: Tensor<B, 2>,
    config: &KlBalanceConfig,
) -> Tensor<B, 1> {
    // Dynamics loss: KL(sg(posterior) || prior) — trains prior
    let dyn_kl = categorical_kl(
        posterior_logits.clone().detach(), // stop-gradient on posterior
        prior_logits.clone(),
    );
    let dyn_loss = dyn_kl.clamp_min(config.free_bits).mean(); // free bits + average

    // Representation loss: KL(posterior || sg(prior)) — trains posterior
    let rep_kl = categorical_kl(
        posterior_logits,
        prior_logits.detach(), // stop-gradient on prior
    );
    let rep_loss = rep_kl.clamp_min(config.free_bits).mean(); // free bits + average

    // Weighted sum
    dyn_loss * config.dyn_weight + rep_loss * config.rep_weight
}

/// Same as [`kl_balanced_loss`] but for grouped categoricals (e.g., RSSM 32x32).
///
/// # Arguments
/// * `posterior_logits` - `[batch, n_groups, n_classes]`
/// * `prior_logits` - `[batch, n_groups, n_classes]`
/// * `config` - KL balance configuration
///
/// # Returns
/// Scalar loss tensor `[1]`
pub fn kl_balanced_loss_groups<B: Backend>(
    posterior_logits: Tensor<B, 3>,
    prior_logits: Tensor<B, 3>,
    config: &KlBalanceConfig,
) -> Tensor<B, 1> {
    let dyn_kl = categorical_kl_groups(
        posterior_logits.clone().detach(),
        prior_logits.clone(),
    );
    let dyn_loss = dyn_kl.clamp_min(config.free_bits).mean();

    let rep_kl = categorical_kl_groups(
        posterior_logits,
        prior_logits.detach(),
    );
    let rep_loss = rep_kl.clamp_min(config.free_bits).mean();

    dyn_loss * config.dyn_weight + rep_loss * config.rep_weight
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    fn dev() -> <B as Backend>::Device {
        Default::default()
    }

    // -- categorical_kl -------------------------------------------------------

    #[test]
    fn kl_analytical_known_value() {
        // P = [0.7, 0.1, 0.1, 0.1] vs uniform Q = [0.25, 0.25, 0.25, 0.25]
        // KL(P||Q) = sum(p * ln(p/q))
        //   = 0.7*ln(0.7/0.25) + 3*0.1*ln(0.1/0.25)
        //   = 0.7*ln(2.8) + 0.3*ln(0.4)
        //   ≈ 0.7*1.02962 + 0.3*(-0.91629)
        //   ≈ 0.72073 - 0.27489 ≈ 0.44584
        let expected_kl: f32 = 0.7 * (0.7_f32 / 0.25).ln()
            + 0.1 * (0.1_f32 / 0.25).ln() * 3.0;

        // Convert probabilities to logits (log space).
        let p_logits = Tensor::<B, 2>::from_floats(
            [[0.7_f32.ln(), 0.1_f32.ln(), 0.1_f32.ln(), 0.1_f32.ln()]],
            &dev(),
        );
        let q_logits = Tensor::<B, 2>::from_floats(
            [[0.25_f32.ln(), 0.25_f32.ln(), 0.25_f32.ln(), 0.25_f32.ln()]],
            &dev(),
        );

        let kl: f32 = categorical_kl(p_logits, q_logits).into_scalar();
        assert!(
            (kl - expected_kl).abs() < 1e-4,
            "KL = {kl}, expected ≈ {expected_kl}"
        );
    }

    #[test]
    fn kl_identical_distributions_is_zero() {
        let logits = Tensor::<B, 2>::from_floats(
            [[1.0, 2.0, 3.0, 4.0]],
            &dev(),
        );
        let kl: f32 = categorical_kl(logits.clone(), logits).into_scalar();
        assert!(kl.abs() < 1e-6, "KL of identical distributions = {kl}, expected 0");
    }

    #[test]
    fn kl_is_non_negative() {
        let p = Tensor::<B, 2>::from_floats([[1.0, -1.0, 0.5, 2.0]], &dev());
        let q = Tensor::<B, 2>::from_floats([[-0.5, 1.0, 0.0, -1.0]], &dev());
        let kl: f32 = categorical_kl(p, q).into_scalar();
        assert!(kl >= -1e-6, "KL should be non-negative, got {kl}");
    }

    // -- free bits clamping ---------------------------------------------------

    #[test]
    fn free_bits_clamps_small_kl() {
        // When KL < free_bits, the clamped value should equal free_bits.
        let free_bits = 1.0_f32;
        // Two distributions with small KL (close logits).
        let p = Tensor::<B, 2>::from_floats([[0.0, 0.1, 0.0, 0.0]], &dev());
        let q = Tensor::<B, 2>::from_floats([[0.0, 0.0, 0.0, 0.0]], &dev());
        let kl: f32 = categorical_kl(p, q).into_scalar();
        assert!(kl < free_bits, "test setup: KL={kl} should be < {free_bits}");

        let kl_tensor = Tensor::<B, 1>::from_floats([kl], &dev());
        let clamped: f32 = kl_tensor.clamp_min(free_bits).into_scalar();
        assert!(
            (clamped - free_bits).abs() < 1e-6,
            "clamped = {clamped}, expected {free_bits}"
        );
    }

    #[test]
    fn free_bits_passes_large_kl() {
        // When KL > free_bits, the value passes through unchanged.
        let free_bits = 1.0_f32;
        let large_kl = 2.5_f32;
        let kl_tensor = Tensor::<B, 1>::from_floats([large_kl], &dev());
        let clamped: f32 = kl_tensor.clamp_min(free_bits).into_scalar();
        assert!(
            (clamped - large_kl).abs() < 1e-6,
            "clamped = {clamped}, expected {large_kl}"
        );
    }

    // -- kl_balanced_loss -----------------------------------------------------

    #[test]
    fn balanced_loss_is_positive_scalar() {
        let posterior = Tensor::<B, 2>::from_floats(
            [[1.0, 2.0, 0.5, -1.0], [0.0, 0.0, 3.0, 1.0]],
            &dev(),
        );
        let prior = Tensor::<B, 2>::from_floats(
            [[-0.5, 1.0, 0.0, 0.0], [1.0, -1.0, 0.5, 0.5]],
            &dev(),
        );
        let config = KlBalanceConfig::default();
        let loss = kl_balanced_loss(posterior, prior, &config);
        assert_eq!(loss.dims(), [1], "loss should be scalar [1]");
        let val: f32 = loss.into_scalar();
        assert!(val > 0.0, "balanced loss should be positive, got {val}");
    }

    #[test]
    fn balanced_loss_zero_free_bits_identical() {
        // Identical distributions with free_bits=0 should give ~0 loss.
        let logits = Tensor::<B, 2>::from_floats(
            [[1.0, 2.0, 3.0]],
            &dev(),
        );
        let config = KlBalanceConfig {
            dyn_weight: 0.5,
            rep_weight: 0.1,
            free_bits: 0.0,
        };
        let loss: f32 =
            kl_balanced_loss(logits.clone(), logits, &config).into_scalar();
        assert!(loss.abs() < 1e-5, "identical dists should give ~0 loss, got {loss}");
    }

    // -- grouped KL -----------------------------------------------------------

    #[test]
    fn grouped_kl_shape_and_positive() {
        let batch = 3;
        let n_groups = 4;
        let n_classes = 8;
        let p = Tensor::<B, 3>::random(
            [batch, n_groups, n_classes],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &dev(),
        );
        let q = Tensor::<B, 3>::random(
            [batch, n_groups, n_classes],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &dev(),
        );
        let kl = categorical_kl_groups(p, q);
        assert_eq!(kl.dims(), [batch], "grouped KL should have shape [batch]");
        let vals: Vec<f32> = kl.to_data().to_vec().unwrap();
        for (i, &v) in vals.iter().enumerate() {
            assert!(v >= -1e-5, "grouped KL[{i}] = {v}, should be non-negative");
        }
    }

    #[test]
    fn grouped_kl_identical_is_zero() {
        let logits = Tensor::<B, 3>::random(
            [2, 4, 8],
            burn::tensor::Distribution::Uniform(-2.0, 2.0),
            &dev(),
        );
        let kl = categorical_kl_groups(logits.clone(), logits);
        let vals: Vec<f32> = kl.to_data().to_vec().unwrap();
        for (i, &v) in vals.iter().enumerate() {
            assert!(v.abs() < 1e-5, "grouped KL[{i}] = {v}, expected 0");
        }
    }

    #[test]
    fn grouped_balanced_loss_positive() {
        let posterior = Tensor::<B, 3>::random(
            [4, 32, 32],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &dev(),
        );
        let prior = Tensor::<B, 3>::random(
            [4, 32, 32],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &dev(),
        );
        let config = KlBalanceConfig::default();
        let loss = kl_balanced_loss_groups(posterior, prior, &config);
        assert_eq!(loss.dims(), [1]);
        let val: f32 = loss.into_scalar();
        assert!(val > 0.0, "grouped balanced loss should be positive, got {val}");
    }

    // -- autodiff smoke test --------------------------------------------------

    #[test]
    fn balanced_loss_autodiff() {
        use burn::backend::Autodiff;
        type AB = Autodiff<NdArray>;

        let dev = <AB as Backend>::Device::default();
        let posterior = Tensor::<AB, 2>::from_floats(
            [[1.0, 2.0, 0.5, -1.0], [0.0, 0.0, 3.0, 1.0]],
            &dev,
        )
        .require_grad();
        let prior = Tensor::<AB, 2>::from_floats(
            [[-0.5, 1.0, 0.0, 0.0], [1.0, -1.0, 0.5, 0.5]],
            &dev,
        )
        .require_grad();

        let config = KlBalanceConfig::default();
        let loss = kl_balanced_loss(posterior.clone(), prior.clone(), &config);
        let val: f32 = loss.clone().into_scalar();
        assert!(val > 0.0, "autodiff loss should be positive, got {val}");

        // Backward pass should not panic
        let grads = loss.backward();
        let _p_grad = posterior.grad(&grads);
        let _q_grad = prior.grad(&grads);
    }
}
