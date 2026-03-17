//! Loss functions for RL training.
//!
//! Backend-generic — works with any Burn backend.
//!
//! All loss functions return a `Tensor<B, 1>` with shape [1] (scalar).
//! This is compatible with `.backward()` for autodiff backends.

use burn::prelude::*;
use burn::tensor::activation::log_softmax;

/// Advantage-weighted regression loss for deterministic continuous actions.
///
/// `loss = mean(clamp(advantage, 0, ∞) * ||pred - target||²)`
///
/// Only positive advantages contribute gradient (reinforce good actions).
/// Negative advantage + MSE is degenerate (pushes away from all positions).
pub fn policy_loss_continuous<B: Backend>(
    pred: Tensor<B, 2>,      // [B, action_dim]
    target: Tensor<B, 2>,    // [B, action_dim]
    advantage: Tensor<B, 1>, // [B]
) -> Tensor<B, 1> {
    let mse: Tensor<B, 1> = (pred - target).powf_scalar(2.0)
        .sum_dim(1).squeeze_dim::<1>(1);
    let pos_adv = advantage.clamp_min(0.0);
    (mse * pos_adv).mean().unsqueeze()
}

/// Policy gradient loss for discrete actions (standard REINFORCE).
///
/// `loss = -mean(advantage * log_prob(action))`
///
/// `logits` are raw (pre-softmax). `actions` are indices of taken actions.
/// `mask` is [B, n_actions] with 1.0 for valid, 0.0 for invalid.
pub fn policy_loss_discrete<B: Backend>(
    logits: Tensor<B, 2>,     // [B, n_actions]
    actions: Tensor<B, 2, Int>, // [B, 1] action indices
    mask: Tensor<B, 2>,       // [B, n_actions] valid=1.0
    advantage: Tensor<B, 1>,  // [B]
) -> Tensor<B, 1> {
    let masked = logits + (mask - 1.0) * 1e9;
    let log_probs = log_softmax(masked, 1);
    let action_lp: Tensor<B, 1> = log_probs.gather(1, actions).squeeze_dim::<1>(1);
    (action_lp * advantage).mean().neg().unsqueeze()
}

/// Huber value loss (smooth L1), δ=1.0.
///
/// - |x| ≤ 1: 0.5 * x²
/// - |x| > 1: |x| - 0.5
///
/// Bounds per-sample gradient to 1.0, preventing outlier targets from
/// dominating the value head update.
pub fn value_loss<B: Backend>(
    pred: Tensor<B, 1>,
    target: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let delta = (pred - target).abs();
    let quadratic = delta.clone().clamp_max(1.0).powf_scalar(2.0) * 0.5;
    let linear = (delta - 1.0).clamp_min(0.0);
    (quadratic + linear).mean().unsqueeze()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    type B = NdArray;

    #[test]
    fn huber_zero_error() {
        let dev = Default::default();
        let pred = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &dev);
        let target = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &dev);
        let loss = value_loss(pred, target);
        assert!(loss.into_scalar().abs() < 1e-6);
    }

    #[test]
    fn huber_small_error_is_quadratic() {
        let dev = Default::default();
        let pred = Tensor::<B, 1>::from_floats([0.0], &dev);
        let target = Tensor::<B, 1>::from_floats([0.5], &dev);
        let loss: f32 = value_loss(pred, target).into_scalar();
        let expected = 0.5 * 0.5 * 0.5; // 0.5 * x²
        assert!((loss - expected).abs() < 1e-5, "loss={loss}, expected={expected}");
    }

    #[test]
    fn huber_large_error_is_linear() {
        let dev = Default::default();
        let pred = Tensor::<B, 1>::from_floats([0.0], &dev);
        let target = Tensor::<B, 1>::from_floats([3.0], &dev);
        let loss: f32 = value_loss(pred, target).into_scalar();
        let expected = 3.0 - 0.5; // |x| - 0.5
        assert!((loss - expected).abs() < 1e-5, "loss={loss}, expected={expected}");
    }

    #[test]
    fn huber_less_than_mse_for_large_errors() {
        let dev = Default::default();
        let _pred = Tensor::<B, 1>::from_floats([0.0], &dev);
        let _target = Tensor::<B, 1>::from_floats([5.0], &dev);
        let huber: f32 = value_loss(
            Tensor::<B, 1>::from_floats([0.0], &dev),
            Tensor::<B, 1>::from_floats([5.0], &dev),
        ).into_scalar();
        let mse = 0.5 * 25.0;
        assert!(huber < mse, "huber={huber} should be < mse={mse}");
    }

    #[test]
    fn continuous_loss_zero_when_pred_equals_target() {
        let dev = Default::default();
        let pred = Tensor::<B, 2>::from_floats([[1.0, 2.0]], &dev);
        let target = Tensor::<B, 2>::from_floats([[1.0, 2.0]], &dev);
        let adv = Tensor::<B, 1>::from_floats([1.0], &dev);
        let loss: f32 = policy_loss_continuous(pred, target, adv).into_scalar();
        assert!(loss.abs() < 1e-6);
    }

    #[test]
    fn continuous_loss_zero_for_negative_advantage() {
        let dev = Default::default();
        let pred = Tensor::<B, 2>::from_floats([[0.0, 0.0]], &dev);
        let target = Tensor::<B, 2>::from_floats([[10.0, 10.0]], &dev);
        let adv = Tensor::<B, 1>::from_floats([-5.0], &dev);
        let loss: f32 = policy_loss_continuous(pred, target, adv).into_scalar();
        assert!(loss.abs() < 1e-6, "negative advantage should give zero loss, got {loss}");
    }

    #[test]
    fn discrete_loss_is_scalar() {
        let dev = Default::default();
        let logits = Tensor::<B, 2>::from_floats([[1.0, 2.0, 3.0]], &dev);
        let actions = Tensor::<B, 2, Int>::from_ints([[1i32]], &dev);
        let mask = Tensor::<B, 2>::from_floats([[1.0, 1.0, 1.0]], &dev);
        let adv = Tensor::<B, 1>::from_floats([1.0], &dev);
        let loss = policy_loss_discrete(logits, actions, mask, adv);
        assert_eq!(loss.dims(), [1]);
    }
}
