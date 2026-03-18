//! Privileged (asymmetric) actor-critic (Issue #13).
//!
//! In many environments the critic can observe information that is unavailable
//! to the actor at deployment time (e.g. full game state, enemy positions,
//! fog-of-war–free maps). This module provides:
//!
//! - [`PrivilegedActorCritic`] — trait for models with asymmetric information.
//! - [`make_critic_input`] — helper to concatenate partial obs with privileged info.

use burn::prelude::*;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Trait for models with a privileged critic.
///
/// The actor sees only partial observations; the critic additionally
/// receives privileged information (e.g., full game state, enemy positions).
///
/// Implementors should use the partial observation for the actor and
/// the concatenation of partial + privileged for the critic.
pub trait PrivilegedActorCritic<B: Backend> {
    /// Actor forward pass with partial observation only.
    ///
    /// Returns logits `[batch, n_actions]`.
    fn actor_forward(&self, obs: Tensor<B, 2>) -> Tensor<B, 2>;

    /// Critic forward pass with partial observation + privileged info.
    ///
    /// Returns values `[batch]`.
    fn critic_forward(&self, obs: Tensor<B, 2>, privileged: Tensor<B, 2>) -> Tensor<B, 1>;
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Concatenate partial observation and privileged information for critic input.
///
/// # Arguments
/// * `obs` — Partial observation `[batch, obs_dim]`
/// * `privileged` — Privileged information `[batch, priv_dim]`
///
/// # Returns
/// Concatenated tensor `[batch, obs_dim + priv_dim]`
pub fn make_critic_input<B: Backend>(
    obs: Tensor<B, 2>,
    privileged: Tensor<B, 2>,
) -> Tensor<B, 2> {
    Tensor::cat(vec![obs, privileged], 1)
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

    #[test]
    fn make_critic_input_shape() {
        let obs = Tensor::<B, 2>::zeros([4, 8], &dev());
        let priv_info = Tensor::<B, 2>::zeros([4, 3], &dev());
        let combined = make_critic_input(obs, priv_info);
        assert_eq!(combined.dims(), [4, 11]);
    }

    #[test]
    fn make_critic_input_values() {
        let obs = Tensor::<B, 2>::ones([2, 3], &dev());
        let priv_info = Tensor::<B, 2>::ones([2, 2], &dev()) * 2.0;
        let combined = make_critic_input(obs, priv_info);
        let vals: Vec<f32> = combined.into_data().to_vec().unwrap();
        // Row 0: [1, 1, 1, 2, 2]
        assert_eq!(vals.len(), 10);
        assert!((vals[0] - 1.0).abs() < 1e-6);
        assert!((vals[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn single_sample() {
        let obs = Tensor::<B, 2>::zeros([1, 5], &dev());
        let priv_info = Tensor::<B, 2>::zeros([1, 7], &dev());
        let combined = make_critic_input(obs, priv_info);
        assert_eq!(combined.dims(), [1, 12]);
    }
}
