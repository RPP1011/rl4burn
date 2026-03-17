//! Goal-conditioned RL with strategy embeddings (Issue #19).
//!
//! Z-conditioning augments the policy input with a strategy embedding `z`
//! that specifies *what* the agent should do (e.g. "rush", "expand",
//! "defend"). The embedding is projected via a learned linear layer and
//! concatenated with the observation before being fed to the policy network.
//!
//! Includes a pseudo-reward function [`z_reward`] that measures how well
//! the agent's observed behavior statistics match the target strategy `z`.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for z-conditioning (goal-conditioned RL).
#[derive(Config, Debug)]
pub struct ZConditioningConfig {
    /// Dimension of the strategy embedding z.
    pub z_dim: usize,
    /// Dimension of the observation space.
    pub obs_dim: usize,
    /// Hidden dimension for the z encoder.
    #[config(default = 64)]
    pub hidden_dim: usize,
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// Z-conditioning module: projects strategy embedding and concatenates with obs.
///
/// Given observation `o` and strategy embedding `z`, produces
/// `concat(o, proj(z))` as input to the policy network.
///
/// The output dimension is `obs_dim + hidden_dim`.
#[derive(Module, Debug)]
pub struct ZConditioning<B: Backend> {
    z_proj: Linear<B>,
    #[module(skip)]
    obs_dim: usize,
    #[module(skip)]
    hidden_dim: usize,
}

impl ZConditioningConfig {
    /// Initialize a z-conditioning module on the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ZConditioning<B> {
        ZConditioning {
            z_proj: LinearConfig::new(self.z_dim, self.hidden_dim).init(device),
            obs_dim: self.obs_dim,
            hidden_dim: self.hidden_dim,
        }
    }
}

impl<B: Backend> ZConditioning<B> {
    /// Combine observation and strategy embedding.
    ///
    /// # Arguments
    /// * `obs` — Partial observation `[batch, obs_dim]`
    /// * `z` — Strategy embedding `[batch, z_dim]`
    ///
    /// # Returns
    /// Concatenated tensor `[batch, obs_dim + hidden_dim]`.
    pub fn forward(&self, obs: Tensor<B, 2>, z: Tensor<B, 2>) -> Tensor<B, 2> {
        let z_embed = self.z_proj.forward(z);
        Tensor::cat(vec![obs, z_embed], 1)
    }

    /// Output dimension after conditioning: `obs_dim + hidden_dim`.
    pub fn output_dim(&self) -> usize {
        self.obs_dim + self.hidden_dim
    }
}

// ---------------------------------------------------------------------------
// Z-reward
// ---------------------------------------------------------------------------

/// Compute z-based pseudo-reward for strategy following.
///
/// Measures how well the agent's behavior matches the target strategy `z`
/// using negative L2 distance: `reward = -||observed - target||₂`.
///
/// - Zero distance → zero reward (best possible).
/// - Nonzero distance → negative reward.
///
/// # Arguments
/// * `observed_stats` — Observed behavior statistics (e.g. resource counts).
/// * `target_z` — Target strategy embedding to match.
///
/// # Panics
/// Panics if slices have different lengths.
pub fn z_reward(observed_stats: &[f32], target_z: &[f32]) -> f32 {
    assert_eq!(
        observed_stats.len(),
        target_z.len(),
        "observed_stats and target_z must have the same length"
    );
    let l2_sq: f32 = observed_stats
        .iter()
        .zip(target_z.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    -l2_sq.sqrt()
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

    // -- ZConditioning shape -------------------------------------------------

    #[test]
    fn output_shape() {
        let cfg = ZConditioningConfig::new(8, 16);
        let module = cfg.init::<B>(&dev());
        let obs = Tensor::<B, 2>::zeros([4, 16], &dev());
        let z = Tensor::<B, 2>::zeros([4, 8], &dev());
        let output = module.forward(obs, z);
        assert_eq!(output.dims(), [4, 16 + 64]); // obs_dim + default hidden_dim
    }

    #[test]
    fn output_shape_custom_hidden() {
        let cfg = ZConditioningConfig::new(4, 10).with_hidden_dim(32);
        let module = cfg.init::<B>(&dev());
        let obs = Tensor::<B, 2>::zeros([2, 10], &dev());
        let z = Tensor::<B, 2>::zeros([2, 4], &dev());
        let output = module.forward(obs, z);
        assert_eq!(output.dims(), [2, 10 + 32]);
    }

    #[test]
    fn output_dim_method() {
        let cfg = ZConditioningConfig::new(8, 16).with_hidden_dim(32);
        let module = cfg.init::<B>(&dev());
        assert_eq!(module.output_dim(), 16 + 32);
    }

    #[test]
    fn single_sample() {
        let cfg = ZConditioningConfig::new(3, 5);
        let module = cfg.init::<B>(&dev());
        let obs = Tensor::<B, 2>::zeros([1, 5], &dev());
        let z = Tensor::<B, 2>::zeros([1, 3], &dev());
        let output = module.forward(obs, z);
        assert_eq!(output.dims(), [1, 5 + 64]);
    }

    // -- z_reward ------------------------------------------------------------

    #[test]
    fn z_reward_zero_distance() {
        let stats = vec![1.0, 2.0, 3.0];
        let target = vec![1.0, 2.0, 3.0];
        let r = z_reward(&stats, &target);
        assert!(r.abs() < 1e-6, "zero distance should give zero reward, got {r}");
    }

    #[test]
    fn z_reward_nonzero_distance() {
        let stats = vec![0.0, 0.0];
        let target = vec![3.0, 4.0];
        let r = z_reward(&stats, &target);
        assert!(r < 0.0, "nonzero distance should give negative reward, got {r}");
        // L2 = sqrt(9+16) = 5.0
        assert!((r - (-5.0)).abs() < 1e-5, "expected -5.0, got {r}");
    }

    #[test]
    fn z_reward_closer_is_better() {
        let target = vec![1.0, 0.0];
        let close = vec![0.9, 0.1];
        let far = vec![5.0, 5.0];
        let r_close = z_reward(&close, &target);
        let r_far = z_reward(&far, &target);
        assert!(
            r_close > r_far,
            "closer should have higher (less negative) reward: close={r_close}, far={r_far}"
        );
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn z_reward_mismatched_lengths() {
        z_reward(&[1.0], &[1.0, 2.0]);
    }

    // -- config --------------------------------------------------------------

    #[test]
    fn config_defaults() {
        let cfg = ZConditioningConfig::new(8, 16);
        assert_eq!(cfg.z_dim, 8);
        assert_eq!(cfg.obs_dim, 16);
        assert_eq!(cfg.hidden_dim, 64);
    }
}
