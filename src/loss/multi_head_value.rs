//! Multi-head value decomposition for reward-decomposed critics.
//!
//! Instead of a single V(s), decompose into N value heads, each estimating
//! value for a specific reward component (e.g., farming, KDA, damage).
//! Each head has its own GAE computation. The total advantage is a weighted sum.

use crate::collect::gae;

/// Configuration for multi-head value decomposition.
#[derive(Debug, Clone)]
pub struct MultiHeadValueConfig {
    /// Number of value heads.
    pub n_heads: usize,
    /// Per-head discount factors γ. If shorter than n_heads, last value is repeated.
    pub gammas: Vec<f32>,
    /// Weights for combining head advantages into a single advantage.
    /// Must sum to 1.0 (or will be normalized). Default: uniform weights.
    pub weights: Vec<f32>,
    /// GAE lambda (shared across heads, or per-head if len == n_heads).
    pub gae_lambdas: Vec<f32>,
}

impl MultiHeadValueConfig {
    /// Create config with N heads, all sharing the same gamma and lambda.
    pub fn new(n_heads: usize, gamma: f32, gae_lambda: f32) -> Self {
        Self {
            n_heads,
            gammas: vec![gamma; n_heads],
            weights: vec![1.0 / n_heads as f32; n_heads],
            gae_lambdas: vec![gae_lambda; n_heads],
        }
    }

    pub fn with_gammas(mut self, gammas: Vec<f32>) -> Self {
        self.gammas = gammas;
        self
    }

    pub fn with_weights(mut self, weights: Vec<f32>) -> Self {
        // Normalize weights to sum to 1
        let sum: f32 = weights.iter().sum();
        self.weights = weights.iter().map(|w| w / sum).collect();
        self
    }

    pub fn with_gae_lambdas(mut self, lambdas: Vec<f32>) -> Self {
        self.gae_lambdas = lambdas;
        self
    }

    fn gamma(&self, head: usize) -> f32 {
        self.gammas
            .get(head)
            .copied()
            .unwrap_or(*self.gammas.last().unwrap())
    }

    fn gae_lambda(&self, head: usize) -> f32 {
        self.gae_lambdas
            .get(head)
            .copied()
            .unwrap_or(*self.gae_lambdas.last().unwrap())
    }
}

/// Result of multi-head GAE computation.
pub struct MultiHeadGaeResult {
    /// Per-head advantages: `[n_heads][T]`
    pub per_head_advantages: Vec<Vec<f32>>,
    /// Per-head returns (advantage + value): `[n_heads][T]`
    pub per_head_returns: Vec<Vec<f32>>,
    /// Combined advantage (weighted sum): `[T]`
    pub combined_advantages: Vec<f32>,
}

/// Compute GAE separately for each value head, then combine advantages.
///
/// # Arguments
/// * `per_head_rewards` - Rewards per head: `[n_heads][T]`
/// * `per_head_values` - Value estimates per head: `[n_heads][T]`
/// * `dones` - Episode termination flags: `[T]`
/// * `per_head_last_values` - Bootstrap values per head: `[n_heads]`
/// * `config` - Multi-head value configuration
pub fn multi_head_gae(
    per_head_rewards: &[Vec<f32>],
    per_head_values: &[Vec<f32>],
    dones: &[bool],
    per_head_last_values: &[f32],
    config: &MultiHeadValueConfig,
) -> MultiHeadGaeResult {
    let n_heads = config.n_heads;
    let t = dones.len();

    let mut per_head_advantages = Vec::with_capacity(n_heads);
    let mut per_head_returns = Vec::with_capacity(n_heads);

    for head in 0..n_heads {
        let (adv, ret) = gae::gae(
            &per_head_rewards[head],
            &per_head_values[head],
            dones,
            per_head_last_values[head],
            config.gamma(head),
            config.gae_lambda(head),
        );
        per_head_advantages.push(adv);
        per_head_returns.push(ret);
    }

    // Combine advantages with weights
    let mut combined = vec![0.0f32; t];
    for step in 0..t {
        for head in 0..n_heads {
            combined[step] += config.weights[head] * per_head_advantages[head][step];
        }
    }

    MultiHeadGaeResult {
        per_head_advantages,
        per_head_returns,
        combined_advantages: combined,
    }
}

/// Compute per-head value losses (MSE).
///
/// Returns individual losses per head: `[n_heads]` as f32 values.
/// These can be weighted and summed for the total value loss.
///
/// # Arguments
/// * `per_head_predictions` - Model's value predictions per head: `[n_heads][batch]`
/// * `per_head_targets` - Target returns per head: `[n_heads][batch]`
pub fn multi_head_value_loss(
    per_head_predictions: &[Vec<f32>],
    per_head_targets: &[Vec<f32>],
) -> Vec<f32> {
    per_head_predictions
        .iter()
        .zip(per_head_targets.iter())
        .map(|(preds, targets)| {
            let n = preds.len() as f32;
            preds
                .iter()
                .zip(targets.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f32>()
                / n
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_head_matches_standard_gae() {
        let rewards = vec![1.0, 2.0, 3.0];
        let values = vec![0.5, 0.5, 0.5];
        let dones = vec![false, false, true];
        let last_value = 1.0;
        let gamma = 0.99;
        let lambda = 0.95;

        // Standard GAE
        let (expected_adv, expected_ret) =
            gae::gae(&rewards, &values, &dones, last_value, gamma, lambda);

        // Single-head multi-head GAE
        let config = MultiHeadValueConfig::new(1, gamma, lambda);
        let result = multi_head_gae(
            &[rewards],
            &[values],
            &dones,
            &[last_value],
            &config,
        );

        assert_eq!(result.per_head_advantages.len(), 1);
        for i in 0..3 {
            assert!(
                (result.per_head_advantages[0][i] - expected_adv[i]).abs() < 1e-5,
                "advantage mismatch at step {i}"
            );
            assert!(
                (result.per_head_returns[0][i] - expected_ret[i]).abs() < 1e-5,
                "return mismatch at step {i}"
            );
            assert!(
                (result.combined_advantages[i] - expected_adv[i]).abs() < 1e-5,
                "combined advantage mismatch at step {i}"
            );
        }
    }

    #[test]
    fn two_heads_different_gammas() {
        let rewards = vec![1.0, 1.0, 1.0];
        let values = vec![0.0, 0.0, 0.0];
        let dones = vec![false, false, false];

        let config = MultiHeadValueConfig::new(2, 0.99, 0.95)
            .with_gammas(vec![0.5, 0.99]);

        let result = multi_head_gae(
            &[rewards.clone(), rewards],
            &[values.clone(), values],
            &dones,
            &[0.0, 0.0],
            &config,
        );

        // With lower gamma (head 0), advantages should be smaller than high gamma (head 1)
        // because future rewards are discounted more aggressively.
        assert_eq!(result.per_head_advantages.len(), 2);
        for i in 0..3 {
            assert!(
                result.per_head_advantages[0][i] <= result.per_head_advantages[1][i],
                "head 0 (gamma=0.5) should have smaller advantages than head 1 (gamma=0.99) at step {i}"
            );
        }
        // Per-head advantages should actually differ
        assert!(
            (result.per_head_advantages[0][0] - result.per_head_advantages[1][0]).abs() > 0.01,
            "per-head advantages should differ with different gammas"
        );
    }

    #[test]
    fn uniform_weights_produce_average() {
        let rewards_a = vec![1.0, 2.0];
        let rewards_b = vec![3.0, 4.0];
        let values = vec![0.0, 0.0];
        let dones = vec![false, false];

        let config = MultiHeadValueConfig::new(2, 0.99, 0.95);
        // Default weights are uniform: [0.5, 0.5]

        let result = multi_head_gae(
            &[rewards_a, rewards_b],
            &[values.clone(), values],
            &dones,
            &[0.0, 0.0],
            &config,
        );

        for i in 0..2 {
            let expected =
                0.5 * result.per_head_advantages[0][i] + 0.5 * result.per_head_advantages[1][i];
            assert!(
                (result.combined_advantages[i] - expected).abs() < 1e-5,
                "combined should be average of per-head at step {i}"
            );
        }
    }

    #[test]
    fn custom_weights_produce_weighted_sum() {
        let rewards_a = vec![1.0, 2.0];
        let rewards_b = vec![3.0, 4.0];
        let values = vec![0.0, 0.0];
        let dones = vec![false, false];

        let config = MultiHeadValueConfig::new(2, 0.99, 0.95)
            .with_weights(vec![0.3, 0.7]);

        let result = multi_head_gae(
            &[rewards_a, rewards_b],
            &[values.clone(), values],
            &dones,
            &[0.0, 0.0],
            &config,
        );

        // Weights should be normalized: 0.3/1.0 = 0.3, 0.7/1.0 = 0.7
        for i in 0..2 {
            let expected =
                0.3 * result.per_head_advantages[0][i] + 0.7 * result.per_head_advantages[1][i];
            assert!(
                (result.combined_advantages[i] - expected).abs() < 1e-5,
                "weighted sum mismatch at step {i}"
            );
        }
    }

    #[test]
    fn done_flag_resets_all_heads() {
        let rewards = vec![1.0, 1.0, 1.0, 1.0];
        let values = vec![0.0; 4];
        let dones = vec![false, true, false, false];

        let config = MultiHeadValueConfig::new(2, 0.99, 0.95)
            .with_gammas(vec![0.5, 0.99]);

        let result = multi_head_gae(
            &[rewards.clone(), rewards],
            &[values.clone(), values],
            &dones,
            &[0.0, 0.0],
            &config,
        );

        // At the terminal step (index 1), both heads should have advantage = reward = 1.0
        // because done resets the bootstrap.
        for head in 0..2 {
            assert!(
                (result.per_head_advantages[head][1] - 1.0).abs() < 1e-5,
                "head {head} should have advantage=1.0 at terminal step"
            );
        }
    }

    #[test]
    fn value_loss_matches_manual_mse() {
        let preds = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let targets = vec![vec![1.5, 2.5, 3.5], vec![4.0, 5.0, 6.0]];

        let losses = multi_head_value_loss(&preds, &targets);

        // Head 0: mean((0.5^2 + 0.5^2 + 0.5^2) / 3) = 0.25
        assert!((losses[0] - 0.25).abs() < 1e-5, "head 0 loss mismatch");
        // Head 1: perfect predictions → 0.0
        assert!((losses[1] - 0.0).abs() < 1e-5, "head 1 loss mismatch");
    }
}
