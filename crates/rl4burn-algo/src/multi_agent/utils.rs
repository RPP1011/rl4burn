//! Multi-agent shared-weight training utilities (Issue #12).
//!
//! When multiple agents in an environment share the same neural network
//! weights, their observations can be batched together for a single forward
//! pass and then unbatched for per-agent environment stepping.

use burn::prelude::*;
use burn::tensor::TensorData;

// ---------------------------------------------------------------------------
// Batching / unbatching
// ---------------------------------------------------------------------------

/// Batch observations from multiple agents across multiple environments
/// into a single tensor for efficient shared-weight inference.
///
/// # Arguments
/// * `per_env_per_agent_obs` - `[n_envs][n_agents][obs_dim]`
///
/// # Returns
/// Flat tensor `[n_envs * n_agents, obs_dim]` and metadata for unbatching.
pub fn batch_multi_agent_obs<B: Backend>(
    per_env_per_agent_obs: &[Vec<Vec<f32>>],
    device: &B::Device,
) -> (Tensor<B, 2>, usize, usize) {
    let n_envs = per_env_per_agent_obs.len();
    let n_agents = per_env_per_agent_obs[0].len();
    let obs_dim = per_env_per_agent_obs[0][0].len();

    let flat: Vec<f32> = per_env_per_agent_obs
        .iter()
        .flat_map(|env| env.iter().flat_map(|obs| obs.iter().copied()))
        .collect();

    let tensor = Tensor::from_data(
        TensorData::new(flat, [n_envs * n_agents, obs_dim]),
        device,
    );

    (tensor, n_envs, n_agents)
}

/// Unbatch actions from shared-weight inference back to per-env, per-agent.
///
/// # Arguments
/// * `flat_actions` - Actions for all agents: `[n_envs * n_agents]`
/// * `n_envs` - Number of environments
/// * `n_agents` - Number of agents per environment
///
/// # Returns
/// `[n_envs][n_agents]` nested actions
pub fn unbatch_actions<T: Clone>(
    flat_actions: &[T],
    n_envs: usize,
    n_agents: usize,
) -> Vec<Vec<T>> {
    debug_assert_eq!(flat_actions.len(), n_envs * n_agents);
    flat_actions.chunks(n_agents).map(|chunk| chunk.to_vec()).collect()
}

// ---------------------------------------------------------------------------
// Reward broadcasting
// ---------------------------------------------------------------------------

/// Broadcast per-environment team rewards to per-agent rewards.
///
/// Typically all agents in a team share the same reward. This function
/// replicates each environment reward `n_agents` times.
pub fn broadcast_team_reward(env_rewards: &[f32], n_agents: usize) -> Vec<f32> {
    env_rewards
        .iter()
        .flat_map(|r| std::iter::repeat_n(*r, n_agents))
        .collect()
}

// ---------------------------------------------------------------------------
// Multi-agent rollout data
// ---------------------------------------------------------------------------

/// Aggregated rollout data from multiple agents for shared-weight PPO update.
///
/// When multiple agents share weights, their experiences are pooled
/// into a single dataset for the PPO update.
pub struct MultiAgentRolloutData {
    /// Flattened observations: `[total_steps * n_agents, obs_dim]`
    pub observations: Vec<Vec<f32>>,
    /// Flattened actions.
    pub actions: Vec<Vec<f32>>,
    /// Flattened log-probs.
    pub log_probs: Vec<f32>,
    /// Flattened values.
    pub values: Vec<f32>,
    /// Flattened advantages.
    pub advantages: Vec<f32>,
    /// Flattened returns.
    pub returns: Vec<f32>,
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
    fn batch_unbatch_round_trip() {
        // 2 envs, 3 agents each, obs_dim=4
        let obs = vec![
            vec![
                vec![1.0, 2.0, 3.0, 4.0],
                vec![5.0, 6.0, 7.0, 8.0],
                vec![9.0, 10.0, 11.0, 12.0],
            ],
            vec![
                vec![13.0, 14.0, 15.0, 16.0],
                vec![17.0, 18.0, 19.0, 20.0],
                vec![21.0, 22.0, 23.0, 24.0],
            ],
        ];

        let (tensor, n_envs, n_agents) = batch_multi_agent_obs::<B>(&obs, &dev());
        assert_eq!(n_envs, 2);
        assert_eq!(n_agents, 3);

        let dims = tensor.dims();
        assert_eq!(dims, [6, 4]); // 2*3=6 rows, 4 columns

        // Check values round-trip
        let data: Vec<f32> = tensor.into_data().to_vec().unwrap();
        assert_eq!(data.len(), 24);
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[4] - 5.0).abs() < 1e-6);
        assert!((data[12] - 13.0).abs() < 1e-6);
    }

    #[test]
    fn unbatch_actions_shape() {
        let flat = vec![0usize, 1, 2, 3, 4, 5];
        let result = unbatch_actions(&flat, 2, 3);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], vec![0, 1, 2]);
        assert_eq!(result[1], vec![3, 4, 5]);
    }

    #[test]
    fn broadcast_team_reward_shape() {
        let env_rewards = vec![1.0, 2.0, 3.0];
        let result = broadcast_team_reward(&env_rewards, 4);
        assert_eq!(result.len(), 12); // 3 envs * 4 agents
        assert_eq!(result[0..4], [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(result[4..8], [2.0, 2.0, 2.0, 2.0]);
        assert_eq!(result[8..12], [3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn broadcast_empty() {
        let result = broadcast_team_reward(&[], 3);
        assert!(result.is_empty());
    }
}
