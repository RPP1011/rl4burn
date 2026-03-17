//! PPO with multi-discrete action spaces and action masking.
//!
//! Generalized versions of `ppo_collect` / `ppo_update` that work with
//! any `ActionDist` (Discrete, MultiDiscrete, or Continuous) and optional
//! per-step masks.
//!
//! For simple discrete without masking, the original `ppo_collect`/`ppo_update`
//! from `algo::ppo` are more ergonomic.

use crate::algo::ppo::{PpoConfig, PpoStats};
use crate::collect::gae;
use crate::env::vec_env::SyncVecEnv;
use crate::env::Env;
use crate::nn::clip::clip_grad_norm;
use crate::nn::dist::ActionDist;

use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::relu;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::TensorData;

use rand::Rng;

// ---------------------------------------------------------------------------
// Model trait
// ---------------------------------------------------------------------------

/// Actor-critic model for masked/multi-discrete/continuous PPO.
///
/// The model outputs flat logits and values. The `ActionDist` handles
/// interpretation of logits as one or more categorical or continuous distributions.
pub trait MaskedActorCritic<B: Backend> {
    /// Forward pass. Returns `(logits [batch, n_logits], values [batch])`.
    fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>);

    /// For `Continuous`/`Separate` mode: return the learnable log_std parameter.
    ///
    /// Default: `None` (discrete models don't need this).
    fn log_std(&self) -> Option<Tensor<B, 1>> {
        None
    }
}

/// Convenience macro-free wrapper: forwards `DiscreteActorCritic` to `MaskedActorCritic`.
///
/// If your model already implements `DiscreteActorCritic`, you can implement
/// `MaskedActorCritic` with a trivial delegation:
///
/// ```ignore
/// impl<B: Backend> MaskedActorCritic<B> for MyModel<B> {
///     fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
///         let out = DiscreteActorCritic::forward(self, obs);
///         (out.logits, out.values)
///     }
/// }
/// ```

// ---------------------------------------------------------------------------
// Rollout data
// ---------------------------------------------------------------------------

/// Collected experience from a masked/multi-discrete PPO rollout.
pub struct MaskedPpoRollout {
    /// Observations at each step. `[total_steps][obs_dim]`.
    pub observations: Vec<Vec<f32>>,
    /// Actions per step: `[total_steps][n_action_dims]`.
    /// Values are integer-valued f32 for discrete, float for continuous.
    pub actions: Vec<Vec<f32>>,
    /// Summed log-probs across all action dimensions.
    pub log_probs: Vec<f32>,
    /// Value estimates V(s_t).
    pub values: Vec<f32>,
    /// Rewards received.
    pub rewards: Vec<f32>,
    /// Whether the episode ended at each step.
    pub dones: Vec<bool>,
    /// Per-step action masks. `None` if env doesn't provide masks.
    /// When present: `[total_steps][n_logits]`, `1.0`=valid `0.0`=invalid.
    pub masks: Option<Vec<Vec<f32>>>,
    /// GAE advantages.
    pub advantages: Vec<f32>,
    /// Value targets = advantages + values.
    pub returns: Vec<f32>,
    /// Completed episode returns during this rollout.
    pub episode_returns: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Rollout collection
// ---------------------------------------------------------------------------

/// Collect a rollout from vectorized environments using masked/multi-discrete PPO.
///
/// Works identically to `ppo_collect` but supports any `ActionDist` and optional
/// per-step action masks from the environment.
///
/// `current_obs` holds the current per-env observations. Initialize with
/// `envs.reset()` before the first call; the collect function updates it
/// at the end so subsequent calls continue where the last rollout left off.
///
/// `episode_returns_acc` tracks per-env cumulative reward across rollout boundaries.
/// Initialize to `vec![0.0; n_envs]` before the first call.
pub fn masked_ppo_collect<B, M, E>(
    model: &M,
    envs: &mut SyncVecEnv<E>,
    action_dist: &ActionDist,
    config: &PpoConfig,
    device: &B::Device,
    rng: &mut impl Rng,
    current_obs: &mut Vec<Vec<f32>>,
    episode_returns_acc: &mut Vec<f32>,
) -> MaskedPpoRollout
where
    B: Backend,
    M: MaskedActorCritic<B>,
    E: Env<Observation = Vec<f32>, Action = Vec<f32>>,
{
    let n_envs = envs.num_envs();
    let obs_dim = match envs.observation_space() {
        crate::env::space::Space::Box { ref low, .. } => low.len(),
        s => s.flat_dim(),
    };
    let n_logits = action_dist.n_logits();
    let n_steps = config.n_steps;
    let total = n_steps * n_envs;

    let mut observations = Vec::with_capacity(total);
    let mut actions_vec: Vec<Vec<f32>> = Vec::with_capacity(total);
    let mut log_probs_vec = Vec::with_capacity(total);
    let mut values_vec = Vec::with_capacity(total);
    let mut rewards = Vec::with_capacity(total);
    let mut dones = Vec::with_capacity(total);
    let mut stored_masks: Option<Vec<Vec<f32>>> = None;
    let mut episode_returns = Vec::new();

    if episode_returns_acc.is_empty() {
        episode_returns_acc.resize(n_envs, 0.0);
    }

    for _step in 0..n_steps {
        // Store current observations
        for obs in current_obs.iter() {
            observations.push(obs.clone());
        }

        // Collect masks from environments
        let env_masks = envs.action_masks();
        let mask_tensor: Option<Tensor<B, 2>> = env_masks.as_ref().map(|masks| {
            let flat: Vec<f32> = masks.iter().flat_map(|m| m.iter().copied()).collect();
            Tensor::from_data(TensorData::new(flat, [n_envs, n_logits]), device)
        });

        // Store masks for this step
        if let Some(ref masks) = env_masks {
            let sm = stored_masks.get_or_insert_with(Vec::new);
            for mask in masks {
                sm.push(mask.clone());
            }
        }

        // Build observation tensor [n_envs, obs_dim]
        let obs_flat: Vec<f32> = current_obs.iter().flat_map(|o| o.iter().copied()).collect();
        let obs_tensor: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(obs_flat, [n_envs, obs_dim]), device);

        // Forward pass
        let (logits, values) = model.forward(obs_tensor);
        let model_log_std = model.log_std();

        // Extract values to CPU
        let vals_data: Vec<f32> = values.into_data().to_vec().unwrap();

        // Sample actions
        let step_actions = action_dist.sample(
            &logits,
            mask_tensor.as_ref(),
            model_log_std.as_ref(),
            rng,
        );

        // Compute log-probs (no autodiff needed during collection)
        let lp_tensor = action_dist.log_prob(
            logits,
            &step_actions,
            mask_tensor.as_ref(),
            model_log_std.as_ref(),
            device,
        );
        let lp_data: Vec<f32> = lp_tensor.into_data().to_vec().unwrap();

        // Store per-env data
        for i in 0..n_envs {
            actions_vec.push(step_actions[i].clone());
            log_probs_vec.push(lp_data[i]);
            values_vec.push(vals_data[i]);
        }

        // Step environments
        let steps = envs.step(step_actions);
        for (env_idx, step) in steps.iter().enumerate() {
            rewards.push(step.reward);
            dones.push(step.done());
            episode_returns_acc[env_idx] += step.reward;
            if step.done() {
                episode_returns.push(episode_returns_acc[env_idx]);
                episode_returns_acc[env_idx] = 0.0;
            }
        }

        *current_obs = steps.into_iter().map(|s| s.observation).collect();
    }

    // Bootstrap value for GAE
    let obs_flat: Vec<f32> = current_obs.iter().flat_map(|o| o.iter().copied()).collect();
    let obs_tensor: Tensor<B, 2> =
        Tensor::from_data(TensorData::new(obs_flat, [n_envs, obs_dim]), device);
    let (_, last_values_tensor) = model.forward(obs_tensor);
    let last_values: Vec<f32> = last_values_tensor.into_data().to_vec().unwrap();

    // Compute GAE for each environment's trajectory slice
    let mut advantages = vec![0.0f32; total];
    let mut returns = vec![0.0f32; total];

    for env_idx in 0..n_envs {
        let env_rewards: Vec<f32> = (0..n_steps).map(|s| rewards[s * n_envs + env_idx]).collect();
        let env_values: Vec<f32> =
            (0..n_steps).map(|s| values_vec[s * n_envs + env_idx]).collect();
        let env_dones: Vec<bool> = (0..n_steps).map(|s| dones[s * n_envs + env_idx]).collect();

        let (env_adv, env_ret) = gae::gae(
            &env_rewards,
            &env_values,
            &env_dones,
            last_values[env_idx],
            config.gamma,
            config.gae_lambda,
        );

        for s in 0..n_steps {
            advantages[s * n_envs + env_idx] = env_adv[s];
            returns[s * n_envs + env_idx] = env_ret[s];
        }
    }

    MaskedPpoRollout {
        observations,
        actions: actions_vec,
        log_probs: log_probs_vec,
        values: values_vec,
        rewards,
        dones,
        masks: stored_masks,
        advantages,
        returns,
        episode_returns,
    }
}

// ---------------------------------------------------------------------------
// PPO update
// ---------------------------------------------------------------------------

/// Perform a PPO update with masked/multi-discrete/continuous action distributions.
///
/// Works identically to `ppo_update` but uses `ActionDist` for log-prob and entropy
/// computation, supporting multi-discrete and continuous action spaces with optional masking.
pub fn masked_ppo_update<B, M, O>(
    mut model: M,
    optim: &mut O,
    rollout: &MaskedPpoRollout,
    action_dist: &ActionDist,
    config: &PpoConfig,
    current_lr: f64,
    device: &B::Device,
    rng: &mut impl Rng,
) -> (M, PpoStats)
where
    B: AutodiffBackend,
    M: MaskedActorCritic<B> + burn::module::AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    let total = rollout.observations.len();
    let obs_dim = rollout.observations[0].len();
    let n_logits = action_dist.n_logits();

    let mut total_policy_loss = 0.0f32;
    let mut total_value_loss = 0.0f32;
    let mut total_entropy = 0.0f32;
    let mut total_kl = 0.0f32;
    let mut n_updates = 0u32;

    let mut indices: Vec<usize> = (0..total).collect();

    for _epoch in 0..config.update_epochs {
        // Fisher-Yates shuffle
        for i in (1..indices.len()).rev() {
            let j = rng.random_range(0..=i);
            indices.swap(i, j);
        }

        for start in (0..total).step_by(config.minibatch_size) {
            let end = (start + config.minibatch_size).min(total);
            let batch_size = end - start;
            if batch_size == 0 {
                continue;
            }

            let batch_indices = &indices[start..end];

            // Build batch tensors
            let obs_flat: Vec<f32> = batch_indices
                .iter()
                .flat_map(|&i| rollout.observations[i].iter().copied())
                .collect();
            let obs_tensor: Tensor<B, 2> =
                Tensor::from_data(TensorData::new(obs_flat, [batch_size, obs_dim]), device);

            let batch_actions: Vec<Vec<f32>> = batch_indices
                .iter()
                .map(|&i| rollout.actions[i].clone())
                .collect();

            let old_lp_data: Vec<f32> =
                batch_indices.iter().map(|&i| rollout.log_probs[i]).collect();
            let old_lp: Tensor<B, 1> =
                Tensor::from_data(TensorData::new(old_lp_data, [batch_size]), device);

            // Per-minibatch advantage normalization (matches CleanRL)
            let raw_adv: Vec<f32> = batch_indices
                .iter()
                .map(|&i| rollout.advantages[i])
                .collect();
            let mb_mean: f32 = raw_adv.iter().sum::<f32>() / batch_size as f32;
            let mb_var: f32 =
                raw_adv.iter().map(|a| (a - mb_mean).powi(2)).sum::<f32>() / batch_size as f32;
            let mb_std = mb_var.sqrt().max(1e-8);
            let norm_adv: Vec<f32> = raw_adv.iter().map(|a| (a - mb_mean) / mb_std).collect();
            let adv_tensor: Tensor<B, 1> =
                Tensor::from_data(TensorData::new(norm_adv, [batch_size]), device);

            let ret_data: Vec<f32> =
                batch_indices.iter().map(|&i| rollout.returns[i]).collect();
            let ret_tensor: Tensor<B, 1> =
                Tensor::from_data(TensorData::new(ret_data, [batch_size]), device);

            let old_val_data: Vec<f32> =
                batch_indices.iter().map(|&i| rollout.values[i]).collect();
            let old_values: Tensor<B, 1> =
                Tensor::from_data(TensorData::new(old_val_data, [batch_size]), device);

            // Reconstruct mask tensor for this minibatch
            let mask_tensor: Option<Tensor<B, 2>> = rollout.masks.as_ref().map(|masks| {
                let flat: Vec<f32> = batch_indices
                    .iter()
                    .flat_map(|&i| masks[i].iter().copied())
                    .collect();
                Tensor::from_data(TensorData::new(flat, [batch_size, n_logits]), device)
            });

            // Forward pass
            let (logits, new_values) = model.forward(obs_tensor);
            let model_log_std = model.log_std();

            // New log-probs and entropy via ActionDist
            let new_lp = action_dist.log_prob(
                logits.clone(),
                &batch_actions,
                mask_tensor.as_ref(),
                model_log_std.as_ref(),
                device,
            );
            let entropy: Tensor<B, 1> = action_dist
                .entropy(logits, mask_tensor.as_ref(), model_log_std.as_ref())
                .mean()
                .unsqueeze();

            // Policy ratio
            let log_ratio = new_lp - old_lp;
            let ratio = log_ratio.clone().exp();

            // Approximate KL divergence for monitoring
            let approx_kl_val: f32 = log_ratio
                .powf_scalar(2.0)
                .mean()
                .into_data()
                .to_vec::<f32>()
                .unwrap()[0]
                * 0.5;

            // Clipped surrogate objective
            let surr1 = ratio.clone() * adv_tensor.clone();
            let surr2 =
                ratio.clamp(1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv_tensor;
            let min_surr = surr2.clone() - relu(surr2 - surr1);
            let policy_loss: Tensor<B, 1> = min_surr.mean().neg().unsqueeze();

            // Value loss with optional clipping
            let value_loss: Tensor<B, 1> = if config.clip_vloss {
                let v_clipped = old_values.clone()
                    + (new_values.clone() - old_values)
                        .clamp(-config.clip_eps, config.clip_eps);
                let v_loss_unclipped = (new_values - ret_tensor.clone()).powf_scalar(2.0);
                let v_loss_clipped = (v_clipped - ret_tensor).powf_scalar(2.0);
                let v_loss_max =
                    v_loss_unclipped.clone() + relu(v_loss_clipped - v_loss_unclipped);
                (v_loss_max.mean() * 0.5).unsqueeze()
            } else {
                ((new_values - ret_tensor).powf_scalar(2.0).mean() * 0.5).unsqueeze()
            };

            // Total loss
            let loss = policy_loss.clone()
                + value_loss.clone() * config.vf_coef
                - entropy.clone() * config.ent_coef;

            // Track stats
            total_policy_loss += policy_loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            total_value_loss += value_loss.clone().into_data().to_vec::<f32>().unwrap()[0];
            total_entropy += entropy.clone().into_data().to_vec::<f32>().unwrap()[0];
            total_kl += approx_kl_val;
            n_updates += 1;

            // Backward pass with global gradient norm clipping
            let grads = loss.backward();
            let mut grads = GradientsParams::from_grads(grads, &model);
            if config.max_grad_norm > 0.0 {
                grads = clip_grad_norm(&model, grads, config.max_grad_norm);
            }
            model = optim.step(current_lr, model, grads);
        }
    }

    let n = n_updates.max(1) as f32;
    let stats = PpoStats {
        policy_loss: total_policy_loss / n,
        value_loss: total_value_loss / n,
        entropy: total_entropy / n,
        approx_kl: total_kl / n,
    };

    (model, stats)
}
