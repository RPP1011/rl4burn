//! Proximal Policy Optimization (Schulman et al., 2017).
//!
//! Provides rollout collection and PPO update as composable functions.
//! The user supplies the actor-critic model, optimizer, and environments.
//!
//! Matches the CleanRL reference implementation: per-minibatch advantage
//! normalization, value loss clipping, and support for LR annealing.
//! Configure gradient clipping (`max_grad_norm=0.5`) on your optimizer.

use rl4burn_collect::gae;
use rl4burn_core::env::vec_env::SyncVecEnv;
use rl4burn_core::env::Env;
use rl4burn_nn::clip::clip_grad_norm;
use rl4burn_nn::policy::{DiscreteAcOutput, DiscreteActorCritic};

use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::{log_softmax, relu, softmax};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::TensorData;

use rand::{Rng, RngExt};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// PPO hyperparameters.
///
/// Defaults match CleanRL's `ppo.py`. Configure gradient clipping
/// (typically `max_grad_norm=0.5`) on your optimizer — e.g. via
/// `AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(0.5)))`.
#[derive(Debug, Clone)]
pub struct PpoConfig {
    /// Learning rate for the optimizer.
    pub lr: f64,
    /// Discount factor γ.
    pub gamma: f32,
    /// GAE smoothing parameter λ.
    pub gae_lambda: f32,
    /// Clipping range ε for the policy ratio.
    pub clip_eps: f32,
    /// Value loss coefficient.
    pub vf_coef: f32,
    /// Entropy bonus coefficient.
    pub ent_coef: f32,
    /// Number of optimization epochs per rollout.
    pub update_epochs: usize,
    /// Minibatch size for each gradient step.
    pub minibatch_size: usize,
    /// Number of steps to collect per environment per rollout.
    pub n_steps: usize,
    /// Whether to clip the value function loss (CleanRL default: true).
    pub clip_vloss: bool,
    /// Maximum gradient norm for global gradient clipping (0.0 = disabled).
    /// Matches PyTorch's `clip_grad_norm_` (global, not per-parameter).
    pub max_grad_norm: f32,
    /// Target KL divergence for early stopping of update epochs.
    /// When approx KL exceeds this threshold, remaining epochs are skipped.
    /// `None` disables early stopping (default). CleanRL uses `None` by default.
    pub target_kl: Option<f32>,
    /// Dual-clip coefficient for pessimistic PPO (Ye et al., 2020).
    /// When `Some(c)` with `c > 1` (typically 3.0), adds a floor of `c * advantage`
    /// to the clipped surrogate objective for negative advantages, preventing
    /// excessively negative objectives when the policy ratio deviates far.
    /// `None` disables dual clipping (default, standard PPO behavior).
    pub dual_clip_coef: Option<f32>,
}

impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            lr: 2.5e-4,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_eps: 0.2,
            vf_coef: 0.5,
            ent_coef: 0.01,
            update_epochs: 4,
            minibatch_size: 128,
            n_steps: 128,
            clip_vloss: true,
            max_grad_norm: 0.5,
            target_kl: None,
            dual_clip_coef: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Rollout data
// ---------------------------------------------------------------------------

/// Collected experience from a PPO rollout.
pub struct PpoRollout {
    /// Observations at each step. `[total_steps][obs_dim]`.
    pub observations: Vec<Vec<f32>>,
    /// Discrete actions taken (as indices). Length: `total_steps`.
    pub actions: Vec<i32>,
    /// Log probabilities of the taken actions under the collection policy.
    pub log_probs: Vec<f32>,
    /// Value estimates V(s_t). Length: `total_steps`.
    pub values: Vec<f32>,
    /// Rewards received. Length: `total_steps`.
    pub rewards: Vec<f32>,
    /// Whether the episode ended at each step.
    pub dones: Vec<bool>,
    /// GAE advantages (computed after collection).
    pub advantages: Vec<f32>,
    /// Value targets = advantages + values (computed after collection).
    pub returns: Vec<f32>,
    /// Completed episode returns during this rollout.
    /// Correctly tracks episodes that span multiple rollouts
    /// (accumulated via `episode_returns_acc`).
    pub episode_returns: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Training statistics
// ---------------------------------------------------------------------------

/// Statistics from a PPO update step.
#[derive(Debug, Clone)]
pub struct PpoStats {
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub approx_kl: f32,
}

// ---------------------------------------------------------------------------
// Categorical sampling
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Rollout collection
// ---------------------------------------------------------------------------

/// Collect a rollout from vectorized environments using the given model.
///
/// The model's backend can be anything (inference or autodiff). Observations
/// are converted to tensors for the forward pass, then extracted back to
/// plain floats. No computation graph is retained.
///
/// After stepping, GAE advantages and returns are computed.
///
/// `current_obs` holds the current per-env observations. Initialize with
/// `envs.reset()` before the first call; the collect function updates it
/// at the end so subsequent calls continue where the last rollout left off.
///
/// `episode_returns_acc` tracks per-env cumulative reward across rollout
/// boundaries. Initialize it to `vec![0.0; n_envs]` before the first call
/// and pass the same Vec on every subsequent call. Completed episode returns
/// are stored in `PpoRollout::episode_returns`.
pub fn ppo_collect<B, M, E>(
    model: &M,
    envs: &mut SyncVecEnv<E>,
    config: &PpoConfig,
    device: &B::Device,
    rng: &mut impl Rng,
    current_obs: &mut Vec<Vec<f32>>,
    episode_returns_acc: &mut Vec<f32>,
) -> PpoRollout
where
    B: Backend,
    M: DiscreteActorCritic<B>,
    E: Env<Observation = Vec<f32>, Action = usize>,
{
    let n_envs = envs.num_envs();
    let obs_dim = match envs.observation_space() {
        rl4burn_core::env::space::Space::Box { ref low, .. } => low.len(),
        s => s.flat_dim(),
    };
    let n_steps = config.n_steps;
    let total = n_steps * n_envs;

    let mut observations = Vec::with_capacity(total);
    let mut actions = Vec::with_capacity(total);
    let mut log_probs_vec = Vec::with_capacity(total);
    let mut values_vec = Vec::with_capacity(total);
    let mut rewards = Vec::with_capacity(total);
    let mut dones = Vec::with_capacity(total);
    let mut episode_returns = Vec::new();

    if episode_returns_acc.is_empty() {
        episode_returns_acc.resize(n_envs, 0.0);
    }

    for _step in 0..n_steps {
        // Store current observations
        for obs in current_obs.iter() {
            observations.push(obs.clone());
        }

        // Build observation tensor [n_envs, obs_dim]
        let obs_flat: Vec<f32> = current_obs.iter().flat_map(|o| o.iter().copied()).collect();
        let obs_tensor: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(obs_flat, [n_envs, obs_dim]), device);

        // Forward pass
        let output: DiscreteAcOutput<B> = model.forward(obs_tensor);

        // Extract logits and values to CPU
        let logits = output.logits;
        let values = output.values;

        let probs = softmax(logits.clone(), 1);
        let lp = log_softmax(logits, 1);

        let probs_data: Vec<f32> = probs.into_data().to_vec().unwrap();
        let lp_data: Vec<f32> = lp.into_data().to_vec().unwrap();
        let vals_data: Vec<f32> = values.into_data().to_vec().unwrap();

        let n_actions = probs_data.len() / n_envs;

        // Sample actions and collect log probs
        let mut step_actions = Vec::with_capacity(n_envs);
        for i in 0..n_envs {
            let row = &probs_data[i * n_actions..(i + 1) * n_actions];
            let action = sample_categorical(row, rng);
            let action_lp = lp_data[i * n_actions + action];

            step_actions.push(action);
            actions.push(action as i32);
            log_probs_vec.push(action_lp);
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

        // Update current observations
        *current_obs = steps.into_iter().map(|s| s.observation).collect();
    }

    // Bootstrap value for GAE
    let obs_flat: Vec<f32> = current_obs.iter().flat_map(|o| o.iter().copied()).collect();
    let obs_tensor: Tensor<B, 2> =
        Tensor::from_data(TensorData::new(obs_flat, [n_envs, obs_dim]), device);
    let last_output = model.forward(obs_tensor);
    let last_values: Vec<f32> = last_output.values.into_data().to_vec().unwrap();

    // Compute GAE for each environment's trajectory slice
    let mut advantages = vec![0.0f32; total];
    let mut returns = vec![0.0f32; total];

    for env_idx in 0..n_envs {
        // Extract this env's data (interleaved: step0_env0, step0_env1, ..., step1_env0, ...)
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

    PpoRollout {
        observations,
        actions,
        log_probs: log_probs_vec,
        values: values_vec,
        rewards,
        dones,
        advantages,
        returns,
        episode_returns,
    }
}

// ---------------------------------------------------------------------------
// PPO update
// ---------------------------------------------------------------------------

/// Perform a PPO update: multiple epochs of minibatch gradient descent
/// on the collected rollout.
///
/// Pass `current_lr` to support learning rate annealing (e.g. linear decay).
/// If not annealing, pass `config.lr`.
///
/// Returns the updated model and training statistics.
pub fn ppo_update<B, M, O>(
    mut model: M,
    optim: &mut O,
    rollout: &PpoRollout,
    config: &PpoConfig,
    current_lr: f64,
    device: &B::Device,
    rng: &mut impl Rng,
) -> (M, PpoStats)
where
    B: AutodiffBackend,
    M: DiscreteActorCritic<B> + burn::module::AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    let total = rollout.observations.len();
    let obs_dim = rollout.observations[0].len();

    let mut total_policy_loss = 0.0f32;
    let mut total_value_loss = 0.0f32;
    let mut total_entropy = 0.0f32;
    let mut total_kl = 0.0f32;
    let mut n_updates = 0u32;

    // Index array, shuffled each epoch to avoid systematic minibatch bias
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

            let actions_data: Vec<i32> =
                batch_indices.iter().map(|&i| rollout.actions[i]).collect();

            let old_lp_data: Vec<f32> =
                batch_indices.iter().map(|&i| rollout.log_probs[i]).collect();
            let old_lp: Tensor<B, 1> =
                Tensor::from_data(TensorData::new(old_lp_data, [batch_size]), device);

            // Per-minibatch advantage normalization (matches CleanRL — sample std)
            let raw_adv: Vec<f32> = batch_indices
                .iter()
                .map(|&i| rollout.advantages[i])
                .collect();
            let mb_mean: f32 = raw_adv.iter().sum::<f32>() / batch_size as f32;
            let mb_var: f32 = raw_adv
                .iter()
                .map(|a| (a - mb_mean).powi(2))
                .sum::<f32>()
                / (batch_size as f32 - 1.0).max(1.0);
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

            // Forward pass
            let output = model.forward(obs_tensor);

            // New log probabilities
            let new_log_probs = log_softmax(output.logits.clone(), 1);
            let action_indices: Tensor<B, 2, Int> = Tensor::from_data(
                TensorData::new(actions_data, [batch_size, 1]),
                device,
            );
            let new_action_lp: Tensor<B, 1> = new_log_probs
                .clone()
                .gather(1, action_indices)
                .squeeze_dim::<1>(1);

            // Entropy: H = -sum(p * log_p, dim=1).mean()
            let probs = softmax(output.logits, 1);
            let entropy_per_sample: Tensor<B, 1> = (probs * new_log_probs)
                .sum_dim(1)
                .squeeze_dim::<1>(1)
                .neg();
            let entropy: Tensor<B, 1> = entropy_per_sample.mean().unsqueeze();

            // Policy ratio
            let log_ratio = new_action_lp - old_lp;
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
                ratio.clamp(1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv_tensor.clone();
            // min(surr1, surr2) = surr2 + min(surr1 - surr2, 0)
            //                   = surr2 - relu(surr2 - surr1)
            let min_surr = surr2.clone() - relu(surr2 - surr1);
            let policy_loss: Tensor<B, 1> = if let Some(c) = config.dual_clip_coef {
                // Dual-clip PPO: when advantage < 0, add floor of c * advantage.
                // dual_floor = c * min(adv, 0). When adv >= 0, dual_floor = 0.
                // Since min_surr >= 0 when adv >= 0, max(min_surr, 0) = min_surr.
                // So max(min_surr, dual_floor) is correct for all samples.
                let neg_adv = adv_tensor.clone().neg().clamp_min(0.0).neg(); // min(adv, 0)
                let dual_floor = neg_adv * c;
                // max(a, b) = a + relu(b - a)
                let dual_surr = min_surr.clone() + relu(dual_floor - min_surr);
                dual_surr.mean().neg().unsqueeze()
            } else {
                min_surr.mean().neg().unsqueeze()
            };

            // Value loss with optional clipping (CleanRL default: clipped)
            // Uses max(a,b) = a + relu(b - a) to avoid mask_where gradient issues.
            let new_values = output.values;
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

        // Target KL early stopping (matches CleanRL)
        if let Some(target) = config.target_kl {
            if n_updates > 0 && (total_kl / n_updates as f32) > target {
                break;
            }
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::prelude::*;
    use burn::tensor::activation::relu;
    use burn::tensor::TensorData;

    type B = NdArray;

    fn dev() -> <B as Backend>::Device {
        Default::default()
    }

    /// Helper: compute dual-clip surrogate for given ratio, advantage, eps, c.
    /// Returns the per-element dual-clip objective value.
    fn dual_clip_value(ratio_val: f32, adv_val: f32, eps: f32, c: f32) -> f32 {
        let ratio = Tensor::<B, 1>::from_data(TensorData::new(vec![ratio_val], [1]), &dev());
        let adv = Tensor::<B, 1>::from_data(TensorData::new(vec![adv_val], [1]), &dev());

        let surr1 = ratio.clone() * adv.clone();
        let surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv.clone();
        let min_surr = surr2.clone() - relu(surr2 - surr1);

        let neg_adv = adv.neg().clamp_min(0.0).neg(); // min(adv, 0)
        let dual_floor = neg_adv * c;
        let dual_surr = min_surr.clone() + relu(dual_floor - min_surr);

        dual_surr.into_data().to_vec::<f32>().unwrap()[0]
    }

    /// Helper: compute standard clip surrogate (no dual clip).
    fn standard_clip_value(ratio_val: f32, adv_val: f32, eps: f32) -> f32 {
        let ratio = Tensor::<B, 1>::from_data(TensorData::new(vec![ratio_val], [1]), &dev());
        let adv = Tensor::<B, 1>::from_data(TensorData::new(vec![adv_val], [1]), &dev());

        let surr1 = ratio.clone() * adv.clone();
        let surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv;
        let min_surr = surr2.clone() - relu(surr2 - surr1);

        min_surr.into_data().to_vec::<f32>().unwrap()[0]
    }

    // -- Dual-clip PPO exact worked cases ----------------------------------------

    #[test]
    fn dual_clip_positive_advantage_is_standard_ppo() {
        let result = dual_clip_value(1.5, 2.0, 0.2, 3.0);
        let standard = standard_clip_value(1.5, 2.0, 0.2);

        assert!(
            (result - standard).abs() < 1e-5,
            "Dual clip should not activate for positive advantages"
        );
        assert!((result - 2.4).abs() < 1e-5, "Expected 2.4, got {result}");
    }

    #[test]
    fn dual_clip_activates_for_negative_advantage_extreme_ratio() {
        let result = dual_clip_value(5.0, -0.1, 0.2, 3.0);
        assert!(
            (result - (-0.3)).abs() < 1e-5,
            "Dual clip should raise objective to -0.3, got {result}"
        );
    }

    #[test]
    fn dual_clip_ratio_one_equals_advantage() {
        for adv_val in [-2.0f32, -0.5, 0.0, 0.5, 2.0] {
            let result = dual_clip_value(1.0, adv_val, 0.2, 3.0);
            assert!(
                (result - adv_val).abs() < 1e-5,
                "At ratio=1, output should equal advantage. adv={adv_val}, got={result}"
            );
        }
    }

    #[test]
    fn dual_clip_standard_wins_for_small_ratio_negative_adv() {
        let result = dual_clip_value(0.01, -0.5, 0.2, 3.0);
        assert!(
            (result - (-0.4)).abs() < 1e-5,
            "Standard clip should win, got {result}"
        );
    }

    #[test]
    fn dual_clip_negative_adv_output_bounded() {
        let c = 3.0f32;
        let eps = 0.2f32;

        for ratio_val in [0.01f32, 0.5, 1.0, 1.5, 5.0, 10.0] {
            for adv_val in [-2.0f32, -1.0, -0.5, -0.1] {
                let result = dual_clip_value(ratio_val, adv_val, eps, c);
                let lower = c * adv_val;
                assert!(
                    result >= lower - 1e-5,
                    "Output {result} below lower bound {lower} for ratio={ratio_val}, adv={adv_val}"
                );
                assert!(
                    result <= 1e-5,
                    "Output {result} above 0 for ratio={ratio_val}, adv={adv_val}"
                );
            }
        }
    }
}
