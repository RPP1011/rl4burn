//! Actor-critic update functions.
//!
//! A family of small update functions that share loss computation but differ
//! in how they compute targets/advantages.
//!
//! - [`ac_vtrace_update`] — off-policy, uses V-trace correction
//! - Future: `ac_gae_update` — on-policy, uses GAE
//! - Future: `ac_upgo_update` — uses UPGO

use rl4burn_collect::trajectory::Trajectory;
use rl4burn_collect::vtrace::vtrace_targets;
use rl4burn_nn::clip::clip_grad_norm;
use rl4burn_nn::policy::{DiscreteAcOutput, DiscreteActorCritic};

use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::TensorData;

/// Statistics from an actor-critic update.
#[derive(Debug, Clone)]
pub struct AcStats {
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    /// Mean importance weight rho (>1 means policy diverged from behavior).
    pub mean_rho: f32,
}

/// Perform a V-trace corrected actor-critic update on a batch of trajectories.
///
/// All trajectories must have the same unroll length. A single gradient step
/// is taken per batch, as V-trace already handles off-policy correction.
///
/// Returns the updated model and training statistics.
pub fn ac_vtrace_update<B, M, O>(
    mut model: M,
    optim: &mut O,
    trajectories: &[Trajectory],
    gamma: f32,
    clip_rho: f32,
    clip_c: f32,
    vf_coef: f32,
    ent_coef: f32,
    max_grad_norm: f32,
    lr: f64,
    device: &B::Device,
) -> (M, AcStats)
where
    B: AutodiffBackend,
    M: DiscreteActorCritic<B> + burn::module::AutodiffModule<B>,
    O: Optimizer<M, B>,
{
    let n_traj = trajectories.len();
    assert!(n_traj > 0, "need at least one trajectory");
    let t_len = trajectories[0].actions.len();
    let obs_dim = trajectories[0].observations[0].len();

    debug_assert!(
        trajectories.iter().all(|t| t.actions.len() == t_len),
        "all trajectories must have the same unroll length"
    );

    // --- Forward pass on step observations (gradient tracked) ---
    let step_count = n_traj * t_len;
    let step_obs_flat: Vec<f32> = trajectories
        .iter()
        .flat_map(|traj| {
            traj.observations[..t_len]
                .iter()
                .flat_map(|o| o.iter().copied())
        })
        .collect();
    let step_obs =
        Tensor::<B, 2>::from_data(TensorData::new(step_obs_flat, [step_count, obs_dim]), device);

    let step_output: DiscreteAcOutput<B> = model.forward(step_obs);
    let step_logits = step_output.logits; // [step_count, n_actions]
    let step_values = step_output.values; // [step_count]

    // --- Bootstrap values (no gradient needed) ---
    let bootstrap_obs_flat: Vec<f32> = trajectories
        .iter()
        .flat_map(|traj| traj.observations[t_len].iter().copied())
        .collect();
    let bootstrap_obs =
        Tensor::<B, 2>::from_data(TensorData::new(bootstrap_obs_flat, [n_traj, obs_dim]), device);

    let bootstrap_output = model.forward(bootstrap_obs);
    let bootstrap_vals: Vec<f32> = bootstrap_output.values.into_data().to_vec().unwrap();

    // --- Compute log-probs and detached data for V-trace ---
    let step_lp = log_softmax(step_logits.clone(), 1);
    let step_vals_data: Vec<f32> = step_values.clone().into_data().to_vec().unwrap();
    let step_lp_data: Vec<f32> = step_lp.clone().into_data().to_vec().unwrap();
    let n_act = step_lp_data.len() / step_count;

    // --- Per-trajectory V-trace ---
    let mut all_targets = Vec::with_capacity(step_count);
    let mut all_advantages = Vec::with_capacity(step_count);
    let mut total_rho = 0.0f32;

    for (i, traj) in trajectories.iter().enumerate() {
        let base = i * t_len;

        let traj_values: Vec<f32> = (0..t_len).map(|t| step_vals_data[base + t]).collect();
        let bootstrap = bootstrap_vals[i];

        let log_rhos: Vec<f32> = (0..t_len)
            .map(|t| {
                let idx = base + t;
                let action = traj.actions[t] as usize;
                let new_lp = step_lp_data[idx * n_act + action];
                let old_lp = traj.behavior_log_probs[t];
                new_lp - old_lp
            })
            .collect();

        total_rho += log_rhos
            .iter()
            .map(|lr| lr.clamp(-20.0, 20.0).exp())
            .sum::<f32>();

        let discounts: Vec<f32> = traj
            .dones
            .iter()
            .map(|&d| if d { 0.0 } else { gamma })
            .collect();

        let (vs, adv) = vtrace_targets(
            &log_rhos,
            &discounts,
            &traj.rewards,
            &traj_values,
            bootstrap,
            clip_rho,
            clip_c,
        );

        all_targets.extend_from_slice(&vs);
        all_advantages.extend_from_slice(&adv);
    }

    // --- Gather log-probs for taken actions ---
    let actions_flat: Vec<i32> = trajectories
        .iter()
        .flat_map(|t| t.actions.iter().copied())
        .collect();
    let action_indices =
        Tensor::<B, 2, Int>::from_data(TensorData::new(actions_flat, [step_count, 1]), device);
    let new_action_lp: Tensor<B, 1> = step_lp
        .clone()
        .gather(1, action_indices)
        .squeeze_dim::<1>(1);

    // --- Losses (targets/advantages are constants, no gradient) ---
    let adv_tensor =
        Tensor::<B, 1>::from_data(TensorData::new(all_advantages, [step_count]), device);
    let target_tensor =
        Tensor::<B, 1>::from_data(TensorData::new(all_targets, [step_count]), device);

    // Policy loss: -E[advantage * log pi(a|s)]
    let policy_loss = (new_action_lp * adv_tensor).mean().neg();

    // Value loss: 0.5 * E[(V(s) - target)^2]
    let value_loss = (step_values - target_tensor)
        .powf_scalar(2.0)
        .mean()
        .mul_scalar(0.5);

    // Entropy: -E[sum(p * log p)]
    let step_probs = softmax(step_logits, 1);
    let entropy = (step_probs * step_lp)
        .sum_dim(1)
        .squeeze_dim::<1>(1)
        .neg()
        .mean();

    // Extract stats before backward (safe — clones don't affect the graph)
    let pl = policy_loss.clone().into_data().to_vec::<f32>().unwrap()[0];
    let vl = value_loss.clone().into_data().to_vec::<f32>().unwrap()[0];
    let ent = entropy.clone().into_data().to_vec::<f32>().unwrap()[0];

    // Total loss
    let total_loss = policy_loss
        + value_loss.mul_scalar(vf_coef)
        - entropy.mul_scalar(ent_coef);

    let grads = total_loss.backward();
    let mut grads = GradientsParams::from_grads(grads, &model);
    if max_grad_norm > 0.0 {
        grads = clip_grad_norm(&model, grads, max_grad_norm);
    }
    model = optim.step(lr, model, grads);

    let stats = AcStats {
        policy_loss: pl,
        value_loss: vl,
        entropy: ent,
        mean_rho: total_rho / step_count.max(1) as f32,
    };

    (model, stats)
}
