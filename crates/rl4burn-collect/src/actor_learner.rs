//! IMPALA-style local collection: each actor runs its own model copy.
//!
//! Use [`batched_inference`] + [`actor_learner_collect`] for the IMPALA
//! collection pattern where each actor has a local model for inference.

use crate::trajectory::Trajectory;
use rl4burn_core::env::vec_env::SyncVecEnv;
use rl4burn_core::env::Env;
use rl4burn_nn::policy::{DiscreteAcOutput, DiscreteActorCritic};

use burn::prelude::*;
use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::TensorData;

use rand::{Rng, RngExt};

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
// Batched inference
// ---------------------------------------------------------------------------

/// Run batched policy inference on a set of observations.
///
/// Returns `(action, log_prob)` pairs, one per observation.
/// Works on any backend — use the inference backend, not necessarily autodiff.
pub fn batched_inference<B, M>(
    model: &M,
    observations: &[Vec<f32>],
    device: &B::Device,
    rng: &mut impl Rng,
) -> Vec<(i32, f32)>
where
    B: Backend,
    M: DiscreteActorCritic<B>,
{
    let batch = observations.len();
    if batch == 0 {
        return Vec::new();
    }
    let obs_dim = observations[0].len();

    let obs_flat: Vec<f32> = observations.iter().flat_map(|o| o.iter().copied()).collect();
    let obs_tensor =
        Tensor::<B, 2>::from_data(TensorData::new(obs_flat, [batch, obs_dim]), device);

    let output: DiscreteAcOutput<B> = model.forward(obs_tensor);
    let probs = softmax(output.logits.clone(), 1);
    let lp = log_softmax(output.logits, 1);

    let probs_data: Vec<f32> = probs.into_data().to_vec().unwrap();
    let lp_data: Vec<f32> = lp.into_data().to_vec().unwrap();
    let n_actions = probs_data.len() / batch;

    let mut results = Vec::with_capacity(batch);
    for i in 0..batch {
        let row = &probs_data[i * n_actions..(i + 1) * n_actions];
        let action = sample_categorical(row, rng);
        let log_prob = lp_data[i * n_actions + action];
        results.push((action as i32, log_prob));
    }
    results
}

// ---------------------------------------------------------------------------
// Local collection (IMPALA-style)
// ---------------------------------------------------------------------------

/// Collect trajectories by running environments with local model inference.
///
/// Produces one [`Trajectory`] per environment. The model is evaluated on
/// the given backend for action sampling; trajectories store behavior
/// log-probabilities for later V-trace correction by the learner.
///
/// `current_obs` and `episode_returns_acc` are carried across calls (same
/// semantics as [`ppo_collect`](crate::algo::base::ppo::ppo_collect)).
///
/// Returns `(trajectories, completed_episode_returns)`.
pub fn actor_learner_collect<B, M, E>(
    model: &M,
    envs: &mut SyncVecEnv<E>,
    unroll_length: usize,
    device: &B::Device,
    rng: &mut impl Rng,
    current_obs: &mut Vec<Vec<f32>>,
    episode_returns_acc: &mut Vec<f32>,
) -> (Vec<Trajectory>, Vec<f32>)
where
    B: Backend,
    M: DiscreteActorCritic<B>,
    E: Env<Observation = Vec<f32>, Action = usize>,
{
    let n_envs = envs.num_envs();
    let mut episode_returns = Vec::new();

    if episode_returns_acc.is_empty() {
        episode_returns_acc.resize(n_envs, 0.0);
    }

    // Per-env accumulators
    let mut obs_acc: Vec<Vec<Vec<f32>>> =
        (0..n_envs).map(|_| Vec::with_capacity(unroll_length + 1)).collect();
    let mut act_acc: Vec<Vec<i32>> = (0..n_envs).map(|_| Vec::with_capacity(unroll_length)).collect();
    let mut rew_acc: Vec<Vec<f32>> = (0..n_envs).map(|_| Vec::with_capacity(unroll_length)).collect();
    let mut done_acc: Vec<Vec<bool>> = (0..n_envs).map(|_| Vec::with_capacity(unroll_length)).collect();
    let mut lp_acc: Vec<Vec<f32>> = (0..n_envs).map(|_| Vec::with_capacity(unroll_length)).collect();

    for _step in 0..unroll_length {
        // Store current observations
        for (i, obs) in current_obs.iter().enumerate() {
            obs_acc[i].push(obs.clone());
        }

        // Batched inference
        let results = batched_inference::<B, M>(model, current_obs, device, rng);

        // Step environments
        let actions: Vec<usize> = results.iter().map(|&(a, _)| a as usize).collect();
        let steps = envs.step(actions);

        for (i, step) in steps.iter().enumerate() {
            let (action, log_prob) = results[i];
            act_acc[i].push(action);
            lp_acc[i].push(log_prob);
            rew_acc[i].push(step.reward);
            done_acc[i].push(step.done());

            episode_returns_acc[i] += step.reward;
            if step.done() {
                episode_returns.push(episode_returns_acc[i]);
                episode_returns_acc[i] = 0.0;
            }
        }

        *current_obs = steps.into_iter().map(|s| s.observation).collect();
    }

    // Bootstrap observations (T+1)
    for (i, obs) in current_obs.iter().enumerate() {
        obs_acc[i].push(obs.clone());
    }

    let trajectories = (0..n_envs)
        .map(|i| Trajectory {
            observations: std::mem::take(&mut obs_acc[i]),
            actions: std::mem::take(&mut act_acc[i]),
            rewards: std::mem::take(&mut rew_acc[i]),
            dones: std::mem::take(&mut done_acc[i]),
            behavior_log_probs: std::mem::take(&mut lp_acc[i]),
        })
        .collect();

    (trajectories, episode_returns)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_categorical_in_range() {
        let probs = [0.1, 0.2, 0.3, 0.4];
        let mut rng = rand::rng();
        for _ in 0..100 {
            let idx = sample_categorical(&probs, &mut rng);
            assert!(idx < probs.len());
        }
    }

    #[test]
    fn sample_categorical_degenerate() {
        let probs = [0.0, 0.0, 1.0, 0.0];
        let mut rng = rand::rng();
        for _ in 0..20 {
            assert_eq!(sample_categorical(&probs, &mut rng), 2);
        }
    }
}
