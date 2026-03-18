//! Deep Q-Network (Mnih et al., 2015).
//!
//! Off-policy value-based RL with:
//! - Experience replay (uses `ReplayBuffer`)
//! - Target network (uses `polyak_update`)
//! - Epsilon-greedy exploration
//!
//! # Architecture
//!
//! DQN maintains two copies of the Q-network: the online network (trained via
//! gradient descent) and the target network (updated slowly via Polyak
//! averaging). The target network provides stable TD targets for the Bellman
//! backup, preventing the moving-target instability of naive Q-learning.

use rl4burn_collect::replay::ReplayBuffer;

use burn::module::AutodiffModule;
use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::TensorData;

use rand::{Rng, RngExt};

// ---------------------------------------------------------------------------
// Q-Network trait
// ---------------------------------------------------------------------------

/// A Q-network maps observations to Q-values for each discrete action.
///
/// # Example
///
/// ```ignore
/// #[derive(Module)]
/// struct MyQNet<B: Backend> {
///     fc1: Linear<B>,
///     fc2: Linear<B>,
///     q_head: Linear<B>,
/// }
///
/// impl<B: Backend> QNetwork<B> for MyQNet<B> {
///     fn q_values(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
///         let h = relu(self.fc1.forward(obs));
///         let h = relu(self.fc2.forward(h));
///         self.q_head.forward(h)
///     }
/// }
/// ```
pub trait QNetwork<B: Backend> {
    /// Compute Q-values for all actions given a batch of observations.
    ///
    /// Input: `[batch, obs_dim]`. Output: `[batch, n_actions]`.
    fn q_values(&self, obs: Tensor<B, 2>) -> Tensor<B, 2>;
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// DQN hyperparameters.
#[derive(Debug, Clone)]
pub struct DqnConfig {
    /// Learning rate.
    pub lr: f64,
    /// Discount factor γ.
    pub gamma: f32,
    /// Replay buffer capacity.
    pub buffer_capacity: usize,
    /// Minibatch size for each gradient step.
    pub batch_size: usize,
    /// Polyak averaging coefficient τ for target network updates.
    pub tau: f32,
    /// Initial exploration rate.
    pub eps_start: f32,
    /// Final exploration rate.
    pub eps_end: f32,
    /// Number of steps to linearly anneal ε from start to end.
    pub eps_decay_steps: usize,
    /// Number of random steps before training starts.
    pub learning_starts: usize,
}

impl Default for DqnConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            gamma: 0.99,
            buffer_capacity: 10_000,
            batch_size: 32,
            tau: 0.005,
            eps_start: 1.0,
            eps_end: 0.05,
            eps_decay_steps: 10_000,
            learning_starts: 1_000,
        }
    }
}

// ---------------------------------------------------------------------------
// Transition
// ---------------------------------------------------------------------------

/// A single transition stored in the replay buffer.
#[derive(Clone)]
pub struct Transition {
    pub obs: Vec<f32>,
    pub action: i32,
    pub reward: f32,
    pub next_obs: Vec<f32>,
    pub done: bool,
}

// ---------------------------------------------------------------------------
// Training statistics
// ---------------------------------------------------------------------------

/// Statistics from a DQN training step.
#[derive(Debug, Clone)]
pub struct DqnStats {
    pub loss: f32,
    pub mean_q: f32,
    pub epsilon: f32,
}

// ---------------------------------------------------------------------------
// Epsilon schedule
// ---------------------------------------------------------------------------

/// Linear annealing of ε from `eps_start` to `eps_end` over `eps_decay_steps`.
pub fn epsilon_schedule(config: &DqnConfig, step: usize) -> f32 {
    if step >= config.eps_decay_steps {
        config.eps_end
    } else {
        let frac = step as f32 / config.eps_decay_steps as f32;
        config.eps_start + (config.eps_end - config.eps_start) * frac
    }
}

/// Select an action using ε-greedy: random with probability ε, greedy otherwise.
pub fn epsilon_greedy<B: Backend, M: QNetwork<B>>(
    model: &M,
    obs: &[f32],
    n_actions: usize,
    epsilon: f32,
    device: &B::Device,
    rng: &mut impl Rng,
) -> usize {
    if rng.random::<f32>() < epsilon {
        rng.random_range(0..n_actions)
    } else {
        let obs_tensor: Tensor<B, 2> =
            Tensor::from_data(TensorData::new(obs.to_vec(), [1, obs.len()]), device);
        let q = model.q_values(obs_tensor);
        let q_data: Vec<f32> = q.into_data().to_vec().unwrap();
        q_data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }
}

// ---------------------------------------------------------------------------
// DQN update
// ---------------------------------------------------------------------------

/// Perform a single DQN gradient step on a minibatch from the replay buffer.
///
/// Uses the target network for stable Bellman targets:
///   `y = r + γ * (1 - done) * max_a' Q_target(s', a')`
///   `loss = mean((Q(s, a) - y)²)`
///
/// Returns updated online model and training stats. The caller is responsible
/// for calling `polyak_update` on the target network afterward.
pub fn dqn_update<B, M, O, R>(
    online: M,
    target: &M,
    optim: &mut O,
    buffer: &mut ReplayBuffer<Transition, R>,
    config: &DqnConfig,
    device: &B::Device,
) -> (M, DqnStats)
where
    B: AutodiffBackend,
    M: QNetwork<B> + AutodiffModule<B>,
    O: Optimizer<M, B>,
    R: rand::Rng,
{
    let batch = buffer.sample_cloned(config.batch_size);
    let batch_size = batch.len();
    let obs_dim = batch[0].obs.len();

    // Build batch tensors
    let obs_flat: Vec<f32> = batch.iter().flat_map(|t| t.obs.iter().copied()).collect();
    let obs_tensor: Tensor<B, 2> =
        Tensor::from_data(TensorData::new(obs_flat, [batch_size, obs_dim]), device);

    let next_obs_flat: Vec<f32> = batch
        .iter()
        .flat_map(|t| t.next_obs.iter().copied())
        .collect();
    let next_obs_tensor: Tensor<B, 2> =
        Tensor::from_data(TensorData::new(next_obs_flat, [batch_size, obs_dim]), device);

    let actions_data: Vec<i32> = batch.iter().map(|t| t.action).collect();
    let action_indices: Tensor<B, 2, Int> =
        Tensor::from_data(TensorData::new(actions_data, [batch_size, 1]), device);

    let rewards_data: Vec<f32> = batch.iter().map(|t| t.reward).collect();
    let rewards: Tensor<B, 1> =
        Tensor::from_data(TensorData::new(rewards_data, [batch_size]), device);

    let not_done_data: Vec<f32> = batch
        .iter()
        .map(|t| if t.done { 0.0 } else { 1.0 })
        .collect();
    let not_done: Tensor<B, 1> =
        Tensor::from_data(TensorData::new(not_done_data, [batch_size]), device);

    // Current Q-values for taken actions
    let q_all = online.q_values(obs_tensor);
    let q_taken: Tensor<B, 1> = q_all.gather(1, action_indices).squeeze_dim::<1>(1);

    // Target Q-values: forward through target network, then extract data
    // to break the computation graph (equivalent to .detach() in PyTorch).
    let next_q_all = target.q_values(next_obs_tensor);
    let next_q_max_data: Vec<f32> = next_q_all
        .max_dim(1)
        .squeeze_dim::<1>(1)
        .into_data()
        .to_vec()
        .unwrap();
    let next_q_max: Tensor<B, 1> =
        Tensor::from_data(TensorData::new(next_q_max_data, [batch_size]), device);

    let targets = rewards + not_done * next_q_max * config.gamma;

    // MSE loss
    let td_error = q_taken.clone() - targets;
    let loss: Tensor<B, 1> = td_error.powf_scalar(2.0).mean().unsqueeze();

    // Stats (before backward)
    let loss_val: f32 = loss.clone().into_data().to_vec::<f32>().unwrap()[0];
    let mean_q: f32 = q_taken.into_data().to_vec::<f32>().unwrap().iter().sum::<f32>()
        / batch_size as f32;

    // Backward + optimize
    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &online);
    let online = optim.step(config.lr, online, grads);

    let stats = DqnStats {
        loss: loss_val,
        mean_q,
        epsilon: 0.0, // caller sets this
    };

    (online, stats)
}
