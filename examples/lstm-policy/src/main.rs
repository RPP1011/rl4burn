//! # Example 12 — LSTM Policies for Partial Observability
//!
//! Demonstrates recurrent (LSTM) policies for games with hidden information.
//! This is the most complex example because the standard `ppo_collect` and
//! `masked_ppo_collect` functions do not natively handle recurrent state
//! passing. We implement a custom training loop that:
//!
//! 1. Manually manages LSTM hidden state across steps during collection.
//! 2. Resets hidden state on episode boundaries (when `done = true`).
//! 3. During PPO update, processes sequences **in temporal order** (not shuffled)
//!    to maintain hidden state consistency.
//!
//! ## Why recurrence?
//!
//! In partially observable environments, the current observation does not
//! contain enough information to act optimally. The agent needs *memory*
//! of past observations. An LSTM hidden state serves as this memory,
//! compressing the observation history into a fixed-size vector.
//!
//! ## The Memory Game
//!
//! A sequence of symbols is shown one at a time. At a "recall" step,
//! the agent must output the symbol it saw `K` steps ago. Without memory,
//! performance is random (1/NUM_SYMBOLS). With an LSTM, the agent can
//! learn to store and retrieve past symbols.
//!
//! - Observation: one-hot of current symbol + a "recall flag" + step counter
//! - Action: Discrete(NUM_SYMBOLS) — guess which symbol was shown K steps ago
//! - Reward: +1 for correct recall, 0 otherwise
//!
//! ## Key implementation detail: sequence-based PPO update
//!
//! Standard PPO shuffles all timesteps into random minibatches. This breaks
//! LSTM hidden state continuity. Instead, we:
//! - Collect full episodes as contiguous sequences.
//! - During the update, iterate over sequences in order, running the LSTM
//!   forward to reconstruct hidden states before computing log-probs.
//! - This is less sample-efficient per gradient step but necessary for
//!   correct recurrent policy training.
//!
//! Run with: `cargo run -p lstm-policy --release`

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::activation::{log_softmax, relu, softmax};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::TensorData;

use rand::{Rng, RngExt, SeedableRng};

use rl4burn::env::space::Space;
use rl4burn::{clip_grad_norm, gae, Env, LstmCell, LstmCellConfig, LstmState, Step};

type TrainB = Autodiff<NdArray>;
/// The inference backend (inner backend of Autodiff). Used when calling
/// `model.valid()` to strip autodiff for data collection.
#[allow(dead_code)]
type InferB = NdArray;

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// Number of distinct symbols in the memory game.
const NUM_SYMBOLS: usize = 4;
/// How many steps back the agent must recall.
const RECALL_DELAY: usize = 3;
/// Episode length (number of steps).
const EPISODE_LEN: usize = 20;

/// Observation dimension:
/// - NUM_SYMBOLS for current symbol one-hot
/// - 1 for recall flag (is this a recall step?)
/// - 1 for normalized step counter
const OBS_DIM: usize = NUM_SYMBOLS + 2;

/// A memory game that requires recurrence to solve.
///
/// The environment shows a sequence of random symbols. On designated "recall"
/// steps (every RECALL_DELAY+1 steps after the initial delay), the agent
/// must output the symbol shown RECALL_DELAY steps ago.
///
/// On non-recall steps, the agent's action is ignored (reward = 0).
/// On recall steps, reward = +1 if correct, 0 if wrong.
struct MemoryGameEnv<R> {
    rng: R,
    step_idx: usize,
    /// History of symbols shown so far this episode.
    history: Vec<usize>,
    /// Current symbol being shown.
    current_symbol: usize,
    /// Whether this step is a recall step.
    is_recall: bool,
}

impl<R: Rng> MemoryGameEnv<R> {
    fn new(mut rng: R) -> Self {
        let sym = rng.random_range(0..NUM_SYMBOLS);
        Self {
            rng,
            step_idx: 0,
            history: vec![sym],
            current_symbol: sym,
            is_recall: false,
        }
    }

    fn obs(&self) -> Vec<f32> {
        let mut o = vec![0.0f32; OBS_DIM];
        // Current symbol one-hot
        o[self.current_symbol] = 1.0;
        // Recall flag
        o[NUM_SYMBOLS] = if self.is_recall { 1.0 } else { 0.0 };
        // Normalized step counter
        o[NUM_SYMBOLS + 1] = self.step_idx as f32 / EPISODE_LEN as f32;
        o
    }
}

impl<R: Rng + Clone> Env for MemoryGameEnv<R> {
    type Observation = Vec<f32>;
    type Action = Vec<f32>;

    fn reset(&mut self) -> Vec<f32> {
        self.step_idx = 0;
        let sym = self.rng.random_range(0..NUM_SYMBOLS);
        self.history = vec![sym];
        self.current_symbol = sym;
        self.is_recall = false;
        self.obs()
    }

    fn step(&mut self, action: Vec<f32>) -> Step<Vec<f32>> {
        let guessed = action[0] as usize;
        self.step_idx += 1;

        // Determine reward: only on recall steps
        let reward = if self.is_recall && self.history.len() > RECALL_DELAY {
            let target_idx = self.history.len() - 1 - RECALL_DELAY;
            if guessed == self.history[target_idx] {
                1.0
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Generate next symbol
        let new_sym = self.rng.random_range(0..NUM_SYMBOLS);
        self.history.push(new_sym);
        self.current_symbol = new_sym;

        // Determine if next observation is a recall step.
        // Recall every (RECALL_DELAY+1) steps, starting after initial delay.
        self.is_recall = self.step_idx >= RECALL_DELAY
            && (self.step_idx - RECALL_DELAY) % (RECALL_DELAY + 1) == 0;

        let truncated = self.step_idx >= EPISODE_LEN;

        Step {
            observation: self.obs(),
            reward,
            terminated: false,
            truncated,
        }
    }

    fn observation_space(&self) -> Space {
        Space::Box {
            low: vec![0.0; OBS_DIM],
            high: vec![1.0; OBS_DIM],
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(NUM_SYMBOLS)
    }
}

// ---------------------------------------------------------------------------
// LSTM Actor-Critic Model
// ---------------------------------------------------------------------------

/// Hidden size for the LSTM layer.
const HIDDEN_SIZE: usize = 64;

/// LSTM-based actor-critic for the memory game.
///
/// Architecture:
/// - Linear encoder: obs_dim -> hidden_size
/// - LstmCell: hidden_size -> hidden_size (the recurrent layer)
/// - Policy head: hidden_size -> NUM_SYMBOLS logits
/// - Value head: hidden_size -> 1 scalar
///
/// The LSTM cell maintains hidden state (h, c) across timesteps within
/// an episode. This hidden state acts as the agent's memory, allowing
/// it to recall information from earlier observations.
#[derive(Module, Debug)]
struct LstmActorCritic<B: Backend> {
    encoder: Linear<B>,
    lstm: LstmCell<B>,
    policy_head: Linear<B>,
    value_head: Linear<B>,
}

impl<B: Backend> LstmActorCritic<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            encoder: LinearConfig::new(OBS_DIM, HIDDEN_SIZE).init(device),
            lstm: LstmCellConfig::new(HIDDEN_SIZE, HIDDEN_SIZE).init(device),
            policy_head: LinearConfig::new(HIDDEN_SIZE, NUM_SYMBOLS).init(device),
            value_head: LinearConfig::new(HIDDEN_SIZE, 1).init(device),
        }
    }

    /// Forward pass for a single timestep. Takes the current observation
    /// and the previous LSTM state, returns logits, value, and new state.
    fn forward_step(
        &self,
        obs: Tensor<B, 2>,
        state: &LstmState<B>,
    ) -> (Tensor<B, 2>, Tensor<B, 1>, LstmState<B>) {
        let encoded = self.encoder.forward(obs).tanh();
        let new_state = self.lstm.forward(encoded, state);
        let logits = self.policy_head.forward(new_state.h.clone());
        let values = self.value_head.forward(new_state.h.clone()).squeeze_dim::<1>(1);
        (logits, values, new_state)
    }
}

// ---------------------------------------------------------------------------
// Custom rollout collection with LSTM state management
// ---------------------------------------------------------------------------

/// A single step of collected experience (includes LSTM state info).
struct RolloutStep {
    obs: Vec<f32>,
    action: usize,
    log_prob: f32,
    value: f32,
    reward: f32,
    done: bool,
}

/// Collect a rollout from a single environment with LSTM state tracking.
///
/// Unlike the standard `ppo_collect` which batches across vectorized envs,
/// this collects from one env at a time with manual hidden state management.
/// The hidden state is:
/// - Carried forward between steps within an episode.
/// - Reset to zeros when an episode ends (done = true).
///
/// This is the fundamental difference from feedforward PPO: the hidden state
/// creates temporal dependencies between steps, so we cannot treat them
/// as independent samples.
fn collect_episode<B: Backend>(
    model: &LstmActorCritic<B>,
    env: &mut impl Env<Observation = Vec<f32>, Action = Vec<f32>>,
    device: &B::Device,
    rng: &mut impl Rng,
) -> (Vec<RolloutStep>, f32) {
    let mut steps = Vec::with_capacity(EPISODE_LEN);
    let mut obs = env.reset();
    let mut state = LstmState::zeros(1, HIDDEN_SIZE, device);

    for _ in 0..EPISODE_LEN {
        let obs_tensor: Tensor<B, 2> = Tensor::from_data(
            TensorData::new(obs.clone(), [1, OBS_DIM]),
            device,
        );

        let (logits, values, new_state) = model.forward_step(obs_tensor, &state);

        // Sample action from softmax
        let probs_data: Vec<f32> = softmax(logits.clone(), 1).into_data().to_vec().unwrap();
        let log_probs_data: Vec<f32> = log_softmax(logits, 1).into_data().to_vec().unwrap();
        let value: f32 = values.into_data().to_vec::<f32>().unwrap()[0];

        // Sample from categorical
        let u: f32 = rng.random();
        let mut cum = 0.0;
        let mut action = NUM_SYMBOLS - 1;
        for (i, &p) in probs_data.iter().enumerate() {
            cum += p;
            if u < cum {
                action = i;
                break;
            }
        }

        let log_prob = log_probs_data[action];

        // Step the environment
        let step = env.step(vec![action as f32]);
        let done = step.done();

        steps.push(RolloutStep {
            obs,
            action,
            log_prob,
            value,
            reward: step.reward,
            done,
        });

        obs = step.observation;

        // Reset LSTM state on episode boundary
        if done {
            state = LstmState::zeros(1, HIDDEN_SIZE, device);
        } else {
            state = new_state;
        }
    }

    // Bootstrap value for the last observation
    let obs_tensor: Tensor<B, 2> = Tensor::from_data(
        TensorData::new(obs, [1, OBS_DIM]),
        device,
    );
    let (_, last_val, _) = model.forward_step(obs_tensor, &state);
    let last_value: f32 = last_val.into_data().to_vec::<f32>().unwrap()[0];

    (steps, last_value)
}

// ---------------------------------------------------------------------------
// Custom PPO update for LSTM policies
// ---------------------------------------------------------------------------

/// Perform a PPO update on collected episodes, processing sequences in
/// temporal order to maintain LSTM hidden state consistency.
///
/// Unlike standard PPO which shuffles timesteps, we must process steps
/// sequentially within each episode so the LSTM hidden states are correct.
/// We still run multiple epochs over the data for sample efficiency.
///
/// Trade-off: we lose the variance reduction benefits of random shuffling,
/// but gain correct hidden state propagation. In practice, collecting
/// from multiple environments and interleaving their episodes provides
/// enough diversity.
fn lstm_ppo_update<B: AutodiffBackend, O: Optimizer<LstmActorCritic<B>, B>>(
    mut model: LstmActorCritic<B>,
    optim: &mut O,
    episodes: &[(Vec<RolloutStep>, f32)], // (steps, last_value) per episode
    config: &LstmPpoConfig,
    current_lr: f64,
    device: &B::Device,
) -> (LstmActorCritic<B>, f32, f32, f32) {
    let mut total_policy_loss = 0.0f32;
    let mut total_value_loss = 0.0f32;
    let mut total_entropy = 0.0f32;
    let mut n_updates = 0u32;

    for _epoch in 0..config.update_epochs {
        for (steps, last_value) in episodes {
            if steps.is_empty() {
                continue;
            }

            // Compute GAE advantages for this episode
            let rewards: Vec<f32> = steps.iter().map(|s| s.reward).collect();
            let values: Vec<f32> = steps.iter().map(|s| s.value).collect();
            let dones: Vec<bool> = steps.iter().map(|s| s.done).collect();
            let (advantages, returns) = gae(&rewards, &values, &dones, *last_value, config.gamma, config.gae_lambda);

            // Normalize advantages
            let adv_mean: f32 = advantages.iter().sum::<f32>() / advantages.len() as f32;
            let adv_var: f32 = advantages.iter().map(|a| (a - adv_mean).powi(2)).sum::<f32>()
                / (advantages.len() as f32 - 1.0).max(1.0);
            let adv_std = adv_var.sqrt().max(1e-8);
            let norm_advantages: Vec<f32> = advantages.iter().map(|a| (a - adv_mean) / adv_std).collect();

            // Process the episode sequentially to maintain LSTM state
            let mut lstm_state = LstmState::zeros(1, HIDDEN_SIZE, device);
            let mut step_losses: Vec<Tensor<B, 1>> = Vec::new();

            for (t, step) in steps.iter().enumerate() {
                let obs_tensor: Tensor<B, 2> = Tensor::from_data(
                    TensorData::new(step.obs.clone(), [1, OBS_DIM]),
                    device,
                );

                let (logits, new_values, new_state) = model.forward_step(obs_tensor, &lstm_state);

                // Compute new log-prob under current policy
                let new_log_probs = log_softmax(logits.clone(), 1);
                let action_idx: Tensor<B, 2, Int> = Tensor::from_data(
                    TensorData::new(vec![step.action as i32], [1, 1]),
                    device,
                );
                let new_lp: Tensor<B, 1> = new_log_probs.clone().gather(1, action_idx).squeeze_dim::<1>(1);

                // Entropy: H = -sum(p * log_p)
                let probs = softmax(logits, 1);
                let log_p = new_log_probs;
                let entropy: Tensor<B, 1> = (probs * log_p).sum_dim(1).squeeze_dim::<1>(1).neg();

                // Policy ratio
                let old_lp: Tensor<B, 1> = Tensor::from_data(
                    TensorData::new(vec![step.log_prob], [1]),
                    device,
                );
                let log_ratio = new_lp - old_lp;
                let ratio = log_ratio.clone().exp();

                // Clipped surrogate objective
                let adv_tensor: Tensor<B, 1> = Tensor::from_data(
                    TensorData::new(vec![norm_advantages[t]], [1]),
                    device,
                );
                let surr1 = ratio.clone() * adv_tensor.clone();
                let surr2 = ratio.clamp(1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv_tensor;
                let policy_loss = surr2.clone() - relu(surr2 - surr1);
                let policy_loss = policy_loss.neg();

                // Value loss
                let ret_tensor: Tensor<B, 1> = Tensor::from_data(
                    TensorData::new(vec![returns[t]], [1]),
                    device,
                );
                let value_loss = (new_values - ret_tensor).powf_scalar(2.0) * 0.5;

                // Combined loss for this step
                let step_loss = policy_loss.clone()
                    + value_loss.clone() * config.vf_coef
                    - entropy.clone() * config.ent_coef;

                step_losses.push(step_loss);

                // Track stats (detached)
                total_policy_loss += policy_loss.clone().into_data().to_vec::<f32>().unwrap()[0];
                total_value_loss += value_loss.clone().into_data().to_vec::<f32>().unwrap()[0];
                total_entropy += entropy.clone().into_data().to_vec::<f32>().unwrap()[0];
                n_updates += 1;

                // Update LSTM state, resetting on episode boundaries
                if step.done {
                    lstm_state = LstmState::zeros(1, HIDDEN_SIZE, device);
                } else {
                    lstm_state = new_state;
                }
            }

            // Accumulate loss over the episode and backprop once
            if !step_losses.is_empty() {
                let total_loss = step_losses
                    .into_iter()
                    .reduce(|a, b| a + b)
                    .unwrap()
                    / steps.len() as f32;

                let grads = total_loss.backward();
                let mut grads = GradientsParams::from_grads(grads, &model);
                grads = clip_grad_norm(&model, grads, config.max_grad_norm);
                model = optim.step(current_lr, model, grads);
            }
        }
    }

    let n = n_updates.max(1) as f32;
    (model, total_policy_loss / n, total_value_loss / n, total_entropy / n)
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Simplified PPO config for the LSTM training loop.
struct LstmPpoConfig {
    lr: f64,
    gamma: f32,
    gae_lambda: f32,
    clip_eps: f32,
    vf_coef: f32,
    ent_coef: f32,
    update_epochs: usize,
    max_grad_norm: f32,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let device = NdArrayDevice::Cpu;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    // --- Environments (one per "worker") ---
    let n_envs = 4;
    let mut envs: Vec<MemoryGameEnv<rand::rngs::SmallRng>> = (0..n_envs)
        .map(|i| MemoryGameEnv::new(rand::rngs::SmallRng::seed_from_u64(42 + i as u64)))
        .collect();

    // --- Model ---
    let mut model: LstmActorCritic<TrainB> = LstmActorCritic::new(&device);

    // --- Optimiser ---
    let mut optim = AdamConfig::new().with_epsilon(1e-5).init();

    // --- Hyperparameters ---
    let config = LstmPpoConfig {
        lr: 3e-4,
        gamma: 0.99,
        gae_lambda: 0.95,
        clip_eps: 0.2,
        vf_coef: 0.5,
        ent_coef: 0.01,
        update_epochs: 3,
        max_grad_norm: 0.5,
    };

    // --- Training loop ---
    let n_iterations = 300;
    let _episodes_per_iter = n_envs; // Collect one episode per env per iteration

    let mut recent_returns: Vec<f32> = Vec::new();
    let mut best_avg = f32::NEG_INFINITY;

    println!("=== LSTM Policy: Memory Game ===");
    println!();
    println!("  Symbols:      {NUM_SYMBOLS}");
    println!("  Recall delay: {RECALL_DELAY} steps");
    println!("  Episode len:  {EPISODE_LEN}");
    println!("  Envs:         {n_envs}");
    println!("  Iterations:   {n_iterations}");
    println!();
    println!("Without memory, accuracy is random: {:.0}%", 100.0 / NUM_SYMBOLS as f32);
    println!("A perfect LSTM policy should approach ~100% recall accuracy.");
    println!();

    // Count how many recall steps per episode (for reward normalization in reporting)
    let n_recall_steps = (0..EPISODE_LEN)
        .filter(|&s| s >= RECALL_DELAY && (s - RECALL_DELAY) % (RECALL_DELAY + 1) == 0)
        .count();
    let max_episode_return = n_recall_steps as f32;
    println!("Recall steps per episode: {n_recall_steps}  (max return: {max_episode_return:.0})");
    println!();

    for iter in 0..n_iterations {
        let frac = 1.0 - iter as f64 / n_iterations as f64;
        let current_lr = config.lr * frac;

        // Collect episodes from all environments using the inference model.
        // Each episode runs the LSTM forward step by step, carrying hidden
        // state across timesteps and resetting on episode boundaries.
        let inference_model = model.valid();
        let mut episodes: Vec<(Vec<RolloutStep>, f32)> = Vec::with_capacity(n_envs);

        for env in envs.iter_mut() {
            let (steps, last_value) = collect_episode(
                &inference_model,
                env,
                &device,
                &mut rng,
            );
            let ep_return: f32 = steps.iter().map(|s| s.reward).sum();
            recent_returns.push(ep_return);
            episodes.push((steps, last_value));
        }

        // PPO update with sequential processing for LSTM state consistency
        let (updated_model, policy_loss, value_loss, entropy) = lstm_ppo_update(
            model,
            &mut optim,
            &episodes,
            &config,
            current_lr,
            &device,
        );
        model = updated_model;

        // Logging
        if recent_returns.len() > 50 {
            let start = recent_returns.len() - 50;
            recent_returns = recent_returns[start..].to_vec();
        }

        if !recent_returns.is_empty() {
            let avg: f32 = recent_returns.iter().sum::<f32>() / recent_returns.len() as f32;
            best_avg = best_avg.max(avg);
            let accuracy = avg / max_episode_return * 100.0;

            if (iter + 1) % 20 == 0 || iter == 0 {
                println!(
                    "iter {:>4}/{}: avg_return={:>5.1}  accuracy={:>5.1}%  best={:>5.1}  \
                     ploss={:>8.4}  vloss={:>7.4}  entropy={:.3}",
                    iter + 1,
                    n_iterations,
                    avg,
                    accuracy,
                    best_avg,
                    policy_loss,
                    value_loss,
                    entropy,
                );
            }
        }
    }

    println!();
    let final_accuracy = best_avg / max_episode_return * 100.0;
    println!("Training complete. Best avg return: {best_avg:.1} ({final_accuracy:.0}% accuracy)");
    if final_accuracy > 50.0 {
        println!("LSTM learned to use memory for recall!");
    } else {
        println!(
            "Agent still learning. LSTM recurrence needs more iterations \
             or hyperparameter tuning for this task."
        );
    }
}
