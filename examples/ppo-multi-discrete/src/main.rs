//! # Example 6 — PPO with Multi-Discrete Actions
//!
//! Demonstrates how to use PPO with a **multi-discrete action space** and
//! **action masking** — the two features most critical for tactical game AI
//! where agents must make multiple simultaneous discrete choices per step.
//!
//! ## What is a multi-discrete action space?
//!
//! In many games an agent must choose several things at once:
//!
//! | Sub-action    | Choices                              | Size |
//! |---------------|--------------------------------------|------|
//! | Select unit   | {warrior, archer, mage}              |  3   |
//! | Select action | {move, attack, defend}               |  3   |
//! | Select target | {north, south, east, west}           |  4   |
//!
//! A single `Discrete(36)` space (3*3*4) would ignore the factorial structure
//! and make credit assignment harder.  `MultiDiscrete([3, 3, 4])` instead
//! treats each sub-action as an **independent categorical** — the policy
//! network outputs one set of logits per sub-action, and probabilities
//! factorise:
//!
//! ```text
//! P(a) = P(unit) * P(action) * P(target)
//! ```
//!
//! ## How `ActionDist::MultiDiscrete` works
//!
//! `ActionDist::MultiDiscrete(vec![n1, n2, …])` tells rl4burn that the
//! policy head outputs `n1 + n2 + …` logits.  During sampling/log-prob
//! computation the logit tensor is **split** into chunks of sizes
//! `[n1, n2, …]`, each chunk is turned into an independent Categorical
//! distribution, and:
//!
//! - **Sampling** draws one index from each sub-distribution.
//! - **Log-prob** is the *sum* of the per-sub-action log-probs.
//! - **Entropy** is the *sum* of per-sub-action entropies.
//!
//! The action is returned as `Vec<f32>` with one element per sub-action,
//! where each element is the chosen index cast to `f32`.
//!
//! ## Action masking
//!
//! In game AI many actions are *invalid* at any given moment — a unit at the
//! map edge cannot move off-screen, a cooldown prevents attacking, etc.
//! Letting the agent pick invalid actions and then giving a penalty is
//! wasteful and slows learning.
//!
//! **Action masking** solves this: the environment returns a binary mask
//! (1 = valid, 0 = invalid) concatenated across all sub-actions.  Before
//! the softmax, invalid logits are set to `-inf` so they receive zero
//! probability.  This is both more sample-efficient and mathematically
//! cleaner than penalty-based approaches.
//!
//! ## `MaskedActorCritic` vs `DiscreteActorCritic`
//!
//! | Trait                  | Mask support | Collect fn            | Update fn            |
//! |------------------------|--------------|-----------------------|----------------------|
//! | `DiscreteActorCritic`  | No           | `ppo_collect`         | `ppo_update`         |
//! | `MaskedActorCritic`    | Yes          | `masked_ppo_collect`  | `masked_ppo_update`  |
//!
//! `MaskedActorCritic` has the same `forward` signature — `(obs) -> (logits, values)` —
//! but the collect/update functions additionally query `env.action_mask()` and
//! store masks in the rollout buffer so that log-probs are recomputed
//! correctly during the PPO update epochs.
//!
//! ## This example
//!
//! We train a small agent on a 2D grid navigation task:
//!
//! - **Observation**: `[x, y, target_x, target_y]` normalised to [0, 1].
//! - **Action space**: `MultiDiscrete([3, 3])` — dx in {-1, 0, +1} and
//!   dy in {-1, 0, +1} chosen independently.
//! - **Masking**: at grid boundaries, movement toward walls is masked out.
//! - **Reward**: +10 for reaching the target, small step penalty otherwise.
//!
//! Run: `cargo run -p ppo-multi-discrete --release`

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::optim::AdamConfig;
use burn::prelude::*;

use rand::{RngExt, SeedableRng};

use rl4burn::env::space::Space;
use rl4burn::{
    masked_ppo_collect, masked_ppo_update, orthogonal_linear, ActionDist, Env,
    MaskedActorCritic, PpoConfig, Step, SyncVecEnv,
};

/// Backend aliases — NdArray is CPU-only but requires no GPU drivers,
/// making this example runnable anywhere.
type AutodiffB = Autodiff<NdArray>;

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// Grid size for the navigation world.
const GRID_SIZE: i32 = 5;
/// Maximum steps before an episode is truncated.
const MAX_STEPS: usize = 50;

/// A 2D grid navigation environment with multi-discrete actions and masking.
///
/// The agent starts in the centre and must reach a random target position.
/// At each step it independently chooses a horizontal move (dx) and a
/// vertical move (dy), each from {-1, 0, +1}.
///
/// Boundary masking prevents the agent from choosing a direction that would
/// move it off the grid — this is the simplest possible example of the kind
/// of action masking used in real strategy games.
struct NavEnv<R> {
    x: i32,
    y: i32,
    target_x: i32,
    target_y: i32,
    step_count: usize,
    rng: R,
}

impl<R: rand::Rng> NavEnv<R> {
    fn new(mut rng: R) -> Self {
        let tx = rng.random_range(0..GRID_SIZE);
        let ty = rng.random_range(0..GRID_SIZE);
        Self {
            x: GRID_SIZE / 2,
            y: GRID_SIZE / 2,
            target_x: tx,
            target_y: ty,
            step_count: 0,
            rng,
        }
    }

    /// Build the 4-element observation vector.
    fn obs(&self) -> Vec<f32> {
        vec![
            self.x as f32 / GRID_SIZE as f32,
            self.y as f32 / GRID_SIZE as f32,
            self.target_x as f32 / GRID_SIZE as f32,
            self.target_y as f32 / GRID_SIZE as f32,
        ]
    }
}

impl<R: rand::Rng + Clone> Env for NavEnv<R> {
    type Observation = Vec<f32>;
    /// Actions are `Vec<f32>` with one element per sub-action dimension.
    /// For MultiDiscrete([3, 3]) this is `[dx_index, dy_index]` where each
    /// index is 0, 1, or 2 (cast to f32).
    type Action = Vec<f32>;

    fn reset(&mut self) -> Vec<f32> {
        self.x = GRID_SIZE / 2;
        self.y = GRID_SIZE / 2;
        self.target_x = self.rng.random_range(0..GRID_SIZE);
        self.target_y = self.rng.random_range(0..GRID_SIZE);
        self.step_count = 0;
        self.obs()
    }

    fn step(&mut self, action: Vec<f32>) -> Step<Vec<f32>> {
        // Decode: action[0] in {0,1,2} maps to dx in {-1, 0, +1}
        //         action[1] in {0,1,2} maps to dy in {-1, 0, +1}
        let dx = action[0] as i32 - 1;
        let dy = action[1] as i32 - 1;

        self.x = (self.x + dx).clamp(0, GRID_SIZE - 1);
        self.y = (self.y + dy).clamp(0, GRID_SIZE - 1);
        self.step_count += 1;

        let dist = ((self.x - self.target_x).abs() + (self.y - self.target_y).abs()) as f32;
        let reached = self.x == self.target_x && self.y == self.target_y;

        // Shaped reward: big bonus for reaching the target, small penalty
        // proportional to Manhattan distance otherwise.
        let reward = if reached {
            10.0
        } else {
            -0.1 - dist * 0.05
        };

        let terminated = reached;
        let truncated = self.step_count >= MAX_STEPS;

        Step {
            observation: self.obs(),
            reward,
            terminated,
            truncated,
        }
    }

    fn observation_space(&self) -> Space {
        Space::Box {
            low: vec![0.0; 4],
            high: vec![1.0; 4],
        }
    }

    /// `MultiDiscrete([3, 3])` — two independent sub-actions, each with 3
    /// choices.  The total logit count the policy must output is 3 + 3 = 6.
    fn action_space(&self) -> Space {
        Space::MultiDiscrete(vec![3, 3])
    }

    /// Build the action mask: a flat `Vec<f32>` of length 6 (3 + 3).
    ///
    /// The first 3 elements correspond to dx choices {left, stay, right}.
    /// The next 3 correspond to dy choices {down, stay, up}.
    ///
    /// A value of `1.0` means the action is *valid*; `0.0` means *invalid*.
    /// During policy evaluation, invalid logits are set to `-inf` before
    /// the softmax, guaranteeing zero probability for forbidden moves.
    ///
    /// In a real game this mask might forbid:
    /// - moving into impassable terrain
    /// - attacking during a cooldown
    /// - selecting a dead unit
    fn action_mask(&self) -> Option<Vec<f32>> {
        let mut mask = vec![1.0f32; 6];

        // dx sub-action (indices 0, 1, 2):
        //   index 0 = move left (dx = -1)  → invalid at left wall
        //   index 2 = move right (dx = +1) → invalid at right wall
        if self.x == 0 {
            mask[0] = 0.0;
        }
        if self.x == GRID_SIZE - 1 {
            mask[2] = 0.0;
        }

        // dy sub-action (indices 3, 4, 5):
        //   index 3 = move down (dy = -1)  → invalid at bottom wall
        //   index 5 = move up (dy = +1)    → invalid at top wall
        if self.y == 0 {
            mask[3] = 0.0;
        }
        if self.y == GRID_SIZE - 1 {
            mask[5] = 0.0;
        }

        Some(mask)
    }
}

// ---------------------------------------------------------------------------
// Neural network (actor-critic)
// ---------------------------------------------------------------------------

/// A simple MLP actor-critic for the navigation task.
///
/// The architecture has:
/// - Two hidden layers (64 units, tanh activation).
/// - A **policy head** producing 6 logits (3 per sub-action).  These logits
///   are split into two groups of 3 by `ActionDist::MultiDiscrete([3, 3])`.
/// - A **value head** producing a single scalar state value.
///
/// ### How logits are split across sub-action heads
///
/// The policy head outputs a single tensor of shape `[batch, 6]`.
/// `ActionDist::MultiDiscrete([3, 3])` tells the PPO implementation to
/// split this into `[batch, 3]` and `[batch, 3]`, apply masking and
/// softmax to each independently, and sample one index from each.
///
/// If you had `MultiDiscrete([3, 3, 4])` the policy head would output 10
/// logits, split as [3, 3, 4].  No changes are needed to the collect/update
/// code — just update the `ActionDist` and the environment's `action_space`.
#[derive(Module, Debug)]
struct NavAgent<B: Backend> {
    fc1: burn::nn::Linear<B>,
    fc2: burn::nn::Linear<B>,
    policy_head: burn::nn::Linear<B>,
    value_head: burn::nn::Linear<B>,
}

impl<B: Backend> NavAgent<B> {
    fn new(device: &B::Device, rng: &mut impl rand::Rng) -> Self {
        let sqrt2 = std::f32::consts::SQRT_2;
        Self {
            // Orthogonal init with gain sqrt(2) for hidden layers
            fc1: orthogonal_linear(4, 64, sqrt2, device, rng),
            fc2: orthogonal_linear(64, 64, sqrt2, device, rng),
            // Policy head: 6 logits total (3 for dx + 3 for dy).
            // Small gain (0.01) keeps initial policy near-uniform.
            policy_head: orthogonal_linear(64, 6, 0.01, device, rng),
            // Value head: single scalar, gain 1.0
            value_head: orthogonal_linear(64, 1, 1.0, device, rng),
        }
    }
}

/// Implement `MaskedActorCritic` so we can use the masked PPO pipeline.
///
/// The trait requires a single method:
///
/// ```ignore
/// fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>);
///                                          // logits        // values
/// ```
///
/// The masking is handled *externally* by `masked_ppo_collect` and
/// `masked_ppo_update` — the model itself just outputs raw logits.
impl<B: Backend> MaskedActorCritic<B> for NavAgent<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let h = self.fc1.forward(obs).tanh();
        let h = self.fc2.forward(h).tanh();
        let logits = self.policy_head.forward(h.clone());
        let values = self.value_head.forward(h).squeeze_dim::<1>(1);
        (logits, values)
    }
}

// ---------------------------------------------------------------------------
// Training loop
// ---------------------------------------------------------------------------

fn main() {
    let device = NdArrayDevice::Cpu;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    // --- Vectorised environment ------------------------------------------
    //
    // Running multiple environment copies in parallel gives more diverse
    // data per rollout and is standard practice for on-policy algorithms.
    let n_envs = 8;
    let envs: Vec<NavEnv<rand::rngs::SmallRng>> = (0..n_envs)
        .map(|i| NavEnv::new(rand::rngs::SmallRng::seed_from_u64(42 + i as u64)))
        .collect();
    let mut vec_env = SyncVecEnv::new(envs);

    // --- Model -----------------------------------------------------------
    let model: NavAgent<AutodiffB> = NavAgent::new(&device, &mut rng);

    // --- Action distribution ---------------------------------------------
    //
    // `ActionDist::MultiDiscrete(vec![3, 3])` is the key piece that tells
    // the PPO collect/update code how to interpret the 6 logits from the
    // policy head.  It splits them into two groups of 3, treating each as
    // an independent categorical distribution.
    let action_dist = ActionDist::MultiDiscrete(vec![3, 3]);

    // --- Optimiser -------------------------------------------------------
    let mut optim = AdamConfig::new().with_epsilon(1e-5).init();

    // --- PPO hyperparameters ---------------------------------------------
    let config = PpoConfig {
        lr: 3e-4,
        gamma: 0.99,
        gae_lambda: 0.95,
        clip_eps: 0.2,
        vf_coef: 0.5,
        ent_coef: 0.01,        // encourage exploration across sub-actions
        update_epochs: 4,
        minibatch_size: 64,
        n_steps: 64,            // steps per env per rollout
        clip_vloss: true,
        max_grad_norm: 0.5,
        target_kl: None,
        dual_clip_coef: None,
    };

    // --- Training loop ---------------------------------------------------
    let mut model = model;
    let total_timesteps = 200_000;
    let steps_per_iter = config.n_steps * n_envs; // 64 * 8 = 512
    let n_iterations = total_timesteps / steps_per_iter;

    let mut recent_returns: Vec<f32> = Vec::new();
    let mut best_avg = f32::NEG_INFINITY;
    let mut current_obs = vec_env.reset();
    let mut ep_acc = vec![0.0f32; n_envs];

    println!("PPO Multi-Discrete Navigation");
    println!("  Grid:        {GRID_SIZE}x{GRID_SIZE}");
    println!("  Action space: MultiDiscrete([3, 3])  (dx, dy)");
    println!("  Masking:     boundary walls");
    println!("  Envs:        {n_envs}");
    println!("  Timesteps:   {total_timesteps}");
    println!("  Iterations:  {n_iterations}");
    println!();

    for iter in 0..n_iterations {
        // Linear learning-rate annealing
        let frac = 1.0 - iter as f64 / n_iterations as f64;
        let current_lr = config.lr * frac;

        // --- Collect rollout with masking --------------------------------
        //
        // `masked_ppo_collect` does the following for each of the n_steps:
        //   1. Calls `model.forward(obs)` to get raw logits and values.
        //   2. Queries each env's `action_mask()`.
        //   3. Applies the mask to the logits (setting invalid to -inf).
        //   4. Splits masked logits by sub-action sizes [3, 3].
        //   5. Samples one index from each sub-distribution.
        //   6. Stores obs, actions, log-probs, values, rewards, and masks.
        let inference_model = model.valid();
        let rollout = masked_ppo_collect::<NdArray, _, _>(
            &inference_model,
            &mut vec_env,
            &action_dist,
            &config,
            &device,
            &mut rng,
            &mut current_obs,
            &mut ep_acc,
        );

        recent_returns.extend_from_slice(&rollout.episode_returns);

        // --- PPO update with masking -------------------------------------
        //
        // `masked_ppo_update` recomputes log-probs using the stored masks
        // so that the probability ratios are correct even though the policy
        // has changed since collection.  This is essential — without
        // reapplying masks during the update, the clipped objective would
        // be computed with wrong probabilities.
        let stats;
        (model, stats) = masked_ppo_update(
            model,
            &mut optim,
            &rollout,
            &action_dist,
            &config,
            current_lr,
            &device,
            &mut rng,
        );

        // Keep a rolling window of recent episode returns for logging.
        if recent_returns.len() > 50 {
            let start = recent_returns.len() - 50;
            recent_returns = recent_returns[start..].to_vec();
        }

        if !recent_returns.is_empty() {
            let avg: f32 =
                recent_returns.iter().sum::<f32>() / recent_returns.len() as f32;
            best_avg = best_avg.max(avg);

            if (iter + 1) % 20 == 0 || iter == 0 {
                println!(
                    "iter {:>4}/{}: avg_return={:>6.1}  best={:>6.1}  \
                     policy_loss={:>8.4}  entropy={:.3}  lr={:.2e}",
                    iter + 1,
                    n_iterations,
                    avg,
                    best_avg,
                    stats.policy_loss,
                    stats.entropy,
                    current_lr,
                );
            }
        }
    }

    println!();
    println!("Training complete.  Best rolling average return: {best_avg:.1}");
    if best_avg > 0.0 {
        println!("Agent learned to navigate to the target!");
    } else {
        println!(
            "Agent did not converge (avg {best_avg:.1}).  \
             Try more timesteps or tune hyperparameters."
        );
    }
}
