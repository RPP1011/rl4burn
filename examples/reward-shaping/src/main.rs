//! Reward shaping for game AI.
//!
//! Demonstrates intrinsic rewards and reward shaping techniques for
//! environments with sparse rewards where the agent receives no signal
//! until it reaches a distant goal.
//!
//! # Why sparse rewards make RL hard
//!
//! In dense-reward environments (like CartPole, +1 every step), the agent
//! gets immediate gradient signal from every action. In sparse-reward
//! environments (like a maze where reward=1 only at the goal), the agent
//! must stumble upon the goal through random exploration before it can
//! learn anything. With high-dimensional state spaces, random exploration
//! almost never finds the goal, so training stalls at zero reward.
//!
//! # Solution: intrinsic motivation
//!
//! Intrinsic rewards provide a "curiosity" bonus that rewards visiting
//! novel states, giving the agent gradient signal even before finding the
//! extrinsic reward. This example uses count-based exploration:
//!
//!   intrinsic_reward(s') = 1 / sqrt(N(s'))
//!
//! where N(s') is the number of times state s' has been visited. Novel
//! states get high bonuses; frequently visited states get small ones.
//!
//! # Combining rewards
//!
//!   total_reward = extrinsic + intrinsic_coef * intrinsic
//!
//! The `intrinsic_coef` hyperparameter controls the exploration/exploitation
//! trade-off. Too high and the agent ignores the real objective; too low
//! and exploration is insufficient. Typical values: 0.01 to 0.5.
//!
//! # Other shaping techniques (discussed in comments below)
//!
//! - **Potential-based shaping**: F(s,s') = gamma*phi(s') - phi(s)
//!   guarantees the same optimal policy as the unshaped reward (Ng et al., 1999).
//! - **Reward normalization**: running mean/std normalization prevents
//!   reward scale from dominating the value function.

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::nn::{Linear, LinearConfig};
use burn::optim::AdamConfig;
use burn::prelude::*;
use rand::{Rng, SeedableRng};

use rand::RngExt;

use rl4burn::{
    combine_rewards, ppo_collect, ppo_update, CountBasedReward, DiscreteAcOutput,
    DiscreteActorCritic, Env, IntrinsicReward, PpoConfig, Step, SyncVecEnv,
};

// ---------------------------------------------------------------------------
// MazeEnv: a simple sparse-reward grid environment
// ---------------------------------------------------------------------------

/// A 5x5 grid maze with sparse reward.
///
/// The agent starts at (0,0) and must reach the goal at (4,4).
/// Reward is 0 at every step EXCEPT when the agent reaches the goal (+10).
/// Actions: 0=up, 1=right, 2=down, 3=left.
///
/// Observation: [row/4, col/4] normalized to [0,1] range.
///
/// This is intentionally hard for RL: with random exploration the agent
/// must take ~8 correct steps in a row out of 4 possible actions per step.
/// The probability of randomly reaching the goal is vanishingly small
/// without exploration bonuses.
struct MazeEnv<R> {
    row: i32,
    col: i32,
    step_count: usize,
    max_steps: usize,
    rng: R,
}

impl<R: Rng> MazeEnv<R> {
    fn new(rng: R) -> Self {
        Self {
            row: 0,
            col: 0,
            step_count: 0,
            max_steps: 50,
            rng,
        }
    }

    fn obs(&self) -> Vec<f32> {
        vec![self.row as f32 / 4.0, self.col as f32 / 4.0]
    }
}

impl<R: Rng> Env for MazeEnv<R> {
    type Observation = Vec<f32>;
    type Action = usize;

    fn reset(&mut self) -> Vec<f32> {
        self.row = 0;
        self.col = 0;
        self.step_count = 0;
        self.obs()
    }

    fn step(&mut self, action: usize) -> Step<Vec<f32>> {
        // Apply action with some stochasticity (10% random action).
        // This makes the maze harder and more realistic.
        let actual_action = if self.rng.random_range(0.0..1.0) < 0.1 {
            self.rng.random_range(0..4)
        } else {
            action
        };

        match actual_action {
            0 => self.row = (self.row - 1).max(0),     // up
            1 => self.col = (self.col + 1).min(4),      // right
            2 => self.row = (self.row + 1).min(4),      // down
            3 => self.col = (self.col - 1).max(0),      // left
            _ => {}
        }

        self.step_count += 1;

        let at_goal = self.row == 4 && self.col == 4;
        // Sparse reward: ONLY at the goal
        let reward = if at_goal { 10.0 } else { 0.0 };

        let terminated = at_goal;
        let truncated = self.step_count >= self.max_steps;

        let obs = if terminated || truncated {
            self.reset()
        } else {
            self.obs()
        };

        Step {
            observation: obs,
            reward,
            terminated,
            truncated,
        }
    }

    fn observation_space(&self) -> rl4burn::Space {
        rl4burn::Space::Box {
            low: vec![0.0, 0.0],
            high: vec![1.0, 1.0],
        }
    }

    fn action_space(&self) -> rl4burn::Space {
        rl4burn::Space::Discrete(4)
    }
}

// ---------------------------------------------------------------------------
// Actor-critic model for the maze
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct MazeModel<B: Backend> {
    actor1: Linear<B>,
    actor2: Linear<B>,
    actor_out: Linear<B>,
    critic1: Linear<B>,
    critic2: Linear<B>,
    critic_out: Linear<B>,
}

impl<B: Backend> MazeModel<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            actor1: LinearConfig::new(2, 64).init(device),
            actor2: LinearConfig::new(64, 64).init(device),
            actor_out: LinearConfig::new(64, 4).init(device),
            critic1: LinearConfig::new(2, 64).init(device),
            critic2: LinearConfig::new(64, 64).init(device),
            critic_out: LinearConfig::new(64, 1).init(device),
        }
    }
}

impl<B: Backend> DiscreteActorCritic<B> for MazeModel<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> DiscreteAcOutput<B> {
        let a = self.actor1.forward(obs.clone()).tanh();
        let a = self.actor2.forward(a).tanh();
        let logits = self.actor_out.forward(a);

        let c = self.critic1.forward(obs).tanh();
        let c = self.critic2.forward(c).tanh();
        let values = self.critic_out.forward(c).squeeze_dim::<1>(1);

        DiscreteAcOutput { logits, values }
    }
}

// ---------------------------------------------------------------------------
// Potential-based reward shaping (theory reference)
// ---------------------------------------------------------------------------

/// Potential-based reward shaping: F(s, s') = gamma * phi(s') - phi(s).
///
/// This is the ONLY form of reward shaping that provably preserves the
/// optimal policy (Ng, Harada, Russell 1999). Any potential function phi(s)
/// can be used:
///   - phi(s) = -manhattan_distance(s, goal) for grid worlds
///   - phi(s) = -euclidean_distance(s, goal) for continuous spaces
///   - phi(s) = learned value function from a prior task (transfer learning)
///
/// The shaped reward becomes:
///   r_shaped = r_extrinsic + gamma * phi(s') - phi(s)
///
/// Key insight: potential-based shaping is equivalent to initializing the
/// value function to phi(s). It speeds up learning without changing what
/// is optimal.
fn potential_based_shaping(
    phi_s: f32,
    phi_s_prime: f32,
    gamma: f32,
) -> f32 {
    gamma * phi_s_prime - phi_s
}

/// Example potential function: negative Manhattan distance to goal (4,4).
fn maze_potential(obs: &[f32]) -> f32 {
    let row = (obs[0] * 4.0).round();
    let col = (obs[1] * 4.0).round();
    -((4.0 - row).abs() + (4.0 - col).abs())
}

// ---------------------------------------------------------------------------
// Reward normalization (running mean/std)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
/// Running reward normalizer using Welford's online algorithm.
///
/// Normalizes rewards to zero mean and unit variance, which stabilizes
/// training when reward scales vary across environments or change over time.
///
/// Usage: normalize each reward as (r - running_mean) / running_std.
///
/// WARNING: Be careful with sparse rewards. If most rewards are 0, the
/// running mean will be near 0 and std near 0, which can amplify noise.
/// Only use normalization when you have sufficiently frequent non-zero rewards.
struct RewardNormalizer {
    mean: f64,
    var: f64,
    count: f64,
}

#[allow(dead_code)]
impl RewardNormalizer {
    fn new() -> Self {
        Self {
            mean: 0.0,
            var: 1.0,
            count: 1e-4, // small epsilon to avoid division by zero
        }
    }

    fn update(&mut self, reward: f32) {
        let r = reward as f64;
        self.count += 1.0;
        let delta = r - self.mean;
        self.mean += delta / self.count;
        let delta2 = r - self.mean;
        self.var += delta * delta2;
    }

    fn normalize(&self, reward: f32) -> f32 {
        let std = (self.var / self.count).sqrt().max(1e-8);
        ((reward as f64 - self.mean) / std) as f32
    }
}

// ---------------------------------------------------------------------------
// Training helper
// ---------------------------------------------------------------------------

type B = Autodiff<NdArray>;

struct TrainResult {
    avg_return: f32,
    goal_rate: f32,
    unique_states: usize,
}

/// Run a PPO training loop on the maze, optionally with intrinsic rewards.
fn train_maze(
    seed: u64,
    intrinsic_coef: f32,
    use_potential_shaping: bool,
    total_steps: usize,
    label: &str,
) -> TrainResult {
    let device = NdArrayDevice::Cpu;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);

    let n_envs = 4;
    let envs: Vec<_> = (0..n_envs)
        .map(|i| MazeEnv::new(rand::rngs::SmallRng::seed_from_u64(seed + 100 + i as u64)))
        .collect();
    let mut vec_env = SyncVecEnv::new(envs);

    let mut model: MazeModel<B> = MazeModel::new(&device);
    let mut optim = AdamConfig::new().with_epsilon(1e-5).init();

    let config = PpoConfig {
        lr: 2.5e-4,
        gamma: 0.99,
        gae_lambda: 0.95,
        clip_eps: 0.2,
        vf_coef: 0.5,
        ent_coef: 0.01,
        update_epochs: 4,
        minibatch_size: 64,
        n_steps: 64,
        clip_vloss: true,
        max_grad_norm: 0.5,
        target_kl: None,
        dual_clip_coef: None,
    };

    let steps_per_iter = config.n_steps * n_envs;
    let n_iters = total_steps / steps_per_iter;

    let mut current_obs = vec_env.reset();
    let mut ep_acc = vec![0.0f32; n_envs];
    let mut all_returns: Vec<f32> = Vec::new();

    // Count-based intrinsic reward module — tracks how many times each
    // discretized state has been visited. Resolution 0.25 means each
    // grid cell maps to its own bin.
    let mut count_reward = CountBasedReward::new(0.25);

    eprintln!("  [{label}] Starting training ({total_steps} steps, intrinsic_coef={intrinsic_coef})");

    for iter in 0..n_iters {
        let lr = config.lr * (1.0 - iter as f64 / n_iters as f64);

        let rollout = ppo_collect::<NdArray, _, _>(
            &model.valid(),
            &mut vec_env,
            &config,
            &device,
            &mut rng,
            &mut current_obs,
            &mut ep_acc,
        );

        // If using intrinsic rewards, compute exploration bonuses and
        // combine them with the extrinsic rewards before the PPO update.
        //
        // The key insight: we modify the rewards in the rollout AFTER
        // collection but BEFORE the update. The PPO update uses these
        // combined rewards to compute advantages via GAE.
        let mut modified_rollout = rollout;

        if intrinsic_coef > 0.0 {
            let n_total = modified_rollout.observations.len();
            let mut intrinsic_rewards = Vec::with_capacity(n_total);

            for i in 0..n_total {
                let obs = &modified_rollout.observations[i];
                // For the "next observation", use the next step's obs or the current one
                // if we're at the end of the rollout.
                let next_obs = if i + n_envs < n_total {
                    &modified_rollout.observations[i + n_envs]
                } else {
                    obs
                };
                let action = modified_rollout.actions[i] as usize;

                // Compute intrinsic reward BEFORE updating counts so the
                // first visit to a state gets the maximum bonus.
                let ir = count_reward.reward(obs, action, next_obs);
                count_reward.update(obs, action, next_obs);
                intrinsic_rewards.push(ir);
            }

            // combine_rewards: combined[i] = extrinsic[i] + coef * intrinsic[i]
            //
            // Tuning intrinsic_coef:
            //   - 0.0:  no exploration bonus (pure extrinsic)
            //   - 0.01: gentle nudge toward novel states
            //   - 0.1:  moderate exploration pressure (good default)
            //   - 1.0:  strong exploration, may overwhelm sparse extrinsic signal
            //
            // A common schedule: start with high coef and anneal to 0 as
            // training progresses and the agent has explored sufficiently.
            let combined = combine_rewards(
                &modified_rollout.rewards,
                &intrinsic_rewards,
                intrinsic_coef,
            );
            modified_rollout.rewards = combined;
        }

        if use_potential_shaping {
            // Potential-based reward shaping example.
            // F(s,s') = gamma * phi(s') - phi(s)
            //
            // This adds a shaping reward proportional to how much closer
            // the agent moved to the goal. Moving toward (4,4) gets a
            // positive bonus; moving away gets a negative one.
            //
            // Unlike arbitrary reward shaping, potential-based shaping
            // preserves the optimal policy (Ng et al., 1999).
            let n_total = modified_rollout.observations.len();
            for i in 0..n_total {
                let obs = &modified_rollout.observations[i];
                let next_obs = if i + n_envs < n_total {
                    &modified_rollout.observations[i + n_envs]
                } else {
                    obs
                };
                let phi_s = maze_potential(obs);
                let phi_s_prime = maze_potential(next_obs);
                let shaping = potential_based_shaping(phi_s, phi_s_prime, config.gamma);
                modified_rollout.rewards[i] += shaping;
            }
        }

        all_returns.extend_from_slice(&modified_rollout.episode_returns);

        let stats;
        (model, stats) = ppo_update(
            model,
            &mut optim,
            &modified_rollout,
            &config,
            lr,
            &device,
            &mut rng,
        );

        // Progress reporting
        if (iter + 1) % 20 == 0 {
            let recent: Vec<f32> = if all_returns.len() > 20 {
                all_returns[all_returns.len() - 20..].to_vec()
            } else {
                all_returns.clone()
            };
            if !recent.is_empty() {
                let avg = recent.iter().sum::<f32>() / recent.len() as f32;
                let goals = recent.iter().filter(|&&r| r > 0.0).count();
                eprintln!(
                    "  [{label}] step {:>6} | avg_return {avg:>5.1} | goals {goals}/{} | \
                     entropy {:.3} | states_visited {}",
                    (iter + 1) * steps_per_iter,
                    recent.len(),
                    stats.entropy,
                    count_reward.num_visited(),
                );
            }
        }
    }

    let recent: Vec<f32> = if all_returns.len() > 20 {
        all_returns[all_returns.len() - 20..].to_vec()
    } else {
        all_returns.clone()
    };

    let avg_return = if recent.is_empty() {
        0.0
    } else {
        recent.iter().sum::<f32>() / recent.len() as f32
    };
    let goal_rate = if recent.is_empty() {
        0.0
    } else {
        recent.iter().filter(|&&r| r > 0.0).count() as f32 / recent.len() as f32
    };

    TrainResult {
        avg_return,
        goal_rate,
        unique_states: count_reward.num_visited(),
    }
}

// ---------------------------------------------------------------------------
// Main: compare training with and without intrinsic rewards
// ---------------------------------------------------------------------------

fn main() {
    eprintln!("=== Reward Shaping for Game AI ===");
    eprintln!();
    eprintln!("Environment: 5x5 Maze (sparse reward: +10 at goal only)");
    eprintln!("Without exploration bonuses, the agent rarely discovers the goal.");
    eprintln!("Count-based intrinsic rewards provide curiosity-driven exploration.");
    eprintln!();

    let total_steps = 50_000;

    // Experiment 1: No intrinsic rewards (baseline)
    eprintln!("--- Experiment 1: No intrinsic rewards (baseline) ---");
    let baseline = train_maze(42, 0.0, false, total_steps, "baseline");

    eprintln!();

    // Experiment 2: With count-based intrinsic rewards
    // intrinsic_coef = 0.1 provides moderate exploration pressure.
    eprintln!("--- Experiment 2: With count-based exploration (coef=0.1) ---");
    let with_intrinsic = train_maze(42, 0.1, false, total_steps, "intrinsic");

    eprintln!();

    // Experiment 3: With potential-based shaping
    eprintln!("--- Experiment 3: With potential-based shaping ---");
    let with_potential = train_maze(42, 0.0, true, total_steps, "potential");

    eprintln!();

    // Experiment 4: Combined intrinsic + potential shaping
    eprintln!("--- Experiment 4: Intrinsic + potential shaping ---");
    let combined = train_maze(42, 0.1, true, total_steps, "combined");

    // Print results comparison
    eprintln!();
    eprintln!("=== Results Comparison ===");
    eprintln!("{:<30} {:>10} {:>10} {:>15}", "Method", "Avg Return", "Goal Rate", "States Visited");
    eprintln!("{:-<65}", "");
    eprintln!(
        "{:<30} {:>10.1} {:>9.0}% {:>15}",
        "Baseline (no shaping)", baseline.avg_return, baseline.goal_rate * 100.0, baseline.unique_states
    );
    eprintln!(
        "{:<30} {:>10.1} {:>9.0}% {:>15}",
        "Count-based (coef=0.1)", with_intrinsic.avg_return, with_intrinsic.goal_rate * 100.0, with_intrinsic.unique_states
    );
    eprintln!(
        "{:<30} {:>10.1} {:>9.0}% {:>15}",
        "Potential-based", with_potential.avg_return, with_potential.goal_rate * 100.0, with_potential.unique_states
    );
    eprintln!(
        "{:<30} {:>10.1} {:>9.0}% {:>15}",
        "Intrinsic + potential", combined.avg_return, combined.goal_rate * 100.0, combined.unique_states
    );

    eprintln!();
    eprintln!("Key takeaways:");
    eprintln!("  - Intrinsic rewards drive the agent to explore more unique states.");
    eprintln!("  - Potential-based shaping provides a safe curriculum toward the goal.");
    eprintln!("  - Combining both approaches often works best for sparse-reward tasks.");
    eprintln!();
    eprintln!("In production game AI, you would also consider:");
    eprintln!("  - Annealing intrinsic_coef to 0 over training (explore early, exploit late).");
    eprintln!("  - Random Network Distillation (RND) for high-dimensional observations.");
    eprintln!("  - Reward normalization via running mean/std for multi-objective rewards.");
}
