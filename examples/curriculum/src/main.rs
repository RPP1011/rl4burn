//! Curriculum learning — progressive environment difficulty for game AI.
//!
//! Trains a PPO agent on a simple "Platformer" environment where the agent
//! must jump across gaps on a 1D line. Difficulty increases as the agent
//! improves: small gaps first, then wider gaps with narrower platforms.
//!
//! ## Why curriculum learning?
//!
//! Many RL tasks are too hard to learn from scratch. If the agent never
//! receives positive reward, it has no gradient signal to improve. Curriculum
//! learning solves this by starting with an easy version of the task and
//! progressively increasing difficulty as the agent demonstrates competence.
//!
//! This is common in game AI (e.g., OpenAI Five started with shorter game
//! durations) and robotics (e.g., increasing target distances).
//!
//! ## When to advance difficulty
//!
//! Two main approaches:
//! - **Performance threshold** (used here): advance when the rolling average
//!   return exceeds a target. This is adaptive — the agent stays at each
//!   level until it truly masters it.
//! - **Fixed schedule**: advance after N training steps regardless of
//!   performance. Simpler but risks advancing before the agent is ready.
//!
//! ## Catastrophic forgetting
//!
//! A key risk: after advancing to harder levels, the agent may "forget" how
//! to solve easier ones. This matters because harder levels often require
//! the skills from easier ones. Mitigation strategies:
//! - **Mix difficulties** (used here): sample 70% from the current level,
//!   30% from previous levels. This provides a rehearsal signal.
//! - **Domain randomization**: randomize difficulty uniformly across all
//!   levels from the start. Simpler but may slow initial learning.
//! - **Progressive networks**: freeze old weights and add new capacity.
//!   More complex but guarantees no forgetting.
//!
//! Run: `cargo run --release -p curriculum`

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::nn::{Linear, LinearConfig};
use burn::optim::AdamConfig;
use burn::prelude::*;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use rl4burn::{
    ppo_collect, ppo_update, DiscreteAcOutput, DiscreteActorCritic, Env, PpoConfig, Space, Step,
    SyncVecEnv,
};

// ---------------------------------------------------------------------------
// Difficulty levels
// ---------------------------------------------------------------------------

/// Parameters that define a difficulty level for the Platformer environment.
#[derive(Debug, Clone, Copy)]
struct DifficultyLevel {
    /// Human-readable name for logging.
    name: &'static str,
    /// Width of the gap the agent must jump across (larger = harder).
    gap_width: f32,
    /// Width of each platform (smaller = harder, less room for error).
    platform_width: f32,
    /// Number of obstacles (gaps) to cross to complete the level.
    num_gaps: usize,
}

/// The three-tier curriculum.
const LEVELS: [DifficultyLevel; 3] = [
    DifficultyLevel {
        name: "Easy",
        gap_width: 1.0,
        platform_width: 4.0,
        num_gaps: 1,
    },
    DifficultyLevel {
        name: "Medium",
        gap_width: 2.0,
        platform_width: 3.0,
        num_gaps: 2,
    },
    DifficultyLevel {
        name: "Hard",
        gap_width: 3.0,
        platform_width: 2.0,
        num_gaps: 3,
    },
];

// ---------------------------------------------------------------------------
// Platformer environment
// ---------------------------------------------------------------------------

/// A 1D platformer where the agent moves right, jumping across gaps.
///
/// The world is a sequence of platforms separated by gaps. The agent starts
/// on the leftmost platform and must reach the rightmost one.
///
/// - **Observation**: `[position, velocity, next_gap_start, next_gap_end, gaps_remaining]`
/// - **Actions**: 0 = step right (+0.5), 1 = jump right (+gap_width + 0.5),
///               2 = small step right (+0.2), 3 = stay (no-op)
/// - **Reward**: +3 per gap successfully crossed, +5 bonus for completing
///   all gaps, -1 for falling into a gap, small step reward for progress.
/// - **Terminated**: fell into a gap or completed all gaps.
/// - **Truncated**: exceeded max steps.
///
/// The environment picks its own difficulty on each reset, based on the
/// `max_unlocked_level` field. This lets the curriculum manager control
/// difficulty without needing mutable access to individual environments
/// inside the vectorized wrapper.
struct Platformer {
    position: f32,
    velocity: f32,
    /// Current episode's difficulty.
    current_level: DifficultyLevel,
    /// Highest difficulty level unlocked by the curriculum. On each reset,
    /// the environment samples: 70% current level, 30% earlier levels.
    /// This mixing prevents catastrophic forgetting — the agent continues
    /// to see easier problems while learning harder ones.
    max_unlocked_level: usize,
    /// Start positions of each gap.
    gap_starts: Vec<f32>,
    /// Number of gaps the agent has crossed so far.
    gaps_crossed: usize,
    step_count: usize,
    max_steps: usize,
    rng: SmallRng,
}

impl Platformer {
    fn new(max_unlocked_level: usize, rng: SmallRng) -> Self {
        let mut env = Self {
            position: 0.0,
            velocity: 0.0,
            current_level: LEVELS[0],
            max_unlocked_level,
            gap_starts: Vec::new(),
            gaps_crossed: 0,
            step_count: 0,
            max_steps: 50,
            rng,
        };
        env.current_level = env.sample_level();
        env.generate_gaps();
        env
    }

    /// Sample a difficulty level, mixing current and previous levels.
    ///
    /// This mixing is crucial to prevent catastrophic forgetting. Without it,
    /// the agent would only train on the hardest unlocked level and might
    /// forget the foundational skills learned at earlier levels.
    fn sample_level(&mut self) -> DifficultyLevel {
        let max = self.max_unlocked_level;
        if max == 0 || self.rng.random::<f32>() < 0.7 {
            LEVELS[max]
        } else {
            // Sample uniformly from previous levels.
            let idx = self.rng.random_range(0..max);
            LEVELS[idx]
        }
    }

    fn generate_gaps(&mut self) {
        self.gap_starts.clear();
        let mut pos = self.current_level.platform_width;
        for _ in 0..self.current_level.num_gaps {
            // Add some randomness to gap placement so the agent generalizes.
            let jitter: f32 = self.rng.random_range(-0.3_f32..0.3);
            self.gap_starts.push(pos + jitter.max(0.0));
            pos += self.current_level.gap_width + self.current_level.platform_width;
        }
    }

    fn next_gap_start(&self) -> f32 {
        if self.gaps_crossed < self.gap_starts.len() {
            self.gap_starts[self.gaps_crossed]
        } else {
            // No more gaps — return a far-away position.
            100.0
        }
    }

    fn next_gap_end(&self) -> f32 {
        self.next_gap_start() + self.current_level.gap_width
    }

    fn is_in_gap(&self) -> bool {
        if self.gaps_crossed >= self.gap_starts.len() {
            return false;
        }
        let gap_start = self.gap_starts[self.gaps_crossed];
        let gap_end = gap_start + self.current_level.gap_width;
        self.position > gap_start && self.position < gap_end
    }

    fn obs(&self) -> Vec<f32> {
        vec![
            self.position / 20.0,                         // normalized position
            self.velocity,                                 // velocity
            self.next_gap_start() / 20.0,                 // next gap start (normalized)
            self.next_gap_end() / 20.0,                   // next gap end (normalized)
            self.gaps_crossed as f32 / 3.0,               // progress
        ]
    }
}

impl Env for Platformer {
    type Observation = Vec<f32>;
    type Action = usize;

    fn reset(&mut self) -> Vec<f32> {
        // Each reset picks a new difficulty level based on the curriculum.
        self.current_level = self.sample_level();
        self.position = 0.0;
        self.velocity = 0.0;
        self.gaps_crossed = 0;
        self.step_count = 0;
        self.generate_gaps();
        self.obs()
    }

    fn step(&mut self, action: usize) -> Step<Vec<f32>> {
        self.step_count += 1;

        // Apply action.
        let old_pos = self.position;
        match action {
            0 => self.velocity = 0.5,                                              // step right
            1 => self.velocity = self.current_level.gap_width + 0.5,               // jump
            2 => self.velocity = 0.2,                                              // small step
            3 => self.velocity = 0.0,                                              // stay
            _ => panic!("Invalid action {action}, expected 0..4"),
        }
        self.position += self.velocity;

        // Check if the agent crossed a gap.
        let mut reward = 0.01 * (self.position - old_pos); // small progress reward

        if self.gaps_crossed < self.gap_starts.len() {
            let gap_end = self.gap_starts[self.gaps_crossed] + self.current_level.gap_width;
            if self.position >= gap_end {
                // Successfully jumped the gap.
                self.gaps_crossed += 1;
                reward += 3.0;
            }
        }

        // Check if the agent fell into a gap.
        let fell = self.is_in_gap();
        if fell {
            reward = -1.0;
        }

        // Check completion (past all gaps).
        let completed = self.gaps_crossed >= self.current_level.num_gaps
            && self.gaps_crossed > 0
            && self.position >= self.next_gap_start();
        if completed {
            reward += 5.0;
        }

        let terminated = fell || completed;
        let truncated = !terminated && self.step_count >= self.max_steps;

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

    fn observation_space(&self) -> Space {
        Space::Box {
            low: vec![-1.0; 5],
            high: vec![5.0; 5],
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(4)
    }
}

// ---------------------------------------------------------------------------
// Curriculum manager
// ---------------------------------------------------------------------------

/// Tracks agent performance and decides when to advance difficulty.
struct Curriculum {
    current_level: usize,
    /// Rolling window of recent episode returns.
    recent_returns: Vec<f32>,
    /// Size of the rolling window.
    window_size: usize,
    /// Thresholds for advancing to the next level.
    advance_thresholds: Vec<f32>,
}

impl Curriculum {
    fn new(window_size: usize) -> Self {
        Self {
            current_level: 0,
            recent_returns: Vec::new(),
            window_size,
            advance_thresholds: vec![7.0, 10.0, f32::INFINITY],
        }
    }

    /// Record episode returns and check whether to advance.
    /// Returns true if the level advanced.
    fn update(&mut self, returns: &[f32]) -> bool {
        self.recent_returns.extend_from_slice(returns);
        if self.recent_returns.len() > self.window_size {
            let start = self.recent_returns.len() - self.window_size;
            self.recent_returns = self.recent_returns[start..].to_vec();
        }

        if self.recent_returns.len() >= self.window_size / 2
            && self.current_level < LEVELS.len() - 1
        {
            let avg = self.avg_return();
            let threshold = self.advance_thresholds[self.current_level];
            if avg >= threshold {
                self.current_level += 1;
                self.recent_returns.clear();
                eprintln!(
                    "*** CURRICULUM ADVANCE: avg return {avg:.1} >= threshold {threshold:.1} ***"
                );
                eprintln!(
                    "*** Now training on level {} ({}) ***",
                    self.current_level,
                    LEVELS[self.current_level].name,
                );
                return true;
            }
        }
        false
    }

    fn avg_return(&self) -> f32 {
        if self.recent_returns.is_empty() {
            return 0.0;
        }
        self.recent_returns.iter().sum::<f32>() / self.recent_returns.len() as f32
    }
}

// ---------------------------------------------------------------------------
// Model (obs_dim=5, n_actions=4)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct Model<B: Backend> {
    actor1: Linear<B>,
    actor2: Linear<B>,
    actor_out: Linear<B>,
    critic1: Linear<B>,
    critic2: Linear<B>,
    critic_out: Linear<B>,
}

impl<B: Backend> Model<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            actor1: LinearConfig::new(5, 64).init(device),
            actor2: LinearConfig::new(64, 64).init(device),
            actor_out: LinearConfig::new(64, 4).init(device),
            critic1: LinearConfig::new(5, 64).init(device),
            critic2: LinearConfig::new(64, 64).init(device),
            critic_out: LinearConfig::new(64, 1).init(device),
        }
    }
}

impl<B: Backend> DiscreteActorCritic<B> for Model<B> {
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
// Main
// ---------------------------------------------------------------------------

type B = Autodiff<NdArray>;

fn main() {
    let device = NdArrayDevice::Cpu;
    let mut rng = SmallRng::seed_from_u64(42);

    let n_envs: usize = 8;

    // Start all environments at the easiest difficulty.
    // Each environment internally samples its level on reset, so we control
    // the curriculum by updating the `max_unlocked_level` field.
    let envs: Vec<_> = (0..n_envs)
        .map(|i| Platformer::new(0, SmallRng::seed_from_u64(1000 + i as u64)))
        .collect();
    let mut vec_env = SyncVecEnv::new(envs);

    let mut model: Model<B> = Model::new(&device);
    let mut optim = AdamConfig::new().with_epsilon(1e-5).init();
    let config = PpoConfig::new()
        .with_n_steps(64)
        .with_minibatch_size(128);

    // Curriculum: advance when 30-episode rolling average exceeds threshold.
    // 70% of environments train on the current (hardest) level, 30% replay
    // earlier levels to prevent catastrophic forgetting.
    let mut curriculum = Curriculum::new(30);

    let total_steps = 200_000;
    let steps_per_iter = config.n_steps * n_envs;
    let n_iters = total_steps / steps_per_iter;

    let mut current_obs = vec_env.reset();
    let mut ep_acc = vec![0.0f32; n_envs];

    eprintln!("=== Curriculum Learning: Platformer ===");
    eprintln!("Starting at level 0 ({})", LEVELS[0].name);
    eprintln!(
        "Levels: {} -> {} -> {}",
        LEVELS[0].name, LEVELS[1].name, LEVELS[2].name
    );
    eprintln!("{:-<70}", "");

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

        // Feed completed episode returns to the curriculum manager.
        // If the level advanced, we need to rebuild the environments with
        // the new max_unlocked_level. Since SyncVecEnv's environments
        // auto-reset, they will pick up the new difficulty on their next
        // episode. We rebuild to immediately expose all envs to the new
        // curriculum distribution.
        let advanced = curriculum.update(&rollout.episode_returns);
        if advanced {
            let new_level = curriculum.current_level;
            let envs: Vec<_> = (0..n_envs)
                .map(|i| {
                    Platformer::new(new_level, SmallRng::seed_from_u64(rng.random::<u64>() + i as u64))
                })
                .collect();
            vec_env = SyncVecEnv::new(envs);
            current_obs = vec_env.reset();
            ep_acc = vec![0.0f32; n_envs];
        }

        let stats;
        (model, stats) =
            ppo_update(model, &mut optim, &rollout, &config, lr, &device, &mut rng);

        if (iter + 1) % 10 == 0 {
            let avg = curriculum.avg_return();
            let level = &LEVELS[curriculum.current_level];
            let threshold = if curriculum.current_level < LEVELS.len() - 1 {
                curriculum.advance_thresholds[curriculum.current_level]
            } else {
                f32::INFINITY
            };
            eprintln!(
                "step {:>6} | level {} ({:<6}) | avg return {:>6.2} (threshold {:.1}) | \
                 policy_loss {:.4} | entropy {:.4}",
                (iter + 1) * steps_per_iter,
                curriculum.current_level,
                level.name,
                avg,
                threshold,
                stats.policy_loss,
                stats.entropy,
            );
        }
    }

    eprintln!("{:-<70}", "");
    eprintln!(
        "Training complete. Final level: {} ({})",
        curriculum.current_level,
        LEVELS[curriculum.current_level].name,
    );
    eprintln!("Final avg return: {:.2}", curriculum.avg_return());

    // Summary of the curriculum approach:
    //
    // - Started easy so the agent could learn basic movement and jumping.
    // - Advanced difficulty only when performance was good enough (adaptive).
    // - Mixed old and new difficulties to retain foundational skills.
    //
    // Alternative: domain randomization
    // Instead of a curriculum, you can randomize difficulty uniformly from
    // the start. This works well when:
    //   - The task is learnable (some reward signal) even at high difficulty.
    //   - You want a single policy that handles all difficulty levels.
    //   - You don't want to tune advancement thresholds.
    // However, it can be much slower when the hardest levels give zero reward
    // without prerequisite skills.
}
