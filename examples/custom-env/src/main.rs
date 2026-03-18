//! Building a custom environment.
//!
//! Implements the `Env` trait for a simple "NumberLine" game where the agent
//! starts at 0 and must reach a target position by moving left or right.
//! Demonstrates how to define your own environment and train PPO on it.

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::nn::{Linear, LinearConfig};
use burn::optim::AdamConfig;
use burn::prelude::*;
use rand::SeedableRng;

use rl4burn::{
    ppo_collect, ppo_update, DiscreteAcOutput, DiscreteActorCritic, Env, PpoConfig, Space, Step,
    SyncVecEnv,
};

// ---------------------------------------------------------------------------
// Custom environment: NumberLine
// ---------------------------------------------------------------------------

/// The agent starts at position 0.0 on a number line and must reach the
/// target (default: 5.0) within a limited number of steps.
///
/// - Observation: `[position, target]` (2 floats)
/// - Action: 0 = move left (-1), 1 = stay (0), 2 = move right (+1)
/// - Reward: -|position - target| / max_steps (shaped to encourage progress)
///           +10.0 bonus when |position - target| < 0.5 (success)
/// - Terminated: when the agent reaches the target
/// - Truncated: after max_steps
struct NumberLine {
    position: f32,
    target: f32,
    step_count: usize,
    max_steps: usize,
}

impl NumberLine {
    fn new(target: f32, max_steps: usize) -> Self {
        Self {
            position: 0.0,
            target,
            step_count: 0,
            max_steps,
        }
    }
}

impl Env for NumberLine {
    type Observation = Vec<f32>;
    type Action = usize;

    fn reset(&mut self) -> Vec<f32> {
        self.position = 0.0;
        self.step_count = 0;
        vec![self.position, self.target]
    }

    fn step(&mut self, action: usize) -> Step<Vec<f32>> {
        // Apply action: 0 = left, 1 = stay, 2 = right
        let delta = match action {
            0 => -1.0,
            1 => 0.0,
            2 => 1.0,
            _ => panic!("Invalid action {action}, expected 0..3"),
        };
        self.position += delta;
        self.step_count += 1;

        let distance = (self.position - self.target).abs();
        let terminated = distance < 0.5;
        let truncated = self.step_count >= self.max_steps;

        // Shaped reward: small penalty proportional to distance, big bonus for reaching goal
        let reward = if terminated {
            10.0
        } else {
            -distance / self.max_steps as f32
        };

        let obs = if terminated || truncated {
            self.reset()
        } else {
            vec![self.position, self.target]
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
            low: vec![-10.0, -10.0],
            high: vec![10.0, 10.0],
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(3)
    }
}

// ---------------------------------------------------------------------------
// Model (obs_dim=2, n_actions=3)
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
            actor1: LinearConfig::new(2, 32).init(device),
            actor2: LinearConfig::new(32, 32).init(device),
            actor_out: LinearConfig::new(32, 3).init(device),
            critic1: LinearConfig::new(2, 32).init(device),
            critic2: LinearConfig::new(32, 32).init(device),
            critic_out: LinearConfig::new(32, 1).init(device),
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
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    let n_envs = 4;
    let envs: Vec<_> = (0..n_envs).map(|_| NumberLine::new(5.0, 20)).collect();
    let mut vec_env = SyncVecEnv::new(envs);

    let mut model: Model<B> = Model::new(&device);
    let mut optim = AdamConfig::new().with_epsilon(1e-5).init();
    let config = PpoConfig::new()
        .with_n_steps(64)
        .with_minibatch_size(64);

    let total_steps = 100_000;
    let steps_per_iter = config.n_steps * n_envs;
    let n_iters = total_steps / steps_per_iter;

    let mut current_obs = vec_env.reset();
    let mut ep_acc = vec![0.0f32; n_envs];
    let mut recent: Vec<f32> = Vec::new();

    eprintln!("Training PPO on NumberLine (target=5.0, {total_steps} steps, {n_envs} envs)");
    eprintln!("The agent starts at 0 and must reach 5 by moving left/right.");
    eprintln!("{:-<80}", "");

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
        recent.extend_from_slice(&rollout.episode_returns);
        if recent.len() > 20 {
            recent = recent[recent.len() - 20..].to_vec();
        }

        let stats;
        (model, stats) =
            ppo_update(model, &mut optim, &rollout, &config, lr, &device, &mut rng);

        if !recent.is_empty() && (iter + 1) % 10 == 0 {
            let avg: f32 = recent.iter().sum::<f32>() / recent.len() as f32;
            eprintln!(
                "step {:>6} | avg return {avg:>7.2} | policy_loss {:.4} | entropy {:.4}",
                (iter + 1) * steps_per_iter,
                stats.policy_loss,
                stats.entropy,
            );
        }
    }

    if !recent.is_empty() {
        let avg: f32 = recent.iter().sum::<f32>() / recent.len() as f32;
        eprintln!("{:-<80}", "");
        eprintln!("Final avg return (last 20 eps): {avg:.2}");
        if avg > 5.0 {
            eprintln!("NumberLine solved! The agent learned to reach the target.");
        }
    }
}
