//! Integration test: train DQN on CartPole and verify convergence.

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig};
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::tensor::activation::relu;

use rand::SeedableRng;

use rl4burn::envs::CartPole;
use rl4burn::{dqn_update, epsilon_greedy, epsilon_schedule, DqnConfig, Env, Transition};
use rl4burn::{polyak_update, ReplayBuffer};

type AB = Autodiff<NdArray>;

// ---------------------------------------------------------------------------
// Q-Network: simple MLP
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct QNet<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    q_head: Linear<B>,
}

impl<B: Backend> QNet<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            fc1: LinearConfig::new(4, 64).init(device),
            fc2: LinearConfig::new(64, 64).init(device),
            q_head: LinearConfig::new(64, 2).init(device),
        }
    }
}

impl<B: Backend> rl4burn::QNetwork<B> for QNet<B> {
    fn q_values(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = relu(self.fc1.forward(obs));
        let h = relu(self.fc2.forward(h));
        self.q_head.forward(h)
    }
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

#[test]
#[ignore] // slow: run with `cargo test -- --ignored`
fn dqn_solves_cartpole() {
    let device = NdArrayDevice::Cpu;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(1);

    let mut env = CartPole::new(rand::rngs::SmallRng::seed_from_u64(42));

    let mut online: QNet<AB> = QNet::new(&device);
    let mut target: QNet<AB> = online.clone();
    let mut optim = AdamConfig::new().with_epsilon(1e-5).init();

    let config = DqnConfig {
        lr: 1e-3,
        gamma: 0.99,
        buffer_capacity: 10_000,
        batch_size: 64,
        tau: 1.0, // hard update
        eps_start: 1.0,
        eps_end: 0.05,
        eps_decay_steps: 5_000,
        learning_starts: 256,
    };
    let train_frequency = 1;
    let target_update_freq = 250;
    let total_timesteps = 50_000;

    let mut buffer = ReplayBuffer::new(config.buffer_capacity, rand::rngs::SmallRng::seed_from_u64(99));
    let mut obs = env.reset();
    let mut recent_returns: Vec<f32> = Vec::new();
    let mut best_avg = 0.0f32;
    let mut episode_return = 0.0f32;

    for step in 0..total_timesteps {
        let epsilon = epsilon_schedule(&config, step);

        // Use inner model for inference (no autodiff overhead)
        let action = {
            let inner = online.valid();
            epsilon_greedy::<NdArray, _>(&inner, &obs, 2, epsilon, &device, &mut rng)
        };

        let result = env.step(action);
        episode_return += result.reward;

        buffer.extend(std::iter::once(Transition {
            obs: obs.clone(),
            action: action as i32,
            reward: result.reward,
            next_obs: result.observation.clone(),
            done: result.done(),
        }));

        if result.done() {
            recent_returns.push(episode_return);
            episode_return = 0.0;
            obs = env.reset();
        } else {
            obs = result.observation;
        }

        // Train
        if step >= config.learning_starts
            && step % train_frequency == 0
            && buffer.len() >= config.batch_size
        {
            let stats;
            (online, stats) = dqn_update(online, &target, &mut optim, &mut buffer, &config, &device);

            // Hard target update
            if step % target_update_freq == 0 {
                target = polyak_update(target, &online, config.tau);
            }
        }

        // Report
        if recent_returns.len() > 20 {
            let start = recent_returns.len() - 20;
            recent_returns = recent_returns[start..].to_vec();
        }

        if !recent_returns.is_empty() && (step + 1) % 5000 == 0 {
            let avg: f32 = recent_returns.iter().sum::<f32>() / recent_returns.len() as f32;
            best_avg = best_avg.max(avg);
            let eps = epsilon_schedule(&config, step);
            eprintln!(
                "step {:>6}: avg_return={:>6.1} (best={:>6.1}) eps={:.3}",
                step + 1, avg, best_avg, eps
            );
        }

        if !recent_returns.is_empty() {
            let avg: f32 = recent_returns.iter().sum::<f32>() / recent_returns.len() as f32;
            best_avg = best_avg.max(avg);
            if best_avg > 450.0 {
                eprintln!("DQN solved at step {}!", step + 1);
                break;
            }
        }
    }

    assert!(
        best_avg > 200.0,
        "DQN should learn CartPole (best avg return {best_avg:.1}, expected >200)"
    );
}
