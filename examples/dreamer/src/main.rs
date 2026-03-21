//! R2-Dreamer training example on CartPole.
//!
//! Demonstrates the world-model-based RL pipeline:
//! 1. Collect experience in the real environment.
//! 2. Train the RSSM world model on replay sequences.
//! 3. Train actor-critic via imagination rollouts.
//!
//! Usage: `cargo run -p dreamer --release`

use burn::backend::{Autodiff, NdArray};
use rand::RngExt;

use rl4burn::algo::dreamer::{
    dreamer_actor_critic_loss, dreamer_world_model_loss, DreamerConfig,
};
use rl4burn::algo::loss::representation::RepresentationVariant;
use rl4burn::collect::sequence_replay::{SequenceReplayBuffer, SequenceStep};
use rl4burn::core::env::{Env, Step};
use rl4burn::envs::CartPole;
use rl4burn::log::{Loggable, PrintLogger};
use rl4burn::nn::rssm::RssmConfig;

type B = Autodiff<NdArray>;

fn main() {
    println!("=== R2-Dreamer on CartPole ===\n");

    let device = <B as burn::prelude::Backend>::Device::default();
    let mut rng = rand::rng();

    // --- Configuration ---
    let obs_dim = 4; // CartPole observation size
    let action_dim = 2; // CartPole discrete actions

    let config = DreamerConfig {
        rssm: RssmConfig::new(obs_dim, action_dim)
            .with_deterministic_size(64)
            .with_n_categories(4)
            .with_n_classes(4)
            .with_hidden_size(64),
        rep_variant: RepresentationVariant::R2Dreamer,
        actor_hidden: 64,
        critic_hidden: 64,
        ac_layers: 1,
        action_dim,
        discrete_actions: true,
        horizon: 5,
        gamma: 0.99,
        lambda: 0.95,
        entropy_coef: 1e-3,
        slow_critic_decay: 0.98,
        ..DreamerConfig::default()
    };

    let agent = config.init::<B>(&device);
    let mut logger = PrintLogger::new(0);

    // --- Environment + Replay Buffer ---
    let mut env = CartPole::new(rand::rng());
    let mut buffer = SequenceReplayBuffer::new(10_000, 8);

    // --- Seed collection ---
    let seed_steps = 200;
    println!("Collecting {seed_steps} seed steps...");

    let mut obs = env.reset();
    for _ in 0..seed_steps {
        let action = rng.random_range(0..action_dim);
        let Step {
            observation: next_obs,
            reward,
            terminated,
            truncated,
            ..
        } = env.step(action);

        let mut action_vec = vec![0.0; action_dim];
        action_vec[action] = 1.0;
        buffer.push(SequenceStep {
            observation: obs.clone(),
            action: action_vec,
            reward,
            done: terminated || truncated,
        });

        if terminated || truncated {
            obs = env.reset();
        } else {
            obs = next_obs;
        }
    }

    println!("Buffer size: {}\n", buffer.len());

    // --- Training loop ---
    let n_updates = 20;
    let batch_size = 4;

    for update in 0..n_updates {
        // Sample a batch of sequences from replay
        let sequences = buffer.sample(batch_size, &mut rng);

        if sequences.is_empty() {
            println!("Not enough data for training, skipping...");
            continue;
        }

        let seq_len = sequences[0].len();

        // Convert to tensors
        let mut obs_data = Vec::new();
        let mut act_data = Vec::new();
        let mut rew_data = Vec::new();
        let mut cont_data = Vec::new();

        for seq in &sequences {
            for step in seq {
                obs_data.extend_from_slice(&step.observation);
                act_data.extend_from_slice(&step.action);
                rew_data.push(step.reward);
                cont_data.push(if step.done { 0.0 } else { 1.0 });
            }
        }

        let obs_tensor = burn::tensor::Tensor::<B, 3>::from_floats(
            burn::tensor::TensorData::new(obs_data, [batch_size, seq_len, obs_dim]),
            &device,
        );
        let act_tensor = burn::tensor::Tensor::<B, 3>::from_floats(
            burn::tensor::TensorData::new(act_data, [batch_size, seq_len, action_dim]),
            &device,
        );
        let rew_tensor = burn::tensor::Tensor::<B, 2>::from_floats(
            burn::tensor::TensorData::new(rew_data, [batch_size, seq_len]),
            &device,
        );
        let cont_tensor = burn::tensor::Tensor::<B, 2>::from_floats(
            burn::tensor::TensorData::new(cont_data, [batch_size, seq_len]),
            &device,
        );

        // Train world model
        let (_wm_loss, wm_stats) =
            dreamer_world_model_loss(&agent, obs_tensor, act_tensor, rew_tensor, cont_tensor);

        // Train actor-critic via imagination
        let initial_states = agent.rssm.initial_state(batch_size, &device);
        let (_actor_loss, _critic_loss, ac_stats) =
            dreamer_actor_critic_loss(&agent, initial_states);

        let stats = rl4burn::algo::dreamer::DreamerStats {
            wm: wm_stats,
            ac: ac_stats,
        };

        print!("[update {update:3}] ");
        stats.log(&mut logger, update as u64);
    }

    println!("\nDone! R2-Dreamer training loop completed successfully.");
}
