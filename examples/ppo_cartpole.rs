//! Train PPO on CartPole-v1 using rl4burn.
//!
//! Demonstrates:
//! - Defining a separate actor/critic model with Burn's `#[derive(Module)]`
//! - Implementing `DiscreteActorCritic` for the model
//! - Using `SyncVecEnv` for parallel environments
//! - The `ppo_collect` / `ppo_update` training loop with LR annealing
//!
//! Run: `cargo run --example ppo_cartpole --features ndarray`

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig};
use burn::grad_clipping::GradientClippingConfig;
use burn::optim::AdamConfig;
use burn::prelude::*;

use rand::SeedableRng;

use rl4burn::envs::CartPole;
use rl4burn::policy::{DiscreteAcOutput, DiscreteActorCritic};
use rl4burn::ppo::{ppo_collect, ppo_update, PpoConfig};
use rl4burn::vec_env::SyncVecEnv;

// ---------------------------------------------------------------------------
// Model: separate actor/critic MLPs with tanh (matches CleanRL)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct ActorCritic<B: Backend> {
    actor_fc1: Linear<B>,
    actor_fc2: Linear<B>,
    actor_out: Linear<B>,
    critic_fc1: Linear<B>,
    critic_fc2: Linear<B>,
    critic_out: Linear<B>,
}

impl<B: Backend> ActorCritic<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            actor_fc1: LinearConfig::new(4, 64).init(device),
            actor_fc2: LinearConfig::new(64, 64).init(device),
            actor_out: LinearConfig::new(64, 2).init(device),
            critic_fc1: LinearConfig::new(4, 64).init(device),
            critic_fc2: LinearConfig::new(64, 64).init(device),
            critic_out: LinearConfig::new(64, 1).init(device),
        }
    }
}

impl<B: Backend> DiscreteActorCritic<B> for ActorCritic<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> DiscreteAcOutput<B> {
        let a = self.actor_fc1.forward(obs.clone()).tanh();
        let a = self.actor_fc2.forward(a).tanh();
        let logits = self.actor_out.forward(a);

        let c = self.critic_fc1.forward(obs).tanh();
        let c = self.critic_fc2.forward(c).tanh();
        let values = self.critic_out.forward(c).squeeze_dim::<1>(1);

        DiscreteAcOutput { logits, values }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

type AutodiffB = Autodiff<NdArray>;

fn main() {
    let device = NdArrayDevice::Cpu;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    let n_envs = 4;
    let envs: Vec<CartPole<rand::rngs::SmallRng>> = (0..n_envs)
        .map(|i| CartPole::new(rand::rngs::SmallRng::seed_from_u64(1000 + i as u64)))
        .collect();
    let mut vec_env = SyncVecEnv::new(envs);

    let mut model: ActorCritic<AutodiffB> = ActorCritic::new(&device);
    let mut optim = AdamConfig::new()
        .with_epsilon(1e-5)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(0.5)))
        .init();

    let config = PpoConfig {
        lr: 2.5e-4,
        gamma: 0.99,
        gae_lambda: 0.95,
        clip_eps: 0.2,
        vf_coef: 0.5,
        ent_coef: 0.01,
        update_epochs: 4,
        minibatch_size: 128,
        n_steps: 128,
        clip_vloss: true,
        max_grad_norm: 0.5,
    };

    let total_timesteps = 500_000;
    let steps_per_iter = config.n_steps * n_envs;
    let n_iterations = total_timesteps / steps_per_iter;
    let mut recent_returns: Vec<f32> = Vec::new();
    let mut ep_acc = vec![0.0f32; n_envs];

    println!("Training PPO on CartPole-v1 ({total_timesteps} timesteps, {n_envs} envs)");
    println!("{:-<80}", "");

    for iter in 0..n_iterations {
        let frac = 1.0 - iter as f64 / n_iterations as f64;
        let current_lr = config.lr * frac;

        let inference_model = model.valid();
        let rollout = ppo_collect::<NdArray, _, _>(
            &inference_model,
            &mut vec_env,
            &config,
            &device,
            &mut rng,
            &mut ep_acc,
        );
        recent_returns.extend_from_slice(&rollout.episode_returns);

        let stats;
        (model, stats) =
            ppo_update(model, &mut optim, &rollout, &config, current_lr, &device, &mut rng);

        if recent_returns.len() > 20 {
            let start = recent_returns.len() - 20;
            recent_returns = recent_returns[start..].to_vec();
        }

        if !recent_returns.is_empty() && (iter + 1) % 5 == 0 {
            let avg: f32 = recent_returns.iter().sum::<f32>() / recent_returns.len() as f32;
            let timestep = (iter + 1) * steps_per_iter;
            println!(
                "step {:>6} | avg_return {:>6.1} | ploss {:>8.4} | vloss {:>8.2} | entropy {:>5.3} | kl {:>6.4}",
                timestep, avg, stats.policy_loss, stats.value_loss, stats.entropy, stats.approx_kl
            );
        }
    }

    if !recent_returns.is_empty() {
        let avg: f32 = recent_returns.iter().sum::<f32>() / recent_returns.len() as f32;
        println!("{:-<80}", "");
        println!("Final average return (last 20 episodes): {avg:.1}");
        if avg > 475.0 {
            println!("CartPole-v1 solved!");
        }
    }
}
