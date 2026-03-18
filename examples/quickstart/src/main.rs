//! Your first game AI in 30 lines.
//!
//! Minimal PPO on CartPole — `cargo run --release` and watch it learn.

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::nn::{Linear, LinearConfig};
use burn::optim::AdamConfig;
use burn::prelude::*;
use rand::SeedableRng;

use rl4burn::envs::CartPole;
use rl4burn::{
    ppo_collect, ppo_update, DiscreteAcOutput, DiscreteActorCritic, PpoConfig, SyncVecEnv,
};

// A small actor-critic network.
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
            actor1: LinearConfig::new(4, 64).init(device),
            actor2: LinearConfig::new(64, 64).init(device),
            actor_out: LinearConfig::new(64, 2).init(device),
            critic1: LinearConfig::new(4, 64).init(device),
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

type B = Autodiff<NdArray>;

fn main() {
    let device = NdArrayDevice::Cpu;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    // 4 parallel environments
    let envs: Vec<_> = (0..4)
        .map(|i| CartPole::new(rand::rngs::SmallRng::seed_from_u64(1000 + i)))
        .collect();
    let mut vec_env = SyncVecEnv::new(envs);

    let mut model: Model<B> = Model::new(&device);
    let mut optim = AdamConfig::new().with_epsilon(1e-5).init();
    let config = PpoConfig::new();

    let total_steps = 100_000;
    let steps_per_iter = config.n_steps * 4;
    let n_iters = total_steps / steps_per_iter;

    let mut current_obs = vec_env.reset();
    let mut ep_acc = vec![0.0f32; 4];
    let mut recent: Vec<f32> = Vec::new();

    eprintln!("Training PPO on CartPole ({total_steps} steps, 4 envs)");

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

        if !recent.is_empty() && (iter + 1) % 5 == 0 {
            let avg: f32 = recent.iter().sum::<f32>() / recent.len() as f32;
            eprintln!(
                "step {:>6} | avg return {avg:>6.1} | policy_loss {:.4} | entropy {:.4}",
                (iter + 1) * steps_per_iter,
                stats.policy_loss,
                stats.entropy,
            );
        }
    }

    if !recent.is_empty() {
        let avg: f32 = recent.iter().sum::<f32>() / recent.len() as f32;
        eprintln!("Final avg return (last 20 eps): {avg:.1}");
    }
}
