//! Configuration-driven PPO training on CartPole.
//!
//! Loads hyperparameters from a TOML file. Pass a custom config path as the
//! first CLI argument, or it defaults to `configs/default.toml`.
//!
//! ```text
//! cargo run --release
//! cargo run --release -- configs/fast.toml
//! ```

use std::fs;

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::nn::{Linear, LinearConfig};
use burn::optim::AdamConfig;
use burn::prelude::*;
use rand::SeedableRng;
use serde::Deserialize;

use rl4burn::envs::CartPole;
use rl4burn::log::{CompositeLogger, PrintLogger};
use rl4burn::{
    ppo_collect, ppo_update, DiscreteAcOutput, DiscreteActorCritic, Loggable, Logger, PpoConfig,
    SyncVecEnv,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct Config {
    training: TrainingConfig,
    ppo: PpoToml,
    model: ModelConfig,
}

#[derive(Deserialize)]
struct TrainingConfig {
    total_timesteps: usize,
    n_envs: usize,
    seed: u64,
}

#[derive(Deserialize)]
struct PpoToml {
    lr: f64,
    gamma: f32,
    gae_lambda: f32,
    clip_eps: f32,
    vf_coef: f32,
    ent_coef: f32,
    update_epochs: usize,
    minibatch_size: usize,
    n_steps: usize,
    clip_vloss: bool,
    max_grad_norm: f32,
}

impl PpoToml {
    fn into_ppo_config(self) -> PpoConfig {
        PpoConfig::new()
            .with_lr(self.lr)
            .with_gamma(self.gamma)
            .with_gae_lambda(self.gae_lambda)
            .with_clip_eps(self.clip_eps)
            .with_vf_coef(self.vf_coef)
            .with_ent_coef(self.ent_coef)
            .with_update_epochs(self.update_epochs)
            .with_minibatch_size(self.minibatch_size)
            .with_n_steps(self.n_steps)
            .with_clip_vloss(self.clip_vloss)
            .with_max_grad_norm(self.max_grad_norm)
    }
}

#[derive(Deserialize)]
struct ModelConfig {
    hidden_size: usize,
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct ActorCritic<B: Backend> {
    actor1: Linear<B>,
    actor2: Linear<B>,
    actor_out: Linear<B>,
    critic1: Linear<B>,
    critic2: Linear<B>,
    critic_out: Linear<B>,
}

impl<B: Backend> ActorCritic<B> {
    fn new(hidden: usize, device: &B::Device) -> Self {
        Self {
            actor1: LinearConfig::new(4, hidden).init(device),
            actor2: LinearConfig::new(hidden, hidden).init(device),
            actor_out: LinearConfig::new(hidden, 2).init(device),
            critic1: LinearConfig::new(4, hidden).init(device),
            critic2: LinearConfig::new(hidden, hidden).init(device),
            critic_out: LinearConfig::new(hidden, 1).init(device),
        }
    }
}

impl<B: Backend> DiscreteActorCritic<B> for ActorCritic<B> {
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
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "configs/default.toml".to_string());

    let toml_str = fs::read_to_string(&config_path)
        .unwrap_or_else(|e| panic!("Failed to read config {config_path}: {e}"));
    let cfg: Config =
        toml::from_str(&toml_str).unwrap_or_else(|e| panic!("Failed to parse config: {e}"));

    let config = cfg.ppo.into_ppo_config();
    let n_envs = cfg.training.n_envs;
    let total_timesteps = cfg.training.total_timesteps;
    let seed = cfg.training.seed;

    let device = NdArrayDevice::Cpu;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);

    let envs: Vec<_> = (0..n_envs)
        .map(|i| CartPole::new(rand::rngs::SmallRng::seed_from_u64(1000 + i as u64)))
        .collect();
    let mut vec_env = SyncVecEnv::new(envs);

    let mut model: ActorCritic<B> = ActorCritic::new(cfg.model.hidden_size, &device);
    let mut optim = AdamConfig::new().with_epsilon(1e-5).init();

    let steps_per_iter = config.n_steps * n_envs;
    let n_iterations = total_timesteps / steps_per_iter;

    let mut current_obs = vec_env.reset();
    let mut ep_acc = vec![0.0f32; n_envs];
    let mut recent: Vec<f32> = Vec::new();

    let loggers: Vec<Box<dyn Logger>> = vec![Box::new(PrintLogger::new(0))];
    let mut logger = CompositeLogger::new(loggers);

    eprintln!("Config: {config_path}");
    eprintln!(
        "Training PPO on CartPole ({total_timesteps} timesteps, {n_envs} envs, hidden={})",
        cfg.model.hidden_size
    );
    eprintln!("{:-<80}", "");

    for iter in 0..n_iterations {
        let frac = 1.0 - iter as f64 / n_iterations as f64;
        let lr = config.lr * frac;

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

        let timestep = ((iter + 1) * steps_per_iter) as u64;
        if !recent.is_empty() && (iter + 1) % 5 == 0 {
            let avg: f32 = recent.iter().sum::<f32>() / recent.len() as f32;
            logger.log_scalar("rollout/avg_return", avg as f64, timestep);
            stats.log(&mut logger, timestep);
        }
    }
    logger.flush();

    if !recent.is_empty() {
        let avg: f32 = recent.iter().sum::<f32>() / recent.len() as f32;
        eprintln!("{:-<80}", "");
        eprintln!("Final avg return (last 20 eps): {avg:.1}");
    }
}
