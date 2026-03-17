//! Train continuous PPO on Pendulum-v1 using rl4burn.
//!
//! Demonstrates:
//! - Continuous action spaces with `ActionDist::Continuous`
//! - Implementing `MaskedActorCritic` for a continuous model
//! - The `masked_ppo_collect` / `masked_ppo_update` training loop
//! - Logging via the `Logger` trait
//!
//! Run: `cargo run --example ppo_pendulum --features ndarray`
//!
//! With TensorBoard:
//!   `cargo run --example ppo_pendulum --features "ndarray,tensorboard"`
//!   `tensorboard --logdir runs/`

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::nn::{Linear, LinearConfig};
use burn::optim::AdamConfig;
use burn::prelude::*;

use rand::SeedableRng;

use rl4burn::envs::Pendulum;
use rl4burn::log::{CompositeLogger, PrintLogger};
use rl4burn::wrapper::{NormalizeObservation, NormalizeReward};
use rl4burn::{
    masked_ppo_collect, masked_ppo_update, ActionDist, Loggable, LogStdMode, Logger,
    MaskedActorCritic, PpoConfig, SyncVecEnv,
};

#[cfg(feature = "tensorboard")]
use rl4burn::TensorBoardLogger;

#[cfg(feature = "json-log")]
use rl4burn::JsonLogger;

// ---------------------------------------------------------------------------
// Model: continuous actor-critic with ModelOutput log_std
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct ContinuousAC<B: Backend> {
    actor_fc1: Linear<B>,
    actor_fc2: Linear<B>,
    // Outputs [mean, log_std] for 1-d action
    actor_out: Linear<B>,
    critic_fc1: Linear<B>,
    critic_fc2: Linear<B>,
    critic_out: Linear<B>,
}

impl<B: Backend> ContinuousAC<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            actor_fc1: LinearConfig::new(3, 64).init(device),
            actor_fc2: LinearConfig::new(64, 64).init(device),
            actor_out: LinearConfig::new(64, 2).init(device), // mean + log_std
            critic_fc1: LinearConfig::new(3, 64).init(device),
            critic_fc2: LinearConfig::new(64, 64).init(device),
            critic_out: LinearConfig::new(64, 1).init(device),
        }
    }
}

impl<B: Backend> MaskedActorCritic<B> for ContinuousAC<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let a = self.actor_fc1.forward(obs.clone()).tanh();
        let a = self.actor_fc2.forward(a).tanh();
        let logits = self.actor_out.forward(a);

        let c = self.critic_fc1.forward(obs).tanh();
        let c = self.critic_fc2.forward(c).tanh();
        let values = self.critic_out.forward(c).squeeze_dim::<1>(1);

        (logits, values)
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
    // Wrap with observation and reward normalization (matching CleanRL)
    let envs: Vec<NormalizeReward<NormalizeObservation<Pendulum<rand::rngs::SmallRng>>>> =
        (0..n_envs)
            .map(|i| {
                let env = Pendulum::new(rand::rngs::SmallRng::seed_from_u64(1000 + i as u64));
                let env = NormalizeObservation::new(env, 10.0);
                NormalizeReward::new(env, 0.99, 10.0)
            })
            .collect();
    let mut vec_env = SyncVecEnv::new(envs);

    let mut model: ContinuousAC<AutodiffB> = ContinuousAC::new(&device);
    let action_dist = ActionDist::Continuous {
        action_dim: 1,
        log_std_mode: LogStdMode::ModelOutput,
    };

    let mut optim = AdamConfig::new().with_epsilon(1e-5).init();

    let config = PpoConfig {
        lr: 3e-4,
        gamma: 0.99,
        gae_lambda: 0.95,
        clip_eps: 0.2,
        vf_coef: 0.5,
        ent_coef: 0.0,
        update_epochs: 10,
        minibatch_size: 64,
        n_steps: 256,
        clip_vloss: true,
        max_grad_norm: 0.5,
        target_kl: None,
    };

    let total_timesteps = 1_000_000;
    let steps_per_iter = config.n_steps * n_envs;
    let n_iterations = total_timesteps / steps_per_iter;
    let mut recent_returns: Vec<f32> = Vec::new();
    let mut current_obs = vec_env.reset();
    let mut ep_acc = vec![0.0f32; n_envs];

    #[allow(unused_mut)]
    let mut loggers: Vec<Box<dyn Logger>> = vec![Box::new(PrintLogger::new(0))];

    #[cfg(feature = "tensorboard")]
    loggers.push(Box::new(
        TensorBoardLogger::new("runs/ppo_pendulum").expect("failed to create TensorBoardLogger"),
    ));

    #[cfg(feature = "json-log")]
    loggers.push(Box::new(JsonLogger::new(Box::new(std::io::stderr()))));

    let mut logger = CompositeLogger::new(loggers);

    eprintln!("Training continuous PPO on Pendulum-v1 ({total_timesteps} timesteps, {n_envs} envs)");
    eprintln!("{:-<80}", "");

    for iter in 0..n_iterations {
        let frac = 1.0 - iter as f64 / n_iterations as f64;
        let current_lr = config.lr * frac;

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

        if recent_returns.len() > 30 {
            let start = recent_returns.len() - 30;
            recent_returns = recent_returns[start..].to_vec();
        }

        let timestep = ((iter + 1) * steps_per_iter) as u64;

        if !recent_returns.is_empty() && (iter + 1) % 5 == 0 {
            let avg: f32 = recent_returns.iter().sum::<f32>() / recent_returns.len() as f32;
            logger.log_scalar("rollout/avg_return", avg as f64, timestep);
            stats.log(&mut logger, timestep);
        }
    }
    logger.flush();

    if !recent_returns.is_empty() {
        let avg: f32 = recent_returns.iter().sum::<f32>() / recent_returns.len() as f32;
        eprintln!("{:-<80}", "");
        eprintln!("Final average return (last 30 episodes): {avg:.1}");
    }
}
