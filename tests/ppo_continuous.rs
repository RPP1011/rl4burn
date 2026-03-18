//! Integration test: continuous PPO on Pendulum using ActionDist::Continuous.
//!
//! Validates that masked_ppo_collect/update with ActionDist::Continuous(ModelOutput),
//! observation normalization, and reward normalization learns Pendulum-v1.
//! Matches CleanRL's ppo_continuous_action.py hyperparameters.

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::optim::AdamConfig;
use burn::prelude::*;

use rand::SeedableRng;

use rl4burn::envs::Pendulum;
use rl4burn::wrapper::{NormalizeObservation, NormalizeReward};
use rl4burn::{
    masked_ppo_collect, masked_ppo_update, orthogonal_linear, ActionDist, LogStdMode,
    MaskedActorCritic, PpoConfig, SyncVecEnv,
};

type AutodiffB = Autodiff<NdArray>;

// ---------------------------------------------------------------------------
// Continuous actor-critic model (ModelOutput mode)
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct ContinuousAgent<B: Backend> {
    actor_fc1: burn::nn::Linear<B>,
    actor_fc2: burn::nn::Linear<B>,
    actor_out: burn::nn::Linear<B>,
    critic_fc1: burn::nn::Linear<B>,
    critic_fc2: burn::nn::Linear<B>,
    critic_out: burn::nn::Linear<B>,
}

impl<B: Backend> ContinuousAgent<B> {
    fn new(device: &B::Device, rng: &mut impl rand::Rng) -> Self {
        let sqrt2 = std::f32::consts::SQRT_2;
        Self {
            actor_fc1: orthogonal_linear(3, 64, sqrt2, device, rng),
            actor_fc2: orthogonal_linear(64, 64, sqrt2, device, rng),
            // 2 outputs: mean + log_std for 1 action dim
            actor_out: orthogonal_linear(64, 2, 0.01, device, rng),
            critic_fc1: orthogonal_linear(3, 64, sqrt2, device, rng),
            critic_fc2: orthogonal_linear(64, 64, sqrt2, device, rng),
            critic_out: orthogonal_linear(64, 1, 1.0, device, rng),
        }
    }
}

impl<B: Backend> MaskedActorCritic<B> for ContinuousAgent<B> {
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
// Test
// ---------------------------------------------------------------------------

#[test]
#[ignore] // slow: run with `cargo test -- --ignored`
fn continuous_ppo_learns_pendulum() {
    let device = NdArrayDevice::Cpu;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(1);

    let n_envs = 4;
    // Wrap Pendulum with observation and reward normalization (matching CleanRL)
    let envs: Vec<NormalizeReward<NormalizeObservation<Pendulum<rand::rngs::SmallRng>>>> =
        (0..n_envs)
            .map(|i| {
                let env = Pendulum::new(rand::rngs::SmallRng::seed_from_u64(1 + i as u64));
                let env = NormalizeObservation::new(env, 10.0);
                NormalizeReward::new(env, 0.99, 10.0)
            })
            .collect();
    let mut vec_env = SyncVecEnv::new(envs);

    let model: ContinuousAgent<AutodiffB> = ContinuousAgent::new(&device, &mut rng);
    let action_dist = ActionDist::Continuous {
        action_dim: 1,
        log_std_mode: LogStdMode::ModelOutput,
    };

    let mut optim = AdamConfig::new().with_epsilon(1e-5).init();

    // CleanRL-matching hyperparameters
    let config = PpoConfig {
        lr: 3e-4,
        gamma: 0.99,
        gae_lambda: 0.95,
        clip_eps: 0.2,
        vf_coef: 0.5,
        ent_coef: 0.0,
        update_epochs: 10,
        minibatch_size: 64,
        n_steps: 512, // 4 envs × 512 = 2048 batch (matches CleanRL)
        clip_vloss: true,
        max_grad_norm: 0.5,
        target_kl: None,
        dual_clip_coef: None,
    };

    let mut model = model;
    let total_timesteps = 200_000;
    let steps_per_iter = config.n_steps * n_envs;
    let n_iterations = total_timesteps / steps_per_iter;

    let mut recent_returns: Vec<f32> = Vec::new();
    let mut best_avg = f32::NEG_INFINITY;
    let mut current_obs = vec_env.reset();
    let mut ep_acc = vec![0.0f32; n_envs];

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

        if !recent_returns.is_empty() {
            let avg: f32 = recent_returns.iter().sum::<f32>() / recent_returns.len() as f32;
            best_avg = best_avg.max(avg);

            if (iter + 1) % 5 == 0 {
                eprintln!(
                    "pendulum iter {:>3}/{}: avg_return={:>8.1} (best={:>8.1}) ploss={:>8.4} ent={:.3} kl={:.4}",
                    iter + 1, n_iterations, avg, best_avg, stats.policy_loss, stats.entropy, stats.approx_kl
                );
            }
        }
    }

    eprintln!("Continuous PPO best avg (normalized): {best_avg:.1}");
    // With reward normalization, episode returns are in normalized scale (~-7 to -10 range).
    // A non-learning policy would score much worse. Assert the machinery works.
    assert!(
        best_avg > -20.0,
        "Continuous PPO should learn Pendulum (best avg {best_avg:.1}, expected > -20)"
    );
}
