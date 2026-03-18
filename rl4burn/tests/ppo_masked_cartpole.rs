//! Integration test: masked PPO on CartPole via DiscreteEnvAdapter.
//!
//! Validates that the generalized masked_ppo_collect/update works correctly
//! with a simple discrete environment using the DiscreteEnvAdapter and
//! ActionDist::Discrete. Should converge similarly to the original PPO test.

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::optim::AdamConfig;
use burn::prelude::*;

use rand::SeedableRng;

use rl4burn::envs::CartPole;
use rl4burn::{
    masked_ppo_collect, masked_ppo_update, orthogonal_linear, ActionDist,
    DiscreteEnvAdapter, MaskedActorCritic, PpoConfig, SyncVecEnv,
};

type AutodiffB = Autodiff<NdArray>;

// ---------------------------------------------------------------------------
// Model: same as the standard PPO test
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct Agent<B: Backend> {
    actor_fc1: burn::nn::Linear<B>,
    actor_fc2: burn::nn::Linear<B>,
    actor_out: burn::nn::Linear<B>,
    critic_fc1: burn::nn::Linear<B>,
    critic_fc2: burn::nn::Linear<B>,
    critic_out: burn::nn::Linear<B>,
}

impl<B: Backend> Agent<B> {
    fn new(device: &B::Device, rng: &mut impl rand::Rng) -> Self {
        let sqrt2 = std::f32::consts::SQRT_2;
        Self {
            actor_fc1: orthogonal_linear(4, 64, sqrt2, device, rng),
            actor_fc2: orthogonal_linear(64, 64, sqrt2, device, rng),
            actor_out: orthogonal_linear(64, 2, 0.01, device, rng),
            critic_fc1: orthogonal_linear(4, 64, sqrt2, device, rng),
            critic_fc2: orthogonal_linear(64, 64, sqrt2, device, rng),
            critic_out: orthogonal_linear(64, 1, 1.0, device, rng),
        }
    }
}

impl<B: Backend> MaskedActorCritic<B> for Agent<B> {
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
fn masked_ppo_solves_cartpole() {
    let device = NdArrayDevice::Cpu;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(1);

    let n_envs = 4;
    // Wrap CartPole with DiscreteEnvAdapter for Vec<f32> actions
    let envs: Vec<DiscreteEnvAdapter<CartPole<rand::rngs::SmallRng>>> = (0..n_envs)
        .map(|i| DiscreteEnvAdapter(CartPole::new(rand::rngs::SmallRng::seed_from_u64(1 + i as u64))))
        .collect();
    let mut vec_env = SyncVecEnv::new(envs);

    let model: Agent<AutodiffB> = Agent::new(&device, &mut rng);
    let action_dist = ActionDist::Discrete(2);

    let mut optim = AdamConfig::new().with_epsilon(1e-5).init();

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
        target_kl: None,
        dual_clip_coef: None,
    };

    let mut model = model;
    let total_timesteps = 500_000;
    let steps_per_iter = config.n_steps * n_envs;
    let n_iterations = total_timesteps / steps_per_iter;

    let mut recent_returns: Vec<f32> = Vec::new();
    let mut best_avg = 0.0f32;
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

        if recent_returns.len() > 20 {
            let start = recent_returns.len() - 20;
            recent_returns = recent_returns[start..].to_vec();
        }

        if !recent_returns.is_empty() {
            let avg: f32 = recent_returns.iter().sum::<f32>() / recent_returns.len() as f32;
            best_avg = best_avg.max(avg);

            if (iter + 1) % 25 == 0 {
                eprintln!(
                    "masked iter {:>4}/{}: avg_return={:>6.1} (best={:>6.1}) ploss={:>8.4} vloss={:>8.1} ent={:.3} kl={:.4}",
                    iter + 1, n_iterations, avg, best_avg, stats.policy_loss, stats.value_loss, stats.entropy, stats.approx_kl
                );
            }

            if best_avg > 450.0 {
                eprintln!("Masked PPO solved at iter {}!", iter + 1);
                break;
            }
        }
    }

    assert!(
        best_avg > 400.0,
        "Masked PPO should solve CartPole (best avg return {best_avg:.1}, expected >400)"
    );
}
