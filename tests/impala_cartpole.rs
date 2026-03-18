//! Integration test: train actor-critic with V-trace on CartPole and verify convergence.
//!
//! Uses the local collection path (actor_learner_collect + ac_vtrace_update)
//! with V-trace off-policy correction.

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::optim::AdamConfig;
use burn::prelude::*;

use rand::SeedableRng;

use rl4burn::envs::CartPole;
use rl4burn::{
    ac_vtrace_update, actor_learner_collect, orthogonal_linear, DiscreteAcOutput,
    DiscreteActorCritic, SyncVecEnv,
};

type AutodiffB = Autodiff<NdArray>;

// ---------------------------------------------------------------------------
// Model: same architecture as PPO test
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

impl<B: Backend> DiscreteActorCritic<B> for Agent<B> {
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
// Test
// ---------------------------------------------------------------------------

#[test]
#[ignore] // slow: run with `cargo test -- --ignored`
fn impala_solves_cartpole() {
    let device = NdArrayDevice::Cpu;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(1);

    let n_envs = 8;
    let envs: Vec<CartPole<rand::rngs::SmallRng>> = (0..n_envs)
        .map(|i| CartPole::new(rand::rngs::SmallRng::seed_from_u64(1 + i as u64)))
        .collect();
    let mut vec_env = SyncVecEnv::new(envs);

    let mut model: Agent<AutodiffB> = Agent::new(&device, &mut rng);

    let mut optim = AdamConfig::new().with_epsilon(1e-5).init();

    // Hyperparameters (previously in ImpalaConfig)
    let lr = 5e-4;
    let gamma = 0.99f32;
    let unroll_length = 40;
    let vtrace_clip_rho = 1.0f32;
    let vtrace_clip_c = 1.0f32;
    let vf_coef = 0.5f32;
    let ent_coef = 0.01f32;
    let max_grad_norm = 40.0f32;

    let total_timesteps = 750_000;
    let steps_per_iter = unroll_length * n_envs;
    let n_iterations = total_timesteps / steps_per_iter;

    let mut recent_returns: Vec<f32> = Vec::new();
    let mut best_avg = 0.0f32;
    let mut current_obs = vec_env.reset();
    let mut ep_acc = vec![0.0f32; n_envs];

    for iter in 0..n_iterations {
        let frac = 1.0 - iter as f64 / n_iterations as f64;
        let current_lr = lr * frac;

        let inference_model = model.valid();
        let (trajectories, ep_returns) = actor_learner_collect::<NdArray, _, _>(
            &inference_model,
            &mut vec_env,
            unroll_length,
            &device,
            &mut rng,
            &mut current_obs,
            &mut ep_acc,
        );

        recent_returns.extend_from_slice(&ep_returns);

        let stats;
        (model, stats) = ac_vtrace_update(
            model,
            &mut optim,
            &trajectories,
            gamma,
            vtrace_clip_rho,
            vtrace_clip_c,
            vf_coef,
            ent_coef,
            max_grad_norm,
            current_lr,
            &device,
        );

        if recent_returns.len() > 20 {
            let start = recent_returns.len() - 20;
            recent_returns = recent_returns[start..].to_vec();
        }

        if !recent_returns.is_empty() {
            let avg: f32 = recent_returns.iter().sum::<f32>() / recent_returns.len() as f32;
            best_avg = best_avg.max(avg);

            if (iter + 1) % 50 == 0 {
                eprintln!(
                    "iter {:>4}/{}: avg_return={:>6.1} (best={:>6.1}) ploss={:>8.4} vloss={:>8.1} ent={:.3} rho={:.3}",
                    iter + 1, n_iterations, avg, best_avg,
                    stats.policy_loss, stats.value_loss, stats.entropy, stats.mean_rho
                );
            }

            if best_avg > 450.0 {
                eprintln!("Solved at iter {}!", iter + 1);
                break;
            }
        }
    }

    assert!(
        best_avg > 400.0,
        "IMPALA should solve CartPole (best avg return {best_avg:.1}, expected >400)"
    );
}
