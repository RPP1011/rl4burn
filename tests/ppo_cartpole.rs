//! Integration test: train PPO on CartPole and verify convergence.
//!
//! Carbon copy of CleanRL's ppo.py: orthogonal init, tanh, separate
//! actor/critic, Adam(eps=1e-5), gradient clipping, LR annealing.

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::grad_clipping::GradientClippingConfig;
use burn::module::{AutodiffModule, Module};
use burn::optim::AdamConfig;
use burn::prelude::*;

use rand::SeedableRng;

use rl4burn::envs::CartPole;
use rl4burn::init::orthogonal_linear;
use rl4burn::policy::{DiscreteAcOutput, DiscreteActorCritic};
use rl4burn::ppo::{ppo_collect, ppo_update, PpoConfig};
use rl4burn::vec_env::SyncVecEnv;

type AutodiffB = Autodiff<NdArray>;

// ---------------------------------------------------------------------------
// Model: exact CleanRL Agent architecture
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct Agent<B: Backend> {
    // Actor: obs -> 64 (tanh) -> 64 (tanh) -> n_actions
    actor_fc1: burn::nn::Linear<B>,
    actor_fc2: burn::nn::Linear<B>,
    actor_out: burn::nn::Linear<B>,
    // Critic: obs -> 64 (tanh) -> 64 (tanh) -> 1
    critic_fc1: burn::nn::Linear<B>,
    critic_fc2: burn::nn::Linear<B>,
    critic_out: burn::nn::Linear<B>,
}

impl<B: Backend> Agent<B> {
    fn new(device: &B::Device, rng: &mut impl rand::Rng) -> Self {
        let sqrt2 = std::f32::consts::SQRT_2;
        Self {
            // Hidden layers: orthogonal with gain=sqrt(2) (CleanRL default)
            actor_fc1: orthogonal_linear(4, 64, sqrt2, device, rng),
            actor_fc2: orthogonal_linear(64, 64, sqrt2, device, rng),
            // Actor output: orthogonal with gain=0.01 (near-uniform initial policy)
            actor_out: orthogonal_linear(64, 2, 0.01, device, rng),
            // Critic hidden: orthogonal with gain=sqrt(2)
            critic_fc1: orthogonal_linear(4, 64, sqrt2, device, rng),
            critic_fc2: orthogonal_linear(64, 64, sqrt2, device, rng),
            // Critic output: orthogonal with gain=1.0
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
fn ppo_solves_cartpole() {
    let device = NdArrayDevice::Cpu;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(1); // CleanRL default seed

    // CleanRL: 4 envs, SyncVectorEnv
    let n_envs = 4;
    let envs: Vec<CartPole<rand::rngs::SmallRng>> = (0..n_envs)
        .map(|i| CartPole::new(rand::rngs::SmallRng::seed_from_u64(1 + i as u64)))
        .collect();
    let mut vec_env = SyncVecEnv::new(envs);

    // Model with orthogonal init (matching CleanRL's layer_init)
    let model: Agent<AutodiffB> = Agent::new(&device, &mut rng);

    // Adam(lr=2.5e-4, eps=1e-5) with gradient clipping at 0.5
    let mut optim = AdamConfig::new()
        .with_epsilon(1e-5)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(0.5)))
        .init();

    // CleanRL defaults
    let config = PpoConfig {
        lr: 2.5e-4,
        gamma: 0.99,
        gae_lambda: 0.95,
        clip_eps: 0.2,
        vf_coef: 0.5,
        ent_coef: 0.01,
        update_epochs: 4,
        minibatch_size: 128, // batch_size(512) / num_minibatches(4)
        n_steps: 128,
        clip_vloss: true,
    };

    let mut model = model;
    let total_timesteps = 500_000;
    let steps_per_iter = config.n_steps * n_envs; // 512
    let n_iterations = total_timesteps / steps_per_iter;

    let mut recent_returns: Vec<f32> = Vec::new();
    let mut best_avg = 0.0f32;

    for iter in 0..n_iterations {
        // Linear LR annealing: frac = 1.0 - (iteration - 1) / num_iterations
        // CleanRL iteration is 1-indexed, ours is 0-indexed
        let frac = 1.0 - iter as f64 / n_iterations as f64;
        let current_lr = config.lr * frac;

        let inference_model = model.valid();
        let rollout = ppo_collect::<NdArray, _, _>(
            &inference_model,
            &mut vec_env,
            &config,
            &device,
            &mut rng,
        );

        // Track completed episode returns
        let mut ep_return = vec![0.0f32; n_envs];
        for step in 0..config.n_steps {
            for env_idx in 0..n_envs {
                let idx = step * n_envs + env_idx;
                ep_return[env_idx] += rollout.rewards[idx];
                if rollout.dones[idx] {
                    recent_returns.push(ep_return[env_idx]);
                    ep_return[env_idx] = 0.0;
                }
            }
        }

        // Debug: print actor weight norm before/after update for first few iters
        let actor_norm_before: f32 = if iter < 3 {
            model.actor_out.weight.val().powf_scalar(2.0).sum().into_scalar()
        } else { 0.0 };

        let stats;
        (model, stats) =
            ppo_update(model, &mut optim, &rollout, &config, current_lr, &device, &mut rng);

        if iter < 3 {
            let actor_norm_after: f32 = model.actor_out.weight.val().into_data().to_vec::<f32>().unwrap()
                .iter().map(|x| x * x).sum::<f32>();
            let critic_norm_after: f32 = model.critic_out.weight.val().into_data().to_vec::<f32>().unwrap()
                .iter().map(|x| x * x).sum::<f32>();
            let critic_fc1_after: f32 = model.critic_fc1.weight.val().into_data().to_vec::<f32>().unwrap()
                .iter().map(|x| x * x).sum::<f32>();
            let actor_fc1_after: f32 = model.actor_fc1.weight.val().into_data().to_vec::<f32>().unwrap()
                .iter().map(|x| x * x).sum::<f32>();
            eprintln!(
                "iter {}: actor_out_L2={:.8} critic_out_L2={:.6} actor_fc1_L2={:.4} critic_fc1_L2={:.4}",
                iter, actor_norm_after, critic_norm_after, actor_fc1_after, critic_fc1_after
            );
        }

        if recent_returns.len() > 20 {
            let start = recent_returns.len() - 20;
            recent_returns = recent_returns[start..].to_vec();
        }

        if !recent_returns.is_empty() {
            let avg: f32 = recent_returns.iter().sum::<f32>() / recent_returns.len() as f32;
            best_avg = best_avg.max(avg);

            if (iter + 1) % 25 == 0 {
                eprintln!(
                    "iter {:>4}/{}: avg_return={:>6.1} (best={:>6.1}) ploss={:>8.4} vloss={:>8.1} ent={:.3} kl={:.4}",
                    iter + 1, n_iterations, avg, best_avg, stats.policy_loss, stats.value_loss, stats.entropy, stats.approx_kl
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
        "PPO should solve CartPole (best avg return {best_avg:.1}, expected >400)"
    );
}
