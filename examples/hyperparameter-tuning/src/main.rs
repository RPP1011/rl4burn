//! Hyperparameter tuning for PPO.
//!
//! Demonstrates grid search over PPO hyperparameters, training short runs
//! and comparing results. This is the simplest tuning approach and works
//! well for understanding the sensitivity of each hyperparameter.
//!
//! # Known-good hyperparameters by environment
//!
//! | Environment   | lr      | ent_coef | gamma | clip_eps | n_steps | n_envs |
//! |---------------|---------|----------|-------|----------|---------|--------|
//! | CartPole-v1   | 2.5e-4  | 0.01     | 0.99  | 0.2      | 128     | 4      |
//! | Pendulum-v1   | 1e-3    | 0.0      | 0.99  | 0.2      | 2048    | 1      |
//! | LunarLander   | 2.5e-4  | 0.01     | 0.99  | 0.2      | 128     | 16     |
//! | Atari (PPO)   | 2.5e-4  | 0.01     | 0.99  | 0.1      | 128     | 8      |
//! | MuJoCo (PPO)  | 3e-4    | 0.0      | 0.99  | 0.2      | 2048    | 1      |
//!
//! # Key hyperparameter effects
//!
//! **Learning rate (lr)**:
//!   - Too high: policy oscillates, KL explodes, training diverges
//!   - Too low: slow convergence, wastes compute
//!   - Sweet spot: approx_kl stays in 0.005-0.02 range
//!
//! **Entropy coefficient (ent_coef)**:
//!   - Too high: policy stays random, never commits to good actions
//!   - Too low: premature convergence to suboptimal deterministic policy
//!   - 0.0 works for continuous control; 0.01 typical for discrete
//!
//! **Other important hyperparameters (not tuned here)**:
//!   - gamma: higher = more foresight, lower = more myopic
//!   - clip_eps: 0.1-0.3, controls policy update aggressiveness
//!   - n_steps: more steps = lower variance but higher bias in GAE
//!   - update_epochs: more epochs = more sample efficiency but risk overfitting

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
// Single training run
// ---------------------------------------------------------------------------

type B = Autodiff<NdArray>;

struct RunResult {
    lr: f64,
    ent_coef: f32,
    avg_return: f32,
    final_entropy: f32,
    final_kl: f32,
}

/// Train PPO on CartPole with the given hyperparameters and return results.
///
/// Uses a fixed seed so results are reproducible across configurations.
/// In production, you would average over multiple seeds for robustness.
fn train_run(lr: f64, ent_coef: f32, total_steps: usize, seed: u64) -> RunResult {
    let device = NdArrayDevice::Cpu;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);

    let n_envs = 4;
    let envs: Vec<CartPole<rand::rngs::SmallRng>> = (0..n_envs)
        .map(|i| CartPole::new(rand::rngs::SmallRng::seed_from_u64(seed + 100 + i as u64)))
        .collect();
    let mut vec_env = SyncVecEnv::new(envs);

    let mut model: ActorCritic<B> = ActorCritic::new(&device);
    let mut optim = AdamConfig::new().with_epsilon(1e-5).init();

    let config = PpoConfig {
        lr,
        gamma: 0.99,
        gae_lambda: 0.95,
        clip_eps: 0.2,
        vf_coef: 0.5,
        ent_coef,
        update_epochs: 4,
        minibatch_size: 128,
        n_steps: 128,
        clip_vloss: true,
        max_grad_norm: 0.5,
        target_kl: None,
        dual_clip_coef: None,
    };

    let steps_per_iter = config.n_steps * n_envs;
    let n_iters = total_steps / steps_per_iter;

    let mut current_obs = vec_env.reset();
    let mut ep_acc = vec![0.0f32; n_envs];
    let mut recent_returns: Vec<f32> = Vec::new();
    let mut last_entropy = 0.0f32;
    let mut last_kl = 0.0f32;

    for iter in 0..n_iters {
        // Linear LR annealing (standard practice for PPO)
        let frac = 1.0 - iter as f64 / n_iters as f64;
        let current_lr = config.lr * frac;

        let rollout = ppo_collect::<NdArray, _, _>(
            &model.valid(),
            &mut vec_env,
            &config,
            &device,
            &mut rng,
            &mut current_obs,
            &mut ep_acc,
        );

        recent_returns.extend_from_slice(&rollout.episode_returns);
        if recent_returns.len() > 20 {
            let start = recent_returns.len() - 20;
            recent_returns = recent_returns[start..].to_vec();
        }

        let stats;
        (model, stats) = ppo_update(
            model,
            &mut optim,
            &rollout,
            &config,
            current_lr,
            &device,
            &mut rng,
        );

        last_entropy = stats.entropy;
        last_kl = stats.approx_kl;
    }

    let avg_return = if recent_returns.is_empty() {
        0.0
    } else {
        recent_returns.iter().sum::<f32>() / recent_returns.len() as f32
    };

    RunResult {
        lr,
        ent_coef,
        avg_return,
        final_entropy: last_entropy,
        final_kl: last_kl,
    }
}

// ---------------------------------------------------------------------------
// Main: grid search
// ---------------------------------------------------------------------------

fn main() {
    eprintln!("=== PPO Hyperparameter Tuning (Grid Search) ===");
    eprintln!();

    // Define the search grid.
    //
    // We tune the two most impactful hyperparameters:
    //   - Learning rate: controls step size of gradient descent
    //   - Entropy coefficient: controls exploration vs exploitation
    //
    // Grid search is simple but exhaustive: every combination is tried.
    // For larger grids, consider random search (Bergstra & Bengio, 2012)
    // or Bayesian optimization (e.g., Optuna).
    let learning_rates: &[f64] = &[1e-4, 2.5e-4, 5e-4];
    let entropy_coefs: &[f32] = &[0.0, 0.01, 0.05];

    // Short training runs for fast comparison.
    // In production, use longer runs (100k-500k steps) for reliable results.
    let total_steps = 50_000;
    let seed = 42;

    let total_configs = learning_rates.len() * entropy_coefs.len();
    eprintln!("Grid: {} lr x {} ent_coef = {} configurations", learning_rates.len(), entropy_coefs.len(), total_configs);
    eprintln!("Steps per config: {total_steps}");
    eprintln!("Environment: CartPole-v1");
    eprintln!();

    let mut results: Vec<RunResult> = Vec::new();
    let mut run_idx = 0;

    for &lr in learning_rates {
        for &ent_coef in entropy_coefs {
            run_idx += 1;
            eprintln!(
                "[{run_idx}/{total_configs}] lr={lr:.1e}, ent_coef={ent_coef:.2} ... ",
            );

            let result = train_run(lr, ent_coef, total_steps, seed);

            eprintln!(
                "  -> avg_return={:.1}, entropy={:.3}, kl={:.5}",
                result.avg_return, result.final_entropy, result.final_kl
            );

            results.push(result);
        }
    }

    // Sort by average return (descending) to find best config
    results.sort_by(|a, b| b.avg_return.partial_cmp(&a.avg_return).unwrap());

    // Print results table
    eprintln!();
    eprintln!("=== Results (sorted by avg_return, descending) ===");
    eprintln!();
    eprintln!(
        "{:<6} {:>10} {:>10} {:>12} {:>10} {:>10}",
        "Rank", "LR", "Ent Coef", "Avg Return", "Entropy", "KL"
    );
    eprintln!("{:-<68}", "");

    for (i, r) in results.iter().enumerate() {
        let marker = if i == 0 { " <-- BEST" } else { "" };
        eprintln!(
            "{:<6} {:>10.1e} {:>10.2} {:>12.1} {:>10.3} {:>10.5}{}",
            i + 1,
            r.lr,
            r.ent_coef,
            r.avg_return,
            r.final_entropy,
            r.final_kl,
            marker,
        );
    }

    let best = &results[0];
    eprintln!();
    eprintln!("Best configuration: lr={:.1e}, ent_coef={:.2}", best.lr, best.ent_coef);
    eprintln!("Best avg return: {:.1}", best.avg_return);

    // Analysis: what the grid search reveals about each hyperparameter
    eprintln!();
    eprintln!("=== Analysis ===");
    eprintln!();

    // Group by learning rate
    eprintln!("Average return by learning rate:");
    for &lr in learning_rates {
        let matching: Vec<&RunResult> = results.iter().filter(|r| r.lr == lr).collect();
        let avg: f32 = matching.iter().map(|r| r.avg_return).sum::<f32>() / matching.len() as f32;
        eprintln!("  lr={lr:.1e}: avg_return={avg:.1}");
    }

    eprintln!();
    eprintln!("Average return by entropy coefficient:");
    for &ec in entropy_coefs {
        let matching: Vec<&RunResult> = results.iter().filter(|r| r.ent_coef == ec).collect();
        let avg: f32 = matching.iter().map(|r| r.avg_return).sum::<f32>() / matching.len() as f32;
        eprintln!("  ent_coef={ec:.2}: avg_return={avg:.1}");
    }

    // Practical recommendations
    eprintln!();
    eprintln!("=== Practical Recommendations ===");
    eprintln!();
    eprintln!("1. Start with known-good defaults (lr=2.5e-4, ent_coef=0.01 for discrete).");
    eprintln!("2. If training diverges: lower lr, check KL (should be < 0.02).");
    eprintln!("3. If entropy collapses: increase ent_coef.");
    eprintln!("4. If learning is slow: try higher lr or more n_envs.");
    eprintln!("5. For production: run each config with 3-5 different seeds.");
    eprintln!("6. For larger search spaces: use random search or Bayesian optimization.");
}
