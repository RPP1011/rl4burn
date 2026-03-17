//! Integration test: masked multi-discrete PPO on a toy 2D navigation env.
//!
//! Environment: agent on a grid must reach a target position.
//! Action space: MultiDiscrete([3, 3]) — [dx ∈ {-1,0,1}, dy ∈ {-1,0,1}]
//! Masking: at boundaries, movement toward walls is masked out.

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::optim::AdamConfig;
use burn::prelude::*;

use rand::SeedableRng;

use rl4burn::env::space::Space;
use rl4burn::{
    masked_ppo_collect, masked_ppo_update, orthogonal_linear, ActionDist, Env,
    MaskedActorCritic, PpoConfig, Step, SyncVecEnv,
};

type AutodiffB = Autodiff<NdArray>;

// ---------------------------------------------------------------------------
// Multi-discrete navigation environment
// ---------------------------------------------------------------------------

const GRID_SIZE: i32 = 5;
const MAX_STEPS: usize = 50;

struct NavEnv<R> {
    x: i32,
    y: i32,
    target_x: i32,
    target_y: i32,
    step_count: usize,
    rng: R,
}

impl<R: rand::Rng> NavEnv<R> {
    fn new(mut rng: R) -> Self {
        let tx = rng.random_range(0..GRID_SIZE);
        let ty = rng.random_range(0..GRID_SIZE);
        Self {
            x: GRID_SIZE / 2,
            y: GRID_SIZE / 2,
            target_x: tx,
            target_y: ty,
            step_count: 0,
            rng,
        }
    }

    fn obs(&self) -> Vec<f32> {
        vec![
            self.x as f32 / GRID_SIZE as f32,
            self.y as f32 / GRID_SIZE as f32,
            self.target_x as f32 / GRID_SIZE as f32,
            self.target_y as f32 / GRID_SIZE as f32,
        ]
    }
}

impl<R: rand::Rng + Clone> Env for NavEnv<R> {
    type Observation = Vec<f32>;
    type Action = Vec<f32>;

    fn reset(&mut self) -> Vec<f32> {
        self.x = GRID_SIZE / 2;
        self.y = GRID_SIZE / 2;
        self.target_x = self.rng.random_range(0..GRID_SIZE);
        self.target_y = self.rng.random_range(0..GRID_SIZE);
        self.step_count = 0;
        self.obs()
    }

    fn step(&mut self, action: Vec<f32>) -> Step<Vec<f32>> {
        // action[0] ∈ {0,1,2} → dx ∈ {-1,0,1}
        // action[1] ∈ {0,1,2} → dy ∈ {-1,0,1}
        let dx = action[0] as i32 - 1;
        let dy = action[1] as i32 - 1;

        self.x = (self.x + dx).clamp(0, GRID_SIZE - 1);
        self.y = (self.y + dy).clamp(0, GRID_SIZE - 1);
        self.step_count += 1;

        let dist = ((self.x - self.target_x).abs() + (self.y - self.target_y).abs()) as f32;
        let reached = self.x == self.target_x && self.y == self.target_y;

        let reward = if reached {
            10.0
        } else {
            -0.1 - dist * 0.05
        };

        let terminated = reached;
        let truncated = self.step_count >= MAX_STEPS;

        Step {
            observation: self.obs(),
            reward,
            terminated,
            truncated,
        }
    }

    fn observation_space(&self) -> Space {
        Space::Box {
            low: vec![0.0; 4],
            high: vec![1.0; 4],
        }
    }

    fn action_space(&self) -> Space {
        Space::MultiDiscrete(vec![3, 3])
    }

    fn action_mask(&self) -> Option<Vec<f32>> {
        // 6 mask values: first 3 for dx, next 3 for dy
        let mut mask = vec![1.0f32; 6];

        // dx: action 0 = move left (-1), action 2 = move right (+1)
        if self.x == 0 {
            mask[0] = 0.0; // can't go left
        }
        if self.x == GRID_SIZE - 1 {
            mask[2] = 0.0; // can't go right
        }

        // dy: action 3 = move down (-1), action 5 = move up (+1)
        if self.y == 0 {
            mask[3] = 0.0; // can't go down
        }
        if self.y == GRID_SIZE - 1 {
            mask[5] = 0.0; // can't go up
        }

        Some(mask)
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[derive(Module, Debug)]
struct NavAgent<B: Backend> {
    fc1: burn::nn::Linear<B>,
    fc2: burn::nn::Linear<B>,
    policy_head: burn::nn::Linear<B>,
    value_head: burn::nn::Linear<B>,
}

impl<B: Backend> NavAgent<B> {
    fn new(device: &B::Device, rng: &mut impl rand::Rng) -> Self {
        let sqrt2 = std::f32::consts::SQRT_2;
        Self {
            fc1: orthogonal_linear(4, 64, sqrt2, device, rng),
            fc2: orthogonal_linear(64, 64, sqrt2, device, rng),
            // 6 logits: 3 for dx, 3 for dy
            policy_head: orthogonal_linear(64, 6, 0.01, device, rng),
            value_head: orthogonal_linear(64, 1, 1.0, device, rng),
        }
    }
}

impl<B: Backend> MaskedActorCritic<B> for NavAgent<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let h = self.fc1.forward(obs).tanh();
        let h = self.fc2.forward(h).tanh();
        let logits = self.policy_head.forward(h.clone());
        let values = self.value_head.forward(h).squeeze_dim::<1>(1);
        (logits, values)
    }
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

#[test]
fn multi_discrete_ppo_learns_navigation() {
    let device = NdArrayDevice::Cpu;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    let n_envs = 8;
    let envs: Vec<NavEnv<rand::rngs::SmallRng>> = (0..n_envs)
        .map(|i| NavEnv::new(rand::rngs::SmallRng::seed_from_u64(42 + i as u64)))
        .collect();
    let mut vec_env = SyncVecEnv::new(envs);

    let model: NavAgent<AutodiffB> = NavAgent::new(&device, &mut rng);
    let action_dist = ActionDist::MultiDiscrete(vec![3, 3]);

    let mut optim = AdamConfig::new().with_epsilon(1e-5).init();

    let config = PpoConfig {
        lr: 3e-4,
        gamma: 0.99,
        gae_lambda: 0.95,
        clip_eps: 0.2,
        vf_coef: 0.5,
        ent_coef: 0.01,
        update_epochs: 4,
        minibatch_size: 64,
        n_steps: 64,
        clip_vloss: true,
        max_grad_norm: 0.5,
    };

    let mut model = model;
    let total_timesteps = 200_000;
    let steps_per_iter = config.n_steps * n_envs;
    let n_iterations = total_timesteps / steps_per_iter;

    let mut recent_returns: Vec<f32> = Vec::new();
    let mut best_avg = f32::NEG_INFINITY;
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

        if recent_returns.len() > 50 {
            let start = recent_returns.len() - 50;
            recent_returns = recent_returns[start..].to_vec();
        }

        if !recent_returns.is_empty() {
            let avg: f32 = recent_returns.iter().sum::<f32>() / recent_returns.len() as f32;
            best_avg = best_avg.max(avg);

            if (iter + 1) % 50 == 0 {
                eprintln!(
                    "nav iter {:>4}/{}: avg_return={:>6.1} (best={:>6.1}) ploss={:>8.4} ent={:.3}",
                    iter + 1, n_iterations, avg, best_avg, stats.policy_loss, stats.entropy
                );
            }
        }
    }

    // The agent should learn to reach the target and get positive returns.
    // A random policy gets ~-5 on average. A decent policy should get > 0.
    eprintln!("Multi-discrete best avg: {best_avg:.1}");
    assert!(
        best_avg > 0.0,
        "Multi-discrete PPO should learn navigation (best avg {best_avg:.1}, expected >0)"
    );
}
