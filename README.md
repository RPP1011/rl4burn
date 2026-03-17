# rl4burn

Reinforcement learning library for the [Burn](https://burn.dev) deep learning framework.

Write your model once with `B: Backend` and train it on **NdArray** (CPU), **WGPU** (GPU), **CUDA**, or **LibTorch** with zero code changes.

## Algorithms

| Algorithm | Action space | Model trait | Status |
|-----------|-------------|-------------|--------|
| **PPO** (discrete) | `Discrete` | `DiscreteActorCritic` | Solves CartPole in <1s |
| **PPO** (masked / multi-discrete / continuous) | `Discrete`, `MultiDiscrete`, `Continuous` | `MaskedActorCritic` | Solves Pendulum, GridWorld |
| **DQN** | `Discrete` | `QNetwork` | Solves CartPole in ~9s |

## Quick start

```toml
[dependencies]
rl4burn = { git = "https://github.com/RPP1011/rl4burn" }
burn = { version = "0.20", features = ["std", "ndarray", "autodiff"] }
rand = "0.9"
```

### PPO on CartPole (discrete)

```rust
use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig};
use burn::optim::AdamConfig;
use burn::prelude::*;
use rand::SeedableRng;

use rl4burn::envs::CartPole;
use rl4burn::policy::{DiscreteAcOutput, DiscreteActorCritic};
use rl4burn::ppo::{ppo_collect, ppo_update, PpoConfig};
use rl4burn::vec_env::SyncVecEnv;

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

type AB = Autodiff<NdArray>;

fn main() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    let n_envs = 4;
    let envs: Vec<CartPole<rand::rngs::SmallRng>> = (0..n_envs)
        .map(|i| CartPole::new(rand::rngs::SmallRng::seed_from_u64(i as u64)))
        .collect();
    let mut vec_env = SyncVecEnv::new(envs);
    let mut model: ActorCritic<AB> = ActorCritic::new(&device);
    let mut optim = AdamConfig::new().with_epsilon(1e-5).init();
    let config = PpoConfig::default();
    let mut ep_acc = vec![0.0f32; n_envs];

    for iter in 0..100 {
        let rollout = ppo_collect::<burn::backend::NdArray, _, _>(
            &model.valid(), &mut vec_env, &config, &device, &mut rng, &mut ep_acc,
        );
        let (new_model, stats) = ppo_update(
            model, &mut optim, &rollout, &config, config.lr, &device, &mut rng,
        );
        model = new_model;

        if !rollout.episode_returns.is_empty() {
            let avg: f32 = rollout.episode_returns.iter().sum::<f32>()
                / rollout.episode_returns.len() as f32;
            println!("iter {iter}: avg_return={avg:.0}");
        }
    }
}
```

### PPO with continuous actions (Pendulum)

```rust
use rl4burn::{ActionDist, LogStdMode, MaskedActorCritic, masked_ppo_collect, masked_ppo_update};

let action_dist = ActionDist::Continuous {
    action_dim: 1,
    log_std_mode: LogStdMode::ModelOutput,
};

// MaskedActorCritic returns (logits, values) — ActionDist handles the rest
impl<B: Backend> MaskedActorCritic<B> for ContinuousAgent<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let h = self.encoder.forward(obs);
        let logits = self.policy_head.forward(h.clone());   // [batch, 2] = [mean, log_std]
        let values = self.value_head.forward(h).squeeze(1);
        (logits, values)
    }
}
```

### PPO with action masking (GridWorld)

```rust
let action_dist = ActionDist::MultiDiscrete(vec![3, 3]);

// Environments provide masks via the Env trait
impl Env for MyEnv {
    fn action_mask(&self) -> Option<Vec<f32>> {
        let mut mask = vec![1.0; 6]; // 3+3 logits
        if self.at_left_wall()  { mask[0] = 0.0; } // can't go left
        if self.at_right_wall() { mask[2] = 0.0; } // can't go right
        Some(mask)
    }
    // ...
}
```

### DQN on CartPole

```rust
use rl4burn::dqn::*;
use rl4burn::replay::ReplayBuffer;
use rl4burn::polyak::polyak_update;

let config = DqnConfig::default();
let mut buffer = ReplayBuffer::new(config.buffer_capacity, rng);

for step in 0..50_000 {
    let eps = epsilon_schedule(&config, step);
    let action = epsilon_greedy::<NdArray, _>(&online.valid(), &obs, 2, eps, &device, &mut rng);
    let result = env.step(action);
    buffer.extend(std::iter::once(Transition { obs, action: action as i32, reward: result.reward, next_obs: result.observation.clone(), done: result.done() }));
    obs = if result.done() { env.reset() } else { result.observation };

    if step >= config.learning_starts && buffer.len() >= config.batch_size {
        (online, _) = dqn_update(online, &target, &mut optim, &mut buffer, &config, &device);
        if step % 250 == 0 { target = polyak_update(target, &online, 1.0); }
    }
}
```

## Environments

| Environment | Observation | Action | Notes |
|-------------|------------|--------|-------|
| **CartPole** | `[x, x_dot, θ, θ_dot]` | Discrete(2) | Classic balance task, 500-step truncation |
| **Pendulum** | `[cos(θ), sin(θ), θ_dot]` | Continuous(1) | Swing-up, torque in [-2, 2] |
| **GridWorld** | `[agent_x, agent_y, goal_x, goal_y]` | Discrete(4) | 7x7 grid with boundary masking |

Implement `Env` for your own:

```rust
pub trait Env {
    type Observation: Clone;
    type Action: Clone;
    fn reset(&mut self) -> Self::Observation;
    fn step(&mut self, action: Self::Action) -> Step<Self::Observation>;
    fn observation_space(&self) -> Space;
    fn action_space(&self) -> Space;
    fn action_mask(&self) -> Option<Vec<f32>> { None }
}
```

Wrappers: `EpisodeStats`, `NormalizeObservation`, `NormalizeReward`, `DiscreteEnvAdapter`.

## Core building blocks

| Module | Description |
|--------|-------------|
| `gae::gae` | Generalized Advantage Estimation |
| `vtrace::vtrace_targets` | V-trace off-policy correction |
| `replay::ReplayBuffer` | FIFO replay buffer with uniform sampling |
| `polyak::polyak_update` | Soft/hard target network updates |
| `nn::dist::ActionDist` | Discrete, MultiDiscrete, and Continuous distributions with masking |
| `init::orthogonal_linear` | Orthogonal weight init (CleanRL's `layer_init`) |
| `clip::clip_grad_norm` | Global gradient norm clipping (PyTorch-compatible) |

## Extras

The library also includes building blocks for more advanced use cases — neural network layers (transformers, attention, RNNs, pointer networks), model-based RL components (RSSM, symlog, imagination rollouts), multi-agent infrastructure (self-play, league training, PFSP matchmaking), and training pipelines (behavioral cloning, distillation, curriculum self-play). See the [book](https://rpp1011.github.io/rl4burn/) and module docs for details.

## Logging and visualization

```bash
cargo run --release --example ppo_cartpole --features "ndarray,tensorboard"
tensorboard --logdir runs/
```

| Feature flag | What you get |
|-------------|-------------|
| `tensorboard` | TFEvent files for `tensorboard --logdir` |
| `json-log` | JSONL output for wandb/mlflow/custom dashboards |
| `video` | `write_gif()` + `Renderable::render()` for episode recording |

## Tests

```bash
cargo test --release
```

```
test ppo_solves_cartpole         ... ok  (~0.7s, discrete)
test ppo_masked_cartpole         ... ok  (~0.8s, MaskedActorCritic adapter)
test ppo_multi_discrete          ... ok  (~1s, MultiDiscrete + boundary masking)
test ppo_continuous_pendulum     ... ok  (~30s, continuous + normalization)
test dqn_solves_cartpole         ... ok  (~9s)
```

## Burn compatibility notes

1. **`Param::initialized` does not set `require_grad`**: Use `Param::from_data` + `load_record`. `orthogonal_linear` handles this correctly.
2. **Gradient clipping is per-parameter**: Burn clips each tensor independently. Use `clip_grad_norm` for PyTorch-compatible global norm clipping.
3. **`mask_where` gradient flow**: Use arithmetic alternatives like `min(a,b) = b - relu(b - a)`.

## License

MIT OR Apache-2.0
