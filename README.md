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
use rl4burn::{
    ActionDist, LogStdMode, MaskedActorCritic,
    masked_ppo_collect, masked_ppo_update,
    PpoConfig,
};
use rl4burn::envs::Pendulum;

// ActionDist handles sampling, log-prob, and entropy for any action space
let action_dist = ActionDist::Continuous {
    action_dim: 1,
    log_std_mode: LogStdMode::ModelOutput,
};

// Your model implements MaskedActorCritic instead of DiscreteActorCritic
impl<B: Backend> MaskedActorCritic<B> for ContinuousAgent<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        // Return (logits [batch, n_logits], values [batch])
        // For Continuous/ModelOutput: logits = [means, log_stds]
        let h = self.encoder.forward(obs);
        let logits = self.policy_head.forward(h.clone());   // [batch, 2]
        let values = self.value_head.forward(h).squeeze(1);  // [batch]
        (logits, values)
    }
}
```

### PPO with action masking (GridWorld)

```rust
use rl4burn::ActionDist;
use rl4burn::envs::GridWorld;

// Multi-discrete: two independent dimensions, 3 choices each
let action_dist = ActionDist::MultiDiscrete(vec![3, 3]);

// Environments provide masks via the Env trait
impl Env for MyGameEnv {
    // ...
    fn action_mask(&self) -> Option<Vec<f32>> {
        let mut mask = vec![0.0; 49];  // 11 action types + 30 targets + 8 abilities
        // 1.0 = valid, 0.0 = invalid
        for i in 0..11 { mask[i] = 1.0; }
        for (i, unit) in self.enemies.iter().enumerate() {
            if unit.alive { mask[11 + i] = 1.0; }
        }
        Some(mask)
    }
}
```

### DQN on CartPole

```rust
use rl4burn::dqn::*;
use rl4burn::envs::CartPole;
use rl4burn::polyak::polyak_update;
use rl4burn::replay::ReplayBuffer;

let config = DqnConfig::default();
let mut buffer = ReplayBuffer::new(config.buffer_capacity, rng);

for step in 0..50_000 {
    let eps = epsilon_schedule(&config, step);
    let action = epsilon_greedy::<NdArray, _>(&online.valid(), &obs, 2, eps, &device, &mut rng);

    let result = env.step(action);
    buffer.extend(std::iter::once(Transition {
        obs: obs.clone(), action: action as i32,
        reward: result.reward, next_obs: result.observation.clone(),
        done: result.done(),
    }));

    if step >= config.learning_starts && buffer.len() >= config.batch_size {
        (online, _) = dqn_update(online, &target, &mut optim, &mut buffer, &config, &device);
        if step % 250 == 0 {
            target = polyak_update(target, &online, 1.0);
        }
    }
}
```

## Environments

| Environment | Observation | Action | Notes |
|-------------|------------|--------|-------|
| **CartPole** | `[x, x_dot, θ, θ_dot]` | Discrete(2) | Classic balance task, 500-step truncation |
| **Pendulum** | `[cos(θ), sin(θ), θ_dot]` | Continuous(1) | Swing-up with torque in [-2, 2] |
| **GridWorld** | `[agent_x, agent_y, goal_x, goal_y]` | Discrete(4) | 7x7 grid with boundary masking |

All environments implement the `Env` trait. CartPole and GridWorld implement `Renderable` for GIF recording.

To use your own environment, implement `Env`:

```rust
pub trait Env {
    type Observation: Clone;
    type Action: Clone;
    fn reset(&mut self) -> Self::Observation;
    fn step(&mut self, action: Self::Action) -> Step<Self::Observation>;
    fn observation_space(&self) -> Space;
    fn action_space(&self) -> Space;
    fn action_mask(&self) -> Option<Vec<f32>> { None }  // optional
}
```

### Wrappers

| Wrapper | Description |
|---------|-------------|
| `EpisodeStats` | Tracks per-episode reward and length |
| `NormalizeObservation` | Running mean/std observation normalization |
| `NormalizeReward` | Reward normalization with discounted return tracking |
| `DiscreteEnvAdapter` | Bridges discrete (`Action = usize`) envs to the masked PPO pipeline (`Action = Vec<f32>`) |

## Action distributions

`ActionDist` unifies sampling, log-probability, and entropy computation across all action space types:

```rust
pub enum ActionDist {
    Discrete(usize),              // Single categorical
    MultiDiscrete(Vec<usize>),    // Independent categoricals
    Continuous {                   // Diagonal Gaussian
        action_dim: usize,
        log_std_mode: LogStdMode, // ModelOutput or Separate
    },
}
```

Masks are `Option<Tensor<B, 2>>` with shape `[batch, n_logits]` (1.0 = valid, 0.0 = invalid), applied before softmax. Continuous distributions clamp log_std to [-5, 2] for numerical stability.

## Building blocks

### Core RL

| Module | Description |
|--------|-------------|
| `gae::gae` | Generalized Advantage Estimation |
| `vtrace::vtrace_targets` | V-trace off-policy correction |
| `replay::ReplayBuffer` | FIFO replay buffer with uniform sampling |
| `polyak::polyak_update` | Soft/hard target network updates |
| `advantage::normalize` | Advantage normalization with clamping |
| `percentile_normalize` | Percentile-based return normalization (DreamerV3-style) |

### Neural network utilities

| Module | Description |
|--------|-------------|
| `init::orthogonal_linear` | Orthogonal weight init (CleanRL's `layer_init`) |
| `clip::clip_grad_norm` | Global gradient norm clipping (PyTorch-compatible) |
| `loss::value_loss` | Huber value loss |
| `loss::policy_loss_discrete` | REINFORCE policy gradient |
| `loss::policy_loss_continuous` | Advantage-weighted regression |
| `nn::dist::ActionDist` | Action distributions with masking |

### Attention and sequence models

| Module | Description |
|--------|-------------|
| `nn::attention` | Multi-head attention, attention-based pooling |
| `nn::transformer` | Transformer encoder blocks with LayerNorm |
| `nn::rnn` | LSTM cells, GRU cells, block-diagonal GRU |
| `nn::pointer` | Pointer networks for entity selection |
| `nn::film` | Feature-wise Linear Modulation conditioning |

### Model-based RL (DreamerV3 components)

| Module | Description |
|--------|-------------|
| `nn::rssm` | Recurrent State-Space Model |
| `nn::symlog` | Symlog transform and twohot distributional predictions |
| `nn::kl_balance` | KL balancing with free bits |
| `nn::vae` | Variational autoencoder |
| `algo::imagination` | Imagination rollout engine with lambda-returns |
| `collect::sequence_replay` | Episode-based sequence replay buffer |

### Multi-agent and self-play

| Module | Description |
|--------|-------------|
| `algo::self_play` | Self-play pool with agent snapshots |
| `algo::pfsp` | Prioritized Fictitious Self-Play matchmaking |
| `algo::league` | Multi-agent league management |
| `algo::multi_agent` | Batched multi-agent observation/action utilities |
| `algo::privileged_critic` | Asymmetric actor-critic with privileged information |

### Training pipelines

| Module | Description |
|--------|-------------|
| `algo::behavioral_cloning` | Behavioral cloning from demonstrations |
| `algo::distillation` | Policy and value distillation |
| `algo::cspl` | Curriculum Self-Play Learning pipeline |
| `algo::z_conditioning` | Goal-conditioned RL with strategy embeddings |

## Logging and visualization

Log training metrics to TensorBoard, JSONL, or stdout. Record agent behavior as GIFs.

```bash
cargo run --release --example ppo_cartpole --features "ndarray,tensorboard"
tensorboard --logdir runs/

cargo run --release --example ppo_cartpole --features "ndarray,json-log" 2>&1 \
  | python scripts/wandb_bridge.py
```

```rust
use rl4burn::{Loggable, PrintLogger};

let mut logger = PrintLogger::new(0);
stats.log(&mut logger, step);  // PpoStats, DqnStats, etc. implement Loggable
```

Optional feature flags (core crate has zero logging dependencies):

| Feature | What you get |
|---------|-------------|
| `tensorboard` | TFEvent files for `tensorboard --logdir` |
| `json-log` | JSONL output for wandb/mlflow/custom dashboards |
| `video` | `write_gif()` + `Renderable::render()` for episode recording |

## Model saving and loading

```rust
use burn::record::{CompactRecorder, Recorder};

model.save_file("checkpoints/ppo_cartpole", &CompactRecorder::new()).unwrap();

let model = ActorCritic::new(&device)
    .load_file("checkpoints/ppo_cartpole", &CompactRecorder::new(), &device)
    .unwrap();
```

## Running tests

```bash
cargo test --release
```

The test suite trains agents to convergence across multiple action space types:

```
test ppo_solves_cartpole         ... ok  (~0.7s, discrete PPO)
test ppo_masked_cartpole         ... ok  (~0.8s, MaskedActorCritic adapter)
test ppo_multi_discrete          ... ok  (~1s, MultiDiscrete with boundary masking)
test ppo_continuous_pendulum     ... ok  (~30s, continuous PPO with normalization)
test dqn_solves_cartpole         ... ok  (~9s, DQN with experience replay)
test gradient_flow               ... ok  (<1s, autodiff through orthogonal init)
```

## Burn compatibility notes

This crate works around several Burn 0.20 behaviors:

1. **`Param::initialized` does not set `require_grad`**: Custom weight initialization must use `Param::from_data` + `load_record`, not `Param::initialized(id, tensor)`. `orthogonal_linear` handles this correctly.

2. **Gradient clipping is per-parameter**: Burn's `GradientClippingConfig::Norm` clips each parameter tensor independently. Use `clip::clip_grad_norm` for PyTorch-compatible global norm clipping.

3. **`mask_where` gradient flow**: Burn's autodiff may not propagate gradients through `mask_where`'s source argument. Use arithmetic alternatives like `min(a,b) = b - relu(b - a)`.

## License

MIT OR Apache-2.0
