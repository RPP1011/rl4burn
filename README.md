# rl4burn

Reinforcement learning algorithms for the [Burn](https://burn.dev) ML framework.

Write your algorithm once with `B: AutodiffBackend` and run it on WGPU, CUDA, NdArray, or LibTorch — no reimplementation per backend.

## Algorithms

| Algorithm | Type | Trait | Status |
|-----------|------|-------|--------|
| **PPO** | On-policy, actor-critic | `DiscreteActorCritic` | Solves CartPole in <1s |
| **DQN** | Off-policy, value-based | `QNetwork` | Solves CartPole in ~9s |

## Quick start

Add to `Cargo.toml`:

```toml
[dependencies]
rl4burn = { git = "https://github.com/RPP1011/rl4burn" }
burn = { version = "0.20", features = ["std", "ndarray", "autodiff"] }
rand = "0.9"
```

### PPO on CartPole

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

// 1. Define your model
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

// 2. Implement the trait
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

// 3. Train
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

### DQN on CartPole

```rust
use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module};
use burn::nn::{Linear, LinearConfig};
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::tensor::activation::relu;
use rand::SeedableRng;

use rl4burn::dqn::*;
use rl4burn::env::Env;
use rl4burn::envs::CartPole;
use rl4burn::polyak::polyak_update;
use rl4burn::replay::ReplayBuffer;

// 1. Define your Q-network
#[derive(Module, Debug)]
struct QNet<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    q_head: Linear<B>,
}

impl<B: Backend> QNetwork<B> for QNet<B> {
    fn q_values(&self, obs: Tensor<B, 2>) -> Tensor<B, 2> {
        let h = relu(self.fc1.forward(obs));
        let h = relu(self.fc2.forward(h));
        self.q_head.forward(h)
    }
}

// 2. Train
type AB = Autodiff<NdArray>;

fn main() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
    let mut env = CartPole::new(rand::rngs::SmallRng::seed_from_u64(0));
    let config = DqnConfig::default();

    let mut online: QNet<AB> = QNet {
        fc1: LinearConfig::new(4, 64).init(&device),
        fc2: LinearConfig::new(64, 64).init(&device),
        q_head: LinearConfig::new(64, 2).init(&device),
    };
    let mut target = online.clone();
    let mut optim = AdamConfig::new().init();
    let mut buffer = ReplayBuffer::new(config.buffer_capacity, rand::rngs::SmallRng::seed_from_u64(42));
    let mut obs = env.reset();

    for step in 0..50_000 {
        let eps = epsilon_schedule(&config, step);
        let action = {
            let inner = online.valid();
            epsilon_greedy::<NdArray, _>(&inner, &obs, 2, eps, &device, &mut rng)
        };

        let result = env.step(action);
        buffer.extend(std::iter::once(Transition {
            obs: obs.clone(), action: action as i32,
            reward: result.reward, next_obs: result.observation.clone(),
            done: result.done(),
        }));
        obs = if result.done() { env.reset() } else { result.observation };

        if step >= config.learning_starts && buffer.len() >= config.batch_size {
            (online, _) = dqn_update(online, &target, &mut optim, &mut buffer, &config, &device);
            if step % 250 == 0 {
                target = polyak_update(target, &online, 1.0);
            }
        }
    }
}
```

## Logging & visualization

Log training metrics to TensorBoard, save runs as JSONL, or record agent behavior as GIFs:

```bash
# Train PPO with TensorBoard logging, then view in browser
cargo run --release --example ppo_cartpole --features "ndarray,tensorboard"
tensorboard --logdir runs/

# Export training metrics as JSONL and pipe to Weights & Biases
cargo run --release --example ppo_cartpole --features "ndarray,json-log" 2>&1 \
  | python scripts/wandb_bridge.py          # requires `wandb login` first
  # or: | python scripts/wandb_bridge.py --offline  # no account needed
```

Use the `Loggable` trait to log stats from any algorithm in one line:

```rust,ignore
use rl4burn::{Loggable, PrintLogger};

let mut logger = PrintLogger::new(0);
let (model, stats) = ppo_update(model, &mut optim, &rollout, &config, lr, &device, &mut rng);
stats.log(&mut logger, step);
```

Save and load model weights with Burn's recorder system:

```rust,ignore
use burn::record::{CompactRecorder, Recorder};

// Save
model.save_file("checkpoints/ppo_cartpole", &CompactRecorder::new()).unwrap();

// Load
let model = ActorCritic::new(&device)
    .load_file("checkpoints/ppo_cartpole", &CompactRecorder::new(), &device)
    .unwrap();
```

Optional feature flags — the core crate has zero logging dependencies:

| Feature | What you get |
|---------|-------------|
| `tensorboard` | TFEvent files for `tensorboard --logdir` |
| `json-log` | JSONL output for wandb/mlflow/custom dashboards |
| `video` | `write_gif()` + `Renderable::render()` for episode recording |

## Architecture

```
rl4burn
  Algorithms:     ppo, dqn
  Env:            Env trait, SyncVecEnv, EpisodeStats/RewardClip/NormalizeObservation wrappers
  Environments:   CartPole
  Building blocks: GAE, V-trace, replay buffer, polyak updates, loss functions,
                   advantage normalization, orthogonal init, global gradient clipping
  Spaces:         Discrete, Box, MultiDiscrete
```

### Core traits

**`Env`** — Gymnasium-style environment with `reset()` / `step()`, separate `terminated` / `truncated` flags:

```rust
pub trait Env {
    type Observation: Clone;
    type Action: Clone;
    fn reset(&mut self) -> Self::Observation;
    fn step(&mut self, action: Self::Action) -> Step<Self::Observation>;
    fn observation_space(&self) -> Space;
    fn action_space(&self) -> Space;
}
```

**`DiscreteActorCritic`** — Actor-critic model for PPO (logits + values in one forward pass):

```rust
pub trait DiscreteActorCritic<B: Backend> {
    fn forward(&self, obs: Tensor<B, 2>) -> DiscreteAcOutput<B>;
}
```

**`QNetwork`** — Q-value network for DQN:

```rust
pub trait QNetwork<B: Backend> {
    fn q_values(&self, obs: Tensor<B, 2>) -> Tensor<B, 2>;
}
```

### PPO details

`ppo_collect` and `ppo_update` are separate functions — you own the training loop:

- **Per-minibatch advantage normalization** (matches CleanRL)
- **Value loss clipping** (configurable)
- **Global gradient norm clipping** via `clip_grad_norm` (PyTorch-compatible, not Burn's per-parameter clipping)
- **LR annealing** via the `current_lr` parameter
- **Episode return tracking** built into `ppo_collect` via the `episode_returns_acc` accumulator

Default hyperparameters match [CleanRL's ppo.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py).

### DQN details

- **Experience replay** via `ReplayBuffer`
- **Target network** via `polyak_update` (hard or soft updates)
- **Epsilon-greedy** exploration with linear annealing
- Caller owns the training loop and target update schedule

### Utilities

| Module | Description |
|--------|-------------|
| `init::orthogonal_linear` | Orthogonal weight initialization (CleanRL's `layer_init`) |
| `clip::clip_grad_norm` | Global gradient norm clipping (matches PyTorch's `clip_grad_norm_`) |
| `gae::gae` | Generalized Advantage Estimation |
| `vtrace::vtrace_targets` | V-trace off-policy correction |
| `polyak::polyak_update` | Soft/hard target network updates |
| `loss::value_loss` | Huber value loss |
| `loss::policy_loss_discrete` | REINFORCE policy gradient |
| `loss::policy_loss_continuous` | Advantage-weighted regression |
| `advantage::normalize` | Advantage normalization with clamping |
| `replay::ReplayBuffer` | Uniform replay buffer with trajectory grouping |

## Burn compatibility notes

This crate works around several Burn 0.20 behaviors:

1. **`Param::initialized` does not set `require_grad`**: Custom weight initialization must use `Param::from_data` + `load_record`, not `Param::initialized(id, tensor)`. The latter creates parameters invisible to autodiff. `orthogonal_linear` handles this correctly.

2. **Gradient clipping is per-parameter**: Burn's `GradientClippingConfig::Norm` clips each parameter tensor independently. PyTorch's `clip_grad_norm_` clips the global norm across all parameters. Use `clip::clip_grad_norm` for PyTorch-compatible behavior.

3. **`mask_where` gradient flow**: Burn's autodiff may not propagate gradients through the `source` argument of `mask_where`. Use arithmetic alternatives like `min(a,b) = b - relu(b - a)` or `max(a,b) = a + relu(b - a)`.

## Running tests

```bash
cargo test --release
```

The test suite includes integration tests that train PPO and DQN on CartPole to convergence:

```
test ppo_solves_cartpole ... ok    (0.7s, asserts avg_return > 400)
test dqn_solves_cartpole ... ok    (9s, asserts avg_return > 200)
```

## License

MIT OR Apache-2.0
