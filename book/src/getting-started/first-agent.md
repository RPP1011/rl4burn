# Your First Agent: PPO on CartPole

This walkthrough trains a PPO agent to balance a pole on a cart. By the end, you'll understand the three pieces every rl4burn training script needs: a **model**, **environments**, and a **training loop**.

## The model

PPO needs an actor-critic model: given an observation, produce action logits and a value estimate. Define a Burn module and implement `DiscreteActorCritic`:

```rust,ignore
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use rl4burn::{DiscreteAcOutput, DiscreteActorCritic};

#[derive(Module, Debug)]
struct ActorCritic<B: Backend> {
    // Separate actor and critic networks (no shared layers)
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
```

Key points:
- `#[derive(Module)]` gives you parameter management, serialization, and device transfer for free.
- `DiscreteAcOutput` holds `logits: Tensor<B, 2>` (shape `[batch, n_actions]`) and `values: Tensor<B, 1>` (shape `[batch]`).
- The model is generic over `B: Backend`. The same struct works on any Burn backend.

## The environments

CartPole is built in. Wrap it in `SyncVecEnv` to run multiple copies in parallel:

```rust,ignore
use rl4burn::envs::CartPole;
use rl4burn::SyncVecEnv;
use rand::SeedableRng;

let n_envs = 4;
let envs: Vec<CartPole<rand::rngs::SmallRng>> = (0..n_envs)
    .map(|i| CartPole::new(rand::rngs::SmallRng::seed_from_u64(i as u64)))
    .collect();
let mut vec_env = SyncVecEnv::new(envs);
```

`SyncVecEnv` steps all environments in lockstep and auto-resets when episodes end.

## The training loop

PPO training alternates between two phases: **collect** a rollout of experience, then **update** the model on that experience.

```rust,ignore
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::optim::AdamConfig;
use rl4burn::{ppo_collect, ppo_update, PpoConfig};
use rl4burn::{Loggable, Logger, PrintLogger};

type AB = Autodiff<NdArray>;

let device = burn::backend::ndarray::NdArrayDevice::Cpu;
let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

let mut model: ActorCritic<AB> = ActorCritic::new(&device);
let mut optim = AdamConfig::new().with_epsilon(1e-5).init();
let config = PpoConfig::default();
let mut logger = PrintLogger::new(0);

// Episode return accumulator — persists across rollouts
let mut ep_acc = vec![0.0f32; n_envs];
// Current observations — persists across rollouts
let mut current_obs = vec_env.reset();

for iter in 0..100 {
    // Collect: use the non-autodiff model for inference
    let rollout = ppo_collect::<NdArray, _, _>(
        &model.valid(),
        &mut vec_env,
        &config,
        &device,
        &mut rng,
        &mut current_obs,
        &mut ep_acc,
    );

    // Update: train on the collected data
    let (new_model, stats) = ppo_update(
        model, &mut optim, &rollout, &config,
        config.lr, // or use LR annealing
        &device, &mut rng,
    );
    model = new_model;

    // Log training stats
    let step = (iter + 1) as u64 * (config.n_steps * n_envs) as u64;
    stats.log(&mut logger, step);

    if !rollout.episode_returns.is_empty() {
        let avg = rollout.episode_returns.iter().sum::<f32>()
            / rollout.episode_returns.len() as f32;
        logger.log_scalar("rollout/avg_return", avg as f64, step);
    }
}
logger.flush();
```

Key points:
- `model.valid()` strips the autodiff layer for efficient inference during collection.
- `current_obs` holds the latest observations from the environments, persisting across rollout boundaries so the next collection starts from where the last one left off.
- `ep_acc` tracks per-env cumulative reward across rollout boundaries. Without this, episodes longer than `n_steps` would have their returns split.
- `ppo_update` returns the updated model (Burn modules are moved through optimizers, not mutated in place).
- `stats.log(...)` uses the `Loggable` trait to log all PPO metrics. See the [Logging](../building-blocks/logging.md) chapter for details on logger setup.

## Run it

```bash
cargo run -p quickstart --release
```

You should see episode returns climb from ~20 (random policy) to 500 (solved) within seconds.
