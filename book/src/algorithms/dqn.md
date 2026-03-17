# DQN (Deep Q-Network)

DQN is an off-policy, value-based algorithm. It learns a Q-function that estimates the expected return for each action in a given state, then acts by taking the argmax.

## API

- **`QNetwork` trait** — Your model implements `fn q_values(&self, obs) -> Tensor` returning Q-values for all actions.
- **`dqn_update`** — One gradient step on a minibatch sampled from the replay buffer, using the target network for stable Bellman targets.
- **`epsilon_greedy`** — Action selection with exploration.
- **`epsilon_schedule`** — Linear epsilon annealing.
- **`polyak_update`** — Target network update (hard or soft).

## The QNetwork trait

```rust,ignore
pub trait QNetwork<B: Backend> {
    fn q_values(&self, obs: Tensor<B, 2>) -> Tensor<B, 2>;
}
```

Input shape: `[batch, obs_dim]`. Output shape: `[batch, n_actions]`.

Example implementation:

```rust,ignore
use burn::tensor::activation::relu;

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
```

## Training loop

DQN differs from PPO: it uses a **single environment**, a **replay buffer**, and **epsilon-greedy** exploration.

```rust,ignore
use rl4burn::dqn::*;
use rl4burn::replay::ReplayBuffer;
use rl4burn::polyak::polyak_update;
use rl4burn::env::Env;

let config = DqnConfig::default();
let mut buffer = ReplayBuffer::new(config.buffer_capacity);
let mut online: QNet<AB> = QNet::new(&device);
let mut target = online.clone();
let mut optim = AdamConfig::new().init();
let mut obs = env.reset();

for step in 0..50_000 {
    // Epsilon-greedy action selection (use non-autodiff model)
    let eps = epsilon_schedule(&config, step);
    let action = {
        let inner = online.valid();
        epsilon_greedy::<NdArray, _>(&inner, &obs, 2, eps, &device, &mut rng)
    };

    // Step environment, store transition
    let result = env.step(action);
    buffer.extend(std::iter::once(Transition {
        obs: obs.clone(),
        action: action as i32,
        reward: result.reward,
        next_obs: result.observation.clone(),
        done: result.done(),
    }));
    obs = if result.done() { env.reset() } else { result.observation };

    // Train after warmup
    if step >= config.learning_starts && buffer.len() >= config.batch_size {
        (online, _) = dqn_update(
            online, &target, &mut optim, &mut buffer, &config, &device,
        );

        // Hard target update every N steps
        if step % 250 == 0 {
            target = polyak_update(target, &online, 1.0);
        }
    }
}
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 1e-4 | Learning rate |
| `gamma` | 0.99 | Discount factor |
| `buffer_capacity` | 10,000 | Replay buffer size |
| `batch_size` | 32 | Minibatch size |
| `tau` | 0.005 | Polyak coefficient (1.0 = hard copy) |
| `eps_start` | 1.0 | Initial exploration rate |
| `eps_end` | 0.05 | Final exploration rate |
| `eps_decay_steps` | 10,000 | Steps to anneal epsilon |
| `learning_starts` | 1,000 | Random steps before training |
| `train_frequency` | 1 | Env steps per gradient step |

## Target network

DQN uses a slowly-updated target network for stable Bellman targets. Two strategies:

- **Hard updates** (`tau = 1.0`): Copy all weights every N steps. Simpler, what CleanRL uses.
- **Soft updates** (`tau = 0.005`): Polyak average every step. Smoother, what SAC/TD3 use.

The caller is responsible for calling `polyak_update` — `dqn_update` only updates the online network.

## How dqn_update works

1. Sample a minibatch from the replay buffer
2. Compute Q(s, a) for taken actions using the **online** network
3. Compute max Q(s', a') using the **target** network (detached from the computation graph by extracting tensor data)
4. Bellman target: `y = r + γ * (1 - done) * max_a' Q_target(s', a')`
5. MSE loss: `mean((Q(s, a) - y)²)`
6. Backward + optimizer step
