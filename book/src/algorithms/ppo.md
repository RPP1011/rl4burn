# PPO (Proximal Policy Optimization)

PPO is an on-policy actor-critic algorithm. It collects a batch of experience using the current policy, computes advantages, then performs multiple epochs of minibatch gradient descent with a clipped surrogate objective.

Our implementation matches [CleanRL's ppo.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py).

## API

PPO is split into two functions:

- **`ppo_collect`** — Run the policy on vectorized environments, collect transitions, compute GAE advantages.
- **`ppo_update`** — Perform clipped PPO gradient steps on the collected data.

You compose them in your own training loop.

## The DiscreteActorCritic trait

```rust,ignore
pub trait DiscreteActorCritic<B: Backend> {
    fn forward(&self, obs: Tensor<B, 2>) -> DiscreteAcOutput<B>;
}

pub struct DiscreteAcOutput<B: Backend> {
    pub logits: Tensor<B, 2>,  // [batch, n_actions]
    pub values: Tensor<B, 1>,  // [batch]
}
```

Implement this on any `#[derive(Module)]` struct. The model must output both action logits (for the policy) and value estimates (for the critic) in a single forward pass.

## Configuration

`PpoConfig` defaults match CleanRL:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 2.5e-4 | Learning rate |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE smoothing parameter |
| `clip_eps` | 0.2 | Surrogate clipping range |
| `vf_coef` | 0.5 | Value loss coefficient |
| `ent_coef` | 0.01 | Entropy bonus coefficient |
| `update_epochs` | 4 | Optimization epochs per rollout |
| `minibatch_size` | 128 | Minibatch size |
| `n_steps` | 128 | Rollout length per env |
| `clip_vloss` | true | Whether to clip value loss |
| `max_grad_norm` | 0.5 | Global gradient norm clipping (0 = disabled) |

## LR annealing

`ppo_update` takes a `current_lr` parameter. For linear annealing:

```rust,ignore
let frac = 1.0 - iter as f64 / n_iterations as f64;
let current_lr = config.lr * frac;
```

For constant LR, just pass `config.lr`.

## Episode return tracking

`ppo_collect` accepts an `&mut Vec<f32>` accumulator for per-env episode returns. This handles episodes that span multiple rollouts correctly. Completed episode returns are in `PpoRollout::episode_returns`.

```rust,ignore
let mut current_obs = vec_env.reset(); // create once before the loop
let mut ep_acc = vec![0.0f32; n_envs];

let rollout = ppo_collect(..., &mut current_obs, &mut ep_acc);
for &ret in &rollout.episode_returns {
    println!("completed episode return: {ret}");
}
```

## Multi-discrete actions and action masking

For complex action spaces (multiple discrete dimensions, per-step validity masks), use `masked_ppo_collect` and `masked_ppo_update` with an `ActionDist`:

```rust,ignore
use rl4burn::{ActionDist, MaskedActorCritic, masked_ppo_collect, masked_ppo_update};

// Action space: [action_type(5), target(10)]
let action_dist = ActionDist::MultiDiscrete(vec![5, 10]);
```

### The MaskedActorCritic trait

```rust,ignore
pub trait MaskedActorCritic<B: Backend> {
    fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>);
    fn log_std(&self) -> Option<Tensor<B, 1>> { None } // continuous only
}
```

If you already have a `DiscreteActorCritic` model, the delegation is trivial:

```rust,ignore
impl<B: Backend> MaskedActorCritic<B> for MyModel<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let out = DiscreteActorCritic::forward(self, obs);
        (out.logits, out.values)
    }
}
```

### Action masking

Environments provide per-step masks via `Env::action_mask()`:

```rust,ignore
fn action_mask(&self) -> Option<Vec<f32>> {
    let mut mask = vec![0.0; 15]; // 5 + 10
    for valid_type in &self.valid_action_types { mask[*valid_type] = 1.0; }
    for valid_target in &self.valid_targets { mask[5 + *valid_target] = 1.0; }
    Some(mask)
}
```

Masked actions are never sampled and receive zero probability during training.

### Env action type

The masked pipeline expects `Env<Action = Vec<f32>>`. For existing discrete envs (`Action = usize`), use `DiscreteEnvAdapter`:

```rust,ignore
use rl4burn::DiscreteEnvAdapter;

let envs: Vec<DiscreteEnvAdapter<CartPole<_>>> = (0..4)
    .map(|i| DiscreteEnvAdapter(CartPole::new(rng)))
    .collect();
```

## Continuous action spaces

For continuous control (e.g. Pendulum, MuJoCo), use `ActionDist::Continuous`. The model outputs means (and optionally log standard deviations) for a diagonal Gaussian distribution.

### ModelOutput mode

The model outputs `[batch, 2 * action_dim]` — first half is means, second half is log_stds:

```rust,ignore
let action_dist = ActionDist::Continuous {
    action_dim: 1,
    log_std_mode: LogStdMode::ModelOutput,
};

// Model outputs [batch, 2]: [mean, log_std]
impl<B: Backend> MaskedActorCritic<B> for MyModel<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let h = self.encoder.forward(obs);
        let logits = self.policy_head.forward(h.clone()); // [batch, 2]
        let values = self.value_head.forward(h).squeeze_dim::<1>(1);
        (logits, values)
    }
}
```

### Separate mode

For state-independent log_std (CleanRL's default), the model outputs only means and provides log_std via a separate learnable parameter:

```rust,ignore
let action_dist = ActionDist::Continuous {
    action_dim: 1,
    log_std_mode: LogStdMode::Separate,
};

impl<B: Backend> MaskedActorCritic<B> for MyModel<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        // logits = [batch, action_dim] (means only)
        ...
    }
    fn log_std(&self) -> Option<Tensor<B, 1>> {
        Some(self.log_std_param.val())
    }
}
```

### log_std clamping

`ActionDist::Continuous` automatically clamps log_std to `[-5, 2]` in all operations (sampling, log-prob, entropy). This prevents numerical instability from excessively large or small standard deviations — a common source of training divergence in continuous RL.

### Continuous PPO tips

- Set `ent_coef: 0.0` — entropy bonus can destabilize continuous policies
- Use `update_epochs: 10` — more gradient steps per rollout helps with continuous
- Longer rollouts (`n_steps: 256+`) improve value estimation for dense-reward tasks
- Environments should accept `Vec<f32>` actions (Pendulum does this natively)

See `examples/ppo_pendulum.rs` for a complete working example.

## Implementation details

- **Per-minibatch advantage normalization**: Advantages are z-normalized within each minibatch, not globally across the full rollout.
- **Clipped value loss**: `max(unclipped, clipped)` using `a + relu(b - a)` to avoid `mask_where` gradient issues in Burn's autodiff.
- **Clipped surrogate**: `min(surr1, surr2)` using `b - relu(b - a)` for the same reason.
- **Global gradient clipping**: Uses `clip_grad_norm` (PyTorch-compatible global norm), not Burn's built-in per-parameter clipping.
- **Minibatch shuffling**: Fisher-Yates shuffle each epoch.
