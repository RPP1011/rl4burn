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
let mut ep_acc = vec![0.0f32; n_envs]; // create once, pass every iteration

let rollout = ppo_collect(..., &mut ep_acc);
for &ret in &rollout.episode_returns {
    println!("completed episode return: {ret}");
}
```

## Implementation details

- **Per-minibatch advantage normalization**: Advantages are z-normalized within each minibatch, not globally across the full rollout.
- **Clipped value loss**: `max(unclipped, clipped)` using `a + relu(b - a)` to avoid `mask_where` gradient issues in Burn's autodiff.
- **Clipped surrogate**: `min(surr1, surr2)` using `b - relu(b - a)` for the same reason.
- **Global gradient clipping**: Uses `clip_grad_norm` (PyTorch-compatible global norm), not Burn's built-in per-parameter clipping.
- **Minibatch shuffling**: Fisher-Yates shuffle each epoch.
