# GAE (Generalized Advantage Estimation)

GAE (Schulman et al., 2015) computes advantages that smoothly interpolate between high-bias/low-variance (TD) and low-bias/high-variance (Monte Carlo) estimates.

## API

```rust,ignore
pub fn gae(
    rewards: &[f32],
    values: &[f32],
    dones: &[bool],
    last_value: f32,
    gamma: f32,
    lambda: f32,
) -> (Vec<f32>, Vec<f32>)  // (advantages, returns)
```

Pure f32 computation — no tensors, no backend dependency.

## How it works

For each timestep t, GAE computes:

- TD error: `δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)`
- Advantage: `A_t = Σ_{l=0}^{T-t-1} (γλ)^l * δ_{t+l}`

The `lambda` parameter controls the bias-variance tradeoff:
- `λ = 0`: TD(0) — just the one-step TD error. Low variance, high bias.
- `λ = 1`: Monte Carlo — full discounted return minus baseline. Low bias, high variance.
- `λ = 0.95`: Standard default. Good tradeoff for most tasks.

Returns are computed as `returns = advantages + values`.

## Done handling

When `dones[t]` is true, the next state is from a new episode. GAE correctly zeroes out both the bootstrap value and the accumulated advantage at episode boundaries.

## Usage

GAE is called internally by `ppo_collect`. You only need it directly if building a custom algorithm:

```rust,ignore
use rl4burn::gae;

let rewards = vec![1.0, 1.0, 1.0, 0.0];
let values = vec![5.0, 4.0, 3.0, 2.0];
let dones = vec![false, false, false, true];
let (advantages, returns) = gae::gae(&rewards, &values, &dones, 0.0, 0.99, 0.95);
```
