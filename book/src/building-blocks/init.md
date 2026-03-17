# Orthogonal Initialization

Orthogonal weight initialization is critical for PPO convergence. It's the default in CleanRL, Stable Baselines3, and OpenAI baselines.

## API

```rust,ignore
pub fn orthogonal_linear<B: Backend>(
    d_in: usize,
    d_out: usize,
    gain: f32,
    device: &B::Device,
    rng: &mut impl Rng,
) -> Linear<B>
```

Creates a `Linear` layer with orthogonal weights and zero bias. Matches PyTorch's `nn.init.orthogonal_`.

## Gain values

| Layer | Gain | Why |
|-------|------|-----|
| Hidden (tanh) | `sqrt(2)` ≈ 1.414 | Preserves gradient norms through tanh |
| Actor output | 0.01 | Near-uniform initial policy (good exploration) |
| Critic output | 1.0 | Reasonable initial value scale |

## Usage

```rust,ignore
use rl4burn::init::orthogonal_linear;
let sqrt2 = std::f32::consts::SQRT_2;

let actor_fc1 = orthogonal_linear(4, 64, sqrt2, &device, &mut rng);
let actor_out = orthogonal_linear(64, 2, 0.01, &device, &mut rng);
let critic_out = orthogonal_linear(64, 1, 1.0, &device, &mut rng);
```

## Why not use Burn's built-in initializers?

Two reasons:

1. **Burn doesn't have orthogonal initialization.** The closest is `XavierUniform`, which has similar scale but lacks the orthogonality property.
2. **Burn initializes bias with the same initializer as weights.** CleanRL always initializes bias to zero. `orthogonal_linear` handles both correctly.

## Implementation

Uses modified Gram-Schmidt orthogonalization on a random Gaussian matrix. Weights are loaded via `Param::from_data` + `load_record` to preserve Burn's autodiff tracking (see [Working with Burn's Autodiff](../burn-compat/autodiff.md)).
