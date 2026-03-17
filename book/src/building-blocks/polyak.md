# Polyak Updates

Polyak (soft) target network updates interpolate between a source model's weights and a target model's weights:

```
target = τ * source + (1 - τ) * target
```

## API

```rust,ignore
pub fn polyak_update<B: Backend, M: Module<B>>(
    target: M,
    source: &M,
    tau: f32,
) -> M
```

- `tau = 1.0`: Hard copy (replace target with source entirely).
- `tau = 0.005`: Soft update (slowly track source). Standard for SAC/TD3.
- `tau = 0.0`: No-op (target unchanged).

## Usage

```rust,ignore
use rl4burn::polyak::polyak_update;

// Hard target update (DQN-style, every N steps)
if step % 250 == 0 {
    target = polyak_update(target, &online, 1.0);
}

// Soft target update (SAC/TD3-style, every step)
target = polyak_update(target, &online, 0.005);
```

## How it works

Uses Burn's `ModuleVisitor` to collect all parameter tensors from the source model, then `ModuleMapper` to interpolate each target parameter in place. Works with any `Module<B>` — nested modules, custom architectures, any number of layers.
