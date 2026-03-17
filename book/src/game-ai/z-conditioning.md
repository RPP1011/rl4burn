# Goal-Conditioned RL (z-Conditioning)

Condition policies on strategy descriptors to enable rapid specialization. Used by ROA-Star and SCC for exploiter training.

## API

```rust,ignore
use rl4burn::algo::z_conditioning::{ZConditioning, ZConditioningConfig, z_reward};

let z_mod = ZConditioningConfig::new(16, obs_dim).init(&device);
// z_dim=16 (strategy embedding), obs_dim from environment

let conditioned_obs = z_mod.forward(obs, z);
// conditioned_obs: [batch, obs_dim + 64] — ready for policy network

// Pseudo-reward for following target strategy
let reward = z_reward(&observed_stats, &target_z);
// negative L2 distance: closer to target = higher reward
```

## What is z?

A low-dimensional vector describing a play style, computed from human replay statistics. Examples:
- Aggressive: high damage, low farming
- Defensive: low damage, high survival
- Rush: high early-game activity

By conditioning on different z vectors, the same policy can exhibit different strategies.
