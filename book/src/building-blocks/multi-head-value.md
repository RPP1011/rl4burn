# Multi-Head Value Decomposition

Decompose value estimation into N heads, each tracking a different reward component. Used by JueWu (Honor of Kings) with 5 heads: farming, KDA, damage, pushing, and winning.

## API

```rust,ignore
use rl4burn::{MultiHeadValueConfig, multi_head_gae, multi_head_value_loss};

let config = MultiHeadValueConfig::new(5, 0.99, 0.95)
    .with_weights(vec![0.1, 0.2, 0.2, 0.2, 0.3]);

let result = multi_head_gae(
    &per_head_rewards,    // [5][T]
    &per_head_values,     // [5][T]
    &dones,               // [T]
    &per_head_last_values, // [5]
    &config,
);

// result.combined_advantages: [T] — weighted sum across heads
// result.per_head_returns: [5][T] — targets for each value head
```

## Why decompose?

With a single value function, the agent knows *how well* it's doing but not *why*. Multi-head decomposition provides credit assignment: "I'm farming well but my pushing is weak."

Each head can have its own discount factor — short-term heads (damage) use lower gamma, long-term heads (winning) use higher gamma.

## Per-head value loss

```rust,ignore
let losses = multi_head_value_loss(&predictions, &targets);
// losses: [5] — MSE per head
let total_loss: f32 = losses.iter().sum();
```
