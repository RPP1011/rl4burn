# UPGO (Self-Imitation Learning)

UPGO (Upgoing Policy Gradient) reinforces only trajectories where the agent performed better than expected. Used by ROA-Star alongside V-trace.

## API

```rust,ignore
use rl4burn::upgo_advantages;

let advantages = upgo_advantages(&rewards, &values, &dones, last_value, gamma);
```

## How it works

At each timestep, UPGO checks if the one-step TD error is positive (did better than the value predicted):
- **Positive TD**: Propagate the actual return backward (learn from this)
- **Negative TD**: Truncate to the value estimate (ignore this)

This creates a self-imitation effect: the agent only reinforces actions that led to above-average outcomes.

## When to use

UPGO is complementary to V-trace, not a replacement. ROA-Star uses both:
- V-trace for stable off-policy value targets
- UPGO for the policy gradient (only reinforce good trajectories)
