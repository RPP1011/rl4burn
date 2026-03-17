# Imagination Rollouts

Generate trajectories entirely within the RSSM latent space for actor-critic training.

## API

```rust,ignore
use rl4burn::algo::imagination::{imagine_rollout, lambda_returns};

let trajectory = imagine_rollout(
    &rssm,
    initial_states,
    |h, z| actor_network.forward(h, z),  // actor closure
    15,  // horizon (DreamerV3 default)
);

// trajectory.states: [16] states (initial + 15 imagined)
// trajectory.reward_logits: [15] reward predictions
// trajectory.continue_logits: [15] continue predictions
```

## Lambda-returns

Compute value targets from imagined rewards:

```rust,ignore
let returns = lambda_returns(
    &rewards,     // decoded from reward_logits
    &values,      // critic predictions at each state
    &continues,   // sigmoid(continue_logits)
    0.997,        // gamma
    0.95,         // lambda
);
```

## Stop-gradient rules

During imagination training:
1. **World model**: frozen (no gradients). The actor learns to generate actions that lead to high-value states.
2. **Value targets**: stop-gradiented. The critic trains on fixed targets.
3. **Rewards**: gradients flow through the dynamics model to the actor (the actor is indirectly optimizing for states that the world model predicts will be rewarding).
