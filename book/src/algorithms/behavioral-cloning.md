# Behavioral Cloning

Train a policy to imitate expert demonstrations via supervised learning. JueWu showed this provides ~64% of final RL performance as initialization.

## API

```rust,ignore
use rl4burn::{bc_loss_discrete, bc_step};

// Single loss computation
let loss = bc_loss_discrete(logits, expert_actions, &device);

// Full training step (forward + backward + optimizer step)
let (model, loss_val) = bc_step(model, &mut optim, obs, expert_actions, lr, &device);
```

## Multi-head actions

For hierarchical action spaces:

```rust,ignore
use rl4burn::bc_loss_multi_head;

let loss = bc_loss_multi_head(logits, expert_actions, &[11, 30, 8], &device);
// head_sizes: action_type(11), target(30), ability(8)
```

## Tips

- The uniform-policy cross-entropy loss should equal `ln(K)` where K is the number of actions. If your initial loss is much higher, something is wrong.
- BC is most useful as RL weight initialization, not as a standalone method. BC policies are brittle — they fail on states not in the training data.
