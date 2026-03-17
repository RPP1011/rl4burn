# Loss Functions

Backend-generic loss functions for RL training. All return `Tensor<B, 1>` with shape `[1]` (scalar), compatible with `.backward()`.

## Value loss (Huber)

```rust,ignore
pub fn value_loss<B: Backend>(pred: Tensor<B, 1>, target: Tensor<B, 1>) -> Tensor<B, 1>
```

Smooth L1 (Huber) loss with δ=1.0. Quadratic for small errors, linear for large errors. Prevents outlier targets from dominating the value head update.

## Discrete policy loss (REINFORCE)

```rust,ignore
pub fn policy_loss_discrete<B: Backend>(
    logits: Tensor<B, 2>,       // [batch, n_actions]
    actions: Tensor<B, 2, Int>, // [batch, 1] action indices
    mask: Tensor<B, 2>,         // [batch, n_actions] valid=1.0
    advantage: Tensor<B, 1>,    // [batch]
) -> Tensor<B, 1>
```

Standard REINFORCE: `-mean(advantage * log_prob(action))`. Supports action masking for environments with invalid actions.

## Continuous policy loss

```rust,ignore
pub fn policy_loss_continuous<B: Backend>(
    pred: Tensor<B, 2>,      // [batch, action_dim]
    target: Tensor<B, 2>,    // [batch, action_dim]
    advantage: Tensor<B, 1>, // [batch]
) -> Tensor<B, 1>
```

Advantage-weighted regression for deterministic continuous actions. Only positive advantages contribute gradient (negative advantage + MSE is degenerate).

## Note

These loss functions are standalone building blocks. PPO and DQN implement their own loss computation internally (clipped surrogate for PPO, Bellman MSE for DQN). Use these when building custom algorithms.
