# Policy Distillation

Train a student network to match a teacher's behavior. Used in CSPL (Phase 2) to merge multiple specialist teachers into one generalist.

## API

```rust,ignore
use rl4burn::algo::distillation::{distillation_loss, DistillationConfig};

let config = DistillationConfig {
    temperature: 2.0,
    soft_weight: 1.0,
    hard_weight: 0.0,
    t_squared_scaling: true,
};

let loss = distillation_loss(teacher_logits, student_logits, &config);
```

## Temperature

Higher temperature produces softer probability distributions. The student learns more from the relative ordering of actions, not just the best one.

- T=1: standard softmax (peaked)
- T=5: much softer (exposes teacher's "second choice" preferences)

## T-squared scaling

Hinton et al. recommend scaling the soft-target loss by T-squared. Without this, gradients from soft targets are 1/T-squared too small.

## Value distillation

```rust,ignore
use rl4burn::algo::distillation::value_distillation_loss;
let vloss = value_distillation_loss(teacher_values, student_values);
```
