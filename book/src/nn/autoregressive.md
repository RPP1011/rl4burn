# Auto-Regressive Action Distributions

For games where actions decompose into sequential decisions: *what action* -> *which target* -> *which ability*. Every competitive game AI paper uses this pattern.

## CompositeDistribution

```rust,ignore
use rl4burn::{CompositeDistribution, ActionHead};

// 3-head action space: action_type(11) -> target(30) -> ability(8)
let dist = CompositeDistribution::from_heads(
    &["action_type", "target", "ability"],
    &[11, 30, 8],
);

// Total logits needed from the model: 11 + 30 + 8 = 49
assert_eq!(dist.total_logits(), 49);
```

## Sampling

Given flat logits from the model (all heads concatenated), sample independently per head:

```rust,ignore
let actions = dist.sample(&logits, mask.as_ref(), &mut rng);
// actions: Vec<Vec<f32>> — [batch][n_heads], integer-valued
```

For fully auto-regressive sampling (where head 2's logits depend on head 1's sample), call the model multiple times and feed actions back.

## Log-probabilities

Joint log-prob is the sum of per-head log-probs:

```rust,ignore
let log_prob = dist.log_prob(logits, &actions, mask.as_ref(), &device);
// log_prob: [batch] — log P(a) = log P(a1) + log P(a2) + log P(a3)
```

## Entropy

Sum of per-head entropies (exact when heads are independent, upper bound otherwise):

```rust,ignore
let entropy = dist.entropy(logits, mask.as_ref());
// entropy: [batch]
```

## With action masking

Pass a flat mask tensor `[batch, total_logits]` where `1.0` = valid, `0.0` = invalid. Masked actions are never sampled and get zero probability.
