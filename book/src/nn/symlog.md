# Symlog and Twohot Encoding

DreamerV3's solution for scale-free predictions. Symlog compresses large values; twohot turns regression into classification.

## Symlog / Symexp

```rust,ignore
use rl4burn::{symlog, symexp};

let compressed = symlog(values);   // sign(x) * ln(|x| + 1)
let recovered = symexp(compressed); // sign(x) * (exp(|x|) - 1)
// Round-trip: symexp(symlog(x)) ≈ x
```

Key properties:
- `symlog(0) = 0`
- `symlog(1000) ≈ 6.9` (massive compression)
- `symlog(-x) = -symlog(x)` (symmetric)
- Monotonically increasing

## Twohot Encoder

Encodes scalar values as soft distributions over 255 bins in symlog space.

```rust,ignore
use rl4burn::TwohotEncoder;

let encoder = TwohotEncoder::new(); // 255 bins, [-20, 20]

// Encode: scalar → distribution
let targets = encoder.encode(values, &device);  // [batch, 255]

// Decode: distribution → scalar
let values = encoder.decode(probs, &device);  // [batch]

// Loss: cross-entropy against twohot targets
let loss = encoder.loss(logits, values, &device);  // [1]
```

## Why this matters

Without symlog+twohot, you need to tune learning rates per domain. A reward of 1000 produces 1000x larger gradients than a reward of 1. Symlog compresses this to ~7x. Twohot converts regression to classification, further stabilizing gradients.
