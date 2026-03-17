# Percentile Return Normalization

DreamerV3-style advantage normalization using EMA-smoothed percentiles. More robust than per-minibatch normalization for sparse or heterogeneous reward scales.

## API

```rust,ignore
use rl4burn::PercentileNormalizer;

let mut normalizer = PercentileNormalizer::new();
// default: 5th-95th percentile, EMA decay 0.99

// Update with observed returns
normalizer.update(&returns);

// Normalize advantages
let normalized = normalizer.normalize(&advantages);
// Divides by max(1.0, P95 - P5)

// Or combine both steps:
let normalized = normalizer.update_and_normalize(&returns, &advantages);
```

## The max(1, ...) floor

The critical detail: when the percentile range is less than 1.0 (sparse rewards, all-zero returns), the scale is clamped to 1.0. Without this, you'd amplify noise.

## Customization

```rust,ignore
let normalizer = PercentileNormalizer::with_percentiles(0.1, 0.9)
    .with_decay(0.999);
```
