# FiLM Conditioning

FiLM (Feature-wise Linear Modulation) applies a context-dependent affine transform to features. Used by SCC to condition spatial action heads on action type.

## API

```rust,ignore
use rl4burn::{Film, FilmConfig};

let film = FilmConfig::new(32, 128).init(&device);
// context_dim=32, feature_dim=128

let output = film.forward(features, context);
// features: [batch, 128], context: [batch, 32]
// output: [batch, 128]
```

## How it works

```
output = (1 + gamma(context)) * features + beta(context)
```

The `+1` on gamma ensures the layer starts as an identity transform, improving training stability.
