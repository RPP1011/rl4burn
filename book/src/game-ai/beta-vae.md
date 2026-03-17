# Beta-VAE Opponent Modeling

ROA-Star's approach: train a frozen encoder to predict opponent behavior behind fog of war, then use the latent embedding as extra context for all agents.

## API

```rust,ignore
use rl4burn::nn::vae::{BetaVae, BetaVaeConfig};

let vae = BetaVaeConfig::new(obs_dim)
    .with_latent_dim(32)
    .with_beta(4.0)
    .init(&device);

// Training
let output = vae.forward(opponent_features);
let loss = vae.loss(opponent_features, &output);

// Inference: extract strategy embedding
let z = vae.strategy_embedding(opponent_features);
// z: [batch, 32] — feed this as extra context to the policy
```

## Why beta-VAE?

A standard VAE often ignores the latent space (posterior collapse). Higher beta forces the model to use the latent space, producing more disentangled and interpretable strategy embeddings.

## Scouting reward

The entropy of the opponent model's predictions can be used as an intrinsic reward: the agent is rewarded for actions that reduce uncertainty about the opponent.

```rust,ignore
use rl4burn::collect::intrinsic::EntropyReductionReward;
```
