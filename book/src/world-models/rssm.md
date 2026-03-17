# RSSM (Recurrent State-Space Model)

The world model at the heart of DreamerV3. Predicts what happens next in a compact latent space.

## State representation

The RSSM state has two parts:
- **h** (deterministic): GRU hidden state, captures long-term memory
- **z** (stochastic): 32 categorical distributions x 32 classes, captures uncertainty

Together they form a 1024+ dimensional state sufficient to reconstruct observations, predict rewards, and determine if episodes continue.

## API

```rust,ignore
use rl4burn::{Rssm, RssmConfig, RssmState};

let config = RssmConfig::new(obs_dim, action_dim);
let rssm = config.init(&device);

// Initial state (all zeros)
let state = rssm.initial_state(batch_size, &device);

// Training: observe → update
let (new_state, posterior_logits, prior_logits) = rssm.obs_step(&state, action, obs);

// Imagination: predict without observing
let new_state = rssm.imagine_step(&state, action);

// Predictions
let reward_logits = rssm.predict_reward(state.h, state.z);   // [batch, 255]
let cont_logits = rssm.predict_continue(state.h, state.z);   // [batch, 1]
```

## Training the RSSM

Train with KL-balanced loss between posterior and prior:

```rust,ignore
use rl4burn::{kl_balanced_loss, KlBalanceConfig, TwohotEncoder};

let kl_loss = kl_balanced_loss(posterior_logits, prior_logits, &KlBalanceConfig::default());
let reward_loss = TwohotEncoder::new().loss(reward_logits, actual_rewards, &device);
let total_loss = kl_loss + reward_loss;
```

## Configuration

```rust,ignore
let config = RssmConfig {
    obs_dim: 64,
    action_dim: 11,
    deterministic_size: 512,   // GRU hidden size
    n_categories: 32,          // stochastic groups
    n_classes: 32,             // classes per group
    hidden_size: 512,          // MLP hidden dim
    n_blocks: 8,               // block GRU blocks (0 = standard GRU)
    unimix: 0.01,              // uniform mixture for categoricals
};
```
