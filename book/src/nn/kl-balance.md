# KL Balancing with Free Bits

DreamerV3's method for training the RSSM world model without the latent space collapsing.

## The problem

The RSSM has an encoder (posterior: what actually happened) and a dynamics predictor (prior: what the model predicts). They're trained with KL divergence, but:
- If KL goes to zero: the latent space collapses (useless)
- If KL grows unchecked: the world model ignores observations

## The solution

Split the KL loss into two terms with different stop-gradients:

| Term | Trains | Stop-gradient on | Weight |
|------|--------|-----------------|--------|
| Dynamics loss | Prior (predictor) | Posterior | 0.5 |
| Representation loss | Posterior (encoder) | Prior | 0.1 |

Plus **free bits**: ignore KL below 1 nat (don't waste capacity eliminating tiny differences).

## API

```rust,ignore
use rl4burn::{kl_balanced_loss, KlBalanceConfig};

let config = KlBalanceConfig::default();
// dyn_weight: 0.5, rep_weight: 0.1, free_bits: 1.0

let loss = kl_balanced_loss(posterior_logits, prior_logits, &config);
```

For RSSM's 32x32 grouped categoricals:

```rust,ignore
use rl4burn::kl_balanced_loss_groups;

// posterior_logits: [batch, 32, 32]
let loss = kl_balanced_loss_groups(posterior_logits, prior_logits, &config);
```

## Standalone KL

```rust,ignore
use rl4burn::{categorical_kl, categorical_kl_groups};

let kl = categorical_kl(p_logits, q_logits);  // [batch]
```
