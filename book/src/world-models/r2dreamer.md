# R2-Dreamer

R2-Dreamer (ICLR 2026) is a computationally efficient world model for RL that achieves strong performance **without decoders or augmentation**. It replaces the standard reconstruction loss with self-supervised representation objectives.

## Key Idea

Standard DreamerV3 trains the encoder via a decoder that reconstructs observations. R2-Dreamer eliminates this bottleneck by using **redundancy reduction** (Barlow Twins loss) to learn representations directly.

## Representation Variants

rl4burn supports all four variants from the paper:

| Variant | Loss | Description |
|---------|------|-------------|
| `Dreamer` | Decoder MSE | Standard DreamerV3 reconstruction baseline |
| `R2Dreamer` | Barlow Twins | Invariance + decorrelation on cross-correlation matrix |
| `InfoNCE` | Contrastive | Positive pair matching with temperature-scaled cosine similarity |
| `DreamerPro` | Prototype | Sinkhorn-Knopp assignment to learned prototypes |

## Usage

```rust
use rl4burn::algo::dreamer::{DreamerConfig, dreamer_world_model_loss, dreamer_actor_critic_loss};
use rl4burn::algo::loss::representation::RepresentationVariant;

// Configure with R2-Dreamer (Barlow Twins)
let config = DreamerConfig {
    rep_variant: RepresentationVariant::R2Dreamer,
    action_dim: 4,
    discrete_actions: true,
    ..DreamerConfig::default()
};
let agent = config.init::<B>(&device);

// Train world model on observed sequences
let (wm_loss, wm_stats) = dreamer_world_model_loss(
    &agent, observations, actions, rewards, continues,
);

// Train actor-critic via imagination
let (actor_loss, critic_loss, ac_stats) = dreamer_actor_critic_loss(
    &agent, initial_states,
);
```

## Architecture

The agent composes existing rl4burn building blocks:

- **RSSM** (`rl4burn_nn::rssm`) — recurrent state-space model with deterministic GRU + stochastic categorical states
- **Imagination rollouts** (`rl4burn_algo::planning::imagination`) — generate trajectories in latent space
- **KL-balanced loss** (`rl4burn_algo::loss::kl_balance`) — train posterior and prior with free bits
- **Symlog + Twohot** (`rl4burn_nn::symlog`) — distributional value prediction
- **Representation losses** (`rl4burn_algo::loss::representation`) — Barlow Twins, InfoNCE, DreamerPro, decoder
- **MLP with RMSNorm** (`rl4burn_nn::mlp`) — prediction heads and actor/critic networks
- **CNN encoder/decoder** (`rl4burn_nn::conv`) — image observation processing

## New Modules

| Module | Crate | Description |
|--------|-------|-------------|
| `mlp` | `rl4burn-nn` | Configurable MLP with RMSNorm or LayerNorm |
| `conv` | `rl4burn-nn` | CNN encoder (images → features) and decoder (features → images) |
| `multi_encoder` | `rl4burn-nn` | Routes mixed observations (images + vectors) |
| `representation` | `rl4burn-algo` | Four self-supervised representation losses |
| `dreamer` | `rl4burn-algo` | DreamerAgent, world model loss, actor-critic loss |

## Example

See `examples/dreamer/` for a complete training loop on CartPole.

## Reference

Nauman & Straffelini, "R2-Dreamer: Redundancy Reduction for Computationally Efficient World Models" (ICLR 2026).
