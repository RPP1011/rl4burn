# DreamerV3

## The 30-second version

DreamerV3 (Hafner et al., Nature 2025) learns a *model of the world* and then trains a policy entirely inside imagined trajectories. It works across wildly different domains (Atari, robotic control, Minecraft) with **zero hyperparameter tuning**.

The secret: symlog transforms that make gradient magnitudes independent of reward scale.

## What's a world model?

Most RL algorithms learn by trial and error in the real environment. World models flip this:

1. **Play the game** a bit, store transitions
2. **Train a model** to predict what happens next (the "world model")
3. **Imagine** thousands of trajectories inside the model
4. **Train the policy** on imagined data

Step 3 is free — no environment interaction needed. This makes DreamerV3 extremely sample-efficient.

## The RSSM (how the world model works)

The RSSM (Recurrent State-Space Model) has 5 networks:

| Component | Input | Output | Purpose |
|-----------|-------|--------|---------|
| **Sequence model** (GRU) | h_{t-1}, z_{t-1}, a_{t-1} | h_t | Deterministic memory |
| **Encoder** (posterior) | h_t, observation | z_t | What actually happened |
| **Dynamics** (prior) | h_t | z_hat_t | What the model predicts |
| **Reward predictor** | h_t, z_t | reward | Expected reward |
| **Continue predictor** | h_t, z_t | continue prob | Is the episode over? |

The state is `(h_t, z_t)` where h is a deterministic GRU hidden state and z is a stochastic categorical variable (32 groups x 32 classes = 1024 dims).

```rust,ignore
use rl4burn::{Rssm, RssmConfig};

let rssm = RssmConfig::new(obs_dim, action_dim).init(&device);
let state = rssm.initial_state(batch_size, &device);

// Training: use observations
let (next_state, post_logits, prior_logits) = rssm.obs_step(&state, action, obs);

// Imagination: no observations needed
let next_state = rssm.imagine_step(&state, action);
```

See [RSSM](../world-models/rssm.md).

## Symlog: the key to fixed hyperparameters

The biggest problem with RL across domains is reward scale. Atari rewards are 0-1000. Robotic rewards are -1 to 0. Without normalization, you need different learning rates for each.

DreamerV3 solves this with **symlog**: `symlog(x) = sign(x) * ln(|x| + 1)`. This compresses large values and keeps small values linear. Combined with **twohot encoding** (distributional predictions), gradient magnitudes become independent of value scale.

```rust,ignore
use rl4burn::{symlog, symexp, TwohotEncoder};

let encoder = TwohotEncoder::new();  // 255 bins, [-20, 20] symlog space
let targets = encoder.encode(values, &device);   // [batch, 255]
let loss = encoder.loss(logits, values, &device); // cross-entropy
let decoded = encoder.decode(softmax(logits, 1), &device);  // back to scalars
```

See [Symlog and Twohot Encoding](../nn/symlog.md).

## KL balancing: training the world model

The RSSM is trained with two KL losses:
- **Dynamics loss**: Make the prior match the posterior (train the predictor)
- **Representation loss**: Make the posterior predictable (don't be too complex)

Each has a stop-gradient on one side, plus a "free bits" threshold (ignore KL below 1 nat).

```rust,ignore
use rl4burn::{kl_balanced_loss, KlBalanceConfig};

let config = KlBalanceConfig {
    dyn_weight: 0.5,
    rep_weight: 0.1,
    free_bits: 1.0,
};
let loss = kl_balanced_loss(posterior_logits, prior_logits, &config);
```

See [KL Balancing with Free Bits](../nn/kl-balance.md).

## Imagination rollouts

Once the world model is trained, generate trajectories purely in latent space:

```rust,ignore
use rl4burn::algo::imagination::{imagine_rollout, lambda_returns};

let trajectory = imagine_rollout(&rssm, initial_states, |h, z| actor(h, z), 15);
// trajectory.states: 16 states (initial + 15 steps)
// trajectory.reward_logits: 15 predicted reward distributions
```

Compute lambda-returns on the imagined rewards, then train actor and critic on these imagined trajectories. The world model parameters are frozen during actor-critic training.

See [Imagination Rollouts](../world-models/imagination.md).

## Sequence replay buffer

DreamerV3 samples contiguous sequences (T=64) from a FIFO buffer, never crossing episode boundaries.

```rust,ignore
use rl4burn::{SequenceReplayBuffer, SequenceStep};
let mut buffer = SequenceReplayBuffer::new(1_000_000, 64);
```

See [Sequence Replay Buffer](../building-blocks/sequence-replay.md).

## Percentile return normalization

Instead of per-minibatch normalization, DreamerV3 tracks the 5th-95th percentile range of returns with an EMA and divides by `max(1, range)`. The floor of 1 prevents amplifying noise.

```rust,ignore
use rl4burn::PercentileNormalizer;
let mut normalizer = PercentileNormalizer::new();
normalizer.update(&returns);
let normalized = normalizer.normalize(&advantages);
```

See [Percentile Return Normalization](../building-blocks/percentile-normalize.md).

## Further reading

- [DreamerV3 paper](https://arxiv.org/abs/2301.04104) (Nature, 2025)
- [DreamerV2 paper](https://arxiv.org/abs/2010.02193) (ICLR, 2021)
- [Original Dreamer paper](https://arxiv.org/abs/1912.01603) (ICLR, 2020)
