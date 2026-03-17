# DreamerV3 Overview

DreamerV3 learns a model of the world, then trains a policy entirely inside imagined trajectories. It's architecturally different from the model-free papers (AlphaStar, JueWu) but its sample efficiency could be transformative for fast simulations.

## The DreamerV3 training loop

```
repeat:
    1. Collect experience in the real environment
    2. Store in sequence replay buffer
    3. Sample sequences, train the world model (RSSM)
    4. Imagine trajectories from the world model
    5. Train actor-critic on imagined data
```

Steps 4-5 are "free" — no environment interaction needed.

## rl4burn modules for DreamerV3

| Component | Module | Page |
|-----------|--------|------|
| World model | `Rssm` | [RSSM](./rssm.md) |
| Imagination | `imagine_rollout` | [Imagination](./imagination.md) |
| Value targets | `lambda_returns` | [Imagination](./imagination.md) |
| Replay | `SequenceReplayBuffer` | [Sequence Replay](../building-blocks/sequence-replay.md) |
| Transforms | `symlog`, `TwohotEncoder` | [Symlog](../nn/symlog.md) |
| KL training | `kl_balanced_loss` | [KL Balance](../nn/kl-balance.md) |
| Normalization | `PercentileNormalizer` | [Percentile](../building-blocks/percentile-normalize.md) |
| Block GRU | `BlockGruCell` | [RNN](../nn/rnn.md) |
