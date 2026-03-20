# Summary

[Introduction](./introduction.md)

# Architecture

- [Workspace Structure](./architecture.md)

# Getting Started

- [Installation](./getting-started/installation.md)
- [Your First Agent: PPO on CartPole](./getting-started/first-agent.md)

# Cookbook

- [Examples Overview](./cookbook.md)

# Core Concepts

- [Environments](./concepts/environments.md)
- [Spaces](./concepts/spaces.md)
- [Vectorized Environments](./concepts/vec-env.md)
- [Environment Wrappers](./concepts/wrappers.md)

# Algorithms

- [PPO (Proximal Policy Optimization)](./algorithms/ppo.md)
- [DQN (Deep Q-Network)](./algorithms/dqn.md)
- [Dual-Clip PPO](./algorithms/dual-clip-ppo.md)
- [Behavioral Cloning](./algorithms/behavioral-cloning.md)
- [Policy Distillation](./algorithms/distillation.md)

# Loss Functions

- [Policy & Value Losses](./building-blocks/losses.md)
- [Multi-Head Value Decomposition](./building-blocks/multi-head-value.md)
- [KL Balancing with Free Bits](./nn/kl-balance.md)

# Building Blocks

- [GAE (Generalized Advantage Estimation)](./building-blocks/gae.md)
- [V-trace](./building-blocks/vtrace.md)
- [UPGO (Self-Imitation Learning)](./building-blocks/upgo.md)
- [Replay Buffer](./building-blocks/replay-buffer.md)
- [Sequence Replay Buffer](./building-blocks/sequence-replay.md)
- [Percentile Return Normalization](./building-blocks/percentile-normalize.md)
- [Intrinsic Rewards](./building-blocks/intrinsic-rewards.md)
- [CSPL (Curriculum Self-Play Learning)](./game-ai/cspl.md)
- [Polyak Updates](./building-blocks/polyak.md)
- [Orthogonal Initialization](./building-blocks/init.md)
- [Global Gradient Clipping](./building-blocks/grad-clip.md)
- [Logging](./building-blocks/logging.md)
- [Saving & Sharing](./building-blocks/saving-sharing.md)

# Neural Network Modules

- [LSTM, GRU, and Block GRU](./nn/rnn.md)
- [Transformer Encoder](./nn/transformer.md)
- [Attention Mechanisms](./nn/attention.md)
- [Auto-Regressive Action Distributions](./nn/autoregressive.md)
- [FiLM Conditioning](./nn/film.md)
- [Symlog and Twohot Encoding](./nn/symlog.md)

# World Models (DreamerV3)

- [DreamerV3 Overview](./world-models/dreamer-overview.md)
- [RSSM (Recurrent State-Space Model)](./world-models/rssm.md)
- [Imagination Rollouts](./world-models/imagination.md)

# Game AI Infrastructure

- [Self-Play](./game-ai/self-play.md)
- [League Training](./game-ai/league.md)
- [PFSP Matchmaking](./game-ai/pfsp.md)
- [Multi-Agent Shared-Weight Training](./game-ai/multi-agent.md)
- [Privileged Critic](./game-ai/privileged-critic.md)
- [Goal-Conditioned RL (z-Conditioning)](./game-ai/z-conditioning.md)
- [Agent Branching](./game-ai/agent-branching.md)
- [MCTS for Drafting](./game-ai/mcts.md)
- [Beta-VAE Opponent Modeling](./game-ai/beta-vae.md)
- [Distributed Training](./game-ai/distributed.md)
- [Cloud GPU Deployment](./game-ai/cloud-deploy.md)

# Paper Guides (ELI5)

- [AlphaStar & ROA-Star](./papers/alphastar.md)
- [SCC (StarCraft Commander)](./papers/scc.md)
- [JueWu & Honor of Kings](./papers/juewu.md)
- [DreamerV3](./papers/dreamerv3.md)

# Burn Compatibility

- [Working with Burn's Autodiff](./burn-compat/autodiff.md)
