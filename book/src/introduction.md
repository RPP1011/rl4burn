# rl4burn

**Reinforcement learning algorithms for the [Burn](https://burn.dev) ML framework.**

rl4burn lets you write RL algorithms once with `B: AutodiffBackend` and run them on any Burn backend — WGPU, CUDA, NdArray, or LibTorch. No per-backend reimplementation.

## What's included

### Algorithms

| Algorithm | Type | Status |
|-----------|------|--------|
| **PPO** | On-policy, actor-critic | Solves CartPole in <1s |
| **Dual-Clip PPO** | PPO for distributed training | JueWu/HoK-style |
| **DQN** | Off-policy, value-based | Solves CartPole in ~9s |
| **Behavioral Cloning** | Supervised imitation | Cross-entropy on demonstrations |
| **Policy Distillation** | Teacher-student transfer | Temperature-scaled KL |

### Neural Network Modules

LSTM, GRU, and block-diagonal GRU cells. Transformer encoder blocks. Multi-head attention, target attention, attention pooling, and pointer networks. FiLM conditioning. Auto-regressive action distributions.

### World Models (DreamerV3)

RSSM world model with imagination rollouts. Symlog/twohot distributional encoding. KL balancing with free bits. Sequence replay buffer. Percentile return normalization.

### Game AI Infrastructure

Self-play with opponent pools. League training with agent roles (AlphaStar-style). PFSP matchmaking. Multi-agent shared-weight training. Privileged critic. Goal-conditioned RL. Agent branching. MCTS for drafting. Beta-VAE opponent modeling. Curriculum self-play learning (CSPL).

### Building Blocks

GAE, V-trace, UPGO, replay buffers, multi-head value decomposition, intrinsic rewards, polyak updates, loss functions, orthogonal initialization, global gradient clipping, and logging.

## Workspace architecture

rl4burn is organized as a Cargo workspace of five focused crates (`rl4burn-core`, `rl4burn-nn`, `rl4burn-collect`, `rl4burn-algo`, `rl4burn-envs`) plus an umbrella `rl4burn` crate that re-exports the full API. Users depend only on `rl4burn` — no need to manage individual crate dependencies. See the [Architecture](./architecture.md) chapter for details.

## Cookbook

The repository includes 15 runnable examples organized into five tiers:

1. **Fundamentals** — quickstart, annotated PPO, config-driven training
2. **Environment Variations** — custom environments, continuous actions, multi-discrete actions
3. **Techniques** — action masking, reward shaping, LSTM policies
4. **Multi-Agent & Game AI** — self-play, multi-agent, curriculum learning
5. **Production** — diagnostics, hyperparameter tuning, policy deployment

Run any example with `cargo run -p <name> --release`. See the [Cookbook](./cookbook.md) for the full list and a decision guide for choosing the right algorithm.

## Why Burn?

Burn's `Backend` trait lets you write generic code:

```rust,ignore
fn train<B: AutodiffBackend>(model: MyModel<B>, device: &B::Device) {
    // This works on WGPU, CUDA, NdArray, LibTorch...
}
```

For RL, this means:
- **Train on GPU** with `Autodiff<Wgpu>` or `Autodiff<LibTorch>`
- **Deploy on edge** with `NdArray` (no GPU needed, `no_std` capable)
- **Run in the browser** with WASM via the WGPU backend

No other Rust RL library achieves this level of backend portability.

## Design philosophy

- **You own the training loop.** `ppo_collect` and `ppo_update` are functions, not a framework. Compose them however you want.
- **Minimal API surface.** Each algorithm is ~200 lines. Read the source — it's meant to be understood.
- **Correctness first.** Integration tests train both PPO and DQN on CartPole to convergence. Contract annotations enforce preconditions on core functions.
- **Match reference implementations.** PPO defaults match [CleanRL's ppo.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py). When Burn's behavior differs from PyTorch (gradient clipping, parameter initialization), we provide compatible alternatives.
