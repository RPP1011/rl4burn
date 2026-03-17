# rl4burn

**Reinforcement learning algorithms for the [Burn](https://burn.dev) ML framework.**

rl4burn lets you write RL algorithms once with `B: AutodiffBackend` and run them on any Burn backend — WGPU, CUDA, NdArray, or LibTorch. No per-backend reimplementation.

## What's included

| Algorithm | Type | Status |
|-----------|------|--------|
| **PPO** | On-policy, actor-critic | Solves CartPole in <1s |
| **DQN** | Off-policy, value-based | Solves CartPole in ~9s |

Plus the building blocks to implement your own algorithms: GAE, V-trace, replay buffers, polyak updates, loss functions, orthogonal initialization, and global gradient clipping.

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
