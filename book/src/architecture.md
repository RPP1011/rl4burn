# Architecture

rl4burn is organized as a Cargo workspace with five focused crates and one umbrella crate that re-exports everything.

## Workspace layout

```
crates/
  rl4burn-core     — Env trait, spaces, SyncVecEnv, wrappers, Logger
  rl4burn-nn       — Neural network utilities (LSTM, GRU, attention, FiLM, policy traits, init)
  rl4burn-collect  — GAE, V-trace, UPGO, replay buffers, collection patterns
  rl4burn-algo     — PPO, DQN, AC, imitation, multi-agent, planning, losses
  rl4burn-envs     — CartPole, Pendulum, GridWorld
rl4burn/           — Umbrella crate re-exporting everything
examples/          — 15 runnable cookbook examples (see Cookbook)
```

## Dependency DAG

The crates form a clean dependency hierarchy:

```
rl4burn-core          (no internal deps)
    |
    +--- rl4burn-nn       (depends on core)
    |       |
    |       +--- rl4burn-collect  (depends on core, nn)
    |       |       |
    |       |       +--- rl4burn-algo  (depends on core, nn, collect)
    |       |
    +--- rl4burn-envs     (depends on core)
```

Each crate has a single responsibility:

- **rl4burn-core** defines the foundational abstractions: the `Env` trait, observation/action spaces, vectorized environments (`SyncVecEnv`), environment wrappers, and the logging system.
- **rl4burn-nn** provides neural network building blocks: RNN cells (LSTM, GRU, block-diagonal GRU), transformer encoders, attention mechanisms, FiLM conditioning, policy traits (`DiscreteActorCritic`, `MaskedActorCritic`, `QNetwork`), orthogonal initialization, gradient clipping, and polyak updates.
- **rl4burn-collect** handles data collection: GAE, V-trace, UPGO, replay buffers, sequence replay, intrinsic rewards, percentile normalization, and distributed collection patterns (actor-learner, centralized inference, trajectory queues).
- **rl4burn-algo** contains the algorithms: PPO, DQN, actor-critic with V-trace, behavioral cloning, policy distillation, multi-agent infrastructure (self-play, league training, PFSP), planning (MCTS, imagination rollouts), and loss functions.
- **rl4burn-envs** provides built-in environments for testing and examples: CartPole, Pendulum, and GridWorld.

## The umbrella crate

Most users should depend only on `rl4burn` in their `Cargo.toml`. The umbrella crate re-exports the full public API at the top level:

```rust,ignore
// All of these work — no intermediate module paths needed:
use rl4burn::SyncVecEnv;
use rl4burn::PpoConfig;
use rl4burn::{ppo_collect, ppo_update};
use rl4burn::{DiscreteActorCritic, DiscreteAcOutput};
use rl4burn::{Logger, PrintLogger};
use rl4burn::ReplayBuffer;
```

If you need access to the sub-crate modules directly, they are also available:

```rust,ignore
use rl4burn::core;    // rl4burn_core
use rl4burn::nn;      // rl4burn_nn
use rl4burn::collect; // rl4burn_collect
use rl4burn::algo;    // rl4burn_algo
use rl4burn::envs;    // rl4burn_envs
```

## When to depend on individual crates

For most projects, the umbrella `rl4burn` crate is the right choice. You might depend on individual crates if:

- You only need environments (`rl4burn-core` + `rl4burn-envs`) and want minimal compile times.
- You are building a custom algorithm and only need the collection primitives (`rl4burn-collect`).
- You are writing a library that extends rl4burn and want to minimize your dependency footprint.
