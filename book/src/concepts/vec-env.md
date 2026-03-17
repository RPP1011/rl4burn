# Vectorized Environments

`SyncVecEnv` runs N copies of an environment in lockstep, collecting N transitions per step. This is essential for PPO, which needs batched data from parallel environments.

## Usage

```rust,ignore
use rl4burn::vec_env::SyncVecEnv;
use rl4burn::envs::CartPole;
use rand::SeedableRng;

let envs: Vec<CartPole<_>> = (0..8)
    .map(|i| CartPole::new(rand::rngs::SmallRng::seed_from_u64(i as u64)))
    .collect();
let mut vec_env = SyncVecEnv::new(envs);

// Reset all environments
let observations = vec_env.reset(); // Vec of 8 observations

// Step all environments with one action each
let actions = vec![0, 1, 0, 1, 1, 0, 1, 0];
let steps = vec_env.step(actions); // Vec of 8 Step results
```

## Auto-reset

When an environment reaches a terminal or truncated state, `SyncVecEnv` automatically resets it. The returned observation is the **initial observation of the new episode**, not the terminal observation. This matches Gymnasium's `SyncVectorEnv` behavior.

The reward and done flags in the returned `Step` are from the terminal step — only the observation is replaced.

## When to use SyncVecEnv

| Algorithm | Vectorized? | Why |
|-----------|-------------|-----|
| PPO | Yes (required) | Needs batched rollouts from parallel envs |
| DQN | No (typically) | Single-env stepping with replay buffer |
