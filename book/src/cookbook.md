# Cookbook

rl4burn ships with 15 runnable examples in the `examples/` directory, organized into five tiers of increasing complexity. Each example is a standalone Cargo package that you can run with `cargo run -p <name> --release`.

## Tier 1: Fundamentals

| Example | Command | Description |
|---------|---------|-------------|
| **quickstart** | `cargo run -p quickstart --release` | Minimal PPO on CartPole — the "hello world" of RL |
| **ppo-annotated** | `cargo run -p ppo-annotated --release` | Same as quickstart but with detailed comments explaining every line |
| **config-driven** | `cargo run -p config-driven --release` | Load hyperparameters from a TOML file instead of hardcoding them |

## Tier 2: Environment Variations

| Example | Command | Description |
|---------|---------|-------------|
| **custom-env** | `cargo run -p custom-env --release` | Implement the `Env` trait for your own environment |
| **ppo-continuous** | `cargo run -p ppo-continuous --release` | PPO with continuous actions on Pendulum |
| **ppo-multi-discrete** | `cargo run -p ppo-multi-discrete --release` | PPO with multi-discrete action spaces |

## Tier 3: Techniques

| Example | Command | Description |
|---------|---------|-------------|
| **action-masking** | `cargo run -p action-masking --release` | Invalid action masking with the masked PPO pipeline |
| **reward-shaping** | `cargo run -p reward-shaping --release` | Intrinsic rewards and reward shaping wrappers |
| **lstm-policy** | `cargo run -p lstm-policy --release` | Recurrent policy for partially observable environments |

## Tier 4: Multi-Agent & Game AI

| Example | Command | Description |
|---------|---------|-------------|
| **self-play** | `cargo run -p self-play --release` | Self-play training with an opponent pool |
| **multi-agent** | `cargo run -p multi-agent --release` | Shared-weight multi-agent training |
| **curriculum** | `cargo run -p curriculum --release` | Curriculum self-play learning (CSPL) |

## Tier 5: Production

| Example | Command | Description |
|---------|---------|-------------|
| **diagnostics** | `cargo run -p diagnostics --release` | TensorBoard logging, video recording, and training diagnostics |
| **hyperparameter-tuning** | `cargo run -p hyperparameter-tuning --release` | Systematic hyperparameter sweeps |
| **deploy-policy** | `cargo run -p deploy-policy --release` | Export a trained policy for inference on a different backend |

## Which algorithm should I use?

Use this decision guide to pick the right starting point:

| Scenario | Recommended algorithm | Start from example |
|----------|----------------------|-------------------|
| **Discrete actions** (e.g., CartPole, Atari) | PPO or DQN | `quickstart` |
| **Continuous actions** (e.g., Pendulum, MuJoCo) | PPO with Gaussian policy | `ppo-continuous` |
| **Multi-discrete actions** (e.g., RTS games) | PPO with multi-head | `ppo-multi-discrete` |
| **Invalid actions vary per step** | Masked PPO | `action-masking` |
| **Competitive game** (1v1 or teams) | Self-play PPO | `self-play` |
| **Partial observability** | LSTM policy + PPO | `lstm-policy` |
| **Multiple cooperating agents** | Shared-weight PPO | `multi-agent` |
| **Large observation space / model-based** | DreamerV3 (future) | — |

When in doubt, start with PPO (`quickstart`). It is the most versatile algorithm and works well across a wide range of problems. Switch to DQN only if you need off-policy learning or have a small discrete action space where sample efficiency matters.
