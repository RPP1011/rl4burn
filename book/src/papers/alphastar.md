# AlphaStar & ROA-Star

## The 30-second version

AlphaStar (DeepMind, 2019) was the first AI to beat a top professional StarCraft II player. ROA-Star (Tencent, NeurIPS 2023) achieves the same level with **4x less compute** by adding opponent modeling and smarter exploiter training.

Both are massive RL systems, but their core ideas decompose into modular building blocks — most of which are in rl4burn.

## What makes StarCraft II hard for RL?

Imagine playing chess, except:
- You can only see part of the board (fog of war)
- Both players move simultaneously
- You control 200 pieces at once
- Each piece has 10+ possible actions
- Games last 20+ minutes (thousands of decisions)

Standard RL algorithms break under this complexity. AlphaStar's solution: decompose the problem.

## Key ideas (and where they are in rl4burn)

### 1. Auto-regressive action space

Instead of choosing from millions of possible joint actions, AlphaStar samples one decision at a time:

> action_type → delay → queue → selected_units → target_unit → target_location

Each head is conditioned on the previous samples. This is exactly what `CompositeDistribution` provides.

```rust,ignore
use rl4burn::CompositeDistribution;

let dist = CompositeDistribution::from_heads(
    &["action_type", "target", "ability"],
    &[11, 30, 8],
);
```

See [Auto-Regressive Action Distributions](../nn/autoregressive.md) for details.

### 2. V-trace for off-policy correction

With thousands of parallel actors, the behavior policy is always slightly stale. V-trace corrects for this. Already in rl4burn as `vtrace_targets`.

See [V-trace](../building-blocks/vtrace.md).

### 3. UPGO (self-imitation learning)

Only learn from experiences where you did better than expected. If the return exceeds the value baseline, reinforce it. Otherwise, ignore it.

```rust,ignore
use rl4burn::upgo_advantages;
let advantages = upgo_advantages(&rewards, &values, &dones, last_value, gamma);
```

See [UPGO](../building-blocks/upgo.md).

### 4. League training with PFSP

Instead of just self-play, AlphaStar trains a *league* of agents:
- **Main agent**: plays against everyone
- **Main exploiter**: specializes in beating the main agent
- **League exploiters**: find weaknesses across the entire pool

Opponents are sampled using PFSP — harder opponents (lower win rate) get sampled more often.

```rust,ignore
use rl4burn::{League, AgentRole, LeagueAgentConfig, PfspMatchmaking};
```

See [League Training](../game-ai/league.md) and [PFSP Matchmaking](../game-ai/pfsp.md).

### 5. ROA-Star's additions

ROA-Star adds two ideas:
- **Beta-VAE opponent modeling**: A frozen encoder predicts what the opponent is doing behind fog of war. The latent embedding is fed to all agents as extra context. See [Beta-VAE Opponent Modeling](../game-ai/beta-vae.md).
- **Goal-conditioned exploiters**: Exploiters are conditioned on strategy descriptors *z*, letting them specialize rapidly. See [Goal-Conditioned RL](../game-ai/z-conditioning.md).

## Further reading

- [AlphaStar paper](https://www.nature.com/articles/s41586-019-1724-z) (Nature, 2019)
- [ROA-Star paper](https://arxiv.org/abs/2312.08826) (NeurIPS, 2023)
