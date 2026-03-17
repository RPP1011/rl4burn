# JueWu & Honor of Kings

## The 30-second version

JueWu (Tencent, NeurIPS 2020) is the first AI to beat top professional players in Honor of Kings, a 5v5 MOBA. The key insight: **macro strategy (which lane to go to, when to fight) emerges from micro rewards** — you don't need a separate strategic layer.

## Why MOBAs are different from StarCraft

In a MOBA, you control 1 hero (or 5 with shared weights). The action space is simpler but the strategic depth comes from teamwork, timing, and map control. Games are shorter (~15 minutes) but the reward is extremely sparse (win/lose).

## Key ideas

### Multi-head value decomposition

Instead of one value function, JueWu uses 5:
- Farming (gold/XP)
- KDA (kills/deaths/assists)
- Damage dealt
- Tower pushing
- Win/lose

Each head learns independently with its own discount factor. The combined advantage drives the policy.

```rust,ignore
use rl4burn::{MultiHeadValueConfig, multi_head_gae};

let config = MultiHeadValueConfig::new(5, 0.99, 0.95)
    .with_weights(vec![0.1, 0.2, 0.2, 0.2, 0.3]);  // win/lose weighted highest
```

This helps with credit assignment: the agent knows *why* it's doing well, not just *that* it's doing well.

See [Multi-Head Value Decomposition](../building-blocks/multi-head-value.md).

### Dual-clip PPO

Standard PPO clips the policy ratio to prevent too-large updates. Dual-clip adds a second constraint: when the advantage is negative, the objective can't go below `c * advantage` (c=3). This prevents catastrophic updates in distributed training where trajectories are slightly off-policy.

```rust,ignore
let config = PpoConfig {
    dual_clip_coef: Some(3.0),
    ..Default::default()
};
```

See [Dual-Clip PPO](../algorithms/dual-clip-ppo.md).

### Supervised pre-training matters (a lot)

JueWu-SL (a separate paper) showed that behavioral cloning from top human players provides **64% of the final RL performance**. RL then refines and exceeds human play.

```rust,ignore
use rl4burn::bc_loss_discrete;
```

See [Behavioral Cloning](../algorithms/behavioral-cloning.md).

### Curriculum Self-Play Learning (CSPL)

Training 40+ heroes at once doesn't converge. CSPL breaks it into 3 phases:
1. **Specialist training**: Train small models on fixed team compositions
2. **Distillation**: Merge all specialists into one big model
3. **Generalization**: Continue RL with random compositions

Without CSPL, training fails after 480+ hours. With it, convergence in ~264 hours.

```rust,ignore
use rl4burn::{CsplPipeline, CsplConfig, CsplPhase};
```

See [CSPL](../game-ai/cspl.md).

### Privileged critic

During training, the value function sees *everything* — including enemy positions behind fog of war. The policy only sees what the player would see. This dramatically improves value estimation.

```rust,ignore
use rl4burn::algo::privileged_critic::PrivilegedActorCritic;
```

See [Privileged Critic](../game-ai/privileged-critic.md).

## The architecture in one sentence

Shared-weight policy across 5 heroes → LSTM for temporal memory → multi-head value for credit assignment → dual-clip PPO for stability → CSPL for scaling to many heroes.

## Further reading

- [Honor of Kings paper](https://arxiv.org/abs/2011.12692) (NeurIPS, 2020)
- [JueWu-SL paper](https://ieeexplore.ieee.org/document/9248616) (IEEE TNNLS, 2020)
