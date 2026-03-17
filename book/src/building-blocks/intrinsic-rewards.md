# Intrinsic Rewards

Exploration bonuses based on internal state. Useful when extrinsic rewards are sparse.

## API

```rust,ignore
use rl4burn::collect::intrinsic::{IntrinsicReward, CountBasedReward, combine_rewards};

let mut explorer = CountBasedReward::new(0.1); // discretization resolution
explorer.update(&obs, action, &next_obs);
let bonus = explorer.reward(&obs, action, &next_obs);
// bonus = 1 / sqrt(visit_count)

let combined = combine_rewards(&extrinsic, &intrinsic, 0.01);
// combined[i] = extrinsic[i] + 0.01 * intrinsic[i]
```

## Count-Based Exploration

Reward = `1 / sqrt(N(s))` where N(s) is how many times the agent has visited a discretized version of state s. Novel states get high reward; familiar states get low reward.

## Entropy-Reduction Reward

ROA-Star's scouting reward: `max(H_{prev} - H_{current}, 0)`. Rewards the agent for reducing uncertainty about the opponent's strategy.

```rust,ignore
use rl4burn::collect::intrinsic::EntropyReductionReward;
let mut scouting = EntropyReductionReward::new();
let reward = scouting.reward_from_entropy(current_entropy);
```

## The IntrinsicReward trait

Implement for custom exploration strategies:

```rust,ignore
pub trait IntrinsicReward {
    type Observation;
    fn reward(&self, obs: &Self::Observation, action: usize, next_obs: &Self::Observation) -> f32;
    fn update(&mut self, obs: &Self::Observation, action: usize, next_obs: &Self::Observation);
}
```
