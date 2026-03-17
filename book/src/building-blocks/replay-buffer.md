# Replay Buffer

`ReplayBuffer<S, R>` stores transitions for off-policy algorithms like DQN. It's generic over the sample type and a deterministic RNG for reproducible sampling.

## API

```rust,ignore
use rand::SeedableRng;

let mut buffer = ReplayBuffer::new(10_000, rand::rngs::SmallRng::seed_from_u64(42));

buffer.extend(transitions);          // add samples
let batch = buffer.sample(64);       // random references
let batch = buffer.sample_cloned(64); // random clones (for owned data)
let groups = buffer.group_by(|t| t.episode_id); // group by key
```

## Eviction

When the buffer exceeds capacity, the oldest samples are dropped first (FIFO).

## With DQN

```rust,ignore
use rl4burn::dqn::Transition;
use rl4burn::replay::ReplayBuffer;

let mut buffer = ReplayBuffer::new(10_000, rand::rngs::SmallRng::seed_from_u64(42));

// Store transitions
buffer.extend(std::iter::once(Transition {
    obs: obs.clone(),
    action: action as i32,
    reward: result.reward,
    next_obs: result.observation.clone(),
    done: result.done(),
}));

// dqn_update samples from the buffer internally
(online, stats) = dqn_update(online, &target, &mut optim, &mut buffer, &config, &device);
```

## Trajectory grouping

The `group_by` method groups sample indices by an arbitrary key function. Useful for V-trace rescoring where you need to process entire trajectories:

```rust,ignore
let groups = buffer.group_by(|sample| sample.trajectory_id);
for (traj_id, indices) in &groups {
    let trajectory: Vec<_> = indices.iter().map(|&i| &buffer.samples()[i]).collect();
    // rescore this trajectory
}
```
