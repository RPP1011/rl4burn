# Environment Wrappers

Wrappers transform an environment's observations, rewards, or tracking without modifying the environment itself. They implement `Env` and wrap an inner `Env`.

## EpisodeStats

Tracks cumulative episode reward and length. Updated when episodes complete.

```rust,ignore
use rl4burn::wrapper::EpisodeStats;

let mut env = EpisodeStats::new(CartPole::new(rng));
env.reset();

loop {
    let step = env.step(action);
    if step.done() {
        println!("Episode return: {}", env.last_episode_reward.unwrap());
        println!("Episode length: {}", env.last_episode_length.unwrap());
        break;
    }
}
```

## RewardClip

Clips rewards to `[-limit, limit]`. Useful for environments with large or unbounded rewards.

```rust,ignore
use rl4burn::wrapper::RewardClip;

let env = RewardClip::new(my_env, 1.0); // rewards clipped to [-1, 1]
```

## NormalizeObservation

Normalizes observations to zero mean, unit variance using Welford's online algorithm. Observations are also clipped to `[-clip, clip]`.

```rust,ignore
use rl4burn::wrapper::NormalizeObservation;

let env = NormalizeObservation::new(my_env, 10.0).unwrap(); // clip normalized obs to [-10, 10]
```

Requires the environment to have `Observation = Vec<f32>` and a `Box` observation space.

## Composing wrappers

Wrappers compose naturally:

```rust,ignore
let env = EpisodeStats::new(
    RewardClip::new(
        NormalizeObservation::new(my_env, 10.0).unwrap(),
        1.0
    )
);
```
