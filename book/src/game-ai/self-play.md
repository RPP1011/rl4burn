# Self-Play

Train agents by playing against past versions of themselves. The core mechanism for competitive game AI.

## API

```rust,ignore
use rl4burn::algo::self_play::{SelfPlayPool, branch_agent};

let mut pool = SelfPlayPool::new();

// Snapshot current model every N steps
pool.add_snapshot(&model, training_step);

// Get a random past opponent
if let Some(opponent) = pool.sample(&mut rng) {
    // Run game: model vs opponent
}

// Keep only the 50 most recent
pool.retain_recent(50);
```

## How it works

`SelfPlayPool` stores cloned copies of the model at different training stages. Opponents are sampled uniformly. For smarter opponent selection, see [PFSP Matchmaking](./pfsp.md).

## Important: snapshots are deep copies

When you call `add_snapshot`, the model is `.clone()`'d. Mutating the original model afterward does not affect stored snapshots. This is essential — without true deep copies, all "opponents" would have the same weights as the current model.
