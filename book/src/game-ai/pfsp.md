# PFSP Matchmaking

Prioritized Fictitious Self-Play samples harder opponents more frequently. The opponent you lose to most often is the one you practice against most.

## API

```rust,ignore
use rl4burn::{PfspMatchmaking, PfspConfig};

let mut mm = PfspMatchmaking::new(PfspConfig {
    power: 1.0,    // higher = more focus on hard opponents
    min_prob: 0.01, // every opponent has at least 1% chance
});

mm.add_opponent(0);
mm.add_opponent(1);
mm.add_opponent(2);

// Record results
mm.record_result(0, true, false);   // beat opponent 0
mm.record_result(1, false, false);  // lost to opponent 1

// Sample: opponent 1 (harder) is sampled more often
let opponent = mm.sample_opponent(&mut rng);
```

## Weighting formula

Selection probability is proportional to `(1 - win_rate) ^ power`:
- Win rate 90% → weight 0.1
- Win rate 50% → weight 0.5
- Win rate 10% → weight 0.9

Higher `power` makes the distribution more extreme.
