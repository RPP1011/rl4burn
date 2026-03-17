# V-trace

V-trace (Espeholt et al., 2018) is an off-policy correction algorithm used in IMPALA. It computes value targets and policy gradient advantages from trajectories collected by a potentially stale behavior policy.

## API

```rust,ignore
pub fn vtrace_targets(
    log_rhos: &[f32],     // log importance ratios log(π/μ)
    discounts: &[f32],    // per-step γ (can vary for terminal steps)
    rewards: &[f32],
    values: &[f32],       // V(s_t) from critic
    bootstrap: f32,       // V(s_T) for the last state
    clip_rho: f32,        // importance weight clipping (typically 1.0)
    clip_c: f32,          // trace accumulation clipping (typically 1.0)
) -> (Vec<f32>, Vec<f32>)  // (value_targets, advantages)
```

Pure f32 computation. Contract annotations enforce preconditions (non-empty inputs, matching lengths, positive clip thresholds).

## When to use V-trace

V-trace is for **actor-learner architectures** (like IMPALA) where the acting policy may be several updates behind the learning policy. For standard on-policy PPO, use GAE instead.

## Key parameters

- `clip_rho` (ρ̄): Clips importance weights for value targets. Higher = lower bias but higher variance.
- `clip_c` (c̄): Clips importance weights for trace accumulation. Controls how far back off-policy corrections propagate.
- Both typically set to 1.0.
