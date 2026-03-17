# Distributed Training

Abstractions for multi-GPU/multi-machine gradient synchronization.

## The GradientSync Trait

```rust,ignore
use rl4burn::algo::distributed::{GradientSync, ReduceStrategy};

pub trait GradientSync {
    fn all_reduce_f32(&self, values: &[f32], strategy: ReduceStrategy) -> Vec<f32>;
    fn rank(&self) -> usize;
    fn world_size(&self) -> usize;
    fn barrier(&self);
}
```

## Local Development

Use `LocalSync` for single-machine development. All operations are no-ops.

```rust,ignore
use rl4burn::algo::distributed::LocalSync;
let sync = LocalSync;
assert_eq!(sync.world_size(), 1);
```

## Custom Implementations

Implement `GradientSync` for your cluster's communication library (MPI, NCCL, gRPC, etc.):

```rust,ignore
struct MpiSync { /* ... */ }

impl GradientSync for MpiSync {
    fn all_reduce_f32(&self, values: &[f32], strategy: ReduceStrategy) -> Vec<f32> {
        // Call MPI_Allreduce
    }
    // ...
}
```

At scale (SCC: ~1000 envs per agent, HoK: 320 GPUs), ring all-reduce is the standard choice for gradient averaging.
