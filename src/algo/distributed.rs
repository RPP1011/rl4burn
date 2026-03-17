//! Distributed gradient synchronization abstractions (Issue #30).
//!
//! Provides a [`GradientSync`] trait that implementations back with their
//! transport layer (MPI, NCCL, gRPC, etc.).  The library ships a
//! [`LocalSync`] no-op implementation for single-process development.

// ---------------------------------------------------------------------------
// Reduce strategy
// ---------------------------------------------------------------------------

/// Strategy for distributed gradient aggregation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceStrategy {
    /// Average gradients across workers.
    Mean,
    /// Sum gradients across workers.
    Sum,
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Trait for distributed gradient synchronization.
///
/// Implementations provide the transport layer (MPI, NCCL, gRPC, etc.).
/// The library provides the interface; users implement for their cluster.
pub trait GradientSync {
    /// Synchronize gradient scalars across workers.
    /// Returns the reduced values.
    fn all_reduce_f32(&self, values: &[f32], strategy: ReduceStrategy) -> Vec<f32>;

    /// Get the rank of this worker (0-indexed).
    fn rank(&self) -> usize;

    /// Get total number of workers.
    fn world_size(&self) -> usize;

    /// Barrier: wait for all workers to reach this point.
    fn barrier(&self);
}

// ---------------------------------------------------------------------------
// Local (single-process) implementation
// ---------------------------------------------------------------------------

/// Single-process "distributed" implementation for local development.
/// All operations are no-ops that return the input unchanged.
pub struct LocalSync;

impl GradientSync for LocalSync {
    fn all_reduce_f32(&self, values: &[f32], _strategy: ReduceStrategy) -> Vec<f32> {
        values.to_vec()
    }

    fn rank(&self) -> usize {
        0
    }

    fn world_size(&self) -> usize {
        1
    }

    fn barrier(&self) {}
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Scale gradients by 1/world_size after summing (implements mean reduction).
///
/// This is a utility for manual gradient averaging when using Sum reduction.
pub fn scale_gradients(gradients: &mut [f32], world_size: usize) {
    let scale = 1.0 / world_size as f32;
    for g in gradients.iter_mut() {
        *g *= scale;
    }
}

/// Configuration for distributed training.
#[derive(Debug, Clone)]
pub struct DistributedConfig {
    /// Gradient reduction strategy. Default: Mean
    pub reduce_strategy: ReduceStrategy,
    /// Number of samples per GPU per minibatch.
    pub per_worker_batch_size: usize,
    /// Whether to synchronize batch normalization statistics.
    pub sync_batch_norm: bool,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            reduce_strategy: ReduceStrategy::Mean,
            per_worker_batch_size: 256,
            sync_batch_norm: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_sync_returns_identity() {
        let sync = LocalSync;
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let result = sync.all_reduce_f32(&values, ReduceStrategy::Mean);
        assert_eq!(result, values);
    }

    #[test]
    fn local_sync_world_size_is_one() {
        let sync = LocalSync;
        assert_eq!(sync.world_size(), 1);
    }

    #[test]
    fn local_sync_rank_is_zero() {
        let sync = LocalSync;
        assert_eq!(sync.rank(), 0);
    }

    #[test]
    fn local_sync_barrier_is_noop() {
        let sync = LocalSync;
        sync.barrier(); // should not panic
    }

    #[test]
    fn scale_gradients_divides_by_world_size() {
        let mut grads = vec![4.0, 8.0, 12.0];
        scale_gradients(&mut grads, 4);
        assert!((grads[0] - 1.0).abs() < 1e-6);
        assert!((grads[1] - 2.0).abs() < 1e-6);
        assert!((grads[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn scale_gradients_world_size_one_is_identity() {
        let mut grads = vec![1.0, 2.0, 3.0];
        let original = grads.clone();
        scale_gradients(&mut grads, 1);
        assert_eq!(grads, original);
    }

    #[test]
    fn distributed_config_defaults() {
        let config = DistributedConfig::default();
        assert_eq!(config.reduce_strategy, ReduceStrategy::Mean);
        assert_eq!(config.per_worker_batch_size, 256);
        assert!(!config.sync_batch_norm);
    }

    #[test]
    fn local_sync_sum_strategy() {
        let sync = LocalSync;
        let values = vec![10.0, 20.0];
        let result = sync.all_reduce_f32(&values, ReduceStrategy::Sum);
        assert_eq!(result, values);
    }
}
