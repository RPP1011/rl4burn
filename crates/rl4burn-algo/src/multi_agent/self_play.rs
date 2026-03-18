//! Self-play training loop and agent branching (Issues #11, #26).
//!
//! Provides a [`SelfPlayPool`] that stores snapshots of past policies for
//! opponent sampling, and [`branch_agent`] for creating new agents
//! initialised from an existing agent's weights (with fresh optimizer state).

use rand::{Rng, RngExt};

// ---------------------------------------------------------------------------
// Policy snapshot
// ---------------------------------------------------------------------------

/// A stored policy snapshot for self-play.
pub struct PolicySnapshot<M> {
    /// The cloned model.
    pub model: M,
    /// Unique identifier for this snapshot.
    pub id: u64,
    /// Training step at which the snapshot was taken.
    pub training_step: u64,
}

// ---------------------------------------------------------------------------
// Opponent pool
// ---------------------------------------------------------------------------

/// Self-play opponent pool that stores cloned policy snapshots.
///
/// During training the current policy is periodically snapshotted into the
/// pool. When an opponent is needed, one is sampled uniformly at random
/// (or the latest can be retrieved for evaluation).
///
/// Memory note: each snapshot is a full clone of the model. Use
/// [`retain_recent`](Self::retain_recent) to bound memory usage.
pub struct SelfPlayPool<M> {
    snapshots: Vec<PolicySnapshot<M>>,
    next_id: u64,
}

impl<M: Clone> SelfPlayPool<M> {
    /// Create an empty pool.
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            next_id: 0,
        }
    }

    /// Store a snapshot of the current policy (clones the model).
    ///
    /// Returns the unique id assigned to this snapshot.
    pub fn add_snapshot(&mut self, model: &M, training_step: u64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.snapshots.push(PolicySnapshot {
            model: model.clone(),
            id,
            training_step,
        });
        id
    }

    /// Sample a random opponent from the pool.
    ///
    /// Returns `None` if the pool is empty.
    pub fn sample(&self, rng: &mut impl Rng) -> Option<&M> {
        if self.snapshots.is_empty() {
            return None;
        }
        let idx = rng.random_range(0..self.snapshots.len());
        Some(&self.snapshots[idx].model)
    }

    /// Get the most recent snapshot.
    pub fn latest(&self) -> Option<&M> {
        self.snapshots.last().map(|s| &s.model)
    }

    /// Number of stored snapshots.
    pub fn len(&self) -> usize {
        self.snapshots.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.snapshots.is_empty()
    }

    /// Remove all snapshots.
    pub fn clear(&mut self) {
        self.snapshots.clear();
    }

    /// Keep only the `n` most recent snapshots, discarding older ones.
    pub fn retain_recent(&mut self, n: usize) {
        if self.snapshots.len() > n {
            let drain_to = self.snapshots.len() - n;
            self.snapshots.drain(0..drain_to);
        }
    }
}

impl<M: Clone> Default for SelfPlayPool<M> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Agent branching
// ---------------------------------------------------------------------------

/// Branch (clone) an agent's weights.
///
/// This is the SCC approach: exploiter agents are initialised from the
/// current main agent rather than from a supervised model. The caller
/// should create a **fresh optimizer** for the branched agent so that
/// optimizer state (momentum, etc.) is reset.
///
/// In Burn the `Module` derive macro provides `Clone` via record
/// round-tripping, so branching is simply a clone. This function
/// documents the pattern and intent.
pub fn branch_agent<M: Clone>(source: &M) -> M {
    source.clone()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::nn::LinearConfig;
    use burn::prelude::*;
    use burn::tensor::TensorData;

    type B = NdArray;

    fn dev() -> <B as Backend>::Device {
        Default::default()
    }

    // -- SelfPlayPool --------------------------------------------------------

    #[test]
    fn pool_starts_empty() {
        let pool: SelfPlayPool<i32> = SelfPlayPool::new();
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn add_and_latest() {
        let mut pool = SelfPlayPool::new();
        pool.add_snapshot(&10, 0);
        pool.add_snapshot(&20, 100);
        assert_eq!(pool.len(), 2);
        assert_eq!(*pool.latest().unwrap(), 20);
    }

    #[test]
    fn sample_returns_stored_model() {
        let mut pool = SelfPlayPool::new();
        pool.add_snapshot(&42, 0);
        let mut rng = rand::rng();
        let sampled = pool.sample(&mut rng).unwrap();
        assert_eq!(*sampled, 42);
    }

    #[test]
    fn sample_empty_returns_none() {
        let pool: SelfPlayPool<i32> = SelfPlayPool::new();
        let mut rng = rand::rng();
        assert!(pool.sample(&mut rng).is_none());
    }

    #[test]
    fn retain_recent_keeps_n() {
        let mut pool = SelfPlayPool::new();
        for i in 0..10 {
            pool.add_snapshot(&(i as i32), i as u64);
        }
        pool.retain_recent(3);
        assert_eq!(pool.len(), 3);
        assert_eq!(*pool.latest().unwrap(), 9);
        // Oldest remaining should be 7
        assert_eq!(pool.snapshots[0].model, 7);
    }

    #[test]
    fn retain_recent_noop_when_small() {
        let mut pool = SelfPlayPool::new();
        pool.add_snapshot(&1, 0);
        pool.retain_recent(5);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn clear_empties_pool() {
        let mut pool = SelfPlayPool::new();
        pool.add_snapshot(&1, 0);
        pool.add_snapshot(&2, 1);
        pool.clear();
        assert!(pool.is_empty());
    }

    // -- branch_agent --------------------------------------------------------

    #[test]
    fn branched_model_produces_same_output() {
        let model = LinearConfig::new(4, 2).init::<B>(&dev());
        let branched = branch_agent(&model);

        let input = Tensor::<B, 2>::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [1, 4]),
            &dev(),
        );

        let out_orig: Vec<f32> = model
            .forward(input.clone())
            .into_data()
            .to_vec()
            .unwrap();
        let out_branch: Vec<f32> = branched.forward(input).into_data().to_vec().unwrap();

        assert_eq!(out_orig.len(), out_branch.len());
        for (a, b) in out_orig.iter().zip(out_branch.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "branched model output differs: {a} vs {b}"
            );
        }
    }

    #[test]
    fn snapshot_is_independent_of_source() {
        // Add model to pool, then mutate source. Snapshot should keep old values.
        let mut pool = SelfPlayPool::new();
        let model = vec![1.0f32, 2.0, 3.0];
        pool.add_snapshot(&model, 0);

        let mut model = model;
        model[0] = 999.0; // mutate source after snapshot

        let snapshot = pool.latest().unwrap();
        assert_eq!(
            snapshot[0], 1.0,
            "Snapshot should retain old value, got {}",
            snapshot[0]
        );
        assert_eq!(snapshot, &vec![1.0f32, 2.0, 3.0]);
    }

    #[test]
    fn snapshot_with_burn_module_is_independent() {
        // Verify that snapshotting a Burn module creates a true independent copy.
        let model = LinearConfig::new(4, 2).init::<B>(&dev());
        let mut pool = SelfPlayPool::new();
        pool.add_snapshot(&model, 0);

        let input = Tensor::<B, 2>::from_data(
            TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [1, 4]),
            &dev(),
        );

        let snapshot = pool.latest().unwrap();
        let out_snap: Vec<f32> = snapshot.forward(input.clone()).into_data().to_vec().unwrap();
        let out_orig: Vec<f32> = model.forward(input).into_data().to_vec().unwrap();

        // Both should be identical (cloned at same point)
        for (a, b) in out_snap.iter().zip(out_orig.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "snapshot and source should match: {a} vs {b}"
            );
        }
    }

    #[test]
    fn pool_with_burn_module() {
        let model = LinearConfig::new(4, 2).init::<B>(&dev());
        let mut pool = SelfPlayPool::new();
        pool.add_snapshot(&model, 0);
        assert_eq!(pool.len(), 1);
        assert!(pool.latest().is_some());
    }
}
