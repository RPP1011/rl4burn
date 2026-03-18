//! Replay buffer with trajectory-aware operations.
//!
//! Generic over sample type — users define their own sample struct.

use contracts::*;
use rand::{Rng, RngExt};

/// A replay buffer that stores samples with trajectory metadata.
///
/// Supports:
/// - FIFO eviction when over capacity (oldest samples dropped first)
/// - Trajectory grouping by ID for V-trace rescoring
/// - Capacity-bounded storage
///
/// The buffer is parameterized by a deterministic RNG for reproducible sampling.
pub struct ReplayBuffer<S, R> {
    samples: Vec<S>,
    capacity: usize,
    rng: R,
}

impl<S, R: Rng> ReplayBuffer<S, R> {
    /// Create a new replay buffer with the given capacity and RNG.
    #[requires(capacity > 0, "capacity must be positive")]
    pub fn new(capacity: usize, rng: R) -> Self {
        Self {
            samples: Vec::new(),
            capacity,
            rng,
        }
    }

    /// Add samples, dropping the oldest (FIFO) if over capacity.
    #[ensures(self.len() <= self.capacity)]
    pub fn extend(&mut self, new_samples: impl IntoIterator<Item = S>) {
        self.samples.extend(new_samples);
        if self.samples.len() > self.capacity {
            let excess = self.samples.len() - self.capacity;
            self.samples.drain(..excess);
        }
    }

    /// Number of samples currently stored.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Sample a random minibatch of `size` (with replacement).
    #[ensures(ret.len() <= size)]
    pub fn sample(&mut self, size: usize) -> Vec<&S> {
        let n = self.samples.len();
        if n == 0 {
            return vec![];
        }
        let size = size.min(n);
        (0..size)
            .map(|_| {
                let idx = self.rng.random_range(0..n);
                &self.samples[idx]
            })
            .collect()
    }

    /// Sample a random minibatch of cloned samples.
    #[ensures(ret.len() <= size)]
    pub fn sample_cloned(&mut self, size: usize) -> Vec<S>
    where
        S: Clone,
    {
        let n = self.samples.len();
        if n == 0 {
            return vec![];
        }
        let size = size.min(n);
        (0..size)
            .map(|_| {
                let idx = self.rng.random_range(0..n);
                self.samples[idx].clone()
            })
            .collect()
    }

    /// Access all samples (for rescoring).
    pub fn samples(&self) -> &[S] {
        &self.samples
    }

    /// Mutable access to all samples (for rescoring).
    pub fn samples_mut(&mut self) -> &mut Vec<S> {
        &mut self.samples
    }

    /// Group sample indices by a key function (e.g., trajectory ID).
    /// Returns a map from key to Vec of indices into `samples()`.
    pub fn group_by<K, F>(&self, key_fn: F) -> std::collections::HashMap<K, Vec<usize>>
    where
        K: std::hash::Hash + Eq,
        F: Fn(&S) -> K,
    {
        let mut groups = std::collections::HashMap::new();
        for (i, s) in self.samples.iter().enumerate() {
            groups.entry(key_fn(s)).or_insert_with(Vec::new).push(i);
        }
        groups
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn rng() -> rand::rngs::SmallRng {
        rand::rngs::SmallRng::seed_from_u64(42)
    }

    #[test]
    fn capacity_enforced() {
        let mut buf = ReplayBuffer::new(5, rng());
        buf.extend(0..10);
        assert_eq!(buf.len(), 5);
    }

    #[test]
    fn empty_buffer() {
        let buf: ReplayBuffer<i32, _> = ReplayBuffer::new(10, rng());
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn sample_returns_correct_size() {
        let mut buf = ReplayBuffer::new(100, rng());
        buf.extend(0..50);
        let batch = buf.sample(10);
        assert_eq!(batch.len(), 10);
    }

    #[test]
    fn sample_cloned_returns_correct_size() {
        let mut buf = ReplayBuffer::new(100, rng());
        buf.extend(0..50);
        let batch = buf.sample_cloned(10);
        assert_eq!(batch.len(), 10);
    }

    #[test]
    fn group_by_works() {
        let mut buf = ReplayBuffer::new(100, rng());
        buf.extend(vec![
            (1u32, "a"),
            (1, "b"),
            (2, "c"),
            (2, "d"),
            (3, "e"),
        ]);
        let groups = buf.group_by(|(id, _)| *id);
        assert_eq!(groups[&1].len(), 2);
        assert_eq!(groups[&2].len(), 2);
        assert_eq!(groups[&3].len(), 1);
    }

    #[test]
    fn extend_under_capacity_keeps_all() {
        let mut buf = ReplayBuffer::new(100, rng());
        buf.extend(0..10);
        assert_eq!(buf.len(), 10);
    }

    #[test]
    fn fifo_keeps_newest() {
        let mut buf = ReplayBuffer::new(5, rng());
        buf.extend(0..10);
        // Should keep 5..10 (the newest)
        let samples: Vec<&i32> = buf.samples().iter().collect();
        assert_eq!(samples, vec![&5, &6, &7, &8, &9]);
    }

    #[test]
    fn deterministic_sampling() {
        let mut buf1 = ReplayBuffer::new(100, rand::rngs::SmallRng::seed_from_u64(7));
        let mut buf2 = ReplayBuffer::new(100, rand::rngs::SmallRng::seed_from_u64(7));
        buf1.extend(0..50);
        buf2.extend(0..50);
        let s1: Vec<i32> = buf1.sample_cloned(10);
        let s2: Vec<i32> = buf2.sample_cloned(10);
        assert_eq!(s1, s2, "same seed should produce same samples");
    }
}
