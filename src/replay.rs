//! Replay buffer with trajectory-aware operations.
//!
//! Generic over sample type — users define their own sample struct.

use contracts::*;

/// A replay buffer that stores samples with trajectory metadata.
///
/// Supports:
/// - Uniform random eviction when over capacity
/// - Trajectory grouping by ID for V-trace rescoring
/// - Capacity-bounded storage
pub struct ReplayBuffer<S> {
    samples: Vec<S>,
    capacity: usize,
    rng: u64,
}

impl<S> ReplayBuffer<S> {
    /// Create a new replay buffer with the given capacity.
    #[requires(capacity > 0, "capacity must be positive")]
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: Vec::new(),
            capacity,
            rng: 0xDEADBEEF,
        }
    }

    /// Add samples, evicting uniformly at random if over capacity.
    #[ensures(self.len() <= self.capacity)]
    pub fn extend(&mut self, new_samples: impl IntoIterator<Item = S>) {
        self.samples.extend(new_samples);
        if self.samples.len() > self.capacity {
            // Fisher-Yates partial shuffle: randomly select `capacity` to keep
            for i in 0..self.capacity {
                self.rng = self.rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let j = i + (self.rng >> 33) as usize % (self.samples.len() - i);
                self.samples.swap(i, j);
            }
            self.samples.truncate(self.capacity);
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
        if n == 0 { return vec![]; }
        let size = size.min(n);
        (0..size).map(|_| {
            self.rng = self.rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let idx = (self.rng >> 33) as usize % n;
            &self.samples[idx]
        }).collect()
    }

    /// Sample a random minibatch of cloned samples.
    #[ensures(ret.len() <= size)]
    pub fn sample_cloned(&mut self, size: usize) -> Vec<S> where S: Clone {
        let n = self.samples.len();
        if n == 0 { return vec![]; }
        let size = size.min(n);
        (0..size).map(|_| {
            self.rng = self.rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let idx = (self.rng >> 33) as usize % n;
            self.samples[idx].clone()
        }).collect()
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

    #[test]
    fn capacity_enforced() {
        let mut buf = ReplayBuffer::new(5);
        buf.extend(0..10);
        assert_eq!(buf.len(), 5);
    }

    #[test]
    fn empty_buffer() {
        let buf: ReplayBuffer<i32> = ReplayBuffer::new(10);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn sample_returns_correct_size() {
        let mut buf = ReplayBuffer::new(100);
        buf.extend(0..50);
        let batch = buf.sample(10);
        assert_eq!(batch.len(), 10);
    }

    #[test]
    fn sample_cloned_returns_correct_size() {
        let mut buf = ReplayBuffer::new(100);
        buf.extend(0..50);
        let batch = buf.sample_cloned(10);
        assert_eq!(batch.len(), 10);
    }

    #[test]
    fn group_by_works() {
        let mut buf = ReplayBuffer::new(100);
        buf.extend(vec![
            (1u32, "a"), (1, "b"), (2, "c"), (2, "d"), (3, "e"),
        ]);
        let groups = buf.group_by(|(id, _)| *id);
        assert_eq!(groups[&1].len(), 2);
        assert_eq!(groups[&2].len(), 2);
        assert_eq!(groups[&3].len(), 1);
    }

    #[test]
    fn extend_under_capacity_keeps_all() {
        let mut buf = ReplayBuffer::new(100);
        buf.extend(0..10);
        assert_eq!(buf.len(), 10);
    }
}
