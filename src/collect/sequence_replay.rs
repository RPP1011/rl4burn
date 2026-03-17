//! Sequence replay buffer for world model training (Issue #24).
//!
//! FIFO buffer that samples contiguous sequences of a fixed length,
//! maintaining episode boundaries so sequences never cross episodes.
//! Used by DreamerV3 for world model training (typically T=64 timesteps).

use rand::Rng;

// ---------------------------------------------------------------------------
// Timestep
// ---------------------------------------------------------------------------

/// A single timestep stored in the sequence replay buffer.
#[derive(Clone, Debug)]
pub struct SequenceStep<O: Clone> {
    /// Observation at this timestep.
    pub observation: O,
    /// Action taken (as a float vector, supporting both discrete and continuous).
    pub action: Vec<f32>,
    /// Scalar reward received.
    pub reward: f32,
    /// Whether this step ends the episode.
    pub done: bool,
}

// ---------------------------------------------------------------------------
// Buffer
// ---------------------------------------------------------------------------

/// FIFO buffer that samples contiguous sequences of a fixed length.
///
/// Maintains episode boundaries: sequences never cross episode boundaries.
/// Used by DreamerV3 for world model training (T=64 timesteps).
pub struct SequenceReplayBuffer<O: Clone> {
    steps: Vec<SequenceStep<O>>,
    /// Indices where new episodes start.
    episode_starts: Vec<usize>,
    capacity: usize,
    sequence_length: usize,
}

impl<O: Clone> SequenceReplayBuffer<O> {
    /// Create a new buffer with the given capacity and sequence length.
    pub fn new(capacity: usize, sequence_length: usize) -> Self {
        Self {
            steps: Vec::with_capacity(capacity.min(1024)),
            episode_starts: vec![0], // first episode starts at 0
            capacity,
            sequence_length,
        }
    }

    /// Add a step to the buffer. Call with `done=true` at episode boundaries.
    pub fn push(&mut self, step: SequenceStep<O>) {
        let is_done = step.done;

        if self.steps.len() >= self.capacity {
            // FIFO: remove oldest step
            self.steps.remove(0);
            // Adjust episode_starts
            self.episode_starts.iter_mut().for_each(|s| {
                *s = s.saturating_sub(1);
            });
            self.episode_starts.retain(|&s| s < self.steps.len());
        }

        self.steps.push(step);

        if is_done {
            // Next step starts a new episode
            self.episode_starts.push(self.steps.len());
        }
    }

    /// Add multiple steps from an episode.
    pub fn extend(&mut self, steps: Vec<SequenceStep<O>>) {
        for step in steps {
            self.push(step);
        }
    }

    /// Sample a batch of contiguous sequences.
    ///
    /// Each sequence has exactly `sequence_length` steps and does not cross
    /// episode boundaries.
    ///
    /// Returns `Vec` of sequences, each `Vec<SequenceStep<O>>` of length
    /// `sequence_length`. Returns an empty vec if no valid sequences exist.
    pub fn sample(
        &self,
        batch_size: usize,
        rng: &mut impl Rng,
    ) -> Vec<Vec<SequenceStep<O>>> {
        let valid_starts = self.valid_sequence_starts();
        if valid_starts.is_empty() {
            return vec![];
        }

        let mut sequences = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let idx = rng.random_range(0..valid_starts.len());
            let start = valid_starts[idx];
            let seq: Vec<SequenceStep<O>> = self.steps[start..start + self.sequence_length]
                .iter()
                .cloned()
                .collect();
            sequences.push(seq);
        }
        sequences
    }

    /// Find all valid starting indices for sequences.
    ///
    /// A valid start index `i` means `steps[i..i+sequence_length]` are all
    /// within the same episode.
    fn valid_sequence_starts(&self) -> Vec<usize> {
        if self.steps.len() < self.sequence_length {
            return vec![];
        }

        // Build episode ranges
        let mut sorted_starts = self.episode_starts.clone();
        sorted_starts.sort();
        sorted_starts.dedup();

        let mut valid = Vec::new();
        for i in 0..sorted_starts.len() {
            let start = sorted_starts[i];
            let end = if i + 1 < sorted_starts.len() {
                sorted_starts[i + 1]
            } else {
                self.steps.len()
            };
            if end - start >= self.sequence_length {
                for s in start..=end - self.sequence_length {
                    valid.push(s);
                }
            }
        }
        valid
    }

    /// Number of steps currently stored.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Maximum capacity of the buffer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_step(obs: i32, done: bool) -> SequenceStep<i32> {
        SequenceStep {
            observation: obs,
            action: vec![0.0],
            reward: obs as f32,
            done,
        }
    }

    #[test]
    fn push_and_len() {
        let mut buf = SequenceReplayBuffer::new(100, 3);
        assert!(buf.is_empty());
        buf.push(make_step(1, false));
        buf.push(make_step(2, false));
        buf.push(make_step(3, false));
        assert_eq!(buf.len(), 3);
    }

    #[test]
    fn fifo_eviction() {
        let mut buf = SequenceReplayBuffer::new(5, 2);
        for i in 0..10 {
            buf.push(make_step(i, false));
        }
        assert_eq!(buf.len(), 5);
        // Should contain steps 5..10
        assert_eq!(buf.steps[0].observation, 5);
        assert_eq!(buf.steps[4].observation, 9);
    }

    #[test]
    fn sample_basic() {
        let mut buf = SequenceReplayBuffer::new(100, 3);
        // Single episode of 5 steps
        for i in 0..5 {
            buf.push(make_step(i, i == 4));
        }

        let mut rng = rand::rng();
        let sequences = buf.sample(2, &mut rng);
        assert_eq!(sequences.len(), 2);
        for seq in &sequences {
            assert_eq!(seq.len(), 3);
        }
    }

    #[test]
    fn empty_sample_when_too_short() {
        let mut buf = SequenceReplayBuffer::new(100, 5);
        buf.push(make_step(1, false));
        buf.push(make_step(2, false));

        let mut rng = rand::rng();
        let sequences = buf.sample(1, &mut rng);
        assert!(sequences.is_empty());
    }

    #[test]
    fn episode_boundaries_respected() {
        let mut buf = SequenceReplayBuffer::new(100, 3);

        // Episode 1: 2 steps (too short for seq_len=3)
        buf.push(make_step(10, false));
        buf.push(make_step(11, true));

        // Episode 2: 4 steps (enough for seq_len=3)
        buf.push(make_step(20, false));
        buf.push(make_step(21, false));
        buf.push(make_step(22, false));
        buf.push(make_step(23, true));

        let mut rng = rand::rng();
        for _ in 0..50 {
            let sequences = buf.sample(1, &mut rng);
            assert_eq!(sequences.len(), 1);
            let seq = &sequences[0];
            assert_eq!(seq.len(), 3);

            // All observations should be from episode 2 (20-23)
            for step in seq {
                assert!(
                    step.observation >= 20 && step.observation <= 23,
                    "got observation {} from wrong episode",
                    step.observation
                );
            }
        }
    }

    #[test]
    fn extend_works() {
        let mut buf = SequenceReplayBuffer::new(100, 2);
        let steps = vec![
            make_step(1, false),
            make_step(2, false),
            make_step(3, true),
        ];
        buf.extend(steps);
        assert_eq!(buf.len(), 3);

        let mut rng = rand::rng();
        let sequences = buf.sample(1, &mut rng);
        assert_eq!(sequences.len(), 1);
        assert_eq!(sequences[0].len(), 2);
    }

    #[test]
    fn fifo_eviction_preserves_episode_boundaries() {
        let mut buf = SequenceReplayBuffer::new(6, 3);

        // Episode 1: 4 steps
        for i in 0..4 {
            buf.push(make_step(i, i == 3));
        }
        // Episode 2: 4 steps — first 2 steps of ep1 get evicted
        for i in 10..14 {
            buf.push(make_step(i, i == 13));
        }

        assert_eq!(buf.len(), 6);

        // Episode 2 should be fully intact (4 steps, seq_len=3 fits)
        let mut rng = rand::rng();
        let sequences = buf.sample(10, &mut rng);
        assert!(!sequences.is_empty());
        for seq in &sequences {
            assert_eq!(seq.len(), 3);
        }
    }

    #[test]
    fn capacity_accessor() {
        let buf: SequenceReplayBuffer<i32> = SequenceReplayBuffer::new(42, 5);
        assert_eq!(buf.capacity(), 42);
    }
}
