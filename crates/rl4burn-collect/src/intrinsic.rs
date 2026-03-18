//! Intrinsic reward framework (Issue #28).
//!
//! Pluggable intrinsic reward modules that provide exploration bonuses
//! based on internal state (curiosity, prediction error, coverage, etc.).
//!
//! Includes:
//! - [`IntrinsicReward`] — trait for computing intrinsic rewards.
//! - [`CountBasedReward`] — count-based exploration: `reward = 1 / sqrt(N(s))`.
//! - [`EntropyReductionReward`] — ROA-Star scouting reward based on entropy reduction.
//! - [`combine_rewards`] — combine extrinsic and intrinsic rewards.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Trait for intrinsic reward modules.
///
/// Intrinsic rewards provide exploration bonuses based on internal
/// state (curiosity, prediction error, coverage, etc.).
pub trait IntrinsicReward {
    /// Observation type.
    type Observation;

    /// Compute intrinsic reward for a transition.
    fn reward(
        &self,
        obs: &Self::Observation,
        action: usize,
        next_obs: &Self::Observation,
    ) -> f32;

    /// Update internal state after observing a transition.
    fn update(
        &mut self,
        obs: &Self::Observation,
        action: usize,
        next_obs: &Self::Observation,
    );
}

// ---------------------------------------------------------------------------
// Combine rewards
// ---------------------------------------------------------------------------

/// Combine extrinsic and intrinsic rewards element-wise.
///
/// `combined[i] = extrinsic[i] + intrinsic_coef * intrinsic[i]`
///
/// # Panics
/// Panics if the slices have different lengths.
pub fn combine_rewards(
    extrinsic: &[f32],
    intrinsic: &[f32],
    intrinsic_coef: f32,
) -> Vec<f32> {
    assert_eq!(
        extrinsic.len(),
        intrinsic.len(),
        "extrinsic and intrinsic reward slices must have the same length"
    );
    extrinsic
        .iter()
        .zip(intrinsic.iter())
        .map(|(e, i)| e + intrinsic_coef * i)
        .collect()
}

// ---------------------------------------------------------------------------
// Count-based exploration
// ---------------------------------------------------------------------------

/// Count-based exploration: `reward = 1 / sqrt(N(s))`.
///
/// Observations are discretized to a grid of the given resolution before
/// counting, which allows this to work with continuous observation spaces.
pub struct CountBasedReward {
    counts: HashMap<Vec<i32>, u64>,
    /// Discretization resolution for continuous observations.
    resolution: f32,
}

impl CountBasedReward {
    /// Create a new count-based reward module.
    ///
    /// # Arguments
    /// * `resolution` — bin width for discretizing continuous observations.
    pub fn new(resolution: f32) -> Self {
        Self {
            counts: HashMap::new(),
            resolution,
        }
    }

    fn discretize(&self, obs: &[f32]) -> Vec<i32> {
        obs.iter()
            .map(|x| (x / self.resolution).round() as i32)
            .collect()
    }

    /// Number of distinct states visited.
    pub fn num_visited(&self) -> usize {
        self.counts.len()
    }
}

impl IntrinsicReward for CountBasedReward {
    type Observation = Vec<f32>;

    fn reward(
        &self,
        _obs: &Vec<f32>,
        _action: usize,
        next_obs: &Vec<f32>,
    ) -> f32 {
        let key = self.discretize(next_obs);
        let count = self.counts.get(&key).copied().unwrap_or(0);
        1.0 / ((count as f32).max(1.0)).sqrt()
    }

    fn update(
        &mut self,
        _obs: &Vec<f32>,
        _action: usize,
        next_obs: &Vec<f32>,
    ) {
        let key = self.discretize(next_obs);
        *self.counts.entry(key).or_insert(0) += 1;
    }
}

// ---------------------------------------------------------------------------
// Entropy-reduction reward
// ---------------------------------------------------------------------------

/// Entropy-reduction reward (ROA-Star scouting reward).
///
/// `reward = max(H(t-1) - H(t), 0)` based on prediction entropy reduction.
/// Call [`reward_from_entropy`](Self::reward_from_entropy) with the current
/// prediction entropy after each model update.
pub struct EntropyReductionReward {
    prev_entropy: f32,
}

impl EntropyReductionReward {
    /// Create a new entropy-reduction reward module.
    pub fn new() -> Self {
        Self {
            prev_entropy: f32::MAX,
        }
    }

    /// Compute reward from entropy reduction.
    ///
    /// Returns `max(prev_entropy - current_entropy, 0)` and stores
    /// `current_entropy` for the next call.
    pub fn reward_from_entropy(&mut self, current_entropy: f32) -> f32 {
        let reward = (self.prev_entropy - current_entropy).max(0.0);
        self.prev_entropy = current_entropy;
        reward
    }
}

impl Default for EntropyReductionReward {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- combine_rewards -----------------------------------------------------

    #[test]
    fn combine_rewards_basic() {
        let ext = vec![1.0, 2.0, 3.0];
        let intr = vec![0.5, 0.5, 0.5];
        let combined = combine_rewards(&ext, &intr, 0.1);
        assert_eq!(combined.len(), 3);
        assert!((combined[0] - 1.05).abs() < 1e-6);
        assert!((combined[1] - 2.05).abs() < 1e-6);
        assert!((combined[2] - 3.05).abs() < 1e-6);
    }

    #[test]
    fn combine_rewards_zero_coef() {
        let ext = vec![1.0, 2.0];
        let intr = vec![100.0, 200.0];
        let combined = combine_rewards(&ext, &intr, 0.0);
        assert!((combined[0] - 1.0).abs() < 1e-6);
        assert!((combined[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn combine_rewards_mismatched_lengths() {
        combine_rewards(&[1.0], &[1.0, 2.0], 1.0);
    }

    // -- CountBasedReward ----------------------------------------------------

    #[test]
    fn count_based_reward_decreases_with_visits() {
        let mut module = CountBasedReward::new(1.0);
        let obs = vec![0.0];
        let next_obs = vec![1.0];

        // First visit: count=0 -> reward = 1/sqrt(1) = 1.0
        let r1 = module.reward(&obs, 0, &next_obs);
        module.update(&obs, 0, &next_obs);

        // Second visit: count=1 -> reward = 1/sqrt(1) = 1.0
        let r2 = module.reward(&obs, 0, &next_obs);
        module.update(&obs, 0, &next_obs);

        // Third visit: count=2 -> reward = 1/sqrt(2) ≈ 0.707
        let r3 = module.reward(&obs, 0, &next_obs);

        assert!((r1 - 1.0).abs() < 1e-6, "first visit reward should be 1.0, got {r1}");
        assert!((r2 - 1.0).abs() < 1e-6, "second visit reward should be 1.0, got {r2}");
        assert!(
            r3 < r2,
            "reward should decrease with more visits: r2={r2}, r3={r3}"
        );
    }

    #[test]
    fn count_based_novel_state_high_reward() {
        let mut module = CountBasedReward::new(1.0);
        let obs = vec![0.0];

        // Visit state A many times
        let next_a = vec![1.0];
        for _ in 0..100 {
            module.update(&obs, 0, &next_a);
        }

        // Novel state B should have high reward
        let next_b = vec![5.0];
        let r_a = module.reward(&obs, 0, &next_a);
        let r_b = module.reward(&obs, 0, &next_b);
        assert!(r_b > r_a, "novel state should have higher reward: a={r_a}, b={r_b}");
    }

    #[test]
    fn count_based_num_visited() {
        let mut module = CountBasedReward::new(1.0);
        let obs = vec![0.0];
        module.update(&obs, 0, &vec![1.0]);
        module.update(&obs, 0, &vec![2.0]);
        module.update(&obs, 0, &vec![1.0]); // revisit
        assert_eq!(module.num_visited(), 2);
    }

    // -- EntropyReductionReward ----------------------------------------------

    #[test]
    fn entropy_reduction_basic() {
        let mut module = EntropyReductionReward::new();
        // First call: prev=MAX, current=2.0 -> reward = MAX-2.0 (capped)
        let r1 = module.reward_from_entropy(2.0);
        assert!(r1 > 0.0);

        // Entropy drops: 2.0 -> 1.0 -> reward = 1.0
        let r2 = module.reward_from_entropy(1.0);
        assert!((r2 - 1.0).abs() < 1e-6);

        // Entropy rises: 1.0 -> 1.5 -> reward = 0.0
        let r3 = module.reward_from_entropy(1.5);
        assert!((r3 - 0.0).abs() < 1e-6);
    }
}
