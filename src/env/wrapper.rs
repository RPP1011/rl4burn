//! Environment wrappers.
//!
//! Composable transformations applied around an environment:
//! episode statistics tracking, reward clipping, observation normalization.

use super::{Env, Step};
use super::space::Space;

// ---------------------------------------------------------------------------
// EpisodeStats
// ---------------------------------------------------------------------------

/// Tracks cumulative episode reward and length.
///
/// After each episode ends, `last_episode_reward` and `last_episode_length`
/// are updated with the completed episode's statistics.
pub struct EpisodeStats<E> {
    inner: E,
    episode_reward: f32,
    episode_length: usize,
    /// Reward of the most recently completed episode (None if no episode has finished).
    pub last_episode_reward: Option<f32>,
    /// Length of the most recently completed episode.
    pub last_episode_length: Option<usize>,
}

impl<E: Env> EpisodeStats<E> {
    pub fn new(inner: E) -> Self {
        Self {
            inner,
            episode_reward: 0.0,
            episode_length: 0,
            last_episode_reward: None,
            last_episode_length: None,
        }
    }

    /// Access the wrapped environment.
    pub fn inner(&self) -> &E {
        &self.inner
    }
}

impl<E: Env> Env for EpisodeStats<E> {
    type Observation = E::Observation;
    type Action = E::Action;

    fn reset(&mut self) -> Self::Observation {
        self.episode_reward = 0.0;
        self.episode_length = 0;
        self.inner.reset()
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation> {
        let step = self.inner.step(action);
        self.episode_reward += step.reward;
        self.episode_length += 1;
        if step.done() {
            self.last_episode_reward = Some(self.episode_reward);
            self.last_episode_length = Some(self.episode_length);
            self.episode_reward = 0.0;
            self.episode_length = 0;
        }
        step
    }

    fn observation_space(&self) -> Space {
        self.inner.observation_space()
    }

    fn action_space(&self) -> Space {
        self.inner.action_space()
    }

    fn action_mask(&self) -> Option<Vec<f32>> {
        self.inner.action_mask()
    }
}

// ---------------------------------------------------------------------------
// RewardClip
// ---------------------------------------------------------------------------

/// Clips rewards to `[-limit, limit]`.
pub struct RewardClip<E> {
    inner: E,
    limit: f32,
}

impl<E: Env> RewardClip<E> {
    pub fn new(inner: E, limit: f32) -> Self {
        assert!(limit > 0.0, "limit must be positive");
        Self { inner, limit }
    }
}

impl<E: Env> Env for RewardClip<E> {
    type Observation = E::Observation;
    type Action = E::Action;

    fn reset(&mut self) -> Self::Observation {
        self.inner.reset()
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation> {
        let mut step = self.inner.step(action);
        step.reward = step.reward.clamp(-self.limit, self.limit);
        step
    }

    fn observation_space(&self) -> Space {
        self.inner.observation_space()
    }

    fn action_space(&self) -> Space {
        self.inner.action_space()
    }

    fn action_mask(&self) -> Option<Vec<f32>> {
        self.inner.action_mask()
    }
}

// ---------------------------------------------------------------------------
// NormalizeObservation
// ---------------------------------------------------------------------------

/// Normalizes observations using a running mean and variance estimate.
///
/// Uses Welford's online algorithm to track statistics.
pub struct NormalizeObservation<E> {
    inner: E,
    mean: Vec<f64>,
    var: Vec<f64>,
    count: f64,
    clip: f32,
}

impl<E: Env<Observation = Vec<f32>>> NormalizeObservation<E> {
    pub fn new(inner: E, clip: f32) -> Self {
        let dim = match inner.observation_space() {
            Space::Box { ref low, .. } => low.len(),
            _ => panic!("NormalizeObservation requires a Box observation space"),
        };
        Self {
            inner,
            mean: vec![0.0; dim],
            var: vec![1.0; dim],
            count: 1e-4, // small epsilon to avoid division by zero
            clip,
        }
    }

    fn update_stats(&mut self, obs: &[f32]) {
        let batch_count = 1.0;
        let new_count = self.count + batch_count;
        for i in 0..obs.len() {
            let delta = obs[i] as f64 - self.mean[i];
            self.mean[i] += delta * batch_count / new_count;
            let m_a = self.var[i] * self.count;
            let m_b = (obs[i] as f64 - self.mean[i]).powi(2) * batch_count;
            self.var[i] = (m_a + m_b) / new_count;
        }
        self.count = new_count;
    }

    fn normalize_obs(&self, obs: &[f32]) -> Vec<f32> {
        obs.iter()
            .enumerate()
            .map(|(i, &x)| {
                let std = (self.var[i].max(1e-8)).sqrt() as f32;
                ((x - self.mean[i] as f32) / std).clamp(-self.clip, self.clip)
            })
            .collect()
    }
}

impl<E: Env<Observation = Vec<f32>>> Env for NormalizeObservation<E> {
    type Observation = Vec<f32>;
    type Action = E::Action;

    fn reset(&mut self) -> Vec<f32> {
        let obs = self.inner.reset();
        self.update_stats(&obs);
        self.normalize_obs(&obs)
    }

    fn step(&mut self, action: Self::Action) -> Step<Vec<f32>> {
        let step = self.inner.step(action);
        self.update_stats(&step.observation);
        Step {
            observation: self.normalize_obs(&step.observation),
            reward: step.reward,
            terminated: step.terminated,
            truncated: step.truncated,
        }
    }

    fn observation_space(&self) -> Space {
        self.inner.observation_space()
    }

    fn action_space(&self) -> Space {
        self.inner.action_space()
    }

    fn action_mask(&self) -> Option<Vec<f32>> {
        self.inner.action_mask()
    }
}

// ---------------------------------------------------------------------------
// NormalizeReward
// ---------------------------------------------------------------------------

/// Normalizes rewards by the running standard deviation of discounted returns.
///
/// Matches Gymnasium's `NormalizeReward` wrapper. Tracks the discounted return
/// accumulator and normalizes each reward by the running std of these returns.
/// This stabilizes the value function target scale across training.
///
/// The mean is tracked but not subtracted — only division by std is applied.
/// Normalized rewards are clipped to `[-clip, clip]`.
pub struct NormalizeReward<E> {
    inner: E,
    gamma: f32,
    clip: f32,
    ret: f64,
    // Running stats for returns (Welford's online algorithm)
    mean: f64,
    var: f64,
    count: f64,
}

impl<E: Env> NormalizeReward<E> {
    pub fn new(inner: E, gamma: f32, clip: f32) -> Self {
        Self {
            inner,
            gamma,
            clip,
            ret: 0.0,
            mean: 0.0,
            var: 1.0,
            count: 1e-4,
        }
    }

    fn update_return_stats(&mut self, ret: f64) {
        let new_count = self.count + 1.0;
        let delta = ret - self.mean;
        self.mean += delta / new_count;
        let m_a = self.var * self.count;
        let m_b = (ret - self.mean) * delta;
        self.var = (m_a + m_b) / new_count;
        self.count = new_count;
    }
}

impl<E: Env> Env for NormalizeReward<E> {
    type Observation = E::Observation;
    type Action = E::Action;

    fn reset(&mut self) -> Self::Observation {
        self.ret = 0.0;
        self.inner.reset()
    }

    fn step(&mut self, action: Self::Action) -> Step<Self::Observation> {
        let step = self.inner.step(action);

        // Update discounted return (resets on done, matching Gymnasium)
        let not_done = if step.done() { 0.0f64 } else { 1.0 };
        self.ret = self.ret * self.gamma as f64 * not_done + step.reward as f64;
        self.update_return_stats(self.ret);

        // Normalize by std of returns (no centering)
        let std = (self.var.max(1e-8)).sqrt() as f32;
        let normalized_reward = (step.reward / std).clamp(-self.clip, self.clip);

        Step {
            observation: step.observation,
            reward: normalized_reward,
            terminated: step.terminated,
            truncated: step.truncated,
        }
    }

    fn observation_space(&self) -> Space {
        self.inner.observation_space()
    }

    fn action_space(&self) -> Space {
        self.inner.action_space()
    }

    fn action_mask(&self) -> Option<Vec<f32>> {
        self.inner.action_mask()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::Step;

    struct ConstEnv;

    impl Env for ConstEnv {
        type Observation = Vec<f32>;
        type Action = usize;

        fn reset(&mut self) -> Vec<f32> {
            vec![1.0, 2.0]
        }
        fn step(&mut self, _: usize) -> Step<Vec<f32>> {
            Step {
                observation: vec![1.0, 2.0],
                reward: 5.0,
                terminated: true,
                truncated: false,
            }
        }
        fn observation_space(&self) -> Space {
            Space::Box {
                low: vec![-10.0; 2],
                high: vec![10.0; 2],
            }
        }
        fn action_space(&self) -> Space {
            Space::Discrete(2)
        }
    }

    #[test]
    fn episode_stats_tracks_return() {
        let mut env = EpisodeStats::new(ConstEnv);
        env.reset();
        env.step(0);
        assert_eq!(env.last_episode_reward, Some(5.0));
        assert_eq!(env.last_episode_length, Some(1));
    }

    #[test]
    fn reward_clip_clamps() {
        let mut env = RewardClip::new(ConstEnv, 1.0);
        env.reset();
        let step = env.step(0);
        assert_eq!(step.reward, 1.0); // clipped from 5.0
    }
}
