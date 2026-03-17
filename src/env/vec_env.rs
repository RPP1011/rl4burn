//! Vectorized environment: runs N environments in lockstep.
//!
//! Auto-resets environments when episodes end, returning the initial
//! observation of the new episode in the step result.

use super::{Env, Step};
use super::space::Space;

/// Synchronous vectorized environment.
///
/// Runs N copies of an environment, auto-resetting on done.
/// Observations from all environments are returned as a Vec,
/// suitable for batching into tensors.
pub struct SyncVecEnv<E> {
    envs: Vec<E>,
}

impl<E: Env> SyncVecEnv<E> {
    /// Create a vectorized environment from a list of environment instances.
    pub fn new(envs: Vec<E>) -> Self {
        assert!(!envs.is_empty(), "need at least one environment");
        Self { envs }
    }

    /// Number of parallel environments.
    pub fn num_envs(&self) -> usize {
        self.envs.len()
    }

    /// Reset all environments, returning initial observations.
    pub fn reset(&mut self) -> Vec<E::Observation> {
        self.envs.iter_mut().map(|e| e.reset()).collect()
    }

    /// Step all environments with one action per env.
    ///
    /// When an environment is done, it is automatically reset.
    /// The returned observation is the initial observation of the new episode
    /// (the terminal observation is lost — store it before stepping if needed).
    pub fn step(&mut self, actions: Vec<E::Action>) -> Vec<Step<E::Observation>> {
        assert_eq!(
            actions.len(),
            self.envs.len(),
            "must provide one action per environment"
        );
        actions
            .into_iter()
            .zip(self.envs.iter_mut())
            .map(|(action, env)| {
                let step = env.step(action);
                if step.done() {
                    let obs = env.reset();
                    Step {
                        observation: obs,
                        reward: step.reward,
                        terminated: step.terminated,
                        truncated: step.truncated,
                    }
                } else {
                    step
                }
            })
            .collect()
    }

    /// Observation space (same for all environments).
    pub fn observation_space(&self) -> Space {
        self.envs[0].observation_space()
    }

    /// Action space (same for all environments).
    pub fn action_space(&self) -> Space {
        self.envs[0].action_space()
    }

    /// Collect action masks from all environments.
    ///
    /// Returns `None` if no environment provides masks.
    /// When at least one env provides masks, envs without masks get all-ones masks.
    pub fn action_masks(&self) -> Option<Vec<Vec<f32>>> {
        let masks: Vec<Option<Vec<f32>>> = self.envs.iter().map(|e| e.action_mask()).collect();

        if masks.iter().all(|m| m.is_none()) {
            return None;
        }

        let mask_len = self.action_space().flat_dim();
        Some(
            masks
                .into_iter()
                .map(|m| m.unwrap_or_else(|| vec![1.0; mask_len]))
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal environment that counts steps and terminates at a fixed length.
    struct CountEnv {
        step_count: usize,
        max_steps: usize,
    }

    impl CountEnv {
        fn new(max_steps: usize) -> Self {
            Self {
                step_count: 0,
                max_steps,
            }
        }
    }

    impl Env for CountEnv {
        type Observation = Vec<f32>;
        type Action = usize;

        fn reset(&mut self) -> Vec<f32> {
            self.step_count = 0;
            vec![0.0]
        }

        fn step(&mut self, _action: usize) -> Step<Vec<f32>> {
            self.step_count += 1;
            let done = self.step_count >= self.max_steps;
            Step {
                observation: vec![self.step_count as f32],
                reward: 1.0,
                terminated: done,
                truncated: false,
            }
        }

        fn observation_space(&self) -> Space {
            Space::Box {
                low: vec![0.0],
                high: vec![100.0],
            }
        }

        fn action_space(&self) -> Space {
            Space::Discrete(2)
        }
    }

    #[test]
    fn reset_returns_initial_obs() {
        let mut venv = SyncVecEnv::new(vec![CountEnv::new(3), CountEnv::new(3)]);
        let obs = venv.reset();
        assert_eq!(obs.len(), 2);
        assert_eq!(obs[0], vec![0.0]);
        assert_eq!(obs[1], vec![0.0]);
    }

    #[test]
    fn step_returns_correct_count() {
        let mut venv = SyncVecEnv::new(vec![CountEnv::new(5)]);
        venv.reset();
        let steps = venv.step(vec![0]);
        assert_eq!(steps[0].observation, vec![1.0]);
        assert!(!steps[0].done());
    }

    #[test]
    fn auto_reset_on_done() {
        let mut venv = SyncVecEnv::new(vec![CountEnv::new(2)]);
        venv.reset();
        venv.step(vec![0]); // step 1
        let steps = venv.step(vec![0]); // step 2 -> done -> auto-reset
        assert!(steps[0].terminated);
        // Observation is from the reset (new episode)
        assert_eq!(steps[0].observation, vec![0.0]);
    }

    #[test]
    fn num_envs() {
        let venv = SyncVecEnv::new(vec![CountEnv::new(1), CountEnv::new(1), CountEnv::new(1)]);
        assert_eq!(venv.num_envs(), 3);
    }
}
