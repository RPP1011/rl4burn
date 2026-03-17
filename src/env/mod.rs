//! Environment trait and step result type.
//!
//! Follows modern Gymnasium conventions: separate `terminated` and `truncated`
//! flags, auto-reset semantics for vectorized environments.

pub mod adapter;
pub mod space;
pub mod vec_env;
pub mod wrapper;

use space::Space;

/// Result of a single environment step.
#[derive(Debug, Clone)]
pub struct Step<O> {
    /// Observation after the transition (or initial obs if auto-reset fired).
    pub observation: O,
    /// Scalar reward for this transition.
    pub reward: f32,
    /// True if the episode ended due to environment dynamics (goal reached, failure, etc.).
    pub terminated: bool,
    /// True if the episode ended due to a time limit or external constraint.
    pub truncated: bool,
}

impl<O> Step<O> {
    /// Whether the episode is over (terminated or truncated).
    pub fn done(&self) -> bool {
        self.terminated || self.truncated
    }
}

/// A reinforcement learning environment.
///
/// Implementations define the dynamics, observation/action types, and space metadata.
pub trait Env {
    /// Observation type returned by `reset` and `step`.
    type Observation: Clone;
    /// Action type accepted by `step`.
    type Action: Clone;

    /// Reset the environment and return the initial observation.
    fn reset(&mut self) -> Self::Observation;

    /// Take one step with the given action.
    fn step(&mut self, action: Self::Action) -> Step<Self::Observation>;

    /// Description of the observation space.
    fn observation_space(&self) -> Space;

    /// Description of the action space.
    fn action_space(&self) -> Space;

    /// Return a flat validity mask for the current state.
    ///
    /// Length must equal `action_space().flat_dim()` (or sum of nvec for MultiDiscrete).
    /// `1.0` = valid, `0.0` = invalid.
    ///
    /// Default: all actions valid (returns `None`).
    fn action_mask(&self) -> Option<Vec<f32>> {
        None
    }
}
