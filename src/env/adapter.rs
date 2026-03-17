//! Environment adapters for action type conversion.

use super::space::Space;
use super::{Env, Step};

/// Wraps a discrete (`Action = usize`) environment to accept `Vec<f32>` actions.
///
/// Takes the first element of the action vector and converts to `usize`.
/// This allows using simple discrete environments with the masked/multi-discrete
/// PPO pipeline which expects `Action = Vec<f32>`.
pub struct DiscreteEnvAdapter<E>(pub E);

impl<E: Env<Action = usize>> Env for DiscreteEnvAdapter<E> {
    type Observation = E::Observation;
    type Action = Vec<f32>;

    fn reset(&mut self) -> Self::Observation {
        self.0.reset()
    }

    fn step(&mut self, action: Vec<f32>) -> Step<Self::Observation> {
        self.0.step(action[0] as usize)
    }

    fn observation_space(&self) -> Space {
        self.0.observation_space()
    }

    fn action_space(&self) -> Space {
        self.0.action_space()
    }

    fn action_mask(&self) -> Option<Vec<f32>> {
        self.0.action_mask()
    }
}
