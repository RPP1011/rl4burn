//! Continuous LunarLander environment (LunarLanderContinuous-v3).
//!
//! Same physics as `LunarLander` but with continuous action control.
//!
//! - **Observation**: 8 floats (same as discrete variant)
//! - **Action**: `[main_engine, side_engine]` each in `[-1.0, 1.0]`
//!   - Main engine: off if ≤ 0, throttle = clamp(action[0], 0, 1) otherwise
//!   - Side engine: fire left if < -0.5, fire right if > 0.5
//! - **Reward**: same potential-based shaping as discrete variant
//! - **Terminated**: crash or successful landing
//! - **Truncated**: step count ≥ 1000

use rl4burn_core::env::render::{Renderable, RgbFrame};
use rl4burn_core::env::space::Space;
use rl4burn_core::env::{Env, Step};
use rand::Rng;

use crate::lunar_lander::{render_lander, LanderState};

/// Continuous LunarLander environment.
pub struct LunarLanderContinuous<R> {
    state: LanderState,
    rng: R,
}

impl<R: Rng> LunarLanderContinuous<R> {
    /// Create a new LunarLanderContinuous environment with the given RNG.
    pub fn new(rng: R) -> Self {
        let mut env = Self {
            state: LanderState::new(),
            rng,
        };
        env.reset();
        env
    }
}

impl<R: Rng> Env for LunarLanderContinuous<R> {
    type Observation = Vec<f32>;
    type Action = Vec<f32>;

    fn reset(&mut self) -> Vec<f32> {
        self.state.reset(&mut self.rng);
        self.state.obs()
    }

    fn step(&mut self, action: Vec<f32>) -> Step<Vec<f32>> {
        let raw_main = action[0].clamp(-1.0, 1.0);
        let raw_side = action[1].clamp(-1.0, 1.0);

        // Main engine: off if ≤ 0, otherwise throttle in [0, 1]
        let main_thrust = if raw_main > 0.0 { raw_main } else { 0.0 };

        // Side engine: fire if |side| > 0.5
        let side_thrust = if raw_side < -0.5 {
            -1.0
        } else if raw_side > 0.5 {
            1.0
        } else {
            0.0
        };

        self.state.physics_step(main_thrust, side_thrust, &mut self.rng);
        let (reward, terminated) = self.state.compute_reward(main_thrust, side_thrust.abs());
        let truncated = self.state.is_truncated();

        Step {
            observation: self.state.obs(),
            reward,
            terminated,
            truncated,
        }
    }

    fn observation_space(&self) -> Space {
        Space::Box {
            low: vec![-f32::INFINITY; 8],
            high: vec![f32::INFINITY; 8],
        }
    }

    fn action_space(&self) -> Space {
        Space::Box {
            low: vec![-1.0, -1.0],
            high: vec![1.0, 1.0],
        }
    }
}

impl<R> Renderable for LunarLanderContinuous<R> {
    fn render(&self) -> RgbFrame {
        render_lander(&self.state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_env() -> LunarLanderContinuous<rand::rngs::SmallRng> {
        LunarLanderContinuous::new(rand::rngs::SmallRng::seed_from_u64(42))
    }

    #[test]
    fn reset_returns_8d_obs() {
        let mut env = make_env();
        let obs = env.reset();
        assert_eq!(obs.len(), 8);
    }

    #[test]
    fn reward_is_finite() {
        let mut env = make_env();
        env.reset();
        for _ in 0..10 {
            let step = env.step(vec![0.5, 0.0]);
            assert!(step.reward.is_finite());
            if step.done() {
                break;
            }
        }
    }

    #[test]
    fn main_engine_off_when_negative() {
        let mut env = make_env();
        let obs1 = env.reset();
        // Step with main engine off (negative value)
        let step = env.step(vec![-1.0, 0.0]);
        // Lander should fall (no thrust)
        assert!(step.observation[3] <= obs1[3] || step.observation[1] < obs1[1]);
    }

    #[test]
    fn spaces() {
        let env = make_env();
        assert_eq!(env.observation_space().flat_dim(), 8);
        assert_eq!(env.action_space().flat_dim(), 2);
    }

    #[test]
    fn render_produces_valid_frame() {
        let env = make_env();
        let frame = env.render();
        assert_eq!(frame.data.len(), frame.width as usize * frame.height as usize * 3);
    }
}
