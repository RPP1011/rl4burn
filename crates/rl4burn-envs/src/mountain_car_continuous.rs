//! Continuous MountainCar environment (MountainCarContinuous-v0).
//!
//! Same physics as MountainCar but with a continuous force action and a
//! different reward structure that encourages fuel efficiency.
//!
//! - **Observation**: `[position, velocity]` (2 floats)
//! - **Action**: `[force]` in `[-1.0, 1.0]` (continuous, 1 float)
//! - **Reward**: +100 on reaching goal, minus 0.1 * action², -1 per step otherwise
//! - **Terminated**: position ≥ 0.45
//! - **Truncated**: step count ≥ 999

use rl4burn_core::env::render::{Renderable, RgbFrame};
use rl4burn_core::env::space::Space;
use rl4burn_core::env::{Env, Step};
use rand::{Rng, RngExt};

use crate::render_util::{draw_circle, draw_rect, draw_thick_line};

const MIN_POSITION: f32 = -1.2;
const MAX_POSITION: f32 = 0.6;
const MAX_SPEED: f32 = 0.07;
const GOAL_POSITION: f32 = 0.45;
const POWER: f32 = 0.0015;
const GRAVITY: f32 = 0.0025;
const MAX_STEPS: usize = 999;

/// Continuous MountainCar environment.
pub struct MountainCarContinuous<R> {
    position: f32,
    velocity: f32,
    step_count: usize,
    rng: R,
}

impl<R: Rng> MountainCarContinuous<R> {
    /// Create a new MountainCarContinuous environment with the given RNG.
    pub fn new(rng: R) -> Self {
        let mut env = Self {
            position: 0.0,
            velocity: 0.0,
            step_count: 0,
            rng,
        };
        env.reset();
        env
    }
}

impl<R: Rng> Env for MountainCarContinuous<R> {
    type Observation = Vec<f32>;
    type Action = Vec<f32>;

    fn reset(&mut self) -> Vec<f32> {
        self.position = self.rng.random_range(-0.6..-0.4);
        self.velocity = 0.0;
        self.step_count = 0;
        vec![self.position, self.velocity]
    }

    fn step(&mut self, action: Vec<f32>) -> Step<Vec<f32>> {
        let force = action[0].clamp(-1.0, 1.0);

        self.velocity += force * POWER - (3.0 * self.position).cos() * GRAVITY;
        self.velocity = self.velocity.clamp(-MAX_SPEED, MAX_SPEED);
        self.position += self.velocity;
        self.position = self.position.clamp(MIN_POSITION, MAX_POSITION);

        if self.position <= MIN_POSITION && self.velocity < 0.0 {
            self.velocity = 0.0;
        }

        self.step_count += 1;

        let terminated = self.position >= GOAL_POSITION;
        let truncated = self.step_count >= MAX_STEPS;

        let reward = if terminated {
            100.0
        } else {
            0.0
        } - 0.1 * force * force;

        Step {
            observation: vec![self.position, self.velocity],
            reward,
            terminated,
            truncated,
        }
    }

    fn observation_space(&self) -> Space {
        Space::Box {
            low: vec![MIN_POSITION, -MAX_SPEED],
            high: vec![MAX_POSITION, MAX_SPEED],
        }
    }

    fn action_space(&self) -> Space {
        Space::Box {
            low: vec![-1.0],
            high: vec![1.0],
        }
    }
}

fn mountain_height(x: f32) -> f32 {
    (3.0 * x).sin() * 0.45 + 0.55
}

impl<R> Renderable for MountainCarContinuous<R> {
    fn render(&self) -> RgbFrame {
        let w: u16 = 600;
        let h: u16 = 400;
        let mut pixels = vec![255u8; w as usize * h as usize * 3];

        let map_x = |x: f32| -> i32 {
            ((x - MIN_POSITION) / (MAX_POSITION - MIN_POSITION) * w as f32) as i32
        };
        let map_y = |height: f32| -> i32 { (h as f32 * (1.0 - height)) as i32 };

        // Draw mountain
        for px in 0..w as i32 {
            let x = MIN_POSITION + (px as f32 / w as f32) * (MAX_POSITION - MIN_POSITION);
            let mh = mountain_height(x);
            let py = map_y(mh);
            draw_rect(&mut pixels, w, px, py, px + 1, h as i32, [180, 220, 140]);
        }

        // Goal flag
        let goal_px = map_x(GOAL_POSITION);
        let goal_py = map_y(mountain_height(GOAL_POSITION));
        draw_thick_line(&mut pixels, w, h, goal_px, goal_py, goal_px, goal_py - 40, 2, [40, 40, 40]);
        draw_rect(&mut pixels, w, goal_px, goal_py - 40, goal_px + 15, goal_py - 28, [200, 40, 40]);

        // Car
        let car_px = map_x(self.position);
        let car_py = map_y(mountain_height(self.position));
        draw_circle(&mut pixels, w, h, car_px, car_py - 8, 8, [200, 80, 40]);

        RgbFrame { width: w, height: h, data: pixels }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_env() -> MountainCarContinuous<rand::rngs::SmallRng> {
        MountainCarContinuous::new(rand::rngs::SmallRng::seed_from_u64(42))
    }

    #[test]
    fn reset_returns_2d_obs() {
        let mut env = make_env();
        let obs = env.reset();
        assert_eq!(obs.len(), 2);
        assert!(obs[0] >= -0.6 && obs[0] < -0.4);
        assert_eq!(obs[1], 0.0);
    }

    #[test]
    fn reward_includes_action_cost() {
        let mut env = make_env();
        env.reset();
        let step = env.step(vec![1.0]);
        // Not at goal, so reward = -0.1 * 1.0² = -0.1
        assert!(!step.terminated);
        assert!((step.reward - (-0.1)).abs() < 1e-5);
    }

    #[test]
    fn terminates_at_goal() {
        let mut env = make_env();
        env.reset();
        env.position = GOAL_POSITION + 0.01;
        env.velocity = 0.0;
        let step = env.step(vec![0.0]);
        assert!(step.terminated);
        assert!(step.reward >= 99.0); // 100 - small action cost
    }

    #[test]
    fn truncates_at_max_steps() {
        let mut env = make_env();
        env.reset();
        let mut truncated = false;
        for _ in 0..MAX_STEPS + 10 {
            let step = env.step(vec![0.0]);
            if step.done() {
                truncated = step.truncated;
                break;
            }
        }
        assert!(truncated);
    }

    #[test]
    fn spaces() {
        let env = make_env();
        assert_eq!(env.observation_space().flat_dim(), 2);
        assert_eq!(env.action_space().flat_dim(), 1);
    }

    #[test]
    fn render_produces_valid_frame() {
        let env = make_env();
        let frame = env.render();
        assert_eq!(frame.data.len(), frame.width as usize * frame.height as usize * 3);
    }
}
