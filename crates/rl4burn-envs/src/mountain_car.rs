//! Classic MountainCar environment (MountainCar-v0).
//!
//! A car is on a one-dimensional track between two hills. The goal is to
//! drive up the hill on the right, but the car's engine is not strong
//! enough to do so directly — it must build momentum by rocking back and forth.
//!
//! - **Observation**: `[position, velocity]` (2 floats)
//! - **Action**: `0` (push left), `1` (no push), `2` (push right)
//! - **Reward**: -1.0 per step
//! - **Terminated**: position ≥ 0.5
//! - **Truncated**: step count ≥ 200

use rl4burn_core::env::render::{Renderable, RgbFrame};
use rl4burn_core::env::space::Space;
use rl4burn_core::env::{Env, Step};
use rand::{Rng, RngExt};

use crate::render_util::{draw_circle, draw_rect, draw_thick_line};

const MIN_POSITION: f32 = -1.2;
const MAX_POSITION: f32 = 0.6;
const MAX_SPEED: f32 = 0.07;
const GOAL_POSITION: f32 = 0.5;
const FORCE: f32 = 0.001;
const GRAVITY: f32 = 0.0025;
const MAX_STEPS: usize = 200;

/// Classic MountainCar-v0 environment with discrete actions.
pub struct MountainCar<R> {
    position: f32,
    velocity: f32,
    step_count: usize,
    rng: R,
}

impl<R: Rng> MountainCar<R> {
    /// Create a new MountainCar environment with the given RNG.
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

impl<R: Rng> Env for MountainCar<R> {
    type Observation = Vec<f32>;
    type Action = usize;

    fn reset(&mut self) -> Vec<f32> {
        self.position = self.rng.random_range(-0.6..-0.4);
        self.velocity = 0.0;
        self.step_count = 0;
        vec![self.position, self.velocity]
    }

    fn step(&mut self, action: usize) -> Step<Vec<f32>> {
        assert!(action < 3, "action must be 0, 1, or 2, got {action}");

        let force = (action as f32 - 1.0) * FORCE;
        self.velocity += force - (3.0 * self.position).cos() * GRAVITY;
        self.velocity = self.velocity.clamp(-MAX_SPEED, MAX_SPEED);
        self.position += self.velocity;
        self.position = self.position.clamp(MIN_POSITION, MAX_POSITION);

        // Zero velocity at left wall
        if self.position <= MIN_POSITION && self.velocity < 0.0 {
            self.velocity = 0.0;
        }

        self.step_count += 1;

        let terminated = self.position >= GOAL_POSITION;
        let truncated = self.step_count >= MAX_STEPS;

        Step {
            observation: vec![self.position, self.velocity],
            reward: -1.0,
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
        Space::Discrete(3)
    }
}

fn mountain_height(x: f32) -> f32 {
    (3.0 * x).sin() * 0.45 + 0.55
}

impl<R> Renderable for MountainCar<R> {
    fn render(&self) -> RgbFrame {
        let w: u16 = 600;
        let h: u16 = 400;
        let mut pixels = vec![255u8; w as usize * h as usize * 3];

        // Draw mountain curve
        let map_x = |x: f32| -> i32 {
            ((x - MIN_POSITION) / (MAX_POSITION - MIN_POSITION) * w as f32) as i32
        };
        let map_y = |height: f32| -> i32 { (h as f32 * (1.0 - height)) as i32 };

        for px in 0..w as i32 {
            let x = MIN_POSITION + (px as f32 / w as f32) * (MAX_POSITION - MIN_POSITION);
            let mh = mountain_height(x);
            let py = map_y(mh);
            // Fill from mountain surface to bottom
            draw_rect(&mut pixels, w, px, py, px + 1, h as i32, [180, 220, 140]);
        }

        // Draw goal flag
        let goal_px = map_x(GOAL_POSITION);
        let goal_py = map_y(mountain_height(GOAL_POSITION));
        draw_thick_line(&mut pixels, w, h, goal_px, goal_py, goal_px, goal_py - 40, 2, [40, 40, 40]);
        draw_rect(&mut pixels, w, goal_px, goal_py - 40, goal_px + 15, goal_py - 28, [200, 40, 40]);

        // Draw car
        let car_px = map_x(self.position);
        let car_py = map_y(mountain_height(self.position));
        draw_circle(&mut pixels, w, h, car_px, car_py - 8, 8, [40, 40, 200]);

        RgbFrame { width: w, height: h, data: pixels }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_env() -> MountainCar<rand::rngs::SmallRng> {
        MountainCar::new(rand::rngs::SmallRng::seed_from_u64(42))
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
    fn reward_is_negative_one() {
        let mut env = make_env();
        env.reset();
        let step = env.step(1);
        assert_eq!(step.reward, -1.0);
    }

    #[test]
    fn terminates_at_goal() {
        let mut env = make_env();
        env.reset();
        env.position = GOAL_POSITION + 0.01;
        env.velocity = 0.0;
        let step = env.step(2);
        assert!(step.terminated);
    }

    #[test]
    fn truncates_at_max_steps() {
        let mut env = make_env();
        env.reset();
        let mut truncated = false;
        for _ in 0..MAX_STEPS + 10 {
            let step = env.step(1); // no push
            if step.done() {
                truncated = step.truncated;
                break;
            }
        }
        assert!(truncated);
    }

    #[test]
    fn velocity_zeroed_at_left_wall() {
        let mut env = make_env();
        env.reset();
        env.position = MIN_POSITION;
        env.velocity = -0.05;
        env.step(0); // push left
        assert!(env.velocity >= 0.0);
    }

    #[test]
    fn spaces() {
        let env = make_env();
        assert_eq!(env.observation_space().flat_dim(), 2);
        assert_eq!(env.action_space().flat_dim(), 3);
    }

    #[test]
    fn render_produces_valid_frame() {
        let env = make_env();
        let frame = env.render();
        assert_eq!(frame.data.len(), frame.width as usize * frame.height as usize * 3);
    }
}
