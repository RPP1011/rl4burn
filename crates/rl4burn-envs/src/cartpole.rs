//! Classic CartPole environment (CartPole-v1).
//!
//! A pole is attached by an un-actuated joint to a cart moving along a
//! frictionless track. The pendulum starts upright and the goal is to
//! prevent it from falling by applying force left or right.
//!
//! - **Observation**: `[x, x_dot, theta, theta_dot]` (4 floats)
//! - **Action**: `0` (push left) or `1` (push right)
//! - **Reward**: +1.0 per step
//! - **Terminated**: `|x| > 2.4` or `|theta| > 12°`
//! - **Truncated**: step count ≥ 500

use rl4burn_core::env::{Env, Step};
use rl4burn_core::env::space::Space;
use rand::{Rng, RngExt};

const GRAVITY: f32 = 9.8;
const MASS_CART: f32 = 1.0;
const MASS_POLE: f32 = 0.1;
const TOTAL_MASS: f32 = MASS_CART + MASS_POLE;
const HALF_POLE_LENGTH: f32 = 0.5;
const POLE_MASS_LENGTH: f32 = MASS_POLE * HALF_POLE_LENGTH;
const FORCE_MAG: f32 = 10.0;
const TAU: f32 = 0.02;

const X_THRESHOLD: f32 = 2.4;
const THETA_THRESHOLD: f32 = 12.0 * std::f32::consts::PI / 180.0;
const MAX_STEPS: usize = 500;

/// Classic CartPole-v1 environment.
pub struct CartPole<R> {
    state: [f32; 4],
    step_count: usize,
    rng: R,
}

impl<R: Rng> CartPole<R> {
    /// Create a new CartPole environment with the given RNG.
    pub fn new(rng: R) -> Self {
        let mut env = Self {
            state: [0.0; 4],
            step_count: 0,
            rng,
        };
        env.reset();
        env
    }
}

impl<R: Rng> Env for CartPole<R> {
    type Observation = Vec<f32>;
    type Action = usize;

    fn reset(&mut self) -> Vec<f32> {
        for s in &mut self.state {
            *s = self.rng.random_range(-0.05..0.05);
        }
        self.step_count = 0;
        self.state.to_vec()
    }

    fn step(&mut self, action: usize) -> Step<Vec<f32>> {
        assert!(action < 2, "action must be 0 or 1, got {action}");

        let [x, x_dot, theta, theta_dot] = self.state;

        let force = if action == 1 { FORCE_MAG } else { -FORCE_MAG };
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        let temp = (force + POLE_MASS_LENGTH * theta_dot * theta_dot * sin_theta) / TOTAL_MASS;
        let theta_acc = (GRAVITY * sin_theta - cos_theta * temp)
            / (HALF_POLE_LENGTH * (4.0 / 3.0 - MASS_POLE * cos_theta * cos_theta / TOTAL_MASS));
        let x_acc = temp - POLE_MASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS;

        self.state = [
            x + TAU * x_dot,
            x_dot + TAU * x_acc,
            theta + TAU * theta_dot,
            theta_dot + TAU * theta_acc,
        ];
        self.step_count += 1;

        let terminated =
            self.state[0].abs() > X_THRESHOLD || self.state[2].abs() > THETA_THRESHOLD;
        let truncated = self.step_count >= MAX_STEPS;

        Step {
            observation: self.state.to_vec(),
            reward: 1.0,
            terminated,
            truncated,
        }
    }

    fn observation_space(&self) -> Space {
        Space::Box {
            low: vec![-4.8, -f32::INFINITY, -0.418, -f32::INFINITY],
            high: vec![4.8, f32::INFINITY, 0.418, f32::INFINITY],
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(2)
    }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

use rl4burn_core::env::render::{Renderable, RgbFrame};
use crate::render_util::{draw_rect, draw_circle, draw_thick_line};

impl<R> Renderable for CartPole<R> {
    fn render(&self) -> RgbFrame {
        let w: u16 = 600;
        let h: u16 = 400;
        let mut pixels = vec![255u8; w as usize * h as usize * 3]; // white bg

        let [x, _, theta, _] = self.state;

        // Coordinate mapping: cart x in [-2.4, 2.4] -> pixel x
        let scale = w as f32 / 4.8;
        let cart_cx = (w as f32 / 2.0 + x * scale) as i32;
        let ground_y: i32 = 300;

        // Draw ground line
        draw_rect(&mut pixels, w, 0, ground_y, w as i32, ground_y + 2, [180, 180, 180]);

        // Cart: 80×30 rectangle
        let cart_w = 80i32;
        let cart_h = 30i32;
        let cart_top = ground_y - cart_h;
        draw_rect(
            &mut pixels, w,
            cart_cx - cart_w / 2, cart_top,
            cart_cx + cart_w / 2, ground_y,
            [30, 30, 180],
        );

        // Pole: line from cart top center, length 150px, angle theta from vertical
        let pole_len = 150.0f32;
        let pole_base_x = cart_cx as f32;
        let pole_base_y = cart_top as f32;
        let pole_tip_x = pole_base_x + pole_len * theta.sin();
        let pole_tip_y = pole_base_y - pole_len * theta.cos();
        draw_thick_line(
            &mut pixels, w, h,
            pole_base_x as i32, pole_base_y as i32,
            pole_tip_x as i32, pole_tip_y as i32,
            3,
            [180, 60, 30],
        );

        // Axle dot
        draw_circle(&mut pixels, w, h, cart_cx, cart_top, 5, [40, 40, 40]);

        // Pole tip dot
        draw_circle(&mut pixels, w, h, pole_tip_x as i32, pole_tip_y as i32, 4, [200, 40, 20]);

        RgbFrame { width: w, height: h, data: pixels }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_env() -> CartPole<rand::rngs::SmallRng> {
        use rand::SeedableRng;
        CartPole::new(rand::rngs::SmallRng::seed_from_u64(42))
    }

    #[test]
    fn reset_returns_4d_obs() {
        let mut env = make_env();
        let obs = env.reset();
        assert_eq!(obs.len(), 4);
        // Initial state should be near zero
        for &v in &obs {
            assert!((v as f32).abs() < 0.1, "initial state too large: {v}");
        }
    }

    #[test]
    fn step_returns_reward_1() {
        let mut env = make_env();
        env.reset();
        let step = env.step(0);
        assert_eq!(step.reward, 1.0);
        assert_eq!(step.observation.len(), 4);
    }

    #[test]
    fn terminates_on_large_x() {
        let mut env = make_env();
        env.reset();
        // Set state far past threshold so it terminates after one step
        env.state = [10.0, 0.0, 0.0, 0.0];
        let step = env.step(0);
        assert!(step.terminated);
    }

    #[test]
    fn terminates_on_large_theta() {
        let mut env = make_env();
        env.reset();
        env.state = [0.0, 0.0, 1.0, 0.0]; // ~57 degrees, way past 12
        let step = env.step(0);
        assert!(step.terminated);
    }

    #[test]
    fn truncates_at_max_steps() {
        let mut env = make_env();
        env.reset();
        // Force upright state to avoid termination
        let mut truncated = false;
        for _ in 0..MAX_STEPS + 10 {
            env.state = [0.0, 0.0, 0.0, 0.0]; // keep balanced
            let step = env.step(1);
            if step.truncated {
                truncated = true;
                break;
            }
        }
        assert!(truncated, "should truncate at max steps");
    }

    #[test]
    fn spaces() {
        let env = make_env();
        assert_eq!(env.observation_space().flat_dim(), 4);
        assert_eq!(env.action_space().flat_dim(), 2);
    }
}
