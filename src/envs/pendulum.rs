//! Classic Pendulum environment (Pendulum-v1).
//!
//! A frictionless pendulum starts at a random angle. The goal is to swing
//! it up and balance it upright by applying torque.
//!
//! - **Observation**: `[cos(θ), sin(θ), θ_dot]` (3 floats)
//! - **Action**: `[torque]` in `[-2.0, 2.0]` (continuous, 1 float)
//! - **Reward**: `-(θ² + 0.1·θ̇² + 0.001·u²)` — near zero when balanced
//! - **Terminated**: never (episode only truncates)
//! - **Truncated**: step count ≥ 200
//!
//! Matches Gymnasium's Pendulum-v1 dynamics.

use crate::env::render::{Renderable, RgbFrame};
use crate::env::space::Space;
use crate::env::{Env, Step};
use rand::{Rng, RngExt};
use std::f32::consts::PI;

const GRAVITY: f32 = 10.0;
const MASS: f32 = 1.0;
const LENGTH: f32 = 1.0;
const DT: f32 = 0.05;
const MAX_TORQUE: f32 = 2.0;
const MAX_SPEED: f32 = 8.0;
const MAX_STEPS: usize = 200;

/// Classic Pendulum-v1 environment with continuous torque control.
pub struct Pendulum<R> {
    theta: f32,
    theta_dot: f32,
    step_count: usize,
    rng: R,
}

impl<R: Rng> Pendulum<R> {
    /// Create a new Pendulum environment with the given RNG.
    pub fn new(rng: R) -> Self {
        let mut env = Self {
            theta: 0.0,
            theta_dot: 0.0,
            step_count: 0,
            rng,
        };
        env.reset();
        env
    }

    fn obs(&self) -> Vec<f32> {
        vec![self.theta.cos(), self.theta.sin(), self.theta_dot]
    }
}

fn angle_normalize(x: f32) -> f32 {
    ((x + PI).rem_euclid(2.0 * PI)) - PI
}

impl<R: Rng> Env for Pendulum<R> {
    type Observation = Vec<f32>;
    type Action = Vec<f32>;

    fn reset(&mut self) -> Vec<f32> {
        self.theta = self.rng.random_range(-PI..PI);
        self.theta_dot = self.rng.random_range(-1.0..1.0);
        self.step_count = 0;
        self.obs()
    }

    fn step(&mut self, action: Vec<f32>) -> Step<Vec<f32>> {
        let u = action[0].clamp(-MAX_TORQUE, MAX_TORQUE);
        let th = self.theta;
        let thdot = self.theta_dot;

        // Pendulum dynamics (matches Gymnasium)
        let new_thdot = thdot
            + (3.0 * GRAVITY / (2.0 * LENGTH) * th.sin()
                + 3.0 / (MASS * LENGTH * LENGTH) * u)
                * DT;
        let new_thdot = new_thdot.clamp(-MAX_SPEED, MAX_SPEED);
        let new_th = th + new_thdot * DT;

        // Cost = normalized_angle² + 0.1·ω² + 0.001·u²
        let norm_th = angle_normalize(th);
        let cost = norm_th * norm_th + 0.1 * thdot * thdot + 0.001 * u * u;

        self.theta = new_th;
        self.theta_dot = new_thdot;
        self.step_count += 1;

        Step {
            observation: self.obs(),
            reward: -cost,
            terminated: false,
            truncated: self.step_count >= MAX_STEPS,
        }
    }

    fn observation_space(&self) -> Space {
        Space::Box {
            low: vec![-1.0, -1.0, -MAX_SPEED],
            high: vec![1.0, 1.0, MAX_SPEED],
        }
    }

    fn action_space(&self) -> Space {
        Space::Box {
            low: vec![-MAX_TORQUE],
            high: vec![MAX_TORQUE],
        }
    }
}

impl<R> Renderable for Pendulum<R> {
    fn render(&self) -> RgbFrame {
        let w: u16 = 400;
        let h: u16 = 400;
        let mut pixels = vec![255u8; w as usize * h as usize * 3]; // white bg

        let cx = w as i32 / 2;
        let cy = h as i32 / 2;
        let pole_len = 150i32;

        // Pole tip: θ=0 means pointing UP
        let tip_x = cx + (pole_len as f32 * self.theta.sin()) as i32;
        let tip_y = cy + (pole_len as f32 * self.theta.cos()) as i32;

        // Draw pole
        draw_thick_line(&mut pixels, w, h, cx, cy, tip_x, tip_y, 4, [180, 60, 30]);

        // Draw pivot
        draw_circle(&mut pixels, w, h, cx, cy, 6, [60, 60, 60]);

        // Draw bob
        draw_circle(&mut pixels, w, h, tip_x, tip_y, 12, [30, 30, 180]);

        RgbFrame {
            width: w,
            height: h,
            data: pixels,
        }
    }
}

fn draw_circle(
    pixels: &mut [u8],
    canvas_w: u16,
    canvas_h: u16,
    cx: i32,
    cy: i32,
    r: i32,
    color: [u8; 3],
) {
    let w = canvas_w as i32;
    let h = canvas_h as i32;
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                let px = cx + dx;
                let py = cy + dy;
                if px >= 0 && px < w && py >= 0 && py < h {
                    let idx = (py * w + px) as usize * 3;
                    pixels[idx..idx + 3].copy_from_slice(&color);
                }
            }
        }
    }
}

fn draw_thick_line(
    pixels: &mut [u8],
    canvas_w: u16,
    canvas_h: u16,
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    thickness: i32,
    color: [u8; 3],
) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut cx = x0;
    let mut cy = y0;
    let w = canvas_w as i32;
    let h = canvas_h as i32;

    loop {
        for oy in -(thickness / 2)..=(thickness / 2) {
            for ox in -(thickness / 2)..=(thickness / 2) {
                let px = cx + ox;
                let py = cy + oy;
                if px >= 0 && px < w && py >= 0 && py < h {
                    let idx = (py * w + px) as usize * 3;
                    pixels[idx..idx + 3].copy_from_slice(&color);
                }
            }
        }
        if cx == x1 && cy == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            cx += sx;
        }
        if e2 <= dx {
            err += dx;
            cy += sy;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_env() -> Pendulum<rand::rngs::SmallRng> {
        Pendulum::new(rand::rngs::SmallRng::seed_from_u64(42))
    }

    #[test]
    fn reset_returns_3d_obs() {
        let mut env = make_env();
        let obs = env.reset();
        assert_eq!(obs.len(), 3);
        // cos²θ + sin²θ = 1
        let sum_sq = obs[0] * obs[0] + obs[1] * obs[1];
        assert!((sum_sq - 1.0).abs() < 1e-5);
    }

    #[test]
    fn reward_is_negative() {
        let mut env = make_env();
        env.reset();
        let step = env.step(vec![0.0]);
        assert!(step.reward <= 0.0);
    }

    #[test]
    fn never_terminates() {
        let mut env = make_env();
        env.reset();
        for _ in 0..MAX_STEPS - 1 {
            let step = env.step(vec![0.0]);
            assert!(!step.terminated);
        }
    }

    #[test]
    fn truncates_at_max_steps() {
        let mut env = make_env();
        env.reset();
        let mut truncated = false;
        for _ in 0..MAX_STEPS + 10 {
            let step = env.step(vec![0.0]);
            if step.truncated {
                truncated = true;
                break;
            }
        }
        assert!(truncated);
    }

    #[test]
    fn action_clamps_to_bounds() {
        let mut env = make_env();
        env.reset();
        // Large action should be clamped internally — no panic
        let step = env.step(vec![100.0]);
        assert!(step.reward.is_finite());
    }

    #[test]
    fn render_produces_valid_frame() {
        let env = make_env();
        let frame = env.render();
        assert_eq!(
            frame.data.len(),
            frame.width as usize * frame.height as usize * 3
        );
    }

    #[test]
    fn spaces() {
        let env = make_env();
        assert_eq!(env.observation_space().flat_dim(), 3);
        assert_eq!(env.action_space().flat_dim(), 1);
    }
}
