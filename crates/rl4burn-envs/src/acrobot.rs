//! Classic Acrobot environment (Acrobot-v1).
//!
//! A two-link pendulum with only the second joint actuated. The goal is to
//! swing the tip above the target height by applying torque to the elbow.
//!
//! - **Observation**: `[cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot]` (6 floats)
//! - **Action**: `0` (negative torque), `1` (none), `2` (positive torque)
//! - **Reward**: -1.0 per step
//! - **Terminated**: `-cos(θ1) - cos(θ1 + θ2) > 1.0`
//! - **Truncated**: step count ≥ 500

use rl4burn_core::env::render::{Renderable, RgbFrame};
use rl4burn_core::env::space::Space;
use rl4burn_core::env::{Env, Step};
use rand::{Rng, RngExt};

use crate::render_util::{draw_circle, draw_rect, draw_thick_line};

// Physical constants (matching Gymnasium)
const LINK_LENGTH_1: f32 = 1.0;
const LINK_LENGTH_2: f32 = 1.0;
const LINK_MASS_1: f32 = 1.0;
const LINK_MASS_2: f32 = 1.0;
const LINK_COM_POS_1: f32 = 0.5;
const LINK_COM_POS_2: f32 = 0.5;
const LINK_MOI: f32 = 1.0;
const GRAVITY: f32 = 9.8;
const DT: f32 = 0.2;
const MAX_VEL_1: f32 = 4.0 * std::f32::consts::PI;
const MAX_VEL_2: f32 = 9.0 * std::f32::consts::PI;
const TORQUE_MAG: f32 = 1.0;
const MAX_STEPS: usize = 500;

/// Classic Acrobot-v1 environment.
pub struct Acrobot<R> {
    theta1: f32,
    theta2: f32,
    theta1_dot: f32,
    theta2_dot: f32,
    step_count: usize,
    rng: R,
}

impl<R: Rng> Acrobot<R> {
    /// Create a new Acrobot environment with the given RNG.
    pub fn new(rng: R) -> Self {
        let mut env = Self {
            theta1: 0.0,
            theta2: 0.0,
            theta1_dot: 0.0,
            theta2_dot: 0.0,
            step_count: 0,
            rng,
        };
        env.reset();
        env
    }

    fn obs(&self) -> Vec<f32> {
        vec![
            self.theta1.cos(),
            self.theta1.sin(),
            self.theta2.cos(),
            self.theta2.sin(),
            self.theta1_dot,
            self.theta2_dot,
        ]
    }
}

/// Compute derivatives of the acrobot state for RK4 integration.
fn dsdt(state: [f32; 4], torque: f32) -> [f32; 4] {
    let [theta1, theta2, dtheta1, dtheta2] = state;

    let m1 = LINK_MASS_1;
    let m2 = LINK_MASS_2;
    let l1 = LINK_LENGTH_1;
    let lc1 = LINK_COM_POS_1;
    let lc2 = LINK_COM_POS_2;
    let i1 = LINK_MOI;
    let i2 = LINK_MOI;
    let g = GRAVITY;

    let d1 = m1 * lc1 * lc1
        + m2 * (l1 * l1 + lc2 * lc2 + 2.0 * l1 * lc2 * theta2.cos())
        + i1
        + i2;
    let d2 = m2 * (lc2 * lc2 + l1 * lc2 * theta2.cos()) + i2;

    let phi2 = m2 * lc2 * g * (theta1 + theta2).cos();
    let phi1 = -m2 * l1 * lc2 * dtheta2 * dtheta2 * theta2.sin()
        - 2.0 * m2 * l1 * lc2 * dtheta2 * dtheta1 * theta2.sin()
        + (m1 * lc1 + m2 * l1) * g * theta1.cos()
        + phi2;

    let ddtheta2 = (torque + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 * dtheta1 * theta2.sin() - phi2)
        / (m2 * lc2 * lc2 + i2 - d2 * d2 / d1);
    let ddtheta1 = -(d2 * ddtheta2 + phi1) / d1;

    [dtheta1, dtheta2, ddtheta1, ddtheta2]
}

/// RK4 integration step.
fn rk4_step(state: [f32; 4], torque: f32, dt: f32) -> [f32; 4] {
    let k1 = dsdt(state, torque);

    let mut s2 = [0.0f32; 4];
    for i in 0..4 {
        s2[i] = state[i] + 0.5 * dt * k1[i];
    }
    let k2 = dsdt(s2, torque);

    let mut s3 = [0.0f32; 4];
    for i in 0..4 {
        s3[i] = state[i] + 0.5 * dt * k2[i];
    }
    let k3 = dsdt(s3, torque);

    let mut s4 = [0.0f32; 4];
    for i in 0..4 {
        s4[i] = state[i] + dt * k3[i];
    }
    let k4 = dsdt(s4, torque);

    let mut result = [0.0f32; 4];
    for i in 0..4 {
        result[i] = state[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    result
}

fn wrap(x: f32) -> f32 {
    use std::f32::consts::PI;
    ((x + PI).rem_euclid(2.0 * PI)) - PI
}

impl<R: Rng> Env for Acrobot<R> {
    type Observation = Vec<f32>;
    type Action = usize;

    fn reset(&mut self) -> Vec<f32> {
        self.theta1 = self.rng.random_range(-0.1..0.1);
        self.theta2 = self.rng.random_range(-0.1..0.1);
        self.theta1_dot = self.rng.random_range(-0.1..0.1);
        self.theta2_dot = self.rng.random_range(-0.1..0.1);
        self.step_count = 0;
        self.obs()
    }

    fn step(&mut self, action: usize) -> Step<Vec<f32>> {
        assert!(action < 3, "action must be 0, 1, or 2, got {action}");

        let torque = (action as f32 - 1.0) * TORQUE_MAG;

        let state = [self.theta1, self.theta2, self.theta1_dot, self.theta2_dot];
        let new_state = rk4_step(state, torque, DT);

        self.theta1 = wrap(new_state[0]);
        self.theta2 = wrap(new_state[1]);
        self.theta1_dot = new_state[2].clamp(-MAX_VEL_1, MAX_VEL_1);
        self.theta2_dot = new_state[3].clamp(-MAX_VEL_2, MAX_VEL_2);
        self.step_count += 1;

        let terminated = -self.theta1.cos() - (self.theta1 + self.theta2).cos() > 1.0;
        let truncated = self.step_count >= MAX_STEPS;

        Step {
            observation: self.obs(),
            reward: -1.0,
            terminated,
            truncated,
        }
    }

    fn observation_space(&self) -> Space {
        Space::Box {
            low: vec![-1.0, -1.0, -1.0, -1.0, -MAX_VEL_1, -MAX_VEL_2],
            high: vec![1.0, 1.0, 1.0, 1.0, MAX_VEL_1, MAX_VEL_2],
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(3)
    }
}

impl<R> Renderable for Acrobot<R> {
    fn render(&self) -> RgbFrame {
        let w: u16 = 500;
        let h: u16 = 500;
        let mut pixels = vec![255u8; w as usize * h as usize * 3];

        let cx = w as i32 / 2;
        let cy = h as i32 / 3; // pivot near top
        let scale = 100.0f32;

        // First link
        let x1 = cx + (scale * LINK_LENGTH_1 * self.theta1.sin()) as i32;
        let y1 = cy + (scale * LINK_LENGTH_1 * self.theta1.cos()) as i32;

        // Second link
        let x2 = x1 + (scale * LINK_LENGTH_2 * (self.theta1 + self.theta2).sin()) as i32;
        let y2 = y1 + (scale * LINK_LENGTH_2 * (self.theta1 + self.theta2).cos()) as i32;

        // Target height line: -cos(θ1) - cos(θ1+θ2) > 1.0 means tip above pivot - LINK_LENGTH_1
        let target_y = cy - (scale * LINK_LENGTH_1) as i32;
        draw_rect(&mut pixels, w, 0, target_y, w as i32, target_y + 1, [200, 60, 60]);

        // Draw links
        draw_thick_line(&mut pixels, w, h, cx, cy, x1, y1, 5, [30, 30, 180]);
        draw_thick_line(&mut pixels, w, h, x1, y1, x2, y2, 5, [30, 180, 30]);

        // Joints
        draw_circle(&mut pixels, w, h, cx, cy, 6, [60, 60, 60]);
        draw_circle(&mut pixels, w, h, x1, y1, 5, [60, 60, 60]);
        draw_circle(&mut pixels, w, h, x2, y2, 4, [200, 40, 40]);

        RgbFrame { width: w, height: h, data: pixels }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_env() -> Acrobot<rand::rngs::SmallRng> {
        Acrobot::new(rand::rngs::SmallRng::seed_from_u64(42))
    }

    #[test]
    fn reset_returns_6d_obs() {
        let mut env = make_env();
        let obs = env.reset();
        assert_eq!(obs.len(), 6);
        // cos²+sin² ≈ 1 for both angles
        let cs1 = obs[0] * obs[0] + obs[1] * obs[1];
        assert!((cs1 - 1.0).abs() < 1e-4);
        let cs2 = obs[2] * obs[2] + obs[3] * obs[3];
        assert!((cs2 - 1.0).abs() < 1e-4);
    }

    #[test]
    fn reward_is_negative_one() {
        let mut env = make_env();
        env.reset();
        let step = env.step(1);
        assert_eq!(step.reward, -1.0);
    }

    #[test]
    fn terminates_when_tip_above_threshold() {
        let mut env = make_env();
        env.reset();
        // Verify termination condition: -cos(θ1) - cos(θ1+θ2) > 1.0
        // θ1 = π → -cos(π) = 1.0; θ2 = π → -cos(2π) = -1.0 → total = 0 (not terminated)
        // θ1 = π, θ2 = 0 → -cos(π) - cos(π) = 1 + 1 = 2 > 1.0 (terminated!)
        // After RK4 step, state drifts, so check the condition holds post-step
        env.theta1 = std::f32::consts::PI;
        env.theta2 = 0.0;
        env.theta1_dot = 0.0;
        env.theta2_dot = 0.0;
        // Condition before step: -cos(π) - cos(π+0) = 1+1 = 2 > 1 ✓
        // After a no-op step, should still be above threshold
        let step = env.step(1);
        assert!(step.terminated);
    }

    #[test]
    fn truncates_at_max_steps() {
        let mut env = make_env();
        env.reset();
        let mut truncated = false;
        for _ in 0..MAX_STEPS + 10 {
            // Keep near hanging-down state to avoid termination
            env.theta1 = 0.0;
            env.theta2 = 0.0;
            env.theta1_dot = 0.0;
            env.theta2_dot = 0.0;
            let step = env.step(1);
            if step.truncated {
                truncated = true;
                break;
            }
        }
        assert!(truncated);
    }

    #[test]
    fn spaces() {
        let env = make_env();
        assert_eq!(env.observation_space().flat_dim(), 6);
        assert_eq!(env.action_space().flat_dim(), 3);
    }

    #[test]
    fn render_produces_valid_frame() {
        let env = make_env();
        let frame = env.render();
        assert_eq!(frame.data.len(), frame.width as usize * frame.height as usize * 3);
    }
}
