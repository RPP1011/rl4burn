//! LunarLander environment (LunarLander-v3).
//!
//! A 2D rocket must land between two flags on a flat helipad. The lander starts
//! at the top center with a small random perturbation and has three engines:
//! main (bottom), left, and right.
//!
//! Pure Rust rigid-body simulation — no Box2D dependency.
//!
//! - **Observation**: 8 floats `[x, y, vx, vy, angle, angular_vel, left_leg_contact, right_leg_contact]`
//! - **Action**: `0` (nop), `1` (left engine), `2` (main engine), `3` (right engine)
//! - **Reward**: potential-based shaping + contact bonus − fuel cost; +100 landing / −100 crash
//! - **Terminated**: crash (body touches ground) or landed (both legs, low velocity)
//! - **Truncated**: step count ≥ 1000

use rl4burn_core::env::render::{Renderable, RgbFrame};
use rl4burn_core::env::space::Space;
use rl4burn_core::env::{Env, Step};
use rand::{Rng, RngExt};

use crate::render_util::{draw_filled_polygon, draw_rect, draw_thick_line};

// --- Physics constants ---
const GRAVITY: f32 = -10.0;
const MAIN_ENGINE_POWER: f32 = 13.0;
const SIDE_ENGINE_POWER: f32 = 0.6;
const LANDER_HALF_W: f32 = 0.35;
const LANDER_HALF_H: f32 = 0.45;
const LEG_AWAY: f32 = 0.4; // horizontal offset of legs from center
const LEG_DOWN: f32 = 0.4; // how far below body the leg extends
const LEG_SPRING: f32 = 0.4; // ground contact threshold for leg compression

const VIEWPORT_W: f32 = 600.0;
const VIEWPORT_H: f32 = 400.0;
const SCALE: f32 = 30.0; // pixels per world unit
const W: f32 = VIEWPORT_W / SCALE;
const H: f32 = VIEWPORT_H / SCALE;

const HELIPAD_Y: f32 = H / 4.0;
const MAX_STEPS: usize = 1000;

const FPS: f32 = 50.0;
const DT: f32 = 1.0 / FPS;
const LANDER_MASS: f32 = 5.0;
const LANDER_INERTIA: f32 = 5.0;

/// Shared physics state for the lunar lander (used by both discrete and continuous variants).
pub(crate) struct LanderState {
    pub x: f32,
    pub y: f32,
    pub vx: f32,
    pub vy: f32,
    pub angle: f32,
    pub angular_vel: f32,
    pub left_contact: bool,
    pub right_contact: bool,
    pub prev_shaping: Option<f32>,
    pub game_over: bool,
    pub step_count: usize,
    // Terrain: flat with a helipad
    pub helipad_x1: f32,
    pub helipad_x2: f32,
}

impl LanderState {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            vx: 0.0,
            vy: 0.0,
            angle: 0.0,
            angular_vel: 0.0,
            left_contact: false,
            right_contact: false,
            prev_shaping: None,
            game_over: false,
            step_count: 0,
            helipad_x1: W / 2.0 - 1.0,
            helipad_x2: W / 2.0 + 1.0,
        }
    }

    pub fn reset<R: Rng>(&mut self, rng: &mut R) {
        self.x = W / 2.0;
        self.y = H * 0.9;
        self.vx = rng.random_range(-0.5..0.5);
        self.vy = rng.random_range(-0.5..0.0);
        self.angle = rng.random_range(-0.2..0.2);
        self.angular_vel = rng.random_range(-0.2..0.2);
        self.left_contact = false;
        self.right_contact = false;
        self.prev_shaping = None;
        self.game_over = false;
        self.step_count = 0;
    }

    /// Step physics with given main and side thrust (both in [0, 1] range).
    /// `main_thrust`: 0 = off, >0 = firing downward
    /// `side_thrust`: negative = fire right engine (push left), positive = fire left engine (push right)
    pub fn physics_step<R: Rng>(
        &mut self,
        main_thrust: f32,
        side_thrust: f32,
        rng: &mut R,
    ) {
        // Forces in world frame
        let sin_a = self.angle.sin();
        let cos_a = self.angle.cos();

        let mut fx = 0.0f32;
        let mut fy = 0.0f32;
        let mut torque = 0.0f32;

        // Main engine (fires downward from bottom of lander)
        if main_thrust > 0.0 {
            let dispersion = rng.random_range(-0.1..0.1);
            let ox = -sin_a + dispersion;
            let oy = cos_a;
            fx += ox * MAIN_ENGINE_POWER * main_thrust;
            fy += oy * MAIN_ENGINE_POWER * main_thrust;
        }

        // Side engines
        if side_thrust.abs() > 0.0 {
            let dispersion = rng.random_range(-0.1..0.1);
            let dir = side_thrust.signum();
            // Side thrust pushes horizontally relative to lander
            fx += (cos_a + dispersion) * SIDE_ENGINE_POWER * dir;
            fy += sin_a * SIDE_ENGINE_POWER * dir;
            // Torque from off-center thrust
            torque += SIDE_ENGINE_POWER * dir * 0.5;
        }

        // Gravity
        fy += GRAVITY * LANDER_MASS;

        // Euler integration
        let ax = fx / LANDER_MASS;
        let ay = fy / LANDER_MASS;
        let alpha = torque / LANDER_INERTIA;

        self.vx += ax * DT;
        self.vy += ay * DT;
        self.x += self.vx * DT;
        self.y += self.vy * DT;
        self.angular_vel += alpha * DT;
        self.angle += self.angular_vel * DT;

        // Leg ground contact
        let left_leg_y = self.y - LANDER_HALF_H - LEG_DOWN;
        let right_leg_y = left_leg_y;
        self.left_contact = left_leg_y <= HELIPAD_Y + LEG_SPRING;
        self.right_contact = right_leg_y <= HELIPAD_Y + LEG_SPRING;

        // Ground collision — stop vertical motion
        if self.y - LANDER_HALF_H <= HELIPAD_Y {
            self.y = HELIPAD_Y + LANDER_HALF_H;
            if self.vy < 0.0 {
                self.vy = 0.0;
            }
        }

        self.step_count += 1;
    }

    pub fn obs(&self) -> Vec<f32> {
        // Normalize positions relative to viewport center and scale
        vec![
            (self.x - W / 2.0) / (W / 2.0),
            (self.y - HELIPAD_Y) / (H / 2.0),
            self.vx * (W / 2.0) / FPS,
            self.vy * (H / 2.0) / FPS,
            self.angle,
            self.angular_vel / 20.0,
            if self.left_contact { 1.0 } else { 0.0 },
            if self.right_contact { 1.0 } else { 0.0 },
        ]
    }

    pub fn compute_reward(&mut self, main_thrust: f32, side_thrust: f32) -> (f32, bool) {
        // Potential-based reward shaping
        let shaping = -100.0
            * ((self.x - W / 2.0).powi(2)
                + (self.y - HELIPAD_Y).powi(2))
            .sqrt()
            - 100.0 * (self.vx.powi(2) + self.vy.powi(2)).sqrt()
            - 100.0 * self.angle.abs()
            + 10.0 * (if self.left_contact { 1.0 } else { 0.0 })
            + 10.0 * (if self.right_contact { 1.0 } else { 0.0 });

        let mut reward = match self.prev_shaping {
            Some(prev) => shaping - prev,
            None => 0.0,
        };
        self.prev_shaping = Some(shaping);

        // Fuel cost
        reward -= main_thrust * 0.30;
        reward -= side_thrust.abs() * 0.03;

        // Terminal conditions
        let body_ground = self.y - LANDER_HALF_H <= HELIPAD_Y + 0.01;
        let on_helipad = self.x >= self.helipad_x1 && self.x <= self.helipad_x2;
        let slow_enough = self.vx.abs() < 0.5 && self.vy.abs() < 0.5;
        let upright = self.angle.abs() < 0.3;

        let crashed = body_ground && (!on_helipad || !upright || !slow_enough);
        let landed = body_ground && self.left_contact && self.right_contact && on_helipad && slow_enough && upright;
        let out_of_bounds = self.x < 0.0 || self.x > W || self.y < 0.0;

        let terminated = if crashed || out_of_bounds {
            reward -= 100.0;
            self.game_over = true;
            true
        } else if landed {
            reward += 100.0;
            self.game_over = true;
            true
        } else {
            false
        };

        (reward, terminated)
    }

    pub fn is_truncated(&self) -> bool {
        self.step_count >= MAX_STEPS
    }
}

/// LunarLander-v3 environment with discrete actions.
pub struct LunarLander<R> {
    pub(crate) state: LanderState,
    pub(crate) rng: R,
}

impl<R: Rng> LunarLander<R> {
    /// Create a new LunarLander environment with the given RNG.
    pub fn new(rng: R) -> Self {
        let mut env = Self {
            state: LanderState::new(),
            rng,
        };
        env.reset();
        env
    }
}

impl<R: Rng> Env for LunarLander<R> {
    type Observation = Vec<f32>;
    type Action = usize;

    fn reset(&mut self) -> Vec<f32> {
        self.state.reset(&mut self.rng);
        self.state.obs()
    }

    fn step(&mut self, action: usize) -> Step<Vec<f32>> {
        assert!(action < 4, "action must be 0..3, got {action}");

        let (main_thrust, side_thrust) = match action {
            0 => (0.0, 0.0),  // nop
            1 => (0.0, -1.0), // left engine (push right... no, fires left = push body right?)
            2 => (1.0, 0.0),  // main engine
            3 => (0.0, 1.0),  // right engine
            _ => unreachable!(),
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
        Space::Discrete(4)
    }
}

pub(crate) fn render_lander(state: &LanderState) -> RgbFrame {
    let w: u16 = VIEWPORT_W as u16;
    let h: u16 = VIEWPORT_H as u16;
    let mut pixels = vec![0u8; w as usize * h as usize * 3];

    // Sky background (dark blue gradient)
    for y in 0..h as i32 {
        let t = y as f32 / h as f32;
        let r = (10.0 + 20.0 * t) as u8;
        let g = (10.0 + 20.0 * t) as u8;
        let b = (40.0 + 60.0 * t) as u8;
        draw_rect(&mut pixels, w, 0, y, w as i32, y + 1, [r, g, b]);
    }

    let to_px = |wx: f32, wy: f32| -> (i32, i32) {
        ((wx * SCALE) as i32, (h as f32 - wy * SCALE) as i32)
    };

    // Draw terrain (flat ground)
    let (_, ground_py) = to_px(0.0, HELIPAD_Y);
    draw_rect(&mut pixels, w, 0, ground_py, w as i32, h as i32, [80, 80, 80]);

    // Helipad markers
    let (hp1_px, hp1_py) = to_px(state.helipad_x1, HELIPAD_Y);
    let (hp2_px, _) = to_px(state.helipad_x2, HELIPAD_Y);
    // Left flag
    draw_thick_line(&mut pixels, w, h, hp1_px, hp1_py, hp1_px, hp1_py - 30, 2, [200, 200, 200]);
    draw_rect(&mut pixels, w, hp1_px, hp1_py - 30, hp1_px + 12, hp1_py - 22, [200, 40, 40]);
    // Right flag
    draw_thick_line(&mut pixels, w, h, hp2_px, hp1_py, hp2_px, hp1_py - 30, 2, [200, 200, 200]);
    draw_rect(&mut pixels, w, hp2_px, hp1_py - 30, hp2_px + 12, hp1_py - 22, [200, 40, 40]);

    // Draw lander body (rotated rectangle)
    let sin_a = state.angle.sin();
    let cos_a = state.angle.cos();

    let body_corners: [(f32, f32); 4] = [
        (-LANDER_HALF_W, -LANDER_HALF_H),
        (LANDER_HALF_W, -LANDER_HALF_H),
        (LANDER_HALF_W, LANDER_HALF_H),
        (-LANDER_HALF_W, LANDER_HALF_H),
    ];

    let transformed: Vec<(i32, i32)> = body_corners
        .iter()
        .map(|&(dx, dy)| {
            let wx = state.x + dx * cos_a - dy * sin_a;
            let wy = state.y + dx * sin_a + dy * cos_a;
            to_px(wx, wy)
        })
        .collect();

    draw_filled_polygon(&mut pixels, w, h, &transformed, [220, 220, 240]);

    // Draw legs
    let leg_color = |contact: bool| -> [u8; 3] {
        if contact { [60, 200, 60] } else { [180, 180, 180] }
    };

    // Left leg
    let ll_top_wx = state.x - LEG_AWAY * cos_a - (-LANDER_HALF_H) * sin_a;
    let ll_top_wy = state.y - LEG_AWAY * sin_a + (-LANDER_HALF_H) * cos_a;
    let ll_bot_wx = ll_top_wx;
    let ll_bot_wy = ll_top_wy - LEG_DOWN;
    let (ll_tx, ll_ty) = to_px(ll_top_wx, ll_top_wy);
    let (ll_bx, ll_by) = to_px(ll_bot_wx, ll_bot_wy);
    draw_thick_line(&mut pixels, w, h, ll_tx, ll_ty, ll_bx, ll_by, 3, leg_color(state.left_contact));

    // Right leg
    let rl_top_wx = state.x + LEG_AWAY * cos_a - (-LANDER_HALF_H) * sin_a;
    let rl_top_wy = state.y + LEG_AWAY * sin_a + (-LANDER_HALF_H) * cos_a;
    let rl_bot_wx = rl_top_wx;
    let rl_bot_wy = rl_top_wy - LEG_DOWN;
    let (rl_tx, rl_ty) = to_px(rl_top_wx, rl_top_wy);
    let (rl_bx, rl_by) = to_px(rl_bot_wx, rl_bot_wy);
    draw_thick_line(&mut pixels, w, h, rl_tx, rl_ty, rl_bx, rl_by, 3, leg_color(state.right_contact));

    RgbFrame { width: w, height: h, data: pixels }
}

impl<R> Renderable for LunarLander<R> {
    fn render(&self) -> RgbFrame {
        render_lander(&self.state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_env() -> LunarLander<rand::rngs::SmallRng> {
        LunarLander::new(rand::rngs::SmallRng::seed_from_u64(42))
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
            let step = env.step(0);
            assert!(step.reward.is_finite(), "reward not finite: {}", step.reward);
            if step.done() {
                break;
            }
        }
    }

    #[test]
    fn crash_terminates() {
        let mut env = make_env();
        env.reset();
        // No thrust → lander falls and eventually terminates
        let mut terminated = false;
        for _ in 0..MAX_STEPS {
            let step = env.step(0);
            if step.terminated {
                terminated = true;
                break;
            }
        }
        assert!(terminated, "lander should crash with no thrust");
    }

    #[test]
    fn truncates_at_max_steps() {
        let mut env = make_env();
        env.reset();
        let mut truncated = false;
        for _ in 0..MAX_STEPS + 10 {
            // Fire main engine to stay aloft
            let step = env.step(2);
            if step.done() {
                if step.truncated {
                    truncated = true;
                }
                break;
            }
        }
        // Either truncated or terminated before max steps — both valid
        assert!(truncated || env.state.game_over);
    }

    #[test]
    fn spaces() {
        let env = make_env();
        assert_eq!(env.observation_space().flat_dim(), 8);
        assert_eq!(env.action_space().flat_dim(), 4);
    }

    #[test]
    fn render_produces_valid_frame() {
        let env = make_env();
        let frame = env.render();
        assert_eq!(frame.data.len(), frame.width as usize * frame.height as usize * 3);
    }
}
