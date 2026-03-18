//! Simple grid-world environment for testing multi-discrete and masking abstractions.
//!
//! A 7×7 grid where the agent starts at a random position and must reach a
//! fixed goal. Walls block movement; stepping into a wall is a no-op.
//!
//! - **Observation**: `[agent_row, agent_col, goal_row, goal_col]` normalized to [0, 1]
//! - **Action**: `0`=up, `1`=right, `2`=down, `3`=left
//! - **Reward**: +1.0 on reaching goal, -0.01 per step
//! - **Terminated**: agent reaches goal
//! - **Truncated**: step count ≥ 100

use rl4burn_core::env::render::{Renderable, RgbFrame};
use rl4burn_core::env::space::Space;
use rl4burn_core::env::{Env, Step};
use rand::{Rng, RngExt};

const GRID_SIZE: usize = 7;
const MAX_STEPS: usize = 100;

/// A simple grid-world environment.
///
/// No internal walls — movement is blocked only at grid boundaries, which
/// the agent can learn via `action_mask()`.
pub struct GridWorld<R> {
    agent: (usize, usize),
    goal: (usize, usize),
    step_count: usize,
    rng: R,
}

impl<R: Rng> GridWorld<R> {
    /// Create a new grid world with the given RNG.
    ///
    /// Goal is fixed at `(GRID_SIZE-1, GRID_SIZE-1)`.
    pub fn new(rng: R) -> Self {
        let mut env = Self {
            agent: (0, 0),
            goal: (GRID_SIZE - 1, GRID_SIZE - 1),
            step_count: 0,
            rng,
        };
        env.reset();
        env
    }

    fn obs(&self) -> Vec<f32> {
        let s = GRID_SIZE as f32;
        vec![
            self.agent.0 as f32 / s,
            self.agent.1 as f32 / s,
            self.goal.0 as f32 / s,
            self.goal.1 as f32 / s,
        ]
    }
}

impl<R: Rng> Env for GridWorld<R> {
    type Observation = Vec<f32>;
    type Action = usize;

    fn reset(&mut self) -> Vec<f32> {
        self.agent = (
            self.rng.random_range(0..GRID_SIZE),
            self.rng.random_range(0..GRID_SIZE),
        );
        // Avoid starting on the goal
        while self.agent == self.goal {
            self.agent = (
                self.rng.random_range(0..GRID_SIZE),
                self.rng.random_range(0..GRID_SIZE),
            );
        }
        self.step_count = 0;
        self.obs()
    }

    fn step(&mut self, action: usize) -> Step<Vec<f32>> {
        assert!(action < 4, "action must be 0..3, got {action}");

        let (r, c) = self.agent;
        self.agent = match action {
            0 if r > 0 => (r - 1, c),             // up
            1 if c < GRID_SIZE - 1 => (r, c + 1), // right
            2 if r < GRID_SIZE - 1 => (r + 1, c), // down
            3 if c > 0 => (r, c - 1),             // left
            _ => (r, c),                           // blocked
        };
        self.step_count += 1;

        let reached = self.agent == self.goal;
        let reward = if reached { 1.0 } else { -0.01 };

        Step {
            observation: self.obs(),
            reward,
            terminated: reached,
            truncated: self.step_count >= MAX_STEPS,
        }
    }

    fn observation_space(&self) -> Space {
        Space::Box {
            low: vec![0.0; 4],
            high: vec![1.0; 4],
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(4)
    }

    fn action_mask(&self) -> Option<Vec<f32>> {
        let (r, c) = self.agent;
        Some(vec![
            if r > 0 { 1.0 } else { 0.0 },             // up
            if c < GRID_SIZE - 1 { 1.0 } else { 0.0 },  // right
            if r < GRID_SIZE - 1 { 1.0 } else { 0.0 },  // down
            if c > 0 { 1.0 } else { 0.0 },              // left
        ])
    }
}

impl<R> Renderable for GridWorld<R> {
    fn render(&self) -> RgbFrame {
        let cell = 40u16; // pixels per cell
        let w = GRID_SIZE as u16 * cell;
        let h = GRID_SIZE as u16 * cell;
        let mut pixels = vec![255u8; w as usize * h as usize * 3]; // white bg

        // Draw grid lines
        for r in 0..=GRID_SIZE {
            let py = r as i32 * cell as i32;
            for px in 0..w as i32 {
                set_pixel(&mut pixels, w, px, py.min(h as i32 - 1), [200, 200, 200]);
            }
        }
        for c in 0..=GRID_SIZE {
            let px = c as i32 * cell as i32;
            for py in 0..h as i32 {
                set_pixel(&mut pixels, w, px.min(w as i32 - 1), py, [200, 200, 200]);
            }
        }

        // Draw goal (green square)
        fill_cell(&mut pixels, w, cell, self.goal.0, self.goal.1, [60, 180, 60]);

        // Draw agent (blue square)
        fill_cell(&mut pixels, w, cell, self.agent.0, self.agent.1, [40, 40, 200]);

        RgbFrame {
            width: w,
            height: h,
            data: pixels,
        }
    }
}

fn set_pixel(pixels: &mut [u8], canvas_w: u16, x: i32, y: i32, color: [u8; 3]) {
    let w = canvas_w as i32;
    let h = pixels.len() as i32 / 3 / w;
    if x >= 0 && x < w && y >= 0 && y < h {
        let idx = (y * w + x) as usize * 3;
        pixels[idx..idx + 3].copy_from_slice(&color);
    }
}

fn fill_cell(
    pixels: &mut [u8],
    canvas_w: u16,
    cell_size: u16,
    row: usize,
    col: usize,
    color: [u8; 3],
) {
    let x0 = col as i32 * cell_size as i32 + 2;
    let y0 = row as i32 * cell_size as i32 + 2;
    let x1 = x0 + cell_size as i32 - 4;
    let y1 = y0 + cell_size as i32 - 4;
    let w = canvas_w as i32;
    let h = pixels.len() as i32 / 3 / w;
    for y in y0.max(0)..y1.min(h) {
        for x in x0.max(0)..x1.min(w) {
            let idx = (y * w + x) as usize * 3;
            pixels[idx..idx + 3].copy_from_slice(&color);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_env() -> GridWorld<rand::rngs::SmallRng> {
        GridWorld::new(rand::rngs::SmallRng::seed_from_u64(42))
    }

    #[test]
    fn reset_returns_4d_obs() {
        let mut env = make_env();
        let obs = env.reset();
        assert_eq!(obs.len(), 4);
        for &v in &obs {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn reaches_goal() {
        let mut env = make_env();
        env.reset();
        // Force agent to one step away from goal
        env.agent = (GRID_SIZE - 1, GRID_SIZE - 2);
        let step = env.step(1); // right -> goal
        assert!(step.terminated);
        assert_eq!(step.reward, 1.0);
    }

    #[test]
    fn wall_blocks_movement() {
        let mut env = make_env();
        env.reset();
        env.agent = (0, 0);
        let _ = env.step(0); // up from top row -> no-op
        assert_eq!(env.agent, (0, 0));
        let _ = env.step(3); // left from left col -> no-op
        assert_eq!(env.agent, (0, 0));
    }

    #[test]
    fn mask_reflects_boundaries() {
        let mut env = make_env();
        env.reset();
        env.agent = (0, 0); // top-left corner
        let mask = env.action_mask().unwrap();
        assert_eq!(mask[0], 0.0); // can't go up
        assert_eq!(mask[1], 1.0); // can go right
        assert_eq!(mask[2], 1.0); // can go down
        assert_eq!(mask[3], 0.0); // can't go left
    }

    #[test]
    fn truncates_at_max_steps() {
        let mut env = make_env();
        env.reset();
        // Keep agent away from goal
        env.agent = (0, 0);
        env.goal = (GRID_SIZE - 1, GRID_SIZE - 1);
        let mut truncated = false;
        for _ in 0..MAX_STEPS + 10 {
            env.agent = (0, 0); // reset position each step to avoid reaching goal
            let step = env.step(1); // move right (then reset)
            if step.truncated {
                truncated = true;
                break;
            }
        }
        assert!(truncated);
    }

    #[test]
    fn render_produces_valid_frame() {
        let env = make_env();
        let frame = env.render();
        assert_eq!(frame.data.len(), frame.width as usize * frame.height as usize * 3);
    }

    #[test]
    fn spaces() {
        let env = make_env();
        assert_eq!(env.observation_space().flat_dim(), 4);
        assert_eq!(env.action_space().flat_dim(), 4);
    }
}
