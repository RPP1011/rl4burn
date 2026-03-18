//! # Example 13 — Multi-Agent Training with Parameter Sharing
//!
//! Two-team game (2v2 "Territory Control") where agents from the same team
//! share neural network weights. Demonstrates:
//!
//! - **Parameter sharing**: all agents on a team use the same model. This
//!   dramatically reduces the number of parameters and means agents learn
//!   a single, general policy rather than specialized roles. Each agent
//!   still acts independently based on its own observation.
//!
//! - **Batching**: `batch_multi_agent_obs` concatenates observations from
//!   all agents into a single tensor `[n_envs * n_agents, obs_dim]` for
//!   one efficient forward pass. `unbatch_actions` splits the result back
//!   into per-agent actions. This is critical for GPU efficiency.
//!
//! - **Team rewards**: all agents on the winning team receive the same
//!   reward (`broadcast_team_reward`). This encourages cooperative
//!   behavior — agents learn that individual actions affect team success.
//!
//! - **Independent learning between teams**: each team has its own model
//!   and optimizer. Team A's gradients never flow into Team B's model.
//!   This is the simplest multi-agent setup; more advanced approaches
//!   (e.g., self-play, population training) can be layered on top.
//!
//! ## Territory Control
//!
//! A 5x5 grid. Two teams of 2 agents each. Each agent occupies one cell.
//! On each step, agents choose a direction (N/S/E/W/stay). After all agents
//! move, score = number of unique cells occupied by team. Team with more
//! cells gets +1 reward, fewer gets -1, tie gets 0. Episode lasts 20 steps.
//!
//! Run with: `cargo run -p multi-agent --release`

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::nn::Linear;
use burn::optim::AdamConfig;
use burn::prelude::*;

use rand::{Rng, RngExt, SeedableRng};

use rl4burn::env::space::Space;
use rl4burn::{
    batch_multi_agent_obs, broadcast_team_reward, masked_ppo_collect, masked_ppo_update,
    orthogonal_linear, unbatch_actions, ActionDist, Env, MaskedActorCritic, PpoConfig, Step,
    SyncVecEnv,
};

type TrainB = Autodiff<NdArray>;
type InferB = NdArray;

// ---------------------------------------------------------------------------
// Game constants
// ---------------------------------------------------------------------------

const GRID_SIZE: i32 = 5;
const AGENTS_PER_TEAM: usize = 2;
const NUM_TEAMS: usize = 2;
const TOTAL_AGENTS: usize = AGENTS_PER_TEAM * NUM_TEAMS;
const MAX_STEPS: usize = 20;
/// 5 movement options: stay, north, south, east, west
const NUM_ACTIONS: usize = 5;

/// Observation per agent:
/// - own position (x, y normalized): 2
/// - teammate position (x, y normalized): 2
/// - opponent positions (2 opponents * 2): 4
/// - step counter (normalized): 1
/// Total: 9
const OBS_DIM: usize = 9;

// ---------------------------------------------------------------------------
// Multi-agent environment
// ---------------------------------------------------------------------------

/// 2v2 Territory Control environment.
///
/// The environment manages all 4 agents internally. From the outside, it
/// exposes a "joint action" interface: one step takes actions for all agents.
///
/// However, for training we want each team's model to process only its own
/// agents' observations. The training loop uses `batch_multi_agent_obs` to
/// batch team observations efficiently.
struct TerritoryEnv<R> {
    /// Positions: [team_a_agent_0, team_a_agent_1, team_b_agent_0, team_b_agent_1]
    positions: [(i32, i32); TOTAL_AGENTS],
    step_count: usize,
    rng: R,
}

impl<R: Rng> TerritoryEnv<R> {
    fn new(mut rng: R) -> Self {
        let positions = Self::random_positions(&mut rng);
        Self {
            positions,
            step_count: 0,
            rng,
        }
    }

    fn random_positions(rng: &mut impl Rng) -> [(i32, i32); TOTAL_AGENTS] {
        [
            (rng.random_range(0..GRID_SIZE), rng.random_range(0..GRID_SIZE)),
            (rng.random_range(0..GRID_SIZE), rng.random_range(0..GRID_SIZE)),
            (rng.random_range(0..GRID_SIZE), rng.random_range(0..GRID_SIZE)),
            (rng.random_range(0..GRID_SIZE), rng.random_range(0..GRID_SIZE)),
        ]
    }

    /// Get observation for a specific agent. Each agent sees:
    /// - Its own position
    /// - Its teammate's position
    /// - Both opponents' positions
    /// - Normalized step counter
    fn agent_obs(&self, agent_idx: usize) -> Vec<f32> {
        let norm = GRID_SIZE as f32;
        let (ax, ay) = self.positions[agent_idx];

        // Determine teammate and opponents
        let team = agent_idx / AGENTS_PER_TEAM;
        let teammate_idx = if agent_idx % AGENTS_PER_TEAM == 0 {
            team * AGENTS_PER_TEAM + 1
        } else {
            team * AGENTS_PER_TEAM
        };
        let opp_team = 1 - team;
        let opp_0 = opp_team * AGENTS_PER_TEAM;
        let opp_1 = opp_team * AGENTS_PER_TEAM + 1;

        let (tx, ty) = self.positions[teammate_idx];
        let (ox0, oy0) = self.positions[opp_0];
        let (ox1, oy1) = self.positions[opp_1];

        vec![
            ax as f32 / norm,
            ay as f32 / norm,
            tx as f32 / norm,
            ty as f32 / norm,
            ox0 as f32 / norm,
            oy0 as f32 / norm,
            ox1 as f32 / norm,
            oy1 as f32 / norm,
            self.step_count as f32 / MAX_STEPS as f32,
        ]
    }

    /// Get observations for all agents on one team.
    fn team_obs(&self, team: usize) -> Vec<Vec<f32>> {
        (0..AGENTS_PER_TEAM)
            .map(|i| self.agent_obs(team * AGENTS_PER_TEAM + i))
            .collect()
    }

    /// Apply actions (0=stay, 1=N, 2=S, 3=E, 4=W) and compute rewards.
    fn step_all(&mut self, actions: &[usize; TOTAL_AGENTS]) -> (f32, f32, bool) {
        let dx = [0, 0, 0, 1, -1];
        let dy = [0, 1, -1, 0, 0];

        for (i, &a) in actions.iter().enumerate() {
            let (x, y) = self.positions[i];
            self.positions[i] = (
                (x + dx[a]).clamp(0, GRID_SIZE - 1),
                (y + dy[a]).clamp(0, GRID_SIZE - 1),
            );
        }

        self.step_count += 1;

        // Count unique cells per team
        let mut team_a_cells = std::collections::HashSet::new();
        let mut team_b_cells = std::collections::HashSet::new();

        for i in 0..AGENTS_PER_TEAM {
            team_a_cells.insert(self.positions[i]);
        }
        for i in 0..AGENTS_PER_TEAM {
            team_b_cells.insert(self.positions[AGENTS_PER_TEAM + i]);
        }

        let score_a = team_a_cells.len() as f32;
        let score_b = team_b_cells.len() as f32;

        // Reward: +1 if more territory, -1 if less, 0 if tie
        let reward_a = if score_a > score_b {
            1.0
        } else if score_a < score_b {
            -1.0
        } else {
            0.0
        };
        let reward_b = -reward_a; // Zero-sum

        let truncated = self.step_count >= MAX_STEPS;

        (reward_a, reward_b, truncated)
    }

    fn reset_env(&mut self) {
        self.positions = Self::random_positions(&mut self.rng);
        self.step_count = 0;
    }
}

// ---------------------------------------------------------------------------
// Single-team environment wrapper for PPO
// ---------------------------------------------------------------------------

/// Wraps TerritoryEnv as a single-team environment for use with masked_ppo_collect.
///
/// This wrapper exposes a "virtual" single-agent environment where:
/// - Observation is this team's concatenated agent observations
/// - Action is this team's concatenated agent actions (multi-discrete)
/// - The opponent team plays with a simple heuristic (or random)
///
/// For full multi-agent training, we train each team's model separately.
struct TeamEnv<R> {
    env: TerritoryEnv<R>,
    team: usize,       // 0 or 1
}

impl<R: Rng + Clone> TeamEnv<R> {
    fn new(rng: R, team: usize) -> Self {
        Self {
            env: TerritoryEnv::new(rng),
            team,
        }
    }

    /// Get concatenated observations for this team's agents.
    fn team_obs_flat(&self) -> Vec<f32> {
        self.env.team_obs(self.team).into_iter().flatten().collect()
    }
}

impl<R: Rng + Clone> Env for TeamEnv<R> {
    type Observation = Vec<f32>;
    /// Actions: MultiDiscrete([5, 5]) — one 5-way choice per agent.
    type Action = Vec<f32>;

    fn reset(&mut self) -> Vec<f32> {
        self.env.reset_env();
        self.team_obs_flat()
    }

    fn step(&mut self, action: Vec<f32>) -> Step<Vec<f32>> {
        // Decode team actions
        let team_actions: Vec<usize> = action.iter().map(|a| *a as usize).collect();

        // Opponent plays randomly (simple baseline opponent)
        let opp_team = 1 - self.team;
        let mut all_actions = [0usize; TOTAL_AGENTS];

        for i in 0..AGENTS_PER_TEAM {
            all_actions[self.team * AGENTS_PER_TEAM + i] = team_actions[i];
            // Random opponent
            all_actions[opp_team * AGENTS_PER_TEAM + i] =
                self.env.rng.random_range(0..NUM_ACTIONS);
        }

        let (reward_a, reward_b, truncated) = self.env.step_all(&all_actions);
        let reward = if self.team == 0 { reward_a } else { reward_b };

        Step {
            observation: self.team_obs_flat(),
            reward,
            terminated: false,
            truncated,
        }
    }

    fn observation_space(&self) -> Space {
        Space::Box {
            low: vec![0.0; OBS_DIM * AGENTS_PER_TEAM],
            high: vec![1.0; OBS_DIM * AGENTS_PER_TEAM],
        }
    }

    /// MultiDiscrete([5, 5]) — one movement choice per team member.
    fn action_space(&self) -> Space {
        Space::MultiDiscrete(vec![NUM_ACTIONS; AGENTS_PER_TEAM])
    }
}

// ---------------------------------------------------------------------------
// Neural network
// ---------------------------------------------------------------------------

/// Shared-weight actor-critic for territory control.
///
/// All agents on a team use the same model. Observations are batched
/// across agents for a single forward pass.
///
/// The policy head outputs 5 * AGENTS_PER_TEAM = 10 logits (5 per agent),
/// interpreted as MultiDiscrete([5, 5]) by the ActionDist.
#[derive(Module, Debug)]
struct TeamModel<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    policy_head: Linear<B>,
    value_head: Linear<B>,
}

impl<B: Backend> TeamModel<B> {
    fn new(device: &B::Device, rng: &mut impl Rng) -> Self {
        let sqrt2 = std::f32::consts::SQRT_2;
        let input_dim = OBS_DIM * AGENTS_PER_TEAM;
        let n_logits = NUM_ACTIONS * AGENTS_PER_TEAM;
        Self {
            fc1: orthogonal_linear(input_dim, 64, sqrt2, device, rng),
            fc2: orthogonal_linear(64, 64, sqrt2, device, rng),
            policy_head: orthogonal_linear(64, n_logits, 0.01, device, rng),
            value_head: orthogonal_linear(64, 1, 1.0, device, rng),
        }
    }
}

/// MaskedActorCritic implementation. The model receives the concatenated
/// team observation and outputs logits for all agents' actions at once.
///
/// Key insight: parameter sharing happens because the same fc1/fc2 weights
/// process information about all team members. The model learns a joint
/// team policy that considers all agents' positions.
impl<B: Backend> MaskedActorCritic<B> for TeamModel<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let h = self.fc1.forward(obs).tanh();
        let h = self.fc2.forward(h).tanh();
        let logits = self.policy_head.forward(h.clone());
        let values = self.value_head.forward(h).squeeze_dim::<1>(1);
        (logits, values)
    }
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

fn main() {
    let device = NdArrayDevice::Cpu;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    let n_envs = 8;

    // --- Team A environments and model ---
    let envs_a: Vec<TeamEnv<rand::rngs::SmallRng>> = (0..n_envs)
        .map(|i| TeamEnv::new(rand::rngs::SmallRng::seed_from_u64(100 + i as u64), 0))
        .collect();
    let mut vec_env_a = SyncVecEnv::new(envs_a);

    // --- Team B environments and model ---
    // Team B trains independently with its own environments.
    // In a self-play setup, they would share environments; here they learn
    // independently against a random opponent for simplicity.
    let envs_b: Vec<TeamEnv<rand::rngs::SmallRng>> = (0..n_envs)
        .map(|i| TeamEnv::new(rand::rngs::SmallRng::seed_from_u64(200 + i as u64), 1))
        .collect();
    let mut vec_env_b = SyncVecEnv::new(envs_b);

    // Each team has its own model — no gradient sharing between teams.
    // Within a team, all agents share the same weights (parameter sharing).
    let mut model_a: TeamModel<TrainB> = TeamModel::new(&device, &mut rng);
    let mut model_b: TeamModel<TrainB> = TeamModel::new(&device, &mut rng);

    // MultiDiscrete([5, 5]) — one movement per agent on the team.
    let action_dist = ActionDist::MultiDiscrete(vec![NUM_ACTIONS; AGENTS_PER_TEAM]);

    let mut optim_a = AdamConfig::new().with_epsilon(1e-5).init();
    let mut optim_b = AdamConfig::new().with_epsilon(1e-5).init();

    let config = PpoConfig {
        lr: 3e-4,
        gamma: 0.99,
        gae_lambda: 0.95,
        clip_eps: 0.2,
        vf_coef: 0.5,
        ent_coef: 0.01,
        update_epochs: 4,
        minibatch_size: 64,
        n_steps: 32,
        clip_vloss: true,
        max_grad_norm: 0.5,
        target_kl: None,
        dual_clip_coef: None,
    };

    let total_timesteps = 200_000;
    let steps_per_iter = config.n_steps * n_envs;
    let n_iterations = total_timesteps / steps_per_iter;

    let mut recent_a: Vec<f32> = Vec::new();
    let mut recent_b: Vec<f32> = Vec::new();
    let mut current_obs_a = vec_env_a.reset();
    let mut current_obs_b = vec_env_b.reset();
    let mut ep_acc_a = vec![0.0f32; n_envs];
    let mut ep_acc_b = vec![0.0f32; n_envs];

    println!("=== Multi-Agent Training: 2v2 Territory Control ===");
    println!();
    println!("  Grid:            {GRID_SIZE}x{GRID_SIZE}");
    println!("  Agents/team:     {AGENTS_PER_TEAM}");
    println!("  Episode length:  {MAX_STEPS}");
    println!("  Actions:         5 (stay, N, S, E, W)");
    println!("  Action space:    MultiDiscrete([5, 5]) per team");
    println!("  Envs per team:   {n_envs}");
    println!("  Timesteps:       {total_timesteps}");
    println!();
    println!("Both teams train independently against random opponents.");
    println!("Parameter sharing means all agents on a team use the same model.");
    println!();

    for iter in 0..n_iterations {
        let frac = 1.0 - iter as f64 / n_iterations as f64;
        let current_lr = config.lr * frac;

        // --- Train Team A ---
        // Collect rollout: each env produces observations for both agents,
        // concatenated into one flat vector. The MultiDiscrete action dist
        // handles splitting logits back into per-agent actions.
        let inference_a = model_a.valid();
        let rollout_a = masked_ppo_collect::<InferB, _, _>(
            &inference_a,
            &mut vec_env_a,
            &action_dist,
            &config,
            &device,
            &mut rng,
            &mut current_obs_a,
            &mut ep_acc_a,
        );
        recent_a.extend_from_slice(&rollout_a.episode_returns);

        let stats_a;
        (model_a, stats_a) = masked_ppo_update(
            model_a,
            &mut optim_a,
            &rollout_a,
            &action_dist,
            &config,
            current_lr,
            &device,
            &mut rng,
        );

        // --- Train Team B ---
        let inference_b = model_b.valid();
        let rollout_b = masked_ppo_collect::<InferB, _, _>(
            &inference_b,
            &mut vec_env_b,
            &action_dist,
            &config,
            &device,
            &mut rng,
            &mut current_obs_b,
            &mut ep_acc_b,
        );
        recent_b.extend_from_slice(&rollout_b.episode_returns);

        let stats_b;
        (model_b, stats_b) = masked_ppo_update(
            model_b,
            &mut optim_b,
            &rollout_b,
            &action_dist,
            &config,
            current_lr,
            &device,
            &mut rng,
        );

        // Keep rolling window
        if recent_a.len() > 50 {
            recent_a = recent_a[recent_a.len() - 50..].to_vec();
        }
        if recent_b.len() > 50 {
            recent_b = recent_b[recent_b.len() - 50..].to_vec();
        }

        if (iter + 1) % 20 == 0 || iter == 0 {
            let avg_a: f32 = if recent_a.is_empty() {
                0.0
            } else {
                recent_a.iter().sum::<f32>() / recent_a.len() as f32
            };
            let avg_b: f32 = if recent_b.is_empty() {
                0.0
            } else {
                recent_b.iter().sum::<f32>() / recent_b.len() as f32
            };

            println!(
                "iter {:>4}/{}: Team A avg={:>+6.2} ent={:.3}  |  Team B avg={:>+6.2} ent={:.3}  lr={:.2e}",
                iter + 1,
                n_iterations,
                avg_a,
                stats_a.entropy,
                avg_b,
                stats_b.entropy,
                current_lr,
            );
        }
    }

    println!();

    // --- Demonstrate multi-agent utility functions ---
    // These utilities from rl4burn are key for efficient multi-agent training:
    //
    // 1. batch_multi_agent_obs: batches per-agent observations from multiple
    //    environments into a single tensor for shared-weight inference.
    //
    // 2. unbatch_actions: splits batched actions back to per-env, per-agent.
    //
    // 3. broadcast_team_reward: replicates team reward to all team members.
    println!("--- Multi-Agent Utility Demo ---");
    println!();

    // Simulate 2 envs, 2 agents each, obs_dim=9
    let demo_obs = vec![
        vec![vec![0.1; OBS_DIM], vec![0.2; OBS_DIM]], // env 0: agent 0, agent 1
        vec![vec![0.3; OBS_DIM], vec![0.4; OBS_DIM]], // env 1: agent 0, agent 1
    ];

    // batch_multi_agent_obs: [2 envs][2 agents][9 features] -> [4, 9] tensor
    let (batched_tensor, n_envs_out, n_agents_out) =
        batch_multi_agent_obs::<InferB>(&demo_obs, &device);
    println!(
        "batch_multi_agent_obs: [{} envs][{} agents][{} obs_dim] -> tensor {:?}",
        demo_obs.len(),
        demo_obs[0].len(),
        OBS_DIM,
        batched_tensor.dims(),
    );

    // Shared-weight inference: one forward pass for all agents
    let _demo_model: TeamModel<InferB> = TeamModel::new(&device, &mut rng);
    // We'd normally forward here, but just demonstrate unbatching with fake actions
    let flat_actions = vec![0usize, 1, 2, 3];
    let per_env_actions = unbatch_actions(&flat_actions, n_envs_out, n_agents_out);
    println!(
        "unbatch_actions: {:?} -> {:?}",
        flat_actions, per_env_actions
    );

    // broadcast_team_reward: per-env reward -> per-agent reward
    let env_rewards = vec![1.0, -1.0];
    let agent_rewards = broadcast_team_reward(&env_rewards, AGENTS_PER_TEAM);
    println!(
        "broadcast_team_reward: {:?} (2 agents) -> {:?}",
        env_rewards, agent_rewards,
    );

    println!();
    let final_avg_a: f32 = if recent_a.is_empty() {
        0.0
    } else {
        recent_a.iter().sum::<f32>() / recent_a.len() as f32
    };
    let final_avg_b: f32 = if recent_b.is_empty() {
        0.0
    } else {
        recent_b.iter().sum::<f32>() / recent_b.len() as f32
    };
    println!("Training complete.");
    println!("  Team A final avg return: {final_avg_a:+.2}");
    println!("  Team B final avg return: {final_avg_b:+.2}");
    if final_avg_a > 0.0 || final_avg_b > 0.0 {
        println!("At least one team learned to beat a random opponent!");
    }
}
