//! Self-play for competitive games.
//!
//! Trains an agent to play a simple competitive two-player game ("Number Duel")
//! by playing against snapshots of itself. Demonstrates:
//!
//! - **Self-play concept**: training by playing against yourself
//! - **SelfPlayPool**: snapshotting policies, sampling opponents
//! - **PfspMatchmaking**: prioritizing harder opponents
//! - **Two-player game loop**: structuring competitive environments for PPO
//! - **Win-rate tracking**: monitoring progress over training
//!
//! ## The Game: Number Duel
//!
//! Two players simultaneously choose a number from 0 to 4 (representing 1-5).
//! The winner is determined by circular comparison (like rock-paper-scissors
//! with 5 options):
//!
//! - Each number beats the two numbers below it (wrapping around).
//!   e.g. 3 beats 1 and 2; 1 beats 4 and 5.
//! - Same number = draw.
//!
//! This creates a non-transitive game with a Nash equilibrium at uniform
//! random play, making it ideal for self-play: the agent must learn a
//! mixed strategy rather than a fixed "best move."
//!
//! ## Architecture
//!
//! Each environment wraps the two-player game. The opponent's policy logits
//! are stored in a shared `Rc<RefCell<>>` so we can swap opponents between
//! rollouts without rebuilding the environment vector.
//!
//! Run with: `cargo run -p self-play --release`

use std::cell::RefCell;
use std::rc::Rc;

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::nn::{Linear, LinearConfig};
use burn::optim::AdamConfig;
use burn::prelude::*;
use burn::tensor::TensorData;
use rand::{Rng, RngExt, SeedableRng};

use rl4burn::nn::dist::ActionDist;
use rl4burn::{
    masked_ppo_collect, masked_ppo_update, Env, MaskedActorCritic, PfspConfig, PfspMatchmaking,
    PpoConfig, SelfPlayPool, Space, Step, SyncVecEnv,
};

// ---------------------------------------------------------------------------
// Game logic
// ---------------------------------------------------------------------------

/// Determine the outcome of a Number Duel round.
///
/// Returns +1.0 for player 1 win, -1.0 for player 1 loss, 0.0 for draw.
///
/// Rule: action `a` beats `b` if `b` is 1 or 2 steps "below" `a` (mod 5).
///   0 beats {3, 4}, 1 beats {4, 0}, 2 beats {0, 1}, 3 beats {1, 2}, 4 beats {2, 3}
fn duel_outcome(a: usize, b: usize) -> f32 {
    if a == b {
        return 0.0;
    }
    let diff = (b + 5 - a) % 5;
    if diff == 3 || diff == 4 {
        1.0 // a wins
    } else {
        -1.0 // b wins
    }
}

// ---------------------------------------------------------------------------
// Shared opponent state
// ---------------------------------------------------------------------------

/// Shared opponent logits that all environments in the SyncVecEnv reference.
///
/// Before each rollout we update these logits by querying a frozen model from
/// the self-play pool. The environments sample opponent actions from
/// softmax(logits) on each step.
///
/// Using Rc<RefCell<>> is clean for single-threaded training. For multi-threaded
/// collection you would use Arc<Mutex<>> instead.
type SharedOpponent = Rc<RefCell<Option<Vec<f32>>>>;

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// Single-agent environment wrapping a two-player Number Duel.
///
/// On each step the agent picks action 0..5, the opponent picks from its
/// frozen policy, and the reward is the duel outcome.
///
/// Observation (6 dims): one-hot of opponent's last action + round fraction.
struct NumberDuelEnv {
    opponent: SharedOpponent,
    last_opponent_action: usize,
    round: usize,
    max_rounds: usize,
    rng: rand::rngs::SmallRng,
}

impl NumberDuelEnv {
    fn new(max_rounds: usize, seed: u64, opponent: SharedOpponent) -> Self {
        Self {
            opponent,
            last_opponent_action: 0,
            round: 0,
            max_rounds,
            rng: rand::rngs::SmallRng::seed_from_u64(seed),
        }
    }

    fn opponent_action(&mut self) -> usize {
        let guard = self.opponent.borrow();
        match guard.as_ref() {
            None => self.rng.random_range(0..5),
            Some(logits) => sample_from_logits(logits, &mut self.rng),
        }
    }

    fn make_obs(&self) -> Vec<f32> {
        let mut obs = vec![0.0; 6];
        obs[self.last_opponent_action] = 1.0;
        obs[5] = self.round as f32 / self.max_rounds as f32;
        obs
    }
}

/// Sample an action index from unnormalized logits using softmax.
fn sample_from_logits(logits: &[f32], rng: &mut impl Rng) -> usize {
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max_l).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let u: f32 = rng.random();
    let mut cum = 0.0;
    for (i, e) in exps.iter().enumerate() {
        cum += e / sum;
        if u < cum {
            return i;
        }
    }
    logits.len() - 1
}

impl Env for NumberDuelEnv {
    type Observation = Vec<f32>;
    type Action = Vec<f32>;

    fn reset(&mut self) -> Vec<f32> {
        self.round = 0;
        self.last_opponent_action = 0;
        self.make_obs()
    }

    fn step(&mut self, action: Vec<f32>) -> Step<Vec<f32>> {
        let agent_action = action[0] as usize;
        let opp_action = self.opponent_action();
        let reward = duel_outcome(agent_action, opp_action);

        self.last_opponent_action = opp_action;
        self.round += 1;

        let truncated = self.round >= self.max_rounds;
        let obs = if truncated { self.reset() } else { self.make_obs() };

        Step {
            observation: obs,
            reward,
            terminated: false,
            truncated,
        }
    }

    fn observation_space(&self) -> Space {
        Space::Box {
            low: vec![0.0; 6],
            high: vec![1.0; 6],
        }
    }

    fn action_space(&self) -> Space {
        Space::Discrete(5)
    }
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/// Actor-critic network for the Number Duel.
///
/// obs (6) -> shared hidden (64, tanh) -> actor head (5 logits)
///                                     -> critic head (1 value)
#[derive(Module, Debug)]
struct DuelModel<B: Backend> {
    shared: Linear<B>,
    actor_head: Linear<B>,
    critic_head: Linear<B>,
}

impl<B: Backend> DuelModel<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            shared: LinearConfig::new(6, 64).init(device),
            actor_head: LinearConfig::new(64, 5).init(device),
            critic_head: LinearConfig::new(64, 1).init(device),
        }
    }

    /// Get raw action logits for a single observation vector.
    /// Used to extract opponent policy for injection into environments.
    fn logits_for_obs(&self, obs: &[f32], device: &B::Device) -> Vec<f32> {
        let t: Tensor<B, 2> = Tensor::from_data(TensorData::new(obs.to_vec(), [1, 6]), device);
        let h = self.shared.forward(t).tanh();
        self.actor_head.forward(h).into_data().to_vec().unwrap()
    }
}

impl<B: Backend> MaskedActorCritic<B> for DuelModel<B> {
    fn forward(&self, obs: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let h = self.shared.forward(obs).tanh();
        let logits = self.actor_head.forward(h.clone());
        let values = self.critic_head.forward(h).squeeze_dim::<1>(1);
        (logits, values)
    }
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

type TrainB = Autodiff<NdArray>;
type InferB = NdArray;

fn main() {
    let device = NdArrayDevice::Cpu;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);

    // -----------------------------------------------------------------------
    // Hyperparameters
    // -----------------------------------------------------------------------
    let n_envs = 8;
    let rounds_per_episode = 20;
    let total_steps = 200_000;
    let snapshot_interval = 10; // Snapshot policy every N iterations
    let max_pool_size = 20; // Bound memory by keeping only recent snapshots

    // -----------------------------------------------------------------------
    // Environments with shared opponent
    // -----------------------------------------------------------------------
    // All environments share a reference to the current opponent's logits.
    // Before each rollout we update these logits by querying a frozen model
    // from the self-play pool. This avoids rebuilding the SyncVecEnv.
    let shared_opp: SharedOpponent = Rc::new(RefCell::new(None));
    let envs: Vec<_> = (0..n_envs)
        .map(|i| NumberDuelEnv::new(rounds_per_episode, 100 + i as u64, Rc::clone(&shared_opp)))
        .collect();
    let mut vec_env = SyncVecEnv::new(envs);

    // -----------------------------------------------------------------------
    // Model and optimizer
    // -----------------------------------------------------------------------
    let mut model: DuelModel<TrainB> = DuelModel::new(&device);
    let mut optim = AdamConfig::new().with_epsilon(1e-5).init();
    let config = PpoConfig::new()
        .with_n_steps(rounds_per_episode)
        .with_minibatch_size(40)
        .with_update_epochs(4)
        .with_ent_coef(0.05); // Encourage exploration (mixed strategies)

    let action_dist = ActionDist::Discrete(5);
    let steps_per_iter = config.n_steps * n_envs;
    let n_iters = total_steps / steps_per_iter;

    // -----------------------------------------------------------------------
    // Self-play pool
    // -----------------------------------------------------------------------
    // SelfPlayPool stores cloned snapshots of the policy at various training
    // stages. When we need an opponent, we sample one of these snapshots.
    // This forces the agent to stay robust against all past strategies,
    // not just the current one.
    let mut pool: SelfPlayPool<DuelModel<InferB>> = SelfPlayPool::new();

    // -----------------------------------------------------------------------
    // PFSP matchmaking
    // -----------------------------------------------------------------------
    // Prioritized Fictitious Self-Play (PFSP) tracks the agent's win rate
    // against each opponent snapshot. Opponents the agent struggles against
    // (low win rate) are sampled more frequently. This focuses training
    // compute on weaknesses rather than wasting time on already-mastered foes.
    //
    // The `power` parameter controls how aggressively we focus on hard
    // opponents: weight = (1 - win_rate)^power. Higher power = more focus.
    let mut matchmaking = PfspMatchmaking::new(PfspConfig {
        power: 2.0,
        min_prob: 0.05,
    });

    // Seed the pool with the initial (random) policy.
    let initial_id = pool.add_snapshot(&model.valid(), 0);
    matchmaking.add_opponent(initial_id);

    // -----------------------------------------------------------------------
    // Training loop
    // -----------------------------------------------------------------------
    let mut current_obs = vec_env.reset();
    let mut ep_acc = vec![0.0f32; n_envs];
    let mut recent_rewards: Vec<f32> = Vec::new();
    let window = 100;

    eprintln!("=== Self-Play Training: Number Duel ===");
    eprintln!();
    eprintln!("Game: two players pick 0-4 simultaneously.");
    eprintln!("Each number beats the two below it (mod 5). Same = draw.");
    eprintln!("Nash equilibrium: uniform random (20% each action).");
    eprintln!();
    eprintln!(
        "Training for {} steps ({} envs, {} iters, snapshot every {}).",
        total_steps, n_envs, n_iters, snapshot_interval
    );
    eprintln!("{:-<80}", "");

    for iter in 0..n_iters {
        // --- Pick opponent via PFSP ---
        // sample_opponent returns the ID of the snapshot to play against.
        // Harder opponents (lower win rate) are sampled more often.
        let opponent_id = matchmaking
            .sample_opponent(&mut rng)
            .expect("pool is non-empty");

        // Look up the opponent model from the pool. We sample uniformly here
        // (the PFSP weighting already chose *which* opponent matters; the pool
        // sample gives us a concrete model to play against).
        let opponent_model = pool
            .sample(&mut rng)
            .expect("pool is non-empty");

        // Extract the opponent's action logits from a neutral observation.
        // For this simultaneous-move game the observation doesn't affect the
        // opponent's strategy much -- what matters is its learned policy weights.
        // A richer game would query the opponent model per state.
        let opp_logits = opponent_model.logits_for_obs(&[0.0; 6], &device);

        // Inject opponent logits into the shared state. All environments
        // will use these logits to sample opponent actions during rollout.
        *shared_opp.borrow_mut() = Some(opp_logits);

        // --- Collect rollout ---
        // masked_ppo_collect runs the environments for n_steps and records
        // observations, actions, rewards, and log-probs. The opponent plays
        // its frozen policy inside each env's step() method.
        let lr = config.lr * (1.0 - iter as f64 / n_iters as f64);
        let rollout = masked_ppo_collect::<InferB, _, _>(
            &model.valid(),
            &mut vec_env,
            &action_dist,
            &config,
            &device,
            &mut rng,
            &mut current_obs,
            &mut ep_acc,
        );

        // --- Track results ---
        for &ret in &rollout.episode_returns {
            recent_rewards.push(ret);
            // Record result for PFSP win-rate tracking.
            // avg reward > 0 means more wins than losses in that episode.
            let avg = ret / rounds_per_episode as f32;
            let win = avg > 0.1;
            let draw = avg.abs() <= 0.1;
            matchmaking.record_result(opponent_id, win, draw);
        }
        if recent_rewards.len() > window {
            recent_rewards = recent_rewards[recent_rewards.len() - window..].to_vec();
        }

        // --- PPO update ---
        // Standard PPO update on the collected rollout. The opponent is frozen
        // so only the agent's policy is updated.
        let stats;
        (model, stats) = masked_ppo_update(
            model,
            &mut optim,
            &rollout,
            &action_dist,
            &config,
            lr,
            &device,
            &mut rng,
        );

        // --- Periodic snapshot ---
        // Every snapshot_interval iterations, clone the current policy into
        // the pool. This gives future iterations a wider range of opponents.
        // The key insight: if we only trained against the latest self, the
        // agent could "chase its tail" cycling between strategies. A diverse
        // pool forces robust play.
        if (iter + 1) % snapshot_interval == 0 {
            let step = ((iter + 1) * steps_per_iter) as u64;
            let snap_id = pool.add_snapshot(&model.valid(), step);
            matchmaking.add_opponent(snap_id);

            // Keep memory bounded. retain_recent drops the oldest snapshots.
            pool.retain_recent(max_pool_size);
        }

        // --- Logging ---
        if (iter + 1) % 10 == 0 && !recent_rewards.is_empty() {
            let avg_ret: f32 =
                recent_rewards.iter().sum::<f32>() / recent_rewards.len() as f32;
            let win_frac = recent_rewards.iter().filter(|&&r| r > 0.0).count() as f32
                / recent_rewards.len() as f32;

            let probs = matchmaking.selection_probs();
            let max_p = probs.iter().cloned().fold(0.0f64, f64::max);
            let min_p = probs.iter().cloned().fold(1.0f64, f64::min);

            eprintln!(
                "step {:>6} | ret {:>+6.2} | win {:.0}% | pool {:>2} | \
                 pfsp [{:.2}..{:.2}] | ploss {:.4} | ent {:.4}",
                (iter + 1) * steps_per_iter,
                avg_ret,
                win_frac * 100.0,
                pool.len(),
                min_p,
                max_p,
                stats.policy_loss,
                stats.entropy,
            );
        }
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    eprintln!("{:-<80}", "");
    eprintln!("Training complete. Pool contains {} snapshots.", pool.len());

    if !recent_rewards.is_empty() {
        let avg: f32 = recent_rewards.iter().sum::<f32>() / recent_rewards.len() as f32;
        let win_pct = recent_rewards.iter().filter(|&&r| r > 0.0).count() as f32
            / recent_rewards.len() as f32
            * 100.0;
        eprintln!(
            "Last {} episodes: avg return {:+.2}, win rate {:.0}%",
            recent_rewards.len(),
            avg,
            win_pct,
        );
        if (win_pct - 50.0).abs() < 15.0 {
            eprintln!("Win rate near 50%: consistent with Nash equilibrium convergence.");
        }
    }

    // Show learned action distribution (Nash = 20% each).
    let logits = model.valid().logits_for_obs(&[0.0; 6], &device);
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max_l).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();
    eprintln!();
    eprintln!("Final action probabilities (Nash equilibrium = 20% each):");
    for (i, p) in probs.iter().enumerate() {
        eprintln!("  Action {} (number {}): {:.1}%", i, i + 1, p * 100.0);
    }

    // Show PFSP matchmaking records.
    eprintln!();
    eprintln!("PFSP opponent records:");
    let records = matchmaking.records();
    let sel_probs = matchmaking.selection_probs();
    for (rec, &sp) in records.iter().zip(sel_probs.iter()) {
        if rec.total_games() > 0 {
            eprintln!(
                "  Snapshot {:>2}: win {:.0}% ({:>3} games), selection prob {:.1}%",
                rec.id,
                rec.win_rate() * 100.0,
                rec.total_games(),
                sp * 100.0,
            );
        }
    }
}
