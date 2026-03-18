//! # Example 11 — Action Masking for Game AI
//!
//! Deep dive into masked PPO for a game with complex legal action constraints.
//!
//! ## Why masking instead of penalties?
//!
//! A common approach to handle illegal actions is to apply a negative reward
//! when the agent picks one. This is **wrong** for two reasons:
//!
//! 1. **Wasted exploration**: the agent must discover through trial-and-error
//!    that certain actions are illegal, burning precious environment steps on
//!    transitions that teach it nothing about strategy.
//!
//! 2. **Reward contamination**: penalty signals compete with the true task
//!    reward, distorting the value function and making credit assignment harder.
//!
//! **Action masking** solves both problems. Before the softmax, invalid logits
//! are set to `-inf` (in practice, `logit + (mask - 1) * 1e9`). This guarantees:
//! - Zero probability for illegal actions: `softmax(-inf) = 0`.
//! - 100% of exploration budget goes to legal actions.
//! - No reward signal corruption.
//!
//! ## How masks flow through masked PPO
//!
//! 1. **Collection** (`masked_ppo_collect`): each step queries `env.action_mask()`,
//!    applies it to the model logits, samples from the masked distribution,
//!    and **stores the mask** in the rollout buffer alongside observations.
//!
//! 2. **Update** (`masked_ppo_update`): during PPO update epochs, minibatches
//!    are formed by shuffling indices. The stored masks are **reconstructed
//!    per-minibatch** so that log-probs and entropy are recomputed correctly
//!    under the masked distribution. This is critical: if you forget to
//!    reapply masks during the update, the policy ratio will be computed
//!    against the wrong (unmasked) distribution, leading to training instability.
//!
//! ## The Card Game environment
//!
//! A simple card game that demonstrates dynamic masking:
//! - The player has a hand of 5 card slots (each either empty or holding a card).
//! - Each card has a "suit" (0-3) and a "rank" (0-2).
//! - The game has a "required suit" that changes each turn.
//! - The player must play a card matching the required suit (if they have one).
//! - If no card matches, any non-empty slot is valid.
//! - Playing an empty slot is always invalid.
//!
//! This creates rich, state-dependent masking: the legal move set changes
//! every turn based on the hand contents and the required suit.
//!
//! Run with: `cargo run -p action-masking --release`

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::optim::AdamConfig;
use burn::prelude::*;

use rand::{Rng, RngExt, SeedableRng};

use rl4burn::env::space::Space;
use rl4burn::{
    masked_ppo_collect, masked_ppo_update, orthogonal_linear, ActionDist, Env, MaskedActorCritic,
    PpoConfig, Step, SyncVecEnv,
};

type AutodiffB = Autodiff<NdArray>;

// ---------------------------------------------------------------------------
// Card representation
// ---------------------------------------------------------------------------

/// Number of card slots in the hand.
const HAND_SIZE: usize = 5;
/// Number of suits in the game.
const NUM_SUITS: usize = 4;
/// Number of ranks per suit.
const NUM_RANKS: usize = 3;
/// Maximum turns before episode truncation.
const MAX_TURNS: usize = 30;

/// A card has a suit (0..NUM_SUITS) and rank (0..NUM_RANKS).
#[derive(Clone, Copy, Debug)]
struct Card {
    suit: usize,
    rank: usize,
}

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

/// A card game environment with dynamic action masking.
///
/// Each turn, the game picks a "required suit." The player must play a card
/// of that suit if possible; otherwise, any non-empty slot is valid. Empty
/// slots are always masked out.
///
/// Reward:
/// - +2.0 for playing a card of the required suit
/// - +0.5 for playing any card when no matching suit is available
/// - Cards are replenished randomly to keep the game going
///
/// Observation (per slot): [has_card, suit_one_hot(4), rank_one_hot(3)] = 8
/// Total obs dim: HAND_SIZE * 8 + NUM_SUITS (for required suit one-hot) = 44
struct CardGameEnv<R> {
    hand: [Option<Card>; HAND_SIZE],
    required_suit: usize,
    turn: usize,
    rng: R,
}

/// Observation dimension: 5 slots * 8 features + 4 for required suit = 44.
const OBS_DIM: usize = HAND_SIZE * (1 + NUM_SUITS + NUM_RANKS) + NUM_SUITS;

impl<R: Rng> CardGameEnv<R> {
    fn new(rng: R) -> Self {
        let mut env = Self {
            hand: [None; HAND_SIZE],
            required_suit: 0,
            turn: 0,
            rng,
        };
        env.deal_full_hand();
        env.required_suit = env.rng.random_range(0..NUM_SUITS);
        env
    }

    /// Fill all empty slots with random cards.
    fn deal_full_hand(&mut self) {
        for slot in self.hand.iter_mut() {
            if slot.is_none() {
                *slot = Some(Card {
                    suit: self.rng.random_range(0..NUM_SUITS),
                    rank: self.rng.random_range(0..NUM_RANKS),
                });
            }
        }
    }

    /// Build the observation vector.
    fn obs(&self) -> Vec<f32> {
        let mut o = Vec::with_capacity(OBS_DIM);
        for slot in &self.hand {
            match slot {
                Some(card) => {
                    o.push(1.0); // has_card
                    for s in 0..NUM_SUITS {
                        o.push(if s == card.suit { 1.0 } else { 0.0 });
                    }
                    for r in 0..NUM_RANKS {
                        o.push(if r == card.rank { 1.0 } else { 0.0 });
                    }
                }
                None => {
                    o.push(0.0); // no card
                    o.extend(std::iter::repeat_n(0.0, NUM_SUITS + NUM_RANKS));
                }
            }
        }
        // Required suit one-hot
        for s in 0..NUM_SUITS {
            o.push(if s == self.required_suit { 1.0 } else { 0.0 });
        }
        o
    }

    /// Compute the action mask. The action space is Discrete(HAND_SIZE): choose
    /// which slot to play.
    ///
    /// Rules:
    /// 1. Empty slots are always invalid (mask = 0).
    /// 2. If any card matches the required suit, only matching-suit cards are valid.
    /// 3. If no card matches, all non-empty slots are valid.
    fn compute_mask(&self) -> Vec<f32> {
        let mut mask = vec![0.0f32; HAND_SIZE];

        // Check if any card matches the required suit
        let has_matching = self.hand.iter().any(|slot| {
            slot.map_or(false, |c| c.suit == self.required_suit)
        });

        for (i, slot) in self.hand.iter().enumerate() {
            match slot {
                Some(card) => {
                    if has_matching {
                        // Must play a matching card
                        if card.suit == self.required_suit {
                            mask[i] = 1.0;
                        }
                    } else {
                        // No matching card — any non-empty slot is valid
                        mask[i] = 1.0;
                    }
                }
                None => {} // Always invalid (mask stays 0)
            }
        }
        mask
    }
}

impl<R: Rng + Clone> Env for CardGameEnv<R> {
    type Observation = Vec<f32>;
    type Action = Vec<f32>;

    fn reset(&mut self) -> Vec<f32> {
        self.hand = [None; HAND_SIZE];
        self.deal_full_hand();
        self.required_suit = self.rng.random_range(0..NUM_SUITS);
        self.turn = 0;
        self.obs()
    }

    fn step(&mut self, action: Vec<f32>) -> Step<Vec<f32>> {
        let slot_idx = action[0] as usize;
        self.turn += 1;

        // Determine reward based on whether we played a matching suit
        let reward = match self.hand[slot_idx] {
            Some(card) if card.suit == self.required_suit => {
                // Played a matching card — good move
                2.0
            }
            Some(_) => {
                // Played a non-matching card (only legal if no match available)
                0.5
            }
            None => {
                // Should never happen with correct masking, but handle gracefully
                -1.0
            }
        };

        // Remove the played card
        self.hand[slot_idx] = None;

        // Replenish: deal a new card into the empty slot
        self.hand[slot_idx] = Some(Card {
            suit: self.rng.random_range(0..NUM_SUITS),
            rank: self.rng.random_range(0..NUM_RANKS),
        });

        // Pick a new required suit for next turn
        self.required_suit = self.rng.random_range(0..NUM_SUITS);

        let truncated = self.turn >= MAX_TURNS;

        Step {
            observation: self.obs(),
            reward,
            terminated: false, // The game never terminates early; it's always truncated
            truncated,
        }
    }

    fn observation_space(&self) -> Space {
        Space::Box {
            low: vec![0.0; OBS_DIM],
            high: vec![1.0; OBS_DIM],
        }
    }

    /// Discrete(5) — choose one of 5 hand slots to play.
    fn action_space(&self) -> Space {
        Space::Discrete(HAND_SIZE)
    }

    /// The action mask is the core of this example. It encodes which cards
    /// are legal to play on this turn.
    ///
    /// The `masked_ppo_collect` function calls this method at every step,
    /// applies the mask to the policy logits (setting invalid logits to -inf),
    /// and stores the mask in the rollout buffer. During `masked_ppo_update`,
    /// masks are reconstructed per-minibatch to ensure log-probs are computed
    /// under the correct masked distribution. This is necessary because PPO
    /// shuffles data across minibatches — each minibatch may contain steps
    /// from different turns with different masks.
    fn action_mask(&self) -> Option<Vec<f32>> {
        Some(self.compute_mask())
    }
}

// ---------------------------------------------------------------------------
// Neural network
// ---------------------------------------------------------------------------

/// Actor-critic network for the card game.
///
/// The policy head outputs 5 logits (one per hand slot). The masking is
/// applied *externally* by the PPO pipeline — the model itself does not
/// need to know about masking. This separation of concerns makes the
/// model reusable: swap environments and masks without touching the network.
#[derive(Module, Debug)]
struct CardAgent<B: Backend> {
    fc1: burn::nn::Linear<B>,
    fc2: burn::nn::Linear<B>,
    policy_head: burn::nn::Linear<B>,
    value_head: burn::nn::Linear<B>,
}

impl<B: Backend> CardAgent<B> {
    fn new(device: &B::Device, rng: &mut impl Rng) -> Self {
        let sqrt2 = std::f32::consts::SQRT_2;
        Self {
            fc1: orthogonal_linear(OBS_DIM, 64, sqrt2, device, rng),
            fc2: orthogonal_linear(64, 64, sqrt2, device, rng),
            // Small gain keeps initial logits near-uniform, so the agent
            // explores all legal actions roughly equally at the start.
            policy_head: orthogonal_linear(64, HAND_SIZE, 0.01, device, rng),
            value_head: orthogonal_linear(64, 1, 1.0, device, rng),
        }
    }
}

/// The model outputs raw logits. Masking is handled by `masked_ppo_collect`
/// and `masked_ppo_update`, which read the mask from the environment and
/// apply `logits + (mask - 1) * 1e9` before the softmax. The key insight:
/// `softmax([-inf, 0.5, -inf, 0.3, -inf]) = [0, 0.55, 0, 0.45, 0]`.
/// Invalid actions get exactly zero probability.
impl<B: Backend> MaskedActorCritic<B> for CardAgent<B> {
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

    // --- Vectorised environment ------------------------------------------
    let n_envs = 8;
    let envs: Vec<CardGameEnv<rand::rngs::SmallRng>> = (0..n_envs)
        .map(|i| CardGameEnv::new(rand::rngs::SmallRng::seed_from_u64(42 + i as u64)))
        .collect();
    let mut vec_env = SyncVecEnv::new(envs);

    // --- Model -----------------------------------------------------------
    let model: CardAgent<AutodiffB> = CardAgent::new(&device, &mut rng);

    // --- Action distribution ---------------------------------------------
    // Discrete(5) — the PPO pipeline splits logits into one group of 5.
    // Masking is applied automatically when the env provides action_mask().
    let action_dist = ActionDist::Discrete(HAND_SIZE);

    // --- Optimiser -------------------------------------------------------
    let mut optim = AdamConfig::new().with_epsilon(1e-5).init();

    // --- PPO hyperparameters ---------------------------------------------
    let config = PpoConfig {
        lr: 3e-4,
        gamma: 0.99,
        gae_lambda: 0.95,
        clip_eps: 0.2,
        vf_coef: 0.5,
        ent_coef: 0.01,
        update_epochs: 4,
        minibatch_size: 64,
        n_steps: 64,
        clip_vloss: true,
        max_grad_norm: 0.5,
        target_kl: None,
        dual_clip_coef: None,
    };

    // --- Training loop ---------------------------------------------------
    let mut model = model;
    let total_timesteps = 150_000;
    let steps_per_iter = config.n_steps * n_envs;
    let n_iterations = total_timesteps / steps_per_iter;

    let mut recent_returns: Vec<f32> = Vec::new();
    let mut best_avg = f32::NEG_INFINITY;
    let mut current_obs = vec_env.reset();
    let mut ep_acc = vec![0.0f32; n_envs];

    println!("=== Action Masking: Card Game ===");
    println!();
    println!("  Hand size:   {HAND_SIZE} slots");
    println!("  Suits:       {NUM_SUITS}");
    println!("  Ranks:       {NUM_RANKS}");
    println!("  Obs dim:     {OBS_DIM}");
    println!("  Max turns:   {MAX_TURNS}");
    println!("  Envs:        {n_envs}");
    println!("  Timesteps:   {total_timesteps}");
    println!();
    println!("The agent must learn to play matching-suit cards when available.");
    println!("Action masking ensures it never wastes a step on an empty slot");
    println!("or picks a non-matching card when a match exists.");
    println!();

    // Theoretical optimal: if we always match the required suit, reward = 2.0 per turn.
    // Probability of having a matching suit in 5 random cards with 4 suits:
    // P(at least one match) = 1 - (3/4)^5 = ~76%. So expected reward per turn
    // is roughly 0.76 * 2.0 + 0.24 * 0.5 = 1.64. Over 30 turns: ~49.2.
    println!("Theoretical max avg return: ~49 (matching suit ~76% of turns)");
    println!();

    for iter in 0..n_iterations {
        let frac = 1.0 - iter as f64 / n_iterations as f64;
        let current_lr = config.lr * frac;

        // Collect with masking. At each step:
        // 1. The env provides action_mask() via the Env trait.
        // 2. masked_ppo_collect applies: masked_logits = logits + (mask - 1) * 1e9
        // 3. softmax(masked_logits) gives zero probability to invalid actions.
        // 4. The mask is stored in MaskedPpoRollout.masks for use during update.
        let inference_model = model.valid();
        let rollout = masked_ppo_collect::<NdArray, _, _>(
            &inference_model,
            &mut vec_env,
            &action_dist,
            &config,
            &device,
            &mut rng,
            &mut current_obs,
            &mut ep_acc,
        );

        recent_returns.extend_from_slice(&rollout.episode_returns);

        // Update. The stored masks are reconstructed per-minibatch:
        // for each index i in the shuffled minibatch, masks[i] is looked up
        // and used to recompute log_prob and entropy. Without this, the
        // probability ratio would be wrong: old_log_prob was computed under
        // the masked distribution, so new_log_prob must also be masked.
        let stats;
        (model, stats) = masked_ppo_update(
            model,
            &mut optim,
            &rollout,
            &action_dist,
            &config,
            current_lr,
            &device,
            &mut rng,
        );

        if recent_returns.len() > 50 {
            let start = recent_returns.len() - 50;
            recent_returns = recent_returns[start..].to_vec();
        }

        if !recent_returns.is_empty() {
            let avg: f32 =
                recent_returns.iter().sum::<f32>() / recent_returns.len() as f32;
            best_avg = best_avg.max(avg);

            if (iter + 1) % 20 == 0 || iter == 0 {
                println!(
                    "iter {:>4}/{}: avg_return={:>6.1}  best={:>6.1}  \
                     policy_loss={:>8.4}  entropy={:.3}  lr={:.2e}",
                    iter + 1,
                    n_iterations,
                    avg,
                    best_avg,
                    stats.policy_loss,
                    stats.entropy,
                    current_lr,
                );
            }
        }
    }

    println!();
    println!("Training complete. Best rolling average return: {best_avg:.1}");
    if best_avg > 40.0 {
        println!("Agent learned effective suit-matching strategy with masking!");
    } else {
        println!(
            "Agent still improving. Try more timesteps or tune hyperparameters."
        );
    }
}
