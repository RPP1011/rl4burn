//! Reinforcement learning algorithms for the Burn ML framework.
//!
//! `rl4burn` provides generic, backend-agnostic RL building blocks that
//! exploit Burn's type system: write `PPO<B: AutodiffBackend>` once and
//! run on WGPU, CUDA, NdArray, or LibTorch.
//!
//! # Modules
//!
//! - [`env`] — Environment trait, spaces, vectorized environments, wrappers
//! - [`envs`] — Built-in environments (CartPole)
//! - [`algo`] — Algorithms (PPO, DQN)
//! - [`nn`] — Neural network utilities (init, gradient clipping, polyak, losses, policy traits)
//! - [`collect`] — Data collection (GAE, V-trace, replay buffer, advantage normalization)

/// Environment abstractions: trait, spaces, vectorized envs, wrappers.
pub mod env;

/// Built-in environments (CartPole).
pub mod envs;

/// RL algorithms (PPO, DQN).
pub mod algo;

/// Neural network utilities for RL.
pub mod nn;

/// Data collection and advantage estimation.
pub mod collect;

/// Logging infrastructure for training metrics.
pub mod log;

// ---------------------------------------------------------------------------
// Convenience re-exports
// ---------------------------------------------------------------------------

// Environment
pub use env::adapter::DiscreteEnvAdapter;
pub use env::space::Space;
pub use env::vec_env::SyncVecEnv;
pub use env::wrapper;
pub use env::{Env, Step};

// Algorithms
pub use algo::dqn::{dqn_update, epsilon_greedy, epsilon_schedule, DqnConfig, DqnStats, QNetwork, Transition};
pub use algo::ppo::{ppo_collect, ppo_update, PpoConfig, PpoRollout, PpoStats};
pub use algo::ppo_masked::{
    masked_ppo_collect, masked_ppo_update, MaskedActorCritic, MaskedPpoRollout,
};

// Action distributions
pub use nn::dist::{ActionDist, LogStdMode};

// Neural network utilities
pub use nn::clip::clip_grad_norm;
pub use nn::init::orthogonal_linear;
pub use nn::loss::{policy_loss_continuous, policy_loss_discrete, value_loss};
pub use nn::policy::{greedy_action, DiscreteAcOutput, DiscreteActorCritic, Policy};
pub use nn::polyak::polyak_update;

// Data collection
pub use collect::advantage::normalize;
pub use collect::gae::gae;
pub use collect::replay::ReplayBuffer;
pub use collect::vtrace::vtrace_targets;

// Logging
pub use log::{CompositeLogger, Loggable, Logger, NoopLogger, PrintLogger};
#[cfg(feature = "tensorboard")]
pub use log::TensorBoardLogger;
#[cfg(feature = "json-log")]
pub use log::JsonLogger;
#[cfg(feature = "video")]
pub use log::write_gif;
