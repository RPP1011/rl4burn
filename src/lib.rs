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
//! - [`collect`] — Data collection (GAE, V-trace, UPGO, replay buffer, advantage normalization)

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
pub use nn::film::{Film, FilmConfig};
pub use nn::init::orthogonal_linear;
pub use nn::kl_balance::{
    categorical_kl, categorical_kl_groups, kl_balanced_loss, kl_balanced_loss_groups,
    KlBalanceConfig,
};
pub use nn::loss::{policy_loss_continuous, policy_loss_discrete, value_loss};
pub use nn::rnn::{
    BlockGruCell, BlockGruCellConfig, GruCell, GruCellConfig, LstmCell, LstmCellConfig, LstmState,
};
pub use nn::policy::{greedy_action, DiscreteAcOutput, DiscreteActorCritic};
pub use nn::symlog::{symexp, symlog, TwohotEncoder};

// Rendering
pub use env::render::{Renderable, RgbFrame};
pub use nn::multi_head_value::{multi_head_gae, multi_head_value_loss, MultiHeadGaeResult, MultiHeadValueConfig};
pub use nn::polyak::polyak_update;

// Data collection
pub use collect::advantage::normalize;
pub use collect::gae::gae;
pub use collect::percentile_normalize::PercentileNormalizer;
pub use collect::replay::ReplayBuffer;
pub use collect::upgo::upgo as upgo_advantages;
pub use collect::vtrace::vtrace_targets;

// Logging
pub use log::{CompositeLogger, Loggable, Logger, NoopLogger, PrintLogger};
#[cfg(feature = "tensorboard")]
pub use log::TensorBoardLogger;
#[cfg(feature = "json-log")]
pub use log::JsonLogger;
#[cfg(feature = "video")]
pub use log::write_gif;
