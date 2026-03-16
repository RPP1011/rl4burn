//! Reinforcement learning algorithms for the Burn ML framework.
//!
//! `rl4burn` provides generic, backend-agnostic RL building blocks that
//! exploit Burn's type system: write `PPO<B: AutodiffBackend>` once and
//! run on WGPU, CUDA, NdArray, or LibTorch.
//!
//! # Modules
//!
//! - **Environment** (`env`, `vec_env`, `wrapper`, `envs`): Gymnasium-style
//!   environment trait with vectorized environments and composable wrappers.
//! - **Spaces** (`space`): Action and observation space descriptions.
//! - **Algorithms** (`ppo`): On-policy and off-policy RL algorithms.
//! - **Building blocks** (`loss`, `vtrace`, `gae`, `advantage`, `replay`,
//!   `polyak`, `policy`): Composable components for custom algorithms.

// Core abstractions
pub mod space;
pub mod env;
pub mod vec_env;
pub mod wrapper;
pub mod policy;

// Building blocks
pub mod clip;
pub mod init;
pub mod vtrace;
pub mod replay;
pub mod loss;
pub mod advantage;
pub mod gae;
pub mod polyak;

// Algorithms
pub mod ppo;
pub mod dqn;

// Built-in environments
pub mod envs;

// Re-exports for convenience
pub use space::Space;
pub use env::{Env, Step};
pub use vec_env::SyncVecEnv;
pub use replay::ReplayBuffer;
pub use vtrace::vtrace_targets;
pub use loss::{policy_loss_continuous, policy_loss_discrete, value_loss};
pub use advantage::normalize;
pub use gae::gae;
pub use polyak::polyak_update;
pub use init::orthogonal_linear;
pub use policy::{DiscreteActorCritic, DiscreteAcOutput, Policy, greedy_action};
pub use ppo::{PpoConfig, PpoRollout, PpoStats, ppo_collect, ppo_update};
pub use dqn::{QNetwork, DqnConfig, DqnStats, Transition, dqn_update, epsilon_greedy, epsilon_schedule};
