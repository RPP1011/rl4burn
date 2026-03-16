//! Reinforcement learning algorithms for the Burn ML framework.
//!
//! Provides generic, backend-agnostic RL building blocks:
//! - V-trace off-policy correction
//! - Replay buffer with trajectory-aware eviction and rescoring
//! - Policy gradient loss (unified across action heads)
//! - Huber value loss
//! - Advantage normalization
//! - Training step orchestration

pub mod vtrace;
pub mod replay;
pub mod loss;
pub mod advantage;

pub use vtrace::vtrace_targets;
pub use replay::ReplayBuffer;
pub use loss::{policy_loss_continuous, policy_loss_discrete, value_loss};
pub use advantage::normalize;
