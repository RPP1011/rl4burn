//! Data collection, advantage estimation, and replay buffers for rl4burn.
//!
//! This crate provides:
//! - **Advantage estimation**: GAE, V-trace, UPGO, and normalization utilities.
//! - **Replay buffers**: uniform and sequence-based replay.
//! - **Trajectory infrastructure**: trajectory types and bounded queues.
//! - **Collection patterns**: IMPALA-style actor-learner and SeedRL centralized inference.
//! - **Intrinsic rewards**: count-based exploration and entropy reduction.
//! - **Curriculum**: CSPL pipeline state machine.

pub mod actor_learner;
pub mod advantage;
pub mod centralized_inference;
pub mod cspl;
pub mod gae;
pub mod intrinsic;
pub mod percentile_normalize;
pub mod replay;
pub mod sequence_replay;
pub mod trajectory;
pub mod upgo;
pub mod vtrace;
