//! Core traits and abstractions for rl4burn.
//!
//! This crate provides the foundational types that don't depend on Burn:
//! - **`env`** — `Env` trait, `Step`, `Space`, `SyncVecEnv`, wrappers, adapters, rendering
//! - **`log`** — `Logger` trait, `Loggable` trait, `PrintLogger`, `NoopLogger`, `CompositeLogger`,
//!   plus feature-gated `TensorBoardLogger`, `JsonLogger`, and `write_gif`

pub mod env;
pub mod log;
