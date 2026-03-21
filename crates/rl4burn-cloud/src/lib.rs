//! Cloud GPU provider abstractions for deploying rl4burn training jobs.
//!
//! This crate provides a [`CloudProvider`] trait and implementations for
//! popular GPU marketplaces (Vast.ai, RunPod). It handles instance lifecycle
//! (search, launch, status, stop) so you can script cloud training runs.
//!
//! # Usage
//!
//! ```toml
//! [dependencies]
//! rl4burn-cloud = { git = "..." }
//! ```

mod provider;
mod vastai;
mod runpod;

pub use provider::*;
pub use vastai::VastAiProvider;
pub use runpod::RunPodProvider;
