//! Cloud GPU provider abstractions for deploying rl4burn training jobs.
//!
//! This module provides a [`CloudProvider`] trait and implementations for
//! popular GPU marketplaces (Vast.ai, RunPod). It handles instance lifecycle
//! (search, launch, status, stop) so you can script cloud training runs.
//!
//! # Feature gate
//!
//! All deploy functionality is behind the `deploy` feature flag:
//!
//! ```toml
//! [dependencies]
//! rl4burn = { git = "...", features = ["deploy"] }
//! ```

mod provider;
mod vastai;
mod runpod;

pub use provider::*;
pub use vastai::VastAiProvider;
pub use runpod::RunPodProvider;
