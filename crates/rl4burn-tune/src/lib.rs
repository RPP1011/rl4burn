//! Hyperparameter search algorithms inspired by Optuna.
//!
//! This crate implements core search functionality: distributions, samplers,
//! pruners, and the study/trial orchestration loop.

pub mod distributions;
pub mod pruners;
pub mod samplers;
pub mod study;
pub mod trial;

#[cfg(kani)]
mod proofs;

pub use distributions::*;
pub use pruners::{HyperbandPruner, MedianPruner, PercentilePruner, Pruner};
pub use samplers::{CmaEsConfig, CmaEsSampler, RandomSampler, Sampler, TpeSampler, TpeSamplerConfig};
pub use study::{Direction, Study, StudyConfig};
pub use trial::{FrozenTrial, Trial, TrialState};
