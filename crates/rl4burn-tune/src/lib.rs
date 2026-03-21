//! Hyperparameter search algorithms inspired by Optuna.
//!
//! This crate implements core search functionality: distributions, samplers,
//! pruners, and the study/trial orchestration loop.

pub mod distributions;
pub mod importance;
pub mod multi_objective;
pub mod pruners;
pub mod samplers;
pub mod storage;
pub mod study;
pub mod trial;

#[cfg(kani)]
mod proofs;

pub use distributions::*;
pub use pruners::{
    HyperbandPruner, MedianPruner, NopPruner, PatientPruner, PercentilePruner, Pruner,
    SuccessiveHalvingPruner, ThresholdPruner, WilcoxonPruner,
};
pub use importance::{FanovaImportanceEvaluator, ImportanceEvaluator, PedAnovaImportanceEvaluator};
pub use samplers::{
    calculate_order, default_gamma, default_weights, gaussian_log_pdf, hyperopt_default_gamma,
    BruteForceSampler, CategoricalParzenEstimator, CmaEsConfig, CmaEsSampler, CrossoverType,
    GammaStrategy, GpConfig, GpSampler, GridSampler, NsgaIIConfig, NsgaIISampler,
    ParzenEstimator, QmcConfig, QmcSampler, RandomSampler, Sampler, TpeSampler,
    TpeSamplerConfig,
};
pub use study::{Callback, Direction, Study, StudyConfig};
pub use trial::{FrozenTrial, Trial, TrialState};
