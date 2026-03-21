//! Parameter importance evaluation.
//!
//! Provides evaluators that estimate the importance of each hyperparameter
//! in determining the objective value.

mod ped_anova;

pub use ped_anova::PedAnovaImportanceEvaluator;

use std::collections::HashMap;

use crate::study::Study;

/// Trait for parameter importance evaluators.
pub trait ImportanceEvaluator: Send + Sync {
    /// Evaluate the importance of each parameter.
    ///
    /// Returns a map from parameter name to importance score (0.0 to 1.0,
    /// where higher means more important). Scores are normalized to sum to 1.0.
    fn evaluate(&self, study: &Study) -> HashMap<String, f64>;
}
