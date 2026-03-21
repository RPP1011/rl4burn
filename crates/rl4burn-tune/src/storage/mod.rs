//! Storage backends for persisting study data.
//!
//! The [`Storage`] trait defines the interface for storing and retrieving
//! trial data. Two implementations are provided:
//!
//! - [`InMemoryStorage`]: Default in-memory storage (no persistence).
//! - [`JournalStorage`]: Append-only JSON lines file for single-machine persistence.

mod in_memory;
mod journal;

pub use in_memory::InMemoryStorage;
pub use journal::JournalStorage;

use crate::study::Direction;
use crate::trial::{FrozenTrial, TrialState};

/// Errors that can occur during storage operations.
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("study not found: {0}")]
    StudyNotFound(usize),
    #[error("trial not found: study={study_id}, trial={trial_number}")]
    TrialNotFound {
        study_id: usize,
        trial_number: usize,
    },
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

/// Trait for storage backends.
pub trait Storage: Send + Sync {
    /// Create a new study and return its ID.
    fn create_study(&mut self, direction: Direction) -> Result<usize, StorageError>;

    /// Get the direction of a study.
    fn get_study_direction(&self, study_id: usize) -> Result<Direction, StorageError>;

    /// Get all trials for a study.
    fn get_all_trials(&self, study_id: usize) -> Result<Vec<FrozenTrial>, StorageError>;

    /// Add a trial to a study.
    fn add_trial(&mut self, study_id: usize, trial: FrozenTrial) -> Result<(), StorageError>;

    /// Update the state of a trial.
    fn set_trial_state(
        &mut self,
        study_id: usize,
        trial_number: usize,
        state: TrialState,
    ) -> Result<(), StorageError>;

    /// Set the objective value of a trial.
    fn set_trial_value(
        &mut self,
        study_id: usize,
        trial_number: usize,
        value: f64,
    ) -> Result<(), StorageError>;

    /// Set a parameter on a trial.
    fn set_trial_param(
        &mut self,
        study_id: usize,
        trial_number: usize,
        name: &str,
        value: f64,
    ) -> Result<(), StorageError>;

    /// Set a user attribute on a trial.
    fn set_trial_user_attr(
        &mut self,
        study_id: usize,
        trial_number: usize,
        key: &str,
        value: &str,
    ) -> Result<(), StorageError>;

    /// Set a system attribute on a trial.
    fn set_trial_system_attr(
        &mut self,
        study_id: usize,
        trial_number: usize,
        key: &str,
        value: &str,
    ) -> Result<(), StorageError>;
}
