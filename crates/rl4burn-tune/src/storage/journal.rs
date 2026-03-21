//! Journal file storage backend.
//!
//! Persists study data as an append-only JSON lines file. Each line represents
//! a storage operation that can be replayed to reconstruct the full state.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::study::Direction;
use crate::trial::{FrozenTrial, TrialState};

use super::{InMemoryStorage, Storage, StorageError};

/// A storage operation that gets serialized to the journal file.
#[derive(Debug, Serialize, Deserialize)]
enum JournalOp {
    CreateStudy {
        study_id: usize,
        direction: Direction,
    },
    AddTrial {
        study_id: usize,
        trial: FrozenTrial,
    },
    SetTrialState {
        study_id: usize,
        trial_number: usize,
        state: TrialState,
    },
    SetTrialValue {
        study_id: usize,
        trial_number: usize,
        value: f64,
    },
    SetTrialParam {
        study_id: usize,
        trial_number: usize,
        name: String,
        value: f64,
    },
    SetTrialUserAttr {
        study_id: usize,
        trial_number: usize,
        key: String,
        value: String,
    },
    SetTrialSystemAttr {
        study_id: usize,
        trial_number: usize,
        key: String,
        value: String,
    },
}

/// Append-only journal file storage.
///
/// Each mutation is written as a JSON line to the journal file. On creation,
/// the file is replayed to reconstruct the in-memory state.
///
/// # Example
///
/// ```no_run
/// use rl4burn_tune::storage::{JournalStorage, Storage};
/// use rl4burn_tune::study::Direction;
///
/// let mut storage = JournalStorage::open("study.jsonl").unwrap();
/// let study_id = storage.create_study(Direction::Minimize).unwrap();
/// ```
pub struct JournalStorage {
    /// In-memory mirror of the persisted state.
    inner: InMemoryStorage,
    /// Path to the journal file.
    path: PathBuf,
    /// Open file handle for appending.
    file: File,
}

impl JournalStorage {
    /// Open or create a journal file at the given path.
    ///
    /// If the file exists, replays all operations to reconstruct state.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let path = path.as_ref().to_path_buf();
        let mut inner = InMemoryStorage::new();

        // Replay existing journal if file exists
        if path.exists() {
            let file = File::open(&path)?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let op: JournalOp = serde_json::from_str(&line)?;
                Self::apply_op(&mut inner, &op)?;
            }
        }

        // Open for appending
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;

        Ok(Self { inner, path, file })
    }

    /// Get the path to the journal file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn write_op(&mut self, op: &JournalOp) -> Result<(), StorageError> {
        let line = serde_json::to_string(op)?;
        writeln!(self.file, "{}", line)?;
        self.file.flush()?;
        Ok(())
    }

    fn apply_op(storage: &mut InMemoryStorage, op: &JournalOp) -> Result<(), StorageError> {
        match op {
            JournalOp::CreateStudy {
                direction, ..
            } => {
                storage.create_study(*direction)?;
            }
            JournalOp::AddTrial { study_id, trial } => {
                storage.add_trial(*study_id, trial.clone())?;
            }
            JournalOp::SetTrialState {
                study_id,
                trial_number,
                state,
            } => {
                storage.set_trial_state(*study_id, *trial_number, *state)?;
            }
            JournalOp::SetTrialValue {
                study_id,
                trial_number,
                value,
            } => {
                storage.set_trial_value(*study_id, *trial_number, *value)?;
            }
            JournalOp::SetTrialParam {
                study_id,
                trial_number,
                name,
                value,
            } => {
                storage.set_trial_param(*study_id, *trial_number, name, *value)?;
            }
            JournalOp::SetTrialUserAttr {
                study_id,
                trial_number,
                key,
                value,
            } => {
                storage.set_trial_user_attr(*study_id, *trial_number, key, value)?;
            }
            JournalOp::SetTrialSystemAttr {
                study_id,
                trial_number,
                key,
                value,
            } => {
                storage.set_trial_system_attr(*study_id, *trial_number, key, value)?;
            }
        }
        Ok(())
    }
}

impl Storage for JournalStorage {
    fn create_study(&mut self, direction: Direction) -> Result<usize, StorageError> {
        let study_id = self.inner.create_study(direction)?;
        let op = JournalOp::CreateStudy {
            study_id,
            direction,
        };
        self.write_op(&op)?;
        Ok(study_id)
    }

    fn get_study_direction(&self, study_id: usize) -> Result<Direction, StorageError> {
        self.inner.get_study_direction(study_id)
    }

    fn get_all_trials(&self, study_id: usize) -> Result<Vec<FrozenTrial>, StorageError> {
        self.inner.get_all_trials(study_id)
    }

    fn add_trial(&mut self, study_id: usize, trial: FrozenTrial) -> Result<(), StorageError> {
        let op = JournalOp::AddTrial {
            study_id,
            trial: trial.clone(),
        };
        self.inner.add_trial(study_id, trial)?;
        self.write_op(&op)?;
        Ok(())
    }

    fn set_trial_state(
        &mut self,
        study_id: usize,
        trial_number: usize,
        state: TrialState,
    ) -> Result<(), StorageError> {
        self.inner.set_trial_state(study_id, trial_number, state)?;
        let op = JournalOp::SetTrialState {
            study_id,
            trial_number,
            state,
        };
        self.write_op(&op)?;
        Ok(())
    }

    fn set_trial_value(
        &mut self,
        study_id: usize,
        trial_number: usize,
        value: f64,
    ) -> Result<(), StorageError> {
        self.inner.set_trial_value(study_id, trial_number, value)?;
        let op = JournalOp::SetTrialValue {
            study_id,
            trial_number,
            value,
        };
        self.write_op(&op)?;
        Ok(())
    }

    fn set_trial_param(
        &mut self,
        study_id: usize,
        trial_number: usize,
        name: &str,
        value: f64,
    ) -> Result<(), StorageError> {
        self.inner
            .set_trial_param(study_id, trial_number, name, value)?;
        let op = JournalOp::SetTrialParam {
            study_id,
            trial_number,
            name: name.to_string(),
            value,
        };
        self.write_op(&op)?;
        Ok(())
    }

    fn set_trial_user_attr(
        &mut self,
        study_id: usize,
        trial_number: usize,
        key: &str,
        value: &str,
    ) -> Result<(), StorageError> {
        self.inner
            .set_trial_user_attr(study_id, trial_number, key, value)?;
        let op = JournalOp::SetTrialUserAttr {
            study_id,
            trial_number,
            key: key.to_string(),
            value: value.to_string(),
        };
        self.write_op(&op)?;
        Ok(())
    }

    fn set_trial_system_attr(
        &mut self,
        study_id: usize,
        trial_number: usize,
        key: &str,
        value: &str,
    ) -> Result<(), StorageError> {
        self.inner
            .set_trial_system_attr(study_id, trial_number, key, value)?;
        let op = JournalOp::SetTrialSystemAttr {
            study_id,
            trial_number,
            key: key.to_string(),
            value: value.to_string(),
        };
        self.write_op(&op)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    use std::sync::atomic::{AtomicU64, Ordering};

    fn temp_path() -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "rl4burn_tune_journal_test_{}_{}.jsonl",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        path
    }

    #[test]
    fn test_journal_create_and_replay() {
        let path = temp_path();
        let _ = fs::remove_file(&path);

        // Create and populate
        {
            let mut storage = JournalStorage::open(&path).unwrap();
            let id = storage.create_study(Direction::Minimize).unwrap();
            assert_eq!(id, 0);

            let mut trial = FrozenTrial::new(0);
            trial.state = TrialState::Complete;
            trial.value = Some(3.14);
            trial.params.insert("x".to_string(), 1.0);
            storage.add_trial(id, trial).unwrap();

            let mut trial2 = FrozenTrial::new(1);
            trial2.state = TrialState::Complete;
            trial2.value = Some(2.71);
            storage.add_trial(id, trial2).unwrap();
        }

        // Replay from file
        {
            let storage = JournalStorage::open(&path).unwrap();
            let trials = storage.get_all_trials(0).unwrap();
            assert_eq!(trials.len(), 2);
            assert_eq!(trials[0].value, Some(3.14));
            assert_eq!(trials[0].params["x"], 1.0);
            assert_eq!(trials[1].value, Some(2.71));
        }

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn test_journal_set_operations() {
        let path = temp_path();
        let _ = fs::remove_file(&path);

        {
            let mut storage = JournalStorage::open(&path).unwrap();
            let id = storage.create_study(Direction::Maximize).unwrap();
            let trial = FrozenTrial::new(0);
            storage.add_trial(id, trial).unwrap();
            storage
                .set_trial_state(id, 0, TrialState::Complete)
                .unwrap();
            storage.set_trial_value(id, 0, 99.0).unwrap();
            storage.set_trial_param(id, 0, "lr", 0.01).unwrap();
            storage
                .set_trial_user_attr(id, 0, "model", "gpt")
                .unwrap();
        }

        // Verify replay
        {
            let storage = JournalStorage::open(&path).unwrap();
            assert_eq!(
                storage.get_study_direction(0).unwrap(),
                Direction::Maximize
            );
            let trials = storage.get_all_trials(0).unwrap();
            assert_eq!(trials[0].state, TrialState::Complete);
            assert_eq!(trials[0].value, Some(99.0));
            assert_eq!(trials[0].params["lr"], 0.01);
            assert_eq!(trials[0].user_attrs["model"], "gpt");
        }

        let _ = fs::remove_file(&path);
    }
}
