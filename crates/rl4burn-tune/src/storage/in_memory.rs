//! In-memory storage backend.

use crate::study::Direction;
use crate::trial::{FrozenTrial, TrialState};

use super::{Storage, StorageError};

/// In-memory storage that holds all data in `Vec`s and `HashMap`s.
///
/// This is the default storage backend. Data is lost when the process exits.
#[derive(Debug, Default)]
pub struct InMemoryStorage {
    studies: Vec<StudyRecord>,
}

#[derive(Debug)]
struct StudyRecord {
    direction: Direction,
    trials: Vec<FrozenTrial>,
}

impl InMemoryStorage {
    /// Create a new empty in-memory storage.
    pub fn new() -> Self {
        Self::default()
    }

    fn get_study(&self, study_id: usize) -> Result<&StudyRecord, StorageError> {
        self.studies
            .get(study_id)
            .ok_or(StorageError::StudyNotFound(study_id))
    }

    fn get_study_mut(&mut self, study_id: usize) -> Result<&mut StudyRecord, StorageError> {
        self.studies
            .get_mut(study_id)
            .ok_or(StorageError::StudyNotFound(study_id))
    }

    fn get_trial_mut(
        study: &mut StudyRecord,
        study_id: usize,
        trial_number: usize,
    ) -> Result<&mut FrozenTrial, StorageError> {
        study
            .trials
            .iter_mut()
            .find(|t| t.number == trial_number)
            .ok_or(StorageError::TrialNotFound {
                study_id,
                trial_number,
            })
    }
}

impl Storage for InMemoryStorage {
    fn create_study(&mut self, direction: Direction) -> Result<usize, StorageError> {
        let id = self.studies.len();
        self.studies.push(StudyRecord {
            direction,
            trials: Vec::new(),
        });
        Ok(id)
    }

    fn get_study_direction(&self, study_id: usize) -> Result<Direction, StorageError> {
        Ok(self.get_study(study_id)?.direction)
    }

    fn get_all_trials(&self, study_id: usize) -> Result<Vec<FrozenTrial>, StorageError> {
        Ok(self.get_study(study_id)?.trials.clone())
    }

    fn add_trial(&mut self, study_id: usize, trial: FrozenTrial) -> Result<(), StorageError> {
        self.get_study_mut(study_id)?.trials.push(trial);
        Ok(())
    }

    fn set_trial_state(
        &mut self,
        study_id: usize,
        trial_number: usize,
        state: TrialState,
    ) -> Result<(), StorageError> {
        let study = self.get_study_mut(study_id)?;
        Self::get_trial_mut(study, study_id, trial_number)?.state = state;
        Ok(())
    }

    fn set_trial_value(
        &mut self,
        study_id: usize,
        trial_number: usize,
        value: f64,
    ) -> Result<(), StorageError> {
        let study = self.get_study_mut(study_id)?;
        Self::get_trial_mut(study, study_id, trial_number)?.value = Some(value);
        Ok(())
    }

    fn set_trial_param(
        &mut self,
        study_id: usize,
        trial_number: usize,
        name: &str,
        value: f64,
    ) -> Result<(), StorageError> {
        let study = self.get_study_mut(study_id)?;
        Self::get_trial_mut(study, study_id, trial_number)?
            .params
            .insert(name.to_string(), value);
        Ok(())
    }

    fn set_trial_user_attr(
        &mut self,
        study_id: usize,
        trial_number: usize,
        key: &str,
        value: &str,
    ) -> Result<(), StorageError> {
        let study = self.get_study_mut(study_id)?;
        Self::get_trial_mut(study, study_id, trial_number)?
            .user_attrs
            .insert(key.to_string(), value.to_string());
        Ok(())
    }

    fn set_trial_system_attr(
        &mut self,
        study_id: usize,
        trial_number: usize,
        key: &str,
        value: &str,
    ) -> Result<(), StorageError> {
        let study = self.get_study_mut(study_id)?;
        Self::get_trial_mut(study, study_id, trial_number)?
            .system_attrs
            .insert(key.to_string(), value.to_string());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_retrieve_study() {
        let mut storage = InMemoryStorage::new();
        let id = storage.create_study(Direction::Minimize).unwrap();
        assert_eq!(id, 0);
        assert_eq!(
            storage.get_study_direction(id).unwrap(),
            Direction::Minimize
        );
    }

    #[test]
    fn test_add_and_get_trials() {
        let mut storage = InMemoryStorage::new();
        let id = storage.create_study(Direction::Minimize).unwrap();

        let mut trial = FrozenTrial::new(0);
        trial.state = TrialState::Complete;
        trial.value = Some(1.0);
        trial.params.insert("x".to_string(), 0.5);
        storage.add_trial(id, trial).unwrap();

        let trials = storage.get_all_trials(id).unwrap();
        assert_eq!(trials.len(), 1);
        assert_eq!(trials[0].value, Some(1.0));
        assert_eq!(trials[0].params["x"], 0.5);
    }

    #[test]
    fn test_set_trial_state_and_value() {
        let mut storage = InMemoryStorage::new();
        let id = storage.create_study(Direction::Minimize).unwrap();

        let trial = FrozenTrial::new(0);
        storage.add_trial(id, trial).unwrap();

        storage
            .set_trial_state(id, 0, TrialState::Complete)
            .unwrap();
        storage.set_trial_value(id, 0, 42.0).unwrap();

        let trials = storage.get_all_trials(id).unwrap();
        assert_eq!(trials[0].state, TrialState::Complete);
        assert_eq!(trials[0].value, Some(42.0));
    }

    #[test]
    fn test_trial_not_found() {
        let mut storage = InMemoryStorage::new();
        let id = storage.create_study(Direction::Minimize).unwrap();
        assert!(storage.set_trial_value(id, 99, 1.0).is_err());
    }

    #[test]
    fn test_study_not_found() {
        let storage = InMemoryStorage::new();
        assert!(storage.get_all_trials(99).is_err());
    }

    #[test]
    fn test_trial_attrs() {
        let mut storage = InMemoryStorage::new();
        let id = storage.create_study(Direction::Minimize).unwrap();
        let trial = FrozenTrial::new(0);
        storage.add_trial(id, trial).unwrap();

        storage
            .set_trial_user_attr(id, 0, "note", "test")
            .unwrap();
        storage
            .set_trial_system_attr(id, 0, "generation", "1")
            .unwrap();

        let trials = storage.get_all_trials(id).unwrap();
        assert_eq!(trials[0].user_attrs["note"], "test");
        assert_eq!(trials[0].system_attrs["generation"], "1");
    }
}
