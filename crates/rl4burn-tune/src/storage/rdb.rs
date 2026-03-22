//! SQLite-based storage backend for persisting study data.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use rusqlite::{params, Connection};

use super::{Storage, StorageError};
use crate::study::Direction;
use crate::trial::{FrozenTrial, TrialState};

/// A SQLite-backed storage implementation.
pub struct SqliteStorage {
    conn: Mutex<Connection>,
}

impl SqliteStorage {
    /// Open (or create) a SQLite database at the given path.
    pub fn new(path: impl AsRef<Path>) -> Result<Self, StorageError> {
        let conn = Connection::open(path).map_err(|e| {
            StorageError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
        })?;
        let storage = Self { conn: Mutex::new(conn) };
        storage.init_tables()?;
        Ok(storage)
    }

    /// Create an in-memory SQLite database (useful for testing).
    pub fn in_memory() -> Result<Self, StorageError> {
        let conn = Connection::open_in_memory().map_err(|e| {
            StorageError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
        })?;
        let storage = Self { conn: Mutex::new(conn) };
        storage.init_tables()?;
        Ok(storage)
    }

    fn init_tables(&self) -> Result<(), StorageError> {
        self.conn.lock().unwrap()
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS studies (
                    study_id INTEGER PRIMARY KEY,
                    direction TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trials (
                    trial_id INTEGER PRIMARY KEY,
                    study_id INTEGER NOT NULL,
                    number INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    value REAL,
                    FOREIGN KEY (study_id) REFERENCES studies(study_id)
                );
                CREATE TABLE IF NOT EXISTS trial_params (
                    trial_id INTEGER NOT NULL,
                    param_name TEXT NOT NULL,
                    param_value REAL NOT NULL,
                    FOREIGN KEY (trial_id) REFERENCES trials(trial_id),
                    PRIMARY KEY (trial_id, param_name)
                );",
            )
            .map_err(|e| {
                StorageError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
            })?;
        Ok(())
    }

    fn sqlite_err(e: rusqlite::Error) -> StorageError {
        StorageError::Io(std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }

    fn direction_to_str(d: Direction) -> &'static str {
        match d {
            Direction::Minimize => "Minimize",
            Direction::Maximize => "Maximize",
        }
    }

    fn str_to_direction(s: &str) -> Direction {
        match s {
            "Maximize" => Direction::Maximize,
            _ => Direction::Minimize,
        }
    }

    fn state_to_str(s: TrialState) -> &'static str {
        match s {
            TrialState::Waiting => "Waiting",
            TrialState::Running => "Running",
            TrialState::Complete => "Complete",
            TrialState::Pruned => "Pruned",
            TrialState::Fail => "Fail",
        }
    }

    fn str_to_state(s: &str) -> TrialState {
        match s {
            "Waiting" => TrialState::Waiting,
            "Running" => TrialState::Running,
            "Complete" => TrialState::Complete,
            "Pruned" => TrialState::Pruned,
            "Fail" => TrialState::Fail,
            _ => TrialState::Fail,
        }
    }

    /// Look up the internal rowid for a trial by study_id and trial number.
    fn trial_rowid(&self, study_id: usize, trial_number: usize) -> Result<i64, StorageError> {
        self.conn.lock().unwrap()
            .query_row(
                "SELECT trial_id FROM trials WHERE study_id = ?1 AND number = ?2",
                params![study_id as i64, trial_number as i64],
                |row| row.get(0),
            )
            .map_err(|_| StorageError::TrialNotFound {
                study_id,
                trial_number,
            })
    }
}

impl Storage for SqliteStorage {
    fn create_study(&mut self, direction: Direction) -> Result<usize, StorageError> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO studies (direction) VALUES (?1)",
            params![Self::direction_to_str(direction)],
        ).map_err(Self::sqlite_err)?;
        Ok(conn.last_insert_rowid() as usize)
    }

    fn get_study_direction(&self, study_id: usize) -> Result<Direction, StorageError> {
        let conn = self.conn.lock().unwrap();
        let dir_str: String = conn
            .query_row(
                "SELECT direction FROM studies WHERE study_id = ?1",
                params![study_id as i64],
                |row| row.get(0),
            )
            .map_err(|_| StorageError::StudyNotFound(study_id))?;
        Ok(Self::str_to_direction(&dir_str))
    }

    fn get_all_trials(&self, study_id: usize) -> Result<Vec<FrozenTrial>, StorageError> {
        let conn = self.conn.lock().unwrap();
        // Verify study exists.
        let _dir_str: String = conn
            .query_row(
                "SELECT direction FROM studies WHERE study_id = ?1",
                params![study_id as i64],
                |row| row.get(0),
            )
            .map_err(|_| StorageError::StudyNotFound(study_id))?;

        let mut stmt = conn
            .prepare(
                "SELECT trial_id, number, state, value FROM trials WHERE study_id = ?1 ORDER BY number",
            )
            .map_err(Self::sqlite_err)?;

        let trial_rows: Vec<(i64, i64, String, Option<f64>)> = stmt
            .query_map(params![study_id as i64], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })
            .map_err(Self::sqlite_err)?
            .filter_map(|r| r.ok())
            .collect();

        let mut trials = Vec::new();
        for (trial_id, number, state_str, value) in trial_rows {
            let mut param_stmt = conn
                .prepare("SELECT param_name, param_value FROM trial_params WHERE trial_id = ?1")
                .map_err(Self::sqlite_err)?;
            let trial_params: HashMap<String, f64> = param_stmt
                .query_map(params![trial_id], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
                })
                .map_err(Self::sqlite_err)?
                .filter_map(|r| r.ok())
                .collect();

            let mut trial = FrozenTrial::new(number as usize);
            trial.state = Self::str_to_state(&state_str);
            trial.value = value;
            trial.params = trial_params;
            trials.push(trial);
        }

        Ok(trials)
    }

    fn add_trial(&mut self, study_id: usize, trial: FrozenTrial) -> Result<(), StorageError> {
        let conn = self.conn.lock().unwrap();
        // Verify study exists.
        let _: String = conn
            .query_row(
                "SELECT direction FROM studies WHERE study_id = ?1",
                params![study_id as i64],
                |row| row.get(0),
            )
            .map_err(|_| StorageError::StudyNotFound(study_id))?;

        conn.execute(
            "INSERT INTO trials (study_id, number, state, value) VALUES (?1, ?2, ?3, ?4)",
            params![
                study_id as i64,
                trial.number as i64,
                Self::state_to_str(trial.state),
                trial.value,
            ],
        ).map_err(Self::sqlite_err)?;

        let trial_id = conn.last_insert_rowid();
        for (name, value) in &trial.params {
            conn.execute(
                "INSERT INTO trial_params (trial_id, param_name, param_value) VALUES (?1, ?2, ?3)",
                params![trial_id, name, value],
            ).map_err(Self::sqlite_err)?;
        }

        Ok(())
    }

    fn set_trial_state(&mut self, study_id: usize, trial_number: usize, state: TrialState) -> Result<(), StorageError> {
        let trial_id = self.trial_rowid(study_id, trial_number)?;
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE trials SET state = ?1 WHERE trial_id = ?2",
            params![Self::state_to_str(state), trial_id],
        ).map_err(Self::sqlite_err)?;
        Ok(())
    }

    fn set_trial_value(&mut self, study_id: usize, trial_number: usize, value: f64) -> Result<(), StorageError> {
        let trial_id = self.trial_rowid(study_id, trial_number)?;
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE trials SET value = ?1 WHERE trial_id = ?2",
            params![value, trial_id],
        ).map_err(Self::sqlite_err)?;
        Ok(())
    }

    fn set_trial_param(&mut self, study_id: usize, trial_number: usize, name: &str, value: f64) -> Result<(), StorageError> {
        let trial_id = self.trial_rowid(study_id, trial_number)?;
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO trial_params (trial_id, param_name, param_value) VALUES (?1, ?2, ?3)",
            params![trial_id, name, value],
        ).map_err(Self::sqlite_err)?;
        Ok(())
    }

    fn set_trial_user_attr(&mut self, study_id: usize, trial_number: usize, _key: &str, _value: &str) -> Result<(), StorageError> {
        let _tid = self.trial_rowid(study_id, trial_number)?;
        Ok(())
    }

    fn set_trial_system_attr(&mut self, study_id: usize, trial_number: usize, _key: &str, _value: &str) -> Result<(), StorageError> {
        let _tid = self.trial_rowid(study_id, trial_number)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_study_and_direction() {
        let mut storage = SqliteStorage::in_memory().unwrap();
        let id = storage.create_study(Direction::Minimize).unwrap();
        assert_eq!(storage.get_study_direction(id).unwrap(), Direction::Minimize);

        let id2 = storage.create_study(Direction::Maximize).unwrap();
        assert_eq!(storage.get_study_direction(id2).unwrap(), Direction::Maximize);
        assert_ne!(id, id2);
    }

    #[test]
    fn test_add_trial_and_get_all_trials() {
        let mut storage = SqliteStorage::in_memory().unwrap();
        let study_id = storage.create_study(Direction::Minimize).unwrap();

        let mut trial = FrozenTrial::new(0);
        trial.state = TrialState::Complete;
        trial.value = Some(0.5);
        trial.params.insert("lr".to_string(), 0.01);
        trial.params.insert("batch_size".to_string(), 32.0);

        storage.add_trial(study_id, trial).unwrap();

        let trials = storage.get_all_trials(study_id).unwrap();
        assert_eq!(trials.len(), 1);
        assert_eq!(trials[0].number, 0);
        assert_eq!(trials[0].state, TrialState::Complete);
        assert_eq!(trials[0].value, Some(0.5));
        assert_eq!(trials[0].params["lr"], 0.01);
        assert_eq!(trials[0].params["batch_size"], 32.0);
    }

    #[test]
    fn test_set_trial_state_and_value() {
        let mut storage = SqliteStorage::in_memory().unwrap();
        let study_id = storage.create_study(Direction::Minimize).unwrap();

        let trial = FrozenTrial::new(0);
        storage.add_trial(study_id, trial).unwrap();

        storage
            .set_trial_state(study_id, 0, TrialState::Complete)
            .unwrap();
        storage.set_trial_value(study_id, 0, 1.23).unwrap();

        let trials = storage.get_all_trials(study_id).unwrap();
        assert_eq!(trials[0].state, TrialState::Complete);
        assert_eq!(trials[0].value, Some(1.23));
    }

    #[test]
    fn test_set_trial_param() {
        let mut storage = SqliteStorage::in_memory().unwrap();
        let study_id = storage.create_study(Direction::Minimize).unwrap();

        let trial = FrozenTrial::new(0);
        storage.add_trial(study_id, trial).unwrap();

        storage.set_trial_param(study_id, 0, "x", 42.0).unwrap();

        let trials = storage.get_all_trials(study_id).unwrap();
        assert_eq!(trials[0].params["x"], 42.0);
    }

    #[test]
    fn test_study_not_found() {
        let storage = SqliteStorage::in_memory().unwrap();
        assert!(storage.get_study_direction(999).is_err());
    }

    #[test]
    fn test_multiple_trials_round_trip() {
        let mut storage = SqliteStorage::in_memory().unwrap();
        let study_id = storage.create_study(Direction::Maximize).unwrap();

        for i in 0..5 {
            let mut trial = FrozenTrial::new(i);
            trial.state = TrialState::Complete;
            trial.value = Some(i as f64 * 0.1);
            trial.params.insert("x".to_string(), i as f64);
            storage.add_trial(study_id, trial).unwrap();
        }

        let trials = storage.get_all_trials(study_id).unwrap();
        assert_eq!(trials.len(), 5);
        for (i, t) in trials.iter().enumerate() {
            assert_eq!(t.number, i);
            assert_eq!(t.value, Some(i as f64 * 0.1));
            assert_eq!(t.params["x"], i as f64);
        }
    }
}
