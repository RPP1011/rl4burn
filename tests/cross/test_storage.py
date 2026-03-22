"""Cross-verification tests for storage backends between rl4burn_tune_py and Optuna."""

import os
import tempfile
import pytest

import rl4burn_tune_py as rl

optuna = pytest.importorskip("optuna")

from .conftest import CROSS_SEED


# ---------------------------------------------------------------------------
# Journal storage round-trip
# ---------------------------------------------------------------------------


class TestJournalStorageRoundTrip:
    """Verify journal storage writes and reads back identical trial data."""

    def test_rl4burn_journal_roundtrip(self):
        """Write trials via rl4burn journal storage, read them back."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            # Write
            storage = rl.journal_storage(path)
            study_id = storage.create_study("minimize")
            storage.add_trial(study_id, 0, {"x": 1.0, "y": 2.0}, 3.0, "complete")
            storage.add_trial(study_id, 1, {"x": 3.0, "y": 4.0}, 7.0, "complete")

            # Read back (new storage instance to force replay from file)
            storage2 = rl.journal_storage(path)
            trials = storage2.get_all_trials(study_id)

            assert len(trials) == 2
            assert trials[0]["params"]["x"] == 1.0
            assert trials[0]["params"]["y"] == 2.0
            assert trials[0]["value"] == 3.0
            assert trials[1]["params"]["x"] == 3.0
            assert trials[1]["value"] == 7.0
        finally:
            os.unlink(path)

    def test_in_memory_storage_basic(self):
        """In-memory storage should store and retrieve trials."""
        storage = rl.in_memory_storage()
        study_id = storage.create_study("minimize")
        storage.add_trial(study_id, 0, {"x": 1.5}, 2.25, "complete")

        trials = storage.get_all_trials(study_id)
        assert len(trials) == 1
        assert trials[0]["params"]["x"] == 1.5
        assert trials[0]["value"] == 2.25

    def test_multiple_studies(self):
        """Storage should handle multiple studies independently."""
        storage = rl.in_memory_storage()
        id1 = storage.create_study("minimize")
        id2 = storage.create_study("maximize")

        storage.add_trial(id1, 0, {"x": 1.0}, 1.0, "complete")
        storage.add_trial(id2, 0, {"y": 2.0}, 2.0, "complete")

        trials1 = storage.get_all_trials(id1)
        trials2 = storage.get_all_trials(id2)

        assert len(trials1) == 1
        assert len(trials2) == 1
        assert "x" in trials1[0]["params"]
        assert "y" in trials2[0]["params"]


# ---------------------------------------------------------------------------
# Optuna journal interop (future: Phase 6)
# ---------------------------------------------------------------------------


class TestOptunaJournalInterop:
    """Cross-verify that rl4burn can read Optuna journal files and vice versa."""

    @pytest.mark.skip(reason="Journal format interop requires format alignment work")
    def test_optuna_write_rl4burn_read(self):
        """Write via Optuna JournalFileStorage, read via rl4burn journal_storage."""
        pass

    @pytest.mark.skip(reason="Journal format interop requires format alignment work")
    def test_rl4burn_write_optuna_read(self):
        """Write via rl4burn journal_storage, read via Optuna JournalFileStorage."""
        pass


# ---------------------------------------------------------------------------
# SQLite storage (Phase 6a placeholder)
# ---------------------------------------------------------------------------


class TestSQLiteStorage:
    """Placeholder for SQLite storage (Phase 6a)."""

    @pytest.mark.skip(reason="Phase 6a: SQLite storage not yet implemented")
    def test_sqlite_interop(self):
        pass
