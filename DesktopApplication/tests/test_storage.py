import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from fastapi.testclient import TestClient

from server.main import create_app
from server.storage import DatabaseRecoveryRequired, LocalStore, SCHEMA_VERSION, StorageError


class LocalStoreTests(unittest.TestCase):
    def test_initializes_wal_schema_and_persists_local_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory))
            store.initialize()

            self.assertEqual(store.schema_version(), SCHEMA_VERSION)
            for directory in (
                store.db_dir,
                store.audio_dir,
                store.drafts_dir,
                store.logs_dir,
                store.settings_dir,
            ):
                self.assertTrue(directory.is_dir())

            with closing(sqlite3.connect(store.database_path)) as connection:
                journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
            self.assertEqual(journal_mode.lower(), "wal")
            self.assertTrue({"generations", "drafts", "settings"}.issubset(tables))
            self.assertFalse({"users", "sessions", "auth_tokens"}.intersection(tables))

            store.save_draft("current", "សួស្តី", {"pace": "natural"})
            store.save_setting("last_voice", "vanna")
            store.save_generation("generation-1", "សួស្តី", "vanna", "natural", 4)

            draft = store.get_draft("current")
            self.assertIsNotNone(draft)
            self.assertEqual(draft["text"], "សួស្តី")
            self.assertEqual(draft["settings"], {"pace": "natural"})
            self.assertEqual(store.get_setting("last_voice"), "vanna")

    def test_migration_creates_a_consistent_pre_migration_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory))
            store.db_dir.mkdir(parents=True)
            with closing(sqlite3.connect(store.database_path)) as connection:
                connection.execute("CREATE TABLE legacy_marker (value TEXT NOT NULL)")
                connection.execute("INSERT INTO legacy_marker VALUES ('preserve me')")
                connection.execute("PRAGMA user_version=0")
                connection.commit()

            store.initialize()
            backup = store.latest_backup()

            self.assertIsNotNone(backup)
            self.assertEqual(store.schema_version(), SCHEMA_VERSION)
            if backup is None:
                self.fail("Migration did not create a pre-migration backup.")
            with closing(sqlite3.connect(backup)) as connection:
                self.assertEqual(connection.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 0)
                self.assertEqual(
                    connection.execute("SELECT value FROM legacy_marker").fetchone()[0],
                    "preserve me",
                )

    def test_corruption_requires_recovery_and_restores_latest_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory))
            store.db_dir.mkdir(parents=True)
            with closing(sqlite3.connect(store.database_path)) as connection:
                connection.execute("CREATE TABLE legacy_marker (value TEXT NOT NULL)")
                connection.execute("INSERT INTO legacy_marker VALUES ('recover me')")
                connection.execute("PRAGMA user_version=0")
                connection.commit()
            store.initialize()
            store.database_path.write_bytes(b"not a sqlite database")

            with self.assertRaises(DatabaseRecoveryRequired) as raised:
                LocalStore(Path(temporary_directory)).initialize()

            self.assertIsNotNone(raised.exception.backup_path)
            self.assertIn("Use the recovery action", str(raised.exception))

            restored_store = LocalStore(Path(temporary_directory))
            restored_backup = restored_store.restore_latest_backup()
            restored_store.initialize()

            self.assertTrue(restored_backup.is_file())
            self.assertEqual(restored_store.schema_version(), SCHEMA_VERSION)
            with closing(sqlite3.connect(restored_store.database_path)) as connection:
                self.assertEqual(
                    connection.execute("SELECT value FROM legacy_marker").fetchone()[0],
                    "recover me",
                )

    def test_corrupt_backup_returns_a_recovery_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory))
            store.backup_dir.mkdir(parents=True)
            backup = store.backup_dir / "broken.sqlite3"
            backup.write_bytes(b"invalid backup")

            with self.assertRaisesRegex(StorageError, "Could not restore database backup"):
                store.restore_latest_backup()

    def test_local_api_offers_and_completes_database_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root)
            store.db_dir.mkdir(parents=True)
            with closing(sqlite3.connect(store.database_path)) as connection:
                connection.execute("CREATE TABLE legacy_marker (value TEXT NOT NULL)")
                connection.execute("INSERT INTO legacy_marker VALUES ('recover me')")
                connection.execute("PRAGMA user_version=0")
                connection.commit()
            store.initialize()
            store.database_path.write_bytes(b"corrupted")

            gpu_reader = lambda: {
                "available": True,
                "gpu_name": "Test GPU",
                "free_mib": 4096,
                "total_mib": 6141,
                "minimum_free_mib": 1536,
                "safe_to_load_model": True,
                "warning": "Test GPU status.",
                "checked_at": "test",
            }
            application = create_app(root, gpu_reader)
            with TestClient(application) as client:
                status = client.get("/api/storage/status")
                blocked = client.get("/api/storage/drafts/current")
                restore = client.post("/api/storage/restore")
                ready = client.get("/api/storage/status")

            self.assertEqual(status.json()["status"], "recovery_required")
            self.assertTrue(status.json()["backup_available"])
            self.assertEqual(blocked.status_code, 503)
            self.assertEqual(restore.status_code, 200, restore.text)
            self.assertEqual(ready.json()["status"], "ready")


if __name__ == "__main__":
    unittest.main()
