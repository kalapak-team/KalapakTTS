import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from app.launcher import _read_window_size, _write_window_size
from server.gpu import MIN_FREE_VRAM_MIB
from server.main import create_app
from server.storage import DatabaseRecoveryRequired, LocalStore, SCHEMA_VERSION, StorageError


class LocalStoreTests(unittest.TestCase):
    def test_window_dimensions_are_persisted_and_clamped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory))
            store.initialize()

            self.assertEqual(_read_window_size(store), (960, 680))
            _write_window_size(store, {"width": 1200, "height": 800})
            self.assertEqual(_read_window_size(store), (1200, 800))
            _write_window_size(store, {"width": 100, "height": 9000})
            self.assertEqual(_read_window_size(store), (640, 2160))

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
            store.save_generation(
                "generation-1",
                "សួស្តី",
                "vanna",
                "natural",
                4,
                captions=[{"start_seconds": 0.0, "end_seconds": 1.0, "text": "សួស្តី"}],
            )

            draft = store.get_draft("current")
            self.assertIsNotNone(draft)
            self.assertEqual(draft["text"], "សួស្តី")
            self.assertEqual(draft["settings"], {"pace": "natural"})
            self.assertEqual(store.get_setting("last_voice"), "vanna")
            generation = store.get_generation("generation-1")
            self.assertIsNotNone(generation)
            self.assertEqual(generation["captions"][0]["text"], "សួស្តី")

    def test_stage_one_database_migrates_caption_segments_with_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory))
            store.db_dir.mkdir(parents=True)
            with closing(sqlite3.connect(store.database_path)) as connection:
                connection.execute(
                    """
                    CREATE TABLE generations (
                        id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        title TEXT,
                        text TEXT NOT NULL,
                        voice_id TEXT NOT NULL,
                        pace TEXT NOT NULL,
                        steps INTEGER NOT NULL,
                        speed REAL NOT NULL DEFAULT 1.0,
                        audio_path TEXT,
                        audio_sha256 TEXT,
                        audio_duration REAL,
                        is_favorite INTEGER NOT NULL DEFAULT 0,
                        status TEXT NOT NULL DEFAULT 'completed'
                    )
                    """
                )
                connection.execute("PRAGMA user_version=1")
                connection.commit()

            store.initialize()
            backup = store.latest_backup()

            self.assertEqual(store.schema_version(), SCHEMA_VERSION)
            self.assertIsNotNone(backup)
            with closing(sqlite3.connect(store.database_path)) as connection:
                columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(generations)")
                }
            self.assertIn("captions_json", columns)
            if backup is None:
                self.fail("Stage 1 migration did not create a pre-migration backup.")
            with closing(sqlite3.connect(backup)) as connection:
                self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)

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

    def test_history_search_filters_favorites_duplicate_and_delete_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory))
            store.initialize()
            audio_path = store.audio_dir / "history-1.wav"
            audio_path.write_bytes(b"wave data")
            store.save_generation(
                "history-1",
                "សួស្តី Fit & Sleek 100%",
                "vanna",
                "natural",
                4,
                title="Fit & Sleek",
                speed=1.1,
                audio_path=str(audio_path),
                audio_sha256="hash",
                audio_duration=1.25,
            )
            store.save_generation(
                "history-2",
                "Different sample",
                "vanna",
                "brisk",
                8,
                title="Second",
            )

            store.rename_generation("history-1", "Interview sample")
            store.set_generation_favorite("history-1", True)
            matches = store.list_generations(
                search="Interview",
                date_from=date.today().isoformat(),
                date_to=date.today().isoformat(),
                pace="natural",
                steps=4,
                favorites_only=True,
            )
            duplicate = store.duplicate_generation("history-1")

            self.assertEqual([item["id"] for item in matches], ["history-1"])
            self.assertEqual(matches[0]["title"], "Interview sample")
            self.assertTrue(matches[0]["is_favorite"])
            self.assertFalse(matches[0]["audio_missing"])
            self.assertIsNotNone(duplicate)
            if duplicate is None:
                self.fail("History duplicate was not created.")
            duplicate_audio = Path(duplicate["audio_path"])
            self.assertEqual(duplicate_audio.read_bytes(), b"wave data")
            self.assertTrue(store.delete_generation("history-1"))
            self.assertFalse(audio_path.exists())
            self.assertIsNone(store.get_generation("history-1"))
            self.assertTrue(duplicate_audio.is_file())

    def test_media_folder_move_is_safe_and_persists_across_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "profile")
            store.initialize()
            original_audio = store.audio_dir / "media-1.wav"
            original_audio.write_bytes(b"generated audio")
            store.save_generation(
                "media-1",
                "Sample",
                "vanna",
                "natural",
                4,
                audio_path=str(original_audio),
                audio_sha256="hash",
                audio_duration=1.0,
            )
            new_folder = root / "external-media"

            moved = store.move_audio_folder(str(new_folder))
            restarted = LocalStore(root / "profile")
            restarted.initialize()

            self.assertEqual(moved["moved_files"], 1)
            self.assertEqual(restarted.audio_dir, new_folder.resolve())
            self.assertFalse(original_audio.exists())
            generation = restarted.get_generation("media-1")
            self.assertIsNotNone(generation)
            if generation is None:
                self.fail("Generation was lost while moving media.")
            self.assertEqual(Path(generation["audio_path"]), new_folder / "media-1.wav")
            self.assertEqual(Path(generation["audio_path"]).read_bytes(), b"generated audio")
            self.assertEqual(restarted.audio_storage_bytes(), len(b"generated audio"))

    def test_media_folder_collision_does_not_change_history_or_source_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "profile")
            store.initialize()
            original_audio = store.audio_dir / "collision-1.wav"
            original_audio.write_bytes(b"original audio")
            store.save_generation(
                "collision-1",
                "Sample",
                "vanna",
                "natural",
                4,
                audio_path=str(original_audio),
                audio_sha256="hash",
                audio_duration=1.0,
            )
            target_folder = root / "existing-media"
            target_folder.mkdir()
            target_file = target_folder / "collision-1.wav"
            target_file.write_bytes(b"different existing file")

            with self.assertRaisesRegex(StorageError, "already exists with different contents"):
                store.move_audio_folder(str(target_folder))

            generation = store.get_generation("collision-1")
            self.assertIsNotNone(generation)
            if generation is None:
                self.fail("Media collision unexpectedly deleted its history row.")
            self.assertEqual(Path(generation["audio_path"]), original_audio)
            self.assertEqual(original_audio.read_bytes(), b"original audio")
            self.assertEqual(target_file.read_bytes(), b"different existing file")
            self.assertEqual(list(target_folder.glob(".*.moving.wav")), [])

    def test_cleanup_removes_old_generations_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalStore(Path(temporary_directory))
            store.initialize()
            audio_path = store.audio_dir / "old.wav"
            audio_path.write_bytes(b"old audio")
            store.save_generation(
                "old",
                "Old speech",
                "vanna",
                "natural",
                4,
                audio_path=str(audio_path),
                audio_sha256="hash",
                audio_duration=1.0,
            )

            removed = store.cleanup_generations_before(
                f"{(date.today() + timedelta(days=1)).isoformat()}T00:00:00+00:00"
            )

            self.assertEqual(removed, 1)
            self.assertEqual(store.generation_count(), 0)
            self.assertFalse(audio_path.exists())
            self.assertEqual(store.audio_storage_bytes(), 0)

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
                "free_mib": 6141,
                "total_mib": 6141,
                "minimum_free_mib": MIN_FREE_VRAM_MIB,
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
