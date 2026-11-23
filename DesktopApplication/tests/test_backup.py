import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from server.backup import BackupError, export_profile, import_profile
from server.main import create_app
from server.storage import LocalStore


def gpu_status() -> dict[str, object]:
    return {
        "available": False,
        "cuda_available": False,
        "gpu_name": None,
        "free_mib": None,
        "total_mib": None,
        "minimum_free_mib": None,
        "safe_to_load_model": False,
        "warning": "GPU unavailable in backup tests.",
        "checked_at": "test",
    }


class BackupArchiveTests(unittest.TestCase):
    def test_backup_restores_history_draft_settings_audio_and_voice_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = LocalStore(root / "source")
            source.initialize()
            external_audio = root / "external-media"
            source.move_audio_folder(str(external_audio))
            audio_path = external_audio / "generation-1.wav"
            audio_path.write_bytes(b"local wav data")
            source.save_generation(
                "generation-1",
                "សួស្តី",
                "vanna",
                "natural",
                4,
                title="Khmer sample",
                audio_path=str(audio_path),
                audio_sha256=hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                audio_duration=1.5,
                captions=[{"start_seconds": 0.0, "end_seconds": 1.5, "text": "សួស្តី"}],
            )
            source.save_draft("current", "Draft Khmer", {"pace": "relaxed"})
            source.save_setting("last_voice", "vanna")
            custom_voice = source.voice_dir / "custom" / "speaker.wav"
            custom_voice.parent.mkdir(parents=True)
            custom_voice.write_bytes(b"custom voice reference")
            bundled_voice_dir = root / "bundled-voices"
            bundled_voice_dir.mkdir()
            (bundled_voice_dir / "vanna.wav").write_bytes(b"Vanna preset")
            archive_path = root / "profile.zip"
            export_profile(source, archive_path, bundled_voice_dir)

            restored = LocalStore(root / "fresh-profile")
            restored.initialize()
            result = import_profile(restored, archive_path)

            generation = restored.get_generation("generation-1")
            self.assertIsNotNone(generation)
            if generation is None:
                self.fail("The generation was not restored.")
            restored_audio = Path(generation["audio_path"])
            self.assertEqual(restored_audio, restored.root / "audio" / restored_audio.name)
            self.assertEqual(restored_audio.read_bytes(), b"local wav data")
            self.assertEqual(
                hashlib.sha256(restored_audio.read_bytes()).hexdigest(),
                generation["audio_sha256"],
            )
            self.assertEqual(restored.get_draft("current")["text"], "Draft Khmer")
            self.assertEqual(restored.get_setting("last_voice"), "vanna")
            self.assertEqual(
                (restored.voice_dir / "custom" / "speaker.wav").read_bytes(),
                b"custom voice reference",
            )
            self.assertEqual(
                (restored.voice_dir / "presets" / "vanna.wav").read_bytes(),
                b"Vanna preset",
            )
            self.assertEqual(restored.audio_dir, (restored.root / "audio").resolve())
            self.assertEqual(result["generation_count"], 1)
            self.assertEqual(result["audio_files"], 1)
            self.assertEqual(result["missing_audio"], 0)
            self.assertTrue((restored.backup_dir / result["pre_import_backup"]).is_file())

    def test_api_export_import_round_trip_on_a_fresh_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_app = create_app(root / "source", gpu_status)
            with TestClient(source_app) as source_client:
                source_store = source_client.app.state.local_store
                audio_path = source_store.audio_dir / "api-generation.wav"
                audio_path.write_bytes(b"API audio")
                source_store.save_generation(
                    "api-generation",
                    "សួស្តី from API",
                    "vanna",
                    "natural",
                    4,
                    audio_path=str(audio_path),
                    audio_sha256=hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                    audio_duration=1.0,
                )
                source_store.save_draft("current", "API draft", {"steps": 8})
                exported = source_client.get("/api/storage/export")

            self.assertEqual(exported.status_code, 200, exported.text)
            self.assertEqual(exported.headers["content-type"], "application/zip")

            target_app = create_app(root / "target", gpu_status)
            with TestClient(target_app) as target_client:
                imported = target_client.put(
                    "/api/storage/import",
                    content=exported.content,
                    headers={"Content-Type": "application/zip"},
                )
                restored = target_client.get("/api/tts/generations")
                draft = target_client.get("/api/storage/drafts/current")

            self.assertEqual(imported.status_code, 200, imported.text)
            self.assertEqual(imported.json()["generation_count"], 1)
            self.assertEqual(restored.json()["items"][0]["id"], "api-generation")
            self.assertEqual(draft.json()["text"], "API draft")

    def test_api_import_recovers_a_profile_with_a_corrupt_database(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = LocalStore(root / "source")
            source.initialize()
            source.save_draft("current", "Restored after corruption", {})
            archive_path = root / "valid-backup.zip"
            export_profile(source, archive_path)

            target_root = root / "corrupt-profile"
            damaged = LocalStore(target_root)
            damaged.initialize()
            damaged.database_path.write_bytes(b"corrupt sqlite database")
            target_app = create_app(target_root, gpu_status)
            with TestClient(target_app) as client:
                before = client.get("/api/storage/status")
                imported = client.put(
                    "/api/storage/import",
                    content=archive_path.read_bytes(),
                    headers={"Content-Type": "application/zip"},
                )
                after = client.get("/api/storage/status")
                draft = client.get("/api/storage/drafts/current")

            self.assertEqual(before.json()["status"], "recovery_required")
            self.assertEqual(imported.status_code, 200, imported.text)
            self.assertEqual(after.json()["status"], "ready")
            self.assertEqual(draft.json()["text"], "Restored after corruption")
            self.assertIsNone(imported.json()["pre_import_backup"])

    def test_tampered_backup_is_rejected_without_replacing_current_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = LocalStore(root / "profile")
            store.initialize()
            store.save_draft("current", "Keep this draft", {})
            archive_path = root / "good.zip"
            export_profile(store, archive_path)

            tampered_path = root / "tampered.zip"
            with zipfile.ZipFile(archive_path) as original, zipfile.ZipFile(
                tampered_path,
                "w",
            ) as tampered:
                for entry in original.infolist():
                    content = original.read(entry.filename)
                    if entry.filename == "database/kalapaktts.sqlite3":
                        content += b"tampered"
                    tampered.writestr(entry.filename, content)

            with self.assertRaisesRegex(BackupError, "Checksum verification failed"):
                import_profile(store, tampered_path)

            self.assertEqual(store.get_draft("current")["text"], "Keep this draft")

    def test_path_traversal_archive_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "unsafe.zip"
            manifest = {
                "format": "kalapaktts-desktop-backup",
                "version": 1,
                "files": {
                    "../outside.txt": {
                        "size": 1,
                        "sha256": hashlib.sha256(b"x").hexdigest(),
                    },
                    "database/kalapaktts.sqlite3": {
                        "size": 1,
                        "sha256": hashlib.sha256(b"x").hexdigest(),
                    },
                },
                "audio": {},
            }
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("manifest.json", json.dumps(manifest))
                archive.writestr("../outside.txt", b"x")
                archive.writestr("database/kalapaktts.sqlite3", b"x")

            store = LocalStore(root / "profile")
            store.initialize()
            with self.assertRaisesRegex(BackupError, "unsafe file path"):
                import_profile(store, archive_path)


if __name__ == "__main__":
    unittest.main()
