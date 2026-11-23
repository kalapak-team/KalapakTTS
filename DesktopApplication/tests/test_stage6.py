import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.launcher import _claim_single_instance_mutex
from server.main import create_app
from server.tts_engine import TTSEngine


def gpu_status() -> dict[str, object]:
    return {
        "available": False,
        "cuda_available": False,
        "gpu_name": None,
        "free_mib": None,
        "total_mib": None,
        "minimum_free_mib": None,
        "safe_to_load_model": False,
        "warning": "GPU unavailable in Stage 6 tests.",
        "checked_at": "test",
    }


class SingleInstanceMutexTests(unittest.TestCase):
    def test_import_does_not_remove_the_callers_test_paths(self) -> None:
        resolved_paths = {Path(entry or ".").resolve() for entry in sys.path}

        self.assertIn(Path(__file__).resolve().parent, resolved_paths)

    def test_existing_mutex_is_closed_and_rejected(self) -> None:
        close_handle = Mock()
        handle = _claim_single_instance_mutex(
            Mock(return_value=123),
            close_handle,
            Mock(return_value=183),
        )

        self.assertIsNone(handle)
        close_handle.assert_called_once_with(123)

    def test_new_mutex_handle_is_kept_for_the_process_lifetime(self) -> None:
        close_handle = Mock()
        handle = _claim_single_instance_mutex(
            Mock(return_value=456),
            close_handle,
            Mock(return_value=0),
        )

        self.assertEqual(handle, 456)
        close_handle.assert_not_called()


class EngineShutdownTests(unittest.TestCase):
    def test_storage_status_reports_media_drive_free_space(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            app = create_app(Path(temporary_directory), gpu_status)
            with TestClient(app) as client:
                status = client.get("/api/storage/status")

        self.assertEqual(status.status_code, 200)
        self.assertGreater(status.json()["free_disk_bytes"], 0)

    def test_unload_releases_cpu_model_references(self) -> None:
        engine = TTSEngine()
        engine._model = object()
        engine._device = "cpu"
        engine._model_dtype = "bfloat16"

        asyncio.run(engine.unload())

        self.assertIsNone(engine._model)
        self.assertIsNone(engine._device)
        self.assertIsNone(engine._model_dtype)

    def test_api_shutdown_calls_the_engine_cleanup_hook(self) -> None:
        class TrackingEngine:
            unload_calls = 0

            async def unload(self) -> None:
                self.unload_calls += 1

        engine = TrackingEngine()
        with tempfile.TemporaryDirectory() as temporary_directory:
            app = create_app(Path(temporary_directory), gpu_status, engine)
            with TestClient(app):
                pass

        self.assertEqual(engine.unload_calls, 1)
