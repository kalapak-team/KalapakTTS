import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from server.gpu import MIN_FREE_VRAM_MIB, read_gpu_status
from server.main import create_app


class GpuStatusTests(unittest.TestCase):
    def test_low_free_vram_blocks_model_startup_with_actionable_warning(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=0,
            stdout="NVIDIA GeForce RTX 4050 Laptop GPU, 1016, 6141\n",
            stderr="",
        )
        with (
            patch("server.gpu.shutil.which", return_value="nvidia-smi"),
            patch("server.gpu.subprocess.run", return_value=completed),
        ):
            status = read_gpu_status()

        self.assertEqual(MIN_FREE_VRAM_MIB, 1536)
        self.assertEqual(status["free_mib"], 1016)
        self.assertFalse(status["safe_to_load_model"])
        self.assertIn("Stop the web TTS server/model", status["warning"])

    def test_startup_gpu_check_is_returned_to_the_ui(self) -> None:
        calls: list[int] = []
        expected_status = {
            "available": True,
            "gpu_name": "Test GPU",
            "free_mib": 1016,
            "total_mib": 6141,
            "minimum_free_mib": MIN_FREE_VRAM_MIB,
            "safe_to_load_model": False,
            "warning": "Stop the web TTS server/model.",
            "checked_at": "test",
        }

        def gpu_reader() -> dict[str, object]:
            calls.append(1)
            return expected_status

        with tempfile.TemporaryDirectory() as temporary_directory:
            application = create_app(Path(temporary_directory), gpu_reader)
            with TestClient(application) as client:
                response = client.get("/api/system/gpu")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["startup"], expected_status)
        self.assertEqual(response.json()["current"], expected_status)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
