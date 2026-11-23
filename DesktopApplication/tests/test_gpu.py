import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from server.gpu import MIB, read_gpu_status
from server.main import create_app


def _nvidia_result(free_mib: int) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=0,
        stdout=f"NVIDIA GeForce RTX 4050 Laptop GPU, {free_mib}, 6141\n",
        stderr="",
    )


def _torch_memory(free_mib: int) -> dict[str, object]:
    return {
        "available": True,
        "initialized_before": False,
        "initialized_after_first_read": True,
        "free_before": free_mib * MIB,
        "total_before": 6141 * MIB,
        "free_after": free_mib * MIB,
        "total_after": 6141 * MIB,
        "torch_context_init_delta_mib": 0,
        "first_read_error": None,
    }


class GpuStatusTests(unittest.TestCase):
    def test_nvidia_and_torch_memory_are_reconciled_before_allowing_gpu(self) -> None:
        with (
            patch("server.gpu.shutil.which", return_value="nvidia-smi"),
            patch("server.gpu.subprocess.run", return_value=_nvidia_result(5848)),
            patch("server.gpu._read_cuda_memory", return_value=_torch_memory(5080)),
            patch("server.gpu.MEASURED_MODEL_PEAK_RESERVED_MIB", 4500),
            patch("server.gpu.MEASURED_CUDA_CONTEXT_OVERHEAD_MIB", 768),
        ):
            status = read_gpu_status()

        self.assertEqual(status["nvidia_smi_free_mib"], 5848)
        self.assertEqual(status["free_mib"], 5080)
        self.assertEqual(status["context_overhead_mib"], 768)
        self.assertEqual(status["model_need_mib"], 5268)
        self.assertTrue(status["safe_to_load_model"])

    def test_unmeasured_model_need_is_a_soft_warning_with_try_anyway(self) -> None:
        with (
            patch("server.gpu.shutil.which", return_value="nvidia-smi"),
            patch("server.gpu.subprocess.run", return_value=_nvidia_result(5848)),
            patch("server.gpu._read_cuda_memory", return_value=_torch_memory(5080)),
            patch("server.gpu.MEASURED_MODEL_PEAK_RESERVED_MIB", None),
            patch("server.gpu.MEASURED_CUDA_CONTEXT_OVERHEAD_MIB", None),
            patch("server.gpu._observed_context_overhead_mib", None),
        ):
            status = read_gpu_status()

        self.assertFalse(status["safe_to_load_model"])
        self.assertIsNone(status["model_need_mib"])
        self.assertIn("Try GPU anyway", status["warning"])

    def test_guard_uses_nvidia_free_and_reports_torch_mismatch(self) -> None:
        with (
            patch("server.gpu.shutil.which", return_value="nvidia-smi"),
            patch("server.gpu.subprocess.run", return_value=_nvidia_result(5848)),
            patch("server.gpu._read_cuda_memory", return_value=_torch_memory(4200)),
            patch("server.gpu.MEASURED_MODEL_PEAK_RESERVED_MIB", 4500),
            patch("server.gpu.MEASURED_CUDA_CONTEXT_OVERHEAD_MIB", 768),
        ):
            status = read_gpu_status()

        self.assertTrue(status["safe_to_load_model"])
        self.assertIn("CUDA reports 4200 MiB", status["warning"])
        self.assertIn("Torch and nvidia-smi differ by 1648 MiB", status["warning"])

    def test_guard_blocks_when_nvidia_free_is_below_measured_need(self) -> None:
        with (
            patch("server.gpu.shutil.which", return_value="nvidia-smi"),
            patch("server.gpu.subprocess.run", return_value=_nvidia_result(5200)),
            patch("server.gpu._read_nvidia_smi_free_after_context", return_value=5200),
            patch("server.gpu._read_cuda_memory", return_value=_torch_memory(5080)),
            patch("server.gpu.MEASURED_MODEL_PEAK_RESERVED_MIB", 4500),
            patch("server.gpu.MEASURED_CUDA_CONTEXT_OVERHEAD_MIB", 768),
        ):
            status = read_gpu_status()

        self.assertFalse(status["safe_to_load_model"])
        self.assertIn("measured model need is 5268 MiB", status["warning"])

    def test_nvidia_torch_gap_is_not_counted_as_context_overhead(self) -> None:
        with (
            patch("server.gpu.shutil.which", return_value="nvidia-smi"),
            patch("server.gpu.subprocess.run", return_value=_nvidia_result(5848)),
            patch("server.gpu._read_nvidia_smi_free_after_context", return_value=5848),
            patch("server.gpu._read_cuda_memory", return_value=_torch_memory(5080)),
            patch("server.gpu.MEASURED_MODEL_PEAK_RESERVED_MIB", 4500),
            patch("server.gpu.MEASURED_CUDA_CONTEXT_OVERHEAD_MIB", None),
            patch("server.gpu._observed_context_overhead_mib", None),
        ):
            status = read_gpu_status()

        self.assertEqual(status["nvidia_torch_delta_mib"], 768)
        self.assertEqual(status["nvidia_context_init_delta_mib"], 0)
        self.assertEqual(status["context_overhead_mib"], 0)
        self.assertEqual(status["model_need_mib"], 4500)
        self.assertTrue(status["safe_to_load_model"])

    def test_nvidia_context_delta_is_used_as_context_overhead(self) -> None:
        with (
            patch("server.gpu.shutil.which", return_value="nvidia-smi"),
            patch("server.gpu.subprocess.run", return_value=_nvidia_result(5848)),
            patch("server.gpu._read_nvidia_smi_free_after_context", return_value=5828),
            patch("server.gpu._read_cuda_memory", return_value=_torch_memory(5080)),
            patch("server.gpu.MEASURED_MODEL_PEAK_RESERVED_MIB", 4500),
            patch("server.gpu.MEASURED_CUDA_CONTEXT_OVERHEAD_MIB", None),
            patch("server.gpu._observed_context_overhead_mib", None),
        ):
            status = read_gpu_status()

        self.assertEqual(status["nvidia_context_init_delta_mib"], 20)
        self.assertEqual(status["context_overhead_mib"], 20)
        self.assertEqual(status["model_need_mib"], 4520)

    def test_startup_gpu_check_is_returned_to_the_ui(self) -> None:
        calls: list[int] = []
        expected_status = {
            "available": True,
            "cuda_available": True,
            "gpu_name": "Test GPU",
            "free_mib": 5080,
            "total_mib": 6141,
            "nvidia_smi_free_mib": 5848,
            "torch_free_before_context_mib": 5080,
            "torch_free_after_context_mib": 5080,
            "torch_context_init_delta_mib": 0,
            "context_overhead_mib": 768,
            "model_peak_reserved_mib": None,
            "model_need_mib": None,
            "minimum_free_mib": None,
            "safe_to_load_model": False,
            "warning": "Try GPU anyway.",
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
