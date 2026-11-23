import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from app.config import settings
from app.core.tts_engine import TTSEngine


class DeviceSelectionTests(unittest.TestCase):
    def test_auto_selects_cpu_when_free_vram_is_below_threshold(self):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                mem_get_info=lambda: (4.9 * 1024**3, 6 * 1024**3),
            )
        )
        engine = TTSEngine()

        with (
            patch.object(settings, "DEVICE", "auto"),
            patch.object(settings, "FORCE_CUDA", False),
            patch.object(settings, "MIN_FREE_VRAM_GIB", 5.8),
        ):
            device = engine._resolve_device(torch)

        self.assertEqual(device, "cpu")
        self.assertIn("4.90 GiB VRAM is free", engine._fallback_reason)

    def test_force_cuda_bypasses_the_free_vram_threshold(self):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                mem_get_info=lambda: (4.9 * 1024**3, 6 * 1024**3),
            )
        )

        with (
            patch.object(settings, "DEVICE", "auto"),
            patch.object(settings, "FORCE_CUDA", True),
            patch.object(settings, "MIN_FREE_VRAM_GIB", 5.8),
        ):
            self.assertEqual(TTSEngine()._resolve_device(torch), "cuda")

    def test_explicit_cuda_reports_the_guard_failure(self):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                mem_get_info=lambda: (4.9 * 1024**3, 6 * 1024**3),
            )
        )

        with (
            patch.object(settings, "DEVICE", "cuda"),
            patch.object(settings, "FORCE_CUDA", False),
            patch.object(settings, "MIN_FREE_VRAM_GIB", 5.8),
            self.assertRaisesRegex(RuntimeError, "FORCE_CUDA=true"),
        ):
            TTSEngine()._resolve_device(torch)


class OomFallbackTests(unittest.TestCase):
    def test_cuda_oom_retries_the_chunk_on_cpu(self):
        import torch

        engine = TTSEngine()
        model = SimpleNamespace(tts_model=SimpleNamespace(sample_rate=24000))
        engine._model = model
        engine._device = "cuda"
        audio = np.ones(24000, dtype=np.float32)
        generate = Mock(side_effect=[RuntimeError("CUDA out of memory"), audio])

        def switch_to_cpu(torch_module, reason):
            engine._model = model
            engine._device = "cpu"
            engine._fallback_reason = reason

        with (
            patch.object(engine, "_generate_streaming_audio", new=generate),
            patch.object(engine, "_is_cuda_oom", return_value=True),
            patch.object(engine, "_fallback_to_cpu", side_effect=switch_to_cpu),
            patch.object(torch.cuda, "reset_peak_memory_stats"),
            patch.object(torch.cuda, "max_memory_allocated", return_value=123),
            patch.object(torch.cuda, "max_memory_reserved", return_value=456),
            patch.object(torch.cuda, "empty_cache"),
        ):
            sample_rate, result = engine._run_inference(
                "សួស្តី", "", 2.0, 4, None, 1.0, 123
            )

        self.assertEqual(sample_rate, 24000)
        np.testing.assert_array_equal(result, audio)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(engine._device, "cpu")
        self.assertIn("retrying this chunk", engine._fallback_reason)
        self.assertEqual(engine._peak_vram_bytes, 123)
        self.assertEqual(engine._peak_vram_reserved_bytes, 456)


if __name__ == "__main__":
    unittest.main()
