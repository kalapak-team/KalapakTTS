import hashlib
import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from server.gpu import MIN_FREE_VRAM_MIB
from server.main import create_app
from server.tts_service import format_subtitles
from server.tts_engine import TTSEngine


def available_gpu_status() -> dict[str, object]:
    return {
        "available": True,
        "cuda_available": True,
        "gpu_name": "Test GPU",
        "free_mib": 6141,
        "total_mib": 6141,
        "minimum_free_mib": MIN_FREE_VRAM_MIB,
        "safe_to_load_model": True,
        "warning": "Test GPU status.",
        "checked_at": "test",
    }


class FakeEngine:
    def __init__(self) -> None:
        self.last_load: tuple[str, bool] | None = None

    def status(self) -> dict[str, object]:
        return {
            "model_available": True,
            "model_error": None,
            "model_loaded": True,
            "model_loading": False,
            "model_device": "cuda",
        }

    async def load_model(self, device: str, allow_low_vram: bool = False) -> None:
        self.last_load = (device, allow_low_vram)

    async def generate_audio(self, **_kwargs) -> tuple[int, np.ndarray]:
        sample_rate = 16_000
        samples = np.arange(sample_rate, dtype=np.float32) / sample_rate
        audio = 0.2 * np.sin(2 * np.pi * 440 * samples)
        return sample_rate, audio.astype(np.float32)


class OomRetryFakeEngine(FakeEngine):
    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda"
        self.fallback_reason: str | None = None

    def status(self) -> dict[str, object]:
        return {
            **super().status(),
            "model_device": self.device,
            "fallback_reason": self.fallback_reason,
        }

    async def generate_audio(self, **kwargs) -> tuple[int, np.ndarray]:
        self.device = "cpu"
        self.fallback_reason = (
            "CUDA ran out of memory during speech generation; VoxCPM2 is retrying on CPU."
        )
        return await super().generate_audio(**kwargs)


class DesktopTtsTests(unittest.TestCase):
    def test_history_api_filters_and_manages_generation_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            application = create_app(
                root,
                available_gpu_status,
                FakeEngine(),
            )
            with TestClient(application) as client:
                store = client.app.state.local_store
                audio_path = store.audio_dir / "history-api.wav"
                audio_path.write_bytes(b"history audio")
                store.save_generation(
                    "history-api",
                    "Khmer history search",
                    "vanna",
                    "natural",
                    4,
                    title="History API test",
                    audio_path=str(audio_path),
                    audio_sha256="hash",
                    audio_duration=1.0,
                )

                filtered = client.get(
                    "/api/tts/generations",
                    params={
                        "search": "history",
                        "date_from": date.today().isoformat(),
                        "date_to": date.today().isoformat(),
                        "pace": "natural",
                        "steps": 4,
                    },
                )
                renamed = client.patch(
                    "/api/tts/generations/history-api",
                    json={"title": "Interview recording"},
                )
                favorite = client.put(
                    "/api/tts/generations/history-api/favorite",
                    json={"favorite": True},
                )
                favorite_list = client.get(
                    "/api/tts/generations",
                    params={"favorites_only": True},
                )
                duplicate = client.post("/api/tts/generations/history-api/duplicate")
                deleted = client.delete("/api/tts/generations/history-api")
                missing = client.get("/api/tts/generations/history-api/audio")
                cleanup = client.post(
                    "/api/storage/cleanup",
                    json={"before": (date.today() + timedelta(days=1)).isoformat()},
                )

            self.assertEqual(filtered.status_code, 200, filtered.text)
            self.assertEqual([item["id"] for item in filtered.json()["items"]], ["history-api"])
            self.assertFalse(filtered.json()["items"][0]["audio_missing"])
            self.assertEqual(renamed.status_code, 200, renamed.text)
            self.assertEqual(favorite.status_code, 200, favorite.text)
            self.assertEqual(favorite_list.json()["items"][0]["title"], "Interview recording")
            self.assertEqual(duplicate.status_code, 200, duplicate.text)
            self.assertEqual(deleted.status_code, 200, deleted.text)
            self.assertEqual(missing.status_code, 404)
            self.assertEqual(cleanup.status_code, 200, cleanup.text)

    def test_api_moves_media_folder_and_reports_storage_usage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            application = create_app(root / "profile", available_gpu_status, FakeEngine())
            new_folder = root / "usb-media"
            with TestClient(application) as client:
                store = client.app.state.local_store
                audio_path = store.audio_dir / "move-api.wav"
                audio_path.write_bytes(b"audio")
                store.save_generation(
                    "move-api",
                    "Move media",
                    "vanna",
                    "natural",
                    4,
                    audio_path=str(audio_path),
                    audio_sha256="hash",
                    audio_duration=1.0,
                )
                moved = client.post(
                    "/api/storage/media-folder",
                    json={"path": str(new_folder)},
                )
                status = client.get("/api/storage/status")

            self.assertEqual(moved.status_code, 200, moved.text)
            self.assertEqual(moved.json()["moved_files"], 1)
            self.assertEqual(status.json()["media_folder"], str(new_folder.resolve()))
            self.assertEqual(status.json()["audio_storage_bytes"], len(b"audio"))
            self.assertEqual(status.json()["generation_count"], 1)

    def test_gpu_precheck_requires_explicit_try_anyway_below_measured_need(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine = FakeEngine()
            gpu_status = {
                **available_gpu_status(),
                "safe_to_load_model": False,
                "warning": "Measured GPU memory need is above current free memory.",
            }
            application = create_app(
                Path(temporary_directory),
                lambda: gpu_status,
                engine,
            )
            with TestClient(application) as client:
                blocked = client.post("/api/engine/load", json={"device": "cuda"})
                forced = client.post(
                    "/api/engine/load",
                    json={"device": "cuda", "try_gpu_anyway": True},
                )

        self.assertEqual(blocked.status_code, 409)
        self.assertIn("Measured GPU memory need", blocked.json()["detail"])
        self.assertEqual(forced.status_code, 200, forced.text)
        self.assertEqual(engine.last_load, ("cuda", True))

    def test_sse_generation_persists_wav_and_exports_srt_and_vtt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            voice_dir = root / "voices"
            voice_dir.mkdir()
            (voice_dir / "vanna.wav").write_bytes(b"test reference")
            application = create_app(
                root / "profile",
                available_gpu_status,
                FakeEngine(),
            )
            patched_settings = SimpleNamespace(VOICE_REFERENCE_DIR=voice_dir)

            with (
                patch("server.tts_service.settings", patched_settings),
                TestClient(application) as client,
            ):
                response = client.post(
                    "/api/tts/generate",
                    json={
                        "text": "សួស្តី",
                        "voice": "vanna",
                        "pace": "natural",
                        "steps": 4,
                        "speed": 1.0,
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertIn("event: chunk", response.text)
                complete_event = next(
                    block
                    for block in response.text.split("\n\n")
                    if block.startswith("event: complete")
                )
                complete_payload = json.loads(
                    next(line[5:].strip() for line in complete_event.splitlines() if line.startswith("data:"))
                )

                audio_response = client.get(complete_payload["audio_url"])
                srt_response = client.get(complete_payload["srt_url"])
                vtt_response = client.get(complete_payload["vtt_url"])

            self.assertEqual(audio_response.status_code, 200)
            self.assertTrue(audio_response.content.startswith(b"RIFF"))
            self.assertEqual(srt_response.status_code, 200)
            self.assertIn("00:00:00,000 -->", srt_response.text)
            self.assertEqual(vtt_response.status_code, 200)
            self.assertTrue(vtt_response.text.startswith("WEBVTT\n\n"))
            self.assertEqual(
                complete_payload["audio_sha256"],
                hashlib.sha256(audio_response.content).hexdigest(),
            )

    def test_invalid_subtitle_format_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "format must be srt or vtt"):
            format_subtitles([], "txt")

    def test_cuda_oom_fallback_is_sent_as_a_visible_stream_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            voice_dir = root / "voices"
            voice_dir.mkdir()
            (voice_dir / "vanna.wav").write_bytes(b"test reference")
            application = create_app(
                root / "profile",
                available_gpu_status,
                OomRetryFakeEngine(),
            )
            patched_settings = SimpleNamespace(VOICE_REFERENCE_DIR=voice_dir)

            with (
                patch("server.tts_service.settings", patched_settings),
                TestClient(application) as client,
            ):
                response = client.post(
                    "/api/tts/generate",
                    json={
                        "text": "សួស្តី",
                        "voice": "vanna",
                        "pace": "natural",
                        "steps": 4,
                        "speed": 1.0,
                    },
                )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("event: fallback", response.text)
        self.assertIn("CUDA ran out of memory", response.text)


class TtsEngineFallbackTests(unittest.TestCase):
    def test_status_uses_nvidia_smi_free_memory_for_wddm_reporting(self) -> None:
        engine = TTSEngine()
        torch = SimpleNamespace(
            __version__="test",
            version=SimpleNamespace(cuda="test"),
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_name=lambda _index: "Test GPU",
                is_bf16_supported=lambda: True,
                memory_allocated=lambda: 123,
                memory_reserved=lambda: 456,
            ),
        )
        with (
            patch.dict("sys.modules", {"torch": torch}),
            patch(
                "server.tts_engine.model_asset_status",
                return_value={
                    "available": True,
                    "message": None,
                    "directory": "models",
                    "total_bytes": 100,
                },
            ),
            patch(
                "server.tts_engine.read_gpu_status",
                return_value={
                    "available": True,
                    "cuda_available": True,
                    "gpu_name": "Test GPU",
                    "free_mib": 5080,
                    "nvidia_smi_free_mib": 5848,
                    "total_mib": 6140,
                    "model_need_mib": 5366,
                },
            ),
        ):
            status = engine.status()

        self.assertEqual(status["vram_free_bytes"], 5848 * 1024 * 1024)
        self.assertEqual(status["vram_total_bytes"], 6140 * 1024 * 1024)
        self.assertEqual(status["vram_used_bytes"], (6140 - 5848) * 1024 * 1024)
        self.assertEqual(status["minimum_free_vram_mib"], 5366)

    def test_cuda_oom_retries_the_chunk_on_cpu_and_keeps_reason(self) -> None:
        engine = TTSEngine()
        engine._device = "cuda"
        engine._model = SimpleNamespace(tts_model=SimpleNamespace(sample_rate=24_000))
        cpu_model = SimpleNamespace(tts_model=SimpleNamespace(sample_rate=24_000))
        torch = SimpleNamespace(
            cuda=SimpleNamespace(
                OutOfMemoryError=RuntimeError,
                reset_peak_memory_stats=lambda: None,
                max_memory_allocated=lambda: 1024,
                max_memory_reserved=lambda: 2048,
                empty_cache=lambda: None,
            )
        )

        def switch_to_cpu(_torch, reason: str) -> None:
            engine._fallback_reason = reason
            engine._model = cpu_model
            engine._device = "cpu"

        audio = np.ones(24_000, dtype=np.float32)
        with (
            patch.dict("sys.modules", {"torch": torch}),
            patch.object(engine, "_fallback_to_cpu", side_effect=switch_to_cpu),
            patch.object(
                engine,
                "_generate_streaming_audio",
                side_effect=[RuntimeError("CUDA out of memory"), audio],
            ) as generate,
        ):
            sample_rate, result = engine._run_inference(
                "សួស្តី",
                "",
                2.0,
                4,
                None,
                1.0,
                123,
            )

        self.assertEqual(engine._device, "cpu")
        self.assertIn("switching to CPU", engine._fallback_reason)
        self.assertEqual(sample_rate, 24_000)
        np.testing.assert_array_equal(result, audio)
        self.assertEqual(generate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
