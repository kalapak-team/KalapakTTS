import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import numpy as np
import soundfile as sf

from app.api.tts import _cache_path, _generate_and_save, tts_engine
from app.config import settings
from app.schemas.generation import GenerationCreate


class SpeechCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_request_cache_is_written_for_device_used_after_fallback(self):
        payload = GenerationCreate(text="Fallback cache sample.", seed=12345)
        audio = np.ones(24000, dtype=np.float32)
        user_id = uuid4()
        sentinel = object()

        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            cache_devices = []

            def cache_path_for_device(key):
                cache_devices.append(key["engine_device"])
                return cache_dir / f"{key['engine_device']}.wav"

            cache_path = Mock(side_effect=cache_path_for_device)
            with (
                patch("app.api.tts.CACHE_DIR", cache_dir),
                patch(
                    "app.api.tts.tts_engine.cache_device_key",
                    side_effect=["cuda", "cpu"],
                ),
                patch("app.api.tts._cache_path", new=cache_path),
                patch(
                    "app.api.tts._synthesize_request",
                    new=AsyncMock(return_value=(24000, audio)),
                ),
                patch("app.api.tts._save_generation", new=AsyncMock(return_value=sentinel)),
                patch("app.api.tts._write_audio_cache") as write_cache,
            ):
                result = await _generate_and_save(
                    payload,
                    AsyncMock(),
                    user_id,
                    None,
                    None,
                    "",
                )

        self.assertIs(result, sentinel)
        self.assertEqual(cache_devices, ["cuda", "cpu"])
        self.assertEqual(write_cache.call_args.args[0], cache_dir / "cpu.wav")

    async def test_previous_postprocessing_cache_is_not_reused(self):
        payload = GenerationCreate(text="Previously cached speech.", seed=12345)
        user_id = uuid4()
        voice_id = uuid4()
        control_instruction = ""
        namespace = f"{user_id}:{voice_id}"
        old_cache_key = {
            "namespace": namespace,
            "model_id": settings.VOXCPM_MODEL_ID,
            "postprocessing_version": 3,
            "reference": None,
            "payload": payload.model_dump(mode="json"),
            "control_instruction": control_instruction,
        }
        audio = np.ones(24000, dtype=np.float32)
        sentinel = object()

        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            with patch("app.api.tts.CACHE_DIR", cache_dir):
                old_cache = _cache_path(old_cache_key)
                sf.write(str(old_cache), audio, 24000)
                with (
                    patch(
                        "app.api.tts._synthesize_request",
                        new=AsyncMock(return_value=(24000, audio)),
                    ) as synthesize,
                    patch(
                        "app.api.tts._save_generation",
                        new=AsyncMock(return_value=sentinel),
                    ),
                ):
                    result = await _generate_and_save(
                        payload,
                        AsyncMock(),
                        user_id,
                        voice_id,
                        None,
                        control_instruction,
                    )

        self.assertIs(result, sentinel)
        synthesize.assert_awaited_once()

    async def test_full_request_cache_skips_model_generation_and_is_user_scoped(self):
        payload = GenerationCreate(text="Cached Khmer test.", seed=12345)
        user_id = uuid4()
        voice_id = uuid4()
        control_instruction = "A calm Khmer voice"
        namespace = f"{user_id}:{voice_id}"
        cache_key = {
            "namespace": namespace,
            "model_id": settings.VOXCPM_MODEL_ID,
            "postprocessing_version": 14,
            "engine_device": tts_engine.cache_device_key(),
            "force_cuda": settings.FORCE_CUDA,
            "reference": None,
            "payload": payload.model_dump(mode="json"),
            "control_instruction": control_instruction,
        }
        audio = np.ones(24000, dtype=np.float32)
        sentinel = object()

        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            cache_file = _cache_path(cache_key)
            user_cache = cache_dir / cache_file.name
            sf.write(str(user_cache), audio, 24000)
            other_user_key = dict(cache_key, namespace=f"{uuid4()}:{voice_id}")

            with (
                patch("app.api.tts.CACHE_DIR", cache_dir),
                patch("app.api.tts._synthesize_request", new=AsyncMock(side_effect=AssertionError("cache miss"))),
                patch("app.api.tts._save_generation", new=AsyncMock(return_value=sentinel)) as save_generation,
            ):
                result = await _generate_and_save(
                    payload,
                    AsyncMock(),
                    user_id,
                    voice_id,
                    None,
                    control_instruction,
                )
            self.assertTrue(user_cache.exists())

        self.assertIs(result, sentinel)
        self.assertNotEqual(cache_file, _cache_path(other_user_key))
        saved_audio = save_generation.await_args.args[-1]
        self.assertEqual(len(saved_audio), len(audio))
        self.assertEqual(save_generation.await_args.args[-2], 24000)
