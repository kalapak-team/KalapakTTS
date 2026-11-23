import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from app.api.tts import _synthesize_request
from app.core.tts_chunking import (
    apply_pronunciation_dictionary,
    join_speech_chunks,
    normalize_chunk_pace,
    split_speech_text,
)
from app.schemas.generation import GenerationCreate


class SpeechChunkingTests(unittest.TestCase):
    def test_saved_user_sample_stays_on_whitespace_boundaries(self):
        fixture_path = Path(__file__).parent / "fixtures" / "khmer_long_sample.txt"
        text = fixture_path.read_text(encoding="utf-8").rstrip("\r\n")
        chunks = split_speech_text(text)

        self.assertEqual(len(text.splitlines()), 5)
        self.assertEqual(len(text), 1093)
        self.assertEqual(len(chunks), 6)
        self.assertTrue(0 < len(chunks[0].text) <= 90)
        self.assertTrue(all(len(chunk.text) <= 450 for chunk in chunks))
        self.assertEqual(chunks[-2].pause_after, "paragraph")
        source_words = [word for paragraph in text.splitlines() for word in paragraph.split()]
        chunk_words = [word for chunk in chunks for word in chunk.text.split()]
        self.assertEqual(chunk_words, source_words)

    def test_splits_khmer_english_punctuation_and_paragraph_boundaries(self):
        first = "word " * 24 + "."
        second = "item " * 24 + "?"
        third = "ពាក្យ " * 18 + "។"
        chunks = split_speech_text(f"{first} {second}\n{third}")

        self.assertLessEqual(len(chunks[0].text), 90)
        self.assertEqual(chunks[0].pause_after, "continuation")
        self.assertTrue(any(chunk.pause_after == "paragraph" for chunk in chunks))
        self.assertTrue(any(chunk.text.endswith("?") for chunk in chunks))
        self.assertTrue(chunks[-1].text.endswith("។"))

    def test_long_text_splits_at_whitespace_without_cutting_words(self):
        words = ["សួស្តី", "Full-Stack", "Portfolio", "Source", "Code"] * 80
        text = " ".join(words)
        chunks = split_speech_text(text)

        self.assertGreater(len(chunks), 5)
        self.assertEqual(
            [word for chunk in chunks for word in chunk.text.split()],
            words,
        )
        self.assertTrue(all(len(chunk.text) <= 450 for chunk in chunks))
        self.assertLessEqual(len(chunks[0].text), 90)

    def test_pronunciation_dictionary_is_optional_and_case_insensitive(self):
        source = "Full-Stack, Portfolio, and Source Code"
        self.assertEqual(apply_pronunciation_dictionary(source, False), source)
        pronounced = apply_pronunciation_dictionary(source, True)
        self.assertEqual(pronounced, "ហ្វុលស្តាក់, ផតហ្វូលីយ៉ូ, and សូសកូដ")


class PaceNormalizationTests(unittest.TestCase):
    def test_absolute_pace_stretch_is_limited_to_fifteen_percent(self):
        sample_rate = 24000
        time = np.arange(sample_rate * 10, dtype=np.float32) / sample_rate
        signal = np.sin(2 * np.pi * 440 * time).astype(np.float32)

        slow, slow_cps, slow_rate, _ = normalize_chunk_pace(
            signal, sample_rate, 160, 14, "custom", 0.8
        )
        fast, fast_cps, fast_rate, _ = normalize_chunk_pace(
            signal, sample_rate, 160, 18.2, "custom", 1.2
        )

        self.assertAlmostEqual(slow_rate, 0.875, places=2)
        self.assertAlmostEqual(fast_rate, 1.14, places=2)
        self.assertAlmostEqual(slow_cps, 14.0, delta=0.2)
        self.assertAlmostEqual(fast_cps, 18.2, delta=0.2)
        self.assertGreater(len(slow), len(signal))
        self.assertLess(len(fast), len(signal))

    def test_extreme_pace_correction_is_skipped_instead_of_overstretching(self):
        sample_rate = 24000
        time = np.arange(sample_rate * 10, dtype=np.float32) / sample_rate
        signal = np.sin(2 * np.pi * 440 * time).astype(np.float32)

        normalized, cps, rate, skipped = normalize_chunk_pace(
            signal, sample_rate, 230, 14, "custom", 1.0
        )

        self.assertEqual(rate, 1.0)
        self.assertTrue(skipped)
        self.assertAlmostEqual(cps, 23.0, delta=0.1)
        np.testing.assert_array_equal(normalized, signal)

    def test_natural_pace_slows_a_chunk_slightly_above_the_old_limit(self):
        sample_rate = 24000
        time = np.arange(sample_rate * 10, dtype=np.float32) / sample_rate
        signal = np.sin(2 * np.pi * 440 * time).astype(np.float32)

        normalized, cps, rate, skipped = normalize_chunk_pace(
            signal, sample_rate, 173, 14, "custom", 1.0
        )

        self.assertAlmostEqual(rate, 14 / 17.3, places=2)
        self.assertFalse(skipped)
        self.assertAlmostEqual(cps, 14.0, delta=0.2)
        self.assertGreater(len(normalized), len(signal))

    def test_out_of_range_pace_correction_is_skipped(self):
        sample_rate = 24000
        signal = np.sin(
            2 * np.pi * 440 * np.arange(sample_rate * 10, dtype=np.float32) / sample_rate
        ).astype(np.float32)

        corrected, cps, rate, skipped = normalize_chunk_pace(
            signal, sample_rate, 300, 14, "auto", 1.0
        )

        self.assertIs(corrected, signal)
        self.assertAlmostEqual(cps, 30.0, delta=0.1)
        self.assertEqual(rate, 1.0)
        self.assertTrue(skipped)

    def test_pace_correction_is_skipped_above_maximum_stretch(self):
        sample_rate = 24000
        signal = np.sin(
            2 * np.pi * 440 * np.arange(sample_rate * 10, dtype=np.float32) / sample_rate
        ).astype(np.float32)

        corrected, cps, rate, skipped = normalize_chunk_pace(
            signal, sample_rate, 100, 16, "auto", 1.0
        )

        self.assertIs(corrected, signal)
        self.assertAlmostEqual(cps, 10.0, delta=0.1)
        self.assertEqual(rate, 1.0)
        self.assertTrue(skipped)

    def test_absolute_pace_does_not_stretch_within_twelve_percent(self):
        sample_rate = 24000
        time = np.arange(sample_rate * 10, dtype=np.float32) / sample_rate
        signal = np.sin(2 * np.pi * 440 * time).astype(np.float32)

        _, auto_cps, auto_rate, _ = normalize_chunk_pace(
            signal, sample_rate, 170, 16, "auto", 1.0
        )
        _, custom_cps, custom_rate, _ = normalize_chunk_pace(
            signal, sample_rate, 170, 16, "custom", 1.0
        )

        self.assertEqual(auto_rate, 1.0)
        self.assertAlmostEqual(auto_cps, 17.0, delta=0.1)
        self.assertEqual(custom_rate, 1.0)
        self.assertAlmostEqual(custom_cps, 17.0, delta=0.1)

    def test_chunk_join_uses_natural_pauses_and_short_crossfades(self):
        sample_rate = 1000
        first = np.ones(1000, dtype=np.float32)
        second = np.ones(1000, dtype=np.float32)

        sentence_audio = join_speech_chunks([first, second], ["sentence", "sentence"], sample_rate)
        paragraph_audio = join_speech_chunks([first, second], ["paragraph", "paragraph"], sample_rate)
        continuation_audio = join_speech_chunks(
            [first, second], ["continuation", "continuation"], sample_rate
        )
        with patch("app.core.tts_chunking.PARAGRAPH_PAUSE_MS", 160):
            explicit_paragraph_audio = join_speech_chunks(
                [first, second], ["paragraph", "paragraph"], sample_rate
            )
        with patch("app.core.tts_chunking.SENTENCE_PAUSE_MS", 120):
            explicit_sentence_audio = join_speech_chunks(
                [first, second], ["sentence", "sentence"], sample_rate
            )

        self.assertEqual(len(sentence_audio), 1960)
        self.assertEqual(len(explicit_sentence_audio), 2120)
        self.assertEqual(len(paragraph_audio), 1960)
        self.assertEqual(len(explicit_paragraph_audio), 2160)
        self.assertEqual(len(continuation_audio), 1960)
        self.assertTrue(np.all(np.isfinite(continuation_audio)))


class ChunkedRequestSynthesisTests(unittest.IsolatedAsyncioTestCase):
    async def test_chunk_audio_cache_uses_device_after_oom_fallback(self):
        payload = GenerationCreate(text="A" * 14, target_cps=14)
        audio = np.ones(24000, dtype=np.float32)
        engine = AsyncMock(return_value=(24000, audio))
        cache_device = Mock(side_effect=["cuda", "cpu"])
        cache_devices = []

        def cache_path_for_device(key):
            cache_devices.append(key["engine_device"])
            return Path(f"{key['engine_device']}.wav")

        cache_path = Mock(side_effect=cache_path_for_device)

        with (
            patch("app.api.tts.tts_engine.generate_audio", new=engine),
            patch("app.api.tts.tts_engine.cache_device_key", new=cache_device),
            patch("app.api.tts._cache_path", new=cache_path),
            patch("app.api.tts._write_audio_cache") as write_cache,
            patch(
                "app.api.tts.normalize_chunk_pace",
                side_effect=lambda audio, **_: (audio, 14.0, 1.0, False),
            ),
        ):
            await _synthesize_request(
                payload,
                None,
                "",
                cache_namespace="user:voice",
            )

        self.assertEqual(cache_devices, ["cuda", "cpu"])
        self.assertEqual(write_cache.call_args.args[0], Path("cpu.wav"))

    async def test_short_text_uses_absolute_pace(self):
        payload = GenerationCreate(text="A" * 14, target_cps=14)
        audio = np.ones(24000, dtype=np.float32)
        engine = AsyncMock(return_value=(24000, audio))
        streamed = []

        with (
            patch("app.api.tts.tts_engine.generate_audio", new=engine),
            patch("app.api.tts.noise_reducer.process", new=AsyncMock()) as denoiser,
            patch(
                "app.api.tts.normalize_chunk_pace",
                side_effect=lambda audio, **_: (audio, 14.0, 1.0, False),
            ) as normalizer,
            patch("app.api.tts.trim_audio_edges", return_value=audio) as trim_audio,
        ):
            sample_rate, output = await _synthesize_request(
                payload,
                None,
                "",
                on_audio=lambda *args: streamed.append(args),
            )

        self.assertEqual(sample_rate, 24000)
        self.assertEqual(len(output), len(audio))
        self.assertEqual(engine.await_count, 1)
        normalizer.assert_called_once()
        self.assertEqual(normalizer.call_args.kwargs["target_cps"], 14.0)
        trim_audio.assert_called_once_with(audio, 24000, padding_ms=60)
        self.assertEqual(len(streamed), 1)
        denoiser.assert_not_awaited()

    async def test_noise_reduction_runs_after_speed_and_streams_cleaned_audio(self):
        payload = GenerationCreate(
            text="A" * 14, speed=0.9, target_cps=14, noise_reduction=True
        )
        raw_audio = np.ones(24000, dtype=np.float32)
        paced_audio = np.full(24000, 0.5, dtype=np.float32)
        cleaned_audio = np.full(24000, 0.25, dtype=np.float32)
        joined_audio = np.full(24000, 0.75, dtype=np.float32)
        engine = AsyncMock(return_value=(24000, raw_audio))
        denoiser = AsyncMock(return_value=cleaned_audio)
        streamed = []

        with (
            patch("app.api.tts.tts_engine.generate_audio", new=engine),
            patch(
                "app.api.tts.normalize_chunk_pace",
                return_value=(raw_audio, 14.0, 1.0, False),
            ),
            patch("app.api.tts.time_stretch_audio", return_value=paced_audio) as stretcher,
            patch("app.api.tts.join_speech_chunks", return_value=joined_audio),
            patch("app.api.tts.noise_reducer.process", new=denoiser),
        ):
            _, output = await _synthesize_request(
                payload,
                None,
                "",
                on_audio=lambda *args: streamed.append(args),
            )

        self.assertEqual(stretcher.call_count, 2)
        np.testing.assert_array_equal(stretcher.call_args_list[0].args[0], raw_audio)
        np.testing.assert_array_equal(stretcher.call_args_list[1].args[0], joined_audio)
        self.assertEqual(denoiser.await_count, 2)
        np.testing.assert_array_equal(denoiser.call_args_list[0].args[0], paced_audio)
        np.testing.assert_array_equal(denoiser.call_args_list[1].args[0], paced_audio)
        np.testing.assert_array_equal(output, cleaned_audio)
        np.testing.assert_array_equal(streamed[0][3], cleaned_audio)

    async def test_long_text_streams_each_chunk_as_soon_as_it_is_generated(self):
        payload = GenerationCreate(
            text="សួស្តី portfolio " * 40, seed=12345, target_cps=14
        )
        expected_chunks = split_speech_text(payload.text)

        def fake_generate(text, **_):
            duration = len(text) / 14
            return 24000, np.ones(round(duration * 24000), dtype=np.float32)

        engine = AsyncMock(side_effect=fake_generate)
        progress = []
        streamed = []

        with (
            patch("app.api.tts.tts_engine.generate_audio", new=engine),
            patch(
                "app.api.tts.normalize_chunk_pace",
                side_effect=lambda audio, **_: (audio, 16.0, 1.0, False),
            ),
        ):
            await _synthesize_request(
                payload,
                None,
                "",
                lambda current, total, message: progress.append((current, total)),
                lambda current, total, *_: streamed.append((current, total)),
            )

        self.assertEqual(engine.await_count, len(expected_chunks))
        self.assertEqual(
            [call.kwargs["text"] for call in engine.await_args_list],
            [chunk.text for chunk in expected_chunks],
        )
        self.assertEqual(
            progress,
            [(0, len(expected_chunks))]
            + [(index, len(expected_chunks)) for index in range(1, len(expected_chunks) + 1)],
        )
        self.assertEqual(
            streamed,
            [(index, len(expected_chunks)) for index in range(1, len(expected_chunks) + 1)],
        )

    async def test_speed_is_applied_once_after_chunk_joining(self):
        payload = GenerationCreate(
            text="A short sample.",
            speed=0.8,
            pace_mode="custom",
            target_cps=14,
            seed=12345,
        )
        chunk_audio = np.ones(24000, dtype=np.float32)
        joined_audio = np.full(24000, 0.75, dtype=np.float32)
        stretched_audio = np.ones(37500, dtype=np.float32)
        engine = AsyncMock(return_value=(24000, chunk_audio))
        normalizer = Mock(return_value=(chunk_audio, 16.0, 1.0, False))

        with (
            patch("app.api.tts.tts_engine.generate_audio", new=engine),
            patch("app.api.tts.normalize_chunk_pace", new=normalizer),
            patch("app.api.tts.time_stretch_audio", return_value=stretched_audio) as stretcher,
            patch("app.api.tts.join_speech_chunks", return_value=joined_audio),
        ):
            sample_rate, output = await _synthesize_request(
                payload,
                None,
                "",
            )

        self.assertEqual(sample_rate, 24000)
        self.assertIs(output, stretched_audio)
        self.assertEqual(normalizer.call_args.kwargs["speed"], 1.0)
        stretcher.assert_called_once()
        np.testing.assert_array_equal(stretcher.call_args.args[0], joined_audio)
        self.assertEqual(stretcher.call_args.args[1:], (24000, 0.8))

    async def test_chunked_generation_preserves_seed_voice_settings_and_pronunciation(self):
        payload = GenerationCreate(
            text="Full-Stack Portfolio Source Code " * 18,
            seed=12345,
            pronunciation_dictionary=True,
            inference_timesteps=4,
            target_cps=14,
        )
        expected_chunks = split_speech_text(payload.text)
        progress = []
        raw_chunks = []

        def fake_generate(text, **_):
            duration = len(text) / 14
            return 24000, np.ones(round(duration * 24000), dtype=np.float32)

        engine = AsyncMock(side_effect=fake_generate)

        with (
            patch("app.api.tts.tts_engine.generate_audio", new=engine),
            patch(
                "app.api.tts.normalize_chunk_pace",
                side_effect=lambda audio, **_: (audio, 16.0, 1.0, False),
            ),
        ):
            sample_rate, output = await _synthesize_request(
                payload,
                "/voices/vanna.wav",
                "Same reference voice",
                lambda current, total, message: progress.append((current, total, message)),
                on_raw_chunk=lambda index, total, chunk, rate, audio: raw_chunks.append(
                    (index, total, chunk.text, rate, audio.copy())
                ),
            )

        self.assertEqual(sample_rate, 24000)
        self.assertGreater(len(output), 0)
        self.assertEqual(engine.await_count, len(expected_chunks))
        self.assertEqual(
            [call.kwargs["seed"] for call in engine.await_args_list],
            [12345 + index * 2 for index in range(len(expected_chunks))],
        )
        self.assertTrue(
            all(call.kwargs["reference_wav_path"] == "/voices/vanna.wav" for call in engine.await_args_list)
        )
        self.assertTrue(
            all(call.kwargs["inference_timesteps"] == 4 for call in engine.await_args_list)
        )
        self.assertTrue(
            all("ហ្វុលស្តាក់" in call.kwargs["text"] for call in engine.await_args_list)
        )
        self.assertEqual(progress[0], (0, len(expected_chunks), f"Preparing {len(expected_chunks)} speech chunks"))
        self.assertEqual(len(raw_chunks), len(expected_chunks))
        self.assertEqual(
            [(item[0], item[1]) for item in raw_chunks],
            [(index, len(expected_chunks)) for index in range(1, len(expected_chunks) + 1)],
        )
        self.assertEqual(
            [item[2] for item in raw_chunks],
            [chunk.text for chunk in expected_chunks],
        )
        self.assertTrue(
            all(item[3] == 24000 and len(item[4]) > 0 for item in raw_chunks)
        )

    async def test_out_of_range_pace_retries_once_with_an_alternate_seed(self):
        payload = GenerationCreate(text="A " * 40, seed=12345, target_cps=14)
        text_characters = len(payload.text)
        too_fast_audio = np.ones(24000 * 2, dtype=np.float32)
        target_pace_audio = np.ones(round(24000 * text_characters / 14), dtype=np.float32)
        engine = AsyncMock(side_effect=[
            (24000, too_fast_audio),
            (24000, target_pace_audio),
        ])
        progress = []
        retries = []
        normalizer = Mock(side_effect=lambda audio, **_: (audio, 14.0, 1.0, False))

        with (
            patch("app.api.tts.tts_engine.generate_audio", new=engine),
            patch("app.api.tts.normalize_chunk_pace", new=normalizer),
        ):
            await _synthesize_request(
                payload,
                "/voices/vanna.wav",
                "Same reference voice",
                on_progress=lambda current, total, message: progress.append(message),
                on_pace_retry=retries.append,
            )

        self.assertEqual(
            [call.kwargs["seed"] for call in engine.await_args_list],
            [12345, 12346],
        )
        self.assertTrue(
            all(call.kwargs["reference_wav_path"] == "/voices/vanna.wav" for call in engine.await_args_list)
        )
        self.assertTrue(
            all(call.kwargs["control_instruction"] == "Same reference voice" for call in engine.await_args_list)
        )
        self.assertEqual(engine.await_count, 2)
        self.assertEqual(retries, [1])
        self.assertEqual(progress[0], "Preparing 1 speech chunks")
        self.assertTrue(any("Generating chunk 1 of 1" in message for message in progress))
        self.assertEqual(
            len(normalizer.call_args.kwargs["audio"]),
            len(target_pace_audio),
        )


if __name__ == "__main__":
    unittest.main()
