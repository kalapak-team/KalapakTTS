import shutil
import unittest
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from app.core.tts_engine import time_stretch_audio
from app.schemas.generation import GenerationCreate


class GenerationPreferenceSchemaTests(unittest.TestCase):
    def test_old_request_shape_remains_valid_with_custom_pace_defaults(self):
        request = GenerationCreate(text="សួស្តី")
        self.assertEqual(request.inference_timesteps, 10)
        self.assertEqual(request.speed, 1.0)
        self.assertEqual(request.pace_mode, "custom")
        self.assertIsNone(request.target_cps)
        self.assertFalse(request.pronunciation_dictionary)
        self.assertFalse(request.noise_reduction)
        self.assertIsNone(request.style)

    def test_slow_fast_and_style_values_are_accepted(self):
        slow = GenerationCreate(text="សួស្តី", speed=0.8, style=" news_anchor ", seed=123)
        fast = GenerationCreate(text="សួស្តី", speed=1.2, style="vlogger")
        denoised = GenerationCreate(text="សួស្តី", noise_reduction=True)
        self.assertEqual(slow.speed, 0.8)
        self.assertEqual(slow.style, "news_anchor")
        self.assertEqual(slow.seed, 123)
        self.assertEqual(fast.speed, 1.2)
        self.assertTrue(denoised.noise_reduction)
        self.assertEqual(
            GenerationCreate(text="សួស្តី", pace_mode="auto", target_cps=16).pace_mode,
            "auto",
        )

    def test_unseeded_generations_get_a_fresh_seed_and_explicit_seed_is_preserved(self):
        with patch("app.schemas.generation.secrets.randbits", side_effect=[101, 202]) as random_seed:
            first = GenerationCreate(text="សួស្តី")
            second = GenerationCreate(text="សួស្តី")
        self.assertEqual((first.seed, second.seed), (101, 202))
        self.assertEqual(random_seed.call_args_list[0].args, (32,))
        self.assertEqual(random_seed.call_args_list[1].args, (32,))
        self.assertEqual(GenerationCreate(text="សួស្តី", seed=303).seed, 303)

    def test_speed_outside_api_range_is_rejected(self):
        for speed in (0.49, 2.01, float("nan"), float("inf"), float("-inf")):
            with self.subTest(speed=speed), self.assertRaises(ValidationError):
                GenerationCreate(text="សួស្តី", speed=speed)

    def test_invalid_pace_mode_and_target_are_rejected(self):
        with self.assertRaises(ValidationError):
            GenerationCreate(text="សួស្តី", pace_mode="instant")
        for target_cps in (4.9, 40.1, float("nan")):
            with self.subTest(target_cps=target_cps), self.assertRaises(ValidationError):
                GenerationCreate(text="សួស្តី", target_cps=target_cps)
        for seed in (-1, 2**32):
            with self.subTest(seed=seed), self.assertRaises(ValidationError):
                GenerationCreate(text="សួស្តី", seed=seed)


class SpeedProcessingTests(unittest.TestCase):
    def test_time_stretch_scales_duration_and_preserves_pitch(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("FFmpeg is required for the time-stretch integration test.")
        sample_rate = 24000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        signal = np.sin(2 * np.pi * 440 * time).astype(np.float32)

        for speed in (0.8, 1.0, 1.2):
            with self.subTest(speed=speed):
                output = time_stretch_audio(signal, sample_rate, speed)
                self.assertAlmostEqual(
                    len(output),
                    len(signal) / speed,
                    delta=round(sample_rate * 0.005),
                )
                self.assertTrue(np.all(np.isfinite(output)))
                peak_frequency = (
                    np.argmax(np.abs(np.fft.rfft(output))) * sample_rate / len(output)
                )
                self.assertAlmostEqual(peak_frequency, 440, delta=4)
                if speed != 1.0:
                    self.assertLessEqual(float(np.max(np.abs(output))), 0.980001)

    def test_stretch_reports_missing_ffmpeg(self):
        signal = np.ones(24000, dtype=np.float32)
        with patch("app.core.ffmpeg_audio.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "FFmpeg"):
                time_stretch_audio(signal, 24000, 0.8)
