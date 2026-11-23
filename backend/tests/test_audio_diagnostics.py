import unittest

import numpy as np

from app.core.audio_diagnostics import (
    audio_quality_metrics,
    shorten_internal_pauses,
    speech_timing_metrics,
    trim_audio_edges,
)


class AudioQualityMetricTests(unittest.TestCase):
    def test_tone_reports_duration_rms_and_centroid(self):
        sample_rate = 48000
        time = np.arange(sample_rate, dtype=np.float64) / sample_rate
        tone = np.sin(2 * np.pi * 1000 * time)

        metrics = audio_quality_metrics(tone, sample_rate)

        self.assertEqual(metrics["duration_seconds"], 1.0)
        self.assertAlmostEqual(metrics["rms_dbfs"], -3.01, places=1)
        self.assertAlmostEqual(metrics["spectral_centroid_hz"], 1000, delta=2)
        self.assertLess(metrics["high_frequency_energy_ratio"], 0.001)
        self.assertLess(metrics["spectral_flatness"], 0.001)

    def test_noise_has_high_flatness_and_high_frequency_energy(self):
        sample_rate = 48000
        noise = np.random.default_rng(42).standard_normal(sample_rate)

        metrics = audio_quality_metrics(noise, sample_rate)

        self.assertGreater(metrics["high_frequency_energy_ratio"], 0.80)
        self.assertGreater(metrics["spectral_flatness"], 0.4)

    def test_rejects_empty_or_non_finite_audio(self):
        with self.assertRaises(ValueError):
            audio_quality_metrics(np.array([], dtype=np.float32), 24000)
        with self.assertRaises(ValueError):
            audio_quality_metrics(np.array([0.0, np.nan]), 24000)

    def test_trims_chunk_edges_and_counts_long_internal_pauses(self):
        audio = np.concatenate(
            [
                np.zeros(100, dtype=np.float32),
                np.ones(300, dtype=np.float32) * 0.2,
                np.zeros(300, dtype=np.float32),
                np.ones(300, dtype=np.float32) * 0.2,
                np.zeros(100, dtype=np.float32),
            ]
        )

        trimmed = trim_audio_edges(audio, 1000, padding_ms=40)
        metrics = speech_timing_metrics(trimmed, 1000)

        self.assertLess(len(trimmed), len(audio))
        self.assertEqual(metrics["pause_count_250ms"], 1)
        self.assertAlmostEqual(metrics["pause_seconds_250ms"], 0.28, delta=0.05)

    def test_shortens_long_internal_pauses_without_removing_short_gaps(self):
        audio = np.concatenate(
            [
                np.ones(300, dtype=np.float32) * 0.2,
                np.zeros(500, dtype=np.float32),
                np.ones(300, dtype=np.float32) * 0.2,
            ]
        )

        shortened = shorten_internal_pauses(audio, 1000, max_pause_ms=160)
        metrics = speech_timing_metrics(shortened, 1000)

        self.assertLess(len(shortened), len(audio) - 300)
        self.assertEqual(metrics["pause_count_250ms"], 0)
        self.assertAlmostEqual(metrics["pause_seconds_250ms"], 0.0)

        unchanged = shorten_internal_pauses(audio, 1000, max_pause_ms=600)
        self.assertEqual(len(unchanged), len(audio))


if __name__ == "__main__":
    unittest.main()
