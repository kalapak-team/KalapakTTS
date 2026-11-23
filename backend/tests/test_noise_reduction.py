import unittest
from unittest.mock import patch

import numpy as np

from app.core.noise_reduction import NOISE_REDUCTION_FILTER, NoiseReducer


class NoiseReducerTests(unittest.IsolatedAsyncioTestCase):
    async def test_processing_uses_gentle_filter_and_preserves_loudness(self):
        sample_rate = 48000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        source = (0.1 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
        filtered = source * 0.5

        with patch(
            "app.core.noise_reduction.apply_ffmpeg_audio_filter",
            return_value=filtered,
        ) as apply_filter:
            output = await NoiseReducer().process(source, sample_rate)

        apply_filter.assert_called_once_with(
            source,
            sample_rate,
            NOISE_REDUCTION_FILTER,
            "noise reduction",
        )
        self.assertEqual(len(output), len(source))
        self.assertTrue(np.all(np.isfinite(output)))
        self.assertLessEqual(float(np.max(np.abs(output))), 0.98)
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(output**2))),
            float(np.sqrt(np.mean(source**2))),
            places=3,
        )

    async def test_empty_or_multichannel_audio_is_rejected(self):
        reducer = NoiseReducer()
        with patch("app.core.ffmpeg_audio.shutil.which") as ffmpeg_lookup:
            with self.assertRaises(ValueError):
                await reducer.process(np.array([], dtype=np.float32), 48000)
            with self.assertRaises(ValueError):
                await reducer.process(np.ones((100, 2), dtype=np.float32), 48000)
        ffmpeg_lookup.assert_not_called()
