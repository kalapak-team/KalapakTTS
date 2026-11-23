import asyncio
import logging

import numpy as np
from server.ffmpeg_audio import apply_ffmpeg_audio_filter, match_audio_rms


NOISE_REDUCTION_FILTER = "afftdn=nr=3:nf=-55:tn=1"
logger = logging.getLogger(__name__)


class NoiseReducer:
    def _process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        signal = np.asarray(audio, dtype=np.float32)
        enhanced = apply_ffmpeg_audio_filter(
            signal,
            sample_rate,
            NOISE_REDUCTION_FILTER,
            "noise reduction",
        )
        enhanced = match_audio_rms(enhanced, signal)

        logger.info(
            "Audio noise reduction completed: sample_rate=%d input_seconds=%.2f output_seconds=%.2f",
            sample_rate,
            len(signal) / sample_rate,
            len(enhanced) / sample_rate,
        )
        return enhanced

    async def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return await asyncio.to_thread(self._process, audio, sample_rate)


noise_reducer = NoiseReducer()
