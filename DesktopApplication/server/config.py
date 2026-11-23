from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from server.gpu import MEASURED_MODEL_PEAK_RESERVED_MIB


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DesktopSettings:
    MODEL_DIR: Path = ROOT / "models" / "VoxCPM2"
    VOICE_REFERENCE_DIR: Path = ROOT / "models" / "voices"
    FFMPEG_PATH: Path = ROOT / "runtime" / "bin" / "ffmpeg.exe"
    DEVICE: str = "auto"
    FORCE_CUDA: bool = False
    MIN_FREE_VRAM_MIB: int | None = MEASURED_MODEL_PEAK_RESERVED_MIB
    TTS_MAX_CONCURRENT_JOBS: int = 1
    TTS_SENTENCE_PAUSE_MS: int = 0
    TTS_PARAGRAPH_PAUSE_MS: int = 0
    TTS_CROSSFADE_MS: int = 40

    @property
    def VOXCPM_MODEL_ID(self) -> str:
        return str(self.MODEL_DIR)


settings = DesktopSettings()
