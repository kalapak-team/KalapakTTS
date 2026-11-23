from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from server.gpu import MEASURED_MODEL_PEAK_RESERVED_MIB


ROOT = Path(__file__).resolve().parents[1]


def _asset_root() -> Path:
    configured_root = os.environ.get("KALAPAK_TTS_MODELS")
    return Path(configured_root) if configured_root else ROOT / "models"


@dataclass(frozen=True)
class DesktopSettings:
    MODEL_ROOT: Path = field(default_factory=_asset_root)
    FFMPEG_PATH: Path = ROOT / "runtime" / "bin" / "ffmpeg.exe"
    DEVICE: str = "auto"
    FORCE_CUDA: bool = False
    MIN_FREE_VRAM_MIB: int | None = MEASURED_MODEL_PEAK_RESERVED_MIB
    TTS_MAX_CONCURRENT_JOBS: int = 1
    TTS_SENTENCE_PAUSE_MS: int = 0
    TTS_PARAGRAPH_PAUSE_MS: int = 0
    TTS_CROSSFADE_MS: int = 40

    @property
    def MODEL_DIR(self) -> Path:
        return self.MODEL_ROOT / "VoxCPM2"

    @property
    def VOICE_REFERENCE_DIR(self) -> Path:
        return self.MODEL_ROOT / "voices"

    @property
    def VOXCPM_MODEL_ID(self) -> str:
        return str(self.MODEL_DIR)


settings = DesktopSettings()
