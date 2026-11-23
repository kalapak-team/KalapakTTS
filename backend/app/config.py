from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List
from pathlib import Path
import math


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://postgres@localhost:5432/kalapaktts_db"
    SECRET_KEY: str = ""
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    AUDIO_FILES_DIR: str = str(Path(__file__).resolve().parents[1] / "audio_files")
    VOXCPM_MODEL_ID: str = "openbmb/VoxCPM2"
    DEVICE: str = "auto"
    MIN_FREE_VRAM_GIB: float = 4.5
    FORCE_CUDA: bool = False
    WARM_MODEL_ON_STARTUP: bool = True
    TTS_MAX_CONCURRENT_JOBS: int = 1
    TTS_SENTENCE_PAUSE_MS: int = 0
    TTS_PARAGRAPH_PAUSE_MS: int = 0
    TTS_CROSSFADE_MS: int = 40
    ADMIN_EMAIL: str = "admin@kalapaktts.com"
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = ""

    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:3001"]

    @field_validator("DEVICE")
    @classmethod
    def valid_device(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in ("auto", "cuda", "cpu"):
            raise ValueError("DEVICE must be auto, cuda, or cpu")
        return normalized

    @field_validator("TTS_MAX_CONCURRENT_JOBS")
    @classmethod
    def valid_tts_concurrency(cls, value: int) -> int:
        if value != 1:
            raise ValueError("TTS_MAX_CONCURRENT_JOBS must be 1 to protect GPU memory.")
        return value

    @field_validator("MIN_FREE_VRAM_GIB")
    @classmethod
    def valid_min_free_vram(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("MIN_FREE_VRAM_GIB must be a finite non-negative number.")
        return value

    @field_validator("TTS_SENTENCE_PAUSE_MS")
    @classmethod
    def valid_sentence_pause(cls, value: int) -> int:
        if not 0 <= value <= 250:
            raise ValueError("TTS_SENTENCE_PAUSE_MS must be between 0 and 250")
        return value

    @field_validator("TTS_PARAGRAPH_PAUSE_MS")
    @classmethod
    def valid_paragraph_pause(cls, value: int) -> int:
        if not 0 <= value <= 500:
            raise ValueError("TTS_PARAGRAPH_PAUSE_MS must be between 0 and 500")
        return value

    @field_validator("TTS_CROSSFADE_MS")
    @classmethod
    def valid_crossfade(cls, value: int) -> int:
        if not 30 <= value <= 60:
            raise ValueError("TTS_CROSSFADE_MS must be between 30 and 60")
        return value

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
