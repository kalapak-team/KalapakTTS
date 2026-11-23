import math
import secrets
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from uuid import UUID


class GenerationCreate(BaseModel):
    text: str
    voice_preset_id: Optional[UUID] = None
    control_instruction: Optional[str] = ""
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    speed: float = 1.0
    pace_mode: Literal["auto", "custom"] = "custom"
    target_cps: Optional[float] = None
    seed: Optional[int] = Field(default_factory=lambda: secrets.randbits(32))
    pronunciation_dictionary: bool = False
    noise_reduction: bool = False
    style: Optional[str] = None

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Text cannot be empty")
        if len(v) > 5000:
            raise ValueError("Text cannot exceed 5000 characters")
        return v

    @field_validator("cfg_value")
    @classmethod
    def cfg_value_range(cls, v: float) -> float:
        if v < 0.0 or v > 10.0:
            raise ValueError("cfg_value must be between 0.0 and 10.0")
        return v

    @field_validator("inference_timesteps")
    @classmethod
    def timesteps_range(cls, v: int) -> int:
        if v < 1 or v > 100:
            raise ValueError("inference_timesteps must be between 1 and 100")
        return v

    @field_validator("speed")
    @classmethod
    def speed_range(cls, v: float) -> float:
        if not math.isfinite(v) or not 0.5 <= v <= 2.0:
            raise ValueError("speed must be between 0.5 and 2.0")
        return v

    @field_validator("target_cps")
    @classmethod
    def target_cps_range(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (not math.isfinite(v) or not 5.0 <= v <= 40.0):
            raise ValueError("target_cps must be between 5.0 and 40.0")
        return v

    @field_validator("seed")
    @classmethod
    def seed_range(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not 0 <= v <= 2**32 - 1:
            raise ValueError("seed must be between 0 and 4294967295")
        return v

    @field_validator("style")
    @classmethod
    def clean_style(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if len(v) > 50:
            raise ValueError("style cannot exceed 50 characters")
        return v or None


class GenerationRead(BaseModel):
    id: UUID
    audio_url: str
    duration: Optional[float] = None
    noise_reduction: bool = False
    created_at: datetime

    model_config = {"from_attributes": True}


class GenerationHistory(BaseModel):
    id: UUID
    title: str = "Untitled"
    text_input: str
    control_instruction: Optional[str] = None
    voice_id: Optional[UUID] = None
    cfg_value: float
    inference_timesteps: int
    speed: float
    pace_mode: Literal["auto", "custom"]
    target_cps: Optional[float] = None
    pronunciation_dictionary: bool = False
    noise_reduction: bool = False
    style: Optional[str] = None
    audio_duration: Optional[float] = None
    status: str
    created_at: datetime
    audio_url: str

    model_config = {"from_attributes": True}


class GenerationTitleUpdate(BaseModel):
    title: str = Field(max_length=100)

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Title cannot be empty")
        return value


class GenerationJobRead(BaseModel):
    job_id: UUID
    status: Literal["queued", "running", "completed", "failed"]
    current_chunk: int
    total_chunks: int
    message: str
    queued_position: Optional[int] = None
    queue_wait_seconds: Optional[float] = None
    first_audio_seconds: Optional[float] = None
    generation_seconds: Optional[float] = None
    pace_retry_count: int = 0
    generation: Optional[GenerationRead] = None
    error: Optional[str] = None
