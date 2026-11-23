from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from uuid import UUID


class VoiceBase(BaseModel):
    name: str
    description: Optional[str] = None
    control_instruction: Optional[str] = None
    is_public: bool = True


class VoiceCreate(VoiceBase):
    pass


class VoiceRead(VoiceBase):
    id: UUID
    language: str
    owner_id: Optional[UUID] = None
    created_at: datetime
    has_reference_audio: bool = False

    model_config = {"from_attributes": True}
