import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database import Base


class Voice(Base):
    __tablename__ = "voices"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    name = Column(String(100), nullable=False)
    description = Column(String(500), nullable=True)
    control_instruction = Column(String(1000), nullable=True)
    reference_audio_path = Column(String(1000), nullable=True)
    language = Column(String(10), default="km", nullable=False)
    is_public = Column(Boolean, default=True, nullable=False)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    owner = relationship("User", back_populates="voices", foreign_keys=[owner_id])
    generations = relationship("Generation", back_populates="voice")

    @property
    def has_reference_audio(self) -> bool:
        return bool(self.reference_audio_path)
