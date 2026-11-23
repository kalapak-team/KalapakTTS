import uuid
from sqlalchemy import Boolean, Column, String, Float, Integer, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.database import Base


class Generation(Base):
    __tablename__ = "generations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String(100), nullable=False, default="Untitled", server_default="Untitled")
    text_input = Column(String(5000), nullable=False)
    control_instruction = Column(String(1000), nullable=True, default="")
    voice_id = Column(UUID(as_uuid=True), ForeignKey("voices.id", ondelete="SET NULL"), nullable=True)
    cfg_value = Column(Float, default=2.0, nullable=False)
    inference_timesteps = Column(Integer, default=10, nullable=False)
    speed = Column(Float, default=1.0, nullable=False)
    pace_mode = Column(String(20), default="custom", nullable=False)
    target_cps = Column(Float, nullable=True)
    pronunciation_dictionary = Column(Boolean, default=False, nullable=False)
    noise_reduction = Column(Boolean, default=False, nullable=False)
    style = Column(String(50), nullable=True)
    audio_file_path = Column(String(500), nullable=True)
    audio_duration = Column(Float, nullable=True)
    status = Column(String(20), default="completed", nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user = relationship("User", back_populates="generations")
    voice = relationship("Voice", back_populates="generations")
