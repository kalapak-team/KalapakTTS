import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_

from app.database import get_db
from app.models.user import User
from app.models.voice import Voice
from app.schemas.voice import VoiceCreate, VoiceRead
from app.core.security import get_current_user, get_optional_current_user

router = APIRouter(prefix="/api/voices", tags=["voices"])


@router.get("/", response_model=list[VoiceRead])
async def list_voices(
    db: AsyncSession = Depends(get_db),
    current_user: User | None = Depends(get_optional_current_user),
):
    """Return public voices and, when authenticated, voices owned by the current user."""
    voice_filter = Voice.is_public.is_(True)
    if current_user is not None:
        voice_filter = or_(voice_filter, Voice.owner_id == current_user.id)
    result = await db.execute(
        select(Voice).where(voice_filter)
    )
    return result.scalars().all()


@router.post("/", response_model=VoiceRead, status_code=status.HTTP_201_CREATED)
async def create_voice(
    voice_in: VoiceCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    voice = Voice(
        name=voice_in.name,
        description=voice_in.description,
        control_instruction=voice_in.control_instruction,
        is_public=voice_in.is_public,
        owner_id=current_user.id,
    )
    db.add(voice)
    await db.commit()
    await db.refresh(voice)
    return voice


@router.delete("/{voice_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_voice(
    voice_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(select(Voice).where(Voice.id == voice_id))
    voice = result.scalar_one_or_none()
    if not voice:
        raise HTTPException(status_code=404, detail="Voice not found")
    if voice.owner_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Not authorized to delete this voice")
    await db.delete(voice)
    await db.commit()
