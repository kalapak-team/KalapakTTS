import os
import uuid
import logging
import asyncio
import hashlib
import json
import time
import io
import base64
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from app.database import AsyncSessionLocal, get_db
from app.models.user import User
from app.models.generation import Generation
from app.models.voice import Voice
from app.schemas.generation import (
    GenerationCreate,
    GenerationHistory,
    GenerationJobRead,
    GenerationRead,
    GenerationTitleUpdate,
)
from app.core.security import get_current_user
from app.core.tts_engine import time_stretch_audio, tts_engine
from app.core.audio_diagnostics import (
    audio_quality_metrics,
    shorten_internal_pauses,
    speech_timing_metrics,
    trim_audio_edges,
)
from app.core.noise_reduction import noise_reducer
from app.core.tts_chunking import (
    DEFAULT_TARGET_CPS,
    MAX_CHUNK_STRETCH_RATE,
    MAX_PACE_RETRIES,
    MAX_STRETCH_RATE,
    MIN_CHUNK_STRETCH_RATE,
    MIN_STRETCH_RATE,
    SpeechChunk,
    apply_pronunciation_dictionary,
    join_speech_chunks,
    normalize_chunk_pace,
    split_speech_text,
)
from app.core.tts_jobs import GenerationJob, create_job, generation_slots, get_job
from app.config import settings

router = APIRouter(prefix="/api/tts", tags=["tts"])
logger = logging.getLogger(__name__)

AUDIO_DIR = Path(settings.AUDIO_FILES_DIR)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = AUDIO_DIR / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _reference_identity(reference_wav_path: Optional[str]) -> Optional[dict[str, object]]:
    if not reference_wav_path:
        return None
    return {"path": str(Path(reference_wav_path))}


def _cache_path(key_data: dict[str, object]) -> Path:
    encoded = json.dumps(key_data, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return CACHE_DIR / f"{digest}.wav"


def _write_audio_cache(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    temporary_cache = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp.wav")
    try:
        sf.write(str(temporary_cache), audio, sample_rate)
        temporary_cache.replace(path)
    except (OSError, RuntimeError):
        logger.exception("Could not write optional TTS cache file %s.", path)
        temporary_cache.unlink(missing_ok=True)


@router.get("/status")
async def get_tts_status():
    return tts_engine.status()


async def _resolve_voice(
    payload: GenerationCreate,
    db: AsyncSession,
) -> tuple[Optional[uuid.UUID], Optional[str], str]:
    control_instruction = payload.control_instruction or ""
    voice_id: Optional[uuid.UUID] = None
    reference_wav_path: Optional[str] = None

    if payload.voice_preset_id:
        result = await db.execute(
            select(Voice).where(Voice.id == payload.voice_preset_id)
        )
        voice = result.scalar_one_or_none()
        if not voice:
            raise HTTPException(status_code=404, detail="Voice preset not found")
        voice_id = voice.id
        # Voice's control_instruction takes priority if the caller didn't supply one
        if (
            not payload.control_instruction
            and isinstance(voice.control_instruction, str)
            and voice.control_instruction
        ):
            control_instruction = voice.control_instruction
        if voice.reference_audio_path:
            reference_path = Path(voice.reference_audio_path)
            if not reference_path.is_file():
                logger.error("Reference audio for voice %s is missing.", voice.id)
                raise HTTPException(
                    status_code=500,
                    detail="The selected voice reference audio is unavailable.",
                )
            reference_wav_path = str(reference_path)

    return voice_id, reference_wav_path, control_instruction


ProgressCallback = Callable[[int, int, str], None]
AudioCallback = Callable[[int, int, int, np.ndarray, str], None]
RawAudioCallback = Callable[[int, int, SpeechChunk, int, np.ndarray], None]
PaceRetryCallback = Callable[[int], None]


async def _synthesize_request(
    payload: GenerationCreate,
    reference_wav_path: Optional[str],
    control_instruction: str,
    on_progress: Optional[ProgressCallback] = None,
    on_audio: Optional[AudioCallback] = None,
    cache_namespace: Optional[str] = None,
    on_raw_chunk: Optional[RawAudioCallback] = None,
    on_pace_retry: Optional[PaceRetryCallback] = None,
) -> tuple[int, np.ndarray]:
    if payload.seed is None:
        seed_material = json.dumps(
            {
                "namespace": cache_namespace,
                "reference": _reference_identity(reference_wav_path),
                "control_instruction": control_instruction,
                "cfg_value": payload.cfg_value,
                "inference_timesteps": payload.inference_timesteps,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big")
    else:
        seed = payload.seed
    reference_identity = _reference_identity(reference_wav_path)
    effective_speed = min(MAX_STRETCH_RATE, max(MIN_STRETCH_RATE, payload.speed))

    chunks = split_speech_text(payload.text)
    if not chunks:
        raise ValueError("Text did not produce any speech chunks.")
    if on_progress:
        on_progress(0, len(chunks), f"Preparing {len(chunks)} speech chunks")

    target_cps = payload.target_cps or DEFAULT_TARGET_CPS
    paced_audio_chunks: list[np.ndarray] = []
    sample_rate: int | None = None
    retry_count = 0
    for chunk_index, speech_chunk in enumerate(chunks):
        current = chunk_index + 1
        if on_progress:
            on_progress(current, len(chunks), f"Generating chunk {current} of {len(chunks)}")
        synth_text = apply_pronunciation_dictionary(
            speech_chunk.text, payload.pronunciation_dictionary
        )
        best_candidate: tuple[float, int, np.ndarray, np.ndarray, int] | None = None
        for attempt in range(MAX_PACE_RETRIES + 1):
            chunk_seed = (
                seed + chunk_index * (MAX_PACE_RETRIES + 1) + attempt
            ) % (2**32)
            cache_key = {
                "namespace": cache_namespace,
                "model_id": settings.VOXCPM_MODEL_ID,
                "engine_device": tts_engine.cache_device_key(),
                "force_cuda": settings.FORCE_CUDA,
                "reference": reference_identity,
                "text": synth_text,
                "control_instruction": control_instruction,
                "cfg_value": payload.cfg_value,
                "inference_timesteps": payload.inference_timesteps,
                "seed": chunk_seed,
                "chunk_index": chunk_index,
            }
            audio_cache = _cache_path(cache_key) if cache_namespace else None
            if audio_cache and audio_cache.is_file():
                raw_audio, sample_rate = sf.read(
                    str(audio_cache),
                    dtype="float32",
                    always_2d=False,
                )
                logger.info("TTS chunk %d/%d served from cache.", current, len(chunks))
            else:
                inference_started = time.perf_counter()
                sample_rate, raw_audio = await tts_engine.generate_audio(
                    text=synth_text,
                    control_instruction=control_instruction,
                    cfg_value=payload.cfg_value,
                    inference_timesteps=payload.inference_timesteps,
                    reference_wav_path=reference_wav_path,
                    speed=1.0,
                    seed=chunk_seed,
                )
                logger.info(
                    "TTS chunk %d/%d inference finished: characters=%d seconds=%.2f seed=%d",
                    current,
                    len(chunks),
                    len(speech_chunk.text),
                    time.perf_counter() - inference_started,
                    chunk_seed,
                )
                if audio_cache:
                    cache_key["engine_device"] = tts_engine.cache_device_key()
                    audio_cache = _cache_path(cache_key)
                    _write_audio_cache(audio_cache, raw_audio, sample_rate)

            if len(raw_audio) == 0:
                raise RuntimeError(f"VoxCPM2 returned empty audio for chunk {current}.")
            trimmed_audio = shorten_internal_pauses(
                trim_audio_edges(raw_audio, sample_rate, padding_ms=60),
                sample_rate,
            )
            speech_seconds = float(
                speech_timing_metrics(trimmed_audio, sample_rate)["speech_seconds"]
            )
            if speech_seconds <= 0:
                raise RuntimeError(
                    f"No speech activity was detected in generated chunk {current}."
                )
            measured_cps = len(speech_chunk.text) / speech_seconds
            deviation = abs(measured_cps - target_cps) / target_cps
            candidate = (deviation, sample_rate, raw_audio, trimmed_audio, chunk_seed)
            if best_candidate is None or deviation < best_candidate[0]:
                best_candidate = candidate

            requested_rate = target_cps / measured_cps
            can_stretch = (
                MIN_CHUNK_STRETCH_RATE
                <= requested_rate
                <= MAX_CHUNK_STRETCH_RATE
            )
            if deviation <= 0.12 or can_stretch or attempt >= MAX_PACE_RETRIES:
                break

            retry_count += 1
            if on_pace_retry:
                on_pace_retry(retry_count)
            logger.info(
                "Retrying chunk %d/%d with alternate seed: raw_cps=%.2f target_cps=%.2f "
                "stretch_rate=%.3f retry=%d/%d.",
                current,
                len(chunks),
                measured_cps,
                target_cps,
                requested_rate,
                attempt + 1,
                MAX_PACE_RETRIES,
            )

        if best_candidate is None:
            raise RuntimeError(f"No usable audio candidate was produced for chunk {current}.")
        _, sample_rate, raw_audio, trimmed_audio, selected_seed = best_candidate
        raw_metrics = audio_quality_metrics(raw_audio, sample_rate)
        logger.info(
            "TTS chunk %d/%d raw audio: characters=%d duration_seconds=%.2f "
            "rms_dbfs=%.2f high_frequency_energy_ratio=%.4f "
            "spectral_centroid_hz=%.1f spectral_flatness=%.4f",
            current,
            len(chunks),
            len(speech_chunk.text),
            raw_metrics["duration_seconds"],
            raw_metrics["rms_dbfs"],
            raw_metrics["high_frequency_energy_ratio"],
            raw_metrics["spectral_centroid_hz"],
            raw_metrics["spectral_flatness"],
        )
        if on_raw_chunk:
            on_raw_chunk(current, len(chunks), speech_chunk, sample_rate, raw_audio.copy())
        paced_audio, measured_cps, stretch_rate, stretch_skipped = normalize_chunk_pace(
            audio=trimmed_audio,
            sample_rate=sample_rate,
            character_count=len(speech_chunk.text),
            target_cps=target_cps,
            pace_mode=payload.pace_mode,
            speed=1.0,
        )
        logger.info(
            "TTS chunk %d/%d pace: cps=%.2f target_cps=%.2f stretch_rate=%.3f "
            "stretch_skipped=%s requested_speed=%.2f seed=%d.",
            current,
            len(chunks),
            measured_cps,
            target_cps,
            stretch_rate,
            stretch_skipped,
            payload.speed,
            selected_seed,
        )
        paced_audio_chunks.append(paced_audio)
        if on_audio:
            preview_audio = (
                time_stretch_audio(paced_audio, sample_rate, effective_speed)
                if effective_speed != 1.0
                else paced_audio
            )
            if payload.noise_reduction:
                preview_audio = await noise_reducer.process(preview_audio, sample_rate)
            on_audio(current, len(chunks), sample_rate, preview_audio, speech_chunk.pause_after)

    if sample_rate is None:
        raise RuntimeError("The speech engine did not provide an audio sample rate.")
    joined_audio = join_speech_chunks(
        paced_audio_chunks,
        [chunk.pause_after for chunk in chunks],
        sample_rate,
    )
    if effective_speed != 1.0:
        joined_audio = time_stretch_audio(joined_audio, sample_rate, effective_speed)
    if payload.noise_reduction:
        if on_progress:
            on_progress(len(chunks), len(chunks), "Reducing background noise")
        joined_audio = await noise_reducer.process(joined_audio, sample_rate)
    return sample_rate, joined_audio


async def _save_generation(
    db: AsyncSession,
    payload: GenerationCreate,
    current_user_id: uuid.UUID,
    voice_id: Optional[uuid.UUID],
    control_instruction: str,
    sample_rate: int,
    wav_np: np.ndarray,
) -> GenerationRead:
    duration = float(len(wav_np)) / float(sample_rate)
    gen_id = uuid.uuid4()
    wav_path = AUDIO_DIR / f"{gen_id}.wav"
    try:
        sf.write(str(wav_path), wav_np, sample_rate)
    except Exception as exc:
        logger.exception("Failed to save WAV file")
        wav_path.unlink(missing_ok=True)
        raise RuntimeError(f"Audio save error: {exc}") from exc

    generation = Generation(
        id=gen_id,
        user_id=current_user_id,
        text_input=payload.text,
        control_instruction=control_instruction,
        voice_id=voice_id,
        cfg_value=payload.cfg_value,
        inference_timesteps=payload.inference_timesteps,
        speed=payload.speed,
        pace_mode=payload.pace_mode,
        target_cps=payload.target_cps,
        pronunciation_dictionary=payload.pronunciation_dictionary,
        noise_reduction=payload.noise_reduction,
        style=payload.style,
        audio_file_path=str(wav_path),
        audio_duration=duration,
        status="completed",
    )
    db.add(generation)
    try:
        await db.commit()
        await db.refresh(generation)
    except Exception:
        await db.rollback()
        wav_path.unlink(missing_ok=True)
        raise

    return GenerationRead(
        id=generation.id,
        audio_url=f"/api/tts/audio/{generation.id}",
        duration=generation.audio_duration,
        noise_reduction=generation.noise_reduction,
        created_at=generation.created_at,
    )


async def _generate_and_save(
    payload: GenerationCreate,
    db: AsyncSession,
    current_user_id: uuid.UUID,
    voice_id: Optional[uuid.UUID],
    reference_wav_path: Optional[str],
    control_instruction: str,
    on_progress: Optional[ProgressCallback] = None,
    on_audio: Optional[AudioCallback] = None,
    on_pace_retry: Optional[PaceRetryCallback] = None,
) -> GenerationRead:
    cache_namespace = f"{current_user_id}:{voice_id or 'default'}"
    full_cache_key = {
        "namespace": cache_namespace,
        "model_id": settings.VOXCPM_MODEL_ID,
        "postprocessing_version": 14,
        "engine_device": tts_engine.cache_device_key(),
        "force_cuda": settings.FORCE_CUDA,
        "reference": _reference_identity(reference_wav_path),
        "payload": payload.model_dump(mode="json"),
        "control_instruction": control_instruction,
    }
    full_cache = _cache_path(full_cache_key)
    if full_cache.is_file():
        cached_audio, cached_sample_rate = sf.read(
            str(full_cache),
            dtype="float32",
            always_2d=False,
        )
        logger.info("TTS request cache hit for user %s.", current_user_id)
        return await _save_generation(
            db,
            payload,
            current_user_id,
            voice_id,
            control_instruction,
            cached_sample_rate,
            cached_audio,
        )

    sample_rate, wav_np = await _synthesize_request(
        payload,
        reference_wav_path,
        control_instruction,
        on_progress,
        on_audio=on_audio,
        cache_namespace=cache_namespace,
        on_pace_retry=on_pace_retry,
    )
    full_cache_key["engine_device"] = tts_engine.cache_device_key()
    full_cache = _cache_path(full_cache_key)
    generation = await _save_generation(
        db,
        payload,
        current_user_id,
        voice_id,
        control_instruction,
        sample_rate,
        wav_np,
    )
    _write_audio_cache(full_cache, wav_np, sample_rate)
    return generation


@router.post("/generate", response_model=GenerationRead, status_code=status.HTTP_201_CREATED)
async def generate_tts(
    payload: GenerationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    voice_id, reference_wav_path, control_instruction = await _resolve_voice(payload, db)
    try:
        async with generation_slots:
            return await _generate_and_save(
                payload,
                db,
                current_user.id,
                voice_id,
                reference_wav_path,
                control_instruction,
            )
    except Exception as exc:
        logger.exception("TTS generation failed")
        raise HTTPException(status_code=500, detail=f"TTS generation error: {exc}") from exc


async def _run_generation_job(
    job: GenerationJob,
    payload: GenerationCreate,
    voice_id: Optional[uuid.UUID],
    reference_wav_path: Optional[str],
    control_instruction: str,
    stream_events: Optional[asyncio.Queue[dict[str, object] | None]] = None,
) -> None:
    def update_progress(current: int, total: int, message: str) -> None:
        job.current_chunk = current
        job.total_chunks = total
        job.message = message
        if stream_events is not None:
            stream_events.put_nowait(
                {
                    "type": "progress",
                    "status": job.status,
                    "current_chunk": current,
                    "total_chunks": total,
                    "message": message,
                    "queued_position": job.queued_position,
                }
            )

    def update_pace_retries(count: int) -> None:
        job.pace_retry_count = count
        if stream_events is not None:
            stream_events.put_nowait(
                {
                    "type": "progress",
                    "status": job.status,
                    "current_chunk": job.current_chunk,
                    "total_chunks": job.total_chunks,
                    "message": f"Trying an alternate seed (pace retry {count})",
                    "queued_position": job.queued_position,
                }
            )

    def stream_audio(
        current: int,
        total: int,
        sample_rate: int,
        audio: np.ndarray,
        pause_after: str,
    ) -> None:
        if stream_events is None:
            return
        if job.first_audio_seconds is None:
            job.first_audio_seconds = time.perf_counter() - request_started
            logger.info(
                "TTS job %s time_to_first_audio_seconds=%.2f.",
                job.job_id,
                job.first_audio_seconds,
            )
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format="WAV")
        pause_ms = {
            "continuation": 0,
            "sentence": settings.TTS_SENTENCE_PAUSE_MS,
            "paragraph": settings.TTS_PARAGRAPH_PAUSE_MS,
        }[pause_after]
        stream_events.put_nowait(
            {
                "type": "audio",
                "current_chunk": current,
                "total_chunks": total,
                "sample_rate": sample_rate,
                "pause_ms": pause_ms,
                "audio_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            }
        )

    request_started = time.perf_counter()
    try:
        async with generation_slots:
            job.status = "running"
            job.message = "Generating speech"
            job.queue_wait_seconds = time.perf_counter() - request_started
            generation_started = time.perf_counter()
            async with AsyncSessionLocal() as db:
                generation = await _generate_and_save(
                    payload,
                    db,
                    job.user_id,
                    voice_id,
                    reference_wav_path,
                    control_instruction,
                    update_progress,
                    stream_audio if stream_events is not None else None,
                    update_pace_retries,
                )
            job.generation_seconds = time.perf_counter() - generation_started
        job.generation = generation.model_dump()
        job.status = "completed"
        job.message = "Speech is ready"
        audio_duration = float(job.generation.get("duration") or 0.0)
        final_audio_path = AUDIO_DIR / f"{job.generation['id']}.wav"
        final_audio, final_sample_rate = sf.read(
            str(final_audio_path),
            dtype="float32",
            always_2d=False,
        )
        timing_metrics = speech_timing_metrics(final_audio, final_sample_rate)
        measured_cps = (
            len(payload.text) / float(timing_metrics["speech_seconds"])
            if timing_metrics["speech_seconds"]
            else None
        )
        real_time_factor = (
            job.generation_seconds / audio_duration if audio_duration > 0 else None
        )
        logger.info(
            "TTS job %s finished: queue_wait_seconds=%.2f first_audio_seconds=%s "
            "generation_seconds=%.2f audio_seconds=%.2f real_time_factor=%s.",
            job.job_id,
            job.queue_wait_seconds or 0.0,
            f"{job.first_audio_seconds:.2f}" if job.first_audio_seconds is not None else "cache",
            job.generation_seconds or 0.0,
            audio_duration,
            f"{real_time_factor:.2f}" if real_time_factor is not None else "n/a",
        )
        if stream_events is not None:
            stream_events.put_nowait(
                {
                    "type": "complete",
                    "generation": job.generation,
                    "metrics": {
                        "queue_wait_seconds": job.queue_wait_seconds,
                        "time_to_first_audio_seconds": job.first_audio_seconds,
                        "generation_seconds": job.generation_seconds,
                        "pace_retry_count": job.pace_retry_count,
                        "speech_only_chars_per_second": measured_cps,
                        "total_chars_per_second": (
                            len(payload.text) / audio_duration if audio_duration > 0 else None
                        ),
                        "pause_count_250ms": timing_metrics["pause_count_250ms"],
                        "pause_seconds_250ms": timing_metrics["pause_seconds_250ms"],
                        "audio_duration_seconds": audio_duration,
                    },
                }
            )
    except Exception as exc:
        logger.exception("Background TTS job %s failed", job.job_id)
        job.status = "failed"
        job.error = str(exc)
        job.message = "Speech generation failed"
        if stream_events is not None:
            stream_events.put_nowait({"type": "error", "message": job.error})
    finally:
        if stream_events is not None:
            stream_events.put_nowait(None)


async def _event_stream(
    events: asyncio.Queue[dict[str, object] | None],
    job: GenerationJob,
):
    while True:
        try:
            event = await asyncio.wait_for(events.get(), timeout=2)
        except asyncio.TimeoutError:
            if job.status == "queued":
                snapshot = job.as_response()
                yield f"data: {json.dumps({'type': 'progress', **snapshot}, ensure_ascii=False, default=str)}\n\n"
            else:
                yield ": keep-alive\n\n"
            continue
        if event is None:
            return
        yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


@router.post("/generate/stream")
async def create_streaming_generation(
    payload: GenerationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    voice_id, reference_wav_path, control_instruction = await _resolve_voice(payload, db)
    total_chunks = len(split_speech_text(payload.text))
    job = create_job(current_user.id, total_chunks)
    events: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()
    events.put_nowait(
        {
            "type": "progress",
            "status": job.status,
            "current_chunk": 0,
            "total_chunks": total_chunks,
            "message": job.message,
            "queued_position": job.as_response()["queued_position"],
        }
    )
    job.task = asyncio.create_task(
        _run_generation_job(
            job,
            payload,
            voice_id,
            reference_wav_path,
            control_instruction,
            events,
        )
    )
    return StreamingResponse(
        _event_stream(events, job),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/generate/jobs",
    response_model=GenerationJobRead,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_generation_job(
    payload: GenerationCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    voice_id, reference_wav_path, control_instruction = await _resolve_voice(payload, db)
    total_chunks = len(split_speech_text(payload.text))
    job = create_job(current_user.id, total_chunks)
    job.task = asyncio.create_task(
        _run_generation_job(
            job,
            payload,
            voice_id,
            reference_wav_path,
            control_instruction,
        )
    )
    return job.as_response()


@router.get("/generate/jobs/{job_id}", response_model=GenerationJobRead)
async def get_generation_job(
    job_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
):
    job = get_job(job_id, current_user.id)
    if job is None:
        raise HTTPException(status_code=404, detail="Generation job not found")
    return job.as_response()


@router.get("/history")
async def get_history(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    offset = (page - 1) * limit
    result = await db.execute(
        select(Generation)
        .where(Generation.user_id == current_user.id)
        .order_by(desc(Generation.created_at))
        .offset(offset)
        .limit(limit)
    )
    generations = result.scalars().all()

    total_result = await db.execute(
        select(func.count(Generation.id)).where(Generation.user_id == current_user.id)
    )
    total = total_result.scalar_one()

    items = [
        GenerationHistory(
            id=g.id,
            title=g.title,
            text_input=g.text_input,
            control_instruction=g.control_instruction,
            voice_id=g.voice_id,
            cfg_value=g.cfg_value,
            inference_timesteps=g.inference_timesteps,
            speed=g.speed,
            pace_mode=g.pace_mode,
            target_cps=g.target_cps,
            pronunciation_dictionary=g.pronunciation_dictionary,
            noise_reduction=g.noise_reduction,
            style=g.style,
            audio_duration=g.audio_duration,
            status=g.status,
            created_at=g.created_at,
            audio_url=f"/api/tts/audio/{g.id}",
        )
        for g in generations
    ]

    return {
        "items": items,
        "total": total,
        "page": page,
        "limit": limit,
        "pages": (total + limit - 1) // limit,
    }


@router.patch("/history/{generation_id}")
async def update_history_title(
    generation_id: uuid.UUID,
    payload: GenerationTitleUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Generation).where(
            Generation.id == generation_id,
            Generation.user_id == current_user.id,
        )
    )
    generation = result.scalar_one_or_none()
    if generation is None:
        raise HTTPException(status_code=404, detail="Generation not found")

    generation.title = payload.title
    await db.commit()
    return {"id": str(generation.id), "title": generation.title}


@router.delete("/history/{generation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_history(
    generation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Generation).where(
            Generation.id == generation_id,
            Generation.user_id == current_user.id,
        )
    )
    generation = result.scalar_one_or_none()
    if not generation:
        raise HTTPException(status_code=404, detail="Generation not found")

    # Remove audio file
    if generation.audio_file_path and os.path.exists(generation.audio_file_path):
        try:
            os.remove(generation.audio_file_path)
        except OSError:
            pass

    await db.delete(generation)
    await db.commit()


@router.get("/audio/{generation_id}")
async def stream_audio(
    generation_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Generation).where(
            Generation.id == generation_id,
            Generation.user_id == current_user.id,
        )
    )
    generation = result.scalar_one_or_none()
    if not generation:
        raise HTTPException(status_code=404, detail="Generation not found")

    if not generation.audio_file_path or not os.path.exists(generation.audio_file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        path=generation.audio_file_path,
        media_type="audio/wav",
        filename=f"{generation_id}.wav",
    )
