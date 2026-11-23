from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import secrets
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, SecretStr
import soundfile as sf
from starlette.background import BackgroundTask

from server.app_lock import (
    LOCKOUT_SECONDS,
    MAX_FAILED_ATTEMPTS,
    hash_password,
    verify_password,
)
from server.backup import MAX_BACKUP_BYTES, export_profile, import_profile
from server.config import settings
from server.gpu import read_gpu_status
from server.model_assets import ModelAssetsError
from server.storage import (
    SCHEMA_VERSION,
    DatabaseRecoveryRequired,
    LocalStore,
    StorageError,
    default_data_root,
)
from server.tts_chunking import split_speech_text
from server.tts_engine import tts_engine
from server.tts_jobs import create_job, generation_slots
from server.tts_service import GenerationOptions, format_subtitles, stream_synthesis


ROOT = Path(__file__).resolve().parents[1]
UI_DIR = ROOT / "ui"
AUDIO_EXPORT_ENCODERS: dict[str, tuple[str, ...]] = {
    "mp3": ("-c:a", "libmp3lame", "-b:a", "192k", "-f", "mp3"),
    "flac": ("-c:a", "flac", "-compression_level", "5", "-f", "flac"),
    "m4a": ("-c:a", "aac", "-b:a", "192k", "-f", "ipod"),
    "ogg": ("-c:a", "libopus", "-b:a", "96k", "-vbr", "on", "-f", "ogg"),
}
AUDIO_EXPORT_MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "m4a": "audio/mp4",
    "ogg": "audio/ogg",
}
AudioExportFormat = Literal["mp3", "wav", "flac", "m4a", "ogg"]
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "base-uri 'self'; "
    "object-src 'none'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "media-src 'self' blob: data:; "
    "worker-src 'self' blob:; "
    "frame-src 'none'; "
    "frame-ancestors 'none'; "
    "form-action 'self'"
)
logger = logging.getLogger(__name__)


class DraftPayload(BaseModel):
    text: str = Field(max_length=5000)
    settings: dict[str, Any] = Field(default_factory=dict)


class SettingPayload(BaseModel):
    value: Any


class HistoryTitlePayload(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class FavoritePayload(BaseModel):
    favorite: bool


class MediaFolderPayload(BaseModel):
    path: str = Field(min_length=1, max_length=1024)


class CleanupPayload(BaseModel):
    before: str


class AppLockSetupPayload(BaseModel):
    password: SecretStr = Field(min_length=6, max_length=128)
    confirmation: SecretStr = Field(min_length=6, max_length=128)


class AppLockPasswordPayload(BaseModel):
    password: SecretStr = Field(min_length=1, max_length=128)


class EngineLoadPayload(BaseModel):
    device: Literal["cuda", "cpu"]
    try_gpu_anyway: bool = False


class GenerationPayload(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    voice: Literal["vanna"] = "vanna"
    pace: Literal["relaxed", "natural", "brisk"] = "natural"
    steps: Literal[4, 8, 12] = 4
    speed: float = Field(default=1.0, ge=0.75, le=1.25)
    pronunciation_dictionary: bool = True


def create_app(
    data_root: Path | None = None,
    gpu_reader: Callable[[], dict[str, Any]] = read_gpu_status,
    engine: Any = tts_engine,
    *,
    api_token: str | None = None,
    expected_host: str | None = None,
) -> FastAPI:
    if api_token is not None and (len(api_token) < 32 or not expected_host):
        raise ValueError("Secure desktop API mode requires a long token and the expected Host.")
    store = LocalStore(data_root or default_data_root())
    app_lock_hash_key = "app_lock.password_hash"
    app_lock_failures_key = "app_lock.failed_attempts"
    app_lockout_until_key = "app_lock.lockout_until"
    app_lock_attempt_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.local_store = store
        application.state.storage_error = None
        application.state.app_lock_hash = None
        application.state.app_locked = False
        try:
            store.initialize()
            stored_hash = store.get_setting(app_lock_hash_key)
            if isinstance(stored_hash, str) and stored_hash:
                application.state.app_lock_hash = stored_hash
                application.state.app_locked = True
        except DatabaseRecoveryRequired as exc:
            application.state.storage_error = exc
        application.state.gpu_startup_status = gpu_reader()
        application.state.tts_engine = engine
        try:
            yield
        finally:
            unload = getattr(engine, "unload", None)
            if callable(unload):
                await unload()

    application = FastAPI(
        title="KalapakTTS Desktop",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    application.state.api_token = api_token
    application.state.expected_host = expected_host

    @application.middleware("http")
    async def enforce_desktop_api_security(request: Request, call_next):
        if api_token is not None:
            request_host = request.headers.get("host", "").lower()
            if request_host != expected_host.lower():
                return Response("Invalid local Host header.", status_code=403)
            origin = request.headers.get("origin")
            expected_origin = f"http://{expected_host}"
            if origin is not None and origin != expected_origin:
                return Response("Invalid local Origin header.", status_code=403)
            if request.headers.get("sec-fetch-site") == "cross-site":
                return Response("Cross-site requests are not allowed.", status_code=403)
            if request.url.path.startswith("/api/") or request.url.path == "/health":
                supplied_token = request.headers.get("x-desktop-token", "")
                if not secrets.compare_digest(supplied_token, api_token):
                    return Response("A valid desktop API token is required.", status_code=401)

        lock_exempt_paths = {
            "/api/security/status",
            "/api/security/unlock",
            "/api/security/reset",
        }
        if (
            request.url.path.startswith("/api/")
            and request.url.path not in lock_exempt_paths
            and request.app.state.app_locked
        ):
            return JSONResponse(
                {"detail": "Unlock KalapakTTS Desktop to access local profile data."},
                status_code=423,
            )

        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @application.middleware("http")
    async def apply_content_security_policy(request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        return response

    def require_ready_store(request: Request) -> LocalStore:
        storage_error = request.app.state.storage_error
        if storage_error is not None:
            raise HTTPException(status_code=503, detail=str(storage_error))
        return request.app.state.local_store

    def app_lock_status(request: Request) -> dict[str, Any]:
        store_instance: LocalStore = request.app.state.local_store
        lockout_until = store_instance.get_setting(app_lockout_until_key)
        try:
            remaining = max(0, int(float(lockout_until) - time.time()))
        except (TypeError, ValueError):
            remaining = 0
        return {
            "enabled": request.app.state.app_lock_hash is not None,
            "locked": request.app.state.app_locked,
            "lockout_remaining_seconds": remaining,
            "max_failed_attempts": MAX_FAILED_ATTEMPTS,
        }

    def reset_lockout(store_instance: LocalStore) -> None:
        store_instance.save_setting(app_lock_failures_key, 0)
        store_instance.save_setting(app_lockout_until_key, 0)

    @application.get("/api/security/status")
    def security_status(request: Request) -> dict[str, Any]:
        if request.app.state.storage_error is not None:
            return {
                "enabled": False,
                "locked": False,
                "lockout_remaining_seconds": 0,
                "max_failed_attempts": MAX_FAILED_ATTEMPTS,
            }
        return app_lock_status(request)

    @application.post("/api/security/setup")
    def setup_app_lock(
        payload: AppLockSetupPayload,
        request: Request,
    ) -> dict[str, Any]:
        store_instance = require_ready_store(request)
        if request.app.state.app_lock_hash is not None:
            raise HTTPException(status_code=409, detail="The app lock is already enabled.")
        password = payload.password.get_secret_value()
        confirmation = payload.confirmation.get_secret_value()
        if not secrets.compare_digest(password, confirmation):
            raise HTTPException(status_code=422, detail="The PIN/password entries do not match.")
        try:
            password_hash = hash_password(password)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        store_instance.save_setting(app_lock_hash_key, password_hash)
        reset_lockout(store_instance)
        request.app.state.app_lock_hash = password_hash
        return app_lock_status(request)

    @application.post("/api/security/unlock")
    def unlock_app(
        payload: AppLockPasswordPayload,
        request: Request,
    ) -> dict[str, Any]:
        store_instance = require_ready_store(request)
        with app_lock_attempt_lock:
            password_hash = request.app.state.app_lock_hash
            if password_hash is None:
                request.app.state.app_locked = False
                return app_lock_status(request)

            status = app_lock_status(request)
            if status["lockout_remaining_seconds"] > 0:
                retry_after = status["lockout_remaining_seconds"]
                raise HTTPException(
                    status_code=429,
                    detail=f"Too many incorrect attempts. Try again in {retry_after} seconds.",
                    headers={"Retry-After": str(retry_after)},
                )

            password = payload.password.get_secret_value()
            if not verify_password(password_hash, password):
                try:
                    failures = int(store_instance.get_setting(app_lock_failures_key) or 0) + 1
                except (TypeError, ValueError):
                    failures = 1
                if failures >= MAX_FAILED_ATTEMPTS:
                    store_instance.save_setting(app_lock_failures_key, 0)
                    store_instance.save_setting(
                        app_lockout_until_key,
                        time.time() + LOCKOUT_SECONDS,
                    )
                    raise HTTPException(
                        status_code=429,
                        detail=f"Too many incorrect attempts. Try again in {LOCKOUT_SECONDS} seconds.",
                        headers={"Retry-After": str(LOCKOUT_SECONDS)},
                    )
                store_instance.save_setting(app_lock_failures_key, failures)
                raise HTTPException(status_code=401, detail="Incorrect PIN/password.")

            reset_lockout(store_instance)
            request.app.state.app_locked = False
            return app_lock_status(request)

    @application.post("/api/security/lock")
    def lock_app(request: Request) -> dict[str, Any]:
        if request.app.state.app_lock_hash is None:
            raise HTTPException(status_code=409, detail="Set up an app lock first.")
        if generation_slots.locked():
            raise HTTPException(
                status_code=409,
                detail="Wait for speech generation to finish before locking the app.",
            )
        request.app.state.app_locked = True
        return app_lock_status(request)

    @application.post("/api/security/disable")
    def disable_app_lock(
        payload: AppLockPasswordPayload,
        request: Request,
    ) -> dict[str, Any]:
        store_instance = require_ready_store(request)
        password_hash = request.app.state.app_lock_hash
        if password_hash is None:
            return app_lock_status(request)
        if not verify_password(password_hash, payload.password.get_secret_value()):
            raise HTTPException(status_code=401, detail="Incorrect PIN/password.")
        store_instance.save_setting(app_lock_hash_key, None)
        reset_lockout(store_instance)
        request.app.state.app_lock_hash = None
        request.app.state.app_locked = False
        return app_lock_status(request)

    @application.post("/api/security/reset")
    def reset_app_lock(request: Request) -> dict[str, Any]:
        store_instance = require_ready_store(request)
        store_instance.save_setting(app_lock_hash_key, None)
        reset_lockout(store_instance)
        request.app.state.app_lock_hash = None
        request.app.state.app_locked = False
        return {
            **app_lock_status(request),
            "message": "App lock reset. History, drafts and audio were not deleted.",
        }

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "application": "KalapakTTS Desktop"}

    @application.get("/api/system/gpu")
    def system_gpu_status(request: Request) -> dict[str, Any]:
        return {
            "startup": request.app.state.gpu_startup_status,
            "current": gpu_reader(),
        }

    @application.get("/api/tts/status")
    def tts_status(request: Request) -> dict[str, Any]:
        return request.app.state.tts_engine.status()

    @application.get("/api/tts/generations")
    def list_generations(
        request: Request,
        search: str = Query(default="", max_length=120),
        date_from: str | None = Query(default=None, max_length=10),
        date_to: str | None = Query(default=None, max_length=10),
        voice: str | None = Query(default=None, max_length=64),
        pace: Literal["relaxed", "natural", "brisk"] | None = None,
        steps: int | None = Query(default=None, ge=4, le=12),
        favorites_only: bool = False,
        limit: int = Query(default=50, ge=1, le=100),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        local_store = require_ready_store(request)
        try:
            if date_from is not None:
                date.fromisoformat(date_from)
            if date_to is not None:
                date.fromisoformat(date_to)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="History dates must use YYYY-MM-DD.") from exc
        if steps is not None and steps not in {4, 8, 12}:
            raise HTTPException(status_code=422, detail="History steps must be 4, 8, or 12.")
        items = local_store.list_generations(
            search=search,
            date_from=date_from,
            date_to=date_to,
            voice=voice,
            pace=pace,
            steps=steps,
            favorites_only=favorites_only,
            limit=limit,
            offset=offset,
        )
        for item in items:
            generation_id = item["id"]
            item["audio_url"] = (
                None
                if item["audio_missing"]
                else f"/api/tts/generations/{generation_id}/audio"
            )
            item["srt_url"] = f"/api/tts/generations/{generation_id}/subtitles/srt"
            item["vtt_url"] = f"/api/tts/generations/{generation_id}/subtitles/vtt"
        return {
            "items": items,
            "limit": limit,
            "offset": offset,
            "total": local_store.generation_count(),
        }

    @application.patch("/api/tts/generations/{generation_id}")
    def rename_generation(
        generation_id: str,
        payload: HistoryTitlePayload,
        request: Request,
    ) -> dict[str, Any]:
        local_store = require_ready_store(request)
        try:
            local_store.rename_generation(generation_id, payload.title)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except StorageError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"id": generation_id, "title": payload.title.strip()}

    @application.put("/api/tts/generations/{generation_id}/favorite")
    def set_generation_favorite(
        generation_id: str,
        payload: FavoritePayload,
        request: Request,
    ) -> dict[str, Any]:
        local_store = require_ready_store(request)
        try:
            local_store.set_generation_favorite(generation_id, payload.favorite)
        except StorageError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"id": generation_id, "favorite": payload.favorite}

    @application.post("/api/tts/generations/{generation_id}/duplicate")
    def duplicate_generation(generation_id: str, request: Request) -> dict[str, Any]:
        local_store = require_ready_store(request)
        try:
            duplicate = local_store.duplicate_generation(generation_id)
        except StorageError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if duplicate is None:
            raise HTTPException(status_code=404, detail="Generation history was not found.")
        return {"id": duplicate["id"]}

    @application.delete("/api/tts/generations/{generation_id}")
    def delete_generation(generation_id: str, request: Request) -> dict[str, Any]:
        local_store = require_ready_store(request)
        try:
            deleted = local_store.delete_generation(generation_id)
        except StorageError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        if not deleted:
            raise HTTPException(status_code=404, detail="Generation history was not found.")
        return {"deleted": True}

    @application.post("/api/engine/load")
    async def load_engine(payload: EngineLoadPayload, request: Request) -> dict[str, Any]:
        require_ready_store(request)
        if payload.device == "cuda":
            gpu_status = gpu_reader()
            if not gpu_status.get("available") or not gpu_status.get("cuda_available", True):
                raise HTTPException(status_code=503, detail=gpu_status["warning"])
            if not gpu_status["safe_to_load_model"] and not payload.try_gpu_anyway:
                raise HTTPException(status_code=409, detail=gpu_status["warning"])
        try:
            await request.app.state.tts_engine.load_model(
                payload.device,
                allow_low_vram=payload.try_gpu_anyway,
            )
        except ModelAssetsError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return request.app.state.tts_engine.status()

    @application.post("/api/tts/generate")
    async def generate_speech(
        payload: GenerationPayload,
        request: Request,
    ) -> StreamingResponse:
        local_store = require_ready_store(request)
        engine_instance = request.app.state.tts_engine
        engine_status = engine_instance.status()
        if not engine_status["model_available"]:
            raise HTTPException(
                status_code=503,
                detail=engine_status["model_error"] or "The local VoxCPM2 model is unavailable.",
            )
        if not engine_status["model_loaded"]:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Load VoxCPM2 on the GPU or explicitly choose CPU fallback before generating speech."
                ),
            )

        speech_chunks = split_speech_text(payload.text)
        if not speech_chunks:
            raise HTTPException(status_code=422, detail="Text did not produce any speech chunks.")
        job = create_job(uuid.UUID(int=0), len(speech_chunks))
        queued_at = time.perf_counter()
        options = GenerationOptions(
            text=payload.text,
            voice=payload.voice,
            pace=payload.pace,
            steps=payload.steps,
            speed=payload.speed,
            pronunciation_dictionary=payload.pronunciation_dictionary,
        )

        def encode_event(event_name: str, data: dict[str, Any]) -> str:
            encoded = json.dumps(data, ensure_ascii=False, default=str)
            return f"event: {event_name}\ndata: {encoded}\n\n"

        def encode_wav(audio, sample_rate: int) -> str:
            buffer = io.BytesIO()
            sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
            return base64.b64encode(buffer.getvalue()).decode("ascii")

        def save_generation(
            audio,
            sample_rate: int,
            captions: list[dict[str, Any]],
        ) -> dict[str, Any]:
            generation_id = str(uuid.uuid4())
            audio_path = local_store.audio_dir / f"{generation_id}.wav"
            temporary_path = local_store.audio_dir / f".{generation_id}.tmp.wav"
            try:
                sf.write(str(temporary_path), audio, sample_rate, subtype="PCM_16")
                with temporary_path.open("r+b") as saved_audio:
                    saved_audio.flush()
                    os.fsync(saved_audio.fileno())
                os.replace(temporary_path, audio_path)
                digest = hashlib.sha256()
                with audio_path.open("rb") as saved_audio:
                    for block in iter(lambda: saved_audio.read(1024 * 1024), b""):
                        digest.update(block)
                duration = len(audio) / sample_rate
                local_store.save_generation(
                    generation_id,
                    payload.text,
                    payload.voice,
                    payload.pace,
                    payload.steps,
                    title=payload.text[:80],
                    speed=payload.speed,
                    audio_path=str(audio_path),
                    audio_sha256=digest.hexdigest(),
                    audio_duration=duration,
                    captions=captions,
                )
            except (OSError, RuntimeError, ValueError, sqlite3.DatabaseError, StorageError):
                temporary_path.unlink(missing_ok=True)
                audio_path.unlink(missing_ok=True)
                raise
            return {
                "id": generation_id,
                "duration_seconds": duration,
                "audio_url": f"/api/tts/generations/{generation_id}/audio",
                "srt_url": f"/api/tts/generations/{generation_id}/subtitles/srt",
                "vtt_url": f"/api/tts/generations/{generation_id}/subtitles/vtt",
                "audio_sha256": digest.hexdigest(),
            }

        async def event_stream():
            try:
                yield encode_event("progress", job.as_response())
                async with generation_slots:
                    job.queue_wait_seconds = time.perf_counter() - queued_at
                    reset_peak_metrics = getattr(
                        engine_instance,
                        "reset_peak_memory_metrics",
                        None,
                    )
                    if callable(reset_peak_metrics):
                        reset_peak_metrics()
                    async for event_name, event_data in stream_synthesis(
                        options,
                        engine_instance,
                        job,
                    ):
                        if event_name == "chunk":
                            audio_base64 = encode_wav(
                                event_data.pop("audio"),
                                event_data["sample_rate"],
                            )
                            yield encode_event(
                                "chunk",
                                {**event_data, "audio_base64": audio_base64},
                            )
                        elif event_name == "complete":
                            audio = event_data.pop("audio")
                            generation = save_generation(
                                audio,
                                event_data["sample_rate"],
                                event_data["captions"],
                            )
                            job.status = "completed"
                            job.generation = generation
                            job.message = "Speech generation complete"
                            yield encode_event(
                                "complete",
                                {**job.as_response(), **generation},
                            )
                        else:
                            yield encode_event(event_name, event_data)
            except Exception as exc:
                job.status = "failed"
                job.error = str(exc)
                job.message = "Speech generation failed"
                logger.exception("Local TTS generation job %s failed.", job.job_id)
                yield encode_event("error", job.as_response())

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @application.get("/api/tts/generations/{generation_id}/audio")
    def get_generation_audio(generation_id: str, request: Request) -> FileResponse:
        local_store = require_ready_store(request)
        generation = local_store.get_generation(generation_id)
        if generation is None:
            raise HTTPException(status_code=404, detail="Generated audio was not found.")
        audio_path = Path(generation["audio_path"]) if generation["audio_path"] else None
        if audio_path is None or not audio_path.is_file():
            raise HTTPException(
                status_code=404,
                detail="The generated audio file is missing from this PC.",
            )
        return FileResponse(audio_path, media_type="audio/wav", filename=f"{generation_id}.wav")

    def create_audio_download_response(
        generation_id: str,
        request: Request,
        audio_format: AudioExportFormat,
        file_name: str,
    ) -> FileResponse:
        local_store = require_ready_store(request)
        generation = local_store.get_generation(generation_id)
        if generation is None:
            raise HTTPException(status_code=404, detail="Generated audio was not found.")
        audio_path = Path(generation["audio_path"]) if generation["audio_path"] else None
        if audio_path is None or not audio_path.is_file():
            raise HTTPException(
                status_code=404,
                detail="The generated audio file is missing from this PC.",
            )

        media_type = AUDIO_EXPORT_MEDIA_TYPES[audio_format]
        if audio_format == "wav":
            return FileResponse(audio_path, media_type=media_type, filename=file_name)

        ffmpeg = settings.FFMPEG_PATH
        if not ffmpeg.is_file():
            raise HTTPException(
                status_code=503,
                detail="The bundled audio converter is missing from this PC.",
            )
        with tempfile.NamedTemporaryFile(
            prefix="kalapaktts-export-",
            suffix=f".{audio_format}",
            delete=False,
        ) as temporary_file:
            output_path = Path(temporary_file.name)

        command = [
            str(ffmpeg),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(audio_path),
            "-map",
            "0:a:0",
            "-vn",
            *AUDIO_EXPORT_ENCODERS[audio_format],
            str(output_path),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                check=False,
                timeout=300,
            )
        except subprocess.TimeoutExpired as exc:
            output_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=504,
                detail="Audio conversion took too long. Try downloading WAV instead.",
            ) from exc
        except OSError as exc:
            output_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=503,
                detail="The bundled audio converter could not be started.",
            ) from exc

        if result.returncode != 0:
            output_path.unlink(missing_ok=True)
            details = result.stderr.decode("utf-8", errors="replace").strip()
            raise HTTPException(
                status_code=500,
                detail=f"Audio conversion failed: {details[-500:] or 'unknown FFmpeg error'}",
            )
        if not output_path.is_file() or output_path.stat().st_size == 0:
            output_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=500,
                detail="Audio converter returned an empty file.",
            )

        return FileResponse(
            output_path,
            media_type=media_type,
            filename=file_name,
            background=BackgroundTask(output_path.unlink, missing_ok=True),
        )

    @application.get("/api/tts/generations/{generation_id}/download")
    def download_generation_audio(
        generation_id: str,
        request: Request,
        audio_format: AudioExportFormat = Query(
            default="wav",
            alias="format",
        ),
    ) -> FileResponse:
        return create_audio_download_response(
            generation_id,
            request,
            audio_format,
            f"{generation_id}.{audio_format}",
        )

    @application.get("/api/tts/generations/{generation_id}/subtitles/{format_name}")
    def get_generation_subtitles(
        generation_id: str,
        format_name: Literal["srt", "vtt"],
        request: Request,
    ) -> Response:
        local_store = require_ready_store(request)
        generation = local_store.get_generation(generation_id)
        if generation is None:
            raise HTTPException(status_code=404, detail="Generation history was not found.")
        content = format_subtitles(generation["captions"], format_name)
        media_type = "application/x-subrip" if format_name == "srt" else "text/vtt"
        return Response(
            content,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{generation_id}.{format_name}"',
            },
        )

    @application.get("/api/storage/status")
    def storage_status(request: Request) -> dict[str, Any]:
        local_store: LocalStore = request.app.state.local_store
        storage_error = request.app.state.storage_error
        backup_path = local_store.latest_backup()
        common = {
            "root": str(local_store.root),
            "media_folder": str(local_store.audio_dir),
            "audio_storage_bytes": local_store.audio_storage_bytes(),
            "free_disk_bytes": shutil.disk_usage(local_store.audio_dir).free,
            "backup_available": backup_path is not None,
            "latest_backup": backup_path.name if backup_path else None,
        }
        if storage_error is not None:
            return {
                **common,
                "status": "recovery_required",
                "schema_version": None,
                "generation_count": None,
                "message": str(storage_error),
            }
        return {
            **common,
            "status": "ready",
            "schema_version": SCHEMA_VERSION,
            "generation_count": local_store.generation_count(),
            "message": None,
        }

    @application.get("/api/storage/export")
    def export_storage_backup(request: Request) -> FileResponse:
        local_store = require_ready_store(request)
        export_path = local_store.root / f".export-{uuid.uuid4().hex}.zip"
        try:
            export_profile(
                local_store,
                export_path,
                bundled_voice_dir=ROOT / "models" / "voices",
            )
        except (OSError, sqlite3.DatabaseError, StorageError) as exc:
            export_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return FileResponse(
            export_path,
            media_type="application/zip",
            filename="KalapakTTS-backup.zip",
            background=BackgroundTask(export_path.unlink, missing_ok=True),
        )

    @application.put("/api/storage/import")
    async def import_storage_backup(request: Request) -> dict[str, Any]:
        local_store: LocalStore = request.app.state.local_store
        if generation_slots.locked():
            raise HTTPException(
                status_code=409,
                detail="Wait for speech generation to finish before importing a backup.",
            )
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > MAX_BACKUP_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="Backup files must be no larger than 50 GiB.",
                    )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Invalid Content-Length header.") from exc

        temporary_path = local_store.root / f".import-{uuid.uuid4().hex}.zip"
        total_bytes = 0
        try:
            with temporary_path.open("xb") as destination:
                async for block in request.stream():
                    total_bytes += len(block)
                    if total_bytes > MAX_BACKUP_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail="Backup files must be no larger than 50 GiB.",
                        )
                    destination.write(block)
                destination.flush()
                os.fsync(destination.fileno())
            try:
                result = import_profile(
                    local_store,
                    temporary_path,
                    create_pre_import_backup=request.app.state.storage_error is None,
                )
            except (OSError, sqlite3.DatabaseError, StorageError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            request.app.state.storage_error = None
            return {"status": "ready", **result}
        finally:
            temporary_path.unlink(missing_ok=True)

    @application.post("/api/storage/media-folder")
    def move_media_folder(
        payload: MediaFolderPayload,
        request: Request,
    ) -> dict[str, Any]:
        local_store = require_ready_store(request)
        try:
            result = local_store.move_audio_folder(payload.path)
        except (OSError, StorageError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return result

    @application.post("/api/storage/cleanup")
    def cleanup_history(payload: CleanupPayload, request: Request) -> dict[str, Any]:
        local_store = require_ready_store(request)
        try:
            cutoff_date = date.fromisoformat(payload.before)
            cutoff = f"{cutoff_date.isoformat()}T00:00:00+00:00"
            removed = local_store.cleanup_generations_before(cutoff)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Cleanup date must use YYYY-MM-DD.") from exc
        except StorageError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "deleted_generations": removed,
            "audio_storage_bytes": local_store.audio_storage_bytes(),
        }

    @application.get("/api/storage/drafts/{draft_id}")
    def get_draft(draft_id: str, request: Request) -> dict[str, Any]:
        local_store = require_ready_store(request)
        draft = local_store.get_draft(draft_id)
        if draft is None:
            raise HTTPException(status_code=404, detail="Draft not found.")
        return draft

    @application.put("/api/storage/drafts/{draft_id}")
    def save_draft(draft_id: str, payload: DraftPayload, request: Request) -> dict[str, bool]:
        local_store = require_ready_store(request)
        try:
            local_store.save_draft(draft_id, payload.text, payload.settings)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"saved": True}

    @application.get("/api/storage/settings/{key}")
    def get_setting(key: str, request: Request) -> dict[str, Any]:
        local_store = require_ready_store(request)
        value = local_store.get_setting(key)
        if value is None:
            raise HTTPException(status_code=404, detail="Setting not found.")
        return {"key": key, "value": value}

    @application.put("/api/storage/settings/{key}")
    def save_setting(key: str, payload: SettingPayload, request: Request) -> dict[str, bool]:
        local_store = require_ready_store(request)
        try:
            local_store.save_setting(key, payload.value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"saved": True}

    @application.get("/api/storage/backups")
    def list_backups(request: Request) -> dict[str, Any]:
        local_store: LocalStore = request.app.state.local_store
        backups = sorted(
            local_store.backup_dir.glob("*.sqlite3"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        return {"items": [backup.name for backup in backups]}

    @application.post("/api/storage/restore")
    def restore_backup(request: Request) -> dict[str, str]:
        local_store: LocalStore = request.app.state.local_store
        if request.app.state.storage_error is None:
            raise HTTPException(
                status_code=409,
                detail="Database restore is only available when recovery is required.",
            )
        try:
            backup_path = local_store.restore_latest_backup()
            local_store.initialize()
        except (DatabaseRecoveryRequired, StorageError) as exc:
            request.app.state.storage_error = exc
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        request.app.state.storage_error = None
        return {"status": "ready", "restored_backup": backup_path.name}

    @application.get("/poc", response_class=FileResponse)
    async def poc_page() -> FileResponse:
        return FileResponse(ROOT / "app" / "poc.html", media_type="text/html; charset=utf-8")

    application.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
    return application
