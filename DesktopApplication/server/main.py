from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from server.gpu import read_gpu_status
from server.storage import (
    SCHEMA_VERSION,
    DatabaseRecoveryRequired,
    LocalStore,
    StorageError,
    default_data_root,
)


ROOT = Path(__file__).resolve().parents[1]
UI_DIR = ROOT / "ui"
REPORT_PATH = ROOT / "build" / "pywebview-check.json"
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


class DraftPayload(BaseModel):
    text: str = Field(max_length=5000)
    settings: dict[str, Any] = Field(default_factory=dict)


class SettingPayload(BaseModel):
    value: Any


def create_app(
    data_root: Path | None = None,
    gpu_reader: Callable[[], dict[str, Any]] = read_gpu_status,
) -> FastAPI:
    store = LocalStore(data_root or default_data_root())

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.local_store = store
        application.state.storage_error = None
        try:
            store.initialize()
        except DatabaseRecoveryRequired as exc:
            application.state.storage_error = exc
        application.state.gpu_startup_status = gpu_reader()
        yield

    application = FastAPI(
        title="KalapakTTS Desktop",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def apply_content_security_policy(request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    def require_ready_store(request: Request) -> LocalStore:
        storage_error = request.app.state.storage_error
        if storage_error is not None:
            raise HTTPException(status_code=503, detail=str(storage_error))
        return request.app.state.local_store

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "application": "KalapakTTS Desktop"}

    @application.get("/api/system/gpu")
    def system_gpu_status(request: Request) -> dict[str, Any]:
        return {
            "startup": request.app.state.gpu_startup_status,
            "current": gpu_reader(),
        }

    @application.get("/api/storage/status")
    def storage_status(request: Request) -> dict[str, Any]:
        local_store: LocalStore = request.app.state.local_store
        storage_error = request.app.state.storage_error
        backup_path = local_store.latest_backup()
        common = {
            "root": str(local_store.root),
            "backup_available": backup_path is not None,
            "latest_backup": backup_path.name if backup_path else None,
        }
        if storage_error is not None:
            return {
                **common,
                "status": "recovery_required",
                "schema_version": None,
                "message": str(storage_error),
            }
        return {
            **common,
            "status": "ready",
            "schema_version": SCHEMA_VERSION,
            "message": None,
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

    @application.post("/_poc/report")
    async def save_poc_report(request: Request) -> dict[str, bool]:
        report: Any = await request.json()
        if not isinstance(report, dict):
            raise HTTPException(status_code=422, detail="POC report must be a JSON object.")
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"saved": True}

    @application.get("/poc", response_class=FileResponse)
    async def poc_page() -> FileResponse:
        return FileResponse(ROOT / "app" / "poc.html", media_type="text/html; charset=utf-8")

    application.mount("/", StaticFiles(directory=UI_DIR, html=True), name="ui")
    return application


app = create_app()
