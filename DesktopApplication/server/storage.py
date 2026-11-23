from __future__ import annotations

import json
import os
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator


SCHEMA_VERSION = 1
DATABASE_FILENAME = "kalapaktts.sqlite3"


class StorageError(RuntimeError):
    pass


class DatabaseRecoveryRequired(StorageError):
    def __init__(
        self,
        database_path: Path,
        reason: str,
        backup_path: Path | None,
    ) -> None:
        self.database_path = database_path
        self.reason = reason
        self.backup_path = backup_path
        if backup_path is None:
            recovery = "No valid backup is available."
        else:
            recovery = f"Last backup: {backup_path.name}."
        super().__init__(
            f"The local database failed its integrity check ({reason}). "
            f"{recovery} Use the recovery action to restore a backup."
        )


class UnsupportedDatabaseVersion(StorageError):
    pass


def default_data_root() -> Path:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        appdata = str(Path.home() / "AppData" / "Roaming")
    return Path(appdata) / "KalapakTTS"


class LocalStore:
    def __init__(self, data_root: Path) -> None:
        self.root = Path(data_root)
        self.db_dir = self.root / "db"
        self.database_path = self.db_dir / DATABASE_FILENAME
        self.backup_dir = self.db_dir / "backups"
        self.audio_dir = self.root / "audio"
        self.drafts_dir = self.root / "drafts"
        self.logs_dir = self.root / "logs"
        self.settings_dir = self.root / "settings"

    def initialize(self) -> None:
        for directory in (
            self.db_dir,
            self.backup_dir,
            self.audio_dir,
            self.drafts_dir,
            self.logs_dir,
            self.settings_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        existed = self.database_path.is_file() and self.database_path.stat().st_size > 0
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if integrity is None or integrity[0] != "ok":
                reason = "SQLite reported corruption" if integrity is None else str(integrity[0])
                raise DatabaseRecoveryRequired(
                    self.database_path,
                    reason,
                    self.latest_backup(),
                )

            connection.execute("PRAGMA journal_mode=WAL")
            version_row = connection.execute("PRAGMA user_version").fetchone()
            version = int(version_row[0]) if version_row else 0
            if version > SCHEMA_VERSION:
                raise UnsupportedDatabaseVersion(
                    f"Database schema version {version} is newer than supported version "
                    f"{SCHEMA_VERSION}."
                )
            if version < SCHEMA_VERSION:
                if existed:
                    self._create_backup(connection, version)
                self._migrate(connection, version)
        except sqlite3.DatabaseError as exc:
            raise DatabaseRecoveryRequired(
                self.database_path,
                str(exc),
                self.latest_backup(),
            ) from exc
        finally:
            if connection is not None:
                connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA synchronous=FULL")
        except sqlite3.DatabaseError:
            connection.close()
            raise
        return connection

    def _create_backup(self, connection: sqlite3.Connection, version: int) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_path = self.backup_dir / (
            f"kalapaktts-pre-v{version}-to-v{SCHEMA_VERSION}-{timestamp}.sqlite3"
        )
        backup_connection = sqlite3.connect(backup_path)
        try:
            connection.backup(backup_connection)
        finally:
            backup_connection.close()
        return backup_path

    @staticmethod
    def _migrate(connection: sqlite3.Connection, version: int) -> None:
        if version != 0:
            raise UnsupportedDatabaseVersion(f"No migration is defined from schema version {version}.")

        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS generations (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    title TEXT,
                    text TEXT NOT NULL,
                    voice_id TEXT NOT NULL,
                    pace TEXT NOT NULL,
                    steps INTEGER NOT NULL CHECK (steps > 0),
                    speed REAL NOT NULL DEFAULT 1.0,
                    audio_path TEXT,
                    audio_sha256 TEXT,
                    audio_duration REAL,
                    is_favorite INTEGER NOT NULL DEFAULT 0 CHECK (is_favorite IN (0, 1)),
                    status TEXT NOT NULL DEFAULT 'completed'
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS drafts (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    settings_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_generations_created_at "
                "ON generations(created_at DESC)"
            )
            connection.execute("PRAGMA user_version=1")
            connection.commit()
        finally:
            if connection.in_transaction:
                connection.rollback()

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Connection, None, None]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        finally:
            if connection.in_transaction:
                connection.rollback()
            connection.close()

    @contextmanager
    def _reader(self) -> Generator[sqlite3.Connection, None, None]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def save_draft(self, draft_id: str, text: str, settings: dict[str, Any]) -> None:
        if not draft_id or len(draft_id) > 128:
            raise ValueError("Draft id must contain between 1 and 128 characters.")
        settings_json = json.dumps(settings, ensure_ascii=False, allow_nan=False)
        updated_at = datetime.now(timezone.utc).isoformat()
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO drafts (id, text, settings_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    text=excluded.text,
                    settings_json=excluded.settings_json,
                    updated_at=excluded.updated_at
                """,
                (draft_id, text, settings_json, updated_at),
            )

    def get_draft(self, draft_id: str) -> dict[str, Any] | None:
        with self._reader() as connection:
            row = connection.execute(
                "SELECT id, text, settings_json, updated_at FROM drafts WHERE id = ?",
                (draft_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "text": row["text"],
            "settings": json.loads(row["settings_json"]),
            "updated_at": row["updated_at"],
        }

    def save_setting(self, key: str, value: Any) -> None:
        if not key or len(key) > 128:
            raise ValueError("Setting key must contain between 1 and 128 characters.")
        value_json = json.dumps(value, ensure_ascii=False, allow_nan=False)
        updated_at = datetime.now(timezone.utc).isoformat()
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO settings (key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_at=excluded.updated_at
                """,
                (key, value_json, updated_at),
            )

    def get_setting(self, key: str) -> Any | None:
        with self._reader() as connection:
            row = connection.execute(
                "SELECT value_json FROM settings WHERE key = ?",
                (key,),
            ).fetchone()
        return json.loads(row["value_json"]) if row else None

    def save_generation(
        self,
        generation_id: str,
        text: str,
        voice_id: str,
        pace: str,
        steps: int,
        *,
        title: str | None = None,
        speed: float = 1.0,
        audio_path: str | None = None,
        audio_sha256: str | None = None,
        audio_duration: float | None = None,
    ) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO generations (
                    id, created_at, title, text, voice_id, pace, steps, speed,
                    audio_path, audio_sha256, audio_duration
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    generation_id,
                    created_at,
                    title,
                    text,
                    voice_id,
                    pace,
                    steps,
                    speed,
                    audio_path,
                    audio_sha256,
                    audio_duration,
                ),
            )

    def schema_version(self) -> int:
        with self._reader() as connection:
            row = connection.execute("PRAGMA user_version").fetchone()
        return int(row[0]) if row else 0

    def latest_backup(self) -> Path | None:
        backups = list(self.backup_dir.glob("*.sqlite3")) if self.backup_dir.exists() else []
        return max(backups, key=lambda path: path.stat().st_mtime_ns) if backups else None

    @staticmethod
    def _check_integrity(database_path: Path) -> None:
        connection = sqlite3.connect(database_path)
        try:
            result = connection.execute("PRAGMA integrity_check").fetchone()
        finally:
            connection.close()
        if result is None or result[0] != "ok":
            detail = "no integrity result" if result is None else str(result[0])
            raise StorageError(f"Backup integrity check failed: {detail}.")

    def restore_latest_backup(self) -> Path:
        backup_path = self.latest_backup()
        if backup_path is None:
            raise StorageError("No database backup is available to restore.")

        temporary_path = self.db_dir / f".{DATABASE_FILENAME}.{uuid.uuid4().hex}.restore"
        try:
            self._check_integrity(backup_path)
            with backup_path.open("rb") as source, temporary_path.open("xb") as destination:
                shutil.copyfileobj(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
            self._check_integrity(temporary_path)

            for suffix in ("-wal", "-shm"):
                sidecar = self.database_path.with_name(self.database_path.name + suffix)
                if sidecar.exists():
                    sidecar.unlink()
            os.replace(temporary_path, self.database_path)
        except (OSError, sqlite3.DatabaseError, StorageError) as exc:
            raise StorageError(
                f"Could not restore database backup {backup_path.name}: {exc}"
            ) from exc
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return backup_path
