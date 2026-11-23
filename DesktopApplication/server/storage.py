from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator


SCHEMA_VERSION = 3
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
        self.voice_dir = self.root / "voices"

    def initialize(self) -> None:
        for directory in (
            self.db_dir,
            self.backup_dir,
            self.audio_dir,
            self.drafts_dir,
            self.logs_dir,
            self.settings_dir,
            self.voice_dir,
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
            self._load_media_folder(connection)
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
        connection.execute("BEGIN IMMEDIATE")
        try:
            if version == 0:
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
                version = 1
            if version == 1:
                connection.execute(
                    "ALTER TABLE generations ADD COLUMN captions_json TEXT NOT NULL DEFAULT '[]'"
                )
                version = 2
            if version == 2:
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
                    "CREATE INDEX IF NOT EXISTS idx_generations_voice_pace_steps "
                    "ON generations(voice_id, pace, steps)"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_generations_favorite_created_at "
                    "ON generations(is_favorite, created_at DESC)"
                )
                version = 3
            if version != SCHEMA_VERSION:
                raise UnsupportedDatabaseVersion(
                    f"No migration is defined from schema version {version}."
                )
            connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            connection.commit()
        finally:
            if connection.in_transaction:
                connection.rollback()

    def _load_media_folder(self, connection: sqlite3.Connection) -> None:
        settings_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'settings'"
        ).fetchone()
        if settings_table is None:
            return
        row = connection.execute(
            "SELECT value_json FROM settings WHERE key = 'media_folder'"
        ).fetchone()
        if row is not None:
            configured_path = Path(json.loads(row["value_json"])).expanduser().resolve()
            configured_path.mkdir(parents=True, exist_ok=True)
            self.audio_dir = configured_path

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
        captions: list[dict[str, Any]] | None = None,
    ) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        captions_json = json.dumps(captions or [], ensure_ascii=False, allow_nan=False)
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO generations (
                    id, created_at, title, text, voice_id, pace, steps, speed,
                    audio_path, audio_sha256, audio_duration, captions_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    captions_json,
                ),
            )

    def get_generation(self, generation_id: str) -> dict[str, Any] | None:
        with self._reader() as connection:
            row = connection.execute(
                """
                SELECT id, created_at, title, text, voice_id, pace, steps, speed,
                       audio_path, audio_sha256, audio_duration, captions_json, status
                FROM generations WHERE id = ?
                """,
                (generation_id,),
            ).fetchone()
        if row is None:
            return None
        generation = dict(row)
        generation["captions"] = json.loads(generation.pop("captions_json"))
        return generation

    def list_generations(
        self,
        *,
        search: str = "",
        date_from: str | None = None,
        date_to: str | None = None,
        voice: str | None = None,
        pace: str | None = None,
        steps: int | None = None,
        favorites_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        filters: list[str] = []
        values: list[Any] = []
        if search:
            escaped_search = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            filters.append("(title LIKE ? ESCAPE '\\' OR text LIKE ? ESCAPE '\\')")
            values.extend((f"%{escaped_search}%", f"%{escaped_search}%"))
        if date_from is not None:
            filters.append("created_at >= ?")
            start_date = datetime.combine(
                date.fromisoformat(date_from),
                datetime.min.time(),
            ).astimezone(timezone.utc)
            values.append(start_date.isoformat())
        if date_to is not None:
            filters.append("created_at < ?")
            exclusive_date_to = date.fromisoformat(date_to) + timedelta(days=1)
            end_date = datetime.combine(
                exclusive_date_to,
                datetime.min.time(),
            ).astimezone(timezone.utc)
            values.append(end_date.isoformat())
        if voice is not None:
            filters.append("voice_id = ?")
            values.append(voice)
        if pace is not None:
            filters.append("pace = ?")
            values.append(pace)
        if steps is not None:
            filters.append("steps = ?")
            values.append(steps)
        if favorites_only:
            filters.append("is_favorite = 1")
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        values.extend((limit, offset))

        with self._reader() as connection:
            rows = connection.execute(
                f"""
                SELECT id, created_at, title, text, voice_id, pace, steps, speed,
                       audio_path, audio_sha256, audio_duration, is_favorite, status
                FROM generations
                {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                values,
            ).fetchall()
        generations: list[dict[str, Any]] = []
        for row in rows:
            generation = dict(row)
            audio_path = Path(generation["audio_path"]) if generation["audio_path"] else None
            generation["is_favorite"] = bool(generation["is_favorite"])
            generation["audio_missing"] = audio_path is None or not audio_path.is_file()
            generations.append(generation)
        return generations

    def rename_generation(self, generation_id: str, title: str) -> None:
        normalized_title = title.strip()
        if not normalized_title or len(normalized_title) > 120:
            raise ValueError("History title must contain between 1 and 120 characters.")
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE generations SET title = ? WHERE id = ?",
                (normalized_title, generation_id),
            )
            if cursor.rowcount == 0:
                raise StorageError("Generation history was not found.")

    def set_generation_favorite(self, generation_id: str, favorite: bool) -> None:
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE generations SET is_favorite = ? WHERE id = ?",
                (int(favorite), generation_id),
            )
            if cursor.rowcount == 0:
                raise StorageError("Generation history was not found.")

    def delete_generation(self, generation_id: str) -> bool:
        generation = self.get_generation(generation_id)
        if generation is None:
            return False
        audio_path = Path(generation["audio_path"]) if generation["audio_path"] else None
        staged_path: Path | None = None
        if audio_path is not None and audio_path.is_file():
            staged_path = audio_path.with_name(
                f".{audio_path.stem}.{uuid.uuid4().hex}.deleting{audio_path.suffix}"
            )
            os.replace(audio_path, staged_path)
        try:
            with self._transaction() as connection:
                cursor = connection.execute(
                    "DELETE FROM generations WHERE id = ?",
                    (generation_id,),
                )
                if cursor.rowcount == 0:
                    raise StorageError("Generation history was not found.")
        except Exception:
            if staged_path is not None and staged_path.exists() and audio_path is not None:
                os.replace(staged_path, audio_path)
            raise
        if staged_path is not None:
            try:
                staged_path.unlink()
            except OSError as exc:
                raise StorageError(
                    f"History was deleted, but its staged audio could not be removed: {exc}"
                ) from exc
        return True

    def duplicate_generation(self, generation_id: str) -> dict[str, Any] | None:
        generation = self.get_generation(generation_id)
        if generation is None:
            return None

        duplicate_id = str(uuid.uuid4())
        original_path = Path(generation["audio_path"]) if generation["audio_path"] else None
        duplicate_path = self.audio_dir / f"{duplicate_id}.wav"
        temporary_path = self.audio_dir / f".{duplicate_id}.tmp.wav"
        saved_path: str | None = None
        if original_path is not None and original_path.is_file():
            try:
                with original_path.open("rb") as source, temporary_path.open("xb") as target:
                    shutil.copyfileobj(source, target)
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(temporary_path, duplicate_path)
                saved_path = str(duplicate_path)
            except OSError as exc:
                temporary_path.unlink(missing_ok=True)
                duplicate_path.unlink(missing_ok=True)
                raise StorageError(f"Could not duplicate the generated audio: {exc}") from exc

        title = (generation["title"] or generation["text"][:80]).strip()
        try:
            self.save_generation(
                duplicate_id,
                generation["text"],
                generation["voice_id"],
                generation["pace"],
                generation["steps"],
                title=f"{title[:110]} (copy)",
                speed=generation["speed"],
                audio_path=saved_path,
                audio_sha256=generation["audio_sha256"] if saved_path else None,
                audio_duration=generation["audio_duration"] if saved_path else None,
                captions=generation["captions"],
            )
        except Exception:
            temporary_path.unlink(missing_ok=True)
            duplicate_path.unlink(missing_ok=True)
            raise
        return self.get_generation(duplicate_id)

    def generation_count(self) -> int:
        with self._reader() as connection:
            row = connection.execute("SELECT COUNT(*) FROM generations").fetchone()
        return int(row[0]) if row else 0

    def audio_storage_bytes(self) -> int:
        if not self.audio_dir.is_dir():
            return 0
        return sum(
            path.stat().st_size
            for path in self.audio_dir.glob("*.wav")
            if path.is_file()
        )

    def cleanup_generations_before(self, cutoff: str) -> int:
        with self._reader() as connection:
            rows = connection.execute(
                "SELECT id FROM generations WHERE created_at < ? ORDER BY created_at",
                (cutoff,),
            ).fetchall()
        deleted = 0
        for row in rows:
            if self.delete_generation(row["id"]):
                deleted += 1
        return deleted

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def move_audio_folder(self, requested_path: str) -> dict[str, Any]:
        if not requested_path.strip() or len(requested_path) > 1024:
            raise ValueError("Choose a media folder path between 1 and 1,024 characters.")
        new_folder = Path(requested_path).expanduser().resolve()
        if (
            new_folder == self.root.resolve()
            or new_folder in self.root.resolve().parents
            or self.root.resolve() in new_folder.parents
        ):
            raise ValueError("The media folder must be separate from the KalapakTTS profile folder.")
        old_folder = self.audio_dir.resolve()
        new_folder.mkdir(parents=True, exist_ok=True)
        if new_folder == old_folder:
            return {
                "media_folder": str(new_folder),
                "moved_files": 0,
                "storage_used_bytes": self.audio_storage_bytes(),
            }

        with self._reader() as connection:
            rows = connection.execute(
                "SELECT id, audio_path FROM generations WHERE audio_path IS NOT NULL"
            ).fetchall()

        moves: list[tuple[str, Path, Path]] = []
        created_targets: list[Path] = []
        temporary: Path | None = None
        try:
            for row in rows:
                source = Path(row["audio_path"])
                if not source.is_file():
                    continue
                destination = new_folder / f"{row['id']}{source.suffix.lower()}"
                if source.resolve() == destination.resolve():
                    continue
                if destination.exists():
                    if self._sha256_file(source) != self._sha256_file(destination):
                        raise StorageError(
                            f"Media move stopped because {destination.name} already exists with different contents."
                        )
                else:
                    temporary = new_folder / f".{destination.stem}.{uuid.uuid4().hex}.moving{destination.suffix}"
                    with source.open("rb") as input_file, temporary.open("xb") as output_file:
                        shutil.copyfileobj(input_file, output_file)
                        output_file.flush()
                        os.fsync(output_file.fileno())
                    os.replace(temporary, destination)
                    temporary = None
                    created_targets.append(destination)
                moves.append((row["id"], source, destination))
        except (OSError, StorageError) as exc:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            for copied_path in created_targets:
                copied_path.unlink(missing_ok=True)
            if isinstance(exc, StorageError):
                raise
            raise StorageError(f"Could not prepare the media-folder move: {exc}") from exc

        try:
            with self._transaction() as connection:
                for generation_id, _source, destination in moves:
                    connection.execute(
                        "UPDATE generations SET audio_path = ? WHERE id = ?",
                        (str(destination), generation_id),
                    )
                value_json = json.dumps(str(new_folder), ensure_ascii=False)
                updated_at = datetime.now(timezone.utc).isoformat()
                connection.execute(
                    """
                    INSERT INTO settings (key, value_json, updated_at)
                    VALUES ('media_folder', ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value_json=excluded.value_json,
                        updated_at=excluded.updated_at
                    """,
                    (value_json, updated_at),
                )
        except Exception:
            for copied_path in created_targets:
                copied_path.unlink(missing_ok=True)
            raise

        self.audio_dir = new_folder
        removal_errors: list[str] = []
        for _generation_id, source, _destination in moves:
            if source.resolve().is_relative_to(old_folder) and source.exists():
                try:
                    source.unlink()
                except OSError as exc:
                    removal_errors.append(f"{source.name}: {exc}")
        if removal_errors:
            raise StorageError(
                "The media folder was changed, but old copies could not be removed: "
                + "; ".join(removal_errors)
            )
        return {
            "media_folder": str(new_folder),
            "moved_files": len(moves),
            "storage_used_bytes": self.audio_storage_bytes(),
        }

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
