from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
import uuid
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from server.storage import SCHEMA_VERSION, LocalStore, StorageError


BACKUP_FORMAT = "kalapaktts-desktop-backup"
BACKUP_VERSION = 1
MAX_BACKUP_BYTES = 50 * 1024**3
MAX_BACKUP_FILES = 200_000
DATABASE_ENTRY = "database/kalapaktts.sqlite3"


class BackupError(StorageError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_file(
    archive: zipfile.ZipFile,
    archive_name: str,
    source_path: Path,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with source_path.open("rb") as source, archive.open(
        archive_name,
        "w",
        force_zip64=True,
    ) as destination:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            destination.write(block)
            digest.update(block)
            size += len(block)
    return {"size": size, "sha256": digest.hexdigest()}


def _safe_archive_path(name: str) -> PurePosixPath:
    if "\\" in name:
        raise BackupError("The backup contains a non-portable file path.")
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or (path.parts and ":" in path.parts[0])
        or path.as_posix() != name
    ):
        raise BackupError("The backup contains an unsafe file path.")
    return path


def _archive_manifest(archive: zipfile.ZipFile) -> dict[str, Any]:
    try:
        if archive.getinfo("manifest.json").file_size > 10 * 1024 * 1024:
            raise BackupError("The backup manifest is too large.")
        raw_manifest = archive.read("manifest.json")
        manifest = json.loads(raw_manifest)
    except (KeyError, json.JSONDecodeError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        raise BackupError("The backup manifest is missing or invalid.") from exc

    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != BACKUP_FORMAT
        or manifest.get("version") != BACKUP_VERSION
        or not isinstance(manifest.get("files"), dict)
        or not isinstance(manifest.get("audio"), dict)
    ):
        raise BackupError("This file is not a supported KalapakTTS backup.")

    archive_infos = archive.infolist()
    if len(archive_infos) > MAX_BACKUP_FILES:
        raise BackupError("The backup contains too many files.")
    names: set[str] = set()
    total_size = 0
    for info in archive_infos:
        if info.is_dir():
            raise BackupError("Directory entries are not supported in a backup.")
        if info.filename in names:
            raise BackupError("The backup contains duplicate file entries.")
        names.add(info.filename)
        if info.filename != "manifest.json":
            _safe_archive_path(info.filename)
        if info.file_size < 0:
            raise BackupError("The backup contains an invalid file size.")
        total_size += info.file_size
        if total_size > MAX_BACKUP_BYTES:
            raise BackupError("The backup exceeds the 50 GiB expanded-size limit.")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise BackupError("Symbolic links are not supported in backups.")

    expected_names = set(manifest["files"])
    if names != expected_names | {"manifest.json"}:
        raise BackupError("The backup entries do not match its manifest.")
    if DATABASE_ENTRY not in expected_names:
        raise BackupError("The backup does not contain its local database.")
    for name, metadata in manifest["files"].items():
        path = _safe_archive_path(name)
        if not (
            name == DATABASE_ENTRY
            or (path.parts[0] == "audio" and len(path.parts) == 2 and path.suffix == ".wav")
            or name.startswith("voices/custom/")
            or name.startswith("voices/presets/")
        ):
            raise BackupError("The backup contains an unsupported file path.")
        if (
            not isinstance(metadata, dict)
            or not isinstance(metadata.get("size"), int)
            or metadata["size"] < 0
            or not isinstance(metadata.get("sha256"), str)
            or len(metadata["sha256"]) != 64
        ):
            raise BackupError("The backup manifest contains invalid file metadata.")
    return manifest


def export_profile(
    store: LocalStore,
    destination: Path,
    bundled_voice_dir: Path | None = None,
) -> dict[str, int]:
    if not store.database_path.is_file():
        raise BackupError("The local database is not available for backup.")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="kalapaktts-backup-", dir=store.root) as temporary:
        temporary_root = Path(temporary)
        snapshot_path = temporary_root / "kalapaktts.sqlite3"
        with closing(store._connect()) as source, closing(
            sqlite3.connect(snapshot_path)
        ) as snapshot:
            source.backup(snapshot)

        audio_manifest: dict[str, str | None] = {}
        missing_audio = 0
        with closing(sqlite3.connect(snapshot_path)) as snapshot:
            rows = snapshot.execute(
                "SELECT id, audio_path FROM generations ORDER BY id"
            ).fetchall()
            for generation_id, audio_path_value in rows:
                audio_path = Path(audio_path_value) if audio_path_value else None
                if audio_path is None or not audio_path.is_file():
                    audio_manifest[str(generation_id)] = None
                    missing_audio += 1
                    snapshot.execute(
                        "UPDATE generations SET audio_path = NULL WHERE id = ?",
                        (generation_id,),
                    )
                    continue
                archive_name = (
                    f"audio/{hashlib.sha256(str(generation_id).encode('utf-8')).hexdigest()}.wav"
                )
                audio_manifest[str(generation_id)] = archive_name
                snapshot.execute(
                    "UPDATE generations SET audio_path = ? WHERE id = ?",
                    (f"__backup__/{archive_name}", generation_id),
                )
            snapshot.commit()

        file_manifest: dict[str, dict[str, Any]] = {}
        audio_count = 0
        with zipfile.ZipFile(
            destination,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            file_manifest[DATABASE_ENTRY] = _write_file(
                archive,
                DATABASE_ENTRY,
                snapshot_path,
            )
            for archive_name in sorted(
                {name for name in audio_manifest.values() if name is not None}
            ):
                generation_id = next(
                    key for key, value in audio_manifest.items() if value == archive_name
                )
                with store._reader() as connection:
                    row = connection.execute(
                        "SELECT audio_path, audio_sha256 FROM generations WHERE id = ?",
                        (generation_id,),
                    ).fetchone()
                if row is None or not row["audio_path"]:
                    raise BackupError("History changed while the backup was being created.")
                source_path = Path(row["audio_path"])
                if not source_path.is_file():
                    raise BackupError(
                        f"Audio for generation {generation_id} disappeared during backup."
                    )
                file_manifest[archive_name] = _write_file(
                    archive,
                    archive_name,
                    source_path,
                )
                expected_hash = row["audio_sha256"]
                if (
                    expected_hash
                    and file_manifest[archive_name]["sha256"] != expected_hash
                ):
                    raise BackupError(
                        f"Audio checksum does not match history item {generation_id}."
                    )
                audio_count += 1

            voice_sources: list[tuple[str, Path]] = []
            if store.voice_dir.is_dir():
                voice_sources.extend(
                    (
                        f"voices/custom/{path.relative_to(store.voice_dir).as_posix()}",
                        path,
                    )
                    for path in store.voice_dir.rglob("*")
                    if path.is_file() and not path.is_symlink()
                )
            if bundled_voice_dir is not None and bundled_voice_dir.is_dir():
                voice_sources.extend(
                    (f"voices/presets/{path.name}", path)
                    for path in bundled_voice_dir.iterdir()
                    if path.is_file() and not path.is_symlink()
                )

            voice_hashes: dict[str, str] = {}
            for archive_name, source_path in sorted(voice_sources):
                _safe_archive_path(archive_name)
                if archive_name in file_manifest:
                    raise BackupError("Voice files map to a duplicate backup path.")
                digest = _sha256_file(source_path)
                previous_digest = voice_hashes.get(archive_name)
                if previous_digest is not None:
                    if previous_digest != digest:
                        raise BackupError("Different voice files use the same backup path.")
                    continue
                voice_hashes[archive_name] = digest
                file_manifest[archive_name] = _write_file(
                    archive,
                    archive_name,
                    source_path,
                )

            manifest = {
                "format": BACKUP_FORMAT,
                "version": BACKUP_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "files": file_manifest,
                "audio": audio_manifest,
            }
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
            )
    return {
        "files": len(file_manifest),
        "audio_files": audio_count,
        "missing_audio": missing_audio,
    }


def _verify_and_extract(
    archive_path: Path,
    destination: Path,
) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            manifest = _archive_manifest(archive)
            total_written = 0
            for name, metadata in manifest["files"].items():
                archive_path_value = _safe_archive_path(name)
                output_path = destination.joinpath(*archive_path_value.parts)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size = 0
                with archive.open(name, "r") as source, output_path.open("xb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        size += len(block)
                        total_written += len(block)
                        if total_written > MAX_BACKUP_BYTES:
                            raise BackupError(
                                "The backup exceeds the 50 GiB expanded-size limit."
                            )
                        output.write(block)
                        digest.update(block)
                if size != metadata["size"] or digest.hexdigest() != metadata["sha256"]:
                    raise BackupError(f"Checksum verification failed for {name}.")
            return manifest
    except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
        if isinstance(exc, BackupError):
            raise
        raise BackupError(f"Could not read the backup archive: {exc}") from exc


def _validate_database(
    database_path: Path,
    audio_manifest: dict[str, Any],
    file_manifest: dict[str, dict[str, Any]],
    audio_root: Path,
) -> int:
    try:
        connection = sqlite3.connect(database_path)
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if integrity is None or integrity[0] != "ok":
                raise BackupError("The database inside the backup failed its integrity check.")
            version_row = connection.execute("PRAGMA user_version").fetchone()
            version = int(version_row[0]) if version_row else 0
            if version > SCHEMA_VERSION:
                raise BackupError(
                    f"Backup database version {version} is newer than supported version "
                    f"{SCHEMA_VERSION}."
                )
            LocalStore._migrate(connection, version)
            table_names = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if not {"generations", "drafts", "settings"}.issubset(table_names):
                raise BackupError("The backup database is missing required local data tables.")

            generation_ids = {
                str(row[0])
                for row in connection.execute("SELECT id FROM generations")
            }
            for generation_id, archive_name in audio_manifest.items():
                if not isinstance(generation_id, str) or generation_id not in generation_ids:
                    raise BackupError("The backup audio map refers to an unknown history item.")
                if archive_name is not None and (
                    not isinstance(archive_name, str)
                    or not archive_name.startswith("audio/")
                    or archive_name not in file_manifest
                    or len(PurePosixPath(archive_name).parts) != 2
                    or PurePosixPath(archive_name).suffix != ".wav"
                ):
                    raise BackupError("The backup audio map contains an invalid file reference.")
                row = connection.execute(
                    "SELECT audio_sha256 FROM generations WHERE id = ?",
                    (generation_id,),
                ).fetchone()
                if (
                    row is not None
                    and row[0]
                    and row[0] != file_manifest[archive_name]["sha256"]
                ):
                    raise BackupError(
                        f"Audio checksum does not match history item {generation_id}."
                    )

            for generation_id in generation_ids:
                archive_name = audio_manifest.get(generation_id)
                audio_path = (
                    str(audio_root / PurePosixPath(archive_name).name)
                    if isinstance(archive_name, str)
                    else None
                )
                connection.execute(
                    "UPDATE generations SET audio_path = ? WHERE id = ?",
                    (audio_path, generation_id),
                )

            media_folder_value = json.dumps(str(audio_root), ensure_ascii=False)
            connection.execute(
                """
                INSERT INTO settings (key, value_json, updated_at)
                VALUES ('media_folder', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_at=excluded.updated_at
                """,
                (media_folder_value, datetime.now(timezone.utc).isoformat()),
            )
            connection.commit()
            return len(generation_ids)
        finally:
            connection.close()
    except (sqlite3.DatabaseError, StorageError) as exc:
        raise BackupError(f"The database inside the backup is invalid: {exc}") from exc


def _copy_verified_tree(
    source_root: Path,
    destination_root: Path,
    copied_paths: list[Path],
) -> int:
    if not source_root.is_dir():
        return 0
    destination_root.mkdir(parents=True, exist_ok=True)
    safe_root = destination_root.resolve()
    copied = 0
    for source_path in source_root.rglob("*"):
        if source_path.is_symlink():
            raise BackupError("Symbolic links are not supported in backups.")
        if not source_path.is_file():
            continue
        relative = source_path.relative_to(source_root)
        target_path = destination_root / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.is_symlink() or not target_path.resolve().is_relative_to(
            safe_root
        ):
            raise BackupError("The restore destination contains an unsafe symbolic link.")
        if target_path.exists():
            if _sha256_file(target_path) != _sha256_file(source_path):
                raise BackupError(
                    f"Import stopped because {target_path.name} already exists with different contents."
                )
            continue
        temporary_path = target_path.with_name(
            f".{target_path.name}.{uuid.uuid4().hex}.importing"
        )
        try:
            with source_path.open("rb") as source, temporary_path.open("xb") as target:
                shutil.copyfileobj(source, target)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary_path, target_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        copied_paths.append(target_path)
        copied += 1
    return copied


def import_profile(
    store: LocalStore,
    archive_path: Path,
    *,
    create_pre_import_backup: bool = True,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="kalapaktts-restore-", dir=store.root) as temporary:
        staging = Path(temporary)
        manifest = _verify_and_extract(archive_path, staging)
        audio_map = dict(manifest["audio"])
        staged_database = staging / DATABASE_ENTRY
        staged_audio = staging / "audio"
        staged_voices = staging / "voices" / "custom"
        staged_presets = staging / "voices" / "presets"
        restored_audio_root = store.root / "audio"
        generation_count = _validate_database(
            staged_database,
            audio_map,
            manifest["files"],
            restored_audio_root,
        )

        backup_path: Path | None = None
        if store.database_path.is_file() and create_pre_import_backup:
            backup_path = store.backup_dir / (
                "pre-import-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".zip"
            )
            export_profile(store, backup_path)

        original_database = staging / "pre-import.sqlite3"
        had_original_database = store.database_path.is_file()
        if had_original_database:
            if create_pre_import_backup:
                with closing(store._connect()) as source, closing(
                    sqlite3.connect(original_database)
                ) as snapshot:
                    source.execute("PRAGMA wal_checkpoint(FULL)")
                    source.backup(snapshot)
            else:
                shutil.copy2(store.database_path, original_database)
        copied_paths: list[Path] = []
        try:
            audio_count = _copy_verified_tree(
                staged_audio,
                restored_audio_root,
                copied_paths,
            )
            _copy_verified_tree(staged_voices, store.voice_dir, copied_paths)
            _copy_verified_tree(
                staged_presets,
                store.voice_dir / "presets",
                copied_paths,
            )
            store.database_path.parent.mkdir(parents=True, exist_ok=True)
            Path(f"{store.database_path}-wal").unlink(missing_ok=True)
            Path(f"{store.database_path}-shm").unlink(missing_ok=True)
            os.replace(staged_database, store.database_path)
            store.audio_dir = restored_audio_root
            store.initialize()
        except Exception:
            Path(f"{store.database_path}-wal").unlink(missing_ok=True)
            Path(f"{store.database_path}-shm").unlink(missing_ok=True)
            if had_original_database:
                os.replace(original_database, store.database_path)
            else:
                store.database_path.unlink(missing_ok=True)
            for copied_path in copied_paths:
                copied_path.unlink(missing_ok=True)
            raise

    return {
        "generation_count": generation_count,
        "audio_files": audio_count,
        "missing_audio": sum(value is None for value in manifest["audio"].values()),
        "pre_import_backup": backup_path.name if backup_path else None,
    }
