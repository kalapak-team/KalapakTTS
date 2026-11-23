from __future__ import annotations

import errno
import json
import logging
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any, BinaryIO, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


EXPORT_FORMATS: dict[str, tuple[str, str]] = {
    "mp3": (".mp3", "MP3 audio (*.mp3)"),
    "wav": (".wav", "WAV audio (*.wav)"),
    "flac": (".flac", "FLAC audio (*.flac)"),
    "m4a": (".m4a", "M4A audio (*.m4a)"),
    "ogg": (".ogg", "OGG audio (*.ogg)"),
    "srt": (".srt", "SubRip subtitles (*.srt)"),
    "vtt": (".vtt", "WebVTT subtitles (*.vtt)"),
    "zip": (".zip", "ZIP backup (*.zip)"),
}
INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
KNOWN_EXPORT_SUFFIX = re.compile(r"\.(?:mp3|wav|flac|m4a|ogg|srt|vtt|zip)$", re.IGNORECASE)
RESERVED_WINDOWS_NAME = re.compile(
    r"^(?:CON|PRN|AUX|NUL|COM[1-9\u00b9\u00b2\u00b3]|LPT[1-9\u00b9\u00b2\u00b3])(?:\.|$)",
    re.IGNORECASE,
)
logger = logging.getLogger(__name__)


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        request: Request,
        response: Any,
        code: int,
        message: str,
        headers: Any,
        new_url: str,
    ) -> None:
        return None


def sanitize_suggested_filename(file_name: str, file_format: str) -> str:
    format_details = EXPORT_FORMATS.get(file_format)
    if format_details is None:
        raise ValueError("Unsupported export format.")
    if not isinstance(file_name, str):
        raise ValueError("A suggested file name is required.")

    extension, _ = format_details
    stem = KNOWN_EXPORT_SUFFIX.sub("", file_name.strip())
    stem = INVALID_FILENAME_CHARS.sub("_", unicodedata.normalize("NFC", stem))
    stem = stem.rstrip(". ")[:120].rstrip(". ")
    if not stem:
        raise ValueError("A valid file name is required.")
    if RESERVED_WINDOWS_NAME.match(stem):
        stem = f"_{stem}"
    return f"{stem}{extension}"


def ensure_selected_extension(file_path: str, file_format: str) -> str:
    format_details = EXPORT_FORMATS.get(file_format)
    if format_details is None:
        raise ValueError("Unsupported export format.")
    extension, _ = format_details
    path = Path(file_path)
    path = path.with_name(path.name.rstrip(". "))
    if path.suffix.lower() == extension:
        return str(path)
    if path.suffix:
        return str(path.with_suffix(extension))
    return f"{path}{extension}"


class NativeDownloadApi:
    def __init__(
        self,
        base_url: str,
        api_token: str,
        expected_host: str,
        window_getter: Callable[[], Any],
        save_dialog_type: int,
        *,
        opener: Callable[..., Any] | None = None,
        stream_writer: Callable[[BinaryIO, Path], None] | None = None,
    ) -> None:
        parsed_url = urlsplit(base_url)
        if (
            parsed_url.scheme != "http"
            or parsed_url.hostname != "127.0.0.1"
            or parsed_url.path not in {"", "/"}
            or parsed_url.netloc.lower() != expected_host.lower()
        ):
            raise ValueError("The native save bridge only accepts the loopback API.")
        if not api_token or not expected_host:
            raise ValueError("The native save bridge requires the per-launch API credentials.")
        self._base_url = base_url.rstrip("/")
        self._api_token = api_token
        self._expected_host = expected_host
        self._window_getter = window_getter
        self._save_dialog_type = save_dialog_type
        self._opener = opener or build_opener(_NoRedirectHandler()).open
        self._stream_writer = stream_writer or self._write_response_stream

    def save_audio(self, generation_id: str, audio_format: str, suggested_name: str) -> dict[str, str]:
        if not isinstance(generation_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", generation_id):
            return {"status": "error", "code": "invalid_request"}
        if not isinstance(audio_format, str) or audio_format not in {"mp3", "wav", "flac", "m4a", "ogg"}:
            return {"status": "error", "code": "invalid_request"}
        endpoint = (
            f"/api/tts/generations/{quote(generation_id, safe='')}/download"
            f"?format={quote(audio_format, safe='')}"
        )
        return self._save_export(endpoint, audio_format, suggested_name)

    def save_subtitles(
        self,
        generation_id: str,
        subtitle_format: str,
        suggested_name: str,
    ) -> dict[str, str]:
        if not isinstance(generation_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", generation_id):
            return {"status": "error", "code": "invalid_request"}
        if not isinstance(subtitle_format, str) or subtitle_format not in {"srt", "vtt"}:
            return {"status": "error", "code": "invalid_request"}
        endpoint = (
            f"/api/tts/generations/{quote(generation_id, safe='')}"
            f"/subtitles/{subtitle_format}"
        )
        return self._save_export(endpoint, subtitle_format, suggested_name)

    def save_backup(self, suggested_name: str) -> dict[str, str]:
        return self._save_export("/api/storage/export", "zip", suggested_name)

    def _save_export(
        self,
        endpoint: str,
        file_format: str,
        suggested_name: str,
    ) -> dict[str, str]:
        try:
            file_name = sanitize_suggested_filename(suggested_name, file_format)
        except ValueError:
            return {"status": "error", "code": "invalid_request"}

        window = self._window_getter()
        if window is None:
            return {"status": "error", "code": "dialog_unavailable"}
        _, file_filter = EXPORT_FORMATS[file_format]
        try:
            selected = window.create_file_dialog(
                self._save_dialog_type,
                save_filename=file_name,
                file_types=(file_filter,),
            )
        except (OSError, RuntimeError):
            logger.exception("Could not open the native save dialog.")
            return {"status": "error", "code": "dialog_failed"}
        if not selected:
            return {"status": "cancelled"}
        if (
            not isinstance(selected, (tuple, list))
            or not selected
            or not isinstance(selected[0], str)
            or not selected[0]
        ):
            return {"status": "error", "code": "dialog_failed"}

        try:
            selected_path = Path(ensure_selected_extension(selected[0], file_format))
        except ValueError:
            return {"status": "error", "code": "dialog_failed"}
        if not selected_path.is_absolute():
            return {"status": "error", "code": "dialog_failed"}
        existing_file = selected_path.exists()
        request = Request(
            f"{self._base_url}{endpoint}",
            headers={
                "Host": self._expected_host,
                "X-Desktop-Token": self._api_token,
            },
        )
        try:
            with self._opener(request, timeout=600) as response:
                self._stream_writer(response, selected_path)
        except HTTPError as error:
            detail = self._read_http_error(error)
            logger.error("Native export request failed with HTTP %s: %s", error.code, detail)
            return {"status": "error", "code": "export_failed"}
        except URLError:
            logger.exception("The local API could not provide the requested export.")
            return {"status": "error", "code": "export_failed"}
        except OSError as error:
            if not existing_file:
                try:
                    selected_path.unlink(missing_ok=True)
                except OSError:
                    logger.exception("Could not remove a partial native export.")
            if getattr(error, "winerror", None) in {32, 33}:
                code = "file_in_use"
            elif isinstance(error, PermissionError) or error.errno in {errno.EACCES, errno.EPERM}:
                code = "permission_denied"
            elif error.errno == errno.ENOSPC:
                code = "disk_full"
            else:
                code = "write_failed"
            logger.exception("Could not write native export to the selected path.")
            return {"status": "error", "code": code}

        return {"status": "saved", "path": str(selected_path)}

    @staticmethod
    def _write_response_stream(response: BinaryIO, selected_path: Path) -> None:
        with selected_path.open("wb") as output_file:
            shutil.copyfileobj(response, output_file, length=1024 * 1024)

    @staticmethod
    def _read_http_error(error: HTTPError) -> str:
        try:
            payload = json.loads(error.read(65536).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return "The local export request failed."
        if not isinstance(payload, dict):
            return "The local export request failed."
        detail = payload.get("detail")
        return detail if isinstance(detail, str) else "The local export request failed."
