import errno
import io
import inspect
import tempfile
import unittest
from pathlib import Path
from urllib.request import Request
from unittest.mock import Mock

from app.native_downloads import (
    EXPORT_FORMATS,
    _NoRedirectHandler,
    NativeDownloadApi,
    ensure_selected_extension,
    sanitize_suggested_filename,
)


class FakeWindow:
    def __init__(self, selected_path: str | None) -> None:
        self.selected_path = selected_path
        self.dialog_calls: list[tuple[int, str, tuple[str, ...]]] = []

    def create_file_dialog(
        self,
        dialog_type: int,
        *,
        save_filename: str,
        file_types: tuple[str, ...],
    ) -> tuple[str, ...] | None:
        self.dialog_calls.append((dialog_type, save_filename, file_types))
        return (self.selected_path,) if self.selected_path is not None else None


class FileInUseError(OSError):
    winerror = 32


def make_api(
    window: FakeWindow,
    opener: Mock,
    *,
    stream_writer=None,
) -> NativeDownloadApi:
    return NativeDownloadApi(
        "http://127.0.0.1:49123",
        "test-per-launch-token",
        "127.0.0.1:49123",
        lambda: window,
        30,
        opener=opener,
        stream_writer=stream_writer,
    )


class NativeDownloadTests(unittest.TestCase):
    def test_suggested_filename_is_sanitized_and_reserved_names_are_rewritten(self) -> None:
        self.assertEqual(
            sanitize_suggested_filename("../../bad\r\nname.wav", "wav"),
            ".._.._bad__name.wav",
        )
        self.assertEqual(sanitize_suggested_filename("CON", "mp3"), "_CON.mp3")
        self.assertEqual(sanitize_suggested_filename("COM².txt", "wav"), "_COM².txt.wav")
        self.assertEqual(sanitize_suggested_filename("speech.mp3", "mp3"), "speech.mp3")
        with self.assertRaises(ValueError):
            sanitize_suggested_filename("...", "wav")

    def test_every_supported_audio_and_subtitle_format_has_its_extension(self) -> None:
        expected_extensions = {
            "mp3": ".mp3",
            "wav": ".wav",
            "flac": ".flac",
            "m4a": ".m4a",
            "ogg": ".ogg",
            "srt": ".srt",
            "vtt": ".vtt",
            "zip": ".zip",
        }
        self.assertEqual(
            {name: details[0] for name, details in EXPORT_FORMATS.items()},
            expected_extensions,
        )
        for file_format, extension in expected_extensions.items():
            with self.subTest(file_format=file_format):
                self.assertEqual(
                    sanitize_suggested_filename("speech", file_format),
                    f"speech{extension}",
                )

    def test_name_entered_without_extension_gets_the_selected_format_extension(self) -> None:
        self.assertEqual(
            ensure_selected_extension(r"C:\Music\my-speech", "wav"),
            r"C:\Music\my-speech.wav",
        )
        self.assertEqual(
            ensure_selected_extension(r"C:\Music\my-speech.txt", "mp3"),
            r"C:\Music\my-speech.mp3",
        )

    def test_native_save_uses_the_dialog_path_and_streams_authenticated_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            selected_path = str(Path(temporary_directory) / "my-speech")
            window = FakeWindow(selected_path)
            response = io.BytesIO(b"audio bytes")
            opener = Mock(return_value=response)
            api = make_api(window, opener)

            result = api.save_audio("generation-1", "wav", "speech")

            saved_path = Path(temporary_directory) / "my-speech.wav"
            self.assertEqual(result, {"status": "saved", "path": str(saved_path)})
            self.assertEqual(saved_path.read_bytes(), b"audio bytes")
            self.assertEqual(window.dialog_calls, [(30, "speech.wav", ("WAV audio (*.wav)",))])
            request = opener.call_args.args[0]
            self.assertEqual(request.full_url, "http://127.0.0.1:49123/api/tts/generations/generation-1/download?format=wav")
            self.assertEqual(request.get_header("Host"), "127.0.0.1:49123")
            self.assertEqual(request.get_header("X-desktop-token"), "test-per-launch-token")

    def test_cancelled_dialog_does_not_request_or_write_an_export(self) -> None:
        window = FakeWindow(None)
        opener = Mock()
        api = make_api(window, opener)

        result = api.save_audio("generation-1", "mp3", "speech")

        self.assertEqual(result, {"status": "cancelled"})
        opener.assert_not_called()

    def test_disk_write_failure_is_returned_as_a_clear_error(self) -> None:
        def fail_write(_response, _selected_path) -> None:
            raise OSError(errno.ENOSPC, "disk full")

        with tempfile.TemporaryDirectory() as temporary_directory:
            window = FakeWindow(str(Path(temporary_directory) / "speech.wav"))
            api = make_api(window, Mock(return_value=io.BytesIO(b"audio")), stream_writer=fail_write)

            with self.assertLogs("app.native_downloads", level="ERROR"):
                result = api.save_audio("generation-1", "wav", "speech")

        self.assertEqual(result, {"status": "error", "code": "disk_full"})

    def test_file_in_use_is_reported_separately_from_permission_failure(self) -> None:
        def fail_write(_response, _selected_path) -> None:
            raise FileInUseError(errno.EACCES, "file is open")

        with tempfile.TemporaryDirectory() as temporary_directory:
            window = FakeWindow(str(Path(temporary_directory) / "speech.wav"))
            api = make_api(window, Mock(return_value=io.BytesIO(b"audio")), stream_writer=fail_write)

            with self.assertLogs("app.native_downloads", level="ERROR"):
                result = api.save_audio("generation-1", "wav", "speech")

        self.assertEqual(result, {"status": "error", "code": "file_in_use"})

    def test_js_api_does_not_accept_a_destination_path_from_the_page(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            untrusted_path = str(Path(temporary_directory) / "attacker.wav")
            window = FakeWindow(untrusted_path)
            api = make_api(window, Mock(return_value=io.BytesIO(b"audio")))

            self.assertEqual(
                tuple(inspect.signature(api.save_audio).parameters),
                ("generation_id", "audio_format", "suggested_name"),
            )
            with self.assertRaises(TypeError):
                api.save_audio("generation-1", "wav", "speech", untrusted_path)
            self.assertEqual(window.dialog_calls, [])
            self.assertFalse(Path(untrusted_path).exists())

    def test_non_loopback_server_urls_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback API"):
            NativeDownloadApi(
                "https://example.com",
                "token",
                "example.com",
                lambda: None,
                30,
            )

    def test_api_host_must_match_the_loopback_server_host(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback API"):
            NativeDownloadApi(
                "http://127.0.0.1:49123",
                "token",
                "127.0.0.1:49124",
                lambda: None,
                30,
            )

    def test_external_redirects_are_not_followed(self) -> None:
        handler = _NoRedirectHandler()

        self.assertIsNone(
            handler.redirect_request(
                Request("http://127.0.0.1:49123/export"),
                None,
                302,
                "Found",
                {},
                "https://example.com/",
            )
        )


if __name__ == "__main__":
    unittest.main()
