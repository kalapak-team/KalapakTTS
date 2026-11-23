from __future__ import annotations

import ctypes
import logging
import os
import secrets
import socket
import sqlite3
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PYTHON = ROOT / "runtime" / "python" / "python.exe"
RUNTIME_ROOT = RUNTIME_PYTHON.parent
RUNTIME_BIN = ROOT / "runtime" / "bin"
INSTANCE_MUTEX_NAME = "Local\\KalapakTTS.Desktop"
ERROR_ALREADY_EXISTS = 183


def _profile_models_root() -> Path:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        appdata = str(Path.home() / "AppData" / "Roaming")
    return Path(appdata) / "KalapakTTS" / "models"


def _select_model_root(desktop_root: Path, profile_models_root: Path) -> Path:
    installed_manifest = profile_models_root / "VoxCPM2" / "manifest.json"
    development_manifest = desktop_root / "models" / "VoxCPM2" / "manifest.json"
    if installed_manifest.is_file() or not development_manifest.is_file():
        return profile_models_root
    return desktop_root / "models"


if __name__ == "__main__":
    if (
        Path(sys.executable).resolve().parent != RUNTIME_PYTHON.resolve().parent
        or Path(sys.executable).name.lower() not in {"python.exe", "pythonw.exe"}
    ):
        raise RuntimeError(
            "The desktop app must use its portable Python runtime. "
            "Launch it with .\\app\\run_desktop.ps1."
        )

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PATH"] = str(RUNTIME_BIN) + os.pathsep + os.environ.get("PATH", "")
    os.environ["KALAPAK_TTS_MODELS"] = str(_select_model_root(ROOT, _profile_models_root()))

    filtered_sys_path = []
    for entry in sys.path:
        if not entry:
            continue
        resolved_entry = Path(entry).resolve()
        if resolved_entry == ROOT.resolve() or resolved_entry.is_relative_to(RUNTIME_ROOT.resolve()):
            filtered_sys_path.append(str(resolved_entry))
    sys.path[:] = filtered_sys_path
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from server.storage import DatabaseRecoveryRequired, LocalStore, StorageError, default_data_root
from server.main import create_app


def _read_window_size(store: LocalStore) -> tuple[int, int]:
    saved_size = store.get_setting("window_size")
    if not isinstance(saved_size, dict):
        return 960, 680
    try:
        width = int(saved_size["width"])
        height = int(saved_size["height"])
    except (KeyError, TypeError, ValueError):
        return 960, 680
    return max(640, min(width, 2560)), max(480, min(height, 2160))


def _write_window_size(store: LocalStore, size: dict[str, int]) -> None:
    store.save_setting("window_size", size)


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_server(
    url: str,
    server_thread: threading.Thread,
    api_token: str,
    expected_host: str,
) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if not server_thread.is_alive():
            raise RuntimeError("The desktop API server stopped before it became ready.")
        try:
            request = Request(
                url,
                headers={"Host": expected_host, "X-Desktop-Token": api_token},
            )
            with urlopen(request, timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            time.sleep(0.1)
    raise TimeoutError("The desktop API server did not become ready within 15 seconds.")


def _claim_single_instance_mutex(
    create_mutex: Callable[..., Any],
    close_handle: Callable[[Any], Any],
    get_last_error: Callable[[], int],
) -> Any | None:
    handle = create_mutex(None, False, INSTANCE_MUTEX_NAME)
    if not handle:
        raise OSError(get_last_error(), "Could not create the desktop single-instance mutex.")
    if get_last_error() == ERROR_ALREADY_EXISTS:
        close_handle(handle)
        return None
    return handle


def _acquire_single_instance() -> tuple[Any, Any] | None:
    if os.name != "nt":
        raise RuntimeError("The desktop application requires Windows.")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_mutex = kernel32.CreateMutexW
    create_mutex.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    create_mutex.restype = ctypes.c_void_p
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle.restype = ctypes.c_int
    ctypes.set_last_error(0)
    handle = _claim_single_instance_mutex(
        create_mutex,
        close_handle,
        ctypes.get_last_error,
    )
    return (kernel32, handle) if handle is not None else None


def _show_message(title: str, message: str) -> None:
    if os.name == "nt":
        ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)
    else:
        logging.getLogger(__name__).error("%s: %s", title, message)


def main() -> None:
    log_path = default_data_root() / "logs" / "desktop.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            RotatingFileHandler(
                log_path,
                maxBytes=5 * 1024 * 1024,
                backupCount=4,
                encoding="utf-8",
            )
        ],
    )
    instance: tuple[Any, Any] | None = None
    try:
        instance = _acquire_single_instance()
        if instance is None:
            _show_message(
                "KalapakTTS Desktop is already running",
                "Close the existing KalapakTTS Desktop window before opening another.",
            )
            return
        kernel32, mutex_handle = instance
        _run_desktop(log_path)
    except Exception:
        logging.getLogger(__name__).exception("KalapakTTS Desktop stopped unexpectedly.")
        _show_message(
            "KalapakTTS Desktop stopped",
            f"The application stopped unexpectedly. Diagnostic log:\n{log_path}",
        )
        raise
    finally:
        if instance is not None:
            kernel32, mutex_handle = instance
            if not kernel32.CloseHandle(mutex_handle):
                logging.getLogger(__name__).error(
                    "Could not release the desktop single-instance mutex (Windows error %d).",
                    ctypes.get_last_error(),
                )


def _run_desktop(log_path: Path) -> None:
    startup_started = time.perf_counter()
    window_store = LocalStore(default_data_root())
    window_store_ready = True
    try:
        window_store.initialize()
    except DatabaseRecoveryRequired as exc:
        window_store_ready = False
        logging.getLogger(__name__).warning(
            "Using default window dimensions because local database recovery is required: %s",
            exc,
        )
    width, height = _read_window_size(window_store) if window_store_ready else (960, 680)
    try:
        import uvicorn
        import webview
    except ImportError as exc:
        raise RuntimeError(
            "Install the DesktopApplication POC dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc

    os.environ["WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"] = "--disable-gpu"
    port = _free_loopback_port()
    expected_host = f"127.0.0.1:{port}"
    api_token = secrets.token_urlsafe(32)
    app = create_app(api_token=api_token, expected_host=expected_host)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
        )
    )
    server_thread = threading.Thread(target=server.run, name="desktop-api", daemon=True)
    server_thread.start()
    base_url = f"http://127.0.0.1:{port}"
    _wait_for_server(f"{base_url}/health", server_thread, api_token, expected_host)
    print(
        f"Desktop API ready in {time.perf_counter() - startup_started:.2f}s at {base_url}",
        flush=True,
    )

    resize_lock = threading.Lock()
    resize_timer: threading.Timer | None = None
    window_size = {"width": width, "height": height}

    def persist_window_size() -> None:
        with resize_lock:
            current_size = dict(window_size)
        if not window_store_ready:
            return
        try:
            _write_window_size(window_store, current_size)
        except (OSError, sqlite3.DatabaseError, StorageError, ValueError):
            logging.getLogger(__name__).exception("Could not save the desktop window size.")

    def schedule_window_size_save(new_width: int, new_height: int) -> None:
        nonlocal resize_timer
        with resize_lock:
            window_size["width"] = max(640, min(int(new_width), 2560))
            window_size["height"] = max(480, min(int(new_height), 2160))
            if resize_timer is not None:
                resize_timer.cancel()
            resize_timer = threading.Timer(0.75, persist_window_size)
            resize_timer.daemon = True
            resize_timer.start()

    try:
        window = webview.create_window(
            "KalapakTTS Desktop",
            f"{base_url}/#token={api_token}",
            width=width,
            height=height,
            min_size=(640, 480),
        )
        window.events.resized += schedule_window_size_save
        webview.start(gui="edgechromium", debug=False)
    finally:
        if resize_timer is not None:
            resize_timer.cancel()
        persist_window_size()
        server.should_exit = True
        server_thread.join(timeout=10)
        if server_thread.is_alive():
            raise RuntimeError("The desktop API server did not shut down cleanly.")


if __name__ == "__main__":
    main()
