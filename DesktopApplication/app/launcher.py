from __future__ import annotations

import logging
import os
import socket
import sqlite3
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PYTHON = ROOT / "runtime" / "python" / "python.exe"
RUNTIME_ROOT = RUNTIME_PYTHON.parent
RUNTIME_BIN = ROOT / "runtime" / "bin"

if Path(sys.executable).resolve() != RUNTIME_PYTHON.resolve():
    raise RuntimeError(
        "The desktop app must use its portable Python runtime. "
        "Launch it with .\\app\\run_desktop.ps1."
    )

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PATH"] = str(RUNTIME_BIN) + os.pathsep + os.environ.get("PATH", "")

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
from server.main import app


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


def _wait_for_server(url: str, server_thread: threading.Thread) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if not server_thread.is_alive():
            raise RuntimeError("The desktop API server stopped before it became ready.")
        try:
            with urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            time.sleep(0.1)
    raise TimeoutError("The desktop API server did not become ready within 15 seconds.")


def main() -> None:
    startup_started = time.perf_counter()
    log_path = default_data_root() / "logs" / "desktop.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
    )
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
    _wait_for_server(f"{base_url}/health", server_thread)
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
            f"{base_url}/",
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
