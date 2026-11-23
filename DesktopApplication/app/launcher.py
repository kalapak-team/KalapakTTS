from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.main import app


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
    _wait_for_server(f"http://127.0.0.1:{port}/health", server_thread)

    try:
        webview.create_window(
            "KalapakTTS Desktop",
            f"http://127.0.0.1:{port}/",
            width=960,
            height=680,
            min_size=(640, 480),
        )
        webview.start(gui="edgechromium", debug=False)
    finally:
        server.should_exit = True
        server_thread.join(timeout=10)
        if server_thread.is_alive():
            raise RuntimeError("The desktop API server did not shut down cleanly.")


if __name__ == "__main__":
    main()
