from __future__ import annotations

import csv
import io
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any


MIN_FREE_VRAM_MIB = 1536


def _unavailable_status(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "gpu_name": None,
        "free_mib": None,
        "total_mib": None,
        "minimum_free_mib": MIN_FREE_VRAM_MIB,
        "safe_to_load_model": False,
        "warning": (
            f"{reason} Desktop speech generation is blocked until NVIDIA memory can be checked. "
            "Stop the web TTS server/model before using the desktop engine."
        ),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def read_gpu_status() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return _unavailable_status("nvidia-smi was not found.")

    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return _unavailable_status("nvidia-smi could not be started.")
    except subprocess.TimeoutExpired:
        return _unavailable_status("nvidia-smi did not respond within 5 seconds.")
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip() or f"exit code {exc.returncode}"
        return _unavailable_status(f"nvidia-smi failed: {detail}")
    except OSError as exc:
        return _unavailable_status(f"nvidia-smi could not be run: {exc}")

    rows = list(csv.reader(io.StringIO(result.stdout)))
    if not rows or len(rows[0]) < 3:
        return _unavailable_status("nvidia-smi returned no usable GPU memory values.")

    gpu_name, free_text, total_text = (value.strip() for value in rows[0][:3])
    try:
        free_mib = int(free_text)
        total_mib = int(total_text)
    except ValueError:
        return _unavailable_status("nvidia-smi returned invalid GPU memory values.")

    safe_to_load_model = free_mib >= MIN_FREE_VRAM_MIB
    if safe_to_load_model:
        warning = (
            f"{free_mib} MiB of {total_mib} MiB is free. Before using desktop speech generation, "
            "verify that the web TTS server and its model are stopped; free memory alone cannot "
            "identify which process owns a loaded model."
        )
    else:
        warning = (
            f"Only {free_mib} MiB of {total_mib} MiB is free, below the "
            f"{MIN_FREE_VRAM_MIB} MiB desktop startup threshold. The web TTS model may still be "
            "loaded. Stop the web TTS server/model; desktop speech generation remains blocked."
        )

    return {
        "available": True,
        "gpu_name": gpu_name,
        "free_mib": free_mib,
        "total_mib": total_mib,
        "minimum_free_mib": MIN_FREE_VRAM_MIB,
        "safe_to_load_model": safe_to_load_model,
        "warning": warning,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
