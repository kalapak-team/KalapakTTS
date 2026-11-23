from __future__ import annotations

import csv
import io
import logging
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)
MIB = 1024 * 1024
MEASURED_MODEL_PEAK_RESERVED_MIB: int | None = 5294
MEASURED_CUDA_CONTEXT_OVERHEAD_MIB: int | None = 72
MIN_FREE_VRAM_MIB: int | None = MEASURED_MODEL_PEAK_RESERVED_MIB
MINIMUM_NVIDIA_DRIVER_VERSION = "570.65"
_MINIMUM_NVIDIA_DRIVER_PARTS = (570, 65)
_observed_context_overhead_mib: int | None = None


def _unavailable_status(
    reason: str,
    nvidia_smi_free_mib: int | None = None,
    nvidia_smi_total_mib: int | None = None,
    gpu_name: str | None = None,
    driver_version: str | None = None,
    driver_compatible: bool | None = None,
) -> dict[str, Any]:
    return {
        "available": False,
        "cuda_available": False,
        "gpu_name": gpu_name,
        "driver_version": driver_version,
        "minimum_driver_version": MINIMUM_NVIDIA_DRIVER_VERSION,
        "driver_compatible": driver_compatible,
        "free_mib": None,
        "total_mib": nvidia_smi_total_mib,
        "nvidia_smi_free_mib": nvidia_smi_free_mib,
        "torch_free_before_context_mib": None,
        "torch_free_after_context_mib": None,
        "torch_context_init_delta_mib": None,
        "nvidia_smi_free_after_context_mib": None,
        "nvidia_context_init_delta_mib": None,
        "nvidia_torch_delta_mib": None,
        "context_overhead_mib": None,
        "model_peak_reserved_mib": MEASURED_MODEL_PEAK_RESERVED_MIB,
        "model_need_mib": None,
        "minimum_free_mib": None,
        "safe_to_load_model": False,
        "warning": (
            f"{reason} Minimum NVIDIA driver for CUDA 12.8: "
            f"{MINIMUM_NVIDIA_DRIVER_VERSION}. CUDA headroom could not be verified. "
            "CPU fallback is available; "
            "Try GPU anyway is disabled until CUDA memory can be measured."
        ),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def _read_cuda_memory() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "available": False,
            "error": "PyTorch is missing from the desktop runtime.",
        }

    if not torch.cuda.is_available():
        return {
            "available": False,
            "error": "CUDA is unavailable in this PyTorch runtime.",
        }

    initialized_before = bool(torch.cuda.is_initialized())
    free_before: int | None = None
    total_before: int | None = None
    first_read_error: str | None = None
    try:
        free_before, total_before = torch.cuda.mem_get_info()
    except RuntimeError as exc:
        first_read_error = str(exc)

    initialized_after_first_read = bool(torch.cuda.is_initialized())
    try:
        if not initialized_after_first_read:
            torch.cuda.init()
        free_after, total_after = torch.cuda.mem_get_info()
    except RuntimeError as exc:
        return {
            "available": False,
            "error": f"PyTorch could not read CUDA memory: {exc}",
            "initialized_before": initialized_before,
            "initialized_after_first_read": initialized_after_first_read,
            "free_before": free_before,
            "total_before": total_before,
            "first_read_error": first_read_error,
        }

    return {
        "available": True,
        "initialized_before": initialized_before,
        "initialized_after_first_read": initialized_after_first_read,
        "free_before": int(free_before) if free_before is not None else None,
        "total_before": int(total_before) if total_before is not None else None,
        "free_after": int(free_after),
        "total_after": int(total_after),
        "torch_context_init_delta_mib": (
            int(round((free_before - free_after) / MIB))
            if free_before is not None and not initialized_before
            else None
        ),
        "first_read_error": first_read_error,
    }


def _read_nvidia_smi_free_after_context(executable: str) -> int | None:
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
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Could not measure nvidia-smi free memory after CUDA context creation: %s", exc)
        return None

    rows = list(csv.reader(io.StringIO(result.stdout)))
    if not rows or len(rows[0]) < 3:
        logger.warning("nvidia-smi returned no usable post-context memory values.")
        return None
    try:
        return int(rows[0][1].strip())
    except ValueError:
        logger.warning("nvidia-smi returned an invalid post-context free-memory value.")
        return None


def _find_nvidia_smi() -> str | None:
    executable = shutil.which("nvidia-smi")
    if executable is not None:
        return executable
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    for candidate in (
        program_files / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
        system_root / "System32" / "nvidia-smi.exe",
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def read_gpu_status() -> dict[str, Any]:
    executable = _find_nvidia_smi()
    if executable is None:
        return _unavailable_status("nvidia-smi was not found.")

    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=name,driver_version,memory.free,memory.total",
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
    if not rows or len(rows[0]) < 4:
        return _unavailable_status("nvidia-smi returned no usable GPU memory values.")

    gpu_name, driver_version, free_text, total_text = (
        value.strip() for value in rows[0][:4]
    )
    version_parts = tuple(int(part) for part in re.findall(r"\d+", driver_version))
    if len(version_parts) < 2:
        return _unavailable_status(
            "The NVIDIA driver version could not be parsed. Update or repair the NVIDIA driver.",
            gpu_name=gpu_name,
            driver_version=driver_version,
            driver_compatible=False,
        )
    if version_parts[:2] < _MINIMUM_NVIDIA_DRIVER_PARTS:
        return _unavailable_status(
            f"NVIDIA driver {driver_version} is below the minimum "
            f"{MINIMUM_NVIDIA_DRIVER_VERSION} for the bundled CUDA 12.8 runtime. "
            "Update the NVIDIA driver or use CPU fallback.",
            gpu_name=gpu_name,
            driver_version=driver_version,
            driver_compatible=False,
        )
    try:
        nvidia_smi_free_mib = int(free_text)
        nvidia_smi_total_mib = int(total_text)
    except ValueError:
        return _unavailable_status("nvidia-smi returned invalid GPU memory values.")

    cuda_memory = _read_cuda_memory()
    if not cuda_memory["available"]:
        return _unavailable_status(
            cuda_memory["error"],
            nvidia_smi_free_mib=nvidia_smi_free_mib,
            nvidia_smi_total_mib=nvidia_smi_total_mib,
            gpu_name=gpu_name,
            driver_version=driver_version,
            driver_compatible=True,
        )

    global _observed_context_overhead_mib
    nvidia_smi_free_after_context_mib = (
        _read_nvidia_smi_free_after_context(executable)
        if not cuda_memory["initialized_before"]
        else None
    )
    nvidia_context_init_delta_mib = (
        nvidia_smi_free_mib - nvidia_smi_free_after_context_mib
        if nvidia_smi_free_after_context_mib is not None
        else None
    )
    torch_free_mib = cuda_memory["free_after"] // MIB
    torch_total_mib = cuda_memory["total_after"] // MIB
    nvidia_torch_delta_mib = nvidia_smi_free_mib - torch_free_mib
    if _observed_context_overhead_mib is None and nvidia_context_init_delta_mib is not None:
        _observed_context_overhead_mib = max(0, nvidia_context_init_delta_mib)
    context_overhead_mib = (
        MEASURED_CUDA_CONTEXT_OVERHEAD_MIB
        if MEASURED_CUDA_CONTEXT_OVERHEAD_MIB is not None
        else _observed_context_overhead_mib
    )
    model_peak_reserved_mib = MEASURED_MODEL_PEAK_RESERVED_MIB
    model_need_mib = (
        model_peak_reserved_mib + context_overhead_mib
        if model_peak_reserved_mib is not None and context_overhead_mib is not None
        else None
    )
    safe_to_load_model = bool(
        model_peak_reserved_mib is not None
        and model_need_mib is not None
        and nvidia_smi_free_mib >= model_need_mib
    )
    if safe_to_load_model:
        warning = (
            f"CUDA reports {torch_free_mib} MiB free after context creation; "
            f"nvidia-smi reports {nvidia_smi_free_mib} MiB free. The measured model need is "
            f"{model_need_mib} MiB, including {context_overhead_mib} MiB measured context overhead."
        )
        if nvidia_torch_delta_mib > 0:
            warning += (
                f" Torch and nvidia-smi differ by {nvidia_torch_delta_mib} MiB; "
                "the guard compares nvidia-smi free memory with the measured model need."
            )
    elif model_need_mib is None:
        if model_peak_reserved_mib is None:
            warning = (
                "The desktop model's peak reserved VRAM has not been measured yet. "
                "Choose CPU fallback or explicitly select Try GPU anyway for the first controlled load."
            )
        else:
            warning = (
                "CUDA context initialization overhead has not been measured yet. "
                "Choose CPU fallback or explicitly select Try GPU anyway."
            )
    else:
        warning = (
            f"CUDA reports {torch_free_mib} MiB free after context creation and nvidia-smi reports "
            f"{nvidia_smi_free_mib} MiB free; measured model need is {model_need_mib} MiB. "
            "Choose CPU fallback or explicitly select Try GPU anyway."
        )

    status = {
        "available": True,
        "cuda_available": True,
        "gpu_name": gpu_name,
        "driver_version": driver_version,
        "minimum_driver_version": MINIMUM_NVIDIA_DRIVER_VERSION,
        "driver_compatible": True,
        "free_mib": torch_free_mib,
        "total_mib": torch_total_mib,
        "nvidia_smi_free_mib": nvidia_smi_free_mib,
        "torch_free_before_context_mib": (
            cuda_memory["free_before"] // MIB
            if cuda_memory["free_before"] is not None
            else None
        ),
        "torch_free_after_context_mib": torch_free_mib,
        "torch_context_init_delta_mib": cuda_memory["torch_context_init_delta_mib"],
        "nvidia_smi_free_after_context_mib": nvidia_smi_free_after_context_mib,
        "nvidia_context_init_delta_mib": nvidia_context_init_delta_mib,
        "nvidia_torch_delta_mib": nvidia_torch_delta_mib,
        "context_overhead_mib": context_overhead_mib,
        "model_peak_reserved_mib": model_peak_reserved_mib,
        "model_need_mib": model_need_mib,
        "minimum_free_mib": model_need_mib,
        "safe_to_load_model": safe_to_load_model,
        "warning": warning,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(
        "GPU memory calibration: nvidia_smi_free_before_context_mib=%d "
        "torch_initialized_before=%s torch_free_before_context_mib=%s "
        "torch_initialized_after_first_read=%s torch_free_after_context_mib=%d "
        "torch_context_init_delta_mib=%s nvidia_smi_free_after_context_mib=%s "
        "nvidia_context_init_delta_mib=%s nvidia_torch_delta_mib=%d context_overhead_mib=%s "
        "model_peak_reserved_mib=%s model_need_mib=%s.",
        nvidia_smi_free_mib,
        cuda_memory["initialized_before"],
        status["torch_free_before_context_mib"],
        cuda_memory["initialized_after_first_read"],
        torch_free_mib,
        cuda_memory["torch_context_init_delta_mib"],
        nvidia_smi_free_after_context_mib,
        nvidia_context_init_delta_mib,
        nvidia_torch_delta_mib,
        context_overhead_mib,
        model_peak_reserved_mib,
        model_need_mib,
    )
    return status
