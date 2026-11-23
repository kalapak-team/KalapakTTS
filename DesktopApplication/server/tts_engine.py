"""
TTS Engine — lazy singleton wrapping VoxCPM2.

Usage:
    from server.tts_engine import tts_engine
    sample_rate, wav_np = await tts_engine.generate_audio(text, control_instruction, cfg_value, inference_timesteps)
"""

import sys
import os
import asyncio
import ctypes
import gc
import logging
import threading
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from server.config import settings
from server.ffmpeg_audio import apply_ffmpeg_audio_filter, match_audio_rms
from server.gpu import MEASURED_CUDA_CONTEXT_OVERHEAD_MIB, read_gpu_status
from server.model_assets import model_asset_status, verify_model_assets

logger = logging.getLogger(__name__)

_VOXCPM_VENDOR_DIR = str(Path(__file__).resolve().parent / "vendor")
if _VOXCPM_VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VOXCPM_VENDOR_DIR)


def time_stretch_audio(
    wav_np: np.ndarray,
    sample_rate: int,
    speed: float,
) -> np.ndarray:
    if speed == 1.0:
        return wav_np
    if sample_rate <= 0 or not np.isfinite(speed) or not 0.5 <= speed <= 2.0:
        raise ValueError("Time stretching requires a positive sample rate and speed from 0.5 to 2.0.")

    stretched = apply_ffmpeg_audio_filter(
        wav_np,
        sample_rate,
        f"atempo={speed:.6f}",
        "speech pace correction",
    )
    return match_audio_rms(stretched, wav_np)


class TTSEngine:
    """Lazy-loaded singleton for VoxCPM2."""

    _MODEL_FILES = {
        "audiovae.pth": (376_951_122, "94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1"),
        "model.safetensors": (4_580_080_592, "f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d"),
    }

    def __init__(self):
        self._model = None
        self._lock = asyncio.Lock()
        self._model_id = settings.VOXCPM_MODEL_ID
        self._assets_verified = False
        self._loading = False
        self._device = None
        self._requested_device = None
        self._generation_lock = threading.Lock()
        self._load_seconds: float | None = None
        self._last_inference_seconds: float | None = None
        self._peak_vram_bytes = 0
        self._peak_vram_reserved_bytes = 0
        self._gpu_name: str | None = None
        self._torch_version: str | None = None
        self._cuda_version: str | None = None
        self._model_dtype: str | None = None
        self._vae_device: str | None = None
        self._fallback_reason: str | None = None

    @staticmethod
    def _process_memory_status() -> dict[str, int | None]:
        if os.name == "nt":
            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            get_current_process = kernel32.GetCurrentProcess
            get_current_process.restype = ctypes.c_void_p
            get_process_memory_info = psapi.GetProcessMemoryInfo
            get_process_memory_info.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ProcessMemoryCounters),
                ctypes.c_ulong,
            ]
            get_process_memory_info.restype = ctypes.c_int
            process = get_current_process()
            if get_process_memory_info(
                process,
                ctypes.byref(counters),
                counters.cb,
            ):
                return {
                    "working_set_bytes": int(counters.WorkingSetSize),
                    "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
                }
            logger.warning(
                "Could not read desktop process memory counters (Windows error %d).",
                ctypes.get_last_error(),
            )
            return {"working_set_bytes": None, "peak_working_set_bytes": None}

        try:
            import resource
        except ImportError:
            return {"working_set_bytes": None, "peak_working_set_bytes": None}
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = int(peak_rss if sys.platform == "darwin" else peak_rss * 1024)
        return {"working_set_bytes": None, "peak_working_set_bytes": peak_bytes}

    @staticmethod
    def _is_cuda_oom(error: Exception, torch) -> bool:
        oom_type = getattr(torch.cuda, "OutOfMemoryError", ())
        return (
            (bool(oom_type) and isinstance(error, oom_type))
            or "out of memory" in str(error).lower()
            and ("cuda" in str(error).lower() or "cublas" in str(error).lower())
        )

    @staticmethod
    def _offload_audio_vae_to_cpu(model) -> None:
        vae = model.tts_model.audio_vae
        vae.to("cpu")
        original_encode = vae.encode
        original_decode = vae.decode

        def encode_on_cpu(audio, sample_rate):
            return original_encode(audio.to("cpu"), sample_rate)

        def decode_on_cpu(latents, sr_cond=None):
            cpu_sr = sr_cond.to("cpu") if sr_cond is not None else None
            return original_decode(latents.to("cpu"), cpu_sr)

        vae.encode = encode_on_cpu
        vae.decode = decode_on_cpu

    def status(self) -> dict:
        assets = model_asset_status()
        total_bytes = assets.get(
            "total_bytes",
            sum(size for size, _ in self._MODEL_FILES.values()),
        )
        try:
            import torch
        except ImportError:
            torch = None
        cuda_available = bool(torch and torch.cuda.is_available())
        vram_total_bytes = 0
        vram_used_bytes = 0
        free_vram_bytes = 0
        minimum_free_vram_mib = (
            settings.MIN_FREE_VRAM_MIB + MEASURED_CUDA_CONTEXT_OVERHEAD_MIB
            if settings.MIN_FREE_VRAM_MIB is not None
            and MEASURED_CUDA_CONTEXT_OVERHEAD_MIB is not None
            else None
        )
        if cuda_available:
            try:
                gpu_status = read_gpu_status()
                if not gpu_status.get("available") or not gpu_status.get("cuda_available"):
                    raise RuntimeError(gpu_status["warning"])
                minimum_free_vram_mib = gpu_status["model_need_mib"]
                vram_total_bytes = int(gpu_status["total_mib"]) * 1024 * 1024
                free_vram_bytes = int(gpu_status["nvidia_smi_free_mib"]) * 1024 * 1024
                vram_used_bytes = vram_total_bytes - free_vram_bytes
                self._gpu_name = torch.cuda.get_device_name(0)
            except RuntimeError:
                cuda_available = False
        if torch is not None:
            self._torch_version = torch.__version__
            self._cuda_version = torch.version.cuda
        try:
            bf16_supported = bool(cuda_available and torch.cuda.is_bf16_supported())
        except RuntimeError:
            bf16_supported = False
        process_memory = self._process_memory_status()
        selected_device = self._device or self._requested_device
        fallback_reason = self._fallback_reason
        if selected_device is None:
            if settings.DEVICE == "cpu":
                selected_device = "cpu"
            elif cuda_available:
                free_mib = free_vram_bytes // (1024 * 1024)
                if settings.MIN_FREE_VRAM_MIB is None:
                    selected_device = "blocked"
                    fallback_reason = (
                        "The desktop model's peak VRAM need has not been measured yet. "
                        "Choose CPU fallback or explicitly try the GPU."
                    )
                elif free_mib >= settings.MIN_FREE_VRAM_MIB:
                    selected_device = "cuda"
                else:
                    selected_device = "blocked"
                    fallback_reason = (
                        f"Only {free_mib} MiB CUDA memory is free after context creation; "
                        "the measured model peak requires "
                        f"{settings.MIN_FREE_VRAM_MIB} MiB. Choose CPU fallback or try the GPU anyway."
                    )
            elif settings.FORCE_CUDA or settings.DEVICE == "cuda":
                selected_device = "blocked"
                fallback_reason = "CUDA was requested but is unavailable in this PyTorch environment."
            else:
                selected_device = "blocked"
                fallback_reason = "CUDA is unavailable. Choose CPU fallback to continue."
        return {
            "model_id": self._model_id,
            "model_available": assets["available"],
            "model_error": assets["message"],
            "model_directory": assets["directory"],
            "model_loaded": self._model is not None,
            "model_loading": self._loading,
            "model_device": self._device,
            "selected_device": selected_device,
            "configured_device": settings.DEVICE,
            "force_cuda": settings.FORCE_CUDA,
            "minimum_free_vram_mib": minimum_free_vram_mib,
            "model_peak_reserved_mib": settings.MIN_FREE_VRAM_MIB,
            "torch_version": self._torch_version,
            "cuda_version": self._cuda_version,
            "cuda_available": cuda_available,
            "bf16_supported": bf16_supported,
            "gpu_name": self._gpu_name,
            "model_dtype": self._model_dtype,
            "vae_device": self._vae_device,
            "fallback_reason": fallback_reason,
            "vram_used_bytes": vram_used_bytes,
            "vram_free_bytes": int(free_vram_bytes) if cuda_available else 0,
            "vram_total_bytes": vram_total_bytes,
            "vram_allocated_bytes": (
                int(torch.cuda.memory_allocated()) if cuda_available else 0
            ),
            "vram_reserved_bytes": (
                int(torch.cuda.memory_reserved()) if cuda_available else 0
            ),
            "model_load_seconds": self._load_seconds,
            "last_inference_seconds": self._last_inference_seconds,
            "working_set_bytes": process_memory["working_set_bytes"],
            "peak_ram_bytes": process_memory["peak_working_set_bytes"],
            "peak_vram_bytes": self._peak_vram_bytes,
            "peak_vram_reserved_bytes": self._peak_vram_reserved_bytes,
            "verified_model_bytes": total_bytes if assets["available"] else 0,
            "total_model_bytes": total_bytes,
        }

    def cache_device_key(self) -> str:
        """Return the effective device so CPU and CUDA WAV caches never collide."""
        if self._device is not None:
            return self._device
        import torch

        return self._resolve_device(torch)

    def log_runtime_info(self) -> None:
        """Log the installed PyTorch build and the available accelerator at startup."""
        try:
            import torch
        except ImportError:
            logger.exception("PyTorch is missing from the desktop runtime.")
            raise

        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else "unavailable"
        bf16_supported = bool(cuda_available and torch.cuda.is_bf16_supported())
        logger.info(
            "TTS runtime: torch=%s CUDA build=%s CUDA available=%s GPU=%s BF16 supported=%s "
            "DEVICE=%s FORCE_CUDA=%s MIN_FREE_VRAM_MIB=%s.",
            torch.__version__,
            torch.version.cuda or "none",
            cuda_available,
            gpu_name,
            bf16_supported,
            settings.DEVICE,
            settings.FORCE_CUDA,
            settings.MIN_FREE_VRAM_MIB,
        )

    def _resolve_device(self, torch, allow_low_vram: bool = False) -> str:
        if settings.DEVICE == "cpu":
            return "cpu"
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable in this PyTorch runtime. Choose CPU fallback to load VoxCPM2."
            )

        gpu_status = read_gpu_status()
        if not gpu_status.get("available") or not gpu_status.get("cuda_available"):
            raise RuntimeError(gpu_status["warning"])
        free_mib = int(gpu_status["free_mib"])
        logger.info(
            "CUDA memory before model load: Torch=%d MiB, nvidia-smi=%d MiB, "
            "measured model need=%s MiB.",
            free_mib,
            gpu_status["nvidia_smi_free_mib"],
            gpu_status["model_need_mib"],
        )
        if not gpu_status["safe_to_load_model"]:
            if not (settings.FORCE_CUDA or allow_low_vram):
                self._fallback_reason = gpu_status["warning"]
                raise RuntimeError(gpu_status["warning"])
            if gpu_status["model_peak_reserved_mib"] is None:
                self._fallback_reason = (
                    "The GPU was tried before a model peak-memory requirement was measured. "
                    "CUDA out-of-memory will switch this job to the CPU."
                )
            else:
                self._fallback_reason = (
                    f"GPU was tried with {gpu_status['nvidia_smi_free_mib']} MiB free, below "
                    f"the measured {gpu_status['model_need_mib']} MiB model need. CUDA out-of-memory "
                    "will switch this job to the CPU."
                )
        else:
            self._fallback_reason = None
        return "cuda"

    def reset_peak_memory_metrics(self) -> None:
        self._peak_vram_bytes = 0
        self._peak_vram_reserved_bytes = 0
        if self._device == "cuda":
            import torch

            torch.cuda.reset_peak_memory_stats()

    def _load_model(
        self,
        target_device: str | None = None,
        allow_oom_fallback: bool = True,
        allow_low_vram: bool = False,
    ):
        """Synchronous model loading (called in executor)."""
        if self._model is not None:
            return
        if not self._assets_verified:
            verify_model_assets()
            self._assets_verified = True
        import torch
        from voxcpm import VoxCPM

        if target_device == "cpu":
            device = "cpu"
        elif target_device == "cuda":
            device = self._resolve_device(torch, allow_low_vram)
        else:
            device = self._resolve_device(torch, allow_low_vram)
        if device == "cpu" and self._fallback_reason is None:
            self._fallback_reason = "CPU fallback was selected; generation will run on the CPU."
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA model loading was requested, but CUDA is unavailable.")
        if device == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected NVIDIA device does not support the required BF16 runtime.")
        self._requested_device = device

        started = time.perf_counter()
        logger.info("Loading VoxCPM2 model on %s; this may take a while.", device)
        try:
            model = VoxCPM.from_pretrained(
                self._model_id,
                device=device,
                load_denoiser=False,
                optimize=False,
                local_files_only=True,
            )
            if device == "cuda":
                self._offload_audio_vae_to_cpu(model)
        except Exception as exc:
            if not (allow_oom_fallback and device == "cuda" and self._is_cuda_oom(exc, torch)):
                raise
            reason = f"CUDA ran out of memory while loading VoxCPM2 ({exc}); retrying on CPU."
            logger.warning("%s", reason)
            exc.__traceback__ = None
            gc.collect()
            torch.cuda.empty_cache()
            self._fallback_reason = reason
            return self._load_model(
                target_device="cpu",
                allow_oom_fallback=False,
                allow_low_vram=False,
            )

        self._model = model
        self._device = device
        self._model_dtype = str(getattr(getattr(model.tts_model, "config", None), "dtype", "unknown"))
        self._vae_device = "cpu" if device == "cuda" else "cpu"
        self._gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        self._torch_version = torch.__version__
        self._cuda_version = torch.version.cuda
        self._load_seconds = time.perf_counter() - started
        logger.info(
            "VoxCPM2 model loaded successfully on %s (dtype=%s, AudioVAE=%s) in %.2f seconds.",
            device,
            self._model_dtype,
            self._vae_device,
            self._load_seconds,
        )

    def _generate_streaming_audio(
        self,
        model,
        text: str,
        cfg_value: float,
        inference_timesteps: int,
        reference_wav_path: str | None,
        seed: int | None,
        torch,
    ) -> np.ndarray:
        chunks: list[np.ndarray] = []
        stream = model.generate_streaming(
            text=text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            reference_wav_path=reference_wav_path,
            seed=seed,
            retry_badcase=False,
        )
        try:
            with torch.inference_mode():
                for chunk in stream:
                    audio_chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
                    if audio_chunk.size:
                        chunks.append(audio_chunk)
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        if not chunks:
            raise RuntimeError("VoxCPM2 returned no audio chunks.")
        return np.concatenate(chunks)

    def _fallback_to_cpu(self, torch, reason: str) -> None:
        self._fallback_reason = reason
        logger.warning("%s", reason)
        self._model = None
        self._device = None
        self._model_dtype = None
        self._vae_device = None
        gc.collect()
        torch.cuda.empty_cache()
        self._load_model(target_device="cpu", allow_oom_fallback=False)

    async def _ensure_loaded(
        self,
        target_device: str | None = None,
        allow_low_vram: bool = False,
    ):
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    self._loading = True
                    try:
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(
                            None,
                            self._load_model,
                            target_device,
                            True,
                            allow_low_vram,
                        )
                    finally:
                        self._loading = False
        elif target_device is not None and target_device != self._device:
            raise RuntimeError(
                f"VoxCPM2 is already loaded on {self._device}; restart the desktop app to change devices."
            )

    async def load_model(
        self,
        target_device: str | None = None,
        allow_low_vram: bool = False,
    ):
        """Load the process-wide model instance once and keep it resident."""
        if target_device not in (None, "cpu", "cuda"):
            raise ValueError("The model device must be cpu or cuda.")
        await self._ensure_loaded(target_device, allow_low_vram)

    def _run_inference(
        self,
        text: str,
        control_instruction: str,
        cfg_value: float,
        inference_timesteps: int,
        reference_wav_path: str | None,
        speed: float,
        seed: int | None,
    ) -> Tuple[int, np.ndarray]:
        """Synchronous inference (called in executor)."""
        combined_text = text
        if control_instruction:
            instruction = control_instruction.translate(str.maketrans("", "", "()（）")).strip()
            combined_text = f"({instruction}){text}"

        with self._generation_lock:
            inference_started = time.perf_counter()
            import torch

            measured_cuda = bool(self._device and self._device.startswith("cuda"))
            if measured_cuda:
                torch.cuda.reset_peak_memory_stats()
            model = self._model
            try:
                wav_np = self._generate_streaming_audio(
                    model,
                    combined_text,
                    cfg_value,
                    inference_timesteps,
                    reference_wav_path,
                    seed,
                    torch,
                )
            except Exception as exc:
                if not (
                    self._device == "cuda"
                    and self._is_cuda_oom(exc, torch)
                ):
                    raise
                reason = (
                    f"CUDA ran out of memory during speech generation ({exc}). "
                    "VoxCPM2 is switching to CPU and retrying this chunk."
                )
                exc.__traceback__ = None
                self._model = None
                del model
                gc.collect()
                torch.cuda.empty_cache()
                self._fallback_to_cpu(torch, reason)
                wav_np = self._generate_streaming_audio(
                    self._model,
                    combined_text,
                    cfg_value,
                    inference_timesteps,
                    reference_wav_path,
                    seed,
                    torch,
                )
            finally:
                if measured_cuda:
                    self._peak_vram_bytes = max(
                        self._peak_vram_bytes,
                        torch.cuda.max_memory_allocated(),
                    )
                    self._peak_vram_reserved_bytes = max(
                        self._peak_vram_reserved_bytes,
                        torch.cuda.max_memory_reserved(),
                    )
                    torch.cuda.empty_cache()
            self._last_inference_seconds = time.perf_counter() - inference_started
            logger.info("VoxCPM2 inference finished in %.2f seconds.", self._last_inference_seconds)
        sample_rate: int = self._model.tts_model.sample_rate
        # Ensure float32 numpy array
        if not isinstance(wav_np, np.ndarray):
            wav_np = np.array(wav_np, dtype=np.float32)
        elif wav_np.dtype != np.float32:
            wav_np = wav_np.astype(np.float32)
        wav_np = time_stretch_audio(wav_np, sample_rate, speed)
        return sample_rate, wav_np

    async def generate_audio(
        self,
        text: str,
        control_instruction: str = "",
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        reference_wav_path: str | None = None,
        speed: float = 1.0,
        seed: int | None = None,
    ) -> Tuple[int, np.ndarray]:
        """
        Generate audio for `text`.

        Returns:
            (sample_rate, wav_np) — sample rate in Hz and WAV as float32 numpy array.
        """
        await self._ensure_loaded()
        loop = asyncio.get_event_loop()
        sample_rate, wav_np = await loop.run_in_executor(
            None,
            self._run_inference,
            text,
            control_instruction,
            cfg_value,
            inference_timesteps,
            reference_wav_path,
            speed,
            seed,
        )
        return sample_rate, wav_np


# Module-level singleton
tts_engine = TTSEngine()
