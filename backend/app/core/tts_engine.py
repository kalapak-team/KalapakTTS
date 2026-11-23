"""
TTS Engine — lazy singleton wrapping VoxCPM2.

Usage:
    from app.core.tts_engine import tts_engine
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
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / "model_cache"))

import numpy as np
from app.config import settings
from app.core.ffmpeg_audio import apply_ffmpeg_audio_filter, match_audio_rms
from huggingface_hub.constants import HF_HUB_CACHE

logger = logging.getLogger(__name__)

# Ensure VoxCPM src is on the path before any import
_VOXCPM_SRC = str(Path(__file__).resolve().parents[3] / "VoxCPM" / "src")
if _VOXCPM_SRC not in sys.path:
    sys.path.insert(0, _VOXCPM_SRC)


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
        self._loading = False
        self._device = None
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
    def _peak_rss_bytes() -> int | None:
        if sys.platform == "win32":
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
            process = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                process,
                ctypes.byref(counters),
                counters.cb,
            ):
                return int(counters.PeakWorkingSetSize)
            return None

        try:
            import resource
        except ImportError:
            return None
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak_rss if sys.platform == "darwin" else peak_rss * 1024)

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
        total_bytes = sum(size for size, _ in self._MODEL_FILES.values())
        downloaded_bytes = 0
        if self._model_id == "openbmb/VoxCPM2":
            model_cache = Path(HF_HUB_CACHE) / "models--openbmb--VoxCPM2" / "blobs"
            for expected_size, blob_name in self._MODEL_FILES.values():
                blob = model_cache / blob_name
                size = blob.stat().st_size if blob.is_file() else 0
                if size < expected_size:
                    parts = [
                        part
                        for part in model_cache.glob(f"{blob_name}.part-*")
                        if part.is_file()
                    ]
                    if parts:
                        size += sum(part.stat().st_size for part in parts)
                    else:
                        incomplete = model_cache / f"{blob_name}.incomplete"
                        if incomplete.is_file():
                            size += incomplete.stat().st_size
                downloaded_bytes += min(size, expected_size)
        try:
            import torch
        except ImportError:
            torch = None
        cuda_available = bool(torch and torch.cuda.is_available())
        vram_total_bytes = 0
        vram_used_bytes = 0
        free_vram_bytes = 0
        if cuda_available:
            free_vram_bytes, total_vram = torch.cuda.mem_get_info()
            vram_total_bytes = int(total_vram)
            vram_used_bytes = int(total_vram - free_vram_bytes)
            self._gpu_name = torch.cuda.get_device_name(0)
        if torch is not None:
            self._torch_version = torch.__version__
            self._cuda_version = torch.version.cuda
        try:
            bf16_supported = bool(cuda_available and torch.cuda.is_bf16_supported())
        except RuntimeError:
            bf16_supported = False
        selected_device = self._device
        fallback_reason = self._fallback_reason
        if selected_device is None:
            if settings.DEVICE == "cpu":
                selected_device = "cpu"
            elif cuda_available:
                free_gib = free_vram_bytes / 1024**3
                can_select_cuda = settings.FORCE_CUDA or free_gib >= settings.MIN_FREE_VRAM_GIB
                if can_select_cuda:
                    selected_device = "cuda"
                elif settings.DEVICE == "cuda":
                    selected_device = "blocked"
                    fallback_reason = (
                        f"Only {free_gib:.2f} GiB VRAM is free; the configured minimum is "
                        f"{settings.MIN_FREE_VRAM_GIB:.2f} GiB. Set FORCE_CUDA=true to override."
                    )
                else:
                    selected_device = "cpu"
                    fallback_reason = (
                        f"Only {free_gib:.2f} GiB VRAM is free; the configured minimum is "
                        f"{settings.MIN_FREE_VRAM_GIB:.2f} GiB."
                    )
            elif settings.FORCE_CUDA or settings.DEVICE == "cuda":
                selected_device = "cuda"
                fallback_reason = "CUDA was requested but is unavailable in this PyTorch environment."
            else:
                selected_device = "cpu"
                fallback_reason = "CUDA is unavailable; the server will use CPU."
        return {
            "model_id": self._model_id,
            "model_loaded": self._model is not None,
            "model_loading": self._loading,
            "model_device": self._device,
            "selected_device": selected_device,
            "configured_device": settings.DEVICE,
            "force_cuda": settings.FORCE_CUDA,
            "min_free_vram_gib": settings.MIN_FREE_VRAM_GIB,
            "torch_version": self._torch_version,
            "cuda_version": self._cuda_version,
            "cuda_available": cuda_available,
            "bf16_supported": bf16_supported,
            "gpu_name": self._gpu_name,
            "model_dtype": self._model_dtype,
            "vae_device": self._vae_device,
            "fallback_reason": fallback_reason,
            "vram_used_bytes": vram_used_bytes,
            "vram_total_bytes": vram_total_bytes,
            "vram_allocated_bytes": (
                int(torch.cuda.memory_allocated()) if cuda_available else 0
            ),
            "vram_reserved_bytes": (
                int(torch.cuda.memory_reserved()) if cuda_available else 0
            ),
            "model_load_seconds": self._load_seconds,
            "last_inference_seconds": self._last_inference_seconds,
            "peak_ram_bytes": self._peak_rss_bytes(),
            "peak_vram_bytes": self._peak_vram_bytes,
            "peak_vram_reserved_bytes": self._peak_vram_reserved_bytes,
            "downloaded_bytes": downloaded_bytes,
            "total_bytes": total_bytes,
            "download_percent": round(downloaded_bytes * 100 / total_bytes),
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
            logger.exception("PyTorch is missing from the backend environment.")
            raise

        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else "unavailable"
        bf16_supported = bool(cuda_available and torch.cuda.is_bf16_supported())
        logger.info(
            "TTS runtime: torch=%s CUDA build=%s CUDA available=%s GPU=%s BF16 supported=%s "
            "DEVICE=%s FORCE_CUDA=%s MIN_FREE_VRAM_GIB=%.2f.",
            torch.__version__,
            torch.version.cuda or "none",
            cuda_available,
            gpu_name,
            bf16_supported,
            settings.DEVICE,
            settings.FORCE_CUDA,
            settings.MIN_FREE_VRAM_GIB,
        )

    def _resolve_device(self, torch) -> str:
        if settings.DEVICE == "cpu":
            return "cpu"
        if not torch.cuda.is_available():
            if settings.FORCE_CUDA or settings.DEVICE == "cuda":
                raise RuntimeError(
                    "CUDA was requested but this PyTorch environment has no usable CUDA runtime. "
                    "Install a CUDA-enabled PyTorch build or set DEVICE=cpu."
                )
            logger.warning("CUDA is unavailable; loading VoxCPM2 on CPU.")
            return "cpu"

        free_vram, total_vram = torch.cuda.mem_get_info()
        free_gib = free_vram / 1024**3
        logger.info(
            "CUDA memory before model load: %.2f / %.2f GiB free.",
            free_gib,
            total_vram / 1024**3,
        )
        if not settings.FORCE_CUDA and free_gib < settings.MIN_FREE_VRAM_GIB:
            message = (
                f"Only {free_gib:.2f} GiB VRAM is free; the configured minimum is "
                f"{settings.MIN_FREE_VRAM_GIB:.2f} GiB."
            )
            if settings.DEVICE == "cuda":
                raise RuntimeError(
                    f"{message} Set FORCE_CUDA=true to bypass this guard, or use DEVICE=auto/cpu."
                )
            logger.warning("%s Selecting CPU for this server process.", message)
            self._fallback_reason = message
            return "cpu"
        return "cuda"

    def _load_model(self, target_device: str | None = None, allow_oom_fallback: bool = True):
        """Synchronous model loading (called in executor)."""
        if self._model is not None:
            return
        import torch
        import voxcpm  # noqa: F401 — imported for side-effects / registration
        from voxcpm import VoxCPM

        device = target_device or self._resolve_device(torch)
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA model loading was requested, but CUDA is unavailable.")

        started = time.perf_counter()
        logger.info("Loading VoxCPM2 model on %s; this may take a while.", device)
        try:
            model = VoxCPM.from_pretrained(
                self._model_id,
                device=device,
                load_denoiser=False,
                optimize=False,
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
            return self._load_model(target_device="cpu", allow_oom_fallback=False)

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

    async def _ensure_loaded(self):
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    self._loading = True
                    try:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, self._load_model)
                    finally:
                        self._loading = False

    async def load_model(self):
        """Load the process-wide model instance once and keep it resident."""
        await self._ensure_loaded()

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
