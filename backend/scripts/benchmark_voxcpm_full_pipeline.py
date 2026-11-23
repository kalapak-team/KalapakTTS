from __future__ import annotations

import ctypes
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
MODEL = (
    BACKEND
    / "model_cache"
    / "hub"
    / "models--openbmb--VoxCPM2"
    / "snapshots"
    / "7936080b41163874f38610f46f438e2c0757f9fd"
)
ONNX = BACKEND / "model_cache" / "onnx-uddom-voxcpm2-ad57fe5"
FIXTURE = BACKEND / "tests" / "fixtures" / "khmer_long_sample.txt"
REFERENCE = BACKEND / "voice_references" / "vanna.wav"
OUTPUT = BACKEND / "model_cache" / "onnx-parity-artifacts" / "stage-f-audio"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT / "VoxCPM" / "src"))

from app.core.audio_diagnostics import audio_quality_metrics, speech_timing_metrics
from voxcpm import VoxCPM


class MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def available_ram_gib() -> float:
    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed.")
    return status.ullAvailPhys / 1024**3


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    signal = np.asarray(audio, dtype=np.float32).reshape(-1)
    pcm = np.clip(signal, -1.0, 1.0)
    pcm = np.round(pcm * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def generation_metrics(audio: np.ndarray, sample_rate: int, text: str) -> dict[str, object]:
    signal = np.asarray(audio, dtype=np.float32).reshape(-1)
    quality = audio_quality_metrics(signal, sample_rate)
    timing = speech_timing_metrics(signal, sample_rate)
    thirds: list[dict[str, float | int]] = []
    for index in range(3):
        start = round(len(signal) * index / 3)
        end = round(len(signal) * (index + 1) / 3)
        third_timing = speech_timing_metrics(signal[start:end], sample_rate)
        third_quality = audio_quality_metrics(signal[start:end], sample_rate)
        thirds.append(
            {
                "speech_seconds_estimated": float(third_timing["speech_seconds"]),
                "chars_per_second_estimated": (
                    len(text) / 3 / float(third_timing["speech_seconds"])
                    if third_timing["speech_seconds"]
                    else 0.0
                ),
                "pause_count_at_least_250ms": int(third_timing["pause_count_250ms"]),
                "pause_seconds_at_least_250ms": float(third_timing["pause_seconds_250ms"]),
                "high_frequency_energy_ratio_over_4khz": float(
                    third_quality["high_frequency_energy_ratio"]
                ),
            }
        )
    return {
        "duration_seconds": quality["duration_seconds"],
        "rms_dbfs": quality["rms_dbfs"],
        "high_frequency_energy_ratio_over_4khz": quality["high_frequency_energy_ratio"],
        "pause_count_at_least_250ms": timing["pause_count_250ms"],
        "pause_seconds_at_least_250ms": timing["pause_seconds_250ms"],
        "speech_seconds_estimated": timing["speech_seconds"],
        "speech_only_chars_per_second_estimated": (
            len(text) / float(timing["speech_seconds"]) if timing["speech_seconds"] else 0.0
        ),
        "thirds": thirds,
    }


def offload_vae_to_cpu(model: VoxCPM) -> None:
    vae = model.tts_model.audio_vae
    vae.to("cpu")
    original_encode = vae.encode
    original_decode = vae.decode

    def encode_on_cpu(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return original_encode(audio.to("cpu"), sample_rate)

    def decode_on_cpu(latents: torch.Tensor, sr_cond: torch.Tensor | None = None):
        cpu_sr = sr_cond.to("cpu") if sr_cond is not None else None
        return original_decode(latents.to("cpu"), cpu_sr)

    vae.encode = encode_on_cpu
    vae.decode = decode_on_cpu


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this benchmark specifically requires the RTX 4050.")
    if not REFERENCE.is_file():
        raise FileNotFoundError(f"Vanna reference WAV is missing: {REFERENCE}")
    if available_ram_gib() < 6.0:
        raise MemoryError(
            f"Only {available_ram_gib():.2f} GiB RAM is available; refusing the full-model load."
        )

    fixture = FIXTURE.read_text(encoding="utf-8").strip()
    texts = [("35char", fixture[:35]), ("long-fixture", fixture)]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "event": "preflight",
                "available_ram_gib": round(available_ram_gib(), 2),
                "cuda_free_gib": round(torch.cuda.mem_get_info()[0] / 1024**3, 2),
                "cuda_total_gib": round(torch.cuda.mem_get_info()[1] / 1024**3, 2),
                "model_bytes": (MODEL / "model.safetensors").stat().st_size,
                "audiovae_bytes": (MODEL / "audiovae.pth").stat().st_size,
                "load_denoiser": False,
                "vae_offload": "CPU",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    load_started = time.perf_counter()
    model = VoxCPM.from_pretrained(
        str(MODEL),
        device="cuda",
        load_denoiser=False,
        optimize=False,
        local_files_only=True,
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    offload_vae_to_cpu(model)
    torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "event": "model_loaded",
                "load_seconds": round(load_seconds, 3),
                "cuda_allocated_gib_after_vae_offload": round(
                    torch.cuda.memory_allocated() / 1024**3, 3
                ),
                "cuda_reserved_gib_after_vae_offload": round(
                    torch.cuda.memory_reserved() / 1024**3, 3
                ),
                "cuda_free_gib_after_vae_offload": round(
                    torch.cuda.mem_get_info()[0] / 1024**3, 3
                ),
            }
        ),
        flush=True,
    )

    stop_values: list[int] = []
    latent_patches: list[np.ndarray] = []
    stop_hook = model.tts_model.stop_head.register_forward_hook(
        lambda _module, _inputs, output: stop_values.append(
            int(output.detach().float().argmax(dim=-1)[0].cpu().item())
        )
    )
    feat_decoder = model.tts_model.feat_decoder
    latent_hook = feat_decoder.register_forward_hook(
        lambda _module, _inputs, output: latent_patches.append(
            output.detach().float().cpu().numpy()
        )
    )
    session = ort.InferenceSession(
        str(ONNX / "vae" / "vae_decoder.onnx"),
        providers=["CPUExecutionProvider"],
    )
    results: list[dict[str, object]] = []
    failure: dict[str, str] | None = None

    try:
        for label, text in texts:
            for steps in (4, 6, 10):
                stop_values.clear()
                latent_patches.clear()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                first_audio_seconds: float | None = None
                chunks: list[np.ndarray] = []
                try:
                    stream = model.generate_streaming(
                        text=text,
                        reference_wav_path=str(REFERENCE),
                        cfg_value=2.0,
                        inference_timesteps=steps,
                        seed=123,
                        retry_badcase=False,
                    )
                    for chunk in stream:
                        if first_audio_seconds is None:
                            first_audio_seconds = time.perf_counter() - started
                        chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
                    audio = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    sample_rate = model.tts_model.audio_vae.out_sample_rate
                    name = f"pytorch-{label}-seed123-steps{steps}.wav"
                    write_wav(OUTPUT / name, audio, sample_rate)
                    generated_metrics = generation_metrics(audio, sample_rate, text)
                    if latent_patches:
                        latents = np.concatenate(latent_patches, axis=-1).astype(np.float32)
                        decoder_audio = session.run(["audio"], {"latents": latents})[0].reshape(-1)
                        onnx_name = f"onnx-vae-{label}-seed123-steps{steps}.wav"
                        write_wav(OUTPUT / onnx_name, decoder_audio, sample_rate)
                        decoder_metrics = generation_metrics(decoder_audio, sample_rate, text)
                        decoder_parity: dict[str, object] = {
                            "result": "MEASURED_VAE_DECODER_ONLY",
                            "input_latent_shape": list(latents.shape),
                            "pytorch_audio_samples": int(audio.size),
                            "onnx_audio_samples": int(decoder_audio.size),
                        }
                        if audio.shape == decoder_audio.shape:
                            delta = decoder_audio.astype(np.float64) - audio.astype(np.float64)
                            norm = np.linalg.norm(audio.astype(np.float64))
                            decoder_parity.update(
                                {
                                    "relative_l2": float(
                                        np.linalg.norm(delta) / max(float(norm), 1e-12)
                                    ),
                                    "cosine": float(
                                        np.dot(audio.astype(np.float64), decoder_audio.astype(np.float64))
                                        / max(
                                            float(
                                                np.linalg.norm(audio.astype(np.float64))
                                                * np.linalg.norm(decoder_audio.astype(np.float64))
                                            ),
                                            1e-12,
                                        )
                                    ),
                                }
                            )
                        results.append(
                            {
                                "text": label,
                                "steps": steps,
                                "seed": 123,
                                "status": "MEASURED",
                                "first_audio_seconds": first_audio_seconds,
                                "total_seconds": elapsed,
                                "rtf": elapsed / generated_metrics["duration_seconds"]
                                if generated_metrics["duration_seconds"]
                                else None,
                                "patch_count": len(latent_patches),
                                "stop_token_position_1based": next(
                                    (index + 1 for index, value in enumerate(stop_values) if value == 1),
                                    None,
                                ),
                                "stop_head_steps": len(stop_values),
                                "cuda_peak_allocated_gib": round(
                                    torch.cuda.max_memory_allocated() / 1024**3, 3
                                ),
                                "pytorch_wav": str(OUTPUT / name),
                                "onnx_vae_wav": str(OUTPUT / onnx_name),
                                "pytorch_metrics": generated_metrics,
                                "onnx_vae_metrics": decoder_metrics,
                                "decoder_parity": decoder_parity,
                            }
                        )
                    else:
                        results.append(
                            {
                                "text": label,
                                "steps": steps,
                                "seed": 123,
                                "status": "FAILED_NO_LATENTS",
                                "total_seconds": elapsed,
                            }
                        )
                    print(json.dumps(results[-1], ensure_ascii=False), flush=True)
                except Exception as exc:
                    failure = {"type": type(exc).__name__, "message": str(exc)}
                    print(
                        json.dumps(
                            {
                                "event": "generation_failed",
                                "text": label,
                                "steps": steps,
                                **failure,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    break
            if failure:
                break
    finally:
        stop_hook.remove()
        latent_hook.remove()
        session = None

    report = {
        "result": "PARTIAL_FAILURE" if failure else "MEASURED",
        "load_seconds": load_seconds,
        "device": "RTX 4050 Laptop GPU",
        "dtype": "BF16",
        "seed": 123,
        "reference_voice": str(REFERENCE),
        "load_denoiser": False,
        "vae_device": "CPU",
        "full_onnx_generation": "NOT_IMPLEMENTED; ONNX WAVs decode PyTorch-generated latents only",
        "failure": failure,
        "runs": results,
    }
    (OUTPUT / "stage-f-results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "stage_f_complete", "result": report["result"], "runs": len(results), "failure": failure}))
    return 2 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
