from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import httpx


DESKTOP_ROOT = Path(__file__).resolve().parents[1]
LONG_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "khmer_long_sample.txt"
MODEL_LOAD_TIMEOUT_SECONDS = 180
GPU_GENERATION_TIMEOUT_SECONDS = 420
CPU_GENERATION_TIMEOUT_SECONDS = 900
CPU_MODEL_LOAD_TIMEOUT_SECONDS = 600
WEB_BENCHMARK_REFERENCE = {
    "source": "user-supplied web GPU benchmark",
    "steps": 4,
    "total_seconds": 191.0,
    "first_audio_seconds": 7.8,
    "rtf": 2.6,
}


def read_gpu_memory() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise RuntimeError("nvidia-smi is unavailable; GPU measurements cannot be collected.")
    result = subprocess.run(
        [
            executable,
            "--query-gpu=name,memory.used,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        check=True,
        text=True,
        timeout=5,
    )
    fields = [field.strip() for field in result.stdout.splitlines()[0].split(",")]
    if len(fields) != 4:
        raise RuntimeError(f"Unexpected nvidia-smi output: {result.stdout.strip()}")
    return {
        "gpu_name": fields[0],
        "used_mib": int(fields[1]),
        "free_mib": int(fields[2]),
        "total_mib": int(fields[3]),
    }


class GpuMonitor:
    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gpu-monitor", daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(read_gpu_memory())
            except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                self.error = exc
                self._stop.set()
                return
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, int]:
        self._stop.set()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("The GPU measurement thread did not stop.")
        if self.error is not None:
            raise RuntimeError(f"GPU monitoring failed: {self.error}") from self.error
        if not self.samples:
            raise RuntimeError("The GPU monitor did not collect any samples.")
        return {
            "peak_used_mib": max(sample["used_mib"] for sample in self.samples),
            "minimum_free_mib": min(sample["free_mib"] for sample in self.samples),
            "sample_count": len(self.samples),
        }


def generate(
    client: httpx.Client,
    base_url: str,
    text: str,
    name: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    first_audio_seconds: float | None = None
    result: dict[str, Any] | None = None
    event_name = ""
    with client.stream(
        "POST",
        f"{base_url}/api/tts/generate",
        json={
            "text": text,
            "voice": "vanna",
            "pace": "natural",
            "steps": 4,
            "speed": 1.0,
            "pronunciation_dictionary": True,
        },
        timeout=httpx.Timeout(timeout_seconds, connect=10),
    ) as response:
        if response.status_code != 200:
            response.read()
            raise RuntimeError(
                f"{name} generation failed (HTTP {response.status_code}): {response.text}"
            )
        for line in response.iter_lines():
            if time.perf_counter() - started > timeout_seconds:
                raise TimeoutError(
                    f"{name} generation exceeded its {timeout_seconds}-second time limit."
                )
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                payload_text = line[5:].strip()
                if event_name == "chunk" and first_audio_seconds is None:
                    first_audio_seconds = time.perf_counter() - started
                elif event_name == "complete":
                    result = json.loads(payload_text)
                elif event_name == "error":
                    payload = json.loads(payload_text)
                    raise RuntimeError(
                        f"{name} generation failed: {payload.get('error', 'unknown error')}"
                    )

    if result is None:
        raise RuntimeError(f"{name} generation ended without a completion event.")
    total_seconds = time.perf_counter() - started
    duration_seconds = float(result["duration_seconds"])
    return {
        "name": name,
        "text_characters": len(text),
        "steps": 4,
        "first_audio_seconds": first_audio_seconds,
        "total_seconds": total_seconds,
        "audio_duration_seconds": duration_seconds,
        "rtf": total_seconds / duration_seconds if duration_seconds else None,
        "audio_sha256": result["audio_sha256"],
        "audio_url": result["audio_url"],
        "srt_url": result["srt_url"],
        "vtt_url": result["vtt_url"],
    }


def export_assets(
    client: httpx.Client,
    base_url: str,
    result: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    exported: dict[str, Any] = {}
    for extension, url_key in (
        ("wav", "audio_url"),
        ("srt", "srt_url"),
        ("vtt", "vtt_url"),
    ):
        response = client.get(f"{base_url}{result[url_key]}", timeout=60)
        if response.status_code != 200:
            raise RuntimeError(
                f"{name} {extension.upper()} export failed "
                f"(HTTP {response.status_code}): {response.text}"
            )
        destination = DESKTOP_ROOT / "build" / f"stage5-{name}.{extension}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(response.content)
        exported[extension] = {
            "path": str(destination),
            "size_bytes": destination.stat().st_size,
            "sha256": hashlib.sha256(response.content).hexdigest(),
        }
    if exported["wav"]["sha256"] != result["audio_sha256"]:
        raise RuntimeError(f"{name} WAV export SHA-256 does not match the saved generation.")
    return exported


def engine_status(base_url: str) -> dict[str, Any]:
    response = httpx.get(f"{base_url}/api/tts/status", timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"TTS status check failed (HTTP {response.status_code}): {response.text}")
    return response.json()


def measured_generation(
    client: httpx.Client,
    base_url: str,
    text: str,
    name: str,
    timeout_seconds: int,
) -> tuple[dict[str, Any], dict[str, int], dict[str, Any]]:
    monitor = GpuMonitor()
    monitor.start()
    try:
        result = generate(client, base_url, text, name, timeout_seconds)
    finally:
        nvidia_peak = monitor.stop()
    engine_after = engine_status(base_url)
    result["generation_device"] = engine_after.get("model_device")
    result["torch_peak_vram_allocated_bytes"] = engine_after.get("peak_vram_bytes")
    result["torch_peak_vram_reserved_bytes"] = engine_after.get("peak_vram_reserved_bytes")
    result["torch_free_after_generation_bytes"] = engine_after.get("vram_free_bytes")
    result["process_ram_working_set_bytes"] = engine_after.get("working_set_bytes")
    result["process_peak_ram_bytes"] = engine_after.get("peak_ram_bytes")
    result["nvidia_smi_gpu_peak"] = nvidia_peak
    result["time_limit_seconds"] = timeout_seconds
    return result, nvidia_peak, engine_after


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the Stage 5 desktop TTS runtime.")
    parser.add_argument("--base-url", required=True, help="Local desktop API URL printed by launcher.py.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--short-only", action="store_true")
    parser.add_argument(
        "--try-gpu-anyway",
        action="store_true",
        help="Permit a controlled CUDA attempt below the measured free-VRAM threshold.",
    )
    parser.add_argument("--app-startup-seconds", type=float)
    parser.add_argument(
        "--output",
        default=str(DESKTOP_ROOT / "build" / "stage5-benchmark.json"),
        help="Path for the local JSON measurements.",
    )
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    output_path = Path(args.output).resolve()
    if not output_path.is_relative_to(DESKTOP_ROOT):
        raise ValueError("Benchmark output must be written inside DesktopApplication.")
    if not args.short_only and not LONG_FIXTURE.is_file():
        raise FileNotFoundError(f"Required long fixture is missing: {LONG_FIXTURE}")

    calibration_response = httpx.get(f"{base_url}/api/system/gpu", timeout=30)
    if calibration_response.status_code != 200:
        raise RuntimeError(
            "GPU calibration check failed "
            f"(HTTP {calibration_response.status_code}): {calibration_response.text}"
        )
    gpu_calibration_before_load = calibration_response.json()
    before_load_nvidia_smi = read_gpu_memory()
    load_timeout_seconds = (
        CPU_MODEL_LOAD_TIMEOUT_SECONDS if args.device == "cpu" else MODEL_LOAD_TIMEOUT_SECONDS
    )
    started = time.perf_counter()
    try:
        load_response = httpx.post(
            f"{base_url}/api/engine/load",
            json={"device": args.device, "try_gpu_anyway": args.try_gpu_anyway},
            timeout=httpx.Timeout(load_timeout_seconds, connect=10),
        )
    except httpx.TimeoutException as exc:
        raise TimeoutError(
            f"{args.device.upper()} model load exceeded its {load_timeout_seconds}-second limit."
        ) from exc
    if load_response.status_code != 200:
        raise RuntimeError(
            f"{args.device.upper()} engine load failed "
            f"(HTTP {load_response.status_code}): {load_response.text}"
        )
    load_seconds = time.perf_counter() - started
    engine_after_load = load_response.json()
    idle_gpu_nvidia_smi = read_gpu_memory()
    short_text = "សួស្តី! នេះជាការសាកល្បងសំឡេងខ្មែររបស់ KalapakTTS។"
    generation_timeout = (
        CPU_GENERATION_TIMEOUT_SECONDS
        if engine_after_load.get("model_device") == "cpu"
        else GPU_GENERATION_TIMEOUT_SECONDS
    )

    short_result: dict[str, Any]
    short_gpu_peak: dict[str, int]
    engine_after_short: dict[str, Any]
    long_result: dict[str, Any] | None = None
    long_gpu_peak: dict[str, int] | None = None
    engine_after_long: dict[str, Any] | None = None
    with httpx.Client() as client:
        short_result, short_gpu_peak, engine_after_short = measured_generation(
            client,
            base_url,
            short_text,
            "short",
            generation_timeout,
        )
        short_result["exports"] = export_assets(client, base_url, short_result, "short")
        if not args.short_only:
            long_text = LONG_FIXTURE.read_text(encoding="utf-8")
            long_timeout = (
                CPU_GENERATION_TIMEOUT_SECONDS
                if engine_after_short.get("model_device") == "cpu"
                else GPU_GENERATION_TIMEOUT_SECONDS
            )
            generated_long_result, long_gpu_peak, engine_after_long = measured_generation(
                client,
                base_url,
                long_text,
                "khmer_long_sample",
                long_timeout,
            )
            long_result = generated_long_result
            long_result["exports"] = export_assets(
                client,
                base_url,
                long_result,
                "khmer_long_sample",
            )

    measurement = {
        "classification": "MEASURED",
        "desktop_api_url": base_url,
        "requested_device": args.device,
        "try_gpu_anyway": args.try_gpu_anyway,
        "app_startup_seconds": args.app_startup_seconds,
        "gpu_calibration_before_model_load": gpu_calibration_before_load,
        "gpu_before_model_load_nvidia_smi": before_load_nvidia_smi,
        "model_load_timeout_seconds": load_timeout_seconds,
        "model_load_seconds": load_seconds,
        "model_idle_gpu_nvidia_smi": idle_gpu_nvidia_smi,
        "engine_status_after_load": engine_after_load,
        "model_idle_ram_working_set_bytes": engine_after_load.get("working_set_bytes"),
        "model_idle_ram_peak_bytes": engine_after_load.get("peak_ram_bytes"),
        "short_generation": short_result,
        "short_generation_gpu_peak_nvidia_smi": short_gpu_peak,
        "engine_status_after_short_generation": engine_after_short,
        "long_generation": long_result,
        "long_generation_gpu_peak_nvidia_smi": long_gpu_peak,
        "engine_status_after_long_generation": engine_after_long,
        "generation_timeout_seconds": generation_timeout,
        "web_benchmark_comparison": WEB_BENCHMARK_REFERENCE,
        "measurement_note": (
            "The local API was exercised while the desktop WebView window was open. "
            "The web TTS service and KamVox must remain stopped for the entire measurement. "
            "nvidia-smi and CUDA mem_get_info are recorded separately because their free-memory "
            "readings differ on this Windows/WDDM setup."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(measurement, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(measurement, ensure_ascii=False, indent=2))
    print(f"Benchmark JSON: {output_path}")


if __name__ == "__main__":
    main()
