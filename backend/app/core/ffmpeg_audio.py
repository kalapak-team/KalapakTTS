import io
import shutil
import subprocess

import numpy as np
import soundfile as sf


def apply_ffmpeg_audio_filter(
    audio: np.ndarray,
    sample_rate: int,
    filter_expression: str,
    operation: str,
) -> np.ndarray:
    signal = np.asarray(audio, dtype=np.float32)
    if (
        sample_rate <= 0
        or signal.ndim != 1
        or signal.size == 0
        or not np.all(np.isfinite(signal))
    ):
        raise ValueError(f"{operation.capitalize()} requires finite, non-empty mono audio.")
    if not filter_expression.strip():
        raise ValueError("An FFmpeg audio filter is required.")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            f"FFmpeg is required for {operation} but was not found on PATH."
        )

    source_buffer = io.BytesIO()
    sf.write(source_buffer, signal, sample_rate, format="WAV", subtype="FLOAT")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "wav",
        "-i",
        "pipe:0",
        "-af",
        filter_expression,
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-c:a",
        "pcm_f32le",
        "-f",
        "wav",
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            command,
            input=source_buffer.getvalue(),
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"Could not run FFmpeg for {operation}.") from exc
    if result.returncode != 0:
        details = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"FFmpeg {operation} failed: {details or 'unknown error'}"
        )

    try:
        filtered, output_rate = sf.read(
            io.BytesIO(result.stdout),
            dtype="float32",
            always_2d=False,
        )
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(f"FFmpeg returned invalid audio for {operation}.") from exc
    if (
        output_rate != sample_rate
        or filtered.ndim != 1
        or filtered.size == 0
        or not np.all(np.isfinite(filtered))
    ):
        raise RuntimeError(f"FFmpeg returned audio with an unexpected format for {operation}.")
    return filtered


def match_audio_rms(audio: np.ndarray, reference_audio: np.ndarray) -> np.ndarray:
    processed = np.asarray(audio, dtype=np.float32)
    reference = np.asarray(reference_audio, dtype=np.float32)
    if (
        processed.ndim != 1
        or processed.size == 0
        or reference.ndim != 1
        or reference.size == 0
        or not np.all(np.isfinite(processed))
        or not np.all(np.isfinite(reference))
    ):
        raise ValueError("RMS matching requires finite, non-empty mono audio.")

    reference_rms = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
    processed_rms = float(np.sqrt(np.mean(processed.astype(np.float64) ** 2)))
    if reference_rms > 0.0 and processed_rms > 0.0:
        gain = reference_rms / processed_rms
        peak = float(np.max(np.abs(processed))) * gain
        if peak > 0.98:
            gain *= 0.98 / peak
        processed = (processed * gain).astype(np.float32)
    return processed
