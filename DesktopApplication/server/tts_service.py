from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator

import numpy as np

from server.config import settings
from server.audio_diagnostics import (
    audio_quality_metrics,
    shorten_internal_pauses,
    speech_timing_metrics,
    trim_audio_edges,
)
from server.tts_chunking import (
    MAX_CHUNK_STRETCH_RATE,
    MAX_PACE_RETRIES,
    MIN_CHUNK_STRETCH_RATE,
    SpeechChunk,
    apply_pronunciation_dictionary,
    join_speech_chunks,
    normalize_chunk_pace,
    split_speech_text,
)
from server.tts_engine import time_stretch_audio
from server.tts_jobs import GenerationJob


PACE_TARGETS = {"relaxed": 12.0, "natural": 14.0, "brisk": 16.0}
SUPPORTED_STEPS = {4, 8, 12}


@dataclass(frozen=True)
class GenerationOptions:
    text: str
    voice: str
    pace: str
    steps: int
    speed: float = 1.0
    pronunciation_dictionary: bool = True


def build_captions(
    chunks: list[SpeechChunk],
    audio_chunks: list[np.ndarray],
    sample_rate: int,
    duration_seconds: float,
) -> list[dict[str, Any]]:
    if sample_rate <= 0 or len(chunks) != len(audio_chunks):
        raise ValueError("Caption timing requires matching audio chunks and a positive sample rate.")
    weights = [len(audio) / sample_rate for audio in audio_chunks]
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("Caption timing requires non-empty audio.")

    scale = duration_seconds / total_weight
    captions: list[dict[str, Any]] = []
    cursor = 0.0
    for chunk, weight in zip(chunks, weights):
        start = cursor
        cursor = min(duration_seconds, cursor + weight * scale)
        captions.append(
            {
                "start_seconds": start,
                "end_seconds": cursor,
                "text": chunk.text,
            }
        )
    if captions:
        captions[-1]["end_seconds"] = duration_seconds
    return captions


def _caption_timestamp(seconds: float, separator: str) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02}{separator}{milliseconds:03}"


def format_subtitles(captions: list[dict[str, Any]], format_name: str) -> str:
    if format_name not in {"srt", "vtt"}:
        raise ValueError("Subtitle format must be srt or vtt.")
    lines = ["WEBVTT", ""] if format_name == "vtt" else []
    for index, caption in enumerate(captions, start=1):
        separator = "," if format_name == "srt" else "."
        start = _caption_timestamp(caption["start_seconds"], separator)
        end = _caption_timestamp(caption["end_seconds"], separator)
        if format_name == "srt":
            lines.extend((str(index), f"{start} --> {end}", caption["text"], ""))
        else:
            lines.extend((f"{start} --> {end}", caption["text"], ""))
    return "\n".join(lines)


async def stream_synthesis(
    options: GenerationOptions,
    engine: Any,
    job: GenerationJob,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    if options.pace not in PACE_TARGETS:
        raise ValueError("Choose Relaxed, Natural, or Brisk pace.")
    if options.steps not in SUPPORTED_STEPS:
        raise ValueError("Choose one of the supported inference step presets: 4, 8, or 12.")
    if not 0.75 <= options.speed <= 1.25:
        raise ValueError("Pitch-preserving speed must be between 0.75x and 1.25x.")
    if options.voice != "vanna":
        raise ValueError("The Vanna preset is the only installed reference voice.")

    chunks = split_speech_text(options.text)
    if not chunks:
        raise ValueError("Text did not produce any speech chunks.")
    if len(chunks) != job.total_chunks:
        job.total_chunks = len(chunks)

    reference_wav_path = settings.VOICE_REFERENCE_DIR / "vanna.wav"
    if not reference_wav_path.is_file():
        raise RuntimeError(f"The Vanna reference audio is missing: {reference_wav_path}")

    seed_material = json.dumps(
        {
            "text": options.text,
            "voice": options.voice,
            "pace": options.pace,
            "steps": options.steps,
            "speed": options.speed,
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big")
    target_cps = PACE_TARGETS[options.pace]
    job.status = "running"
    job.message = f"Preparing {len(chunks)} speech chunks"
    queue_started = time.perf_counter()
    yield "progress", job.as_response()

    audio_chunks: list[np.ndarray] = []
    chunk_durations: list[float] = []
    sample_rate: int | None = None
    fallback_reported = False
    for chunk_index, speech_chunk in enumerate(chunks):
        current = chunk_index + 1
        job.current_chunk = current
        job.message = f"Generating chunk {current} of {len(chunks)}"
        progress = job.as_response()
        progress["estimated_seconds_remaining"] = (
            sum(chunk_durations) / len(chunk_durations) * (len(chunks) - chunk_index)
            if chunk_durations
            else None
        )
        yield "progress", progress
        chunk_started = time.perf_counter()
        synth_text = apply_pronunciation_dictionary(
            speech_chunk.text,
            options.pronunciation_dictionary,
        )
        best_candidate: tuple[float, int, np.ndarray, np.ndarray, int] | None = None

        for attempt in range(MAX_PACE_RETRIES + 1):
            chunk_seed = (seed + chunk_index * (MAX_PACE_RETRIES + 1) + attempt) % (2**32)
            rate, raw_audio = await engine.generate_audio(
                text=synth_text,
                cfg_value=2.0,
                inference_timesteps=options.steps,
                reference_wav_path=str(reference_wav_path),
                speed=1.0,
                seed=chunk_seed,
            )
            status_reader = getattr(engine, "status", None)
            if callable(status_reader):
                engine_status = status_reader()
                fallback_reason = engine_status.get("fallback_reason")
                if (
                    not fallback_reported
                    and engine_status.get("model_device") == "cpu"
                    and isinstance(fallback_reason, str)
                    and "CUDA ran out of memory" in fallback_reason
                ):
                    fallback_reported = True
                    job.message = fallback_reason
                    yield "fallback", {
                        "message": fallback_reason,
                        "model_device": "cpu",
                    }
            if len(raw_audio) == 0:
                raise RuntimeError(f"VoxCPM2 returned empty audio for chunk {current}.")

            trimmed_audio = shorten_internal_pauses(
                trim_audio_edges(raw_audio, rate, padding_ms=60),
                rate,
            )
            speech_seconds = float(
                speech_timing_metrics(trimmed_audio, rate)["speech_seconds"]
            )
            if speech_seconds <= 0:
                raise RuntimeError(
                    f"No speech activity was detected in generated chunk {current}."
                )
            measured_cps = len(speech_chunk.text) / speech_seconds
            deviation = abs(measured_cps - target_cps) / target_cps
            candidate = (deviation, rate, raw_audio, trimmed_audio, chunk_seed)
            if best_candidate is None or deviation < best_candidate[0]:
                best_candidate = candidate

            requested_rate = target_cps / measured_cps
            can_stretch = MIN_CHUNK_STRETCH_RATE <= requested_rate <= MAX_CHUNK_STRETCH_RATE
            if deviation <= 0.12 or can_stretch or attempt >= MAX_PACE_RETRIES:
                break
            job.pace_retry_count += 1

        if best_candidate is None:
            raise RuntimeError(f"No usable audio candidate was produced for chunk {current}.")
        _, sample_rate, raw_audio, trimmed_audio, selected_seed = best_candidate
        metrics = audio_quality_metrics(raw_audio, sample_rate)
        paced_audio, measured_cps, stretch_rate, stretch_skipped = normalize_chunk_pace(
            audio=trimmed_audio,
            sample_rate=sample_rate,
            character_count=len(speech_chunk.text),
            target_cps=target_cps,
            pace_mode="auto",
            speed=1.0,
        )
        audio_chunks.append(paced_audio)
        chunk_elapsed = time.perf_counter() - chunk_started
        chunk_durations.append(chunk_elapsed)
        job.message = (
            f"Chunk {current} of {len(chunks)} ready · {measured_cps:.1f} CPS"
            + (" · pace adjustment skipped" if stretch_skipped else "")
        )
        elapsed = time.perf_counter() - queue_started
        if current == 1:
            job.first_audio_seconds = elapsed
        job.generation_seconds = elapsed
        preview = (
            time_stretch_audio(paced_audio, sample_rate, options.speed)
            if options.speed != 1.0
            else paced_audio
        )
        yield "chunk", {
            "job_id": str(job.job_id),
            "current_chunk": current,
            "total_chunks": len(chunks),
            "text": speech_chunk.text,
            "sample_rate": sample_rate,
            "audio": preview,
            "pace_cps": measured_cps,
            "pace_stretch_rate": stretch_rate,
            "pace_stretch_skipped": stretch_skipped,
            "raw_audio_duration_seconds": metrics["duration_seconds"],
            "seed": selected_seed,
            "chunk_seconds": chunk_elapsed,
        }

    if sample_rate is None:
        raise RuntimeError("The speech engine did not provide an audio sample rate.")

    joined_audio = join_speech_chunks(
        audio_chunks,
        [chunk.pause_after for chunk in chunks],
        sample_rate,
    )
    if options.speed != 1.0:
        joined_audio = time_stretch_audio(joined_audio, sample_rate, options.speed)
    duration_seconds = len(joined_audio) / sample_rate
    captions = build_captions(chunks, audio_chunks, sample_rate, duration_seconds)
    job.generation_seconds = time.perf_counter() - queue_started
    job.message = "Speech generation complete"
    yield "complete", {
        "job_id": str(job.job_id),
        "sample_rate": sample_rate,
        "audio": joined_audio,
        "duration_seconds": duration_seconds,
        "captions": captions,
    }
