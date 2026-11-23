import re
from dataclasses import dataclass
from math import pi
from typing import Literal

import numpy as np

from server.config import settings
from server.audio_diagnostics import speech_timing_metrics
from server.tts_engine import time_stretch_audio


PauseKind = Literal["continuation", "sentence", "paragraph"]

DEFAULT_TARGET_CPS = 14.0
MIN_CHUNK_CHARS = 250
MAX_CHUNK_CHARS = 450
FIRST_CHUNK_MAX_CHARS = 90
MAX_PACE_RETRIES = 1
MIN_STRETCH_RATE = 0.75
MAX_STRETCH_RATE = 1.25
MIN_CHUNK_STRETCH_RATE = MIN_STRETCH_RATE
MAX_CHUNK_STRETCH_RATE = MAX_STRETCH_RATE
CHUNK_PACE_TOLERANCE = 0.12
SENTENCE_PAUSE_MS = settings.TTS_SENTENCE_PAUSE_MS
PARAGRAPH_PAUSE_MS = settings.TTS_PARAGRAPH_PAUSE_MS
CROSSFADE_MS = settings.TTS_CROSSFADE_MS

PRONUNCIATION_REPLACEMENTS = (
    (re.compile(r"\bFull[- ]Stack\b", re.IGNORECASE), "ហ្វុលស្តាក់"),
    (re.compile(r"\bPortfolio\b", re.IGNORECASE), "ផតហ្វូលីយ៉ូ"),
    (re.compile(r"\bSource\s+Code\b", re.IGNORECASE), "សូសកូដ"),
)


@dataclass(frozen=True)
class SpeechChunk:
    text: str
    pause_after: PauseKind


def apply_pronunciation_dictionary(text: str, enabled: bool) -> str:
    if not enabled:
        return text
    for pattern, pronunciation in PRONUNCIATION_REPLACEMENTS:
        text = pattern.sub(pronunciation, text)
    return text


def _sentence_units(paragraph: str) -> list[str]:
    units: list[str] = []
    start = 0
    index = 0
    while index < len(paragraph):
        char = paragraph[index]
        boundary = char in "!?។៕"
        if char == "." and index + 1 < len(paragraph):
            boundary = paragraph[index + 1].isspace()
        if boundary:
            end = index + 1
            while end < len(paragraph) and paragraph[end] in "!?។៕":
                end += 1
            unit = paragraph[start:end].strip()
            if unit:
                units.append(unit)
            start = end
            index = end
        else:
            index += 1
    remainder = paragraph[start:].strip()
    if remainder:
        units.append(remainder)
    return units


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    remaining = sentence.strip()
    pieces: list[str] = []
    while len(remaining) > max_chars:
        cut = 0
        for index in range(max_chars, 0, -1):
            if remaining[index].isspace():
                cut = index
                break
            if remaining[index - 1] in ",،;":
                cut = index
                break
        if cut == 0:
            next_space = next(
                (index for index in range(max_chars + 1, len(remaining)) if remaining[index].isspace()),
                None,
            )
            if next_space is None:
                break
            cut = next_space
        piece = remaining[:cut].strip()
        if not piece:
            break
        pieces.append(piece)
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def split_speech_text(
    text: str,
    min_chars: int = MIN_CHUNK_CHARS,
    max_chars: int = MAX_CHUNK_CHARS,
    first_chunk_max_chars: int = FIRST_CHUNK_MAX_CHARS,
) -> list[SpeechChunk]:
    if min_chars < 1 or max_chars < min_chars:
        raise ValueError("Chunk limits must satisfy 1 <= min_chars <= max_chars.")
    if first_chunk_max_chars < 1 or first_chunk_max_chars > max_chars:
        raise ValueError("First chunk limit must be between 1 and max_chars.")

    paragraphs = [
        line.strip()
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        if line.strip()
    ]
    sentence_units: list[tuple[str, PauseKind]] = []
    for paragraph_index, paragraph in enumerate(paragraphs):
        sentences = _sentence_units(paragraph) or [paragraph]
        for sentence_index, sentence in enumerate(sentences):
            final_in_paragraph = sentence_index == len(sentences) - 1
            pause: PauseKind = (
                "paragraph"
                if final_in_paragraph and paragraph_index < len(paragraphs) - 1
                else "sentence"
                if final_in_paragraph
                else "sentence"
            )
            sentence_units.append((sentence, pause))

    if not sentence_units:
        return []

    chunks: list[SpeechChunk] = []
    first_sentence, first_pause = sentence_units.pop(0)
    if len(first_sentence) <= first_chunk_max_chars:
        first_text = first_sentence
    else:
        first_piece = _split_long_sentence(first_sentence, first_chunk_max_chars)
        first_text = first_piece.pop(0)
        remainder = " ".join(first_piece)
        if remainder:
            sentence_units.insert(0, (remainder, first_pause))
        first_pause = "continuation"
    chunks.append(SpeechChunk(first_text, first_pause))

    pending = ""
    pending_pause: PauseKind = "sentence"
    for sentence, pause in sentence_units:
        if pending and pending_pause == "paragraph":
            chunks.append(SpeechChunk(pending, pending_pause))
            pending = ""
        pieces = _split_long_sentence(sentence, max_chars)
        for piece_index, piece in enumerate(pieces):
            piece_pause: PauseKind = "continuation" if piece_index < len(pieces) - 1 else pause
            combined = f"{pending} {piece}".strip()
            if pending and len(combined) > max_chars:
                chunks.append(SpeechChunk(pending, pending_pause))
                pending = piece
                pending_pause = piece_pause
            else:
                pending = combined
                pending_pause = piece_pause
            if len(pending) >= min_chars:
                continue
    if pending:
        chunks.append(SpeechChunk(pending, pending_pause))

    return chunks


def normalize_chunk_pace(
    audio: np.ndarray,
    sample_rate: int,
    character_count: int,
    target_cps: float,
    pace_mode: str,
    speed: float,
) -> tuple[np.ndarray, float, float, bool]:
    if sample_rate <= 0 or character_count <= 0 or len(audio) == 0:
        raise ValueError("Pace normalization requires non-empty audio and text.")
    if pace_mode not in ("auto", "custom"):
        raise ValueError("pace_mode must be 'auto' or 'custom'.")

    speech_seconds = float(speech_timing_metrics(audio, sample_rate)["speech_seconds"])
    if speech_seconds <= 0:
        raise ValueError("Pace normalization requires detectable speech.")
    current_cps = character_count / speech_seconds
    desired_cps = target_cps
    deviation = abs(current_cps - desired_cps) / desired_cps
    if deviation <= CHUNK_PACE_TOLERANCE:
        return audio, current_cps, 1.0, False

    requested_rate = desired_cps / current_cps
    if not MIN_CHUNK_STRETCH_RATE <= requested_rate <= MAX_CHUNK_STRETCH_RATE:
        return audio, current_cps, 1.0, True

    rate = requested_rate
    normalized = time_stretch_audio(audio, sample_rate, rate)
    normalized_speech_seconds = float(
        speech_timing_metrics(normalized, sample_rate)["speech_seconds"]
    )
    final_cps = character_count / normalized_speech_seconds
    return normalized, final_cps, rate, False


def _fade(audio: np.ndarray, sample_count: int, fade_in: bool) -> np.ndarray:
    output = np.asarray(audio, dtype=np.float32).copy()
    length = min(sample_count, len(output))
    if length <= 0:
        return output
    ramp = np.linspace(0.0, 1.0, length, dtype=np.float32)
    if fade_in:
        output[:length] *= ramp
    else:
        output[-length:] *= ramp[::-1]
    return output


def join_speech_chunks(
    audio_chunks: list[np.ndarray],
    pause_kinds: list[PauseKind],
    sample_rate: int,
    crossfade_ms: int = CROSSFADE_MS,
) -> np.ndarray:
    if not audio_chunks or len(audio_chunks) != len(pause_kinds):
        raise ValueError("Each non-empty audio chunk must have a matching boundary.")
    if sample_rate <= 0 or not 30 <= crossfade_ms <= 60:
        raise ValueError("Use a positive sample rate and a 30–60 ms crossfade.")

    edge_samples = max(1, round(sample_rate * crossfade_ms / 1000))
    result = _fade(audio_chunks[0], edge_samples, True)
    for index in range(1, len(audio_chunks)):
        next_audio = np.asarray(audio_chunks[index], dtype=np.float32)
        boundary = pause_kinds[index - 1]
        pause_ms = PARAGRAPH_PAUSE_MS if boundary == "paragraph" else SENTENCE_PAUSE_MS
        if boundary == "continuation" or pause_ms == 0:
            overlap = min(edge_samples, len(result), len(next_audio))
            blend = np.linspace(0.0, 1.0, overlap, dtype=np.float32)
            left = result[-overlap:]
            right = next_audio[:overlap]
            mixed = left * np.cos(blend * pi / 2) + right * np.sin(blend * pi / 2)
            result = np.concatenate((result[:-overlap], mixed, next_audio[overlap:]))
            continue

        result = _fade(result, edge_samples, False)
        silence = np.zeros(round(sample_rate * pause_ms / 1000), dtype=np.float32)
        next_audio = _fade(next_audio, edge_samples, True)
        result = np.concatenate((result, silence, next_audio))

    return _fade(result, edge_samples, False)
