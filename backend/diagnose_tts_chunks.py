import argparse
import asyncio
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.api.tts import _synthesize_request
from app.core.audio_diagnostics import (
    audio_quality_metrics,
    speech_timing_metrics,
)
from app.core.tts_chunking import SpeechChunk
from app.schemas.generation import GenerationCreate


BACKEND_DIR = Path(__file__).resolve().parent
FIXTURE_PATH = BACKEND_DIR / "tests" / "fixtures" / "khmer_long_sample.txt"
VANNA_PATH = BACKEND_DIR / "voice_references" / "vanna.wav"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and measure raw TTS audio before pace processing."
    )
    parser.add_argument("--text-file", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-audio", type=Path, default=VANNA_PATH)
    parser.add_argument("--reference-transcript", type=Path)
    parser.add_argument("--control-instruction", default="")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--pace-mode", choices=("auto", "custom"), default="custom")
    parser.add_argument("--target-cps", type=float, default=18.0)
    parser.add_argument("--pronunciation-dictionary", action="store_true")
    parser.add_argument("--style")
    return parser.parse_args()


async def diagnose(args: argparse.Namespace) -> None:
    if not args.text_file.is_file():
        raise FileNotFoundError(f"Text fixture not found: {args.text_file}")
    if not args.reference_audio.is_file():
        raise FileNotFoundError(f"Reference audio not found: {args.reference_audio}")
    text = args.text_file.read_text(encoding="utf-8").strip()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = GenerationCreate(
        text=text,
        cfg_value=args.cfg_value,
        inference_timesteps=args.steps,
        seed=args.seed,
        speed=args.speed,
        pace_mode=args.pace_mode,
        target_cps=args.target_cps,
        pronunciation_dictionary=args.pronunciation_dictionary,
        style=args.style,
    )
    rows: list[dict[str, Any]] = []

    def save_raw_audio(
        _index: int,
        _total: int,
        chunk: SpeechChunk,
        sample_rate: int,
        audio: np.ndarray,
    ) -> None:
        filename = "raw.wav"
        sf.write(str(args.output_dir / filename), audio, sample_rate)
        speech_metrics = speech_timing_metrics(audio, sample_rate)
        rows.append(
            {
                "mode": "single_pass",
                "characters": len(chunk.text),
                "text": chunk.text,
                "file": filename,
                **audio_quality_metrics(audio, sample_rate),
                **speech_metrics,
                "speech_only_chars_per_second": len(chunk.text)
                / max(float(speech_metrics["speech_seconds"]), 1e-9),
                "steps": args.steps,
                "cfg_value": args.cfg_value,
                "seed": args.seed,
                "speed": args.speed,
                "pace_mode": args.pace_mode,
                "target_cps": args.target_cps,
                "pronunciation_dictionary": args.pronunciation_dictionary,
                "style": args.style or "",
                "reference_audio": str(args.reference_audio),
                "control_instruction": args.control_instruction,
            }
        )
        print(f"Saved raw single-pass audio: {len(chunk.text)} characters")

    def report_progress(index: int, total: int, message: str) -> None:
        print(message)

    def save_processed_audio(
        _index: int,
        _total: int,
        sample_rate: int,
        audio: np.ndarray,
        _pause_after: str,
        chunk_text: str,
    ) -> None:
        filename = "processed.wav"
        sf.write(str(args.output_dir / filename), audio, sample_rate)
        print(f"Saved processed single-pass audio: {len(chunk_text)} chars")

    sample_rate, final_audio = await _synthesize_request(
        payload,
        str(args.reference_audio),
        args.control_instruction,
        on_progress=report_progress,
        on_audio=save_processed_audio,
        on_raw_chunk=save_raw_audio,
    )
    final_path = args.output_dir / "final.wav"
    sf.write(str(final_path), final_audio, sample_rate)
    final_metrics = {
        **audio_quality_metrics(final_audio, sample_rate),
        **speech_timing_metrics(final_audio, sample_rate),
        "text_characters": len(text),
        "total_chars_per_second": len(text) / (len(final_audio) / sample_rate),
        "speech_only_chars_per_second": len(text)
        / max(float(speech_timing_metrics(final_audio, sample_rate)["speech_seconds"]), 1e-9),
        "steps": args.steps,
        "cfg_value": args.cfg_value,
        "seed": args.seed,
        "speed": args.speed,
        "pace_mode": args.pace_mode,
        "target_cps": args.target_cps,
        "style": args.style,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    reference_audio, reference_rate = sf.read(
        str(args.reference_audio),
        dtype="float32",
        always_2d=False,
    )
    reference_metrics = speech_timing_metrics(reference_audio, reference_rate)
    reference_result: dict[str, object] = {
        "reference_audio": str(args.reference_audio),
        **reference_metrics,
    }
    if args.reference_transcript:
        transcript = args.reference_transcript.read_text(encoding="utf-8").strip()
        if not transcript:
            raise ValueError("Reference transcript file is empty.")
        reference_result["transcript_characters"] = len(transcript)
        reference_result["speech_only_chars_per_second"] = (
            len(transcript) / max(float(reference_metrics["speech_seconds"]), 1e-9)
        )
    (args.output_dir / "reference_voice_metrics.json").write_text(
        json.dumps(reference_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metrics_path = args.output_dir / "metrics.csv"
    fields = list(rows[0]) if rows else []
    with metrics_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))
    print(json.dumps(reference_result, ensure_ascii=False, indent=2))
    print(f"Saved single-pass raw audio and metrics to {args.output_dir.resolve()}")


if __name__ == "__main__":
    asyncio.run(diagnose(parse_arguments()))
