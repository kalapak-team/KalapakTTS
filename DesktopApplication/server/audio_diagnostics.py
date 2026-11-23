import numpy as np


def _speech_regions(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: int = 20,
    hop_ms: int = 10,
) -> list[tuple[int, int]]:
    signal = np.asarray(audio, dtype=np.float64)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    if signal.ndim != 1 or sample_rate <= 0 or signal.size == 0:
        raise ValueError("Speech timing requires non-empty mono audio and a positive sample rate.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("Speech timing cannot be computed for non-finite samples.")

    frame_size = max(1, round(sample_rate * frame_ms / 1000))
    hop_size = max(1, round(sample_rate * hop_ms / 1000))
    if signal.size < frame_size:
        frames = np.pad(signal, (0, frame_size - signal.size))[None, :]
    else:
        starts = list(range(0, signal.size - frame_size + 1, hop_size))
        if starts[-1] != signal.size - frame_size:
            starts.append(signal.size - frame_size)
        frames = np.stack([signal[start : start + frame_size] for start in starts])

    rms = np.sqrt(np.mean(frames**2, axis=1))
    threshold = max(float(rms.max()) * 10 ** (-35 / 20), 10 ** (-65 / 20))
    active = rms >= threshold
    gap_frames = max(1, round(50 / hop_ms))
    index = 0
    while index < len(active):
        if active[index]:
            index += 1
            continue
        gap_start = index
        while index < len(active) and not active[index]:
            index += 1
        if gap_start > 0 and index < len(active) and index - gap_start <= gap_frames:
            active[gap_start:index] = True

    regions: list[tuple[int, int]] = []
    active_indexes = np.flatnonzero(active)
    if not len(active_indexes):
        return regions
    start_index = previous_index = int(active_indexes[0])
    for current in active_indexes[1:]:
        current_index = int(current)
        if current_index > previous_index + 1:
            start_sample = max(0, start_index * hop_size)
            end_sample = min(signal.size, previous_index * hop_size + frame_size)
            regions.append((start_sample, end_sample))
            start_index = current_index
        previous_index = current_index
    regions.append(
        (
            max(0, start_index * hop_size),
            min(signal.size, previous_index * hop_size + frame_size),
        )
    )
    return regions


def trim_audio_edges(
    audio: np.ndarray,
    sample_rate: int,
    padding_ms: int = 40,
) -> np.ndarray:
    signal = np.asarray(audio, dtype=np.float32)
    regions = _speech_regions(signal, sample_rate)
    if not regions:
        return signal.copy()
    padding = round(sample_rate * padding_ms / 1000)
    start = max(0, regions[0][0] - padding)
    end = min(len(signal), regions[-1][1] + padding)
    return signal[start:end].copy()


def shorten_internal_pauses(
    audio: np.ndarray,
    sample_rate: int,
    max_pause_ms: int = 160,
) -> np.ndarray:
    signal = np.asarray(audio, dtype=np.float32)
    if max_pause_ms < 0:
        raise ValueError("Maximum pause duration must be non-negative.")
    regions = _speech_regions(signal, sample_rate)
    if len(regions) < 2:
        return signal.copy()

    max_gap = round(sample_rate * max_pause_ms / 1000)
    pieces: list[np.ndarray] = []
    cursor = 0
    for (_, end), (next_start, _) in zip(regions, regions[1:]):
        gap = next_start - end
        if gap <= max_gap:
            continue
        cut_start = end + (gap - max_gap) // 2
        cut_end = cut_start + gap - max_gap
        pieces.append(signal[cursor:cut_start])
        cursor = cut_end
    pieces.append(signal[cursor:])
    return np.concatenate(pieces)


def speech_timing_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, float | int]:
    signal = np.asarray(audio, dtype=np.float32)
    regions = _speech_regions(signal, sample_rate)
    speech_seconds = sum(end - start for start, end in regions) / sample_rate
    pauses = [
        (next_start - end) / sample_rate
        for (_, end), (next_start, _) in zip(regions, regions[1:])
        if (next_start - end) / sample_rate >= 0.25
    ]
    return {
        "duration_seconds": len(signal) / sample_rate,
        "speech_seconds": speech_seconds,
        "pause_count_250ms": len(pauses),
        "pause_seconds_250ms": sum(pauses),
    }


def audio_quality_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    signal = np.asarray(audio, dtype=np.float64)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    if signal.ndim != 1 or sample_rate <= 0 or signal.size == 0:
        raise ValueError("Audio metrics require non-empty mono audio and a positive sample rate.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("Audio metrics cannot be computed for non-finite samples.")

    frame_size = min(2048, 1 << (signal.size.bit_length() - 1))
    frame_size = max(frame_size, 2)
    hop_size = max(1, frame_size // 2)
    if signal.size < frame_size:
        frames = np.pad(signal, (0, frame_size - signal.size))[None, :]
    else:
        starts = list(range(0, signal.size - frame_size + 1, hop_size))
        if starts[-1] != signal.size - frame_size:
            starts.append(signal.size - frame_size)
        frames = np.stack([signal[start : start + frame_size] for start in starts])

    window = np.hanning(frame_size)
    magnitude = np.abs(np.fft.rfft(frames * window, axis=1))
    power = magnitude**2
    mean_magnitude = magnitude.mean(axis=0)
    mean_power = power.mean(axis=0)
    frequencies = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    total_power = float(mean_power.sum())
    total_magnitude = float(mean_magnitude.sum())
    rms = float(np.sqrt(np.mean(signal**2)))
    epsilon = np.finfo(np.float64).tiny
    flatness_by_frame = np.exp(np.mean(np.log(np.maximum(power, epsilon)), axis=1)) / np.maximum(
        power.mean(axis=1), epsilon
    )

    return {
        "duration_seconds": float(signal.size / sample_rate),
        "rms_dbfs": float(20.0 * np.log10(max(rms, epsilon))),
        "high_frequency_energy_ratio": (
            float(mean_power[frequencies > 4000].sum() / total_power)
            if total_power > 0
            else 0.0
        ),
        "spectral_centroid_hz": (
            float(np.dot(frequencies, mean_magnitude) / total_magnitude)
            if total_magnitude > 0
            else 0.0
        ),
        "spectral_flatness": float(flatness_by_frame.mean()),
    }
