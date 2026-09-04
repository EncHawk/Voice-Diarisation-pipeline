"""Beta: binary voice-gender estimation from fundamental frequency (spec §10.1).

Uses only pitch statistics computed from the speaker's own segments.  Adult
male voices centre roughly 85-180 Hz and adult female voices 165-255 Hz, so
the medians are compared against a configurable ambiguous band instead of a
hard cut.  No model, no training data, no recording-specific tuning.
"""
from __future__ import annotations

import numpy as np

DEFAULTS = {
    "enabled": True,
    "male_max_hz": 160.0,
    "female_min_hz": 180.0,
    "f0_min_hz": 60.0,
    "f0_max_hz": 400.0,
    "clarity_threshold": 0.50,
    "frame_size": 1024,
    "hop_size": 160,
    "max_seconds_per_speaker": 45.0,
    "min_voiced_frames": 10,
}


def _frame_autocorrelation(frame: np.ndarray) -> np.ndarray:
    """Biased normalized autocorrelation of one frame via FFT."""
    size = len(frame)
    spectrum = np.fft.rfft(frame, n=2 * size)
    power = np.fft.irfft(spectrum * np.conj(spectrum))[:size]
    if power[0] <= 0:
        return np.zeros(size)
    return power / power[0]


def _voiced_f0s(audio: np.ndarray, sr: int, start: float, end: float, cfg: dict) -> np.ndarray:
    """Yield per-frame F0 estimates (Hz) for one time span."""
    frame_size = int(cfg["frame_size"])
    hop = int(cfg["hop_size"])
    lag_min = max(2, int(sr / cfg["f0_max_hz"]))
    lag_max = min(frame_size - 1, int(sr / cfg["f0_min_hz"]))
    begin = max(0, int(start * sr))
    stop = min(len(audio), int(end * sr))
    f0s = []
    for pos in range(begin, stop - frame_size, hop):
        frame = audio[pos : pos + frame_size]
        if np.sqrt(np.mean(frame.astype(np.float64) ** 2)) < 0.01:
            continue
        corr = _frame_autocorrelation(frame)
        window = corr[lag_min : lag_max + 1]
        if len(window) == 0:
            continue
        peak_index = int(np.argmax(window)) + lag_min
        clarity = corr[peak_index]
        if clarity < cfg["clarity_threshold"]:
            continue
        f0 = sr / peak_index
        f0s.append(f0)
    return np.asarray(f0s)


def classify_gender(median_f0: float | None, male_max_hz: float, female_min_hz: float) -> str | None:
    if median_f0 is None:
        return None
    if median_f0 <= male_max_hz:
        return "male"
    if median_f0 >= female_min_hz:
        return "female"
    return None  # ambiguous band


def estimate_speaker_f0(
    audio: np.ndarray,
    sr: int,
    spans: list[tuple[float, float]],
    cfg: dict | None = None,
) -> float | None:
    """Median F0 across a speaker's spans, capped to keep the cost bounded."""
    cfg = {**DEFAULTS, **(cfg or {})}
    max_samples = int(cfg["max_seconds_per_speaker"] * sr)
    collected: list[np.ndarray] = []
    total = 0
    for start, end in spans:
        if total >= max_samples:
            break
        clipped_end = min(end, start + (max_samples - total) / sr)
        f0s = _voiced_f0s(audio, sr, start, clipped_end, cfg)
        total += int(max(0.0, clipped_end - start) * sr)
        if len(f0s):
            collected.append(f0s)
    if not collected or sum(len(f) for f in collected) < int(cfg["min_voiced_frames"]):
        return None
    return float(np.median(np.concatenate(collected)))


def classify_speakers(
    audio: np.ndarray,
    sr: int,
    speaker_spans: dict[str, list[tuple[float, float]]],
    cfg: dict | None = None,
) -> dict[str, dict]:
    """Return {speaker_id: {"gender": "male"/"female"/None, "median_f0_hz": float|None}}."""
    cfg = {**DEFAULTS, **(cfg or {})}
    results: dict[str, dict] = {}
    if not cfg.get("enabled", True):
        return {sid: {"gender": None, "median_f0_hz": None} for sid in speaker_spans}
    for sid, spans in speaker_spans.items():
        median_f0 = estimate_speaker_f0(audio, sr, spans, cfg)
        results[sid] = {
            "gender": classify_gender(median_f0, cfg["male_max_hz"], cfg["female_min_hz"]),
            "median_f0_hz": round(median_f0, 1) if median_f0 else None,
        }
    return results


def assign_generic_labels(
    speaker_ids: list[str],
    gender_map: dict[str, dict],
    first_start: dict[str, float],
) -> dict[str, str]:
    """Map speakers to beta generic labels: Man N / Woman N / Speaker N.

    Numbering follows each group's order of first speech so the labels read
    naturally in the transcript.  Speakers without a confident gender keep
    the neutral Speaker N form.
    """
    counters = {"male": 0, "female": 0}
    labels: dict[str, str] = {}
    ordered = sorted(speaker_ids, key=lambda sid: first_start.get(sid, float("inf")))
    for sid in ordered:
        gender = (gender_map.get(sid) or {}).get("gender")
        if gender == "male":
            counters["male"] += 1
            labels[sid] = f"Man {counters['male']}"
        elif gender == "female":
            counters["female"] += 1
            labels[sid] = f"Woman {counters['female']}"
        else:
            digits = "".join(ch for ch in sid if ch.isdigit())
            labels[sid] = f"Speaker {int(digits)}" if digits else sid.replace("speaker_", "Speaker ")
    return labels
