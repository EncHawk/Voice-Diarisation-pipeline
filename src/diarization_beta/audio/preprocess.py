"""Audio preprocessing: normalize / trim / chunk helpers."""
from __future__ import annotations

import numpy as np


def normalize(audio: np.ndarray) -> np.ndarray:
    """Peak normalize to [-1, 1] without clipping source."""
    if audio.size == 0:
        return audio
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak * 0.95
    return audio.astype(np.float32)


def chunk_audio(audio: np.ndarray, sr: int, chunk_duration: float = 30.0, overlap: float = 0.5):
    """Yield (chunk, start_time) with overlap."""
    chunk_samples = int(chunk_duration * sr)
    hop = int((chunk_duration - overlap) * sr)
    if chunk_samples <= 0:
        yield audio, 0.0
        return
    n = len(audio)
    start = 0
    while start < n:
        end = min(start + chunk_samples, n)
        yield audio[start:end], start / sr
        if end >= n:
            break
        start += hop


def slice_segment(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    s = max(0, int(start * sr))
    e = min(len(audio), int(end * sr))
    return audio[s:e]
