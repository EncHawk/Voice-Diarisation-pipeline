"""VAD — Silero VAD (ONNX) with energy-based fallback."""
from __future__ import annotations

import dataclasses
from typing import List

import numpy as np


@dataclasses.dataclass
class VADSegment:
    start: float
    end: float
    speech_probability: float = 1.0

    def to_dict(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "speech_probability": round(self.speech_probability, 3)}


class VADDetector:
    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
        window_size_samples: int = 512,
        sample_rate: int = 16000,
        model: str = "silero_vad",
    ):
        self.threshold = threshold
        self.min_speech_ms = min_speech_duration_ms
        self.min_silence_ms = min_silence_duration_ms
        self.pad_ms = speech_pad_ms
        self.window = window_size_samples
        self.sr = sample_rate
        self.model = model
        self._silero = None  # lazy

    # -- silero --
    def _try_load_silero(self):
        if self._silero is not None:
            return self._silero
        try:
            from silero_vad import load_silero_vad

            model = load_silero_vad(onnx=True)
            self._silero = model
            return model
        except Exception:
            self._silero = False
            return None

    def _silero_detect(self, audio: np.ndarray) -> List[VADSegment] | None:
        model = self._try_load_silero()
        if not model:
            return None
        try:
            # silero_vad v5 ONNX accepts float32 numpy directly (no torch needed)
            from silero_vad import get_speech_timestamps

            stamps = get_speech_timestamps(
                audio.astype(np.float32),
                model,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_ms,
                min_silence_duration_ms=self.min_silence_ms,
                speech_pad_ms=self.pad_ms,
                window_size_samples=self.window,
                sampling_rate=self.sr,
            )
            segs = []
            for s in stamps:
                start = s["start"] / self.sr
                end = s["end"] / self.sr
                # approximate probability as 0.96 if present; real model returns probs separately
                segs.append(VADSegment(start=start, end=end, speech_probability=0.96))
            return segs
        except Exception:
            return None

    # -- energy fallback (works well for clean TTS) --
    def _energy_detect(self, audio: np.ndarray) -> List[VADSegment]:
        # Frame energy with 25ms window, 10ms hop
        frame_len = int(0.025 * self.sr)
        hop = int(0.010 * self.sr)
        if len(audio) < frame_len:
            if np.max(np.abs(audio)) > 0.01:
                return [VADSegment(0.0, len(audio) / self.sr, 0.9)]
            return []
        # RMS energy
        n_frames = 1 + (len(audio) - frame_len) // hop
        energies = np.array([np.sqrt(np.mean(audio[i * hop : i * hop + frame_len] ** 2) + 1e-10) for i in range(n_frames)])
        # Adaptive threshold: median + k * std, but with floor
        med = np.median(energies)
        std = np.std(energies)
        thr = max(med + 0.5 * std, 0.02, self.threshold * 0.08)
        # Also absolute floor scaled by peak
        peak = np.max(energies)
        thr = min(thr, peak * 0.15) if peak > 0 else thr

        is_speech = energies > thr
        # Morphological smoothing: remove short silences/speech blips
        min_speech_frames = max(1, int(self.min_speech_ms / 10))
        min_silence_frames = max(1, int(self.min_silence_ms / 10))

        # Merge
        segments = []
        i = 0
        while i < len(is_speech):
            if not is_speech[i]:
                i += 1
                continue
            j = i
            while j < len(is_speech) and is_speech[j]:
                j += 1
            # check gap to next speech for min_silence merging
            # merge silences shorter than min_silence_frames if surrounded
            k = j
            while k < len(is_speech) and not is_speech[k]:
                k += 1
            gap = k - j
            if gap > 0 and gap < min_silence_frames and k < len(is_speech):
                # merge across short silence
                j = k
                while j < len(is_speech) and is_speech[j]:
                    j += 1
            dur_frames = j - i
            if dur_frames >= min_speech_frames:
                start = i * hop / self.sr
                end = min(j * hop + frame_len, len(audio)) / self.sr
                # pad
                pad = self.pad_ms / 1000.0
                start = max(0, start - pad)
                end = min(len(audio) / self.sr, end + pad)
                # speech prob as normalized energy
                prob = float(np.clip(np.mean(energies[i:j]) / (peak + 1e-6) * 1.2, 0.5, 0.99))
                segments.append(VADSegment(start=start, end=end, speech_probability=prob))
            i = j
        # Merge overlapping (due to pad)
        merged: List[VADSegment] = []
        for s in segments:
            if merged and s.start <= merged[-1].end + 0.05:
                merged[-1].end = max(merged[-1].end, s.end)
                merged[-1].speech_probability = max(merged[-1].speech_probability, s.speech_probability)
            else:
                merged.append(s)
        return merged

    def detect(self, audio: np.ndarray) -> List[VADSegment]:
        if audio.size == 0:
            return []
        if self.model == "silero_vad":
            # Silero is the configured VAD; if it cannot load that is a hard
            # failure, never a silent downgrade to the energy heuristic.
            if self._try_load_silero() is None:
                raise RuntimeError(
                    "vad.model=silero_vad but the Silero model could not be loaded. "
                    "Install it with: uv add silero-vad (or uv sync --extra vad)."
                )
            segs = self._silero_detect(audio)
            if segs is None:
                raise RuntimeError("Silero VAD inference failed; refusing energy fallback (vad.model=silero_vad).")
            return segs
        return self._energy_detect(audio)
