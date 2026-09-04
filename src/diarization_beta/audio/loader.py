"""Audio loading + resampling to 16 kHz mono float32."""
from __future__ import annotations

import pathlib
import subprocess
import tempfile

import numpy as np

TARGET_SR = 16000


def _load_with_soundfile(path: pathlib.Path) -> tuple[np.ndarray, int] | None:
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        # ensure float32 in [-1, 1]
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        return data, int(sr)
    except Exception:
        return None


def _load_with_ffmpeg(path: pathlib.Path, target_sr: int = TARGET_SR) -> np.ndarray:
    """Decode any ffmpeg-supported file to mono f32le @ target_sr via pipe."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed for {path}: {result.stderr.decode()[:500]}")
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    return audio


def load_audio(path: str | pathlib.Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load audio file and resample to target_sr mono float32.

    Returns (audio, sample_rate). Never modifies source file.
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {p}")

    # Try soundfile first (fast for wav/flac)
    loaded = _load_with_soundfile(p)
    if loaded is not None:
        data, sr = loaded
        if sr != target_sr:
            # resample via ffmpeg pipe for quality; avoid librosa dependency
            # write temp wav then decode
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                import soundfile as sf

                sf.write(tmp.name, data, sr)
                data = _load_with_ffmpeg(pathlib.Path(tmp.name), target_sr)
                sr = target_sr
        return data.astype(np.float32), target_sr

    # Fallback: ffmpeg for mp3/m4a/ogg etc
    data = _load_with_ffmpeg(p, target_sr)
    return data.astype(np.float32), target_sr


def write_wav(path: str | pathlib.Path, audio: np.ndarray, sr: int = TARGET_SR) -> None:
    import soundfile as sf

    sf.write(str(path), audio.astype(np.float32), sr)
