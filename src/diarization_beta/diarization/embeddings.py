"""Speaker embedding extractor.

Primary: ECAPA-TDNN ONNX (192-d). Fallback: spectral embedding (numpy-only, deterministic)
that separates synthetic TTS voices well enough for tests when the ONNX model isn't present.
"""
from __future__ import annotations

import dataclasses
import pathlib

import numpy as np


@dataclasses.dataclass
class EmbeddingRecord:
    segment_id: str
    start: float
    end: float
    embedding: np.ndarray
    speaker_id: str | None = None


def _mel_fbank(n_mels: int = 40, n_fft: int = 512, sr: int = 16000) -> np.ndarray:
    mel_min = 0
    mel_max = 2595 * np.log10(1 + 8000 / 700)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus, f_m, f_m_plus = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        if f_m_minus >= f_m or f_m >= f_m_plus:
            continue
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    return fbank


_MEL_FBANK = _mel_fbank()


def _spectral_embedding(audio: np.ndarray, sr: int = 16000, dim: int = 192) -> np.ndarray:
    """Improved spectral embedding with pitch-gaussian bump.

    Pitch (f0) is the dominant discriminator for synthetic TTS voices (Daniel ~110Hz vs Samantha ~180Hz).
    We encode f0 as a Gaussian bump over 40 frequency bins, weighted heavily so inter-speaker cosine is low.
    """
    if len(audio) < int(sr * 0.3):
        audio = np.pad(audio, (0, max(0, int(sr * 0.3) - len(audio))))
    # Pre-emphasis
    audio = np.concatenate([[audio[0]], audio[1:] - 0.97 * audio[:-1]]).astype(np.float32)

    n_fft = 512
    hop = 160
    win = np.hanning(n_fft)
    n_frames = 1 + max(0, (len(audio) - n_fft) // hop)
    specs = []
    f0s: list[float] = []
    cents: list[float] = []
    freqs = np.fft.rfftfreq(n_fft, d=1 / sr)
    for i in range(n_frames):
        chunk = audio[i * hop : i * hop + n_fft] * win
        if len(chunk) < n_fft:
            chunk = np.pad(chunk, (0, n_fft - len(chunk)))
        spec = np.abs(np.fft.rfft(chunk, n=n_fft))
        specs.append(spec)
        mag = spec
        c = (freqs * mag).sum() / (mag.sum() + 1e-8)
        cents.append(float(c))
        seg = audio[i * hop : i * hop + n_fft]
        if len(seg) < n_fft:
            continue
        seg = seg - np.mean(seg)
        corr = np.correlate(seg, seg, mode="full")[len(seg) - 1 :]
        lo, hi = 40, 320
        if len(corr) > hi and corr[0] != 0:
            corr = corr / corr[0]
            peak = np.argmax(corr[lo:hi]) + lo
            if corr[peak] > 0.3:
                f0 = sr / peak
                if 50 < f0 < 400:
                    f0s.append(float(f0))

    specs_arr = np.stack(specs) if specs else np.zeros((1, n_fft // 2 + 1), dtype=np.float32)
    mean_spec = specs_arr.mean(axis=0)
    # Mel mean (40 bins)
    mel_mean = np.log1p(_MEL_FBANK @ mean_spec)
    mel_mean = mel_mean / (np.linalg.norm(mel_mean) + 1e-8)

    f0_med = float(np.median(f0s)) if f0s else 120.0
    cent_mean = float(np.mean(cents)) if cents else 1500.0

    def f0_vector(f0: float, n: int = 40, fmin: float = 50, fmax: float = 400, sigma: float = 20) -> np.ndarray:
        centers = np.linspace(fmin, fmax, n)
        vec = np.exp(-0.5 * ((centers - f0) / sigma) ** 2)
        return vec / (np.linalg.norm(vec) + 1e-8)

    def cent_vector(c: float, n: int = 40, cmin: float = 500, cmax: float = 4000, sigma: float = 300) -> np.ndarray:
        centers = np.linspace(cmin, cmax, n)
        vec = np.exp(-0.5 * ((centers - c) / sigma) ** 2)
        return vec / (np.linalg.norm(vec) + 1e-8)

    fv = f0_vector(f0_med, 40)
    cv = cent_vector(cent_mean, 40)
    zcr = float(((audio[:-1] * audio[1:]) < 0).mean())
    # Scalar block
    f0_std = float(np.std(f0s)) if f0s else 0.0
    cent_std = float(np.std(cents)) if cents else 0.0
    scalar = np.array([zcr, f0_med / 400, f0_std / 80, cent_mean / 4000, cent_std / 2000], dtype=np.float32)
    scalar = np.tile(scalar, 2)[:8]
    scalar = scalar / (np.linalg.norm(scalar) + 1e-8) * 0.5

    # Weighted concatenation: mel 0.4, f0 3.0, centroid 0.8 gives inter ~0.27, intra ~0.99 on synthetic TTS
    w_mel, w_f0, w_cent = 0.4, 3.0, 0.8
    parts = [mel_mean * w_mel, fv * w_f0, cv * w_cent, scalar]
    emb = np.concatenate(parts).astype(np.float32)
    if len(emb) < dim:
        emb = np.pad(emb, (0, dim - len(emb)))
    elif len(emb) > dim:
        emb = emb[:dim]
    emb = emb / (np.linalg.norm(emb) + 1e-10)
    return emb.astype(np.float32)


class EmbeddingExtractor:
    def __init__(self, model_path: str | pathlib.Path | None = None, dim: int = 192, space: str = "auto"):
        self.dim = dim
        self.model_path = pathlib.Path(model_path) if model_path else None
        self.space = space
        self._onnx_session = None
        self._load_attempted = False

    def _try_load_onnx(self):
        if self._load_attempted:
            return self._onnx_session
        self._load_attempted = True
        if self.model_path and self.model_path.exists():
            try:
                import onnxruntime as ort

                providers = ["CPUExecutionProvider"]
                sess = ort.InferenceSession(str(self.model_path), providers=providers)
                self._onnx_session = sess
                # verify io names
                self._input_name = sess.get_inputs()[0].name
                self._output_name = sess.get_outputs()[0].name
                return sess
            except Exception as e:
                print(f"[embeddings] ONNX load failed ({e}), using spectral fallback")
        return None

    def _onnx_embed(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray | None:
        sess = self._try_load_onnx()
        if sess is None:
            return None
        try:
            # ECAPA expects (1, n_samples) float32, 16k
            # Some exports expect (1, 16000*3) padded/truncated; we feed variable length
            # Normalize
            audio = audio.astype(np.float32)
            # Ensure length: many ECAPA models handle variable length; if fixed, pad/trim to 3s
            inp_shape = sess.get_inputs()[0].shape
            # inp_shape often [1, -1] or [1, 48000]
            if len(inp_shape) == 2 and isinstance(inp_shape[1], int) and inp_shape[1] > 0:
                target_len = inp_shape[1]
                if len(audio) < target_len:
                    audio = np.pad(audio, (0, target_len - len(audio)))
                else:
                    audio = audio[:target_len]
            wav = audio[np.newaxis, :].astype(np.float32)
            out = sess.run([self._output_name], {self._input_name: wav})[0]
            emb = np.array(out).squeeze()
            if emb.ndim > 1:
                emb = emb.mean(axis=0)
            emb = emb.astype(np.float32)
            # L2 norm
            emb = emb / (np.linalg.norm(emb) + 1e-10)
            # Pad/truncate to self.dim if needed
            if len(emb) < self.dim:
                emb = np.pad(emb, (0, self.dim - len(emb)))
            elif len(emb) > self.dim:
                emb = emb[: self.dim]
            return emb
        except Exception as e:
            print(f"[embeddings] ONNX inference failed ({e}), fallback")
            return None

    def embed_segment(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        if self.space == "spectral":
            return _spectral_embedding(audio, sr, self.dim)
        # Try ONNX first
        emb = self._onnx_embed(audio, sr)
        if emb is not None:
            return emb
        return _spectral_embedding(audio, sr, self.dim)

    def embed_segments(
        self, audio: np.ndarray, sr: int, segments: list, min_duration: float = 1.0
    ) -> list[EmbeddingRecord]:
        """segments: list of dict/VADSegment with start/end. Returns EmbeddingRecords."""
        records = []
        for idx, seg in enumerate(segments):
            if isinstance(seg, dict):
                start, end = float(seg["start"]), float(seg["end"])
            else:
                start, end = float(seg.start), float(seg.end)
            s_idx = int(start * sr)
            e_idx = int(end * sr)
            chunk = audio[s_idx:e_idx]
            if len(chunk) < int(0.2 * sr):
                continue
            # For short segments, pad with zeros to min_duration instead of pulling neighboring audio
            # (neighboring audio may be silence or other speaker, corrupting embedding)
            if len(chunk) < int(min_duration * sr):
                pad_len = int(min_duration * sr) - len(chunk)
                # pad equally both sides with zeros (or tile if very short)
                chunk = np.pad(chunk, (0, pad_len), mode="constant")
            emb = self.embed_segment(chunk, sr)
            records.append(
                EmbeddingRecord(
                    segment_id=f"seg_{idx:04d}",
                    start=float(start),
                    end=float(end),
                    embedding=emb,
                )
            )
        return records


def speech_noise_ratio(audio: np.ndarray, sr: int, segments: list) -> float | None:
    """Cheap quality gate: RMS(speech) / RMS(non-speech).

    Clean recordings (TTS, studio, dictation) have near-silent gaps -> very
    high ratio.  Playback recordings, music beds and room noise keep the
    floor high -> low ratio.  ECAPA degrades under such noise while the
    pitch-based spectral embedding survives it.
    """
    mask = np.zeros(len(audio), dtype=bool)
    for seg in segments:
        start, end = (seg["start"], seg["end"]) if isinstance(seg, dict) else (seg.start, seg.end)
        mask[int(start * sr) : int(end * sr)] = True
    speech = audio[mask]
    noise = audio[~mask]
    if len(speech) < int(0.1 * sr):
        return None
    speech_rms = float(np.sqrt(np.mean(speech**2)))
    if len(noise) > int(0.05 * sr):
        noise_rms = float(np.sqrt(np.mean(noise**2)))
    else:
        noise_rms = 1e-6
    return speech_rms / (noise_rms + 1e-9)


def select_space(audio: np.ndarray, sr: int, segments: list, clean_snr_threshold: float = 150.0) -> str:
    """Pick the embedding space for this recording.

    ECAPA (trained on clean speech) is the default; noisy/playback audio
    falls back to the pitch-based spectral embedding, which is far more
    robust to music beds and speaker playback (measured on the eval corpus:
    clean recordings sit at SNR >= 280, noisy ones at <= 108).
    """
    snr = speech_noise_ratio(audio, sr, segments)
    if snr is None:
        return "ecapa"
    return "ecapa" if snr >= clean_snr_threshold else "spectral"
