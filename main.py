#!/usr/bin/env python3
"""
main.py — single-file pipeline runner (change AUDIO_PATH and run).

Usage:
  python main.py                          # uses AUDIO_PATH below
  python main.py "Testing for diarization.m4a"
  python main.py /path/to/audio.wav --no-llm

All inference is local CPU:
  - VAD: Silero VAD (ONNX, HF)  + energy fallback
  - Embeddings: ECAPA-TDNN ONNX (HF speechbrain) or spectral fallback (no download)
  - Transcription: whisper.cpp ggml-base.en.bin (HF ggerganov/whisper.cpp)
  - LLM reasoning: configured local GGUF via llama-cli
No cloud API. All models under ./models/.

Install models once:
  uv run python scripts/setup_models.py   # whisper + configured GGUF
  uv sync --group export && uv run python scripts/export_ecapa_onnx.py  # optional ECAPA ONNX
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile

# ── CHANGE ME ────────────────────────────────────────────────────────────────
AUDIO_PATH = "audio_data/Lasso_test_insane_set.m4a"  # default; overridden by argv[1] if provided
OUTPUT_DIR = None  # e.g. "outputs/my_run" or None → outputs/<stem>
DB_PATH = None     # e.g. "diarization.db" or None → diarization.db
SKIP_LLM = False   # set True to skip llama.cpp (heuristic only, faster)
SKIP_TRANSCRIPTION = False
# ────────────────────────────────────────────────────────────────────────────

def ensure_wav(audio: pathlib.Path) -> pathlib.Path:
    """Transcode any ffmpeg-supported audio to 16kHz mono wav (temp file, auto-cleaned)."""
    if audio.suffix.lower() == ".wav":
        return audio
    print(f"[main] converting {audio.name} → 16kHz mono wav via ffmpeg ...")
    out = pathlib.Path(tempfile.gettempdir()) / f"diarization_{os.getpid()}_{audio.stem}.wav"
    cmd = ["ffmpeg", "-y", "-i", str(audio), "-vn", "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(out)]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0 or not out.exists():
        raise RuntimeError(f"ffmpeg conversion failed for {audio}: {result.stderr.decode()[:500]}")
    return out


def run_pipeline(audio: str | pathlib.Path, output_dir=None, db_path=None, skip_llm=False, skip_transcription=False):
    from diarization_beta.config import load_config
    from diarization_beta.pipeline.orchestrator import process_file

    audio = pathlib.Path(audio)
    if not audio.exists():
        print(f"[main] audio not found: {audio}", file=sys.stderr)
        sys.exit(1)

    wav = ensure_wav(audio)

    # Auto-download check: whisper base + configured GGUF must exist, else hint
    cfg = load_config()
    whisper_model = pathlib.Path(cfg.get("transcription", {}).get("model_path", "models/ggml-base.en.bin"))
    llm_model = pathlib.Path(cfg.get("reasoning", {}).get("llm_model_path", "models/LFM2-700M-Q5_K_M.gguf"))
    if not whisper_model.exists():
        print(f"[main] missing whisper model at {whisper_model}")
        print("       run: uv run python scripts/setup_models.py  (or curl HF ggerganov/whisper.cpp ggml-base.en.bin)")
        if not skip_transcription:
            print("       continuing with --skip-transcription fallback")
            skip_transcription = True
    if not llm_model.exists() and not skip_llm:
        # also check models/*.gguf fallback
        alt = list(pathlib.Path("models").glob("*.gguf"))
        if alt:
            llm_model = alt[0]
            cfg.setdefault("reasoning", {})["llm_model_path"] = str(llm_model)
            print(f"[main] using fallback LLM {llm_model}")
        else:
            print(f"[main] missing LLM GGUF at {llm_model}")
            print("       run: uv run python scripts/setup_models.py")
            print("       continuing without LLM (heuristic only)")
            skip_llm = True

    # also verify llama-cli exists (brew install llama.cpp)
    import shutil
    if shutil.which(cfg.get("reasoning", {}).get("llama_cli", "llama-cli")) is None and not skip_llm:
        print("[main] llama-cli not found (brew install llama.cpp), skipping LLM")
        skip_llm = True
    if shutil.which(cfg.get("transcription", {}).get("whisper_cli", "whisper-cli")) is None and not skip_transcription:
        print("[main] whisper-cli not found (brew install whisper-cpp), skipping transcription")
        skip_transcription = True

    result = process_file(
        wav,
        config=cfg,
        output_dir=output_dir,
        db_path=db_path,
        skip_llm=skip_llm,
        skip_transcription=skip_transcription,
    )
    if wav.resolve() != audio.resolve():
        wav.unlink(missing_ok=True)
    return result


def main():
    import argparse
    from diarization_beta.config import load_config

    p = argparse.ArgumentParser(description="Local CPU diarization pipeline (whisper.cpp + llama.cpp)")
    p.add_argument("audio", nargs="?", default=AUDIO_PATH, help="input audio (wav/mp3/m4a)")
    p.add_argument("-o", "--output-dir", default=OUTPUT_DIR, help="output dir (default outputs/<stem>)")
    p.add_argument("--db", default=DB_PATH, help="sqlite db path (default diarization.db)")
    p.add_argument("--no-llm", action="store_true", help="skip llama.cpp LLM (heuristic only)")
    p.add_argument("--no-transcription", action="store_true", help="skip whisper")
    args = p.parse_args()

    skip_llm = SKIP_LLM or args.no_llm
    skip_trans = SKIP_TRANSCRIPTION or args.no_transcription

    print(f"[main] audio: {args.audio}")
    print(f"[main] whisper: {'skip' if skip_trans else 'whisper.cpp ggml-base.en.bin (HF)'}")
    llm_name = pathlib.Path(load_config().get("reasoning", {}).get("llm_model_path", "models/local.gguf")).name
    print(f"[main] LLM: {'skip' if skip_llm else f'llama.cpp {llm_name}'}")

    result = run_pipeline(args.audio, args.output_dir, args.db, skip_llm=skip_llm, skip_transcription=skip_trans)

    print("\n=== RESULT ===")
    print(f"Recording ID : {result['recording_id']}")
    print(f"Duration     : {result['duration']:.1f}s  RTF: {result['rtf']:.2f}")
    print(f"JSON         : {result['json_path']}")
    print(f"TXT          : {result['txt_path']}")
    print(f"DB           : {result['db_path']}")
    print(f"Speakers     : {len(result['participants'])}")
    for sp in result["participants"]:
        print(f"  - {sp['speaker_id']}: {sp['display_name']}  status={sp['status']}  conf={sp['confidence']:.2f}  name={sp['name']}")

    print("\n--- Transcript ---")
    for seg in result["segments"]:
        if not seg["text"]:
            continue
        t = seg["start"]
        stamp = f"[{int(t//60):02d}:{int(t%60):02d}]"
        print(f"{stamp} {seg['speaker_name']}: {seg['text']}")

    # also write a concise result log for RESULT.md style
    return result


if __name__ == "__main__":
    main()
