"""Download required models: whisper base.en + LFM2 gguf."""
from __future__ import annotations

import pathlib
import subprocess
import sys
import urllib.request

MODELS_DIR = pathlib.Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

WHISPER_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
WHISPER_PATH = MODELS_DIR / "ggml-base.en.bin"

# Small local model for llama.cpp — LiquidAI LFM2 700M Q5_K_M.
LLM_URL = "https://huggingface.co/LiquidAI/LFM2-700M-GGUF/resolve/main/LFM2-700M-Q5_K_M.gguf"
LLM_PATH = MODELS_DIR / "LFM2-700M-Q5_K_M.gguf"
LLM_FALLBACK_URL = None


def download(url: str, dest: pathlib.Path, fallback: str | None = None):
    if dest.exists() and dest.stat().st_size > 1024 * 1024:
        print(f"[setup] {dest} already exists ({dest.stat().st_size/1e6:.1f} MB), skipping")
        return
    print(f"[setup] downloading {url} -> {dest}")
    try:
        # use curl if available (resume), else urllib
        if subprocess.run(["which", "curl"], stdout=subprocess.DEVNULL).returncode == 0:
            ret = subprocess.run(["curl", "-L", "--progress-bar", "-o", str(dest), url])
            if ret.returncode != 0:
                raise RuntimeError(f"curl failed {ret.returncode}")
        else:
            urllib.request.urlretrieve(url, dest)
        print(f"[setup] done {dest} ({dest.stat().st_size/1e6:.1f} MB)")
    except Exception as e:
        print(f"[setup] download failed: {e}")
        if fallback and url != fallback:
            print(f"[setup] trying fallback {fallback}")
            download(fallback, dest, None)
        elif dest.exists():
            dest.unlink(missing_ok=True)
        raise


def main():
    print("[setup] models dir:", MODELS_DIR.resolve())
    try:
        download(WHISPER_URL, WHISPER_PATH)
    except Exception as e:
        print(f"[setup] whisper download failed (non-fatal): {e}")
        print("  You can manually download from https://huggingface.co/ggerganov/whisper.cpp")
    try:
        download(LLM_URL, LLM_PATH, LLM_FALLBACK_URL)
    except Exception as e:
        print(f"[setup] LLM download failed (non-fatal): {e}")
        print("  You can manually place a GGUF at", LLM_PATH)
    print("[setup] done. Now run: uv run python scripts/export_ecapa_onnx.py (optional)")


if __name__ == "__main__":
    main()
