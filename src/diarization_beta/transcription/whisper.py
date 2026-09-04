"""Whisper.cpp wrapper via whisper-cli subprocess."""
from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile


def _whisper_cli_exists(cli: str = "whisper-cli") -> bool:
    import shutil

    return shutil.which(cli) is not None


def transcribe(
    wav_path: str | pathlib.Path,
    model_path: str | pathlib.Path | None = None,
    whisper_cli: str = "whisper-cli",
    language: str = "en",
    threads: int = 4,
    output_dir: str | pathlib.Path | None = None,
) -> list[dict]:
    """Transcribe wav_path via whisper-cli.

    Returns list of {start, end, text} dicts.
    Falls back to empty list if whisper not available or model missing.
    """
    wav_path = pathlib.Path(wav_path)
    if model_path:
        model_path = pathlib.Path(model_path)
        if not model_path.exists():
            print(f"[whisper] model not found at {model_path}, skipping transcription")
            return []
    else:
        # default location
        default = pathlib.Path("models/ggml-base.en.bin")
        if default.exists():
            model_path = default
        else:
            # try tinydiarize test model
            alt = pathlib.Path("/opt/homebrew/share/whisper-cpp/for-tests-ggml-tiny.bin")
            if alt.exists():
                model_path = alt
            else:
                print("[whisper] no model found, skipping")
                return []

    if not _whisper_cli_exists(whisper_cli):
        print(f"[whisper] {whisper_cli} not found, skipping")
        return []

    # whisper-cli writes JSON to <output_file>.json when -oj -of <prefix> given
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        prefix = tmpdir / "out"
        cmd = [
            whisper_cli,
            "-m",
            str(model_path),
            "-f",
            str(wav_path),
            "-l",
            language,
            "-t",
            str(threads),
            "-oj",
            "-of",
            str(prefix),
            "--no-prints",
        ]
        # Some whisper-cli builds print to stdout; suppress unless error
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            # Try without --no-prints for older builds
            cmd2 = [
                whisper_cli,
                "-m",
                str(model_path),
                "-f",
                str(wav_path),
                "-l",
                language,
                "-t",
                str(threads),
                "-oj",
                "-of",
                str(prefix),
            ]
            result = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                print(f"[whisper] failed: {result.stderr[:500]}")
                return []

        json_path = pathlib.Path(str(prefix) + ".json")
        if not json_path.exists():
            # try tmpdir/out.json
            json_path = prefix.with_suffix(".json")
        if not json_path.exists():
            print(f"[whisper] expected JSON at {json_path} not found; stdout={result.stdout[:500]}")
            return []
        try:
            data = json.loads(json_path.read_text())
        except Exception as e:
            print(f"[whisper] JSON parse failed {e}")
            return []

        # whisper.cpp JSON format: {"transcription": [{"timestamps": {"from": "00:00:00,000", "to": ...}, "text": "..."}]} or {"segments": [...] }
        # Handle multiple known schemas
        segments = []
        if isinstance(data, dict):
            if "transcription" in data:
                for item in data["transcription"]:
                    ts = item.get("timestamps", {})
                    # timestamps as "HH:MM:SS,mmm" or seconds string?
                    def parse_ts(s):
                        if isinstance(s, (int, float)):
                            return float(s)
                        s = str(s).strip()
                        # try "HH:MM:SS,mmm" or "HH:MM:SS.mmm"
                        if ":" in s:
                            parts = s.replace(",", ".").split(":")
                            try:
                                if len(parts) == 3:
                                    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
                                elif len(parts) == 2:
                                    return int(parts[0]) * 60 + float(parts[1])
                            except Exception:
                                return 0.0
                        try:
                            return float(s)
                        except Exception:
                            return 0.0

                    start = parse_ts(ts.get("from", 0))
                    end = parse_ts(ts.get("to", 0))
                    text = item.get("text", "").strip()
                    if text:
                        segments.append({"start": float(start), "end": float(end), "text": text})
            elif "segments" in data:
                for seg in data["segments"]:
                    segments.append(
                        {
                            "start": float(seg.get("start", seg.get("from", 0))),
                            "end": float(seg.get("end", seg.get("to", 0))),
                            "text": str(seg.get("text", "")).strip(),
                        }
                    )
            else:
                # unknown, try to dump keys
                print(f"[whisper] unknown JSON keys: {list(data.keys())}")
        elif isinstance(data, list):
            for seg in data:
                segments.append({"start": float(seg.get("start", 0)), "end": float(seg.get("end", 0)), "text": str(seg.get("text","")).strip()})

        # Filter empties, ensure sorted
        segments = [s for s in segments if s["text"]]
        segments.sort(key=lambda x: x["start"])
        return segments


def transcribe_with_chunks(
    wav_path: str | pathlib.Path,
    model_path: str | pathlib.Path | None = None,
    whisper_cli: str = "whisper-cli",
    language: str = "en",
    threads: int = 4,
    chunk_duration: float = 30.0,
) -> list[dict]:
    """For now just delegates to transcribe (whisper-cli handles long audio internally).
    Chunking kept as hook for future incremental processing.
    """
    return transcribe(wav_path, model_path, whisper_cli, language, threads)
