# diarization pipeline

Local, CPU-only speaker diarization pipeline: audio → VAD → speaker embeddings → clustering(storing on db) → Whisper transcription → LLM name resolution.

*note:all technical details are in the [documentation](./DOCS.md)*

## Setup

```bash
uv sync --python 3.12
uv run python scripts/setup_models.py   # downloads whisper + LLM gguf
brew install llama.cpp whisper-cpp      # CLI binaries
```

## Run

```bash
uv run python main.py path/to/audio.wav 
# there's a couple in audio_data/
```

Options:

```bash
uv run python main.py audio.wav --no-llm       # skip LLM, heuristic name resolution
uv run python main.py audio.wav -o outputs/run # custom output dir
uv run python main.py audio.wav --db my.db     # custom sqlite db
```

Outputs go to `outputs/`, models to `models/`, transcript data to `diarization.db`.
