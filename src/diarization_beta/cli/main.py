"""CLI — typer app."""
from __future__ import annotations

import json
import pathlib

import typer

app = typer.Typer(help="Local CPU Speaker Diarization & Identification Engine")
from ..config import load_config


@app.command()
def process(
    audio: pathlib.Path = typer.Argument(..., help="Input audio file (wav/mp3/m4a)"),
    output_dir: pathlib.Path | None = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    db: pathlib.Path | None = typer.Option(None, "--db", help="SQLite DB path"),
    config: pathlib.Path | None = typer.Option(None, "--config", "-c", help="Config YAML path"),
    skip_transcription: bool = typer.Option(False, "--skip-transcription", help="Skip Whisper (for testing diarization only)"),
    skip_llm: bool = typer.Option(False, "--skip-llm", help="Skip LLM reasoning"),
):
    cfg = load_config(config) if config else load_config()
    if db:
        # inject into config for pipeline
        if "storage" not in cfg:
            cfg["storage"] = {}
        cfg["storage"]["db_path"] = str(db)
    from ..pipeline.orchestrator import process_file

    result = process_file(audio, config=cfg, output_dir=output_dir, db_path=db, skip_transcription=skip_transcription, skip_llm=skip_llm)
    typer.echo(f"Done. Recording ID: {result['recording_id']}")
    typer.echo(f"  JSON: {result['json_path']}")
    typer.echo(f"  TXT:  {result['txt_path']}")
    typer.echo(f"  DB:   {result['db_path']}")
    typer.echo(f"  RTF:  {result['rtf']:.2f}  Duration: {result['duration']:.1f}s  Time: {result['processing_time']:.1f}s")
    for p in result["participants"]:
        typer.echo(f"  - {p['speaker_id']}: {p['display_name']} ({p['status']} @ {p['confidence']:.2f})")


@app.command()
def correct(
    speaker_id: str = typer.Argument(..., help="e.g. speaker_2"),
    name: str = typer.Argument(..., help="Correct name, e.g. Dan"),
    recording_id: str = typer.Option(..., "--recording", "-r", help="Recording ID"),
    db: pathlib.Path | None = typer.Option(None, "--db", help="SQLite DB path"),
):
    cfg = load_config()
    db_path = db or pathlib.Path(cfg.get("storage", {}).get("db_path", "diarization.db") if isinstance(cfg.get("storage"), dict) else "diarization.db")
    from ..storage.sqlite import Storage
    from ..identity.corrections import apply_correction

    storage = Storage(db_path)
    segments = storage.get_segments(recording_id)
    profiles_raw = storage.get_profiles(recording_id)
    # Convert to dict
    profiles = {p["speaker_id"]: p for p in profiles_raw}
    result = apply_correction(speaker_id, name, profiles, segments, storage=storage, recording_id=recording_id)
    # Also need to update JSON output if exists? For now just DB + re-export
    typer.echo(f"Corrected {speaker_id} -> {name} ({result['updated_segments']} segments)")
    # Re-save profiles cache? Already done via storage
    storage.close()


@app.command()
def export(
    recording_id: str = typer.Argument(..., help="Recording ID"),
    db: pathlib.Path | None = typer.Option(None, "--db", help="SQLite DB path"),
    output: pathlib.Path | None = typer.Option(None, "--output", "-o", help="Output JSON path"),
):
    cfg = load_config()
    db_path = db or pathlib.Path(cfg.get("storage", {}).get("db_path", "diarization.db") if isinstance(cfg.get("storage"), dict) else "diarization.db")
    from ..storage.sqlite import Storage

    storage = Storage(db_path)
    segs = storage.get_segments(recording_id)
    profs = storage.get_profiles(recording_id)
    # reformat per spec §18
    participants = []
    for p in profs:
        participants.append(
            {
                "speaker_id": p["speaker_id"],
                "name": p["candidate_name"] if p["status"] in ("identified", "confirmed") else None,
                "display_name": p["display_name"],
                "confidence": p["confidence"],
                "status": p["status"],
            }
        )
    segments = []
    for s in segs:
        segments.append(
            {
                "start": s["start"],
                "end": s["end"],
                "speaker_id": s["speaker_id"],
                "speaker_name": s["speaker_name"],
                "text": s["text"],
            }
        )
    out = {"participants": participants, "segments": segments}
    if output:
        output.write_text(json.dumps(out, indent=2))
        typer.echo(f"Wrote {output}")
    else:
        typer.echo(json.dumps(out, indent=2))
    storage.close()


@app.command("list")
def list_recordings(
    db: pathlib.Path | None = typer.Option(None, "--db", help="SQLite DB path"),
):
    cfg = load_config()
    db_path = db or pathlib.Path(cfg.get("storage", {}).get("db_path", "diarization.db") if isinstance(cfg.get("storage"), dict) else "diarization.db")
    from ..storage.sqlite import Storage

    storage = Storage(db_path)
    recs = storage.list_recordings()
    if not recs:
        typer.echo("No recordings")
    else:
        for r in recs:
            typer.echo(f"{r['id']}: {r['path']} ({r['duration']:.1f}s, {r['num_speakers']} speakers)")
    storage.close()


@app.command()
def info(
    recording_id: str = typer.Argument(..., help="Recording ID"),
    db: pathlib.Path | None = typer.Option(None, "--db", help="SQLite DB path"),
):
    cfg = load_config()
    db_path = db or pathlib.Path(cfg.get("storage", {}).get("db_path", "diarization.db") if isinstance(cfg.get("storage"), dict) else "diarization.db")
    from ..storage.sqlite import Storage

    storage = Storage(db_path)
    profs = storage.get_profiles(recording_id)
    segs = storage.get_segments(recording_id)
    typer.echo(f"Profiles for {recording_id}:")
    for p in profs:
        typer.echo(f"  {p['speaker_id']}: {p['display_name']} status={p['status']} conf={p['confidence']:.2f} candidate={p['candidate_name']}")
    typer.echo(f"\nSegments ({len(segs)}):")
    for s in segs[:20]:
        typer.echo(f"  [{s['start']:.1f}-{s['end']:.1f}] {s['speaker_name']}: {s['text'][:60]}")
    if len(segs) > 20:
        typer.echo(f"  ... and {len(segs)-20} more")
    storage.close()


if __name__ == "__main__":
    app()
