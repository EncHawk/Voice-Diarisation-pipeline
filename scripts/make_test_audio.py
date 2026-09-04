"""Generate synthetic multi-voice test audio using macOS `say`.

Creates 8 test cases per SPEC §26 with ground-truth JSON for verification.
Each case concatenates voiced segments with silences; speakers use distinct macOS voices.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile

import numpy as np

OUT_DIR = pathlib.Path("tests/fixtures/synthetic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Voice mapping: Speaker 1/2/3 -> distinct macOS voices that actually sound different
VOICES = {
    "speaker_1": "Daniel",   # en_GB male
    "speaker_2": "Samantha", # en_US female
    "speaker_3": "Karen",    # en_AU female (alt)
    "speaker_4": "Moira",    # en_IE female
}

# Use aiff intermediate then ffmpeg to 16k wav


def say_to_wav(text: str, voice: str, out_path: pathlib.Path, rate: int = 180):
    """Use `say -v Voice -r rate -o file.aiff --data-format=LEF32@16000` then convert."""
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as tmp:
        tmp_aiff = pathlib.Path(tmp.name)
    try:
        cmd = ["say", "-v", voice, "-r", str(rate), "-o", str(tmp_aiff), text]
        # say wants aiff; with --data-format we can force 16k? Try without first
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            # fallback without -r
            cmd = ["say", "-v", voice, "-o", str(tmp_aiff), text]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                print(f"say failed voice={voice}: {result.stderr}")
                raise RuntimeError(result.stderr)
        # Convert aiff -> wav 16k mono
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(tmp_aiff), "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", str(out_path)],
            check=True,
        )
    finally:
        try:
            tmp_aiff.unlink(missing_ok=True)
        except Exception:
            pass
    return out_path


def concat_wavs(wav_paths: list[pathlib.Path], out_path: pathlib.Path, silence_ms: int = 400):
    """Concatenate wavs with silence between."""
    import soundfile as sf

    parts = []
    sr = 16000
    silence = np.zeros(int(sr * silence_ms / 1000), dtype=np.float32)
    for p in wav_paths:
        data, s = sf.read(str(p))
        if data.ndim == 2:
            data = data.mean(axis=1)
        if s != 16000:
            # resample via ffmpeg already did; but handle if not
            import subprocess
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp2 = pathlib.Path(tmp.name)
            sf.write(str(tmp2), data, s)
            tmp_out = p.with_suffix(".tmp.wav")
            subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(tmp2), "-ar", "16000", "-ac", "1", str(tmp_out)], check=True)
            data, _ = sf.read(str(tmp_out))
            if data.ndim == 2:
                data = data.mean(axis=1)
            tmp2.unlink(missing_ok=True)
            tmp_out.unlink(missing_ok=True)
        parts.append(data.astype(np.float32))
        parts.append(silence)
    if parts and len(parts[-1]) == len(silence):
        parts = parts[:-1]
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    sf.write(str(out_path), audio, 16000)
    return out_path, audio


def build_case(name: str, turns: list[dict], silence_ms: int = 500):
    """
    turns: [{"speaker": "speaker_1", "text": "..."}, ...]
    Produces wav + json ground truth with per-turn timestamps (estimated via segment durations).
    """
    case_dir = OUT_DIR / name
    case_dir.mkdir(parents=True, exist_ok=True)
    wav_parts: list[pathlib.Path] = []
    gt_segments = []
    cur_time = 0.0
    for idx, turn in enumerate(turns):
        spk = turn["speaker"]
        text = turn["text"]
        voice = VOICES.get(spk, "Daniel")
        part_path = case_dir / f"part_{idx:04d}.wav"
        say_to_wav(text, voice, part_path)
        # probe duration
        import soundfile as sf

        data, sr = sf.read(str(part_path))
        dur = len(data) / sr
        wav_parts.append(part_path)
        gt_segments.append(
            {
                "segment_id": f"seg_{idx:04d}",
                "speaker_id": spk,
                "start": round(cur_time, 3),
                "end": round(cur_time + dur, 3),
                "text": text,
                "voice": voice,
            }
        )
        cur_time += dur + silence_ms / 1000.0

    out_wav = OUT_DIR / f"{name}.wav"
    _, _ = concat_wavs(wav_parts, out_wav, silence_ms=silence_ms)
    # cleanup parts? keep for debugging but also produce single wav
    # Update ground truth durations to reflect actual concatenated wav timestamps (re-probe)
    # Recompute with silence gaps: use gt_segments as is but now gaps accounted? Already did.
    # Write ground truth JSON
    gt = {
        "case": name,
        "description": turns[0].get("_desc", ""),
        "speakers": sorted(set(t["speaker"] for t in turns)),
        "segments": gt_segments,
        "wav": str(out_wav),
        "num_speakers": len(set(t["speaker"] for t in turns)),
    }
    # Write expected identity mapping for verification
    expected = {}
    # For cases 2,3 etc., expected name mappings will be injected by caller if needed
    wins = OUT_DIR / f"{name}.json"
    wins.write_text(json.dumps(gt, indent=2))
    print(f"[make] {name}: {len(turns)} turns, {cur_time:.1f}s, {len(set(t['speaker'] for t in turns))} speakers -> {out_wav}")
    # also clean parts
    for p in wav_parts:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass
    # remove case_dir if empty
    try:
        case_dir.rmdir()
    except Exception:
        pass
    return out_wav, gt


def main():
    cases = []

    # Case 1: Two speakers alternating (spec §26.1)
    cases.append(
        (
            "case1_two_speakers",
            [
                {"speaker": "speaker_1", "text": "Hello every one. I think the architecture looks good."},
                {"speaker": "speaker_2", "text": "I disagree with that approach. We should reconsider the data pipeline."},
                {"speaker": "speaker_1", "text": "What specifically concerns you about the current pipeline?"},
                {"speaker": "speaker_2", "text": "The latency is too high. We need to optimize the batch processing."},
            ],
        )
    )

    # Case 2: Explicit introduction (spec §26.2)
    cases.append(
        (
            "case2_introduction",
            [
                {"speaker": "speaker_1", "text": "Hello everyone."},
                {"speaker": "speaker_2", "text": "Hi, I'm Dan. Nice to meet you all."},
                {"speaker": "speaker_1", "text": "Welcome Dan. Let's discuss the design document."},
                {"speaker": "speaker_2", "text": "Thanks. I've reviewed the document and it looks solid."},
            ],
        )
    )

    # Case 3: Direct address (spec §26.3)
    cases.append(
        (
            "case3_direct_address",
            [
                {"speaker": "speaker_1", "text": "Dan, what do you think about this design?"},
                {"speaker": "speaker_2", "text": "I think it is good. The approach makes sense."},
                {"speaker": "speaker_1", "text": "Great. Do you have any concerns?"},
                {"speaker": "speaker_2", "text": "No major concerns. Just a small question about the API."},
            ],
        )
    )

    # Case 4: Multiple names (spec §26.4) — must not assign all mentioned names to one speaker
    cases.append(
        (
            "case4_multiple_names",
            [
                {"speaker": "speaker_1", "text": "Dan, please talk to Priya about the schedule."},
                {"speaker": "speaker_2", "text": "Sure, I will reach out to Priya tomorrow."},
                {"speaker": "speaker_3", "text": "Hi, I'm Priya. I can help with that."},
                {"speaker": "speaker_1", "text": "Thanks Priya."},
            ],
        )
    )

    # Case 5: Ambiguous evidence — no name should be assigned
    cases.append(
        (
            "case5_ambiguous",
            [
                {"speaker": "speaker_1", "text": "Let's discuss the project timeline."},
                {"speaker": "speaker_2", "text": "I think we should aim for next month."},
                {"speaker": "speaker_1", "text": "That seems reasonable. What about the budget?"},
                {"speaker": "speaker_2", "text": "We have enough budget for this phase."},
            ],
        )
    )

    # Case 6: Late identification (spec §26.7) — speaker remains unknown then identified
    cases.append(
        (
            "case6_late_identification",
            [
                {"speaker": "speaker_1", "text": "Let's go over the meeting notes."},
                {"speaker": "speaker_2", "text": "Sure, what is on the agenda today?"},
                {"speaker": "speaker_1", "text": "We need to finalize the deployment plan."},
                {"speaker": "speaker_2", "text": "I can handle the deployment if you want."},
                {"speaker": "speaker_1", "text": "That would be great, Dan. Can you send me that file?"},
                {"speaker": "speaker_2", "text": "Sure, I'll send it right after the call."},
            ],
        )
    )

    # Case 7: Long two-speaker for clustering robustness
    cases.append(
        (
            "case7_long_two_speaker",
            [
                {"speaker": "speaker_1", "text": "Welcome to today's meeting. Let's talk about the roadmap."},
                {"speaker": "speaker_2", "text": "Thanks for organizing this. I have some thoughts on the timeline."},
                {"speaker": "speaker_1", "text": "Please share your perspective on the milestones."},
                {"speaker": "speaker_2", "text": "I think the first milestone should be moved earlier."},
                {"speaker": "speaker_1", "text": "That's an interesting idea. What is the rationale?"},
                {"speaker": "speaker_2", "text": "It will give us more buffer before the release."},
                {"speaker": "speaker_1", "text": "Makes sense. Let's do that."},
                {"speaker": "speaker_2", "text": "Great, I'll update the document."},
            ],
        )
    )

    # Case 8: Similar voices? With TTS we approximate by using same gender but different voice
    # We'll use Daniel vs another male voice if available; fallback to Samantha vs Karen (both female, similar)
    # Keep as two female speakers for near-similar-voice challenge
    # Note: Samantha and Karen are somewhat similar; clustering must still separate
    cases.append(
        (
            "case8_three_speakers",
            [
                {"speaker": "speaker_1", "text": "Hello everyone, let's start the review."},
                {"speaker": "speaker_2", "text": "Hi, I'm Dan, thanks for having me."},
                {"speaker": "speaker_3", "text": "And I'm Priya, happy to be here."},
                {"speaker": "speaker_1", "text": "Dan, what did you think of the proposal?"},
                {"speaker": "speaker_2", "text": "I think the proposal is excellent."},
                {"speaker": "speaker_1", "text": "Priya, do you agree?"},
                {"speaker": "speaker_3", "text": "Yes, I agree completely."},
            ],
        )
    )

    for name, turns in cases:
        build_case(name, turns)

    print(f"[make] done. Fixtures in {OUT_DIR.resolve()}")
    print("WAV files:")
    for p in sorted(OUT_DIR.glob("*.wav")):
        print(" ", p.name, f"({p.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
