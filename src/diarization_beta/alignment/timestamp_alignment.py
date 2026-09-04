"""Align whisper segments to diarized speaker turns."""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class AlignedSegment:
    segment_id: str
    speaker_id: str
    start: float
    end: float
    text: str
    whisper_start: float | None = None
    whisper_end: float | None = None


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = _overlap(a_start, a_end, b_start, b_end)
    if inter == 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def align(
    whisper_segments: list[dict],
    diarized_segments: list[dict],
    iou_threshold: float = 0.1,
    default_speaker: str = "speaker_1",
) -> list[AlignedSegment]:
    """Map each whisper segment to the diarized speaker with max overlap.

    diarized_segments: [{segment_id, speaker_id, start, end}, ...] from clustering/VAD
    whisper_segments: [{start, end, text}, ...]
    """
    aligned = []
    for idx, w in enumerate(whisper_segments):
        ws, we = float(w["start"]), float(w["end"])
        best_sid = None
        best_overlap = 0.0
        best_iou = 0.0
        best_seg_id = f"seg_{idx:04d}"
        for d in diarized_segments:
            ds, de = float(d["start"]), float(d["end"])
            ov = _overlap(ws, we, ds, de)
            if ov > best_overlap:
                best_overlap = ov
                best_sid = d.get("speaker_id", default_speaker)
                best_seg_id = d.get("segment_id", best_seg_id)
                best_iou = _iou(ws, we, ds, de)
        # If no overlap above threshold, pick nearest by center distance
        if best_sid is None or best_overlap < 1e-6:
            # fallback: nearest diarized segment center
            if diarized_segments:
                wc = (ws + we) / 2
                nearest = min(diarized_segments, key=lambda d: abs((d["start"] + d["end"]) / 2 - wc))
                best_sid = nearest.get("speaker_id", default_speaker)
                best_seg_id = nearest.get("segment_id", best_seg_id)
            else:
                best_sid = default_speaker

        aligned.append(
            AlignedSegment(
                segment_id=best_seg_id if len(whisper_segments) == len(diarized_segments) else f"seg_{idx:04d}",
                speaker_id=best_sid,
                start=ws,
                end=we,
                text=w["text"],
                whisper_start=ws,
                whisper_end=we,
            )
        )

    # If whisper produced fewer/more segments than diarized turns, we keep whisper timing
    # but assign speakers per overlap. For cases where whisper segments span multiple diarized turns,
    # we split at diarized boundaries if overlap suggests two speakers.
    # Simple split: if a whisper segment overlaps two diarized segments significantly, split proportionally.
    # For MVP, keep as single assigned segment (max overlap wins) — acceptable.

    return aligned
