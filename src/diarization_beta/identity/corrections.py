"""User corrections (spec §16)."""
from __future__ import annotations


def apply_correction(
    speaker_id: str,
    name: str,
    profiles: dict,
    segments: list[dict],
    storage=None,
    recording_id: str | None = None,
) -> dict:
    """Mark profile as confirmed and retroactively relabel segments."""
    # Update profile
    prof = profiles.get(speaker_id)
    if prof is not None:
        if isinstance(prof, dict):
            prof["status"] = "confirmed"
            prof["display_name"] = name
            prof["candidate_name"] = name
            prof["confidence"] = 1.0
        else:
            prof.status = "confirmed"
            prof.display_name = name
            prof.candidate_name = name
            prof.confidence = 1.0

    # Retroactive relabel (spec §17)
    updated = 0
    for seg in segments:
        if seg.get("speaker_id") == speaker_id:
            seg["speaker_name"] = name
            seg["display_name"] = name
            seg["status"] = "confirmed"
            updated += 1

    # Persist if storage provided
    if storage and recording_id:
        try:
            storage.add_confirmation(recording_id, speaker_id, name)
            storage.update_profile_status(speaker_id, "confirmed", name, recording_id)
        except Exception as e:
            print(f"[corrections] storage update failed: {e}")

    return {"speaker_id": speaker_id, "name": name, "status": "confirmed", "updated_segments": updated}
