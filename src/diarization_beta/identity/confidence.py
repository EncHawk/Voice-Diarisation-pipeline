"""Confidence states (spec §15)."""
from __future__ import annotations

STATUSES = ("unknown", "candidate", "identified", "confirmed")


def status_for_confidence(conf: float, has_user_confirmation: bool = False, thresholds: dict | None = None) -> str:
    thresholds = thresholds or {"candidate": 0.60, "identified": 0.80}
    if has_user_confirmation:
        return "confirmed"
    if conf >= thresholds.get("identified", 0.80):
        return "identified"
    if conf >= thresholds.get("candidate", 0.60):
        return "candidate"
    return "unknown"


def should_display_name(status: str) -> bool:
    return status in ("identified", "confirmed")
