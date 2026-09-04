"""Identity resolver — weighted evidence aggregation (spec §13-14)."""
from __future__ import annotations

import collections

DEFAULT_WEIGHTS = {
    "VOICE_SIMILARITY": 0.25,
    "DIRECT_ADDRESS": 0.30,
    "RESPONSE_TO_ADDRESS": 0.40,
    "SELF_IDENTIFICATION": 0.45,
    "INTRODUCTION": 0.35,
    "CONVERSATIONAL_REFERENCE": 0.22,
    "LLM_SELECTION": 0.45,
    "PREVIOUS_IDENTIFICATION": 0.10,
    "USER_CONFIRMATION": 0.95,
}

# Map evidence types to weight keys
TYPE_TO_WEIGHT = {
    "voice_similarity": "VOICE_SIMILARITY",
    "direct_address": "DIRECT_ADDRESS",
    "response_to_address": "RESPONSE_TO_ADDRESS",
    "self_identification": "SELF_IDENTIFICATION",
    "introduction": "INTRODUCTION",
    "conversational_reference": "CONVERSATIONAL_REFERENCE",
    "llm_selection": "LLM_SELECTION",
    "repeated_contextual_reference": "CONVERSATIONAL_REFERENCE",
    "previous_identification": "PREVIOUS_IDENTIFICATION",
    "user_confirmation": "USER_CONFIRMATION",
}


def _weight_for(ev_type: str, weights: dict) -> float:
    key = TYPE_TO_WEIGHT.get(ev_type, ev_type.upper())
    return float(weights.get(key, weights.get(ev_type, 0.15)))


def resolve_identities(
    evidence_by_speaker: dict[str, list[dict]],
    speaker_ids: list[str],
    weights: dict | None = None,
    thresholds: dict | None = None,
    previous_identities: dict[str, str] | None = None,
    user_confirmations: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Return per-speaker {candidate_name, confidence, evidence, status}."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    thresholds = thresholds or {"candidate": 0.60, "identified": 0.60}
    previous_identities = previous_identities or {}
    user_confirmations = user_confirmations or {}

    result: dict[str, dict] = {}

    # Also need global name conflict avoidance: don't assign same name to two speakers if evidence is ambiguous
    # We'll first compute per-speaker best candidate, then resolve conflicts greedily by confidence.

    candidates: dict[str, dict] = {}  # sid -> {name, score, count}

    for sid in speaker_ids:
        evs = evidence_by_speaker.get(sid, [])
        if sid in user_confirmations:
            # confirmed overrides everything
            result[sid] = {
                "candidate_name": user_confirmations[sid],
                "confidence": 1.0,
                "status": "confirmed",
                "evidence": [{"type": "user_confirmation", "weight": weights["USER_CONFIRMATION"]}],
                "all_evidence": evs,
            }
            continue

        # Count weighted votes per name
        # Evidence items may have: target_name, name, candidate_name
        name_scores: dict[str, float] = collections.defaultdict(float)
        name_evidence: dict[str, list[dict]] = collections.defaultdict(list)
        name_counts: dict[str, int] = collections.defaultdict(int)

        for ev in evs:
            ev_type = ev.get("type", "conversational_reference")
            # Determine which name this evidence supports and for which speaker
            # For heuristic direct_address: the *target* is not the speaker; response_to_address maps correctly.
            # For self_identification: ev["name"] belongs to ev["speaker"]
            # For response_to_address / conversational_reference: ev["candidate_name"]
            name = None
            if ev_type == "self_identification":
                name = ev.get("name") or ev.get("candidate_name")
            elif ev_type == "direct_address":
                # direct_address itself doesn't assign name to its speaker; skip for scoring
                # (its derived response_to_address does)
                continue
            elif ev_type in ("response_to_address", "conversational_reference", "introduction", "repeated_contextual_reference"):
                name = ev.get("candidate_name") or ev.get("name") or ev.get("target_name")
            else:
                name = ev.get("candidate_name") or ev.get("name") or ev.get("target_name")

            if not name:
                continue
            w = _weight_for(ev_type, weights)
            # confidence field if present scales weight
            conf_scale = float(ev.get("confidence", 0.8)) if "confidence" in ev else 0.8
            # weighted contribution: w * conf_scale (or w * conf for LLM)
            # For LLM conversational_reference, conf is already the hypothesis confidence
            score = w * conf_scale
            # Bonus for multiple independent clues pointing to same name (diminishing)
            # We'll aggregate and later normalize
            name_scores[name] += score
            name_evidence[name].append({**ev, "weight": w})
            name_counts[name] += 1

        # Previous identification as faint prior (if speaker previously identified as X and no new evidence, retain)
        if sid in previous_identities and previous_identities[sid] not in name_scores:
            # add small prior with weight PREVIOUS_IDENTIFICATION
            name = previous_identities[sid]
            w = weights.get("PREVIOUS_IDENTIFICATION", 0.10)
            name_scores[name] += w * 0.7
            name_evidence[name].append({"type": "previous_identification", "weight": w})

        if not name_scores:
            result[sid] = {"candidate_name": None, "confidence": 0.0, "status": "unknown", "evidence": [], "all_evidence": evs}
            continue

        # Pick best name
        best_name, best_raw = max(name_scores.items(), key=lambda kv: kv[1])
        # Normalize confidence: logistic-like mapping
        # Sum of weights up to ~1.0-1.5 = high confidence
        # Use: conf = 1 - exp(-k * raw_sum) with bonus for multiplicity
        # k=4.0 ensures single vocative (response 0.40*0.60=0.24) reaches 0.60 threshold (60% as requested)
        k = 4.0
        multiplicity_bonus = min(0.15, (name_counts[best_name] - 1) * 0.06)
        raw_with_bonus = best_raw + multiplicity_bonus
        confidence = 1 - pow(2.718281828, -k * raw_with_bonus)
        confidence = float(max(0.0, min(0.99, confidence)))
        # Self-identification alone should be high: boost if top evidence is self_identification
        if any(e.get("type") == "self_identification" for e in name_evidence[best_name]):
            confidence = max(confidence, 0.93)

        # Determine status from thresholds
        from .confidence import status_for_confidence

        status = status_for_confidence(confidence, thresholds=thresholds)
        result[sid] = {
            "candidate_name": best_name,
            "confidence": round(float(confidence), 3),
            "status": status,
            "evidence": name_evidence[best_name],
            "all_evidence": evs,
            "raw_score": round(float(best_raw), 3),
        }

    # Resolve name collisions: if two speakers map to same name, keep higher confidence, demote other to unknown/candidate
    name_to_sids: dict[str, list[str]] = collections.defaultdict(list)
    for sid, info in result.items():
        if info.get("candidate_name"):
            name_to_sids[info["candidate_name"]].append(sid)
    for name, sids in name_to_sids.items():
        if len(sids) > 1:
            # keep highest confidence
            sids_sorted = sorted(sids, key=lambda s: result[s].get("confidence", 0), reverse=True)
            keep = sids_sorted[0]
            for drop in sids_sorted[1:]:
                # demote to next best candidate if available, else unknown
                # For simplicity, mark as unknown and clear candidate
                # (could look at second-best name_scores, but not stored)
                result[drop]["candidate_name"] = None
                result[drop]["confidence"] = 0.0
                result[drop]["status"] = "unknown"
                result[drop]["evidence"] = []
                # note collision
                result[drop]["collision_with"] = keep

    return result
