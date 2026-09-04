"""Evidence extraction — combines heuristic + LLM hypotheses into weighted evidence."""
from __future__ import annotations


def merge_evidence(
    heuristic_evidence: list[dict],
    llm_output: dict,
    allowed_names: list[str] | set[str] | None = None,
) -> list[dict]:
    """Merge heuristic evidence with LLM hypotheses into uniform evidence records."""
    from ..entities.name_extractor import is_model_name_candidate

    allowed = set(allowed_names) if allowed_names is not None else None

    def _is_plausible_llm_name(name: str) -> bool:
        if not isinstance(name, str) or not name.strip():
            return False
        if allowed is not None and name not in allowed:
            return False
        return is_model_name_candidate(name)

    # Build forbidden map: speaker who directly addressed Name cannot BE that Name
    forbidden = set()  # (speaker, name)
    for ev in heuristic_evidence:
        if ev.get("type") == "direct_address":
            forbidden.add((ev.get("speaker"), ev.get("target_name")))

    merged = list(heuristic_evidence)
    for hyp in llm_output.get("identity_hypotheses", []):
        sid = hyp.get("speaker_id")
        name = hyp.get("candidate_name")
        if not name or not _is_plausible_llm_name(name):
            continue
        if (sid, name) in forbidden:
            # LLM incorrectly assigned addressed name to the addresser
            continue
        conf = float(hyp.get("confidence", 0.5))
        if conf < 0.5:
            continue
        ev_list = hyp.get("evidence", [])
        merged.append(
            {
                "type": "llm_selection",
                "speaker": sid,
                "candidate_name": name,
                "confidence": conf,
                "evidence": "; ".join(ev_list) if ev_list else f"LLM hypothesis {sid}->{name} @ {conf}",
                "source": "llm",
            }
        )
    return merged


def evidence_to_resolver_format(evidence: list[dict]) -> dict[str, list[dict]]:
    """Group evidence by speaker_id -> list."""
    grouped: dict[str, list[dict]] = {}
    for ev in evidence:
        sid = ev.get("speaker") or ev.get("speaker_id")
        if not sid:
            continue
        grouped.setdefault(sid, []).append(ev)
    # also handle evidence that names a candidate for a speaker
    for ev in evidence:
        if ev.get("type") == "direct_address":
            # direct_address evidence belongs to the *addressing* speaker, but resolver needs
            # the response_to_address derived evidence already created. Keep as-is.
            pass
    return grouped
