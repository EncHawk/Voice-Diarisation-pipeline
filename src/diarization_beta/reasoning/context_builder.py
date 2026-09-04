"""Build structured conversation state for the LLM (spec §22, §12)."""
from __future__ import annotations

import json


def build_context(
    speakers: dict,
    aligned_segments: list[dict],
    entities: list[dict],
    recent_turns: int = 12,
) -> dict:
    # speakers: dict speaker_id -> profile dict
    speaker_states = []
    for sid, prof in speakers.items():
        speaker_states.append(
            {
                "id": sid,
                "known_name": prof.get("candidate_name") if isinstance(prof, dict) else getattr(prof, "candidate_name", None),
                "status": prof.get("status", "unknown") if isinstance(prof, dict) else getattr(prof, "status", "unknown"),
                "display_name": prof.get("display_name", sid) if isinstance(prof, dict) else getattr(prof, "display_name", sid),
            }
        )

    # recent turns: last N aligned segments
    recent = aligned_segments[-recent_turns:] if len(aligned_segments) > recent_turns else aligned_segments
    recent_turns_list = [
        {"speaker": s.get("speaker_id", s.get("speaker", "")), "text": s.get("text", "")} for s in recent
    ]

    known_people = [e["name"] for e in entities] if entities else []

    return {
        "speakers": speaker_states,
        "recent_turns": recent_turns_list,
        "known_entities": known_people,
        "duration": float(aligned_segments[-1].get("end", 0)) if aligned_segments else 0.0,
    }


def build_llm_prompt(
    context: dict,
    evidence: list[dict] | None = None,
    aligned_segments: list[dict] | None = None,
    allowed_names: list[str] | None = None,
) -> str:
    """Prompt that asks LLM to return structured JSON only (spec §21)."""
    import json

    if allowed_names is None:
        allowed_names = list(context.get("known_entities", []))
    allowed_names = list(dict.fromkeys(name for name in allowed_names if isinstance(name, str) and name.strip()))

    prompt = """You are a conversation analyst. Given diarized speakers and transcript turns, infer who each speaker likely is.

Rules:
- Voice identity is already established: segments with same Speaker ID are the same physical voice.
- You infer NAMES only from conversational evidence (address, self-identification, introductions, responses).
- This is a CLOSED-WORLD selection task. The allowed candidate names are listed below.
- You MUST use an exact name from that list or null. Never invent, spell-correct, paraphrase, or copy a speaker label.
- Words such as "oh", "uh", "god", "hello", "sure", "all", and "wait" are not participant names unless they appear in the allowed list as an exact candidate.
- Do not assign a name merely because it appears in a turn. A direct address belongs to the person being addressed, usually the following responding speaker, not the addresser.
- Return at most one hypothesis per speaker. If the evidence does not establish a mapping, use null and confidence 0.0.
- Return JSON only, no prose outside JSON.
- Be conservative: if evidence is weak, return low confidence or null candidate_name.
- Evidence types you can cite: direct_address, response_to_address, self_identification, introduction, conversational_reference.

Expected JSON schema:
{
  "entities": [],
  "identity_hypotheses": []
}

The entities list is not an invitation to add names. It may contain only allowed candidate names.
"""
    prompt += "\nAllowed candidate names (select only from this exact list):\n" + json.dumps(allowed_names)
    prompt += "\nConversation state:\n" + json.dumps(context, indent=2)
    if evidence:
        prompt += "\n\nHeuristic evidence already extracted:\n" + json.dumps(evidence[:10], indent=2)
    prompt += "\n\nRespond with JSON only."
    return prompt


def build_name_listing_prompt(
    aligned_segments: list[dict],
    candidate_hints: list[str] | None = None,
) -> str:
    """Ask the LLM for the participant-name list before identity mapping."""
    turns = [
        {
            "speaker": segment.get("speaker_id", segment.get("speaker", "")),
            "text": segment.get("text", ""),
        }
        for segment in aligned_segments
        if str(segment.get("text", "")).strip()
    ]
    candidate_hints = candidate_hints or []
    return """You are the name-finding step of a speaker-identification pipeline.
Read every diarized turn below and explicitly list every human person's name that appears in the transcript.
There may be zero, one, or several names. Do not skip a name because it appears only once.

Rules:
- Return JSON only.
- Return a required `names` array. Put each name in that array exactly once.
- List human names explicitly present in the transcript, including names used to address another person.
- Do not list speaker labels, roles, places, organizations, objects, transcript artifacts, or interjections such as oh or uh.
- Do not infer or invent a name that is not present in the transcript.
- Preserve the spelling used in the transcript.
- Look especially for self-introductions, direct forms of address, and names repeated in conversation.
- A name addressed to another person is still a participant-name candidate; include it even when that person never self-identifies.
- A mechanical pre-scan may provide possible name mentions below. Verify each against the transcript; do not blindly accept or reject the hints.
- For recall, inspect every possible name mention one by one. The next pipeline step will decide which speaker owns each name, so do not omit an addressed name merely because its speaker is unknown.
- If there are no reliable participant names, return an empty names array.

Return exactly this shape:
{"names":["a name from the transcript"]}

Possible name mentions to verify:
""" + json.dumps(candidate_hints) + "\n\nDiarized transcript:\n""" + json.dumps(turns, indent=2) + "\n\nRespond with JSON only."
