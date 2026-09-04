"""Entity store — in-memory + sqlite helper."""
from __future__ import annotations


class EntityStore:
    def __init__(self):
        self.entities: list[dict] = []
        self.evidence: list[dict] = []

    def add_entities(self, entities: list[dict]):
        seen = {e["name"] for e in self.entities}
        for e in entities:
            if e["name"] not in seen:
                self.entities.append(e)
                seen.add(e["name"])

    def add_evidence(self, evidence: list[dict]):
        self.evidence.extend(evidence)

    def to_dict(self) -> dict:
        return {"entities": list(self.entities), "evidence": list(self.evidence)}
