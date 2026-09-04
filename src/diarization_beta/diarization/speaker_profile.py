"""Speaker profiles with centroid and evidence."""
from __future__ import annotations

import dataclasses
from typing import List

import numpy as np


@dataclasses.dataclass
class SpeakerProfile:
    speaker_id: str
    display_name: str
    status: str = "unknown"  # unknown | candidate | identified | confirmed
    centroid_embedding: np.ndarray | None = None
    sample_count: int = 0
    segments: List[str] = dataclasses.field(default_factory=list)
    candidate_names: List[dict] = dataclasses.field(default_factory=list)
    confidence: float = 0.0
    # for quantized thresholds
    candidate_name: str | None = None

    def to_dict(self) -> dict:
        return {
            "speaker_id": self.speaker_id,
            "display_name": self.display_name,
            "status": self.status,
            "sample_count": self.sample_count,
            "segments": list(self.segments),
            "candidate_names": list(self.candidate_names),
            "confidence": round(float(self.confidence), 3),
            "candidate_name": self.candidate_name,
        }


def build_profiles(
    embedding_records, labels: np.ndarray, embeddings: np.ndarray
) -> dict[str, SpeakerProfile]:
    """Create SpeakerProfile per cluster."""
    profiles: dict[str, SpeakerProfile] = {}
    unique = np.unique(labels) if len(labels) else []
    for lab in unique:
        idxs = np.where(labels == lab)[0]
        cluster_embs = embeddings[idxs]
        centroid = cluster_embs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        sid = f"speaker_{int(lab)+1}"
        segs = [embedding_records[i].segment_id for i in idxs]
        profiles[sid] = SpeakerProfile(
            speaker_id=sid,
            display_name=f"Speaker {int(lab)+1}",
            centroid_embedding=centroid.astype(np.float32),
            sample_count=len(idxs),
            segments=segs,
        )
        # link record speaker_id
        for i in idxs:
            embedding_records[i].speaker_id = sid
    return profiles


def update_centroid(profile: SpeakerProfile, new_embedding: np.ndarray, similarity: float, config: dict | None = None) -> None:
    """EMA update guarded by similarity threshold."""
    cfg = config or {}
    alpha = float(cfg.get("centroid_update_weight", 0.15))
    min_sim = float(cfg.get("min_similarity_for_update", 0.55))
    if similarity < min_sim:
        return
    if profile.centroid_embedding is None:
        profile.centroid_embedding = new_embedding / (np.linalg.norm(new_embedding) + 1e-10)
        profile.sample_count = 1
        return
    # EMA
    c = profile.centroid_embedding
    new_c = (1 - alpha) * c + alpha * new_embedding
    new_c = new_c / (np.linalg.norm(new_c) + 1e-10)
    profile.centroid_embedding = new_c.astype(np.float32)
    profile.sample_count += 1


def assign_speaker_to_segments(embedding_records, labels: np.ndarray):
    mapping = {}
    for rec, lab in zip(embedding_records, labels):
        sid = f"speaker_{int(lab)+1}"
        mapping[rec.segment_id] = sid
        rec.speaker_id = sid
    return mapping
