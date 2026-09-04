"""SQLite persistence (spec §25)."""
from __future__ import annotations

import json
import pathlib
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings (
    id TEXT PRIMARY KEY,
    path TEXT,
    duration REAL,
    created_at REAL,
    num_speakers INTEGER
);
CREATE TABLE IF NOT EXISTS segments (
    id TEXT PRIMARY KEY,
    recording_id TEXT,
    segment_id TEXT,
    speaker_id TEXT,
    speaker_name TEXT,
    status TEXT,
    start REAL,
    end REAL,
    text TEXT,
    FOREIGN KEY(recording_id) REFERENCES recordings(id)
);
CREATE TABLE IF NOT EXISTS speaker_profiles (
    recording_id TEXT,
    speaker_id TEXT,
    display_name TEXT,
    status TEXT,
    confidence REAL,
    candidate_name TEXT,
    sample_count INTEGER,
    centroid BLOB,
    PRIMARY KEY (recording_id, speaker_id)
);
CREATE TABLE IF NOT EXISTS embeddings (
    recording_id TEXT,
    segment_id TEXT,
    speaker_id TEXT,
    embedding BLOB,
    dim INTEGER,
    PRIMARY KEY (recording_id, segment_id)
);
CREATE TABLE IF NOT EXISTS entities (
    recording_id TEXT,
    name TEXT,
    type TEXT,
    PRIMARY KEY (recording_id, name)
);
CREATE TABLE IF NOT EXISTS identity_hypotheses (
    recording_id TEXT,
    speaker_id TEXT,
    candidate_name TEXT,
    confidence REAL,
    evidence TEXT
);
CREATE TABLE IF NOT EXISTS confirmations (
    recording_id TEXT,
    speaker_id TEXT,
    name TEXT,
    source TEXT,
    created_at REAL,
    PRIMARY KEY (recording_id, speaker_id)
);
"""


class Storage:
    def __init__(self, db_path: str | pathlib.Path = "diarization.db"):
        self.db_path = pathlib.Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def save_recording(self, recording_id: str, path: str, duration: float, num_speakers: int):
        self.conn.execute(
            "INSERT OR REPLACE INTO recordings (id, path, duration, created_at, num_speakers) VALUES (?,?,?,?,?)",
            (recording_id, path, duration, time.time(), num_speakers),
        )
        self.conn.commit()

    def save_segments(self, recording_id: str, segments: list[dict]):
        for seg in segments:
            sid = seg.get("segment_id", "")
            pk = f"{recording_id}:{sid}"
            self.conn.execute(
                "INSERT OR REPLACE INTO segments (id, recording_id, segment_id, speaker_id, speaker_name, status, start, end, text) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    pk,
                    recording_id,
                    sid,
                    seg.get("speaker_id"),
                    seg.get("speaker_name") or seg.get("display_name") or seg.get("speaker_id"),
                    seg.get("status", "unknown"),
                    float(seg.get("start", 0)),
                    float(seg.get("end", 0)),
                    seg.get("text", ""),
                ),
            )
        self.conn.commit()

    def save_profiles(self, recording_id: str, profiles: dict):
        for sid, prof in profiles.items():
            if isinstance(prof, dict):
                display = prof.get("display_name", sid)
                status = prof.get("status", "unknown")
                conf = float(prof.get("confidence", 0))
                cand = prof.get("candidate_name")
                count = int(prof.get("sample_count", 0))
                centroid = prof.get("centroid_embedding")
            else:
                display = getattr(prof, "display_name", sid)
                status = getattr(prof, "status", "unknown")
                conf = float(getattr(prof, "confidence", 0))
                cand = getattr(prof, "candidate_name", None)
                count = int(getattr(prof, "sample_count", 0))
                centroid = getattr(prof, "centroid_embedding", None)
            blob = None
            if centroid is not None:
                try:
                    import numpy as np

                    blob = np.array(centroid, dtype=np.float32).tobytes()
                except Exception:
                    blob = None
            self.conn.execute(
                "INSERT OR REPLACE INTO speaker_profiles (recording_id, speaker_id, display_name, status, confidence, candidate_name, sample_count, centroid) VALUES (?,?,?,?,?,?,?,?)",
                (recording_id, sid, display, status, conf, cand, count, blob),
            )
        self.conn.commit()

    def save_embeddings(self, recording_id: str, records):
        for rec in records:
            dim = len(rec.embedding) if hasattr(rec, "embedding") else 0
            blob = None
            try:
                import numpy as np

                blob = np.array(rec.embedding, dtype=np.float32).tobytes()
            except Exception:
                blob = None
            self.conn.execute(
                "INSERT OR REPLACE INTO embeddings (recording_id, segment_id, speaker_id, embedding, dim) VALUES (?,?,?,?,?)",
                (recording_id, getattr(rec, "segment_id", ""), getattr(rec, "speaker_id", None), blob, dim),
            )
        self.conn.commit()

    def save_entities(self, recording_id: str, entities: list[dict]):
        for e in entities:
            self.conn.execute(
                "INSERT OR REPLACE INTO entities (recording_id, name, type) VALUES (?,?,?)",
                (recording_id, e.get("name"), e.get("type", "person")),
            )
        self.conn.commit()

    def save_hypotheses(self, recording_id: str, hypotheses: dict):
        # clear previous
        self.conn.execute("DELETE FROM identity_hypotheses WHERE recording_id=?", (recording_id,))
        for sid, info in hypotheses.items():
            self.conn.execute(
                "INSERT INTO identity_hypotheses (recording_id, speaker_id, candidate_name, confidence, evidence) VALUES (?,?,?,?,?)",
                (recording_id, sid, info.get("candidate_name"), float(info.get("confidence", 0)), json.dumps(info.get("evidence", []))),
            )
        self.conn.commit()

    def add_confirmation(self, recording_id: str, speaker_id: str, name: str, source: str = "user"):
        self.conn.execute(
            "INSERT OR REPLACE INTO confirmations (recording_id, speaker_id, name, source, created_at) VALUES (?,?,?,?,?)",
            (recording_id, speaker_id, name, source, time.time()),
        )
        self.conn.commit()

    def update_profile_status(self, speaker_id: str, status: str, name: str, recording_id: str):
        self.conn.execute(
            "UPDATE speaker_profiles SET status=?, display_name=?, candidate_name=?, confidence=1.0 WHERE recording_id=? AND speaker_id=?",
            (status, name, name, recording_id, speaker_id),
        )
        self.conn.execute(
            "UPDATE segments SET speaker_name=?, status=? WHERE recording_id=? AND speaker_id=?",
            (name, status, recording_id, speaker_id),
        )
        self.conn.commit()

    def get_confirmations(self, recording_id: str) -> dict[str, str]:
        cur = self.conn.execute("SELECT speaker_id, name FROM confirmations WHERE recording_id=?", (recording_id,))
        return {row[0]: row[1] for row in cur.fetchall()}

    def get_segments(self, recording_id: str) -> list[dict]:
        cur = self.conn.execute("SELECT segment_id, speaker_id, speaker_name, status, start, end, text FROM segments WHERE recording_id=? ORDER BY start", (recording_id,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_profiles(self, recording_id: str) -> list[dict]:
        cur = self.conn.execute("SELECT speaker_id, display_name, status, confidence, candidate_name, sample_count FROM speaker_profiles WHERE recording_id=?", (recording_id,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def list_recordings(self) -> list[dict]:
        cur = self.conn.execute("SELECT id, path, duration, created_at, num_speakers FROM recordings ORDER BY created_at DESC")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
