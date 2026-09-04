"""Cosine similarity helpers."""
from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


def pairwise_cosine_similarity(embeddings: np.ndarray) -> np.ndarray:
    """(n, d) -> (n, n) cosine similarity matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    normed = embeddings / norms
    return normed @ normed.T
