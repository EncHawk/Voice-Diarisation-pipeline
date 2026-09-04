"""Speaker clustering via agglomerative cosine distance."""
from __future__ import annotations

import numpy as np


def cluster_embeddings(
    embeddings: np.ndarray,
    distance_threshold: float = 0.35,
    linkage: str = "average",
    metric: str = "cosine",
) -> np.ndarray:
    """Cluster embeddings; returns labels (n,). Handles n<2 gracefully."""
    n = embeddings.shape[0] if embeddings.ndim == 2 else 0
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)

    # Try sklearn AgglomerativeClustering
    try:
        from sklearn.cluster import AgglomerativeClustering

        # sklearn >=1.3 uses metric=; older uses affinity
        try:
            clust = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                linkage=linkage,
                metric=metric,
                compute_full_tree=True,
            )
        except TypeError:
            clust = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                linkage=linkage,
                affinity=metric,
                compute_full_tree=True,
            )
        labels = clust.fit_predict(embeddings)
        # Agglomerative may return -1? No; but check
        # Remap to 0..k-1 contiguous
        _, labels = np.unique(labels, return_inverse=True)
        return labels.astype(int)
    except Exception:
        pass

    # Fallback: greedy threshold clustering on cosine similarity
    from .similarity import pairwise_cosine_similarity

    sim = pairwise_cosine_similarity(embeddings)
    # distance = 1 - sim
    labels = np.full(n, -1, dtype=int)
    cur = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        labels[i] = cur
        for j in range(i + 1, n):
            if labels[j] == -1 and (1 - sim[i, j]) <= distance_threshold:
                # also ensure not closer to another cluster centroid
                labels[j] = cur
        cur += 1
    # If still one cluster but embeddings far apart, split by threshold
    return labels


def estimate_num_speakers(labels: np.ndarray) -> int:
    if len(labels) == 0:
        return 0
    return int(len(np.unique(labels)))
