"""Relate papers using their encodings: neighbours, shared topics, clusters."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans

from src.research_map.encoding.embed import l2_normalize
from src.research_map.encoding.persist import Encodings, parse_ids


def _topic_set(frame: pd.DataFrame, row: int) -> set[int]:
    return set(parse_ids(frame.iloc[row]["topic_ids"]))


def _row_of(frame: pd.DataFrame, paper_id: str) -> int:
    matches = np.flatnonzero(frame["paper_id"].to_numpy() == paper_id)
    if len(matches) == 0:
        raise KeyError(f"unknown paper_id {paper_id!r}")
    return int(matches[0])


def _require_vectors(encodings: Encodings) -> np.ndarray:
    if encodings.vectors is None:
        raise ValueError(
            "these encodings have no vectors (sample data); run build_encodings first"
        )
    return encodings.vectors


def shared_topics(encodings: Encodings, paper_a: str, paper_b: str) -> list[int]:
    """Topic ids both papers belong to (sorted)."""
    frame = encodings.frame
    return sorted(_topic_set(frame, _row_of(frame, paper_a))
                  & _topic_set(frame, _row_of(frame, paper_b)))


def nearest_papers(encodings: Encodings, paper_id: str, k: int = 5) -> pd.DataFrame:
    """The ``k`` most similar papers by cosine similarity.

    Columns: ``paper_id``, ``similarity``, ``shared_topics``.  The paper itself is
    excluded.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    vectors = l2_normalize(_require_vectors(encodings))
    frame = encodings.frame
    row = _row_of(frame, paper_id)
    similarity = vectors @ vectors[row]
    similarity[row] = -np.inf
    order = np.argsort(-similarity, kind="stable")[: min(k, len(frame) - 1)]
    mine = _topic_set(frame, row)
    return pd.DataFrame({
        "paper_id": frame["paper_id"].to_numpy()[order],
        "similarity": similarity[order],
        "shared_topics": [sorted(mine & _topic_set(frame, int(i))) for i in order],
    })


def cluster_papers(
    encodings: Encodings,
    method: str = "kmeans",
    *,
    n_clusters: int = 8,
    min_cluster_size: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Cluster papers on their vectors; returns one label per frame row.

    ``method="kmeans"`` needs ``n_clusters`` (capped by the paper count).
    ``method="hdbscan"`` finds the number itself; label ``-1`` means noise.
    """
    vectors = l2_normalize(_require_vectors(encodings))
    if method == "kmeans":
        k = max(1, min(n_clusters, len(vectors)))
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(vectors)
    if method == "hdbscan":
        if len(vectors) < max(2, min_cluster_size):
            return np.full(len(vectors), -1)
        return HDBSCAN(min_cluster_size=min_cluster_size, copy=True).fit_predict(vectors)
    raise ValueError(f"unknown method {method!r}; use 'kmeans' or 'hdbscan'")
