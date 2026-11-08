"""Chunk-level encodings for papers that mix themes (optional extension).

An abstract is split into windows of a few sentences.  Each window is embedded
as ``title [SEP] window`` and stored in its own parquet file.  Chunk vectors are
mean-pooled into a paper vector, and topic memberships are computed per chunk
with an already fitted :class:`TopicModel` and then merged per paper, so a paper
whose sentences fall in different topics keeps both.

A paper with a single chunk reuses its full baseline ``text``, so its vector and
memberships equal the baseline ones.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import contracts
from src.research_map.encoding.embed import Embedder, l2_normalize
from src.research_map.encoding.store import embed_missing
from src.research_map.encoding.text import build_text
from src.research_map.encoding.topics import TopicModel

CHUNK_VECTORS_FILE = "paper_chunk_vectors.parquet"
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def chunk_vectors_path() -> Path:
    return contracts.DATA_DIR / CHUNK_VECTORS_FILE


def split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation followed by a capital letter or digit."""
    text = " ".join(text.split()) if isinstance(text, str) else ""
    return [s for s in _SENTENCE_BOUNDARY.split(text) if s] if text else []


def make_chunks(papers: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Return ``chunk_id, paper_id, chunk_index, text`` for every paper.

    ``papers`` needs ``paper_id``, ``title``, ``abstract`` and ``text``.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    rows = []
    for paper in papers.itertuples(index=False):
        sentences = split_sentences(paper.abstract)
        windows = [" ".join(sentences[i:i + window]) for i in range(0, len(sentences), window)]
        if len(windows) <= 1:
            texts = [paper.text]  # same input as the baseline vector
        else:
            texts = [build_text(paper.title, chunk) for chunk in windows]
        for index, text in enumerate(texts):
            rows.append({
                "chunk_id": f"{paper.paper_id}#{index}",
                "paper_id": paper.paper_id,
                "chunk_index": index,
                "text": text,
            })
    return pd.DataFrame(rows, columns=["chunk_id", "paper_id", "chunk_index", "text"])


def embed_chunks(
    chunks: pd.DataFrame,
    embedder: Embedder,
    path: Path | str | None = None,
) -> tuple[np.ndarray, int]:
    """Embed chunks incrementally into the chunk parquet; returns ``(vectors, n_new)``."""
    stored = chunks[["chunk_id", "text"]].rename(columns={"chunk_id": "paper_id"})
    _, vectors, n_new = embed_missing(stored, embedder, path or chunk_vectors_path())
    return vectors, n_new


def _paper_positions(chunks: pd.DataFrame, paper_ids: list[str]) -> np.ndarray:
    position = {pid: i for i, pid in enumerate(paper_ids)}
    return np.array([position[p] for p in chunks["paper_id"]], dtype=int)


def aggregate_paper_vectors(
    chunks: pd.DataFrame,
    chunk_vectors: np.ndarray,
    paper_ids: list[str],
) -> np.ndarray:
    """Mean-pool unit-length chunk vectors per paper, then re-normalise.

    Independent of chunk order.  Rows follow ``paper_ids``.
    """
    positions = _paper_positions(chunks, paper_ids)
    unit = l2_normalize(chunk_vectors)
    sums = np.zeros((len(paper_ids), unit.shape[1]), dtype=np.float64)
    np.add.at(sums, positions, unit)
    return l2_normalize(sums)


def chunk_memberships(
    model: TopicModel,
    chunks: pd.DataFrame,
    chunk_vectors: np.ndarray,
    paper_ids: list[str],
    how: str = "mean",
) -> np.ndarray:
    """Merge per-chunk topic probabilities into per-paper memberships.

    ``how="mean"`` averages chunks (every row still sums to 1); ``how="max"``
    keeps each topic's strongest chunk then renormalises, which favours papers
    that touch a topic in even one passage.
    """
    if how not in ("mean", "max"):
        raise ValueError("how must be 'mean' or 'max'")
    positions = _paper_positions(chunks, paper_ids)
    probs = model.predict_proba(chunk_vectors)
    merged = np.zeros((len(paper_ids), probs.shape[1]))
    if how == "mean":
        np.add.at(merged, positions, probs)
        merged /= np.bincount(positions, minlength=len(paper_ids))[:, None]
    else:
        np.maximum.at(merged, positions, probs)
        merged /= merged.sum(axis=1, keepdims=True)
    return merged
