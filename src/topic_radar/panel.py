"""Shared inputs for the topic radar: papers, vectors, edges and per-origin group views.

``Corpus`` aligns B's files by paper. ``group_view`` places papers into groups with a
fitted model and returns, per paper, the group, the top membership probability and
the distance to the group centre. ``refit_model(corpus, t)`` fits groups on papers with
year <= t only, so nothing about later years shapes the groups used at origin t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.contracts import DATA_DIR
from src.research_map.encoding.groups import GroupModel, load_groups
from src.research_map.encoding.store import load_vectors
from src.research_map.encoding.topics import fit_topics

N_GROUPS = 25  # same as B's frozen groups
PCA_COMPONENTS = 50
SEED = 0
LAST_TARGET_YEAR = 2025  # 2026 is a partial year


@dataclass
class Corpus:
    paper_id: np.ndarray  # (n,)
    title: np.ndarray  # (n,)
    year: np.ndarray  # (n,) int
    vectors: np.ndarray  # (n, dim) raw vectors (what the group models expect)
    unit: np.ndarray  # (n, dim) L2-normalised, for cosine similarity
    frozen_group: np.ndarray  # (n,) B's frozen group id
    edges: np.ndarray  # (m, 2) paper indices, shared-reference links
    group_names: dict[int, str]

    def __len__(self) -> int:
        return len(self.year)


def normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


def make_corpus(nodes: pd.DataFrame, ids: list[str], vectors: np.ndarray, frozen: pd.Series,
                edges: pd.DataFrame, names: dict[int, str]) -> Corpus:
    """Align nodes, vectors and frozen groups by paper_id; drop papers without a vector."""
    pos = {pid: i for i, pid in enumerate(ids)}
    nodes = nodes[nodes["paper_id"].isin(pos) & nodes["paper_id"].isin(frozen.index)]
    nodes = nodes.drop_duplicates("paper_id").reset_index(drop=True)
    vec = np.asarray(vectors, dtype=np.float64)[[pos[p] for p in nodes["paper_id"]]]
    index = {pid: i for i, pid in enumerate(nodes["paper_id"])}
    e = edges[edges["source_id"].isin(index) & edges["target_id"].isin(index)]
    pairs = np.column_stack([e["source_id"].map(index), e["target_id"].map(index)]).astype(int)
    return Corpus(
        paper_id=nodes["paper_id"].to_numpy(), title=nodes["title"].fillna("").to_numpy(),
        year=nodes["year"].astype(int).to_numpy(), vectors=vec,
        unit=normalise(vec).astype(np.float32),
        frozen_group=frozen.loc[nodes["paper_id"]].astype(int).to_numpy(),
        edges=pairs.reshape(-1, 2), group_names=dict(names),
    )


def load_corpus() -> Corpus:
    """B's committed files: network nodes and edges, paper encodings, vectors, group names."""
    nodes = pd.read_csv(DATA_DIR / "network_nodes.csv")
    enc = pd.read_csv(DATA_DIR / "paper_encodings.csv").set_index("paper_id")["top_topic"]
    edges = pd.read_csv(DATA_DIR / "network_edges.csv")
    ids, vectors, _ = load_vectors()
    _, names = load_groups()
    return make_corpus(nodes, ids, vectors, enc, edges, names)


def frozen_model() -> GroupModel:
    return load_groups()[0]


def refit_model(corpus: Corpus, t: int, k: int = N_GROUPS, seed: int = SEED):
    """PCA + Gaussian mixture fitted on papers with year <= t only (B's fit_topics)."""
    past = corpus.year <= t
    return fit_topics(corpus.vectors[past], k_range=(k, k), n_components=PCA_COMPONENTS,
                      seed=seed).model


def _reduce(model, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(model, GroupModel):
        return (vectors - model.pca_mean) @ model.pca_components.T, model.means
    reduced = model.pca.transform(vectors) if model.pca is not None else vectors
    return reduced, model.gmm.means_


VIEW_CHUNK = 1000  # papers per batch, keeps memory small


def group_view(model, vectors: np.ndarray) -> pd.DataFrame:
    """Per paper: group (argmax), top membership probability, distance to its group centre."""
    parts = []
    for i in range(0, len(vectors), VIEW_CHUNK):
        v = np.asarray(vectors[i:i + VIEW_CHUNK], dtype=np.float64)
        probs = model.predict_proba(v)
        group = probs.argmax(axis=1)
        reduced, means = _reduce(model, v)
        parts.append(pd.DataFrame({"group": group, "top_prob": probs.max(axis=1),
                                   "dist": np.linalg.norm(reduced - means[group], axis=1)}))
    return pd.concat(parts, ignore_index=True)
