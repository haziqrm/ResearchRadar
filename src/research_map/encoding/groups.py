"""Frozen research groups: the shared list that replaces the fixed areas.

Groups are discovered once (PCA + Gaussian mixture, see ``topics.py``) and then
frozen, so group 3 stays group 3 when papers are added.  Two committed files
hold them:

* ``data/group_model.npz`` - the fitted parameters as plain arrays (no pickle),
  used to place any paper, including D's arXiv papers, into the groups.
* ``data/groups.csv`` - ``group_id, name, keywords``.  ``name`` is the string
  every team member uses in their ``area`` columns; edit it by hand once after
  a refit.  ``keywords`` are the automatic c-TF-IDF terms, kept for reference
  and for prompting.

Only refit (``build_encodings --refit-groups``) when the corpus has changed a
lot: it renumbers the groups and resets the names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from src.common import contracts
from src.research_map.encoding.embed import Embedder
from src.research_map.encoding.text import build_text
from src.research_map.encoding.topics import (
    TopicModel,
    assign_topics,
    component_log_likelihoods,
)

GROUP_MODEL_FILE = "group_model.npz"
GROUPS_CONTRACT = "groups"
OTHER = "Other"


def group_model_path(data_dir: Path | str | None = None) -> Path:
    return Path(data_dir or contracts.DATA_DIR) / GROUP_MODEL_FILE


def groups_path(data_dir: Path | str | None = None) -> Path:
    if data_dir is None:
        return contracts.path_for(GROUPS_CONTRACT)
    return Path(data_dir) / f"{GROUPS_CONTRACT}.csv"


@dataclass
class GroupModel:
    """PCA projection plus spherical Gaussian mixture, stored as plain arrays."""

    pca_mean: np.ndarray  # (dim,)
    pca_components: np.ndarray  # (n_components, dim)
    weights: np.ndarray  # (n_groups,)
    means: np.ndarray  # (n_groups, n_components)
    variances: np.ndarray  # (n_groups,)
    temperature: float
    threshold: float
    max_topics: int
    embedding_model: str

    @property
    def n_groups(self) -> int:
        return len(self.weights)

    @classmethod
    def from_topic_model(cls, model: TopicModel, embedding_model: str) -> "GroupModel":
        if model.pca is None:
            raise ValueError("cannot freeze a topic model fitted without PCA")
        if getattr(model.pca, "whiten", False):
            raise ValueError("whitened PCA is not supported")
        gmm = model.gmm
        return cls(
            pca_mean=np.asarray(model.pca.mean_, dtype=np.float64),
            pca_components=np.asarray(model.pca.components_, dtype=np.float64),
            weights=np.asarray(gmm.weights_, dtype=np.float64),
            means=np.asarray(gmm.means_, dtype=np.float64),
            variances=np.asarray(gmm.covariances_, dtype=np.float64),
            temperature=float(model.temperature),
            threshold=float(model.threshold),
            max_topics=int(model.max_topics),
            embedding_model=embedding_model,
        )

    def predict_proba(self, vectors: np.ndarray) -> np.ndarray:
        """``(n, n_groups)`` membership probabilities; rows sum to 1."""
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[1] != len(self.pca_mean):
            raise ValueError(
                f"expected vectors with {len(self.pca_mean)} dims, got shape {vectors.shape}"
            )
        reduced = (vectors - self.pca_mean) @ self.pca_components.T
        log_lik = component_log_likelihoods(reduced, self.weights, self.means, self.variances)
        return softmax(log_lik / self.temperature, axis=1)

    def assign(self, memberships: np.ndarray) -> list[list[tuple[int, float]]]:
        return assign_topics(memberships, self.threshold, self.max_topics)

    def save(self, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as handle:
            np.savez(
                handle,
                pca_mean=self.pca_mean,
                pca_components=self.pca_components,
                weights=self.weights,
                means=self.means,
                variances=self.variances,
                settings=np.array([self.temperature, self.threshold, self.max_topics]),
                embedding_model=np.array(self.embedding_model),
            )
        return out

    @classmethod
    def load(cls, path: Path | str) -> "GroupModel":
        with np.load(Path(path), allow_pickle=False) as data:
            temperature, threshold, max_topics = data["settings"].tolist()
            return cls(
                pca_mean=data["pca_mean"],
                pca_components=data["pca_components"],
                weights=data["weights"],
                means=data["means"],
                variances=data["variances"],
                temperature=float(temperature),
                threshold=float(threshold),
                max_topics=int(max_topics),
                embedding_model=str(data["embedding_model"]),
            )


def unique_names(labels: Mapping[int, str]) -> dict[int, str]:
    """Make names unique by appending the group id to repeats."""
    seen: set[str] = set()
    names: dict[int, str] = {}
    for group_id in sorted(labels):
        name = labels[group_id].strip() or f"group {group_id}"
        if name.casefold() in seen:
            name = f"{name} ({group_id})"
        seen.add(name.casefold())
        names[group_id] = name
    return names


def build_groups_frame(
    keywords: Mapping[int, str],
    names: Mapping[int, str] | None = None,
) -> pd.DataFrame:
    """The ``groups`` contract table; names default to the keywords."""
    names = unique_names(names or keywords)
    rows = [
        {"group_id": g, "name": names[g], "keywords": keywords.get(g, "")}
        for g in sorted(names)
    ]
    return pd.DataFrame(rows, columns=contracts.CONTRACTS[GROUPS_CONTRACT][1])


def save_groups(frame: pd.DataFrame, data_dir: Path | str | None = None) -> Path:
    if data_dir is None:
        return contracts.save(GROUPS_CONTRACT, frame)
    contracts.validate(GROUPS_CONTRACT, frame)
    out = groups_path(data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return out


def load_group_names(data_dir: Path | str | None = None) -> dict[int, str]:
    """``{group_id: name}`` from ``groups.csv``."""
    path = groups_path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run build_encodings first")
    frame = pd.read_csv(path)
    contracts.validate(GROUPS_CONTRACT, frame)
    return dict(zip(frame["group_id"].astype(int), frame["name"]))


def load_groups(
    data_dir: Path | str | None = None,
) -> tuple[GroupModel, dict[int, str]]:
    """The frozen model and its names; checks they describe the same groups."""
    path = group_model_path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run build_encodings first")
    model = GroupModel.load(path)
    names = load_group_names(data_dir)
    if sorted(names) != list(range(model.n_groups)):
        raise ValueError(
            f"groups.csv lists ids {sorted(names)} but the model has {model.n_groups} groups"
        )
    return model, names


def assign_groups(
    titles: Sequence[str],
    abstracts: Sequence[str] | None = None,
    *,
    embedder: Embedder | None = None,
    data_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Place papers into the shared groups (for D's arXiv papers and new papers).

    Returns one row per paper with ``area`` (strongest group name),
    ``area_score``, and ``group_ids`` / ``group_scores`` (``;``-separated, all
    groups above the membership threshold).
    """
    if abstracts is not None and len(abstracts) != len(titles):
        raise ValueError("titles and abstracts must have the same length")
    model, names = load_groups(data_dir)
    if embedder is None:
        from src.research_map.encoding.embed import Specter2Embedder

        embedder = Specter2Embedder()
    if embedder.name != model.embedding_model:
        raise ValueError(
            f"groups were fitted on {model.embedding_model!r} vectors, "
            f"not {embedder.name!r}"
        )
    abstracts = abstracts if abstracts is not None else [""] * len(titles)
    texts = [build_text(t or "", a or "") for t, a in zip(titles, abstracts)]
    columns = ["area", "area_score", "group_ids", "group_scores"]
    if not texts:
        return pd.DataFrame(columns=columns)

    probs = model.predict_proba(embedder.embed(texts))
    top = probs.argmax(axis=1)
    rows = []
    for row, best, groups in zip(probs, top, model.assign(probs)):
        rows.append({
            "area": names[int(best)],
            "area_score": round(float(row[best]), 3),
            "group_ids": ";".join(str(g) for g, _ in groups),
            "group_scores": ";".join(f"{s:.3f}" for _, s in groups),
        })
    return pd.DataFrame(rows, columns=columns)
