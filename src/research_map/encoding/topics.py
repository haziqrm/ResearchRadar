"""Unsupervised, soft topic discovery over paper vectors.

Vectors are reduced with PCA and modelled with a spherical-covariance Gaussian
mixture (one variance per topic).  The number of topics is chosen by BIC.  Each
paper gets a probability for every topic; the topics above ``threshold`` (at most
``max_topics``) are the ones it "belongs to", so a paper can have several, or
none if it is ambiguous.

Mixture posteriors are overconfident in several dimensions (a paper between two
topics still gets ~0.99 for one), so memberships are softened with a
``temperature``: the per-topic log-likelihoods are divided by it before the
softmax.  ``temperature=1`` gives the raw mixture posteriors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

DEFAULT_THRESHOLD = 0.15
DEFAULT_MAX_TOPICS = 3
DEFAULT_TEMPERATURE = 5.0
MAX_PCA_COMPONENTS = 50


@dataclass
class TopicModel:
    """A fitted PCA + Gaussian mixture, usable on new vectors."""

    pca: PCA | None
    gmm: GaussianMixture
    bic_by_k: dict[int, float]
    threshold: float = DEFAULT_THRESHOLD
    max_topics: int = DEFAULT_MAX_TOPICS
    temperature: float = DEFAULT_TEMPERATURE

    @property
    def n_topics(self) -> int:
        return int(self.gmm.n_components)

    def predict_proba(self, vectors: np.ndarray) -> np.ndarray:
        """Return ``(n, n_topics)`` membership probabilities (rows sum to 1)."""
        vectors = np.asarray(vectors, dtype=np.float64)
        reduced = self.pca.transform(vectors) if self.pca is not None else vectors
        return softmax(self._log_likelihoods(reduced) / self.temperature, axis=1)

    def _log_likelihoods(self, reduced: np.ndarray) -> np.ndarray:
        """log(weight_k * N(x | mean_k, var_k * I)) for each paper and topic."""
        gmm = self.gmm
        dim = reduced.shape[1]
        sq_dist = ((reduced[:, None, :] - gmm.means_[None, :, :]) ** 2).sum(axis=2)
        variance = gmm.covariances_[None, :]
        return (
            np.log(gmm.weights_)[None, :]
            - 0.5 * dim * np.log(2 * np.pi * variance)
            - sq_dist / (2 * variance)
        )

    def assign(self, memberships: np.ndarray) -> list[list[tuple[int, float]]]:
        return assign_topics(memberships, self.threshold, self.max_topics)


@dataclass
class TopicResult:
    model: TopicModel
    memberships: np.ndarray  # (n, n_topics)
    top_topic: np.ndarray  # (n,) argmax topic per paper
    assigned: list[list[tuple[int, float]]]  # per paper: [(topic, prob), ...] desc


def assign_topics(
    memberships: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
    max_topics: int = DEFAULT_MAX_TOPICS,
) -> list[list[tuple[int, float]]]:
    """Per paper, the topics with probability >= ``threshold``, strongest first."""
    if max_topics < 1:
        raise ValueError("max_topics must be at least 1")
    result: list[list[tuple[int, float]]] = []
    for row in memberships:
        order = np.argsort(-row, kind="stable")[:max_topics]
        result.append([(int(t), float(row[t])) for t in order if row[t] >= threshold])
    return result


def fit_topics(
    vectors: np.ndarray,
    k_range: tuple[int, int] = (2, 12),
    threshold: float = DEFAULT_THRESHOLD,
    max_topics: int = DEFAULT_MAX_TOPICS,
    n_components: int | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = 0,
) -> TopicResult:
    """Discover topics and return soft memberships for every paper.

    ``k_range`` is inclusive and is capped by the number of papers.  With fewer
    than three papers there is nothing to cluster, so a single topic is used.
    ``n_components`` defaults to a quarter of the paper count, capped at 50:
    small corpora get a coarse PCA because extra dimensions are mostly noise.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or len(vectors) == 0:
        raise ValueError("vectors must be a non-empty 2-D array")
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    k_min, k_max = k_range
    if k_min < 1 or k_max < k_min:
        raise ValueError("k_range must satisfy 1 <= min <= max")

    n_papers = len(vectors)
    if n_papers == 1:  # GaussianMixture needs two samples; one paper is trivially one topic
        vectors = np.vstack([vectors, vectors])
    n = len(vectors)

    if n_components is None:
        n_components = min(MAX_PCA_COMPONENTS, max(2, n // 4))
    n_comp = min(n_components, n - 1, vectors.shape[1])
    pca: PCA | None = None
    reduced = vectors
    if n_comp >= 1:
        pca = PCA(n_components=n_comp, random_state=seed)
        reduced = pca.fit_transform(vectors)

    # Each mixture component needs several papers to estimate its variance.
    k_cap = max(1, n // 3)
    ks = [k for k in range(k_min, k_max + 1) if k <= k_cap] or [1]

    best: GaussianMixture | None = None
    best_bic = np.inf
    bic_by_k: dict[int, float] = {}
    for k in ks:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="spherical",
            n_init=3,
            reg_covar=1e-4,
            random_state=seed,
        ).fit(reduced)
        bic = float(gmm.bic(reduced))
        bic_by_k[k] = bic
        if bic < best_bic:
            best, best_bic = gmm, bic
    assert best is not None

    model = TopicModel(pca, best, bic_by_k, threshold, max_topics, temperature)
    memberships = softmax(model._log_likelihoods(reduced) / temperature, axis=1)[:n_papers]
    return TopicResult(
        model=model,
        memberships=memberships,
        top_topic=memberships.argmax(axis=1),
        assigned=model.assign(memberships),
    )
