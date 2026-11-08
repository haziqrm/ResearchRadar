import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from src.research_map.encoding.persist import Encodings, build_encodings_frame
from src.research_map.encoding.relate import cluster_papers, nearest_papers, shared_topics


def make_encodings(n_per=20, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.eye(3, dim) * 5
    vectors = np.vstack([c + rng.normal(scale=0.3, size=(n_per, dim)) for c in centers])
    truth = np.repeat(np.arange(3), n_per)
    ids = [f"P{i}" for i in range(len(vectors))]
    assigned = [[(int(t), 0.9)] for t in truth]
    assigned[0] = [(0, 0.6), (1, 0.3)]  # P0 is also in topic 1
    frame = build_encodings_frame(ids, truth.tolist(), assigned, {}, "m", ["full"] * len(ids))
    return Encodings(frame, vectors.astype(np.float32)), truth


def test_nearest_papers_are_same_theme_and_exclude_self():
    enc, truth = make_encodings()
    result = nearest_papers(enc, "P25", k=5)
    assert "P25" not in result["paper_id"].tolist()
    assert len(result) == 5
    assert all(truth[int(p[1:])] == 1 for p in result["paper_id"])
    assert result["similarity"].is_monotonic_decreasing


def test_nearest_papers_reports_shared_topics():
    enc, _ = make_encodings()
    result = nearest_papers(enc, "P0", k=3)
    assert all(t == [0] for t in result["shared_topics"])


def test_nearest_neighbour_symmetry_for_mutual_pairs():
    enc, _ = make_encodings(n_per=2, dim=8)
    a = nearest_papers(enc, "P0", k=1).loc[0, "paper_id"]
    sim_ab = nearest_papers(enc, "P0", k=len(enc.frame) - 1).set_index("paper_id").loc[a, "similarity"]
    sim_ba = nearest_papers(enc, a, k=len(enc.frame) - 1).set_index("paper_id").loc["P0", "similarity"]
    assert sim_ab == pytest.approx(sim_ba, abs=1e-6)


def test_k_is_capped_and_validated():
    enc, _ = make_encodings(n_per=2)
    assert len(nearest_papers(enc, "P0", k=100)) == 5
    with pytest.raises(ValueError):
        nearest_papers(enc, "P0", k=0)


def test_unknown_paper_raises():
    enc, _ = make_encodings()
    with pytest.raises(KeyError):
        nearest_papers(enc, "nope")
    with pytest.raises(KeyError):
        shared_topics(enc, "P0", "nope")


def test_shared_topics():
    enc, _ = make_encodings()
    assert shared_topics(enc, "P0", "P1") == [0]
    assert shared_topics(enc, "P0", "P25") == [1]
    assert shared_topics(enc, "P1", "P25") == []


@pytest.mark.parametrize("method,kwargs", [
    ("kmeans", {"n_clusters": 3}),
    ("hdbscan", {"min_cluster_size": 5}),
])
def test_clustering_recovers_blobs(method, kwargs):
    enc, truth = make_encodings()
    labels = cluster_papers(enc, method, **kwargs)
    assert len(labels) == len(truth)
    assert adjusted_rand_score(truth, labels) > 0.95


def test_cluster_edge_cases():
    enc, _ = make_encodings(n_per=2)
    assert len(set(cluster_papers(enc, "kmeans", n_clusters=50))) <= len(enc.frame)
    assert set(cluster_papers(enc, "hdbscan", min_cluster_size=50)) == {-1}
    with pytest.raises(ValueError):
        cluster_papers(enc, "nope")


def test_sample_data_without_vectors_gives_clear_error():
    enc = Encodings(pd.read_csv("data/sample/paper_encodings.csv"), None)
    with pytest.raises(ValueError, match="no vectors"):
        nearest_papers(enc, "W001")
    with pytest.raises(ValueError, match="no vectors"):
        cluster_papers(enc)
