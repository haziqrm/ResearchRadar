import numpy as np
import pytest

from src.research_map.encoding.topics import assign_topics, fit_topics


def blobs(n_per=60, dim=8, seed=1):
    rng = np.random.default_rng(seed)
    centers = np.zeros((3, dim))
    centers[0, 0], centers[1, 1], centers[2, 2] = 8, 8, 8
    data = np.vstack([c + rng.normal(scale=0.5, size=(n_per, dim)) for c in centers])
    labels = np.repeat(np.arange(3), n_per)
    return data, labels, centers


def test_recovers_number_of_blobs_and_assignments():
    data, labels, _ = blobs()
    result = fit_topics(data, k_range=(2, 8))
    assert result.model.n_topics == 3
    # each true blob maps to exactly one discovered topic
    for blob in range(3):
        assert len(set(result.top_topic[labels == blob])) == 1
    assert len(set(result.top_topic)) == 3


def test_membership_rows_sum_to_one_and_shape():
    data, _, _ = blobs()
    result = fit_topics(data)
    assert result.memberships.shape == (len(data), result.model.n_topics)
    assert np.allclose(result.memberships.sum(axis=1), 1.0)


def test_paper_between_overlapping_topics_gets_two_topics():
    rng = np.random.default_rng(2)
    centers = np.zeros((2, 8))
    centers[0, 0], centers[1, 0] = -1.5, 1.5
    data = np.vstack([c + rng.normal(size=(200, 8)) for c in centers])
    result = fit_topics(data, k_range=(2, 2), temperature=1.0)
    probs = result.model.predict_proba(np.zeros((1, 8)))
    assigned = result.model.assign(probs)[0]
    assert len(assigned) == 2
    assert {t for t, _ in assigned} == {int(result.top_topic[0]), int(result.top_topic[200])}
    # a paper at the heart of one topic belongs only to that topic
    core = result.model.assign(result.model.predict_proba(centers[:1] * 2))[0]
    assert len(core) == 1 and core[0][0] == int(result.top_topic[0])


def test_temperature_softens_memberships():
    rng = np.random.default_rng(4)
    centers = np.zeros((2, 8))
    centers[1, 0] = 3.0
    data = np.vstack([c + rng.normal(size=(150, 8)) for c in centers])
    raw = fit_topics(data, k_range=(2, 2), temperature=1.0)
    soft = fit_topics(data, k_range=(2, 2), temperature=8.0)
    assert soft.memberships.max(axis=1).mean() < raw.memberships.max(axis=1).mean()
    multi = lambda r: sum(len(a) > 1 for a in r.assigned)  # noqa: E731
    assert multi(soft) > multi(raw)
    assert np.allclose(soft.memberships.sum(axis=1), 1.0)
    # temperature never changes the strongest topic
    assert np.array_equal(raw.top_topic, soft.top_topic)


def test_default_pca_size_scales_with_corpus():
    small = fit_topics(np.random.default_rng(0).normal(size=(32, 100)), k_range=(2, 3))
    assert small.model.pca.n_components_ == 8
    large = fit_topics(np.random.default_rng(0).normal(size=(400, 100)), k_range=(2, 2))
    assert large.model.pca.n_components_ == 50


def test_clear_point_gets_a_single_topic():
    data, _, _ = blobs()
    result = fit_topics(data, k_range=(2, 8))
    assert all(len(a) == 1 for a in result.assigned[:60])


def test_assign_topics_threshold_cap_and_none():
    probs = np.array([[0.5, 0.3, 0.1, 0.1], [0.3, 0.3, 0.2, 0.2]])
    out = assign_topics(probs, threshold=0.15, max_topics=2)
    assert [t for t, _ in out[0]] == [0, 1]
    assert [t for t, _ in out[1]] == [0, 1]
    assert assign_topics(np.full((1, 10), 0.1), threshold=0.15) == [[]]


def test_deterministic_with_seed():
    data, _, _ = blobs()
    a = fit_topics(data, seed=3).memberships
    b = fit_topics(data, seed=3).memberships
    assert np.allclose(a, b)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_tiny_corpora_do_not_crash(n):
    vectors = np.random.default_rng(0).normal(size=(n, 16))
    result = fit_topics(vectors, k_range=(2, 10))
    assert result.memberships.shape[0] == n
    assert result.model.n_topics <= max(1, n // 3)
    assert np.allclose(result.memberships.sum(axis=1), 1.0)


def test_invalid_arguments():
    data, _, _ = blobs(n_per=5)
    with pytest.raises(ValueError):
        fit_topics(np.zeros((0, 4)))
    with pytest.raises(ValueError):
        fit_topics(data, threshold=0)
    with pytest.raises(ValueError):
        fit_topics(data, k_range=(5, 2))
    with pytest.raises(ValueError):
        fit_topics(data, temperature=0)
    with pytest.raises(ValueError):
        assign_topics(np.ones((1, 2)) / 2, max_topics=0)
