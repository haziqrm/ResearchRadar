import numpy as np
import pytest

from src.research_map.encoding.embed import FakeEmbedder, Specter2Embedder, l2_normalize

TEXTS = [
    "deep hedging with neural networks",
    "hedging derivatives using deep learning",
    "high frequency trading and market microstructure",
]


def test_fake_embedder_shape_dtype_and_norm():
    vecs = FakeEmbedder(dim=32).embed(TEXTS)
    assert vecs.shape == (3, 32) and vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_fake_embedder_is_deterministic_and_batch_invariant():
    emb = FakeEmbedder()
    assert np.array_equal(emb.embed(TEXTS), emb.embed(TEXTS))
    one_by_one = np.vstack([emb.embed([t]) for t in TEXTS])
    assert np.allclose(emb.embed(TEXTS), one_by_one)


def test_fake_embedder_related_texts_are_closer():
    vecs = FakeEmbedder(dim=256).embed(TEXTS)
    assert vecs[0] @ vecs[1] > vecs[0] @ vecs[2]


def test_empty_inputs():
    assert FakeEmbedder().embed([]).shape == (0, 64)
    assert Specter2Embedder().embed([]).shape == (0, 768)
    assert np.all(l2_normalize(np.zeros((1, 4))) == 0)


def test_invalid_arguments():
    with pytest.raises(ValueError):
        FakeEmbedder(dim=0)
    with pytest.raises(ValueError):
        Specter2Embedder(batch_size=0)
    with pytest.raises(ValueError):
        Specter2Embedder(max_length=1000)


@pytest.mark.slow
def test_specter2_real_model():
    """Downloads ~500MB on first run: ``pytest -m slow``."""
    pytest.importorskip("adapters")
    emb = Specter2Embedder(batch_size=2)
    vecs = emb.embed(TEXTS)
    assert vecs.shape == (3, 768) and vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)
    assert np.allclose(vecs, Specter2Embedder(batch_size=3).embed(TEXTS), atol=1e-4)
    assert vecs[0] @ vecs[1] > vecs[0] @ vecs[2]
