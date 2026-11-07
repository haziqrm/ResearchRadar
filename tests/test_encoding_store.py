import numpy as np
import pandas as pd
import pytest

from src.research_map.encoding.embed import FakeEmbedder
from src.research_map.encoding.store import embed_missing, load_vectors, save_vectors


class CountingEmbedder(FakeEmbedder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen: list[str] = []

    def embed(self, texts):
        self.seen.extend(texts)
        return super().embed(texts)


def papers(n, start=0):
    return pd.DataFrame({
        "paper_id": [f"W{i}" for i in range(start, start + n)],
        "text": [f"paper about topic {i}" for i in range(start, start + n)],
    })


def test_round_trip(tmp_path):
    vecs = np.random.default_rng(0).normal(size=(3, 5)).astype(np.float32)
    path = save_vectors(["a", "b", "c"], vecs, "m1", tmp_path / "v.parquet")
    ids, loaded, model = load_vectors(path)
    assert ids == ["a", "b", "c"] and model == "m1"
    assert loaded.dtype == np.float32 and np.array_equal(loaded, vecs)


def test_missing_file_gives_empty_store(tmp_path):
    ids, vecs, model = load_vectors(tmp_path / "none.parquet")
    assert ids == [] and vecs.shape[0] == 0 and model is None


def test_save_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError):
        save_vectors(["a", "a"], np.zeros((2, 3)), "m", tmp_path / "v.parquet")
    with pytest.raises(ValueError):
        save_vectors(["a"], np.zeros((2, 3)), "m", tmp_path / "v.parquet")


def test_second_run_embeds_nothing(tmp_path):
    path = tmp_path / "v.parquet"
    emb = CountingEmbedder()
    _, first, n1 = embed_missing(papers(5), emb, path)
    emb.seen.clear()
    _, second, n2 = embed_missing(papers(5), emb, path)
    assert (n1, n2) == (5, 0) and emb.seen == []
    assert np.array_equal(first, second)


def test_only_new_papers_are_embedded_and_order_follows_input(tmp_path):
    path = tmp_path / "v.parquet"
    emb = CountingEmbedder()
    embed_missing(papers(3), emb, path)
    emb.seen.clear()
    shuffled = papers(5).iloc[::-1].reset_index(drop=True)
    ids, vecs, n_new = embed_missing(shuffled, emb, path)
    assert n_new == 2 and sorted(emb.seen) == ["paper about topic 3", "paper about topic 4"]
    assert ids == ["W4", "W3", "W2", "W1", "W0"]
    assert np.allclose(vecs, FakeEmbedder().embed(shuffled["text"].tolist()))
    assert len(load_vectors(path)[0]) == 5


def test_model_change_triggers_full_reembed(tmp_path):
    path = tmp_path / "v.parquet"
    embed_missing(papers(3), FakeEmbedder(dim=8), path)

    other = CountingEmbedder(dim=16)
    other.name = "other-model"
    _, vecs, n_new = embed_missing(papers(3), other, path)
    assert n_new == 3 and vecs.shape == (3, 16)
    assert load_vectors(path)[2] == "other-model"


def test_duplicate_ids_rejected(tmp_path):
    dup = pd.DataFrame({"paper_id": ["W1", "W1"], "text": ["a", "b"]})
    with pytest.raises(ValueError):
        embed_missing(dup, FakeEmbedder(), tmp_path / "v.parquet")
