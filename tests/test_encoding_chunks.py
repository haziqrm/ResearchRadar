import numpy as np
import pandas as pd
import pytest

from src.research_map.encoding.chunks import (
    aggregate_paper_vectors,
    chunk_memberships,
    embed_chunks,
    make_chunks,
    split_sentences,
)
from src.research_map.encoding.embed import FakeEmbedder
from src.research_map.encoding.text import build_text
from src.research_map.encoding.topics import fit_topics

OPT = "We hedge option portfolios with neural networks. Volatility surfaces are calibrated. Derivatives pricing improves."
CRY = "Bitcoin exchange liquidity is thin. Stablecoin runs hit DeFi lending. Crypto tokens trade on decentralized exchanges."


def paper(pid, title, abstract):
    return {"paper_id": pid, "title": title, "abstract": abstract,
            "text": build_text(title, abstract)}


def test_split_sentences():
    assert split_sentences("One thing. Another thing! Is it? Yes.") == [
        "One thing.", "Another thing!", "Is it?", "Yes."]
    assert split_sentences("") == [] and split_sentences(None) == []
    assert split_sentences("Value is 3.5 percent. Next.") == ["Value is 3.5 percent.", "Next."]


def test_make_chunks_windows_and_single_chunk_uses_baseline_text():
    papers = pd.DataFrame([paper("A", "Mixed", OPT + " " + CRY), paper("B", "Short", "Only one.")])
    chunks = make_chunks(papers, window=3)
    assert chunks["chunk_id"].tolist() == ["A#0", "A#1", "B#0"]
    assert chunks.loc[0, "text"].startswith("Mixed [SEP] We hedge option")
    assert chunks.loc[2, "text"] == papers.loc[1, "text"]  # baseline text reused
    with pytest.raises(ValueError):
        make_chunks(papers, window=0)


def test_single_chunk_paper_vector_equals_baseline():
    papers = pd.DataFrame([paper("B", "Short", "Only one sentence here.")])
    chunks = make_chunks(papers)
    emb = FakeEmbedder(dim=32)
    vecs = emb.embed(chunks["text"].tolist())
    pooled = aggregate_paper_vectors(chunks, vecs, ["B"])
    assert np.allclose(pooled, emb.embed(papers["text"].tolist()), atol=1e-6)


def test_aggregation_is_order_invariant_and_normalised():
    papers = pd.DataFrame([paper("A", "Mixed", OPT + " " + CRY), paper("C", "Other", OPT)])
    chunks = make_chunks(papers)
    vecs = FakeEmbedder(dim=32).embed(chunks["text"].tolist())
    order = np.random.default_rng(0).permutation(len(chunks))
    a = aggregate_paper_vectors(chunks, vecs, ["A", "C"])
    b = aggregate_paper_vectors(chunks.iloc[order].reset_index(drop=True), vecs[order], ["A", "C"])
    assert np.allclose(a, b, atol=1e-6)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)


def test_embed_chunks_is_incremental(tmp_path):
    papers = pd.DataFrame([paper("A", "Mixed", OPT + " " + CRY)])
    chunks = make_chunks(papers)
    path = tmp_path / "chunks.parquet"
    _, first = embed_chunks(chunks, FakeEmbedder(), path)
    _, second = embed_chunks(chunks, FakeEmbedder(), path)
    assert (first, second) == (2, 0)


def _topic_setup():
    rng = np.random.default_rng(0)
    opt_words = "option hedging volatility derivatives pricing smile calibrated surfaces".split()
    cry_words = "bitcoin crypto exchange liquidity defi stablecoin tokens lending".split()
    rows = []
    for i in range(30):
        words = opt_words if i < 15 else cry_words
        abstract = " ".join(rng.choice(words, 6)).capitalize() + "."
        rows.append(paper(f"P{i}", f"Paper {i}", abstract))
    papers = pd.DataFrame(rows)
    emb = FakeEmbedder(dim=128)
    model = fit_topics(emb.embed(papers["text"].tolist()), k_range=(2, 2), temperature=1.0).model
    return papers, emb, model


def test_mixed_paper_chunks_land_in_different_topics():
    _, emb, model = _topic_setup()
    mixed = pd.DataFrame([paper("M", "Markets", OPT + " " + CRY)])
    chunks = make_chunks(mixed)
    vecs = emb.embed(chunks["text"].tolist())
    per_chunk = model.predict_proba(vecs).argmax(axis=1)
    assert per_chunk[0] != per_chunk[1]

    merged = chunk_memberships(model, chunks, vecs, ["M"], how="mean")
    assert np.allclose(merged.sum(axis=1), 1.0)
    assert len(model.assign(merged)[0]) == 2  # both themes kept for the paper


def test_chunk_memberships_max_and_validation():
    _, emb, model = _topic_setup()
    mixed = pd.DataFrame([paper("M", "Markets", OPT + " " + CRY)])
    chunks = make_chunks(mixed)
    vecs = emb.embed(chunks["text"].tolist())
    merged = chunk_memberships(model, chunks, vecs, ["M"], how="max")
    assert np.allclose(merged.sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        chunk_memberships(model, chunks, vecs, ["M"], how="median")


def test_single_chunk_memberships_equal_baseline():
    papers, emb, model = _topic_setup()
    chunks = make_chunks(papers.iloc[:3])
    assert len(chunks) == 3  # one short abstract each
    vecs = emb.embed(chunks["text"].tolist())
    merged = chunk_memberships(model, chunks, vecs, papers["paper_id"][:3].tolist())
    assert np.allclose(merged, model.predict_proba(emb.embed(papers["text"][:3].tolist())))
