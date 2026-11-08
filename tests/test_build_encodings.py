import numpy as np
import pandas as pd
import pytest

from src.common import contracts
from src.common.contracts import validate
from src.research_map import build_encodings
from src.research_map.build_encodings import main, run_pipeline
from src.research_map.encoding.embed import FakeEmbedder
from src.research_map.encoding.persist import load_encodings, parse_ids

THEMES = {
    "options": "option hedging volatility derivatives pricing smile",
    "crypto": "bitcoin crypto exchange liquidity defi stablecoin",
    "factor": "momentum value anomaly factor returns premium",
}


def make_papers(per_theme=12):
    rows = []
    for theme, words in THEMES.items():
        vocab = words.split()
        for i in range(per_theme):
            text = f"{theme} paper {i} " + " ".join(vocab[(i + j) % len(vocab)] for j in range(5))
            rows.append({"paper_id": f"{theme[:2]}{i}", "title": f"{theme} {i}",
                         "text": text, "text_quality": "full"})
    return pd.DataFrame(rows)


def test_pipeline_outputs_satisfy_contract_and_files_agree(tmp_path):
    papers = make_papers()
    result = run_pipeline(papers, FakeEmbedder(dim=128), data_dir=tmp_path, k_range=(2, 6))

    validate("paper_encodings", result.frame)
    assert len(result.frame) == len(papers) and result.n_new_embeddings == len(papers)
    assert result.csv_path.exists() and result.parquet_path.exists()
    assert result.vectors.shape == (len(papers), 128)
    assert result.summary["papers"].sum() == len(papers)
    assert (result.frame["topic_ids"].map(lambda s: len(parse_ids(s))) <= 3).all()
    reread = pd.read_csv(result.csv_path)
    validate("paper_encodings", reread)


def test_second_run_reuses_embeddings(tmp_path):
    papers = make_papers()
    run_pipeline(papers, FakeEmbedder(dim=64), data_dir=tmp_path)
    again = run_pipeline(papers, FakeEmbedder(dim=64), data_dir=tmp_path)
    assert again.n_new_embeddings == 0


def test_new_query_keeps_embedding_cache_of_other_papers(tmp_path):
    papers = make_papers()
    run_pipeline(papers.iloc[:20], FakeEmbedder(dim=64), data_dir=tmp_path)
    second = run_pipeline(papers.iloc[10:], FakeEmbedder(dim=64), data_dir=tmp_path)
    assert second.n_new_embeddings == len(papers) - 20  # only never-seen papers


def test_themes_end_up_in_different_topics(tmp_path):
    papers = make_papers()
    result = run_pipeline(papers, FakeEmbedder(dim=256), data_dir=tmp_path, k_range=(2, 6))
    tops = result.frame["top_topic"].to_numpy()
    majorities = []
    for start in (0, 12, 24):
        values, counts = np.unique(tops[start:start + 12], return_counts=True)
        assert counts.max() >= 10  # at least 10 of 12 papers per theme share a topic
        majorities.append(values[counts.argmax()])
    assert len(set(majorities)) == 3  # each theme has its own dominant topic


def test_empty_input_rejected(tmp_path):
    with pytest.raises(ValueError):
        run_pipeline(make_papers().iloc[:0], FakeEmbedder(), data_dir=tmp_path)


def test_cli_fake_run_writes_to_scratch_dir_not_data(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(contracts, "DATA_DIR", tmp_path)
    monkeypatch.setattr(build_encodings, "fetch_papers", lambda *a, **k: make_papers())
    assert main(["--query", "anything", "--max", "40", "--fake", "--k-max", "5"]) == 0

    scratch = tmp_path / "cache" / "fake_run"
    assert (scratch / "paper_encodings.csv").exists()
    assert not (tmp_path / "paper_encodings.csv").exists()
    assert "group" in capsys.readouterr().out


def test_cli_combines_repeated_queries_and_dedupes(tmp_path, monkeypatch):
    monkeypatch.setattr(contracts, "DATA_DIR", tmp_path)
    everything = make_papers()
    calls = []

    def fake_fetch(query, *a, **k):
        calls.append(query)
        return everything.iloc[:24] if query == "a" else everything.iloc[12:]

    monkeypatch.setattr(build_encodings, "fetch_papers", fake_fetch)
    assert main(["--query", "a", "--query", "b", "--fake", "--k-max", "5"]) == 0
    assert calls == ["a", "b"]
    written = pd.read_csv(tmp_path / "cache" / "fake_run" / "paper_encodings.csv")
    assert len(written) == 36 and written["paper_id"].is_unique


def test_cli_per_year_mode_uses_the_yearly_sample(tmp_path, monkeypatch):
    from src.research_map.encoding.fetch import YearlySample

    monkeypatch.setattr(contracts, "DATA_DIR", tmp_path)
    seen = {}

    def fake_sample(filters, years, per_year, **kwargs):
        seen.update(filters=filters, years=list(years), per_year=per_year, seed=kwargs["seed"])
        return YearlySample(make_papers(), {}, {})

    monkeypatch.setattr(build_encodings, "fetch_yearly_sample", fake_sample)
    assert main(["--per-year", "50", "--years", "2010-2012", "--seed", "3", "--fake"]) == 0
    assert seen["years"] == [2010, 2011, 2012] and seen["per_year"] == 50 and seen["seed"] == 3
    assert seen["filters"].startswith("primary_topic.id:")


def test_cli_requires_exactly_one_source(capsys):
    with pytest.raises(SystemExit):
        main(["--fake"])
    with pytest.raises(SystemExit):
        main(["--query", "a", "--per-year", "5", "--fake"])


@pytest.mark.parametrize("value,expected", [("2000-2026", (2000, 2026)), ("2015", (2015, 2015))])
def test_parse_years(value, expected):
    assert build_encodings.parse_years(value) == expected


@pytest.mark.parametrize("value", ["2026-2000", "abc", "1-2-3"])
def test_parse_years_rejects_bad_ranges(value):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        build_encodings.parse_years(value)


def test_cli_with_no_results_exits_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(contracts, "DATA_DIR", tmp_path)
    monkeypatch.setattr(build_encodings, "fetch_papers",
                        lambda *a, **k: make_papers().iloc[:0])
    assert main(["--query", "x", "--fake"]) == 1


def test_real_outputs_load_back(tmp_path, monkeypatch):
    monkeypatch.setattr(contracts, "DATA_DIR", tmp_path)
    run_pipeline(make_papers(), FakeEmbedder(dim=64))  # writes into patched data dir
    loaded = load_encodings()
    assert loaded.vectors is not None and loaded.vectors.shape == (36, 64)
    assert np.isfinite(loaded.vectors).all()
