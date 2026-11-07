import json

import pytest

from src.research_map.encoding import fetch
from src.research_map.encoding.fetch import (
    CACHE_FILE,
    PAPER_COLUMNS,
    SELECT_FIELDS,
    fetch_papers,
    load_cache,
)


def make_work(n: int) -> dict:
    return {
        "id": f"https://openalex.org/W{n}",
        "title": f"Paper {n}",
        "abstract_inverted_index": {"abstract": [0], str(n): [1]},
        "keywords": [{"display_name": f"kw{n}"}],
        "topics": [{"display_name": "Finance"}],
        "publication_year": 2020 + n,
        "cited_by_count": n * 10,
    }


class FakeClient:
    def __init__(self, works):
        self.works = works
        self.calls = []

    def iter_works(self, query, **kwargs):
        self.calls.append((query, kwargs))
        limit = kwargs.get("max_results")
        yield from self.works[:limit]


def test_fetch_returns_frame_and_writes_cache(tmp_path):
    client = FakeClient([make_work(i) for i in range(1, 4)])
    df = fetch_papers("momentum", 3, client=client, cache_dir=tmp_path)

    assert list(df.columns) == PAPER_COLUMNS
    assert list(df["paper_id"]) == ["W1", "W2", "W3"]
    assert df.loc[0, "year"] == 21 + 2000 and df.loc[1, "citations"] == 20
    assert df.loc[0, "keywords"] == ["kw1"]
    assert df.loc[0, "text"].startswith("Paper 1 [SEP] abstract 1")
    assert len(load_cache(tmp_path)) == 3


def test_fetch_requests_the_expected_fields(tmp_path):
    client = FakeClient([make_work(1)])
    fetch_papers("q", 5, client=client, cache_dir=tmp_path, filters="type:article")
    query, kwargs = client.calls[0]
    assert query == "q"
    assert kwargs["select"] == list(SELECT_FIELDS)
    assert kwargs["filters"] == "type:article"
    assert kwargs["max_results"] == 5


def test_rerun_does_not_duplicate_cache(tmp_path):
    client = FakeClient([make_work(i) for i in range(1, 4)])
    fetch_papers("q", 3, client=client, cache_dir=tmp_path)
    df = fetch_papers("q", 3, client=client, cache_dir=tmp_path)

    lines = (tmp_path / CACHE_FILE).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert len(df) == 3


def test_resume_only_appends_new_works(tmp_path):
    fetch_papers("q", 2, client=FakeClient([make_work(1), make_work(2)]), cache_dir=tmp_path)
    df = fetch_papers("q", 4, client=FakeClient([make_work(i) for i in range(1, 5)]),
                      cache_dir=tmp_path)
    lines = (tmp_path / CACHE_FILE).read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(x)["id"].rsplit("/", 1)[-1] for x in lines] == ["W1", "W2", "W3", "W4"]
    assert list(df["paper_id"]) == ["W1", "W2", "W3", "W4"]


def test_duplicates_and_idless_works_are_skipped(tmp_path):
    works = [make_work(1), make_work(1), {"title": "no id"}]
    df = fetch_papers("q", 10, client=FakeClient(works), cache_dir=tmp_path)
    assert list(df["paper_id"]) == ["W1"]


def test_corrupt_cache_line_is_ignored(tmp_path):
    (tmp_path / CACHE_FILE).write_text(
        json.dumps(make_work(1)) + "\n" + '{"id": "https://openalex.org/W2", "ti', encoding="utf-8"
    )
    assert list(load_cache(tmp_path)) == ["W1"]


def test_append_after_truncated_line_keeps_new_work(tmp_path):
    (tmp_path / CACHE_FILE).write_text(
        json.dumps(make_work(1)) + "\n" + '{"id": "https://openalex.org/W2", "ti', encoding="utf-8"
    )
    fetch_papers("q", 3, client=FakeClient([make_work(3)]), cache_dir=tmp_path)
    assert list(load_cache(tmp_path)) == ["W1", "W3"]


def test_negative_max_results_rejected(tmp_path):
    with pytest.raises(ValueError):
        fetch_papers("q", -1, client=FakeClient([]), cache_dir=tmp_path)


def test_default_cache_dir_is_git_ignored_location():
    assert fetch.CACHE_DIR.parts[-3:] == ("data", "cache", "openalex")


def complete_work(n: int, year: int = 2020) -> dict:
    work = make_work(n)
    work.update({
        "publication_year": year,
        "primary_topic": {"display_name": "Finance"},
        "counts_by_year": [{"year": 2024, "cited_by_count": 3}],
        "referenced_works": [f"https://openalex.org/W{n + 100}", "https://openalex.org/W1"],
        "authorships": [{"author": {"id": f"https://openalex.org/A{n}"}}, {"author": {}}],
    })
    return work


def test_frame_has_short_reference_and_author_ids(tmp_path):
    df = fetch_papers("q", 1, client=FakeClient([complete_work(5)]), cache_dir=tmp_path)
    assert df.loc[0, "references"] == ["W105", "W1"]
    assert df.loc[0, "authors"] == ["A5"]


def test_incomplete_cached_work_is_refreshed(tmp_path):
    fetch_papers("q", 1, client=FakeClient([make_work(1)]), cache_dir=tmp_path)  # old fields only
    fetch_papers("q", 1, client=FakeClient([complete_work(1)]), cache_dir=tmp_path)
    assert "referenced_works" in load_cache(tmp_path)["W1"]
    fetch_papers("q", 1, client=FakeClient([complete_work(1)]), cache_dir=tmp_path)
    lines = (tmp_path / CACHE_FILE).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # complete copy is not written twice


class FakeSampler:
    def __init__(self, by_year, population):
        self.by_year, self.population, self.calls = by_year, population, []

    def count_works_by(self, filters, group_by):
        self.calls.append(("count", filters, group_by))
        return {str(y): n for y, n in self.population.items()}

    def sample_works(self, filters, n, *, seed, select):
        self.calls.append(("sample", filters, n, seed, tuple(select)))
        year = int(filters.rsplit(":", 1)[-1])
        yield from self.by_year.get(year, [])[:n]


def test_yearly_sample_takes_each_year_and_records_population(tmp_path):
    from src.research_map.encoding.fetch import fetch_yearly_sample, load_manifest

    by_year = {2020: [complete_work(i, 2020) for i in range(1, 6)],
               2021: [complete_work(i, 2021) for i in range(6, 9)] + [complete_work(1, 2021)]}
    client = FakeSampler(by_year, {2020: 500, 2021: 3, 2022: 0})
    result = fetch_yearly_sample("base", [2021, 2020, 2022], 4, seed=9,
                                 client=client, cache_dir=tmp_path)

    assert result.population == {2020: 500, 2021: 3, 2022: 0}
    assert result.sampled == {2020: 4, 2021: 3, 2022: 0}
    assert len(result.papers) == 7 and result.papers["paper_id"].is_unique
    count_call = client.calls[0]
    assert count_call == ("count", "base,publication_year:2020-2022", "publication_year")
    samples = [c for c in client.calls if c[0] == "sample"]
    assert [c[1] for c in samples] == ["base,publication_year:2020", "base,publication_year:2021"]
    assert samples[1][2] == 3  # sample size capped by the population
    assert all(c[3] == 9 and c[4] == SELECT_FIELDS for c in samples)

    manifest = load_manifest(tmp_path)
    assert manifest["years"]["2020"] == {"population": 500, "sampled": 4}
    assert manifest["paper_ids"] == result.papers["paper_id"].tolist()


def test_yearly_sample_validation(tmp_path):
    from src.research_map.encoding.fetch import fetch_yearly_sample, load_manifest

    with pytest.raises(ValueError):
        fetch_yearly_sample("f", [], 5, client=FakeSampler({}, {}), cache_dir=tmp_path)
    with pytest.raises(ValueError):
        fetch_yearly_sample("f", [2020], 0, client=FakeSampler({}, {}), cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "none")


def test_corpus_filter():
    from src.research_map.encoding.fetch import corpus_filter

    f = corpus_filter(["T1", "T2"], extra="from_publication_date:2010-01-01,")
    assert f.startswith("primary_topic.id:T1|T2,")
    assert "language:en" in f and "has_abstract:true" in f
    assert f.endswith("from_publication_date:2010-01-01")
    with pytest.raises(ValueError):
        corpus_filter([])
