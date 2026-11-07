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
