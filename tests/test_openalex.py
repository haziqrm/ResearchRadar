import pytest

from src.research_map.openalex import OpenAlexClient, OpenAlexError


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code, self.headers, self.text = payload, status, {}, ""
        self.ok = status < 400

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, pages):
        self.pages, self.calls = list(pages), []

    def get(self, url, params, headers, timeout):
        self.calls.append(dict(params))
        return FakeResponse(self.pages.pop(0))


def client(pages):
    session = FakeSession(pages)
    return OpenAlexClient("key", session=session), session


def test_sample_works_pages_until_n_and_passes_seed():
    c, session = client([
        {"results": [{"id": f"W{i}"} for i in range(100)]},
        {"results": [{"id": f"W{i}"} for i in range(100, 150)]},
    ])
    works = list(c.sample_works("type:article", 150, seed=4, select=["id", "title"]))
    assert len(works) == 150
    assert [call["page"] for call in session.calls] == [1, 2]
    assert session.calls[0]["sample"] == 150 and session.calls[0]["seed"] == 4
    assert session.calls[0]["select"] == "id,title"
    assert "Authorization" not in str(session.calls)  # key is never in params


def test_sample_works_stops_on_empty_page_and_validates():
    c, _ = client([{"results": [{"id": "W1"}]}, {"results": []}])
    assert len(list(c.sample_works("f", 10))) == 1
    with pytest.raises(ValueError):
        list(c.sample_works("", 10))
    with pytest.raises(ValueError):
        list(c.sample_works("f", 10_001))


def test_count_works_by():
    c, session = client([{"group_by": [{"key": "2020", "count": 5},
                                       {"key": 2021, "count": "7"}, {"bad": 1}]}])
    assert c.count_works_by("type:article", "publication_year") == {"2020": 5, "2021": 7}
    assert session.calls[0] == {"group_by": "publication_year", "filter": "type:article"}


def test_count_works_by_rejects_bad_response():
    c, _ = client([{"results": []}])
    with pytest.raises(OpenAlexError):
        c.count_works_by("f", "publication_year")
