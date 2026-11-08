from datetime import date

from src.live import arxiv_feed as af

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <opensearch:totalResults>{total}</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2410.01234v2</id>
    <updated>2024-10-09T12:00:00Z</updated>
    <published>2024-10-02T08:00:00Z</published>
    <title>Momentum   Crashes
      Revisited</title>
    <summary>  We study momentum.
      It crashes.  </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <arxiv:primary_category term="q-fin.PM" scheme="http://arxiv.org/schemas/atom"/>
    <category term="q-fin.PM" scheme="http://arxiv.org/schemas/atom"/>
    <category term="stat.ML" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>"""


def test_month_starts_crosses_year():
    assert af.month_starts(date(2026, 2, 15), 3) == [
        date(2025, 12, 1), date(2026, 1, 1), date(2026, 2, 1)]


def test_next_month_december():
    assert af.next_month(date(2025, 12, 1)) == date(2026, 1, 1)


def test_build_query_covers_whole_month():
    q = af.build_query(date(2024, 2, 1), date(2024, 3, 1))
    assert "cat:q-fin.TR" in q and "cat:q-fin.CP" in q
    assert "submittedDate:[202402010000 TO 202402292359]" in q


def test_parse_feed_cleans_fields():
    rows, total = af.parse_feed(ATOM.format(total=1))
    assert total == 1
    assert rows == [{
        "arxiv_id": "2410.01234",
        "title": "Momentum Crashes Revisited",
        "abstract": "We study momentum. It crashes.",
        "published": "2024-10-02",
        "updated": "2024-10-09",
        "primary_category": "q-fin.PM",
        "categories": "q-fin.PM stat.ML",
        "authors": "Ada Lovelace; Alan Turing",
    }]


def test_fetch_month_pages_until_total(monkeypatch):
    calls = []

    def fake_page(month, offset, use_cache):
        calls.append(offset)
        return ATOM.format(total=2500)

    monkeypatch.setattr(af, "fetch_page", fake_page)
    rows = af.fetch_month(date(2024, 10, 1), today=date(2026, 1, 1), refresh=False)
    assert calls == [0, 1000, 2000]
    assert len(rows) == 3


def test_open_month_refetched_only_on_refresh(monkeypatch):
    seen = []
    monkeypatch.setattr(af, "fetch_page",
                        lambda m, o, use_cache: seen.append(use_cache) or ATOM.format(total=1))
    today = date(2026, 11, 7)
    af.fetch_month(date(2026, 11, 1), today, refresh=True)   # open month
    af.fetch_month(date(2026, 9, 1), today, refresh=True)    # closed month
    af.fetch_month(date(2026, 11, 1), today, refresh=False)
    assert seen == [False, True, True]


def test_fetch_recent_dedups_by_id(monkeypatch):
    monkeypatch.setattr(af, "fetch_page", lambda m, o, u: ATOM.format(total=1))
    df = af.fetch_recent(months=2, today=date(2026, 1, 15))
    assert list(df.columns) == af.PAPER_COLUMNS
    assert len(df) == 1
