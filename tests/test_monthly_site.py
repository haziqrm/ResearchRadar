"""The /monthly alias of the site: monthly_data.py and its routes in html_main.py."""
import csv
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import html_main
import monthly_data as md
import research_data

AREAS = research_data.AREAS


def write_tags(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arxiv_id", "published", "title", "area", "confidence"])
        for i, (published, area) in enumerate(rows):
            w.writerow([f"p{i}", published, f"paper {i}", area, "0.9"])


@pytest.fixture
def tags_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(md, "DATA_DIR", tmp_path)
    return tmp_path


def test_counts_by_month_with_gaps_and_partial_last_month(tags_dir):
    a = AREAS[0]
    write_tags(tags_dir / md.TAGS_FILE, [("2025-01-03", a), ("2025-01-20", "none"),
                                         ("2025-03-02", a), ("2025-03-04", "")])
    net = md.get_network()
    assert net["slider_years"] == ["2025-01", "2025-02", "2025-03"]   # Feb kept, at 0
    node = next(n for n in net["nodes"] if n["id"] == a)
    assert node["by_year"] == {"2025-01": 1, "2025-02": 0, "2025-03": 1}
    assert net["papers"] == 3            # "none" counts towards the total; "" (untagged) doesn't
    assert net["partial"] == "2025-03" and net["start"] == "2025-02"
    assert net["labels"]["2025-03"] == "Mar 2025 (to 2 Mar)"
    assert net["labels"]["2025-01"] == "Jan 2025"


def test_full_last_month_is_not_partial(tags_dir):
    write_tags(tags_dir / md.TAGS_FILE, [("2025-01-03", AREAS[0]), ("2025-02-28", AREAS[1])])
    net = md.get_network()
    assert net["partial"] is None and net["start"] == "2025-02"


def test_crowding_needs_history_and_flags_a_jump():
    months = [f"2025-{m:02d}" for m in range(1, 9)]
    counts = {}
    for m in months:
        counts[(AREAS[0], m)] = 2
        counts[(md.OTHER, m)] = 18
    counts[(AREAS[0], "2025-08")] = 12            # 12 of 30 vs a usual 10%
    from collections import Counter
    scores = md.crowding_scores(months, Counter(counts))[AREAS[0]]
    assert min(scores) == f"2025-{md.MIN_BASELINE + 1:02d}"   # first months: no history
    assert abs(scores["2025-07"]) < 0.5                        # usual share
    assert scores["2025-08"] > 2                               # crowded


def test_missing_tags_give_error(tags_dir):
    assert "error" in md.get_network()
    assert "error" in md.get_crowding_history()


def test_real_tags_cover_every_paper():
    net = md.get_network()
    assert "error" not in net
    assert net["papers"] == sum(1 for _ in open(md.DATA_DIR / md.TAGS_FILE, encoding="utf-8")) - 1
    assert {n["id"] for n in net["nodes"]} == set(AREAS)
    assert all(set(l["by_year"]) == set(net["slider_years"]) for l in net["links"])


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), html_main.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.headers["Content-Type"], r.read()


@pytest.mark.parametrize("path", ["/api/network/monthly", "/api/crowding/history/monthly"])
def test_monthly_routes_return_json(server, path):
    status, ctype, body = get(server + path)
    assert status == 200 and ctype == "application/json"
    assert "error" not in json.loads(body)


def test_monthly_page_uses_monthly_data(server):
    _, ctype, page = get(server + "/monthly")
    assert ctype.startswith("text/html")
    assert b'"network": "/api/network/monthly"' in page
    assert b"papers that month" in page and b'href="/"' in page
    assert b"__" + b"MAP_MODE__" not in page


def test_main_page_still_yearly(server):
    _, _, page = get(server + "/")
    assert b'"network": "/api/network"' in page
    assert b"articles that year" in page
    assert b"/api/network/monthly" not in page and b"This is the monthly view" not in page
