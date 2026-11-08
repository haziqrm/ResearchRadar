"""Real research data for the frontend, read from the research map's CSVs.

The circles and the research topics table use the 8 trading-relevant areas
below, out of the 25 groups in data/groups.csv (spelling must match exactly).

Source files (built on the research_map branch, see src/common/contracts.py):

  data/groups.csv         group_id, name, keywords        the area names
  data/network_nodes.csv  paper_id, title, year, area,    one row per paper
                          citations, lifecycle_stage
  data/network_edges.csv  source_id, target_id            two papers that cite
                                                          the same earlier work
                                                          (undirected)

The papers are a random sample of 380 OpenAlex quant-finance papers per year,
so a count is an area's share of research attention, not every paper written.

Uses only the standard library, so the site still runs with plain `py`.
"""
import csv
import math
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# The areas shown as circles and in the table.
AREAS = [
    "Asset pricing and risk premia",
    "Information and stock reactions",
    "Trading and order flow",
    "Behavioral finance",
    "Volatility forecasting (GARCH)",
    "Deep learning and LLM forecasting",
    "Classical stock price prediction",
    "Crypto, COVID and spillovers",
]

# Each area keeps only its strongest links, so the graph is readable.
LINKS_PER_AREA = 3

# First year on the frontend's year slider (it runs to the latest year).
SLIDER_START = 2010


def _read(name):
    with (DATA_DIR / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _check_areas():
    known = {row["name"] for row in _read("groups.csv")}
    missing = [a for a in AREAS if a not in known]
    if missing:
        raise ValueError(f"areas not in data/groups.csv: {missing}")


def _area_counts():
    """Articles per area: in total, and per year. Plus each area's stage."""
    nodes = _read("network_nodes.csv")
    years = sorted({int(r["year"]) for r in nodes if r["year"]})
    total, by_year, stage = Counter(), Counter(), {}
    for r in nodes:
        area = r["area"]
        if area not in AREAS:
            continue
        total[area] += 1
        by_year[(area, int(r["year"]))] += 1
        stage[area] = r["lifecycle_stage"]
    return nodes, years, total, by_year, stage


def get_research_snapshot():
    """The research topics table: articles per area, and this year vs last."""
    _check_areas()
    _, years, total, by_year, stage = _area_counts()
    this_year, last_year = years[-1], years[-2]
    rows = []
    for area in AREAS:
        now, before = by_year[(area, this_year)], by_year[(area, last_year)]
        rows.append({
            "topic": area,
            "articles": total[area],
            "articles_this_year": now,
            "articles_last_year": before,
            "change": now - before,
            "stage": stage.get(area, ""),
            # articles in each year, keyed by year as text (JSON keys are text)
            "by_year": {str(y): by_year[(area, y)] for y in years},
        })
    rows.sort(key=lambda r: r["articles"], reverse=True)
    return {
        "first_year": years[0],
        "this_year": this_year,
        "last_year": last_year,
        "slider_years": [y for y in years if y >= SLIDER_START],
        "topics": rows,
    }


def get_topic_network():
    """Areas as circles and the links between them.

    Two areas are linked by how many of their papers cite the same earlier
    work. That count is divided by the geometric mean of the two areas' sizes
    (as the research map does for papers) so big areas do not link to
    everything. Each area keeps its LINKS_PER_AREA strongest links. Links have
    no direction.
    """
    _check_areas()
    nodes, years, total, by_year, _ = _area_counts()
    this_year, last_year = years[-1], years[-2]
    area_of = {r["paper_id"]: r["area"] for r in nodes}
    year_of = {r["paper_id"]: int(r["year"]) for r in nodes if r["year"]}

    # shared: all years together. shared_by_year: dated by the newer of the
    # two papers, i.e. the year that link appeared.
    shared, shared_by_year = Counter(), Counter()
    for e in _read("network_edges.csv"):
        a, b = area_of.get(e["source_id"]), area_of.get(e["target_id"])
        if a in AREAS and b in AREAS and a != b:
            pair = tuple(sorted((a, b)))
            shared[pair] += 1
            year = max(year_of.get(e["source_id"], 0), year_of.get(e["target_id"], 0))
            shared_by_year[(pair, year)] += 1

    score = {pair: n / math.sqrt(total[pair[0]] * total[pair[1]])
             for pair, n in shared.items()}
    keep = set()
    for area in AREAS:
        mine = sorted((p for p in score if area in p), key=score.get, reverse=True)
        keep.update(mine[:LINKS_PER_AREA])
    top = max((score[p] for p in keep), default=1)

    links = [
        {
            "source": a,
            "target": b,
            "shared": shared[(a, b)],
            "strength": round(score[(a, b)] / top, 3),
            "by_year": {str(y): shared_by_year[((a, b), y)] for y in years},
        }
        for a, b in sorted(keep, key=score.get, reverse=True)
    ]
    circles = [
        {
            "id": area,
            "articles": total[area],
            "articles_this_year": by_year[(area, this_year)],
            "articles_last_year": by_year[(area, last_year)],
            "by_year": {str(y): by_year[(area, y)] for y in years},
        }
        for area in AREAS
    ]
    return {
        "this_year": this_year,
        "last_year": last_year,
        "slider_years": [y for y in years if y >= SLIDER_START],
        "nodes": circles,
        "links": links,
    }
