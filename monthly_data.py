"""Month-by-month research map from the Gemini-tagged arXiv papers (Person D).

The main page's map counts B's OpenAlex sample by year. This module feeds the
/monthly alias of that page instead: arXiv q-fin papers that Gemini tagged into the
same 8 map areas (data/arxiv_map_tags.csv, from src.live.map_tags), counted by
calendar month.

  get_network()           the 8 areas with their papers per month, and the links
  get_crowding_history()  crowding score per area and month (colours the circles)

Links: arXiv gives no reference lists, so the links are research_data's OpenAlex
links (papers citing the same earlier work), over all years. They are the same in
every month.

Crowding uses the method of src.live.crowding: the area's papers in the month vs
its share of all papers over the BASELINE months before, as a Poisson z-score. Months
with fewer than MIN_BASELINE months of history get no score.

Standard library only, like research_data.py.
"""
import csv
import math
from collections import Counter
from datetime import date
from pathlib import Path

import research_data

DATA_DIR = Path(__file__).resolve().parent / "data"
TAGS_FILE = "arxiv_map_tags.csv"
AREAS = research_data.AREAS
OTHER = "(other)"

BASELINE = 6        # months before the one being scored
MIN_BASELINE = 3    # fewer months of history than this: no score
SHARE_PRIOR = 0.5   # as src.live.crowding: a quiet area never gets share 0

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _tags():
    path = DATA_DIR / TAGS_FILE
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        # area "" = not tagged yet; leave those out (an empty area isn't "none")
        return [r for r in csv.DictReader(f) if r["area"] and r["published"]]


def _months(first, last):
    """Every month from first to last ("YYYY-MM"), gaps included."""
    y, m = int(first[:4]), int(first[5:7])
    out = []
    while f"{y:04d}-{m:02d}" <= last:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _label(month, last_day=None):
    text = f"{MONTH_NAMES[int(month[5:7]) - 1]} {month[:4]}"
    return f"{text} (to {last_day})" if last_day else text


def _counts():
    """months, counts[(area, month)] with OTHER for every non-map paper, and the
    newest paper's date (to mark the last month as partial)."""
    rows = _tags()
    if not rows:
        return None
    counts = Counter()
    for r in rows:
        area = r["area"] if r["area"] in AREAS else OTHER
        counts[(area, r["published"][:7])] += 1
    published = sorted(r["published"][:10] for r in rows)
    return _months(published[0][:7], published[-1][:7]), counts, published[-1]


def _is_partial(newest):
    """Is the newest paper's month still running? (Its last day isn't reached.)"""
    d = date.fromisoformat(newest)
    nxt = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return (nxt - d).days > 1


def crowding_scores(months, counts):
    """{area: {month: z}} by src.live.crowding's method, with a 1-month window."""
    total = {m: sum(counts[(a, m)] for a in AREAS + [OTHER]) for m in months}
    scores = {a: {} for a in AREAS}
    for i, m in enumerate(months):
        before = months[max(0, i - BASELINE):i]
        if len(before) < MIN_BASELINE:
            continue
        base_total = sum(total[b] for b in before) + SHARE_PRIOR * (len(AREAS) + 1)
        for a in AREAS:
            share = (sum(counts[(a, b)] for b in before) + SHARE_PRIOR) / base_total
            expected = share * total[m]
            if expected > 0:
                scores[a][m] = round((counts[(a, m)] - expected) / math.sqrt(expected), 2)
    return scores


def get_crowding_history():
    """{"scores": {area: {"YYYY-MM": z}}}, in the shape of live_data's yearly one."""
    got = _counts()
    if got is None:
        return {"error": f"data/{TAGS_FILE} not found; run: python -m src.live.map_tags"}
    months, counts, _ = got
    return {"source": "arxiv", "unit": "month", "window": 1, "baseline": BASELINE,
            "scores": crowding_scores(months, counts)}


def get_network():
    """The map's data by month, in the shape of research_data.get_topic_network()
    (by_year is keyed by month), plus the labels the slider shows."""
    got = _counts()
    if got is None:
        return {"error": f"data/{TAGS_FILE} not found; run: python -m src.live.map_tags"}
    months, counts, newest = got
    partial = _is_partial(newest)
    last = months[-1]
    labels = {m: _label(m) for m in months}
    if partial:
        labels[last] = _label(last, f"{int(newest[8:10])} {MONTH_NAMES[int(last[5:7]) - 1]}")

    circles = [{
        "id": a,
        "articles": sum(counts[(a, m)] for m in months),
        "by_year": {m: counts[(a, m)] for m in months},
    } for a in AREAS]

    # OpenAlex links over all years, the same in every month
    links = [{
        "source": l["source"], "target": l["target"],
        "shared": l["shared"], "strength": l["strength"],
        "by_year": {m: l["shared"] for m in months},
    } for l in research_data.get_topic_network()["links"]]

    return {
        "unit": "month",
        "slider_years": months,
        # where the slider starts: the newest full month
        "start": months[-2] if partial and len(months) > 1 else last,
        "labels": labels,
        "partial": last if partial else None,
        "papers": sum(counts.values()),
        "tagged_papers": sum(counts[(a, m)] for a in AREAS for m in months),
        "nodes": circles,
        "links": links,
    }
