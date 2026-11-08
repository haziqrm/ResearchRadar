"""Synthetic, hard-coded data for the stock tracker site.

Every function here returns plain JSON-able data. When real data is ready,
replace the body of each function (or point html_main.py at a new module with
the same functions). The website itself does not need to change.
"""
import csv
from datetime import date
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# ---------------------------------------------------------------- account

def get_account():
    """Account summary shown at the top of the page."""
    return {
        "value": 128_450.32,
        "cash": 23_110.05,
        "invested": 105_340.27,
        "day_change": 1_284.77,
        "day_change_pct": 1.01,
        "currency": "USD",
    }


# ---------------------------------------------------------------- stocks

_MOST_TRADED = [
    # ticker, name, price, change %, volume (shares)
    ("NVDA", "NVIDIA Corp", 142.18, 2.84, 312_450_000),
    ("TSLA", "Tesla Inc", 251.06, -1.92, 118_920_000),
    ("AAPL", "Apple Inc", 228.73, 0.41, 64_310_000),
    ("AMD", "Advanced Micro Devices", 156.40, 3.37, 58_770_000),
    ("PLTR", "Palantir Technologies", 41.22, 5.12, 55_040_000),
    ("AMZN", "Amazon.com Inc", 196.88, -0.58, 47_660_000),
    ("SOFI", "SoFi Technologies", 10.94, -2.40, 44_180_000),
    ("INTC", "Intel Corp", 22.67, 1.06, 41_350_000),
    ("F", "Ford Motor Co", 10.51, -0.85, 39_820_000),
    ("MSFT", "Microsoft Corp", 418.12, 0.73, 22_910_000),
]


def get_most_traded():
    """Today's most traded stocks, highest volume first."""
    rows = [
        {"ticker": t, "name": n, "price": p, "change_pct": c, "volume": v}
        for t, n, p, c, v in _MOST_TRADED
    ]
    return sorted(rows, key=lambda r: r["volume"], reverse=True)


# ---------------------------------------------------------------- research

# The Monday that starts the week this snapshot covers.
SNAPSHOT_WEEK = date(2026, 11, 2)

# topic, popularity this week (0-100),
# research papers out last week, research papers out this week
_TOPICS = [
    ("12-1 month momentum", 92, 48, 41),
    ("Low volatility anomaly", 88, 39, 42),
    ("Post-earnings announcement drift", 83, 35, 31),
    ("LLM-generated earnings summaries", 79, 22, 34),
    ("Retail sentiment from Reddit", 71, 27, 30),
    ("Short interest squeeze", 64, 18, 15),
    ("Quality minus junk", 60, 21, 23),
    ("Earnings call tone (NLP)", 52, 16, 19),
    ("Supply chain network momentum", 47, 12, 14),
    ("ESG controversy signals", 41, 15, 11),
    ("Options flow imbalance", 34, 9, 12),
    ("Insider buying clusters", 29, 10, 8),
    ("Job postings growth", 24, 6, 7),
    ("Satellite imagery of retail car parks", 18, 5, 4),
    ("Credit card transaction panels", 13, 3, 5),
    ("Patent citation novelty", 8, 2, 2),
]


def get_research_snapshot():
    """This week's research topics: popularity (0-100) and paper counts."""
    rows = [
        {
            "topic": t,
            "popularity": pop,
            "papers": now,
            "papers_last_week": before,
            "papers_change": now - before,
        }
        for t, pop, before, now in _TOPICS
    ]
    return {
        "week_start": SNAPSHOT_WEEK.isoformat(),
        "topics": sorted(rows, key=lambda r: r["popularity"], reverse=True),
    }


# Knock-on effects between topics come from a data analysis sheet saved as
# CSV, with columns: source, target, strength (0-1). Topic names must match
# the names above exactly. Rows with unknown topics are skipped.
LINKS_FILE = DATA_DIR / "topic_links.csv"


def get_topic_network():
    """Topics as nodes (sized by paper count) and knock-on effects as links."""
    topics = get_research_snapshot()["topics"]
    names = {t["topic"] for t in topics}
    nodes = [
        {
            "id": t["topic"],
            "papers": t["papers"],
            "papers_last_week": t["papers_last_week"],
        }
        for t in topics
    ]

    links, skipped = [], []
    if LINKS_FILE.exists():
        with LINKS_FILE.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                source = (row.get("source") or "").strip()
                target = (row.get("target") or "").strip()
                try:
                    strength = float(row.get("strength") or 0.5)
                except ValueError:
                    strength = 0.5
                if source in names and target in names and source != target:
                    links.append({
                        "source": source,
                        "target": target,
                        "strength": max(0.0, min(1.0, strength)),
                    })
                else:
                    skipped.append(row)

    return {"nodes": nodes, "links": links, "skipped_rows": len(skipped)}
