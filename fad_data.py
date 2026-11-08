"""Fad check (optional add-on): how each map topic's share of research has moved.

GET /api/fads -> get_fads(). Standard library only, like live_data.py. Descriptive:
it shows the trends and labels them; it makes no prediction.
To remove it: delete fad_data.py, static/js/fads.js, static/css/fads.css,
tests/test_fads.py and the lines marked "Fad check" in html_main.py and live.js.

Data: data/network_nodes.csv, B's OpenAlex sample of 380 random quant-finance papers
a year (2026 partial), each placed in one of 25 areas.

  share(a, y)  100 * area a's papers in year y / all papers in y.
  S(a, y)      mean share over y-2..y (trailing 3 years). One year of 380 papers is
               noisy: one paper is 0.26 points.
  label        from S at the last full year, checked in this order:
               booming   S >= BOOM_RATIO * S five years earlier, and BOOM_POINTS higher
               past fad  below FADE_SHARE of its peak, after a boom into that peak
               fading    below FADE_SHARE of its peak, without a boom
               steady    anything else
"""

import csv
from pathlib import Path

import research_data

DATA_DIR = Path(__file__).resolve().parent / "data"

SMOOTH = 3          # years in the trailing mean
LOOK = 5            # years back for a boom
BOOM_RATIO = 1.5
BOOM_POINTS = 1.0   # percentage points
FADE_SHARE = 0.6    # of the peak


def _rows(name):
    path = DATA_DIR / name
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def shares():
    """{area: {year: share %}}, the years, and the last full year (the latest is partial)."""
    total, count = {}, {}
    for r in _rows("network_nodes.csv") or []:
        y = int(r["year"])
        total[y] = total.get(y, 0) + 1
        count.setdefault(r["area"], {})
        count[r["area"]][y] = count[r["area"]].get(y, 0) + 1
    years = sorted(total)
    out = {a: {y: 100 * c.get(y, 0) / total[y] for y in years} for a, c in count.items()}
    full = [y for y in years if total[y] >= 0.97 * max(total.values())] if years else []
    return out, years, max(full) if full else None


def smooth(series, years):
    """Trailing SMOOTH-year mean; None until there are SMOOTH years."""
    out = {}
    for i, y in enumerate(years):
        if i + 1 >= SMOOTH:
            out[y] = sum(series[years[j]] for j in range(i - SMOOTH + 1, i + 1)) / SMOOTH
    return out


def is_boom(s, t):
    if t not in s or t - LOOK not in s:
        return False
    return s[t] >= BOOM_RATIO * s[t - LOOK] and s[t] - s[t - LOOK] >= BOOM_POINTS


def label(s, last_full):
    if is_boom(s, last_full):
        return "booming"
    past = {y: v for y, v in s.items() if y <= last_full - 3}
    peak_year = max(past, key=past.get)
    if s[last_full] <= FADE_SHARE * past[peak_year]:
        rose = any(is_boom(s, y) for y in range(peak_year - 3, peak_year + 1))
        return "past fad" if rose else "fading"
    return "steady"


def get_fads():
    raw, years, last_full = shares()
    if not raw:
        return {"error": "network_nodes.csv not found"}
    topics = []
    for a in research_data.AREAS:
        s = smooth(raw[a], years)
        past = {y: v for y, v in s.items() if y <= last_full}
        peak_year = max(past, key=past.get)
        topics.append({
            "topic": a, "label": label(s, last_full),
            "now": round(s[last_full], 1), "five_ago": round(s[last_full - LOOK], 1),
            "peak": round(past[peak_year], 1), "peak_year": peak_year,
            "series": [{"year": y, "share": round(raw[a][y], 1),
                        "smooth": round(s[y], 1) if y in s else None} for y in years],
        })
    return {"years": [years[0], years[-1]], "last_full": last_full, "topics": topics,
            "rules": {"smooth": SMOOTH, "look": LOOK, "ratio": BOOM_RATIO,
                      "points": BOOM_POINTS, "fade": FADE_SHARE}}


if __name__ == "__main__":
    for t in get_fads()["topics"]:
        print(f'{t["topic"]:36} {t["label"]:9} now {t["now"]:5} 5y ago {t["five_ago"]:5} '
              f'peak {t["peak"]:5} ({t["peak_year"]})')
