"""Crowding score per research-map area: is an area taking more than its usual share?

Usage:
    python -m src.live.crowding                  # OpenAlex, by year (the default for now)
    python -m src.live.crowding --source arxiv   # arXiv q-fin, by week (needs map tags)

Areas are the 8 research-map areas the team chose (src.live.map_tags.MAP_AREAS, the
same list as research_data.AREAS on the site); every other paper counts towards the
total only.

Sources:
    openalex  B's data/network_nodes.csv: 380 sampled OpenAlex quant-finance papers a
              year, already assigned to B's areas. One period = one calendar year.
    arxiv     data/arxiv_map_tags.csv from src.live.map_tags (Gemini tags new arXiv
              q-fin papers into the same 8 areas, or "none"). One period = one week
              (Monday). Papers Gemini hasn't reached yet (area "") are left out.

For each period and area:
    observed  = area's papers in the last `window` periods (this one included)
    share     = area's share of all papers over the `baseline` periods before that
                window (smoothed by +0.5 per column so a quiet area never gets share 0)
    expected  = share * all papers in the same window
    score     = (observed - expected) / sqrt(expected)        Poisson z-score

So 0 means "its usual share", +2 "unusually crowded", -2 "unusually quiet". Using the
share rather than the raw count controls for the field as a whole growing, and for
the current period being partial: observed and expected cover the same days.

Periods with fewer than `min_baseline` periods of history are dropped. paper_count is
the area's count in that single period. Writes data/crowding.csv (the contract) and
data/crowding_meta.json (source, period unit, window) for the briefing and the site.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import contracts
from src.common.contracts import DATA_DIR
from src.live.map_tags import MAP_AREAS, MAP_TAGS_CSV

AREAS = list(MAP_AREAS)
OTHER = "(other)"          # every paper outside the 8 areas, so shares are of all papers
SHARE_PRIOR = 0.5
META_JSON = DATA_DIR / "crowding_meta.json"
NODES_CSV = DATA_DIR / "network_nodes.csv"

SOURCES: dict[str, dict] = {
    # 380 papers a year over 25 areas: about 15 per area a year, so one year is the
    # window and the five years before it are "usual".
    "openalex": {"unit": "year", "window": 1, "baseline": 5, "min_baseline": 3},
    "arxiv": {"unit": "week", "window": 4, "baseline": 52, "min_baseline": 26},
}


def period_counts(period: pd.Series, area: pd.Series, freq: str) -> pd.DataFrame:
    """Periods x (AREAS + OTHER) matrix of paper counts, gaps filled with 0."""
    col = area.where(area.isin(AREAS), OTHER)
    counts = pd.crosstab(period, col)
    index = pd.date_range(counts.index.min(), counts.index.max(), freq=freq)
    return counts.reindex(index=index, columns=AREAS + [OTHER], fill_value=0)


def openalex_counts(nodes: pd.DataFrame) -> pd.DataFrame:
    """Yearly counts from network_nodes.csv (paper_id, title, year, area, ...)."""
    year = pd.to_datetime(nodes["year"].astype(int).astype(str) + "-01-01")
    return period_counts(year, nodes["area"], "YS")


def arxiv_counts(tags: pd.DataFrame) -> pd.DataFrame:
    """Weekly counts (weeks start Monday) from arxiv_map_tags.csv (published, area)."""
    tags = tags[tags["area"].fillna("") != ""]                 # not tagged yet
    published = pd.to_datetime(tags["published"].str[:10])
    monday = (published - pd.to_timedelta(published.dt.weekday, unit="D")).dt.normalize()
    return period_counts(monday, tags["area"], "7D")


def crowding_scores(counts: pd.DataFrame, window: int, baseline: int,
                    min_baseline: int) -> pd.DataFrame:
    """Long table in the crowding.csv contract, for AREAS only."""
    observed = counts.rolling(window, min_periods=1).sum()
    window_total = observed.sum(axis=1)

    # Baseline = the `baseline` periods ending just before the current window.
    base = counts.shift(window).rolling(baseline, min_periods=min_baseline).sum()
    share = (base + SHARE_PRIOR).div(base.sum(axis=1) + SHARE_PRIOR * counts.shape[1], axis=0)
    expected = share.mul(window_total, axis=0)
    score = ((observed - expected) / np.sqrt(expected))[AREAS]

    keep = base.notna().all(axis=1) & (window_total > 0)
    return pd.DataFrame({
        "date": np.repeat(counts.index[keep].strftime("%Y-%m-%d"), len(AREAS)),
        "area": AREAS * int(keep.sum()),
        "paper_count": counts.loc[keep, AREAS].to_numpy().ravel().astype(int),
        "crowding_score": score[keep].to_numpy().ravel().round(2),
    })


def build(source: str) -> pd.DataFrame:
    p = SOURCES[source]
    if source == "openalex":
        if not NODES_CSV.exists():
            raise FileNotFoundError(f"{NODES_CSV} missing; it comes from B's research map")
        counts = openalex_counts(pd.read_csv(NODES_CSV))
    else:
        if not MAP_TAGS_CSV.exists():
            raise FileNotFoundError(f"{MAP_TAGS_CSV} missing; run python -m src.live.map_tags")
        counts = arxiv_counts(pd.read_csv(MAP_TAGS_CSV, dtype=str, keep_default_na=False))
    return crowding_scores(counts, p["window"], p["baseline"], p["min_baseline"])


def load_meta() -> dict:
    """Which source crowding.csv came from. Files from before the meta were weekly arXiv."""
    if META_JSON.exists():
        return json.loads(META_JSON.read_text(encoding="utf-8"))
    return {"source": "arxiv", **SOURCES["arxiv"]}


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Crowding score per research-map area.")
    parser.add_argument("--source", choices=list(SOURCES), default="openalex")
    args = parser.parse_args(argv)

    out = build(args.source)
    path = contracts.save("crowding", out)
    META_JSON.write_text(json.dumps({"source": args.source, **SOURCES[args.source]},
                                    indent=2) + "\n", encoding="utf-8")
    latest = out[out["date"] == out["date"].max()].sort_values("crowding_score",
                                                               ascending=False)
    unit = SOURCES[args.source]["unit"]
    print(f"wrote {len(out)} rows ({out['date'].nunique()} {unit}s, {args.source}) to {path}")
    print(f"{unit} of {latest['date'].iloc[0]}:")
    for r in latest.itertuples():
        print(f"  {r.crowding_score:+5.2f}  {r.paper_count:3d}  {r.area}")
    return path


if __name__ == "__main__":
    main()
