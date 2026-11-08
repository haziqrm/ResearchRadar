"""Weekly crowding score per area from the tagged arXiv papers.

Usage:
    python -m src.live.crowding

For each week (dated by its Monday) and area:
    observed  = area's papers in the last WINDOW_WEEKS weeks (this week included)
    share     = area's share of all tagged papers over the BASELINE_WEEKS before that
                window (smoothed by +0.5 per area so a quiet area never gets share 0)
    expected  = share * all tagged papers in the same window
    score     = (observed - expected) / sqrt(expected)        Poisson z-score

So 0 means "its usual share", +2 means "unusually crowded", -2 "unusually quiet".
Using the share rather than the raw count controls for q-fin as a whole growing
(about 100 papers a month in late 2024, about 240 by late 2026); otherwise
every area would look crowded. The current week may be partial; the share
method handles that, because observed and expected cover the same days.

Weeks with fewer than MIN_BASELINE_WEEKS of history are dropped.
paper_count is the area's count in that single week.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import contracts
from src.common.contracts import AREAS
from src.live.tagging import TAGGED_CSV

WINDOW_WEEKS = 4
BASELINE_WEEKS = 52
MIN_BASELINE_WEEKS = 26
SHARE_PRIOR = 0.5


def weekly_counts(tagged: pd.DataFrame) -> pd.DataFrame:
    """Weeks (Monday dates) x AREAS matrix of paper counts, gaps filled with 0."""
    df = tagged[tagged["area"].isin(AREAS)].copy()
    published = pd.to_datetime(df["published"])
    df["week"] = (published - pd.to_timedelta(published.dt.weekday, unit="D")).dt.normalize()
    counts = df.groupby(["week", "area"]).size().unstack(fill_value=0)
    weeks = pd.date_range(counts.index.min(), counts.index.max(), freq="7D")
    return counts.reindex(index=weeks, columns=AREAS, fill_value=0)


def crowding_scores(counts: pd.DataFrame) -> pd.DataFrame:
    """Long table in the crowding.csv contract."""
    observed = counts.rolling(WINDOW_WEEKS, min_periods=1).sum()
    window_total = observed.sum(axis=1)

    # Baseline = the BASELINE_WEEKS ending just before the current window.
    before = counts.shift(WINDOW_WEEKS)
    base = before.rolling(BASELINE_WEEKS, min_periods=MIN_BASELINE_WEEKS).sum()
    share = (base + SHARE_PRIOR).div(base.sum(axis=1) + SHARE_PRIOR * len(AREAS), axis=0)
    expected = share.mul(window_total, axis=0)
    score = (observed - expected) / np.sqrt(expected)

    keep = base.notna().all(axis=1) & (window_total > 0)
    long = pd.DataFrame({
        "date": np.repeat(counts.index[keep].strftime("%Y-%m-%d"), len(AREAS)),
        "area": AREAS * int(keep.sum()),
        "paper_count": counts[keep].to_numpy().ravel().astype(int),
        "crowding_score": score[keep].to_numpy().ravel().round(2),
    })
    return long


def main(argv: list[str] | None = None) -> Path:
    argparse.ArgumentParser(description="Weekly crowding score per area.").parse_args(argv)
    if not TAGGED_CSV.exists():
        raise FileNotFoundError(f"{TAGGED_CSV} missing; run python -m src.live.tagging")
    tagged = pd.read_csv(TAGGED_CSV, dtype=str, keep_default_na=False)
    out = crowding_scores(weekly_counts(tagged))
    path = contracts.save("crowding", out)
    latest = out[out["date"] == out["date"].max()].sort_values("crowding_score",
                                                               ascending=False)
    print(f"wrote {len(out)} rows ({out['date'].nunique()} weeks) to {path}")
    print(f"week of {latest['date'].iloc[0]}:")
    for r in latest.itertuples():
        print(f"  {r.crowding_score:+5.2f}  {r.paper_count:3d}  {r.area}")
    return path


if __name__ == "__main__":
    main()
