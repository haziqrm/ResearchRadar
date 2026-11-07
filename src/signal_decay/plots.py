"""Before/after charts: cumulative return with a publication marker."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; works without a display

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import contracts  # noqa: E402
from src.signal_decay import CHARTS_DIR  # noqa: E402


def _load_factors() -> pd.DataFrame:
    return pd.read_csv(contracts.path_for("factors"), parse_dates=["date"])


def plot_before_after(
    signal: str,
    factors: pd.DataFrame | None = None,
    papers: pd.DataFrame | None = None,
    save_path=None,
):
    """Plot a signal's cumulative return with a vertical line at publication.

    signal is the name in papers.csv; its ken_french_column picks the series
    in factors.csv (falls back to signal itself if it is not in papers.csv).
    factors and papers default to the real data/factors.csv and data/papers.csv.
    Returns the matplotlib Figure, and saves it if save_path is given.
    """
    from src.signal_decay.papers import load_papers

    if factors is None:
        factors = _load_factors()
    if papers is None:
        papers = load_papers()

    row = papers[papers["signal"] == signal]
    column, pub_date = signal, None
    if not row.empty:
        kf_col = row["ken_french_column"].iloc[0]
        column = kf_col if isinstance(kf_col, str) and kf_col else signal
        pub_date = row["publication_date"].iloc[0]

    series = factors[factors["signal"] == column].set_index("date")["return"].sort_index()
    if series.empty:
        raise ValueError(f"No returns found for signal {signal!r} (column {column!r})")
    cumulative = (1 + series).cumprod() - 1

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(cumulative.index, cumulative.values, lw=1.4, label=column)
    if pub_date is not None and pd.notna(pub_date):
        ax.axvline(pd.Timestamp(pub_date), color="crimson", ls="--", lw=1.2, label="publication")
    ax.legend()
    ax.set_title(f"{signal}: cumulative return")
    ax.set_ylabel("Cumulative return")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def save_all_charts(
    papers: pd.DataFrame, factors: pd.DataFrame | None = None, out_dir: Path = CHARTS_DIR
) -> list[Path]:
    """Save one PNG per paper row to out_dir (data/cache/charts by default)."""
    factors = _load_factors() if factors is None else factors
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for signal in papers["signal"]:
        path = out_dir / f"{signal}.png"
        fig = plot_before_after(signal, factors=factors, papers=papers, save_path=path)
        plt.close(fig)
        paths.append(path)
    return paths
