"""Load and validate data/papers.csv (one row per signal and its source paper).

Rules:
- columns match the shared contract exactly (src.common.contracts)
- signal is non-empty and unique
- publication_date is YYYY-MM-DD (empty means unknown)
- every row with a date uses the same date_rule (team agreement: one rule)
- ken_french_column, if set, is a known non-benchmark Ken French signal
- source_verified is yes/no/true/false/1/0 or empty

Unverified rows are kept in the table but left out of results by
select_for_results() unless allow_unverified=True.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common import contracts
from src.signal_decay.ken_french import ALL_SIGNALS, BENCHMARK_SIGNALS

_TRUE = {"true", "1", "yes", "y"}
_FALSE = {"false", "0", "no", "n"}


def _blank(v) -> bool:
    return pd.isna(v) or str(v).strip() == ""


def validate_papers(df: pd.DataFrame, known_signals: set[str] | None = None) -> pd.DataFrame:
    """Check the rules in the module docstring; return a cleaned copy.

    Raises ValueError listing every problem found.
    """
    contracts.validate("papers", df)
    known = ALL_SIGNALS if known_signals is None else known_signals
    out = df.copy()
    problems = []

    sig = out["signal"].astype("string").str.strip()
    if sig.isna().any() or (sig == "").any():
        problems.append("empty signal name")
    dupes = sig[sig.duplicated()].dropna().unique().tolist()
    if dupes:
        problems.append(f"duplicate signals: {dupes}")

    raw_dates = out["publication_date"].astype("string").str.strip()
    parsed = pd.to_datetime(raw_dates, format="%Y-%m-%d", errors="coerce")
    bad = raw_dates.notna() & (raw_dates != "") & parsed.isna()
    if bad.any():
        problems.append(f"bad publication_date in rows {list(out.index[bad])}")
    out["publication_date"] = parsed

    rules = out.loc[parsed.notna(), "date_rule"]
    if rules.map(_blank).any():
        problems.append("date_rule missing for a row that has a publication_date")
    distinct = sorted({str(r).strip() for r in rules if not _blank(r)})
    if len(distinct) > 1:
        problems.append(f"mixed date rules {distinct}; the team agreed on one rule")

    for i, col in out["ken_french_column"].items():
        if _blank(col):
            continue
        if col in BENCHMARK_SIGNALS:
            problems.append(f"row {i}: {col} is a benchmark and cannot be decay-tested")
        elif col not in known:
            problems.append(f"row {i}: unknown ken_french_column {col!r}")

    def _to_bool(v):
        if _blank(v):
            return pd.NA
        s = str(v).strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
        problems.append(f"bad source_verified value {v!r}")
        return pd.NA

    out["source_verified"] = out["source_verified"].map(_to_bool).astype("boolean")

    if problems:
        raise ValueError("papers.csv invalid: " + "; ".join(problems))
    return out


def load_papers(path: Path | None = None) -> pd.DataFrame:
    """Read the real data/papers.csv (never the sample) and validate it.

    Raises FileNotFoundError if it does not exist, so results are never built
    from the made-up sample rows.
    """
    path = path or contracts.path_for("papers")
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found; add verified papers before running decay tests")
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    return validate_papers(df)


def select_for_results(
    papers: pd.DataFrame, allow_unverified: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split validated papers into (included, excluded) for the final results.

    Rows with source_verified not True are excluded unless allow_unverified
    is set. Included rows get a boolean ``unverified`` column so callers can
    mark them in decay_details.csv.
    """
    verified = papers["source_verified"].fillna(False).astype(bool)
    included = papers if allow_unverified else papers[verified]
    excluded = papers.iloc[0:0] if allow_unverified else papers[~verified]
    included = included.assign(unverified=~verified.loc[included.index])
    return included.copy(), excluded.copy()
