"""Download and tidy monthly factor returns from the Ken French Data Library.

Run directly to download and print each signal's date range:
    python -m src.signal_decay.ken_french

Files used (<name>_CSV.zip) and expected first month:
    F-F_Research_Data_Factors        Mkt-RF, SMB_3f, HML_3f   1926-07
    F-F_ST_Reversal_Factor           ST_Rev                   1926-02
    F-F_LT_Reversal_Factor           LT_Rev                   1931-01
    F-F_Momentum_Factor              Mom                      1927-01
    F-F_Research_Data_5_Factors_2x3  SMB, HML, RMW, CMA       1963-07

SMB and HML are the 5-factor versions; SMB_3f and HML_3f are the 3-factor
versions, which are built differently and start earlier. Mkt-RF is kept in
factors.csv as a benchmark only and is never decay-tested.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

from src.signal_decay import RAW_DIR

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

# File stem -> {raw column name: signal name in factors.csv}.
# Columns not listed (e.g. RF, the 5-factor Mkt-RF) are dropped.
DATASETS: dict[str, dict[str, str]] = {
    "F-F_Research_Data_Factors": {"Mkt-RF": "Mkt-RF", "SMB": "SMB_3f", "HML": "HML_3f"},
    "F-F_ST_Reversal_Factor": {"ST_Rev": "ST_Rev"},
    "F-F_LT_Reversal_Factor": {"LT_Rev": "LT_Rev"},
    "F-F_Momentum_Factor": {"Mom": "Mom"},
    "F-F_Research_Data_5_Factors_2x3": {"SMB": "SMB", "HML": "HML", "RMW": "RMW", "CMA": "CMA"},
}

# Signals kept in factors.csv for reference but excluded from decay tests.
BENCHMARK_SIGNALS = {"Mkt-RF"}

ALL_SIGNALS = {name for mapping in DATASETS.values() for name in mapping.values()}
TESTABLE_SIGNALS = ALL_SIGNALS - BENCHMARK_SIGNALS

# Ken French files use these sentinels (in percent) for missing values.
MISSING_SENTINELS = (-99.99, -999.0)

_MONTH_ROW = re.compile(r"^\s*(\d{6})\s*,")


def _download_zip(name: str, raw_dir: Path, refresh: bool = False) -> Path:
    """Download <name>_CSV.zip into raw_dir unless it is already cached there."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{name}_CSV.zip"
    if path.exists() and not refresh:
        return path
    resp = requests.get(f"{BASE_URL}{name}_CSV.zip", timeout=60)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def _read_zip_text(path: Path) -> str:
    """Return the text of the CSV inside a Ken French zip file."""
    with zipfile.ZipFile(path) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        return zf.read(member).decode("latin-1")


def parse_monthly_section(text: str) -> pd.DataFrame:
    """Parse the first monthly table out of a Ken French CSV.

    The raw files have free-text header lines, then a column-name line
    (starting with a comma), then YYYYMM rows, then an annual table and a
    copyright footer. Only the first block of YYYYMM rows is kept.

    Returns a wide DataFrame indexed by month-end date, with returns in
    decimals (the raw files are in percent).
    """
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if _MONTH_ROW.match(ln)), None)
    if start is None:
        raise ValueError("No monthly YYYYMM rows found in Ken French file")

    # Column names are on the nearest non-blank line above the first data row.
    header_idx = start - 1
    while header_idx >= 0 and not lines[header_idx].strip():
        header_idx -= 1
    if header_idx < 0:
        raise ValueError("No column header found above the monthly rows")
    columns = [c.strip() for c in lines[header_idx].split(",")[1:]]

    rows = []
    for ln in lines[start:]:
        if not _MONTH_ROW.match(ln):
            break  # end of the monthly block (blank line or annual section)
        parts = [p.strip() for p in ln.split(",")]
        rows.append([parts[0]] + parts[1 : len(columns) + 1])

    df = pd.DataFrame(rows, columns=["yyyymm"] + columns)
    df.index = pd.to_datetime(df.pop("yyyymm"), format="%Y%m") + pd.offsets.MonthEnd(0)
    df.index.name = "date"
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.mask(df.isin(MISSING_SENTINELS))
    return df / 100.0


def to_tidy(wide: pd.DataFrame, rename: dict[str, str]) -> pd.DataFrame:
    """Keep the columns in ``rename``, rename them, and return tidy (date, signal, return)."""
    missing = [c for c in rename if c not in wide.columns]
    if missing:
        raise ValueError(f"Expected columns {missing} not found; got {list(wide.columns)}")
    tidy = (
        wide[list(rename)]
        .rename(columns=rename)
        .reset_index()
        .melt(id_vars="date", var_name="signal", value_name="return")
        .dropna(subset=["return"])
    )
    return tidy.sort_values(["signal", "date"]).reset_index(drop=True)


def download_ken_french(
    datasets: dict[str, dict[str, str]] | None = None,
    raw_dir: Path = RAW_DIR,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download Ken French datasets and return tidy monthly returns.

    Raw zip files are saved to data/raw; repeat runs reuse them unless
    refresh=True. Returns columns date, signal, return (decimals).
    """
    frames = []
    for name, rename in (datasets or DATASETS).items():
        text = _read_zip_text(_download_zip(name, raw_dir, refresh=refresh))
        frames.append(to_tidy(parse_monthly_section(text), rename))
    tidy = pd.concat(frames, ignore_index=True)
    dupes = tidy.duplicated(subset=["date", "signal"])
    if dupes.any():
        raise ValueError(f"Duplicate signals across datasets: {sorted(tidy.loc[dupes, 'signal'].unique())}")
    return tidy.sort_values(["signal", "date"]).reset_index(drop=True)


def main() -> None:
    """Download all datasets and print the date range of each signal."""
    df = download_ken_french()
    summary = df.groupby("signal")["date"].agg(["min", "max", "count"])
    summary["min"] = summary["min"].dt.strftime("%Y-%m")
    summary["max"] = summary["max"].dt.strftime("%Y-%m")
    summary.columns = ["first_month", "last_month", "n_months"]
    summary["role"] = ["benchmark" if s in BENCHMARK_SIGNALS else "signal" for s in summary.index]
    print(summary.to_string())


if __name__ == "__main__":
    main()
