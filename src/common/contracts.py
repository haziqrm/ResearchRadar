"""Single source of truth for the shared area list and file contracts.

Changing anything here changes the agreement between A, B, C and D:
tell the team before you edit it.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"

# B and D must use these exact strings.
AREAS: list[str] = [
    "Factor and anomaly research",
    "Machine learning for returns",
    "Market microstructure",
    "Portfolio and risk",
    "Derivatives and volatility",
    "Alternative data and text",
    "Crypto and DeFi",
    "LLMs in finance",
]

LIFECYCLE_STAGES: list[str] = ["emerging", "peaking", "saturated"]

# file stem -> (owner, ordered column list)
CONTRACTS: dict[str, tuple[str, list[str]]] = {
    "factors": ("A", ["date", "signal", "return"]),
    "papers": ("A", ["signal", "paper", "publication_date", "date_rule",
                     "ken_french_column", "source_verified"]),
    "decay_results": ("A", ["signal", "paper", "publication_date", "pre_mean",
                            "post_mean", "half_life_months", "ci_low", "ci_high",
                            "early_citations"]),
    "network_nodes": ("B", ["paper_id", "title", "year", "area", "citations",
                            "lifecycle_stage"]),
    "network_edges": ("B", ["source_id", "target_id"]),
    "area_forecasts": ("B", ["area", "year", "growth_score", "boom_probability",
                             "predicted_boom", "actual_boom"]),
    "crowding": ("D", ["date", "area", "paper_count", "crowding_score"]),
    "backtest_results": ("D", ["date", "strategy", "monthly_return",
                               "cumulative_return"]),
}


def path_for(name: str, sample: bool = False) -> Path:
    if name not in CONTRACTS:
        raise KeyError(f"Unknown contract file {name!r}; known: {sorted(CONTRACTS)}")
    return (SAMPLE_DIR if sample else DATA_DIR) / f"{name}.csv"


def validate(name: str, df: pd.DataFrame) -> None:
    """Raise ValueError if df does not match the agreed contract."""
    expected = CONTRACTS[name][1]
    if list(df.columns) != expected:
        raise ValueError(f"{name}.csv columns {list(df.columns)} != contract {expected}")
    if "area" in df.columns:
        bad = set(df["area"].dropna()) - set(AREAS)
        if bad:
            raise ValueError(f"{name}.csv has areas not in the shared list: {sorted(bad)}")
    if "lifecycle_stage" in df.columns:
        bad = set(df["lifecycle_stage"].dropna()) - set(LIFECYCLE_STAGES)
        if bad:
            raise ValueError(f"{name}.csv has unknown lifecycle stages: {sorted(bad)}")


def load(name: str, prefer_real: bool = True) -> pd.DataFrame:
    """Load data/<name>.csv if it exists, otherwise data/sample/<name>.csv."""
    real = path_for(name)
    path = real if prefer_real and real.exists() else path_for(name, sample=True)
    df = pd.read_csv(path)
    validate(name, df)
    return df


def save(name: str, df: pd.DataFrame) -> Path:
    """Validate then write data/<name>.csv. Use this for every contract output."""
    validate(name, df)
    out = path_for(name)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
