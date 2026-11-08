"""Single source of truth for the shared area list and file contracts.

Changing anything here changes the agreement between A, B, C and D:
tell the team before you edit it.

Areas are the research groups B discovers from the papers, not a hand-written
list.  Their names live in ``data/groups.csv`` (``data/sample/groups.csv`` until
B has real output).  Use :func:`area_names` to read them; every ``area`` column
must use these exact strings.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample"

LIFECYCLE_STAGES: list[str] = ["emerging", "peaking", "saturated"]

# file stem -> (owner, ordered column list)
CONTRACTS: dict[str, tuple[str, list[str]]] = {
    "factors": ("A", ["date", "signal", "return"]),
    "papers": ("A", ["signal", "paper", "publication_date", "date_rule",
                     "ken_french_column", "source_verified"]),
    "decay_results": ("A", ["signal", "paper", "publication_date", "pre_mean",
                            "post_mean", "half_life_months", "ci_low", "ci_high",
                            "early_citations"]),
    # The shared area list: one row per discovered research group. `name` is the
    # string used in every `area` column; `keywords` are the automatic top terms.
    "groups": ("B", ["group_id", "name", "keywords"]),
    "network_nodes": ("B", ["paper_id", "title", "year", "area", "citations",
                            "lifecycle_stage"]),
    # Undirected links between papers that cite the same earlier work. `shared` is
    # the number of references in common; `strength` (0 to 1) is shared divided by
    # the geometric mean of both reference-list lengths. Use it to filter or weight.
    "network_edges": ("B", ["source_id", "target_id", "shared", "strength"]),
    "area_forecasts": ("B", ["area", "year", "growth_score", "boom_probability",
                             "predicted_boom", "actual_boom"]),
    # topic_ids / topic_scores are ";"-separated, topic_labels are " | "-separated,
    # all parallel. Heavy vectors live in data/paper_vectors.parquet (not a contract).
    "paper_encodings": ("B", ["paper_id", "top_topic", "topic_ids", "topic_scores",
                              "topic_labels", "model", "text_quality"]),
    "crowding": ("D", ["date", "area", "paper_count", "crowding_score"]),
    "backtest_results": ("D", ["date", "strategy", "monthly_return",
                               "cumulative_return"]),
}


def path_for(name: str, sample: bool = False) -> Path:
    if name not in CONTRACTS:
        raise KeyError(f"Unknown contract file {name!r}; known: {sorted(CONTRACTS)}")
    return (SAMPLE_DIR if sample else DATA_DIR) / f"{name}.csv"


def area_names(sample: bool = False) -> list[str]:
    """The shared area list: group names from data/groups.csv (or the sample)."""
    real = path_for("groups")
    path = real if real.exists() and not sample else path_for("groups", sample=True)
    return [str(name) for name in pd.read_csv(path)["name"]]


def __getattr__(name: str):
    # `from src.common.contracts import AREAS` keeps working and is always current.
    if name == "AREAS":
        return area_names()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def validate(name: str, df: pd.DataFrame, sample: bool = False) -> None:
    """Raise ValueError if df does not match the agreed contract.

    ``sample=True`` checks areas against the sample group list, for sample files.
    """
    expected = CONTRACTS[name][1]
    if list(df.columns) != expected:
        raise ValueError(f"{name}.csv columns {list(df.columns)} != contract {expected}")
    if name == "groups":
        names = df["name"]
        if names.isna().any() or (names.astype(str).str.strip() == "").any():
            raise ValueError("groups.csv has an empty group name")
        if df["group_id"].duplicated().any() or names.str.casefold().duplicated().any():
            raise ValueError("groups.csv group ids and names must be unique")
    if "area" in df.columns:
        bad = set(df["area"].dropna()) - set(area_names(sample))
        if bad:
            raise ValueError(f"{name}.csv has areas not in the shared list: {sorted(bad)}")
    if "lifecycle_stage" in df.columns:
        bad = set(df["lifecycle_stage"].dropna()) - set(LIFECYCLE_STAGES)
        if bad:
            raise ValueError(f"{name}.csv has unknown lifecycle stages: {sorted(bad)}")


def load(name: str, prefer_real: bool = True) -> pd.DataFrame:
    """Load data/<name>.csv if it exists, otherwise data/sample/<name>.csv."""
    real = path_for(name)
    use_real = prefer_real and real.exists()
    df = pd.read_csv(real if use_real else path_for(name, sample=True))
    validate(name, df, sample=not use_real)
    return df


def save(name: str, df: pd.DataFrame) -> Path:
    """Validate then write data/<name>.csv. Use this for every contract output."""
    validate(name, df)
    out = path_for(name)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
