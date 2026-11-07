"""Signal-decay module (Person A): how much published signals' returns fall after publication.

Contract files (factors, papers, decay_results) go through src.common.contracts.
The paths below are for files this module owns that are not shared contracts.
Importing this package does no network access and needs no API keys.
"""

from src.common.contracts import DATA_DIR, ROOT

ROOT_DIR = ROOT
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
OPENALEX_CACHE_DIR = CACHE_DIR / "openalex"
CHARTS_DIR = CACHE_DIR / "charts"

# Owned by A, not part of the shared contracts.
DECAY_DETAILS_CSV = DATA_DIR / "decay_details.csv"
PAPER_IDS_CSV = DATA_DIR / "paper_ids.csv"
