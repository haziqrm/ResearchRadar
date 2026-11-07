"""Open Source Asset Pricing (Chen and Zimmermann) portfolio returns.

Portfolio returns only: no firm-level characteristics and no WRDS. The
openassetpricing package is imported inside functions because building
OpenAP() downloads metadata from Hugging Face.

TODO: cache downloads in data/cache, convert to the tidy (date, signal, return)
format used by data/factors.csv, and pick which portfolio leg/spread to use.
"""

from __future__ import annotations

import pandas as pd

# "op" = portfolios built as in the original papers. Run list_portfolio_files()
# to see other options for the current release.
DEFAULT_PORTFOLIO = "op"


def _client():
    """Create an OpenAP client (network access: fetches the release file index)."""
    from openassetpricing import OpenAP

    return OpenAP()


def list_portfolio_files() -> None:
    """Print the portfolio files available in the current OSAP release."""
    _client().list_port()


def list_predictors() -> pd.DataFrame:
    """Return the OSAP signal documentation (one row per predictor, column Acronym)."""
    return _client().dl_signal_doc("pandas")


def download_portfolio_returns(
    predictors: list[str] | None = None, portfolio: str = DEFAULT_PORTFOLIO
) -> pd.DataFrame:
    """Download OSAP portfolio returns for the given predictors (all if None).

    Returns the package's raw frame (columns include signalname, port, date, ret).
    TODO: reshape to date, signal, return and decide units (OSAP returns are in percent).
    """
    return _client().dl_port(portfolio, "pandas", predictor=predictors)
