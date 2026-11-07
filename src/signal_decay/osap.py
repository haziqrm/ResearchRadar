"""Open Source Asset Pricing (Chen and Zimmermann): 212 published signals.

Portfolio returns only (original-paper long-short portfolios), no firm-level
characteristics and no WRDS. Tests and rules are fixed in HYPOTHESES.md (H4-H6).

Downloads are cached in data/cache/osap (gitignored). The openassetpricing
package is imported inside functions because building OpenAP() downloads
metadata from Hugging Face.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from src.signal_decay import CACHE_DIR
from src.signal_decay.attention import _get_json

OSAP_CACHE_DIR = CACHE_DIR / "osap"
OP_PORTFOLIO = "op"  # portfolios built as in the original papers

PERIODS = ("IS", "OOS", "POST")

# OSAP journal abbreviation -> OpenAlex source name. WP, Book and Other have no journal.
JOURNALS = {
    "JF": "The Journal of Finance",
    "JFE": "Journal of Financial Economics",
    "AR": "The Accounting Review",
    "JAE": "Journal of Accounting and Economics",
    "RFS": "Review of Financial Studies",
    "RAS": "Review of Accounting Studies",
    "JFQA": "Journal of Financial and Quantitative Analysis",
    "JAR": "Journal of Accounting Research",
    "MS": "Management Science",
    "JPE": "Journal of Political Economy",
    "FAJ": "Financial Analysts Journal",
    "ROF": "Review of Finance",
    "JFM": "Journal of Financial Markets",
    "RFQA": "Review of Quantitative Finance and Accounting",
    "BAR": "The British Accounting Review",
    "JFR": "Journal of Financial Research",
    "JOIM": "Journal of Investment Management",
    "QJE": "The Quarterly Journal of Economics",
    "JBFA": "Journal of Business Finance & Accounting",
    "JEmpFin": "Journal of Empirical Finance",
    "RED": "Review of Economic Dynamics",
    "CAR": "Contemporary Accounting Research",
}


# ---------- download (network only inside these functions) ----------

def _client():
    """Create an OpenAP client (network access: fetches the release file index)."""
    from openassetpricing import OpenAP

    return OpenAP()


def load_signal_doc(cache_dir: Path = OSAP_CACHE_DIR, refresh: bool = False) -> pd.DataFrame:
    """OSAP signal documentation (one row per signal), cached as parquet."""
    path = cache_dir / "signal_doc.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    doc = _client().dl_signal_doc("pandas")
    doc.to_parquet(path)
    return doc


def load_ls_returns(cache_dir: Path = OSAP_CACHE_DIR, refresh: bool = False) -> pd.DataFrame:
    """Long-short monthly returns as tidy (date, signal, return), decimals, month-end dates."""
    path = cache_dir / "op_ports.parquet"
    if path.exists() and not refresh:
        ports = pd.read_parquet(path)
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        ports = _client().dl_port(OP_PORTFOLIO, "pandas")
        ports.to_parquet(path)
    ls = ports[ports["port"].astype(str) == "LS"]
    out = pd.DataFrame({
        "date": pd.to_datetime(ls["date"]) + pd.offsets.MonthEnd(0),
        "signal": ls["signalname"].astype(str),
        "return": ls["ret"].astype(float) / 100.0,
    }).dropna()
    return out.sort_values(["signal", "date"]).reset_index(drop=True)


def predictor_table(doc: pd.DataFrame) -> pd.DataFrame:
    """One row per predictor signal with the fields the tests need."""
    p = doc[doc["Cat.Signal"] == "Predictor"]
    return pd.DataFrame({
        "signal": p["Acronym"].astype(str),
        "authors": p["Authors"].astype(str),
        "year": p["Year"].astype(int),
        "journal": p["Journal"].astype(str),
        "sample_start": p["SampleStartYear"].astype(int),
        "sample_end": p["SampleEndYear"].astype(int),
        "gs_cites": pd.to_numeric(p["GScholarCites202509"], errors="coerce"),
    }).reset_index(drop=True)


def publication_start(year: int) -> pd.Timestamp:
    """First post-publication month under the 'end of publication year' rule."""
    return pd.Timestamp(f"{int(year) + 1}-01-01")


# ---------- three periods ----------

def period_panel(returns: pd.DataFrame, papers: pd.DataFrame) -> pd.DataFrame:
    """Returns joined with each signal's period label; rows before the sample start dropped."""
    df = returns.merge(papers[["signal", "sample_start", "sample_end", "year"]], on="signal")
    y = df["date"].dt.year
    df["period"] = np.select(
        [y < df["sample_start"], y <= df["sample_end"], y <= df["year"]],
        [None, "IS", "OOS"], default="POST",
    )
    return df[df["period"].notna()].drop(columns=["sample_start", "sample_end", "year"])


def signal_period_stats(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-signal n and mean in each period, plus relative decay (1 - POST/IS)."""
    g = panel.groupby(["signal", "period"])["return"].agg(["count", "mean"]).unstack("period")
    out = pd.DataFrame(index=g.index)
    for p in PERIODS:
        out[f"n_{p.lower()}"] = g.get(("count", p), pd.Series(0, index=g.index)).fillna(0).astype(int)
        out[f"{p.lower()}_mean"] = g.get(("mean", p))
    is_pos = out["is_mean"] > 0
    out["relative_decay"] = np.where(is_pos & (out["n_post"] > 0), 1 - out["post_mean"] / out["is_mean"], np.nan)
    out["relative_decay_oos"] = np.where(is_pos & (out["n_oos"] > 0), 1 - out["oos_mean"] / out["is_mean"], np.nan)
    return out.reset_index()


def mp_regression(panel: pd.DataFrame) -> dict:
    """H4: pooled OLS of IS-scaled returns on OOS and POST dummies, SE clustered by month.

    Signals with IS mean <= 0 are excluded. Coefficients are the average change
    as a share of the in-sample mean (McLean and Pontiff found about -0.26, -0.58).
    """
    is_mean = panel[panel["period"] == "IS"].groupby("signal")["return"].mean()
    keep = is_mean[is_mean > 0].index
    df = panel[panel["signal"].isin(keep)].copy()
    df["scaled"] = df["return"] / df["signal"].map(is_mean)
    X = pd.DataFrame({
        "const": 1.0,
        "OOS": (df["period"] == "OOS").astype(float),
        "POST": (df["period"] == "POST").astype(float),
    }, index=df.index)
    groups = pd.factorize(df["date"])[0]
    fit = sm.OLS(df["scaled"], X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    ci = fit.conf_int()
    cov = fit.cov_params()
    diff = fit.params["POST"] - fit.params["OOS"]
    diff_se = float(np.sqrt(cov.loc["POST", "POST"] + cov.loc["OOS", "OOS"] - 2 * cov.loc["POST", "OOS"]))
    z = diff / diff_se
    return {
        "n_signals": int(len(keep)),
        "n_excluded_nonpositive_is": int((is_mean <= 0).sum()),
        "n_obs": int(len(df)),
        "b_oos": float(fit.params["OOS"]), "b_oos_ci": tuple(ci.loc["OOS"]),
        "b_post": float(fit.params["POST"]), "b_post_ci": tuple(ci.loc["POST"]),
        "post_minus_oos": float(diff), "post_minus_oos_se": diff_se,
        "post_minus_oos_p_one_sided": float(stats.norm.cdf(z)),
    }


# ---------- matching papers to OpenAlex ----------

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def parse_surnames(authors: str) -> list[str]:
    """'Chan, Jegadeesh and Lakonishok' -> ['Chan', 'Jegadeesh', 'Lakonishok']; drops 'et al.'."""
    a = re.sub(r"\bet\.? al\.?", "", authors)
    parts = re.split(r",|\band\b|&", a)
    return [p.strip() for p in parts if p.strip()]


def surname_matches(surname: str, full_name: str) -> bool:
    """True if surname equals the last 1, 2 or 3 name tokens (handles 'De Bondt', 'Novy-Marx')."""
    target = _norm(surname)
    tokens = [t for t in re.split(r"[\s.]+", full_name.replace("-", "")) if t]
    tokens = [t for t in tokens if _norm(t) not in {"jr", "sr", "ii", "iii", "iv"}]
    return any(_norm("".join(tokens[-k:])) == target for k in (1, 2, 3) if len(tokens) >= k)


def resolve_source_id(journal: str, **kwargs) -> str | None:
    """OpenAlex source ID for an OSAP journal abbreviation (None for WP/Book/Other)."""
    name = JOURNALS.get(journal)
    if name is None:
        return None
    res = _get_json("sources", {"search": name, "per_page": 5, "select": "id,display_name"},
                    f"source_{journal}", **kwargs)
    want = _norm(name.removeprefix("The "))
    for s in res.get("results", []):
        if _norm(s["display_name"].removeprefix("The ")) == want:
            return s["id"].rsplit("/", 1)[-1]
    return None


def match_paper(authors: str, year: int, journal: str, **kwargs) -> dict:
    """Find the OpenAlex work for an OSAP paper (rule in HYPOTHESES.md, H6)."""
    out = {"openalex_id": None, "match_status": None, "match_title": None, "n_candidates": 0}
    source = resolve_source_id(journal, **kwargs)
    if source is None:
        out["match_status"] = "no_journal"
        return out
    surnames = parse_surnames(authors)
    first = surnames[0]
    res = _get_json(
        "works",
        {"filter": f"primary_location.source.id:{source},publication_year:{year - 1}-{year},"
                   f"raw_author_name.search:{first}",
         "per_page": 50, "select": "id,display_name,publication_year,authorships"},
        f"match_{journal}_{year}_{_norm(first)}",
        **kwargs,
    )
    cands = []
    for w in res.get("results", []):
        names = [a["author"]["display_name"] for a in w["authorships"]]
        names += [a.get("raw_author_name") or "" for a in w["authorships"]]
        if all(any(surname_matches(s, n) for n in names if n) for s in surnames):
            cands.append(w)
    out["n_candidates"] = len(cands)
    if len(cands) == 1:
        out.update(openalex_id=cands[0]["id"].rsplit("/", 1)[-1], match_status="matched",
                   match_title=cands[0]["display_name"])
    else:
        out["match_status"] = "not_found" if not cands else "ambiguous"
    return out


def paper_table(papers: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    """One row per paper: its signals and mean relative decay across them."""
    df = papers.merge(stats_df[["signal", "relative_decay"]], on="signal", how="left")
    g = df.groupby(["authors", "year", "journal"], sort=True)
    return g.agg(
        signals=("signal", lambda s: ";".join(sorted(s))),
        n_signals=("signal", "size"),
        gs_cites=("gs_cites", "max"),
        relative_decay=("relative_decay", "mean"),
    ).reset_index()


def within_year_percentile(values: pd.Series, years: pd.Series) -> pd.Series:
    """Percentile of each value among *other* papers from the same year (ties count half)."""
    out = pd.Series(np.nan, index=values.index)
    for _, idx in years.groupby(years).groups.items():
        for i in idx:
            others = values.loc[[j for j in idx if j != i]].dropna()
            if len(others) and pd.notna(values.loc[i]):
                out.loc[i] = 100 * ((others < values.loc[i]).sum() + 0.5 * (others == values.loc[i]).sum()) / len(others)
    return out
