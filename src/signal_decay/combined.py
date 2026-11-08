"""Combined outputs for the app: one decay table, OSAP OpenAlex IDs, and an event-time curve.

This reorganises existing measurements (H1-H6); it is not a new test and changes no
H1-H10 code or result file. Run it after run_decay (main.py stage "a2" does this).

Usage:
    python -m src.signal_decay.combined [--charts] [--no-curve]

Writes:
    data/decay_results.csv     contract: 7 Ken French + 212 OSAP rows, key = (source, signal)
    data/osap_paper_ids.csv    signal, openalex_id for OSAP signals whose paper matched (H6)
    data/decay_curve.csv       descriptive event-time curve (OSAP), needs the OSAP returns
    data/decay_curve_robust.csv  less noisy version: ratio of averages, 5-year buckets

Inputs (all A-owned): data/papers.csv, data/decay_details.csv (from run_decay),
data/osap_decay_results.csv and data/osap_papers.csv (from run_osap).

Conventions:
- share_kept = post_mean / pre_mean when pre_mean > 0, else blank. Not clipped, so it can be
  below 0 (the edge reversed) or above 1 (it grew).
- Ken French: pre_mean / post_mean are the means before / after publication_date (issue month);
  ci_low / ci_high are the Newey-West 95% CI of post - pre; half_life_months only where the
  HYPOTHESES.md rule allows (none at present).
- OSAP: pre_mean is the in-sample mean; ci is the Newey-West 95% CI of post - in-sample;
  half_life_months is not computed. OSAP gives the publication year only, so
  publication_date is 1 January of the following year: the first post-publication month
  under the existing rule (a month is post if its month-end is on or after the date).
- early_citations: raw citing works in years Y+1 to Y+3 (blank if the paper is unmatched).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.common import contracts
from src.signal_decay import CHARTS_DIR, DATA_DIR, DECAY_DETAILS_CSV

OSAP_RESULTS_CSV = DATA_DIR / "osap_decay_results.csv"
OSAP_PAPERS_CSV = DATA_DIR / "osap_papers.csv"
OSAP_PAPER_IDS_CSV = DATA_DIR / "osap_paper_ids.csv"
CURVE_CSV = DATA_DIR / "decay_curve.csv"
ROBUST_CURVE_CSV = DATA_DIR / "decay_curve_robust.csv"
ROBUST_BUCKET_YEARS = 5
ROBUST_MIN_MONTHS = 24  # months a signal needs inside a 5-year bucket to count in it

CURVE_FIRST_YEAR = -10  # buckets of 12 months; bucket 0 = first 12 months after publication
CURVE_LAST_YEAR = 19  # i.e. up to 20 years after publication
CURVE_MIN_SIGNALS = 50
CURVE_MIN_MONTHS = 6  # months a signal needs inside a bucket to count in it
BOOTSTRAP_DRAWS = 2000
SEED = 0


def share_kept(pre: pd.Series, post: pd.Series) -> pd.Series:
    pre, post = pd.to_numeric(pre, errors="coerce"), pd.to_numeric(post, errors="coerce")
    return (post / pre).where(pre > 0)


def _paper_key(df: pd.DataFrame) -> pd.Series:
    return df["authors"].astype(str) + "|" + df["year"].astype(int).astype(str) + "|" + df["journal"].astype(str)


def kf_rows(papers: pd.DataFrame, details: pd.DataFrame) -> pd.DataFrame:
    d = details.set_index("signal")
    rows = []
    for p in papers.itertuples(index=False):
        r = d.loc[p.signal] if p.signal in d.index else None
        ok = r is not None and r["status"] == "ok"
        reportable = ok and str(r["half_life_reportable"]).lower() == "true"
        rows.append({
            "signal": p.signal, "paper": p.paper, "publication_date": p.publication_date, "source": "ken_french",
            "pre_mean": r["pre_mean"] if ok else np.nan, "post_mean": r["post_mean"] if ok else np.nan,
            "half_life_months": r["half_life_months"] if reportable else np.nan,
            "ci_low": r["diff_ci_low"] if ok else np.nan, "ci_high": r["diff_ci_high"] if ok else np.nan,
            "early_citations": r["early_citations"] if r is not None else np.nan,
        })
    return pd.DataFrame(rows)


def osap_rows(osap: pd.DataFrame, osap_papers: pd.DataFrame) -> pd.DataFrame:
    cites = osap_papers.assign(key=_paper_key(osap_papers)).set_index("key")["early_citations"]
    return pd.DataFrame({
        "signal": osap["signal"],
        "paper": osap["authors"] + " (" + osap["year"].astype(str) + "), " + osap["journal"],
        "publication_date": (osap["year"].astype(int) + 1).astype(str) + "-01-01",
        "source": "osap",
        "pre_mean": osap["is_mean"], "post_mean": osap["post_mean"], "half_life_months": np.nan,
        "ci_low": osap["ci_low"], "ci_high": osap["ci_high"],
        "early_citations": _paper_key(osap).map(cites),
    })


def build_decay_results(papers: pd.DataFrame, details: pd.DataFrame, osap: pd.DataFrame,
                        osap_papers: pd.DataFrame) -> pd.DataFrame:
    out = pd.concat([kf_rows(papers, details), osap_rows(osap, osap_papers)], ignore_index=True)
    out["share_kept"] = share_kept(out["pre_mean"], out["post_mean"])
    out["early_citations"] = pd.to_numeric(out["early_citations"], errors="coerce").astype("Int64")
    out = out[contracts.CONTRACTS["decay_results"][1]]
    if out.duplicated(["source", "signal"]).any() or out["signal"].isna().any():
        raise ValueError("decay_results needs a unique, non-blank (source, signal) key")
    return out


def build_osap_paper_ids(osap_papers: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """One row per OSAP signal whose paper matched in H6, plus counts of papers by match status."""
    counts = osap_papers["match_status"].value_counts().to_dict()
    m = osap_papers[osap_papers["match_status"] == "matched"].copy()
    m["signal"] = m["signals"].str.split(";")
    ids = m.explode("signal")[["signal", "openalex_id"]].sort_values("signal").reset_index(drop=True)
    if ids["signal"].duplicated().any():
        raise ValueError("an OSAP signal maps to more than one OpenAlex ID")
    return ids, counts


# ---------- descriptive event-time curve ----------

def event_panel(returns: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    """Monthly returns / in-sample mean, with months since publication (0 = January of Y+1).

    info: signal, year (publication), sample_start, is_mean (> 0). Months before the
    sample start are dropped, as in the three-period tests.
    """
    df = returns.assign(date=pd.to_datetime(returns["date"])).merge(info, on="signal")
    df = df[df["date"].dt.year >= df["sample_start"]]
    df["k"] = (df["date"].dt.year - (df["year"] + 1)) * 12 + df["date"].dt.month - 1
    df["bucket"] = np.floor_divide(df["k"], 12)
    df["scaled"] = df["return"] / df["is_mean"]
    return df


def decay_curve(panel: pd.DataFrame, first: int = CURVE_FIRST_YEAR, last: int = CURVE_LAST_YEAR,
                min_signals: int = CURVE_MIN_SIGNALS, min_months: int = CURVE_MIN_MONTHS,
                draws: int = BOOTSTRAP_DRAWS, seed: int = SEED) -> pd.DataFrame:
    """Mean scaled return by 12-month bucket, signals weighted equally, bootstrap over signals."""
    p = panel[panel["bucket"].between(first, last)]
    g = p.groupby(["signal", "bucket"])["scaled"].agg(["mean", "count"])
    per_signal = g["mean"].where(g["count"] >= min_months).unstack("bucket")
    per_signal = per_signal.reindex(columns=range(first, last + 1))
    M = per_signal.notna().to_numpy(float)
    V = per_signal.fillna(0.0).to_numpy()
    n = M.sum(axis=0)
    est = np.divide(V.sum(axis=0), n, out=np.full(n.shape, np.nan), where=n > 0)

    rng = np.random.default_rng(seed)
    S = V.shape[0]
    boot = np.empty((draws, V.shape[1]))
    for i in range(draws):
        w = np.bincount(rng.integers(0, S, S), minlength=S).astype(float)
        den = w @ M
        boot[i] = np.divide(w @ V, den, out=np.full(den.shape, np.nan), where=den > 0)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    out = pd.DataFrame({"years_since_publication": per_signal.columns, "n_signals": n.astype(int),
                        "mean_scaled_return": est, "ci_low": lo, "ci_high": hi})
    out["period"] = np.where(out["years_since_publication"] >= 0, "post-publication", "pre-publication")
    return out[out["n_signals"] >= min_signals].reset_index(drop=True)


def robust_curve(panel: pd.DataFrame, bucket_years: int = ROBUST_BUCKET_YEARS,
                 first: int = CURVE_FIRST_YEAR, last: int = CURVE_LAST_YEAR,
                 min_signals: int = CURVE_MIN_SIGNALS, min_months: int = ROBUST_MIN_MONTHS,
                 draws: int = BOOTSTRAP_DRAWS, seed: int = SEED) -> pd.DataFrame:
    """Ratio of averages by multi-year bucket: mean return across signals / mean in-sample mean.

    Dividing averages rather than averaging ratios stops signals with a tiny in-sample mean
    from dominating. Signals in a bucket are weighted equally; bootstrap resamples signals.
    Descriptive only; eras overlap across signals.
    """
    p = panel[panel["bucket"].between(first, last)].copy()
    p["rbucket"] = np.floor_divide(p["bucket"], bucket_years) * bucket_years
    g = p.groupby(["signal", "rbucket"]).agg(ret=("return", "mean"), n=("return", "size"), is_mean=("is_mean", "first"))
    g = g[g["n"] >= min_months]
    R = g["ret"].unstack("rbucket")
    cols = sorted(R.columns)
    R = R[cols]
    Mask = R.notna().to_numpy(float)
    Rv = R.fillna(0.0).to_numpy()
    I = g["is_mean"].groupby(level="signal").first().reindex(R.index).to_numpy()[:, None] * Mask

    def ratio(w):
        num, den = w @ Rv, w @ I
        return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)

    n = Mask.sum(axis=0)
    est = ratio(np.ones(len(R)))
    rng = np.random.default_rng(seed)
    S = len(R)
    boot = np.array([ratio(np.bincount(rng.integers(0, S, S), minlength=S).astype(float)) for _ in range(draws)])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    out = pd.DataFrame({"years_from": cols, "years_to": [c + bucket_years for c in cols], "n_signals": n.astype(int),
                        "share_of_in_sample_mean": est, "ci_low": lo, "ci_high": hi})
    out["period"] = np.where(out["years_from"] >= 0, "post-publication", "pre-publication")
    return out[out["n_signals"] >= min_signals].reset_index(drop=True)


def plot_curve(curve: pd.DataFrame, path, robust: pd.DataFrame | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = curve["years_since_publication"] + 0.5
    ax.plot(x, curve["mean_scaled_return"], marker="o", ms=3, lw=0.8, alpha=0.35, color="grey",
            label="yearly: mean of (return / in-sample mean), noisy")
    if robust is not None:
        xr = (robust["years_from"] + robust["years_to"]) / 2
        ax.fill_between(xr, robust["ci_low"], robust["ci_high"], alpha=0.25, color="C0",
                        label="95% bootstrap interval (signals)")
        ax.plot(xr, robust["share_of_in_sample_mean"], marker="o", color="C0", lw=2,
                label="5-year: mean return / mean in-sample mean")
    ax.axvline(0, color="black", lw=1, label="publication (January after the publication year)")
    ax.axhline(1, color="grey", ls="--", lw=0.8)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylim(-1, 2.5)
    ax.set_xlabel("Years since publication")
    ax.set_ylabel("Return as a share of the in-sample mean")
    ax.set_title(f"Event-time decay, OSAP signals (descriptive; eras overlap; at least {CURVE_MIN_SIGNALS} "
                 "signals per bucket)\nPre-publication buckets partly overlap the in-sample period, "
                 "which averages 1 by construction")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Combined decay table, OSAP IDs and event-time curve.")
    parser.add_argument("--charts", action="store_true")
    parser.add_argument("--no-curve", action="store_true", help="skip the curve (it needs the OSAP returns)")
    args = parser.parse_args(argv)

    papers = pd.read_csv(DATA_DIR / "papers.csv")
    details = pd.read_csv(DECAY_DETAILS_CSV)
    osap, osap_papers = pd.read_csv(OSAP_RESULTS_CSV), pd.read_csv(OSAP_PAPERS_CSV)
    results = build_decay_results(papers, details, osap, osap_papers)
    print(f"Wrote {len(results)} rows to {contracts.save('decay_results', results)}")
    ids, counts = build_osap_paper_ids(osap_papers)
    ids.to_csv(OSAP_PAPER_IDS_CSV, index=False)
    print(f"Wrote {len(ids)} OSAP signal IDs to {OSAP_PAPER_IDS_CSV}; papers by match status: {counts}")
    if args.no_curve:
        return

    from src.signal_decay.osap import load_ls_returns

    info = osap[(osap["is_mean"] > 0) & (osap["n_post"].fillna(0) > 0)][["signal", "year", "sample_start", "is_mean"]]
    curve = decay_curve(event_panel(load_ls_returns(), info))
    curve.to_csv(CURVE_CSV, index=False)
    print(f"Wrote {len(curve)} buckets to {CURVE_CSV}")
    robust = robust_curve(event_panel(load_ls_returns(), info))
    robust.to_csv(ROBUST_CURVE_CSV, index=False)
    print(f"Wrote {len(robust)} buckets to {ROBUST_CURVE_CSV}")
    if args.charts:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        plot_curve(curve, CHARTS_DIR / "decay_curve.png", robust)
        print(f"Saved chart to {CHARTS_DIR / 'decay_curve.png'}")


if __name__ == "__main__":
    main()
