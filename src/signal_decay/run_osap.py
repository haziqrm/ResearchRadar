"""Run the OSAP extension (HYPOTHESES.md, H4-H6). Writes A-owned files only.

Usage:
    python -m src.signal_decay.run_osap [--no-citations] [--charts] [--refresh]

Writes:
    data/osap_decay_results.csv  one row per signal: periods, means, decay, Newey-West CI
    data/osap_papers.csv         one row per paper: OpenAlex match, attention, mean decay
    data/osap_summary.csv        H4 (McLean-Pontiff regression), H5 (placebo), H6 (attention)
    data/cache/charts/osap_*.png with --charts

None of the shared contract files (factors, papers, decay_results) are touched.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.signal_decay import CHARTS_DIR, DATA_DIR
from src.signal_decay.attention import attention_decay_test, era_adjusted_attention
from src.signal_decay.decay import before_after_test
from src.signal_decay.osap import (
    load_ls_returns, load_signal_doc, match_paper, mp_regression, paper_table, period_panel,
    predictor_table, publication_start, signal_period_stats, within_year_percentile,
)
from src.signal_decay.robustness import pooled_placebo_test

OSAP_RESULTS_CSV = DATA_DIR / "osap_decay_results.csv"
OSAP_PAPERS_CSV = DATA_DIR / "osap_papers.csv"
OSAP_SUMMARY_CSV = DATA_DIR / "osap_summary.csv"
OSAP_PEER_N = 30  # HYPOTHESES.md, H6

SUMMARY_COLUMNS = ["test", "statistic", "value", "ci_low", "ci_high", "p_value", "n", "note"]


def signal_results(returns: pd.DataFrame, papers: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Per-signal period stats plus a Newey-West CI for POST minus IS."""
    st = signal_period_stats(panel)
    rows = []
    for _, s in st.iterrows():
        sub = panel[(panel["signal"] == s["signal"]) & panel["period"].isin(["IS", "POST"])]
        series = sub.set_index("date")["return"]
        pub = publication_start(papers.loc[papers["signal"] == s["signal"], "year"].iloc[0])
        if s["n_is"] >= 2 and s["n_post"] >= 2:
            r = before_after_test(series, pub, min_months=1)
            rows.append({"signal": s["signal"], "post_minus_is": r["diff"],
                         "ci_low": r["diff_ci_low"], "ci_high": r["diff_ci_high"], "p_value": r["p_value"]})
        else:
            rows.append({"signal": s["signal"]})
    out = papers.merge(st, on="signal", how="left").merge(pd.DataFrame(rows), on="signal", how="left")
    return out


def attention_for_papers(ptable: pd.DataFrame, citations: bool) -> pd.DataFrame:
    """Add OpenAlex match, early citations and peer percentile to the paper table."""
    cols = ["openalex_id", "match_status", "match_title", "early_citations", "citation_peer_n",
            "citation_percentile"]
    for c in cols:
        ptable[c] = None
    if not citations:
        return ptable
    for i, r in ptable.iterrows():
        try:
            m = match_paper(r["authors"], int(r["year"]), r["journal"])
            ptable.loc[i, ["openalex_id", "match_status", "match_title"]] = (
                m["openalex_id"], m["match_status"], m["match_title"])
            if m["match_status"] == "matched":
                att = era_adjusted_attention(m["openalex_id"], int(r["year"]), peer_n=OSAP_PEER_N)
                ptable.loc[i, ["early_citations", "citation_peer_n", "citation_percentile"]] = (
                    att["early_citations"], att["peer_n"], att["percentile"])
        except RuntimeError as exc:
            print(f"Stopping citation lookups: {exc}")
            ptable.loc[i:, "match_status"] = ptable.loc[i:, "match_status"].fillna("not_attempted")
            break
        if (i + 1) % 20 == 0:
            print(f"  attention: {i + 1}/{len(ptable)} papers")
    return ptable


def build_summary(h4: dict, h5: dict, h6: dict, h6b: dict, ptable: pd.DataFrame) -> pd.DataFrame:
    status = ptable["match_status"].value_counts().to_dict()
    return pd.DataFrame([
        {"test": "H4 McLean-Pontiff", "statistic": "b_oos (share of IS mean)", "value": h4["b_oos"],
         "ci_low": h4["b_oos_ci"][0], "ci_high": h4["b_oos_ci"][1], "p_value": None, "n": h4["n_signals"],
         "note": f"{h4['n_obs']} signal-months; {h4['n_excluded_nonpositive_is']} signals excluded (IS mean <= 0)"},
        {"test": "H4 McLean-Pontiff", "statistic": "b_post (share of IS mean)", "value": h4["b_post"],
         "ci_low": h4["b_post_ci"][0], "ci_high": h4["b_post_ci"][1], "p_value": None, "n": h4["n_signals"],
         "note": "SE clustered by month"},
        {"test": "H4 McLean-Pontiff", "statistic": "b_post - b_oos", "value": h4["post_minus_oos"],
         "ci_low": h4["post_minus_oos"] - 1.96 * h4["post_minus_oos_se"],
         "ci_high": h4["post_minus_oos"] + 1.96 * h4["post_minus_oos_se"],
         "p_value": h4["post_minus_oos_p_one_sided"], "n": h4["n_signals"],
         "note": "one-sided: extra drop at publication beyond the post-sample drop"},
        {"test": "H5 pooled placebo", "statistic": "mean post-pre monthly return", "value": h5["mean_real_diff"],
         "ci_low": None, "ci_high": None, "p_value": h5["p_one_sided"], "n": h5["n_signals"],
         "note": f"mean fake diff {h5.get('mean_fake_diff', np.nan):.5f}; >=120 months each side"},
        {"test": "H6 attention vs decay", "statistic": "Spearman rho (early citation peer percentile)",
         "value": h6["spearman"], "ci_low": None, "ci_high": None, "p_value": h6["p_one_sided"], "n": h6["n"],
         "note": f"paper level; OpenAlex match status {status}"},
        {"test": "H6 secondary", "statistic": "Spearman rho (Google Scholar cites, within-year percentile)",
         "value": h6b["spearman"], "ci_low": None, "ci_high": None, "p_value": h6b["p_one_sided"], "n": h6b["n"],
         "note": "total citations to Sept 2025, can reflect fame after publication"},
    ], columns=SUMMARY_COLUMNS)


def plot_three_periods(h4: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals = [1.0, 1 + h4["b_oos"], 1 + h4["b_post"]]
    lo = [0, h4["b_oos"] - h4["b_oos_ci"][0], h4["b_post"] - h4["b_post_ci"][0]]
    hi = [0, h4["b_oos_ci"][1] - h4["b_oos"], h4["b_post_ci"][1] - h4["b_post"]]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(["In-sample\n(paper's data)", "After sample,\nbefore publication", "After\npublication"],
           [100 * v for v in vals], yerr=[[100 * x for x in lo], [100 * x for x in hi]],
           color=["#4C72B0", "#8172B3", "#DD8452"], capsize=6)
    for i, v in enumerate(vals):
        ax.text(i, 100 * v + 4, f"{100 * v:.0f}%", ha="center")
    ax.set_ylabel("Average return, % of in-sample level")
    ax.set_title(f"{h4['n_signals']} published signals (OSAP): returns by period\n95% CI, clustered by month")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_attention(ptable: pd.DataFrame, h6: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = ptable.dropna(subset=["citation_percentile", "relative_decay"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(d["citation_percentile"].astype(float), 100 * d["relative_decay"].astype(float), s=18, alpha=0.7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylim(-300, 300)
    ax.set_xlabel("Early citations, percentile among same journal and year (%)")
    ax.set_ylabel("Share of in-sample edge lost after publication (%)")
    ax.set_title(f"Attention vs decay, {h6['n']} papers: Spearman {h6['spearman']:.2f}, "
                 f"one-sided p = {h6['p_one_sided']:.2f}\n(y axis clipped at ±300%)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the OSAP decay extension.")
    parser.add_argument("--no-citations", action="store_true", help="skip OpenAlex matching and citations")
    parser.add_argument("--charts", action="store_true", help="save charts to data/cache/charts")
    parser.add_argument("--refresh", action="store_true", help="re-download OSAP files")
    args = parser.parse_args(argv)

    papers = predictor_table(load_signal_doc(refresh=args.refresh))
    returns = load_ls_returns(refresh=args.refresh)
    returns = returns[returns["signal"].isin(papers["signal"])]
    panel = period_panel(returns, papers)
    print(f"{papers['signal'].nunique()} signals, {len(panel)} signal-months in the three periods")

    results = signal_results(returns, papers, panel)
    results.to_csv(OSAP_RESULTS_CSV, index=False)
    print(f"Wrote {len(results)} rows to {OSAP_RESULTS_CSV}")

    h4 = mp_regression(panel)
    series = {s: g.set_index("date")["return"] for s, g in returns.groupby("signal")}
    dates = {s: publication_start(y) for s, y in zip(papers["signal"], papers["year"])}
    h5 = pooled_placebo_test(series, dates)

    ptable = paper_table(papers, results)
    ptable = attention_for_papers(ptable, citations=not args.no_citations)
    ptable["gs_cites_percentile"] = within_year_percentile(ptable["gs_cites"], ptable["year"])
    ptable.to_csv(OSAP_PAPERS_CSV, index=False)
    print(f"Wrote {len(ptable)} rows to {OSAP_PAPERS_CSV}")

    decay = ptable["relative_decay"].astype(float)
    h6 = attention_decay_test(pd.to_numeric(ptable["citation_percentile"], errors="coerce"), decay)
    h6b = attention_decay_test(ptable["gs_cites_percentile"].astype(float), decay)
    summary = build_summary(h4, h5, h6, h6b, ptable)
    summary.to_csv(OSAP_SUMMARY_CSV, index=False)
    print(f"Wrote {len(summary)} rows to {OSAP_SUMMARY_CSV}")

    if args.charts:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        plot_three_periods(h4, CHARTS_DIR / "osap_three_periods.png")
        if h6["n"] >= 3:
            plot_attention(ptable, h6, CHARTS_DIR / "osap_attention_vs_decay.png")
        print(f"Saved charts to {CHARTS_DIR}")


if __name__ == "__main__":
    main()
