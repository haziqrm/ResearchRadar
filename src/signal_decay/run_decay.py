"""Run the signal-decay chain. Tests and rules are fixed in HYPOTHESES.md.

Usage:
    python -m src.signal_decay.run_decay [--refresh] [--allow-unverified]
                                         [--no-citations] [--charts]
                                         [--bootstrap-draws N]

Writes:
    data/factors.csv              contract: date, signal, return (Mkt-RF is a benchmark only)
    data/decay_results.csv        contract; only written when at least one paper is included
    data/decay_details.csv        A-owned: extra stats, intervals, placebo, attention, status
    data/signal_decay_summary.csv A-owned: pooled placebo (H2) and attention-vs-decay (H3)
    data/cache/charts/            per-signal PNGs and attention_vs_decay.png with --charts

decay_results.csv columns:
    pre_mean, post_mean   mean monthly return before / after publication
    ci_low, ci_high       95% CI of (post_mean - pre_mean), Newey-West, 12 lags
    half_life_months      only when the bootstrap rule in HYPOTHESES.md passes, else empty
    early_citations       raw count of works citing the paper in years 1-3 after publication
                          (era-adjusted percentile is in decay_details.csv)
Rows below 120 months on either side keep their row with stats left empty;
the reason is in decay_details.csv (status column).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.common import contracts
from src.signal_decay import DATA_DIR, DECAY_DETAILS_CSV
from src.signal_decay.attention import attention_decay_test, era_adjusted_attention, load_paper_ids
from src.signal_decay.decay import before_after_test, fit_half_life
from src.signal_decay.ken_french import download_ken_french
from src.signal_decay.papers import load_papers, select_for_results
from src.signal_decay.robustness import BOOTSTRAP_DRAWS, bootstrap_decay, placebo_test, pooled_placebo_test

SUMMARY_CSV = DATA_DIR / "signal_decay_summary.csv"
RESULTS_COLUMNS = contracts.CONTRACTS["decay_results"][1]
DETAIL_COLUMNS = [
    "signal", "paper", "ken_french_column", "included", "unverified", "status",
    "pre_n", "post_n", "pre_mean", "post_mean", "pre_se", "post_se",
    "pre_ci_low", "pre_ci_high", "post_ci_low", "post_ci_high",
    "diff", "diff_se", "diff_ci_low", "diff_ci_high", "t_stat", "p_value",
    "welch_se", "welch_ci_low", "welch_ci_high",
    "retained_share", "retained_ci_low", "retained_ci_high", "relative_decay",
    "half_life_months", "half_life_ci_low", "half_life_ci_high", "half_life_fit_rate",
    "half_life_null_p", "half_life_reportable", "placebo_n", "placebo_p",
    "early_citations", "citation_peer_n", "citation_peer_median", "citation_percentile",
    "notes",
]
SUMMARY_COLUMNS = ["test", "statistic", "value", "p_value", "n", "note"]


def _signal_series(factors: pd.DataFrame, column) -> pd.Series:
    return factors[factors["signal"] == column].set_index("date")["return"].sort_index()


def build_results(
    factors: pd.DataFrame,
    included: pd.DataFrame,
    excluded: pd.DataFrame,
    paper_ids: dict[str, str] | None = None,
    citations: bool = True,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (decay_results, decay_details) for the included and excluded papers."""
    paper_ids = paper_ids or {}
    results, details = [], []

    for _, p in included.iterrows():
        res_row = {c: None for c in RESULTS_COLUMNS}
        res_row.update(signal=p["signal"], paper=p["paper"], publication_date=p["publication_date"])
        det = {c: None for c in DETAIL_COLUMNS}
        det.update(signal=p["signal"], paper=p["paper"], ken_french_column=p["ken_french_column"],
                   included=True, unverified=bool(p.get("unverified", False)))
        notes = []

        series = _signal_series(factors, p["ken_french_column"])
        if pd.isna(p["publication_date"]):
            det["status"] = "no_publication_date"
        elif series.empty:
            det["status"] = "no_return_series"
        else:
            pub = p["publication_date"]
            res = before_after_test(series, pub)
            det.update({k: v for k, v in res.items() if k in DETAIL_COLUMNS})
            det["status"] = res["status"]
            det["half_life_months"] = fit_half_life(series, pub)
            det.update(placebo_test(series, pub))
            det.pop("real_diff", None)
            boot = bootstrap_decay(series, pub, draws=bootstrap_draws)
            det.update(boot)
            if np.isfinite(det["retained_share"]):
                det["relative_decay"] = 1 - det["retained_share"]
            if res["status"] == "ok":
                res_row.update(pre_mean=res["pre_mean"], post_mean=res["post_mean"],
                               ci_low=res["diff_ci_low"], ci_high=res["diff_ci_high"])
                if boot["half_life_reportable"]:
                    res_row["half_life_months"] = det["half_life_months"]
                else:
                    notes.append("half-life not reported (fails the HYPOTHESES.md rule; "
                                 "see half_life_null_p and half_life_fit_rate)")
            else:
                print(f"WARNING {p['signal']}: {res['status']}; stats left empty in decay_results")

        if citations and pd.notna(p["publication_date"]) and p["signal"] in paper_ids:
            try:
                att = era_adjusted_attention(paper_ids[p["signal"]], p["publication_date"].year)
                res_row["early_citations"] = det["early_citations"] = att["early_citations"]
                det.update(citation_peer_n=att["peer_n"], citation_peer_median=att["peer_median"],
                           citation_percentile=att["percentile"])
            except Exception as exc:  # keep going; record why the cells are empty
                notes.append(f"citations failed: {exc}")
        elif citations:
            notes.append("no openalex_id in paper_ids.csv")

        det["notes"] = "; ".join(notes) or None
        results.append(res_row)
        details.append(det)

    for _, p in excluded.iterrows():
        det = {c: None for c in DETAIL_COLUMNS}
        det.update(signal=p["signal"], paper=p["paper"], ken_french_column=p["ken_french_column"],
                   included=False, unverified=True, status="excluded_unverified")
        details.append(det)

    out = pd.DataFrame(results, columns=RESULTS_COLUMNS)
    out["publication_date"] = pd.to_datetime(out["publication_date"]).dt.strftime("%Y-%m-%d")
    out["early_citations"] = out["early_citations"].astype("Int64")
    return out, pd.DataFrame(details, columns=DETAIL_COLUMNS)


def build_summary(factors: pd.DataFrame, included: pd.DataFrame, details: pd.DataFrame) -> pd.DataFrame:
    """Pooled placebo (H2) and attention-vs-decay (H3) rows."""
    ok = details[details["included"] & (details["status"] == "ok")]
    ok_signals = set(ok["signal"])
    rows = included[included["signal"].isin(ok_signals)]
    series = {r["signal"]: _signal_series(factors, r["ken_french_column"]) for _, r in rows.iterrows()}
    dates = dict(zip(rows["signal"], rows["publication_date"]))

    h2 = pooled_placebo_test(series, dates)
    h3 = attention_decay_test(ok["citation_percentile"].astype(float), ok["relative_decay"].astype(float))
    return pd.DataFrame(
        [
            {"test": "H2 pooled placebo", "statistic": "mean post-pre monthly return",
             "value": h2["mean_real_diff"], "p_value": h2["p_one_sided"], "n": h2["n_signals"],
             "note": f"one random fake split per signal, 10,000 draws; mean fake diff "
                     f"{h2.get('mean_fake_diff', np.nan):.5f}"},
            {"test": "H3 attention vs decay", "statistic": "Spearman rho",
             "value": h3["spearman"], "p_value": h3["p_one_sided"], "n": h3["n"],
             "note": "citation percentile among journal-year peers vs 1 - post/pre; "
                     "exact one-sided permutation p; suggestive only"},
        ],
        columns=SUMMARY_COLUMNS,
    )


def main(argv: list[str] | None = None) -> None:
    """Download factors, run the tests per paper, and write the output files."""
    parser = argparse.ArgumentParser(description="Run the signal-decay chain.")
    parser.add_argument("--refresh", action="store_true", help="re-download Ken French files")
    parser.add_argument("--allow-unverified", action="store_true",
                        help="include rows with source_verified != yes (marked in decay_details.csv)")
    parser.add_argument("--no-citations", action="store_true", help="skip OpenAlex lookups")
    parser.add_argument("--charts", action="store_true", help="save PNG charts")
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS,
                        help="bootstrap draws for half-life and retained-share intervals")
    args = parser.parse_args(argv)

    factors = download_ken_french(refresh=args.refresh)
    out_factors = factors.assign(date=factors["date"].dt.strftime("%Y-%m-%d"))
    print(f"Wrote {len(out_factors)} rows to {contracts.save('factors', out_factors)}")

    try:
        papers = load_papers()
    except FileNotFoundError as exc:
        print(f"{exc}\nSkipping decay_results.csv so the sample file stays in use.")
        return

    included, excluded = select_for_results(papers, allow_unverified=args.allow_unverified)
    if len(excluded):
        print(f"Excluded {len(excluded)} unverified paper(s): {', '.join(excluded['signal'])}")

    results, details = build_results(
        factors, included, excluded, paper_ids=load_paper_ids(),
        citations=not args.no_citations, bootstrap_draws=args.bootstrap_draws,
    )
    details.to_csv(DECAY_DETAILS_CSV, index=False)
    print(f"Wrote {len(details)} rows to {DECAY_DETAILS_CSV}")
    if results.empty:
        print("No papers included; decay_results.csv not written so the sample file stays in use.")
        return
    print(f"Wrote {len(results)} rows to {contracts.save('decay_results', results)}")

    summary = build_summary(factors, included, details)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote {len(summary)} rows to {SUMMARY_CSV}")

    if args.charts:
        from src.signal_decay.plots import plot_attention_vs_decay, save_all_charts

        paths = save_all_charts(included, factors=factors)
        if details["citation_percentile"].notna().any():
            paths.append(plot_attention_vs_decay(details, save_path=paths[0].parent / "attention_vs_decay.png"))
        print(f"Saved {len(paths)} charts to {paths[0].parent}")


if __name__ == "__main__":
    main()
