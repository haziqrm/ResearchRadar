"""Run the signal-decay chain.

Usage:
    python -m src.signal_decay.run_decay [--refresh] [--allow-unverified]
                                         [--no-citations] [--charts]

Writes:
    data/factors.csv        contract: date, signal, return (Mkt-RF is a benchmark only)
    data/decay_results.csv  contract; only written when at least one paper is included
    data/decay_details.csv  A-owned extra stats, status and flags per paper
    data/cache/charts/      one PNG per paper with --charts

decay_results.csv columns:
    pre_mean, post_mean   mean monthly return before / after publication
    ci_low, ci_high       95% CI of (post_mean - pre_mean), Newey-West, 12 lags
    half_life_months      ln2/k from an exponential fit; empty if no clear decay
    early_citations       works citing the paper in years 1-3 after publication
Rows below 120 months on either side keep their row with stats left empty;
the reason is in decay_details.csv (status column).
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.common import contracts
from src.signal_decay import DECAY_DETAILS_CSV
from src.signal_decay.attention import early_citations, load_paper_ids
from src.signal_decay.decay import before_after_test, fit_half_life
from src.signal_decay.ken_french import download_ken_french
from src.signal_decay.papers import load_papers, select_for_results

RESULTS_COLUMNS = contracts.CONTRACTS["decay_results"][1]
DETAIL_COLUMNS = [
    "signal", "paper", "ken_french_column", "included", "unverified", "status",
    "pre_n", "post_n", "pre_mean", "post_mean", "pre_se", "post_se",
    "pre_ci_low", "pre_ci_high", "post_ci_low", "post_ci_high",
    "diff", "diff_se", "diff_ci_low", "diff_ci_high", "t_stat", "p_value",
    "welch_se", "welch_ci_low", "welch_ci_high", "half_life_months",
    "early_citations", "notes",
]


def build_results(
    factors: pd.DataFrame,
    included: pd.DataFrame,
    excluded: pd.DataFrame,
    paper_ids: dict[str, str] | None = None,
    citations: bool = True,
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

        series = factors[factors["signal"] == p["ken_french_column"]].set_index("date")["return"]
        if pd.isna(p["publication_date"]):
            det["status"] = "no_publication_date"
        elif series.empty:
            det["status"] = "no_return_series"
        else:
            res = before_after_test(series, p["publication_date"])
            det.update({k: v for k, v in res.items() if k in DETAIL_COLUMNS})
            det["status"] = res["status"]
            half_life = fit_half_life(series, p["publication_date"])
            det["half_life_months"] = half_life
            if res["status"] == "ok":
                res_row.update(pre_mean=res["pre_mean"], post_mean=res["post_mean"],
                               ci_low=res["diff_ci_low"], ci_high=res["diff_ci_high"],
                               half_life_months=half_life)
            else:
                print(f"WARNING {p['signal']}: {res['status']}; stats left empty in decay_results")

        if citations and pd.notna(p["publication_date"]) and p["signal"] in paper_ids:
            try:
                n = early_citations(paper_ids[p["signal"]], p["publication_date"].year)
                res_row["early_citations"] = det["early_citations"] = n
            except Exception as exc:  # keep going; record why the cell is empty
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


def main(argv: list[str] | None = None) -> None:
    """Download factors, run the tests per paper, and write the output files."""
    parser = argparse.ArgumentParser(description="Run the signal-decay chain.")
    parser.add_argument("--refresh", action="store_true", help="re-download Ken French files")
    parser.add_argument("--allow-unverified", action="store_true",
                        help="include rows with source_verified != yes (marked in decay_details.csv)")
    parser.add_argument("--no-citations", action="store_true", help="skip OpenAlex lookups")
    parser.add_argument("--charts", action="store_true", help="save one PNG per paper")
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
        factors, included, excluded, paper_ids=load_paper_ids(), citations=not args.no_citations
    )
    details.to_csv(DECAY_DETAILS_CSV, index=False)
    print(f"Wrote {len(details)} rows to {DECAY_DETAILS_CSV}")
    if results.empty:
        print("No papers included; decay_results.csv not written so the sample file stays in use.")
        return
    print(f"Wrote {len(results)} rows to {contracts.save('decay_results', results)}")

    if args.charts:
        from src.signal_decay.plots import save_all_charts

        paths = save_all_charts(included, factors=factors)
        print(f"Saved {len(paths)} charts to {paths[0].parent}")


if __name__ == "__main__":
    main()
