"""Tests for the app-facing decay outputs: combined decay_results, OSAP IDs, curve, haircut."""

import re

import numpy as np
import pandas as pd
import pytest

from src.common import contracts
from src.signal_decay import DATA_DIR
from src.signal_decay.combined import (
    CURVE_MIN_SIGNALS, OSAP_PAPER_IDS_CSV, build_decay_results, decay_curve, event_panel,
)
from src.signal_decay.haircut import CAVEAT, haircut, share_distribution

RESULTS = DATA_DIR / "decay_results.csv"
needs_results = pytest.mark.skipif(not RESULTS.exists(), reason="decay_results.csv not built")


def _results():
    return pd.read_csv(RESULTS)


@needs_results
def test_keys_unique_and_non_blank():
    d = _results()
    contracts.validate("decay_results", d)
    assert d["signal"].notna().all() and (d["signal"].astype(str).str.strip() != "").all()
    assert d["source"].isin(["ken_french", "osap"]).all()
    assert not d.duplicated(["source", "signal"]).any()


@needs_results
def test_share_kept_arithmetic():
    d = _results()
    pos = d["pre_mean"] > 0
    assert np.allclose(d.loc[pos, "share_kept"], d.loc[pos, "post_mean"] / d.loc[pos, "pre_mean"], equal_nan=True)
    assert d.loc[~pos, "share_kept"].isna().all()


@needs_results
def test_every_papers_row_is_in_decay_results():
    papers = pd.read_csv(DATA_DIR / "papers.csv")
    kf = _results().query("source == 'ken_french'")
    assert set(papers["signal"]) == set(kf["signal"])
    osap = pd.read_csv(DATA_DIR / "osap_decay_results.csv")
    assert set(osap["signal"]) == set(_results().query("source == 'osap'")["signal"])


@needs_results
def test_both_sources_keep_shared_names():
    d = _results()
    for name in ["Size", "Investment"]:
        assert set(d.loc[d["signal"] == name, "source"]) == {"ken_french", "osap"}


@needs_results
def test_kf_rows_match_decay_details():
    det = pd.read_csv(DATA_DIR / "decay_details.csv").set_index("signal")
    kf = _results().query("source == 'ken_french'").set_index("signal")
    for col, src in [("pre_mean", "pre_mean"), ("post_mean", "post_mean"), ("ci_low", "diff_ci_low"),
                     ("ci_high", "diff_ci_high"), ("early_citations", "early_citations")]:
        assert np.allclose(kf[col], det.loc[kf.index, src])
    assert kf["half_life_months"].isna().all()  # none passes the reporting rule


@needs_results
def test_osap_publication_date_is_first_post_month():
    osap = pd.read_csv(DATA_DIR / "osap_decay_results.csv").set_index("signal")
    d = _results().query("source == 'osap'").set_index("signal")
    expected = (osap.loc[d.index, "year"] + 1).astype(str) + "-01-01"
    assert (d["publication_date"] == expected).all()


def test_osap_openalex_ids_unique_per_signal():
    ids = pd.read_csv(OSAP_PAPER_IDS_CSV)
    assert ids["signal"].is_unique and ids["openalex_id"].notna().all()
    assert ids["openalex_id"].map(lambda s: bool(re.fullmatch(r"W\d+", s))).all()
    papers = pd.read_csv(DATA_DIR / "osap_papers.csv")
    assert set(ids["openalex_id"]) == set(papers.loc[papers["match_status"] == "matched", "openalex_id"])
    # Ken French IDs are untouched (H3 input); OSAP IDs live in their own file
    assert len(pd.read_csv(DATA_DIR / "paper_ids.csv")) == 7


def test_build_rejects_duplicate_keys():
    papers = pd.DataFrame({"signal": ["A", "A"], "paper": ["p", "p"], "publication_date": ["2000-01-01"] * 2})
    details = pd.DataFrame({"signal": ["A"], "status": ["ok"], "pre_mean": [1.0], "post_mean": [0.5],
                            "half_life_months": [np.nan], "half_life_reportable": [False],
                            "diff_ci_low": [0.0], "diff_ci_high": [0.0], "early_citations": [1]})
    empty = pd.DataFrame(columns=["signal", "authors", "year", "journal", "is_mean", "post_mean",
                                  "ci_low", "ci_high"])
    with pytest.raises(ValueError, match="unique"):
        build_decay_results(papers, details, empty, pd.DataFrame(columns=["authors", "year", "journal",
                                                                          "early_citations"]))


def _curve_inputs(n_signals=60, short=0):
    """Return 1 before publication and 0 after; `short` signals published late (few post years)."""
    dates = pd.date_range("1970-01-31", "2025-12-31", freq="ME")
    rows, info = [], []
    for i in range(n_signals):
        year = 2020 if i < short else 1990
        info.append({"signal": f"S{i}", "year": year, "sample_start": 1970, "is_mean": 0.5})
        post = dates.year > year  # January of year + 1 onward
        rows.append(pd.DataFrame({"date": dates, "signal": f"S{i}", "return": np.where(post, 0.0, 0.5)}))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(info)


def test_curve_uses_months_after_the_correct_publication_date():
    returns, info = _curve_inputs()
    curve = decay_curve(event_panel(returns, info), draws=50).set_index("years_since_publication")
    assert curve.loc[-1, "mean_scaled_return"] == pytest.approx(1.0)  # December of the publication year is pre
    assert curve.loc[0, "mean_scaled_return"] == pytest.approx(0.0)  # January of year + 1 is post
    assert (curve.loc[curve.index >= 0, "period"] == "post-publication").all()


def test_curve_minimum_signals_rule():
    returns, info = _curve_inputs(n_signals=60, short=20)  # 20 signals published in 2020: 5 post years
    curve = decay_curve(event_panel(returns, info), draws=50)
    assert (curve["n_signals"] >= CURVE_MIN_SIGNALS).all()
    # Years 0-4 have all 60 signals; from year 5 only the 40 long-history signals remain (< 50): dropped.
    assert curve["years_since_publication"].max() == 4
    real = DATA_DIR / "decay_curve.csv"
    if real.exists():
        assert (pd.read_csv(real)["n_signals"] >= CURVE_MIN_SIGNALS).all()


def _toy_distribution():
    res = pd.DataFrame({"source": "osap", "share_kept": [-0.5, 0.1, 0.3, 0.4, 0.6, 0.9, 1.4],
                        "paper": list("abcdefg")})
    return share_distribution(res, draws=200)


def test_haircut_is_monotonic_in_claimed_value():
    d = _toy_distribution()
    outs = [haircut(x, d) for x in [0.0, 0.001, 0.005, 0.01, 0.05]]
    for key in ["p25", "median", "p75", "median_ci_low", "median_ci_high"]:
        vals = [o[key] for o in outs]
        assert all(b >= a for a, b in zip(vals, vals[1:])), key
    assert outs[3]["median"] == pytest.approx(0.01 * d["median"])


def test_haircut_reports_values_outside_zero_one_and_rejects_negative():
    d = _toy_distribution()
    out = haircut(0.01, d)
    assert out["share_below_zero"] == pytest.approx(1 / 7) and out["share_above_one"] == pytest.approx(1 / 7)
    with pytest.raises(ValueError):
        haircut(-0.01, d)


def test_haircut_states_its_caveat():
    from src.signal_decay import haircut as module

    for doc in (module.__doc__, haircut.__doc__):
        doc = " ".join(doc.split())  # ignore line wrapping
        assert "not a prediction for any one signal" in doc and "replicable published signals only" in doc
    assert haircut(0.01, _toy_distribution())["caveat"] == CAVEAT
