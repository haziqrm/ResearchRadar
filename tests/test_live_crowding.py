import numpy as np
import pandas as pd

from src.common import contracts
from src.live import crowding as cr
from src.live.crowding import AREAS, OTHER, SOURCES

ARXIV = SOURCES["arxiv"]
OPENALEX = SOURCES["openalex"]


def scores(counts, p):
    return cr.crowding_scores(counts, p["window"], p["baseline"], p["min_baseline"])


def tagged_frame(weeks, per_week, surge_area=None, surge=0, start="2025-01-06"):
    """Same mix of areas every week (plus 'none'); optionally extra papers in the last week."""
    rows = []
    mondays = pd.date_range(start, periods=weeks, freq="7D")
    for i, monday in enumerate(mondays):
        for area in AREAS + ["none"]:
            k = per_week + (surge if area == surge_area and i == weeks - 1 else 0)
            rows += [{"published": monday.strftime("%Y-%m-%d"), "area": area}] * k
    rows.append({"published": mondays[0].strftime("%Y-%m-%d"), "area": ""})  # not tagged yet
    return pd.DataFrame(rows)


def nodes_frame(years, per_year, surge_area=None, surge=0):
    rows = []
    for i, y in enumerate(years):
        for area in AREAS + ["Real estate investment"]:          # a B area not on the map
            k = per_year + (surge if area == surge_area and i == len(years) - 1 else 0)
            rows += [{"year": y, "area": area}] * k
    return pd.DataFrame(rows)


def test_map_areas_match_the_site_and_bs_groups():
    import research_data
    assert AREAS == research_data.AREAS
    assert set(AREAS) <= set(contracts.area_names())


def test_arxiv_matches_contract_and_flat_is_zero():
    out = scores(cr.arxiv_counts(tagged_frame(40, 5)), ARXIV)
    contracts.validate("crowding", out)
    assert out["date"].nunique() == 40 - ARXIV["window"] - ARXIV["min_baseline"] + 1
    assert np.allclose(out["crowding_score"], 0)
    assert (out["paper_count"] == 5).all()
    assert set(out["area"]) == set(AREAS)                        # never the OTHER column


def test_arxiv_flags_a_surge():
    area = "Crypto, COVID and spillovers"
    out = scores(cr.arxiv_counts(tagged_frame(40, 5, area, 15)), ARXIV)
    last = out[out["date"] == out["date"].max()].set_index("area")["crowding_score"]
    assert last[area] > 2
    assert (last.drop(area) < 0).all()


def test_overall_growth_alone_is_not_crowding():
    small = tagged_frame(20, 2)
    big = tagged_frame(20, 6, start="2025-05-26")      # 20 weeks after small starts
    out = scores(cr.arxiv_counts(pd.concat([small, big])), ARXIV)
    assert np.allclose(out["crowding_score"], 0)


def test_untagged_papers_are_left_out_and_none_counts_as_other():
    df = pd.DataFrame({"published": ["2025-01-06", "2025-01-06", "2025-01-27"],
                       "area": [AREAS[0], "", "none"]})
    counts = cr.arxiv_counts(df)
    assert len(counts) == 4                                        # gap weeks filled
    assert counts[AREAS[0]].tolist() == [1, 0, 0, 0]
    assert counts[OTHER].tolist() == [0, 0, 0, 1]


def test_openalex_is_yearly_and_flags_a_surge():
    area = "Deep learning and LLM forecasting"
    out = scores(cr.openalex_counts(nodes_frame(range(2015, 2027), 10, area, 20)), OPENALEX)
    contracts.validate("crowding", out)
    assert out["date"].min() == "2018-01-01"     # 3 years of history before the window
    assert out["date"].max() == "2026-01-01"
    last = out[out["date"] == "2026-01-01"].set_index("area")
    assert last.loc[area, "paper_count"] == 30
    assert last.loc[area, "crowding_score"] > 2
    assert np.allclose(out[out["date"] < "2026-01-01"]["crowding_score"], 0)
