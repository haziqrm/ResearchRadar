import pandas as pd

from src.live import briefing as br


def card(significant=False, held=True):
    return {
        "period_start": "2026-11-02", "unit": "week", "window": 4, "source": "arxiv",
        "busiest": [{"area": "Market microstructure", "score": 2.3, "papers": 30},
                    {"area": "LLMs in finance", "score": 1.1, "papers": 21}],
        "quietest": {"area": "Machine learning for returns", "score": -1.8, "papers": 11},
        "example_paper": {"arxiv_id": "1", "title": "TensorLOB", "published": "2026-11-02"},
        "allocation": {"for_month": "October 2026", "data_through": "2026-09-30",
                       "held": [{"signal": "Value", "weight": 0.52, "expected_edge_net": 0.0012},
                                {"signal": "Momentum", "weight": 0.48, "expected_edge_net": 0.0019}]
                       if held else [],
                       "skipped": ["Size"], "min_edge": 0.001},
        "caveat": {"sharpe_minus_equal_weight": 0.11, "ci_low": -0.18 if not significant else 0.02,
                   "ci_high": 0.4, "significant": significant},
    }


def test_crowding_facts_sums_last_four_weeks():
    weeks = pd.date_range("2026-10-05", periods=5, freq="7D").strftime("%Y-%m-%d")
    rows = [{"date": w, "area": a, "paper_count": n, "crowding_score": s}
            for i, w in enumerate(weeks)
            for a, n, s in [("A", 10, 0.5 + i), ("B", 1, -1.0), ("C", 3, 0.0)]]
    f = br.crowding_facts(pd.DataFrame(rows), 4)
    assert f["period_start"] == weeks[-1]
    assert [r["area"] for r in f["busiest"]] == ["A", "C", "B"]
    assert f["busiest"][0]["papers"] == 40      # 4 weeks, not 5
    assert f["quietest"]["area"] == "B"


def test_template_has_facts_and_caveat():
    text = br.template_text(card())
    assert "week of 2 November 2026" in text
    assert "right now is Market microstructure: 30 papers in the last 4 weeks" in text
    assert text.index("TensorLOB") < text.index("quietest")   # example is the busiest area's
    assert "most crowded area right now" in text and "busiest" not in text
    assert "LLMs in finance is also above its usual share, at plus 1.1." in text
    assert "Value at 52 percent and momentum at 48 percent." in text
    assert "skipping size, because after costs it is not expected" in text
    assert "isn't proven" in text and "minus 0.18" in text and "beat equal" in text


def test_template_cash_and_significant_wording():
    text = br.template_text(card(significant=True, held=False))
    assert "sit in cash" in text
    assert "isn't proven" not in text and "track record" in text


def test_template_negative_and_quiet_wording():
    c = card()
    c["busiest"][1]["score"] = -0.2
    c["caveat"]["sharpe_minus_equal_weight"] = -0.05
    text = br.template_text(c)
    assert "Next come LLMs in finance, at minus 0.2." in text
    assert "trailed equal weighting by 0.05" in text


def test_most_papers_named_when_not_the_most_crowded():
    c = card()
    c["most_papers"] = {"area": "Portfolio and risk", "score": -1.3, "papers": 49}
    text = br.template_text(c)
    assert "Portfolio and risk still has the most papers, 49, but that's below" in text

    c["most_papers"] = c["busiest"][0]                      # same area: no extra line
    assert "still has the most papers" not in br.template_text(c)


def test_crowding_facts_most_papers():
    weeks = pd.date_range("2026-10-05", periods=4, freq="7D").strftime("%Y-%m-%d")
    rows = [{"date": w, "area": a, "paper_count": n, "crowding_score": s}
            for w in weeks for a, n, s in [("A", 2, 2.0), ("B", 9, -1.0)]]
    f = br.crowding_facts(pd.DataFrame(rows), 4)
    assert f["busiest"][0]["area"] == "A"
    assert f["most_papers"] == {"area": "B", "score": -1.0, "papers": 36}


def test_half_up_rounds_like_people_expect():
    assert br.half_up(0.35, 1) == 0.4
    assert br.half_up(-0.35, 1) == -0.4
    assert br.half_up(52.5) == 53
    assert br.half_up(0.125, 2) == 0.13


def test_same_numbers_guards_the_rewrite():
    base = br.template_text(card())
    assert br.same_numbers(base, base.replace("So, what", "Now, what"))
    assert not br.same_numbers(base, base.replace("30 papers", "31 papers"))
    assert not br.same_numbers(base, base + " Up 5 percent.")


def test_yearly_openalex_wording():
    c = card()
    c.update({"period_start": "2026-01-01", "unit": "year", "window": 1, "source": "openalex"})
    text = br.template_text(c)
    assert "research radar for 2026 so far, from a sample of OpenAlex" in text
    assert "Market microstructure: 30 papers in this year's sample" in text
    assert 'Its most cited paper this year is called "TensorLOB"' in text
    assert "week" not in text


def test_most_cited_paper_picks_the_area_and_year():
    nodes = pd.DataFrame({"paper_id": ["W1", "W2", "W3", "W4"],
                          "title": ["old", "  low  cited ", "top", "other area"],
                          "year": [2025, 2026, 2026, 2026], "area": ["A", "A", "A", "B"],
                          "citations": [500, 1, 9, 99]})
    assert br.most_cited_paper(nodes, "A", 2026)["paper_id"] == "W3"
    assert br.most_cited_paper(nodes, "C", 2026) is None
