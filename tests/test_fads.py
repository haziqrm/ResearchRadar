"""Fad check (optional add-on): fad_data.py. Delete with it."""
import fad_data
import research_data


def test_fads_cover_the_map_topics_with_labels():
    d = fad_data.get_fads()
    assert [t["topic"] for t in d["topics"]] == research_data.AREAS
    assert all(t["label"] in {"booming", "past fad", "fading", "steady"} for t in d["topics"])
    assert d["last_full"] < d["years"][1]          # the latest year is partial
    assert "tests" not in d                        # descriptive only


def test_smoothing_uses_only_past_years():
    years = list(range(2000, 2010))
    s = fad_data.smooth({y: float(y - 2000) for y in years}, years)
    assert 2000 not in s and 2001 not in s
    assert s[2002] == 1.0 and s[2009] == 8.0       # mean of y-2..y


def test_boom_rule():
    s = {2000: 2.0, 2005: 3.1, 2006: 2.9}
    assert fad_data.is_boom(s, 2005)               # 1.55x and +1.1 points
    assert not fad_data.is_boom(s, 2006)           # no 2001 to compare with
    assert not fad_data.is_boom({2000: 0.2, 2005: 0.9}, 2005)   # 4.5x but only +0.7 points


def test_labels_match_the_rules():
    d = fad_data.get_fads()
    for t in d["topics"]:
        if t["label"] == "booming":
            assert t["now"] >= 1.5 * t["five_ago"] - 0.1
        if t["label"] in ("fading", "past fad"):
            assert t["now"] <= 0.6 * t["peak"] + 0.1
