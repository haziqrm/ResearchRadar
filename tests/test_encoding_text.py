import pytest

from src.research_map.encoding.text import (
    QUALITY_FULL,
    QUALITY_PARTIAL,
    QUALITY_TITLE_ONLY,
    build_text,
    reconstruct_abstract,
    record_from_work,
)


def test_reconstruct_abstract_orders_by_position():
    index = {"returns": [2], "Momentum": [0], "predicts": [1], "momentum": [4], "and": [3]}
    assert reconstruct_abstract(index) == "Momentum predicts returns and momentum"


def test_reconstruct_abstract_handles_repeated_words_and_empty():
    assert reconstruct_abstract({"a": [0, 2], "b": [1]}) == "a b a"
    assert reconstruct_abstract(None) == ""
    assert reconstruct_abstract({}) == ""


def test_record_with_full_metadata():
    work = {
        "id": "https://openalex.org/W123",
        "title": "Deep   hedging",
        "abstract_inverted_index": {"We": [0], "hedge": [1]},
        "keywords": [{"display_name": "Hedging"}, {"display_name": "hedging"}],
        "topics": [{"display_name": "Derivatives"}, {"display_name": "Derivatives"}],
    }
    rec = record_from_work(work)
    assert rec.paper_id == "W123"
    assert rec.title == "Deep hedging"
    assert rec.keywords == ("Hedging",)
    assert rec.topics == ("Derivatives",)
    assert rec.text == "Deep hedging [SEP] We hedge. Keywords: Hedging. Topics: Derivatives."
    assert rec.text_quality == QUALITY_FULL


def test_missing_abstract_falls_back_to_title_and_keywords():
    rec = record_from_work(
        {"id": "W1", "title": "Pairs trading", "abstract_inverted_index": None,
         "keywords": [{"display_name": "arbitrage"}], "topics": None}
    )
    assert rec.abstract == ""
    assert rec.text == "Pairs trading. Keywords: arbitrage."
    assert rec.text_quality == QUALITY_PARTIAL


def test_title_only_and_none_fields():
    rec = record_from_work(
        {"id": "W2", "title": "Bare title", "keywords": None, "topics": None}
    )
    assert rec.text == "Bare title"
    assert rec.text_quality == QUALITY_TITLE_ONLY


def test_missing_title_uses_display_name_and_missing_id_raises():
    assert record_from_work({"id": "W3", "display_name": "Fallback"}).title == "Fallback"
    with pytest.raises(ValueError):
        record_from_work({"title": "no id"})


def test_build_text_does_not_double_punctuation():
    assert build_text("T", "Ends here.", ["k"]) == "T [SEP] Ends here. Keywords: k."
