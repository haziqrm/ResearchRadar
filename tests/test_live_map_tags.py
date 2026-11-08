import json

import pandas as pd

from src.live import map_tags as mt


class FakeError(Exception):
    def __init__(self, text, code=429):
        super().__init__(text)
        self.code = code


def papers(n):
    return pd.DataFrame({"arxiv_id": [f"p{i}" for i in range(n)],
                         "published": ["2026-11-02"] * n,
                         "title": [f"t{i}" for i in range(n)],
                         "abstract": ["a"] * n})


def test_table_separates_none_from_not_tagged_yet():
    cache = {"p0": {"area": "Behavioral finance", "confidence": 0.9},
             "p1": {"area": "", "confidence": 0.8}}          # Gemini said none
    table = mt.build_table(papers(3), cache)
    assert table["area"].tolist() == ["Behavioral finance", "none", ""]


def test_daily_quota_stops_and_keeps_what_was_tagged(tmp_path):
    calls = []

    def call(client, model, prompt):
        calls.append(prompt)
        if len(calls) == 2:
            raise FakeError("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel")
        ids = [p["id"] for p in json.loads(prompt.split("Papers:\n", 1)[1])]
        return json.dumps([{"id": i, "area": "Behavioral finance", "confidence": 1}
                           for i in ids])

    cache_path = tmp_path / "cache.jsonl"
    cache = mt.tag(papers(mt.BATCH_SIZE * 3), None, "m", cache_path, call=call, gap=0)
    assert len(calls) == 2 and len(cache) == mt.BATCH_SIZE    # stopped, first batch kept
    assert len(cache_path.read_text().splitlines()) == mt.BATCH_SIZE


def test_per_minute_quota_is_not_daily():
    assert mt.is_daily_quota(FakeError("429 RESOURCE_EXHAUSTED PerDay"))
    assert not mt.is_daily_quota(FakeError("429 RESOURCE_EXHAUSTED PerMinute"))
    assert not mt.is_daily_quota(FakeError("500 internal", code=500))
