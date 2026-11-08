"""Tag arXiv q-fin papers into the 8 research-map areas with Gemini.

Usage:
    python -m src.live.map_tags            # every paper in the feed
    python -m src.live.map_tags --days 60  # only the last 60 days

The site and the crowding score use 8 of B's 25 OpenAlex areas (MAP_AREAS, the same
list as research_data.AREAS), chosen by the team. OpenAlex only gives 380 sampled
papers a year, so for a weekly view the arXiv papers are tagged again, by Gemini,
into those 8 areas (or "none"). `python -m src.live.crowding --source arxiv` then
scores them week by week.

Writes data/arxiv_map_tags.csv: arxiv_id, published, title, area, confidence.
area is one of MAP_AREAS, "none", or "" for a paper Gemini hasn't reached yet.
Answers are cached by arXiv ID in data/cache/arxiv/map_tags_gemini.jsonl, so a run
that hits the daily quota stops, writes what it has, and the next run carries on.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.common.contracts import DATA_DIR, ROOT
from src.live import CACHE_DIR
from src.live.arxiv_feed import load_papers
from src.live.tagging import DEFAULT_MODEL, GEMINI_GAP_SECONDS, MAX_RETRIES, error_kind

MAP_TAGS_CSV = DATA_DIR / "arxiv_map_tags.csv"
MAP_CACHE = CACHE_DIR / "map_tags_gemini.jsonl"
MAP_TAG_COLUMNS = ["arxiv_id", "published", "title", "area", "confidence"]
BATCH_SIZE = 50

# B's area names (exact, they key the map) with the keywords B's grouping found,
# so Gemini sees what each cluster actually contains rather than guessing from a name.
MAP_AREAS: dict[str, str] = {
    "Asset pricing and risk premia": "risk premia, expected returns, factor models, the cross-section of returns",
    "Information and stock reactions": "how prices react to earnings, news, disclosures and other information",
    "Trading and order flow": "trading, order flow, liquidity, market making, execution, market microstructure",
    "Behavioral finance": "investor behaviour, biases, sentiment, behavioural explanations of prices",
    "Volatility forecasting (GARCH)": "forecasting volatility and risk, GARCH and realised-volatility models",
    "Deep learning and LLM forecasting": "deep learning, transformers and large language models used to forecast markets",
    "Classical stock price prediction": "predicting stock prices or returns with statistical or classical machine-learning models",
    "Crypto, COVID and spillovers": "cryptocurrencies, the COVID shock, contagion and spillovers between markets",
}

PROMPT = """You are tagging new quantitative-finance papers from arXiv by research area.

For each paper, choose:
- "area": exactly one name from this list, or "none" if the paper does not mainly
  belong to any of them (many papers will be "none"):
{areas}
- "confidence": your confidence in the area, from 0 to 1.

Return a JSON list with one object per paper, using the paper's "id".

Papers:
{papers}"""

RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"id": {"type": "string"}, "area": {"type": "string"},
                       "confidence": {"type": "number"}},
        "required": ["id", "area", "confidence"],
    },
}


def recent_papers(papers: pd.DataFrame, days: int | None = None) -> pd.DataFrame:
    """Papers from the last `days` days (all of them when days is None)."""
    if not days:
        return papers
    pub = pd.to_datetime(papers["published"]).dt.normalize()
    return papers[pub > pub.max() - pd.to_timedelta(days, unit="D")]


def build_prompt(batch: pd.DataFrame) -> str:
    papers = [{"id": r.arxiv_id, "title": r.title, "abstract": str(r.abstract)[:1500]}
              for r in batch.itertuples()]
    areas = "\n".join(f"  - {a}: {desc}" for a, desc in MAP_AREAS.items())
    return PROMPT.format(areas=areas, papers=json.dumps(papers, ensure_ascii=False, indent=1))


def clean_answers(raw: str, ids: list[str]) -> list[dict]:
    """Answers for requested ids only; an area not on the list becomes ""."""
    out = {}
    for item in json.loads(raw):
        pid = str(item.get("id", ""))
        if pid not in ids:
            continue
        area = item.get("area", "")
        try:
            conf = float(item.get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0.0
        out[pid] = {"arxiv_id": pid, "area": area if area in MAP_AREAS else "",
                    "confidence": round(min(max(conf, 0.0), 1.0), 2)}
    return list(out.values())


def call_gemini(client, model: str, prompt: str) -> str:
    from google.genai import types
    resp = client.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(temperature=0,
                                           response_mime_type="application/json",
                                           response_schema=RESPONSE_SCHEMA))
    return resp.text


def load_cache(path: Path, model: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())
    return {r["arxiv_id"]: r for r in rows if r.get("model") == model}


def is_daily_quota(exc: Exception) -> bool:
    """A per-day quota won't clear by retrying in seconds (free tier: 20 requests/day)."""
    text = str(exc)
    return error_kind(exc) == "quota" and ("PerDay" in text or "per day" in text)


def tag(papers: pd.DataFrame, client, model: str, cache_path: Path = MAP_CACHE,
        call=call_gemini, gap: float = GEMINI_GAP_SECONDS) -> dict[str, dict]:
    cache = load_cache(cache_path, model)
    todo = papers[~papers["arxiv_id"].isin(cache)]
    print(f"map tags: {len(cache)} cached, {len(todo)} to tag with {model}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo.iloc[start:start + BATCH_SIZE]
        ids = batch["arxiv_id"].tolist()
        for attempt in range(MAX_RETRIES):
            try:
                answers = clean_answers(call(client, model, build_prompt(batch)), ids)
                break
            except Exception as exc:
                if is_daily_quota(exc):
                    print(f"  daily Gemini quota reached; stopping at {start}/{len(todo)} "
                          f"(cached answers are kept, rerun tomorrow to continue)")
                    return cache
                if error_kind(exc) == "fatal" or attempt == MAX_RETRIES - 1:
                    raise
                print(f"  retrying: {str(exc)[:120]}")
                time.sleep(gap * 2 ** (attempt + 1))
        with cache_path.open("a", encoding="utf-8") as f:
            for a in answers:
                a["model"] = model
                f.write(json.dumps(a) + "\n")
                cache[a["arxiv_id"]] = a
        print(f"  tagged {start + len(batch)}/{len(todo)}")
        time.sleep(gap)
    return cache


def build_table(papers: pd.DataFrame, cache: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for r in papers.itertuples():
        a = cache.get(r.arxiv_id)
        area = "" if a is None else (a.get("area") or "none")   # "" = not tagged yet
        a = a or {}
        rows.append({"arxiv_id": r.arxiv_id, "published": str(r.published)[:10],
                     "title": r.title, "area": area,
                     "confidence": a.get("confidence", "")})
    return pd.DataFrame(rows, columns=MAP_TAG_COLUMNS)


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=None)
    args = ap.parse_args(argv)
    load_dotenv(ROOT / ".env")
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise SystemExit("GEMINI_API_KEY missing in .env")
    model = os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL

    from google import genai
    client = genai.Client(api_key=key)
    papers = recent_papers(load_papers(), args.days)
    table = build_table(papers, tag(papers, client, model))
    table.to_csv(MAP_TAGS_CSV, index=False)
    print(f"wrote {MAP_TAGS_CSV} ({len(table)} papers)")
    print(table["area"].replace("", "(not tagged yet)").value_counts().to_string())
    return MAP_TAGS_CSV


if __name__ == "__main__":
    main()
