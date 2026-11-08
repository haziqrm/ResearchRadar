"""Re-tag arXiv papers into the 8 research-map areas, for the monthly map.

Usage:
    python -m src.live.map_tags            # every paper in the feed
    python -m src.live.map_tags --days 60  # only the last 60 days
    python -m src.live.map_tags --model gemini-3.7-flash --shard 2/3
        # one of 3 parallel workers: tags only its third of the papers, into its own
        # cache file, and doesn't write the CSV. A plain run afterwards merges them.
    python -m src.live.map_tags --merge-only  # write the CSV from the cache, no API calls

The live crowding score keeps D's 8 shared AREAS. The research map instead shows 8 of
B's 25 OpenAlex areas (MAP_AREAS), chosen by the team as the ones that matter for the
pitch. So the map can show how those areas move month by month, the arXiv papers are
tagged again, by Gemini, into those 8 areas (or "none"). real_data.py turns the tags
into monthly shares.

Writes data/arxiv_map_tags.csv: arxiv_id, published, title, area, confidence.
Answers are cached by arXiv ID in data/cache/arxiv/map_tags_gemini*.jsonl (one file
per parallel worker). A paper cached by any model is not tagged again.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zlib
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


def cache_files(path: Path) -> list[Path]:
    """The main cache file plus every worker's file next to it (map_tags_gemini.*.jsonl)."""
    return sorted({path, *path.parent.glob(f"{path.stem}.*{path.suffix}")})


def load_cache(path: Path) -> dict[str, dict]:
    """Cached answers from every cache file, whichever model gave them."""
    cache = {}
    for f in cache_files(path):
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    cache[r["arxiv_id"]] = r
    return cache


def in_shard(arxiv_id: str, shard: tuple[int, int] | None) -> bool:
    """Stable split of papers between parallel workers (shard is (k, n), k from 1)."""
    if not shard:
        return True
    k, n = shard
    return zlib.crc32(arxiv_id.encode()) % n == k - 1


def worker_cache(model: str) -> Path:
    return MAP_CACHE.with_name(f"{MAP_CACHE.stem}.{model}{MAP_CACHE.suffix}")


def tag(papers: pd.DataFrame, client, model: str, cache_path: Path = MAP_CACHE,
        call=call_gemini, gap: float = GEMINI_GAP_SECONDS,
        shard: tuple[int, int] | None = None, write_path: Path | None = None,
        reverse: bool = False) -> dict[str, dict]:
    """Tag uncached papers in batches. With a shard, the cache is re-read before each
    batch, so two workers on the same shard (one with reverse=True, working from the
    other end) skip what the other has done."""
    cache = load_cache(cache_path)
    write_path = write_path or cache_path
    todo = papers[~papers["arxiv_id"].isin(cache)
                  & papers["arxiv_id"].map(lambda i: in_shard(i, shard))]
    print(f"map tags: {len(cache)} cached, {len(todo)} to tag with {model}"
          + (f" (shard {shard[0]}/{shard[1]})" if shard else ""))
    write_path.parent.mkdir(parents=True, exist_ok=True)
    tried: set[str] = set()   # ids sent already (an id Gemini skips isn't retried forever)
    while True:
        if shard:
            cache.update(load_cache(cache_path))
        left = todo[~todo["arxiv_id"].isin(cache) & ~todo["arxiv_id"].isin(tried)]
        if left.empty:
            break
        batch = left.iloc[-BATCH_SIZE:] if reverse else left.iloc[:BATCH_SIZE]
        ids = batch["arxiv_id"].tolist()
        tried.update(ids)
        for attempt in range(MAX_RETRIES):
            try:
                answers = clean_answers(call(client, model, build_prompt(batch)), ids)
                break
            except Exception as exc:
                if error_kind(exc) == "fatal" or attempt == MAX_RETRIES - 1:
                    raise
                print(f"  retrying: {str(exc)[:120]}")
                time.sleep(gap * 2 ** (attempt + 1))
        with write_path.open("a", encoding="utf-8") as f:
            for a in answers:
                a["model"] = model
                f.write(json.dumps(a) + "\n")
                cache[a["arxiv_id"]] = a
        print(f"  tagged {len(tried)}/{len(todo)}")
        time.sleep(gap)
    return cache


def build_table(papers: pd.DataFrame, cache: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for r in papers.itertuples():
        a = cache.get(r.arxiv_id)
        if a is None:   # not tagged yet: leave it out, or it would count as "none"
            continue
        rows.append({"arxiv_id": r.arxiv_id, "published": str(r.published)[:10],
                     "title": r.title, "area": a.get("area", ""),
                     "confidence": a.get("confidence", "")})
    return pd.DataFrame(rows, columns=MAP_TAG_COLUMNS)


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--model", default=None, help="Gemini model (default: GEMINI_MODEL)")
    ap.add_argument("--shard", default=None, help="K/N: tag only this worker's share")
    ap.add_argument("--reverse", action="store_true",
                    help="work from the end of the shard (to share it with another worker)")
    ap.add_argument("--merge-only", action="store_true",
                    help="write the CSV from cached tags without calling Gemini")
    args = ap.parse_args(argv)
    shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
    if shard and not (len(shard) == 2 and 1 <= shard[0] <= shard[1]):
        raise SystemExit("--shard must look like 2/3")
    if args.merge_only:
        papers = recent_papers(load_papers(), args.days)
        return write_table(build_table(papers, load_cache(MAP_CACHE)))
    load_dotenv(ROOT / ".env")
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        raise SystemExit("GEMINI_API_KEY missing in .env")
    model = args.model or os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL

    from google import genai
    client = genai.Client(api_key=key)
    papers = recent_papers(load_papers(), args.days)
    if shard:
        cache = tag(papers, client, model, shard=shard, write_path=worker_cache(model),
                    reverse=args.reverse)
        done = papers["arxiv_id"].isin(cache).sum()
        print(f"shard done; {done}/{len(papers)} papers tagged overall. "
              "Run without --shard to write the CSV.")
        return worker_cache(model)
    return write_table(build_table(papers, tag(papers, client, model)))


def write_table(table: pd.DataFrame) -> Path:
    table.to_csv(MAP_TAGS_CSV, index=False)
    print(f"wrote {MAP_TAGS_CSV} ({len(table)} papers)")
    print(table["area"].replace("", "none").value_counts().to_string())
    return MAP_TAGS_CSV


if __name__ == "__main__":
    main()
