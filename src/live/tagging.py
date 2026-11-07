"""Tag each arXiv paper with an area (shared list) and a matching signal (papers.csv).

Usage:
    python -m src.live.tagging [--tagger auto|gemini|keyword] [--limit N]
    python -m src.live.tagging --audit 50       # write data/tag_audit.csv to check by eye
    python -m src.live.tagging --score-audit    # agreement rate once it's filled in

Two taggers:
- gemini:  BATCH_SIZE (50) abstracts per request, answers forced into the fixed lists,
           cached by arXiv ID in data/cache/arxiv/tags_gemini.jsonl, so a crash or a
           quota stop never means re-tagging. Rerun to continue.
- keyword: transparent rules, no key needed. Used as the fallback for any paper
           Gemini hasn't tagged yet, and as a baseline to compare Gemini against.
`auto` (default) uses Gemini when GEMINI_API_KEY is set, otherwise keywords.

Writes data/arxiv_tagged.csv with TAG_COLUMNS. area/signal are "" when nothing fits.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.common import contracts
from src.common.contracts import AREAS, DATA_DIR, ROOT
from src.live import CACHE_DIR
from src.live.arxiv_feed import load_papers

TAGGED_CSV = DATA_DIR / "arxiv_tagged.csv"
AUDIT_CSV = DATA_DIR / "tag_audit.csv"
GEMINI_CACHE = CACHE_DIR / "tags_gemini.jsonl"
TAG_COLUMNS = ["arxiv_id", "published", "title", "area", "signal", "confidence", "tagger"]

# 50 abstracts per request: matched 25-paper batches on 46/50 areas and 50/50 signals,
# while 75 drifted (66-67/75, more off-topic papers forced into an area). See docs.
BATCH_SIZE = 50
GEMINI_GAP_SECONDS = 6.0      # ~10 requests a minute, inside free-tier limits
# Chosen on 2026-11-07 after a 50-paper comparison (docs/live_layer.md): most accurate
# of the flash models, and 3.8-flash was overloaded. Override with GEMINI_MODEL in .env.
DEFAULT_MODEL = "gemini-3.5-flash"
# Errors that retrying can't fix (wrong model name, bad key): stop the run.
# Matched on the status code, never on digits in the message (a quota message's
# "retry in 10h23m40.440284047s" contains "404").
# 403 PERMISSION_DENIED is not fatal: it appeared once mid-run on a working key and
# cleared on its own, so it is retried like a 503.
FATAL_CODES = {400, 401, 404}
FATAL_STATUSES = ("NOT_FOUND", "API_KEY_INVALID", "UNAUTHENTICATED")


def error_kind(exc: Exception) -> str:
    """'fatal', 'quota' or 'retry' for an exception from the Gemini SDK."""
    code = getattr(exc, "code", None)
    text = str(exc)
    if code == 429 or "RESOURCE_EXHAUSTED" in text:
        return "quota"
    if code in FATAL_CODES or any(s in text for s in FATAL_STATUSES):
        return "fatal"
    return "retry"
MAX_RETRIES = 4

# --- keyword tagger --------------------------------------------------------
# Patterns are regexes matched on lowercased title + abstract. A title hit
# counts double. Ties go to the area listed first here (most specific first).
AREA_KEYWORDS: dict[str, list[str]] = {
    "LLMs in finance": [r"large language models?", r"\bllms?\b", r"\bgpt", r"chatgpt",
                        r"generative ai", r"language models?", r"\bagentic\b"],
    "Crypto and DeFi": [r"bitcoin", r"cryptocurrenc", r"\bcrypto", r"blockchain",
                        r"\bdefi\b", r"decentrali[sz]ed finance", r"ethereum",
                        r"stablecoin", r"automated market makers?", r"\btokens?\b"],
    "Alternative data and text": [r"sentiment", r"textual", r"text mining", r"\bnews\b",
                                  r"twitter", r"social media", r"satellite",
                                  r"alternative data", r"\bnlp\b", r"natural language",
                                  r"earnings calls?", r"reddit"],
    "Machine learning for returns": [r"machine learning", r"deep learning",
                                     r"neural networks?", r"random forests?",
                                     r"gradient boosting", r"reinforcement learning",
                                     r"\blstm\b", r"transformers?\b", r"return prediction",
                                     r"forecasting returns"],
    "Market microstructure": [r"order books?", r"limit orders?", r"market making",
                              r"market makers?", r"bid-ask", r"high-frequency",
                              r"price impact", r"optimal execution", r"\bliquidity\b",
                              r"microstructure", r"order flow"],
    "Derivatives and volatility": [r"\boptions?\b", r"volatility", r"black-scholes",
                                   r"\bderivatives?\b", r"swaptions?", r"variance swaps?",
                                   r"\bvix\b", r"hedging", r"option pricing"],
    "Factor and anomaly research": [r"\bfactors?\b", r"anomal", r"cross-section",
                                    r"asset pricing", r"risk premi", r"fama.french",
                                    r"\balphas?\b", r"stock returns", r"momentum"],
    "Portfolio and risk": [r"portfolios?", r"value-at-risk", r"expected shortfall",
                           r"mean-variance", r"asset allocation", r"risk management",
                           r"\bcvar\b", r"systemic risk", r"credit risk", r"risk measures?"],
}
assert set(AREA_KEYWORDS) == set(AREAS), "AREA_KEYWORDS must cover exactly the shared AREAS"

# Extra wording per signal; a signal not listed here matches its own name.
SIGNAL_KEYWORDS: dict[str, list[str]] = {
    "Size": [r"size effect", r"size premium", r"small-cap", r"\bsmb\b"],
    "Value": [r"value premium", r"value factor", r"book-to-market", r"\bhml\b",
              r"value stocks"],
    "Momentum": [r"momentum"],
    "Short-term reversal": [r"short-term reversal"],
    "Long-term reversal": [r"long-term reversal", r"overreaction"],
    # Bare "profitability" mostly means "is this strategy profitable", not the factor.
    "Profitability": [r"(gross|operating) profitability", r"profitability (premium|factor)",
                      r"gross profits?", r"\brmw\b"],
    "Investment": [r"investment factor", r"asset growth", r"\bcma\b"],
}


def signal_names() -> list[str]:
    return contracts.load("papers")["signal"].dropna().unique().tolist()


def _hits(patterns: list[str], title: str, text: str) -> int:
    return sum(2 * bool(re.search(p, title)) + bool(re.search(p, text)) for p in patterns)


def keyword_tag(title: str, abstract: str, signals: list[str]) -> dict:
    title, text = title.lower(), f"{title} {abstract}".lower()
    scores = {a: _hits(p, title, text) for a, p in AREA_KEYWORDS.items()}
    best = max(scores, key=scores.get)          # dict order breaks ties
    sig_scores = {s: _hits(SIGNAL_KEYWORDS.get(s, [re.escape(s.lower())]), title, text)
                  for s in signals}
    best_sig = max(sig_scores, key=sig_scores.get) if sig_scores else ""
    return {"area": best if scores[best] else "",
            "signal": best_sig if best_sig and sig_scores[best_sig] else "",
            "confidence": round(min(1.0, scores[best] / 4), 2),
            "tagger": "keyword"}


# --- Gemini tagger ---------------------------------------------------------
PROMPT = """You tag quantitative-finance research papers.

For each paper, choose:
- "area": exactly one string from this list, or "none" if none fits:
{areas}
- "signal": the one trading signal from this list that the paper mainly studies,
  or "none" (most papers study none of them):
{signals}
- "confidence": your confidence in the area, from 0 to 1.

Return a JSON list with one object per paper, using the paper's "id".

Papers:
{papers}"""

RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"id": {"type": "string"}, "area": {"type": "string"},
                       "signal": {"type": "string"}, "confidence": {"type": "number"}},
        "required": ["id", "area", "signal", "confidence"],
    },
}


class QuotaExhausted(RuntimeError):
    """Gemini kept refusing for rate/quota reasons; stop and rerun later."""


def build_prompt(batch: pd.DataFrame, signals: list[str]) -> str:
    papers = [{"id": r.arxiv_id, "title": r.title, "abstract": r.abstract[:1500]}
              for r in batch.itertuples()]
    return PROMPT.format(areas="\n".join(f"  - {a}" for a in AREAS),
                         signals="\n".join(f"  - {s}" for s in signals),
                         papers=json.dumps(papers, ensure_ascii=False, indent=1))


def clean_answers(raw: str, ids: list[str], signals: list[str]) -> list[dict]:
    """Keep only answers for requested ids, with values forced into the fixed lists."""
    by_id = {}
    for item in json.loads(raw):
        pid = str(item.get("id", ""))
        if pid not in ids:
            continue
        area, sig = item.get("area", ""), item.get("signal", "")
        try:
            conf = float(item.get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0.0
        by_id[pid] = {"arxiv_id": pid,
                      "area": area if area in AREAS else "",
                      "signal": sig if sig in signals else "",
                      "confidence": round(min(max(conf, 0.0), 1.0), 2),
                      "tagger": "gemini"}
    return list(by_id.values())


def call_gemini(client, model: str, prompt: str) -> str:
    from google.genai import types
    resp = client.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(temperature=0,
                                           response_mime_type="application/json",
                                           response_schema=RESPONSE_SCHEMA))
    return resp.text


def load_gemini_cache(path: Path = GEMINI_CACHE, model: str | None = None) -> dict[str, dict]:
    """Cached answers by arXiv ID. With `model`, only that model's answers count, so
    switching model re-tags everything with one model instead of mixing them."""
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if model is None or row.get("model") == model:
                out[row["arxiv_id"]] = row
    return out


def committed_gemini_tags(path: Path = TAGGED_CSV) -> dict[str, dict]:
    """Gemini tags already in the committed arxiv_tagged.csv. The API cache is git-ignored,
    so without this a teammate with no key would overwrite them with keyword tags."""
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df[df["tagger"] == "gemini"]
    return {r.arxiv_id: {"area": r.area, "signal": r.signal,
                         "confidence": float(r.confidence or 0), "tagger": "gemini"}
            for r in df.itertuples()}


def gemini_tag(papers: pd.DataFrame, signals: list[str], client, model: str,
               cache_path: Path = GEMINI_CACHE, gap: float = GEMINI_GAP_SECONDS,
               call=call_gemini, batch_size: int = BATCH_SIZE) -> dict[str, dict]:
    """Tag every uncached paper; returns the full cache. Stops cleanly on quota."""
    cache = load_gemini_cache(cache_path, model)
    todo = papers[~papers["arxiv_id"].isin(cache)]
    print(f"gemini: {len(cache)} cached, {len(todo)} to tag with {model}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(todo), batch_size):
        batch = todo.iloc[start:start + batch_size]
        ids = batch["arxiv_id"].tolist()
        for attempt in range(MAX_RETRIES):
            try:
                answers = clean_answers(call(client, model, build_prompt(batch, signals)),
                                        ids, signals)
                break
            except Exception as exc:
                kind = error_kind(exc)
                if kind == "fatal":
                    raise
                if kind == "quota" and "per day" in str(exc).lower():
                    raise QuotaExhausted(str(exc)[:200]) from exc   # waiting won't help
                quota = kind == "quota"
                if attempt == MAX_RETRIES - 1:
                    if quota:
                        raise QuotaExhausted(str(exc)[:200]) from exc
                    print(f"  batch at {start} failed, skipping for now: {str(exc)[:200]}")
                    answers = []
                    break
                print(f"  {'quota' if quota else 'error'}; retrying: {str(exc)[:120]}")
                time.sleep(gap * 2 ** (attempt + 1))
        with cache_path.open("a", encoding="utf-8") as f:
            for a in answers:
                a["model"] = model
                f.write(json.dumps(a) + "\n")
                cache[a["arxiv_id"]] = a
        print(f"  tagged {start + len(batch)}/{len(todo)} ({len(answers)} answers)")
        time.sleep(gap)
    return cache


# --- combine, audit, CLI ---------------------------------------------------
def tag_papers(papers: pd.DataFrame, signals: list[str],
               gemini_cache: dict[str, dict] | None = None) -> pd.DataFrame:
    """Gemini answer where cached, keyword tag otherwise."""
    gemini_cache = gemini_cache or {}
    rows = []
    for r in papers.itertuples():
        tag = gemini_cache.get(r.arxiv_id) or keyword_tag(r.title, r.abstract, signals)
        rows.append({"arxiv_id": r.arxiv_id, "published": r.published, "title": r.title,
                     "area": tag["area"], "signal": tag["signal"],
                     "confidence": tag["confidence"], "tagger": tag["tagger"]})
    return pd.DataFrame(rows, columns=TAG_COLUMNS)


def make_audit(tagged: pd.DataFrame, papers: pd.DataFrame, n: int, seed: int = 0) -> Path:
    """Random sample to check by eye: fill correct_area / correct_signal with y or n."""
    sample = tagged.sample(n=min(n, len(tagged)), random_state=seed)
    sample = sample.merge(papers[["arxiv_id", "abstract"]], on="arxiv_id")
    sample["abstract"] = sample["abstract"].str[:600]
    sample["correct_area"] = ""
    sample["correct_signal"] = ""
    sample[["arxiv_id", "title", "abstract", "tagger", "area", "signal",
            "correct_area", "correct_signal"]].to_csv(AUDIT_CSV, index=False)
    return AUDIT_CSV


def score_audit(path: Path = AUDIT_CSV) -> str:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    lines = []
    for col in ["correct_area", "correct_signal"]:
        marked = df[df[col].str.lower().isin(["y", "n"])]
        ok = (marked[col].str.lower() == "y").sum()
        pct = f"{ok / len(marked):.0%}" if len(marked) else "n/a"
        lines.append(f"{col.removeprefix('correct_')}: {ok} of {len(marked)} correct ({pct})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Tag arXiv papers with area and signal.")
    parser.add_argument("--tagger", choices=["auto", "gemini", "keyword"], default="auto")
    parser.add_argument("--limit", type=int, help="only tag the newest N papers (testing)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"abstracts per Gemini request (default {BATCH_SIZE})")
    parser.add_argument("--audit", type=int, metavar="N", help="write N papers to check by eye")
    parser.add_argument("--score-audit", action="store_true")
    args = parser.parse_args(argv)

    if args.score_audit:
        print(score_audit())
        return AUDIT_CSV

    load_dotenv(ROOT / ".env")
    papers = load_papers()
    if args.limit:
        papers = papers.tail(args.limit)
    signals = signal_names()
    key = os.getenv("GEMINI_API_KEY", "").strip()
    use_gemini = args.tagger == "gemini" or (args.tagger == "auto" and key)
    if args.tagger == "gemini" and not key:
        raise SystemExit("GEMINI_API_KEY missing in .env")

    model = os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL
    cache = load_gemini_cache(model=model)
    if use_gemini:
        from google import genai
        try:
            cache = gemini_tag(papers, signals, genai.Client(api_key=key), model,
                               batch_size=args.batch_size)
        except QuotaExhausted as exc:
            print(f"gemini quota reached ({exc}); rerun later to continue. "
                  "Untagged papers use keywords for now.")
            cache = load_gemini_cache(model=model)
    elif args.tagger == "keyword":
        cache = {}
    else:  # auto with no key: keep committed Gemini tags, keywords only for new papers
        cache = {**committed_gemini_tags(), **cache}

    tagged = tag_papers(papers, signals, cache)
    tagged.to_csv(TAGGED_CSV, index=False)
    counts = tagged["tagger"].value_counts().to_dict()
    print(f"wrote {len(tagged)} tagged papers to {TAGGED_CSV} {counts}; "
          f"{(tagged['area'] == '').sum()} with no area")
    if args.audit:
        print(f"audit sheet: {make_audit(tagged, papers, args.audit)}")
    return TAGGED_CSV


if __name__ == "__main__":
    main()
