"""This week's spoken briefing: crowding, the current allocation and one caveat.

Usage:
    python -m src.live.briefing [--no-gemini] [--no-audio]

1. Facts (the "decision card") are computed in code, never by a model:
     - the 3 most crowded areas this week and the quietest one, by crowding score
       (share of papers vs the area's usual share), plus the area with the most
       papers when that's a different one (crowding.csv)
     - the newest tagged paper in the most crowded area (arxiv_tagged.csv)
     - the allocation for next month: decay_tilt weights from the backtest's own
       rule, applied to every return up to the latest month in factors.csv
     - the caveat: decay_tilt vs equal weight from backtest_summary.csv
   Written to data/briefing_card.json for the dashboard.
2. Text: a fixed template fills in the facts. If GEMINI_API_KEY is set (and not
   --no-gemini), Gemini rewrites it to sound natural aloud; the rewrite is kept only
   if it contains exactly the same numbers, otherwise the template text is used.
3. Audio: ElevenLabs reads the text into data/briefing.mp3. If there is no key or
   the call fails, the committed briefing.mp3 stays as the pre-recorded fallback.
   Audio is only regenerated when the text changes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.common import contracts
from src.common.contracts import DATA_DIR, ROOT
from src.live import backtest as bt
from src.live.crowding import WINDOW_WEEKS
from src.live.tagging import DEFAULT_MODEL, TAGGED_CSV

TEXT_PATH = DATA_DIR / "briefing.txt"
AUDIO_PATH = DATA_DIR / "briefing.mp3"
CARD_PATH = DATA_DIR / "briefing_card.json"
AUDIO_STAMP = DATA_DIR / "cache" / "briefing_audio_text.txt"  # text the mp3 was made from

# ElevenLabs premade voice "Brian - Deep, Resonant and Comforting" on the newest model
# (the most natural of the ones compared). Override both in .env.
DEFAULT_VOICE_ID = "nPczCjzI2devNBz1zQrb"
DEFAULT_TTS_MODEL = "eleven_v4"
N_BUSY = 3


# ---------- facts ----------

def half_up(x: float, places: int = 0) -> float:
    """Round the way people expect when hearing it: 0.35 -> 0.4, 52.5 -> 53.
    Python's round() gives 0.3 and 52 (binary floats, banker's rounding)."""
    q = Decimal(1).scaleb(-places)
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))


def crowding_facts(crowding: pd.DataFrame) -> dict:
    """'busiest'/'quietest' rank by crowding score (share vs usual), not by paper count;
    'most_papers' is the area with the most papers in the window, for contrast."""
    crowding = crowding.copy()
    week = crowding["date"].max()
    weeks = sorted(crowding["date"].unique())[-WINDOW_WEEKS:]
    window = crowding[crowding["date"].isin(weeks)].groupby("area")["paper_count"].sum()
    latest = crowding[crowding["date"] == week].sort_values("crowding_score", ascending=False)
    rows = [{"area": r.area, "score": half_up(float(r.crowding_score), 1),
             "papers_4w": int(window.get(r.area, 0))} for r in latest.itertuples()]
    most = max(rows, key=lambda r: r["papers_4w"])
    return {"week": week, "busiest": rows[:N_BUSY], "quietest": rows[-1],
            "most_papers": most}


def newest_paper(tagged: pd.DataFrame, area: str) -> dict | None:
    hits = tagged[tagged["area"] == area].sort_values("published")
    if hits.empty:
        return None
    r = hits.iloc[-1]
    return {"arxiv_id": r["arxiv_id"], "title": " ".join(str(r["title"]).split()),
            "published": str(r["published"])[:10]}


def tagger_share(tagged: pd.DataFrame) -> dict:
    counts = tagged["tagger"].value_counts()
    return {k: int(v) for k, v in counts.items()}


def allocation_facts(returns: pd.DataFrame, pub: pd.Series) -> dict:
    """decay_tilt weights for the month after the latest return, same rule as the backtest."""
    last = returns.index.max()
    est = bt.expected_edges(returns, pub, last)
    w = bt.weights_at(est)["decay_tilt"]
    held = [{"signal": s, "weight": half_up(float(w[s]), 2),
             "expected_edge_net": round(float(est.at[s, "net_edge"]), 4)}
            for s in w.sort_values(ascending=False).index if w[s] > 0]
    skipped = [s for s in est.index if w.get(s, 0) == 0]
    month = (last + pd.offsets.MonthEnd(1)).strftime("%B %Y")
    return {"for_month": month, "data_through": last.strftime("%Y-%m-%d"),
            "held": held, "skipped": skipped, "min_edge": bt.MIN_EDGE}


def caveat_facts(summary: pd.DataFrame) -> dict:
    s = summary.set_index("strategy").loc["decay_tilt"]
    return {"sharpe_minus_equal_weight": half_up(float(s["sharpe_minus_equal_weight"]), 2),
            "ci_low": half_up(float(s["diff_ci_low"]), 2),
            "ci_high": half_up(float(s["diff_ci_high"]), 2),
            "significant": bool(s["diff_ci_low"] > 0 or s["diff_ci_high"] < 0)}


def build_card() -> dict:
    tagged = pd.read_csv(TAGGED_CSV, dtype=str, keep_default_na=False)
    returns, _, pub = bt.load_inputs()
    card = {**crowding_facts(contracts.load("crowding"))}
    card["example_paper"] = newest_paper(tagged, card["busiest"][0]["area"])
    card["allocation"] = allocation_facts(returns, pub)
    card["caveat"] = caveat_facts(pd.read_csv(bt.SUMMARY_CSV))
    card["taggers"] = tagger_share(tagged)
    return card


# ---------- text ----------

def _signed(x: float) -> str:
    return f"plus {x:.1f}" if x >= 0 else f"minus {abs(x):.1f}"


def _signed2(x: float) -> str:
    return f"plus {x:.2f}" if x >= 0 else f"minus {abs(x):.2f}"


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def template_text(card: dict) -> str:
    """Conversational text meant to be heard, not read. Every number comes from card."""
    week = pd.Timestamp(card["week"])
    top, rest = card["busiest"][0], card["busiest"][1:]
    q = card["quietest"]
    lines = [f"Here's your research radar for the week of {week.day} {week:%B %Y}."]
    # Scores compare an area with its own usual share of papers, so say "crowded
    # compared with usual", not "busiest": the area with the most papers can score low.
    lines.append(f"Compared with its usual share of papers, the most crowded area right "
                 f"now is {top['area']}: {top['papers_4w']} papers in the last "
                 f"{WINDOW_WEEKS} weeks, a crowding score of {_signed(top['score'])}.")
    if card.get("example_paper"):
        lines.append(f"One of the newest is called \"{card['example_paper']['title']}\".")
    most = card.get("most_papers")
    if most and most["area"] != top["area"]:
        where = "below" if most["score"] < 0 else "closer to"
        lines.append(f"For scale, {most['area']} still has the most papers, "
                     f"{most['papers_4w']}, but that's {where} its usual share.")
    if rest:
        names = _join([r["area"] for r in rest])
        scores = _join([_signed(r["score"]) for r in rest])
        if all(r["score"] > 0 for r in rest):
            verb = "is" if len(rest) == 1 else "are"
            share = "its" if len(rest) == 1 else "their"
            lines.append(f"{names} {verb} also above {share} usual share, at {scores}.")
        else:
            lines.append(f"Next come {names}, at {scores}.")
    lines.append(f"And the quietest compared with usual? {q['area']}, at "
                 f"{_signed(q['score'])}.")
    a = card["allocation"]
    if a["held"]:
        held = _join([f"{h['signal'].lower()} at {half_up(h['weight'] * 100):.0f} percent"
                      for h in a["held"]])
        lines.append(f"So, what would we hold for {a['for_month']}? {_cap(held)}.")
    else:
        lines.append(f"So, what would we hold for {a['for_month']}? Nothing: we'd sit in "
                     f"cash, because no published signal clears the cost hurdle.")
    if a["skipped"]:
        many = len(a["skipped"]) > 1
        lines.append(f"We're skipping {_join([s.lower() for s in a['skipped']])}, because "
                     f"after costs {'none of them is' if many else 'it is not'} expected to "
                     f"earn more than {a['min_edge'] * 100:.1f} percent a month.")
    c = card["caveat"]
    diff = c["sharpe_minus_equal_weight"]
    verb = "beat" if diff >= 0 else "trailed"
    record = (f"since 1981, this rule {verb} equal weighting by {abs(diff):.2f} in Sharpe "
              f"ratio")
    if c["significant"]:
        lines.append(f"And the track record: {record}, with a 95 percent interval of "
                     f"{_signed2(c['ci_low'])} to {_signed2(c['ci_high'])}.")
    else:
        lines.append(f"One honest caveat, though. {_cap(record)}, but the 95 percent "
                     f"interval runs from {_signed2(c['ci_low'])} to "
                     f"{_signed2(c['ci_high'])}. So the edge isn't proven.")
    return " ".join(lines)


NUMBER = re.compile(r"\d+(?:\.\d+)?")

GEMINI_PROMPT = """Rewrite this research briefing so it sounds like a person talking, in
60 seconds or less. Keep every number, area name, signal name and paper title exactly
as written, and keep the caveat. Do not add any facts, numbers or advice. Crowding
compares an area with its own usual share of papers, so never call an area "busiest"
or say it has the most papers unless the briefing says so. Plain text only, no
markdown, no bullet points.

Briefing:
{text}"""


def same_numbers(a: str, b: str) -> bool:
    return sorted(NUMBER.findall(a)) == sorted(NUMBER.findall(b))


def gemini_rewrite(text: str, key: str, model: str) -> str | None:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(
        model=model, contents=GEMINI_PROMPT.format(text=text),
        config=types.GenerateContentConfig(temperature=0.3))
    out = " ".join((resp.text or "").split())
    ok = out and same_numbers(text, out) and "busiest" not in out.lower()
    return out if ok else None


# ---------- audio ----------

def make_audio(text: str, key: str, voice_id: str, model_id: str, path: Path = AUDIO_PATH) -> None:
    from elevenlabs.client import ElevenLabs
    chunks = ElevenLabs(api_key=key).text_to_speech.convert(
        voice_id, text=text, model_id=model_id, output_format="mp3_44100_128")
    data = b"".join(chunks)
    if not data:
        raise RuntimeError("ElevenLabs returned no audio")
    path.write_bytes(data)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Write this week's briefing.")
    parser.add_argument("--no-gemini", action="store_true", help="use the template text")
    parser.add_argument("--no-audio", action="store_true", help="skip ElevenLabs")
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env")

    card = build_card()
    text, source = template_text(card), "template"
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key and not args.no_gemini:
        model = os.getenv("GEMINI_MODEL", "").strip() or DEFAULT_MODEL
        try:
            rewrite = gemini_rewrite(text, gemini_key, model)
        except Exception as exc:  # any API failure: keep the template text
            print(f"[warn] gemini rewrite failed ({type(exc).__name__}); using template")
            rewrite = None
        if rewrite:
            text, source = rewrite, f"gemini:{model}"
        else:
            print("[warn] gemini rewrite unusable (numbers changed or empty); using template")
    card["text_source"] = source
    TEXT_PATH.write_text(text + "\n", encoding="utf-8")
    CARD_PATH.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {TEXT_PATH.name} ({source}, {len(text)} chars) and {CARD_PATH.name}")
    print(text)

    stamp = AUDIO_STAMP.read_text(encoding="utf-8") if AUDIO_STAMP.exists() else None
    eleven_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    if AUDIO_PATH.exists() and stamp == text:
        print(f"{AUDIO_PATH.name} already matches the text")
    elif args.no_audio or not eleven_key:
        why = "--no-audio" if args.no_audio else "no ELEVENLABS_API_KEY"
        state = "kept the existing (pre-recorded)" if AUDIO_PATH.exists() else "no"
        print(f"[skip] audio ({why}); {state} {AUDIO_PATH.name}")
    else:
        try:
            make_audio(text, eleven_key,
                       os.getenv("ELEVENLABS_VOICE_ID", "").strip() or DEFAULT_VOICE_ID,
                       os.getenv("ELEVENLABS_MODEL", "").strip() or DEFAULT_TTS_MODEL)
            AUDIO_STAMP.parent.mkdir(parents=True, exist_ok=True)
            AUDIO_STAMP.write_text(text, encoding="utf-8")
            print(f"wrote {AUDIO_PATH.name} ({AUDIO_PATH.stat().st_size // 1024} KB)")
        except Exception as exc:  # quota, network: keep whatever mp3 is committed
            kept = f"kept the existing {AUDIO_PATH.name}" if AUDIO_PATH.exists() else "no audio"
            print(f"[warn] ElevenLabs failed ({type(exc).__name__}: {str(exc)[:150]}); {kept}")
    return TEXT_PATH


if __name__ == "__main__":
    main()
