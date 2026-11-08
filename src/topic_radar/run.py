"""Run the topic radar end to end: group forecast (H11), sub-topics, seed papers (H12).

Run: python -m src.topic_radar.run
Writes data/area_forecasts.csv, topic_forecast_eval.csv, subtopics.csv, seed_papers.csv,
seed_eval.csv.
"""

from __future__ import annotations

from src.topic_radar import forecast, seeds, subtopics
from src.topic_radar.panel import load_corpus

SHOW = ["metric", "model", "value", "ci_low", "ci_high", "holm_p"]


def main() -> None:
    corpus = load_corpus()
    print(f"{len(corpus)} papers, {len(corpus.edges)} edges, {len(corpus.group_names)} groups")
    level1 = forecast.run(corpus)
    ev = level1["eval"]
    for mode in ["refit", "frozen"]:
        print(f"\nH11 group forecast, {mode} groups")
        print(ev[ev["groups_mode"] == mode][SHOW].round(3).to_string(index=False))
    print("H11 pass:", bool(ev["h11_pass"].dropna().iloc[0]))
    live = level1["contract"][level1["contract"]["actual_boom"].isna()]
    print("\nLive forecast from 2025 (top 8):")
    print(live.sort_values("growth_score", ascending=False).head(8).to_string(index=False))

    table, papers = subtopics.run(corpus)
    print(f"\nSub-topics: {len(table)}, rising: {int(table['rising'].sum())}")
    print(table[table["rising"]].sort_values("growth", ascending=False).round(3).to_string(index=False))

    level3 = seeds.run(corpus, level1, papers)
    print(f"\nH12 seed papers (tau = {level3['tau']:.3f})")
    print(level3["eval"][SHOW].round(3).to_string(index=False))
    print("H12 pass:", bool(level3["eval"]["h12_pass"].iloc[0]))


if __name__ == "__main__":
    main()
