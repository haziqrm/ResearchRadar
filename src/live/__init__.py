"""Person D: live layer and decision.

Pipeline (run with `python -m src.live.run_live`):
    arxiv_feed  -> data/cache/arxiv/papers.csv     recent q-fin papers (git-ignored cache)
    map_tags    -> data/arxiv_map_tags.csv         Gemini: 8 research-map areas per paper
    crowding    -> data/crowding.csv               Poisson z-score per map area: OpenAlex
                                                   by year for now (arXiv by week later)
    backtest    -> data/backtest_results.csv       walk-forward allocation test
    briefing    -> data/briefing.txt / .mp3        spoken summary
"""

from src.common.contracts import DATA_DIR

CACHE_DIR = DATA_DIR / "cache" / "arxiv"
RAW_DIR = CACHE_DIR / "raw"
PAPERS_CSV = CACHE_DIR / "papers.csv"
