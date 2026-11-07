"""Person D: live layer and decision.

Pipeline (run with `python -m src.live.run_live`):
    arxiv_feed  -> data/cache/arxiv/papers.csv     recent q-fin papers (git-ignored cache)
    tagging     -> data/arxiv_tagged.csv           area + matching signal per paper
    crowding    -> data/crowding.csv               weekly Poisson z-score per area
    backtest    -> data/backtest_results.csv       walk-forward allocation test
    briefing    -> data/briefing.txt / .mp3        spoken summary
"""

from src.common.contracts import DATA_DIR

CACHE_DIR = DATA_DIR / "cache" / "arxiv"
RAW_DIR = CACHE_DIR / "raw"
PAPERS_CSV = CACHE_DIR / "papers.csv"
